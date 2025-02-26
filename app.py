import streamlit as st
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from PIL import Image, ImageOps
import exifread
import numpy as np
import os
from pyproj import Transformer
import io
import math
from affine import Affine
import zipfile
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import base64
import uuid
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString

##############################################
# Dictionnaires pour les classes de défauts
##############################################
class_color = {
    "deformations ornierage": "#FF0000",
    "fissurations": "#00FF00",
    "Faiençage": "#0000FF",
    "fissure de retrait": "#FFFF00",
    "fissure anarchique": "#FF00FF",
    "reparations": "#00FFFF",
    "nid de poule": "#FFA500",
    "arrachements": "#800080",
    "fluage": "#008000",
    "denivellement accotement": "#000080",
    "chaussée detruite": "#FFC0CB",
    "envahissement vegetations": "#A52A2A",
    "assainissements": "#808080",
    "depot de terre": "#8B4513"
}
gravity_sizes = {1: 5, 2: 10, 3: 15}

##############################################
# Gestion des routes (chargement depuis un fichier)
##############################################
routes_ci = []
if os.path.exists("routeQSD.txt"):
    with open("routeQSD.txt", "r") as f:
        routes_data = json.load(f)
    for feature in routes_data["features"]:
        if feature["geometry"]["type"] == "LineString":
            routes_ci.append({
                "coords": feature["geometry"]["coordinates"],
                "nom": feature["properties"].get("ID", "Route inconnue")
            })

##############################################
# Fonctions utilitaires pour la carte
##############################################
def add_image_overlay(map_object, image_path, bounds, layer_name, opacity=1, show=True, control=True):
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    img_data_url = f"data:image/png;base64,{image_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_data_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=opacity,
        show=show,
        control=control
    ).add_to(map_object)

def create_map(center_lat, center_lon, bounds, display_path, marker_data=None,
               hide_osm=False, tiff_opacity=1, tiff_show=True, tiff_control=True,
               draw_routes=True, add_draw_tool=True):
    if hide_osm:
        m = folium.Map(location=[center_lat, center_lon], tiles=None)
    else:
        m = folium.Map(location=[center_lat, center_lon])
    if display_path:
        add_image_overlay(m, display_path, bounds, "TIFF Overlay", opacity=tiff_opacity,
                          show=tiff_show, control=tiff_control)
    if draw_routes:
        for route in routes_ci:
            # Inversion des coordonnées pour obtenir (lat, lon)
            poly_coords = [(lat, lon) for lon, lat in route["coords"]]
            folium.PolyLine(
                locations=poly_coords,
                color="blue",
                weight=3,
                opacity=0.7,
                tooltip=route["nom"]
            ).add_to(m)
    if add_draw_tool:
        draw = Draw(
            draw_options={
                'marker': True,
                'polyline': False,
                'polygon': False,
                'rectangle': False,
                'circle': False,
                'circlemarker': False,
            },
            edit_options={'edit': True}
        )
        draw.add_to(m)
    
    # Ajout des marqueurs si marker_data est fourni
    if marker_data:
        for marker in marker_data:
            lat = marker["lat"]
            lon = marker["long"]
            color = class_color.get(marker["classe"], "#000000")
            radius = gravity_sizes.get(marker["gravite"], 5)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"{marker['classe']} (Gravité {marker['gravite']}) - Route : {marker.get('routes', 'Route inconnue')} - Détection: {marker.get('detection', 'Inconnue')}"
            ).add_to(m)
    
    m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
    folium.LayerControl().add_to(m)
    return m

##############################################
# Fonctions d'extraction et conversion
##############################################
def extract_exif_info(image_file):
    image_file.seek(0)
    tags = exifread.process_file(image_file)
    
    lat = lon = altitude = focal_length = None
    fp_x_res = fp_unit = None
    
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags.get('GPS GPSLatitudeRef', None)
        lon_ref = tags.get('GPS GPSLongitudeRef', None)
        
        if lat_vals and lon_vals and lat_ref and lon_ref:
            lat = (float(lat_vals[0].num) / lat_vals[0].den +
                   float(lat_vals[1].num) / lat_vals[1].den / 60 +
                   float(lat_vals[2].num) / lat_vals[2].den / 3600)
            lon = (float(lon_vals[0].num) / lon_vals[0].den +
                   float(lon_vals[1].num) / lon_vals[1].den / 60 +
                   float(lon_vals[2].num) / lon_vals[2].den / 3600)
            
            if lat_ref.printable.strip().upper() == 'S':
                lat = -lat
            if lon_ref.printable.strip().upper() == 'W':
                lon = -lon
    
    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den
        
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den
    
    if 'EXIF FocalPlaneXResolution' in tags and 'EXIF FocalPlaneResolutionUnit' in tags:
        fp_res_tag = tags['EXIF FocalPlaneXResolution']
        fp_unit_tag = tags['EXIF FocalPlaneResolutionUnit']
        
        fp_x_res = float(fp_res_tag.values[0].num) / fp_res_tag.values[0].den
        fp_unit = int(fp_unit_tag.values[0])
    
    return lat, lon, altitude, focal_length, fp_x_res, fp_unit

def latlon_to_utm(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

def convert_to_tiff_in_memory(image_file, pixel_size, utm_center, utm_crs, rotation_angle=0, scaling_factor=1):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    orig_width, orig_height = img.size
    new_width = int(orig_width * scaling_factor)
    new_height = int(orig_height * scaling_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    effective_pixel_size = pixel_size / scaling_factor

    center_x, center_y = utm_center
    T1 = Affine.translation(-width/2, -height/2)
    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
    T3 = Affine.rotation(rotation_angle)
    T4 = Affine.translation(center_x, center_y)
    transform = T4 * T3 * T2 * T1

    memfile = MemoryFile()
    with memfile.open(
        driver='GTiff',
        height=height,
        width=width,
        count=3 if len(img_array.shape) == 3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform
    ) as dst:
        if len(img_array.shape) == 3:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            dst.write(img_array, 1)
    return memfile.read()

def process_uploaded_image(uploaded_file, pixel_size):
    file_bytes = uploaded_file.read()
    file_buffer = io.BytesIO(file_bytes)
    lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
    if lat is None or lon is None:
        return None  # Image ignorée si pas de coordonnées GPS.
    
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    img_width, img_height = img.size
    
    sensor_width_mm = None
    if fp_x_res and fp_unit:
        if fp_unit == 2:   # pouces
            sensor_width_mm = (img_width / fp_x_res) * 25.4
        elif fp_unit == 3: # cm
            sensor_width_mm = (img_width / fp_x_res) * 10
        elif fp_unit == 4: # mm
            sensor_width_mm = (img_width / fp_x_res)
    
    utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
    
    # On utilise flight_angle = 0 pour le premier passage.
    flight_angle = 0

    # Conversion initiale
    tiff_petit_bytes = convert_to_tiff_in_memory(
        image_file=io.BytesIO(file_bytes),
        pixel_size=pixel_size,
        utm_center=(utm_x, utm_y),
        utm_crs=utm_crs,
        rotation_angle=-flight_angle,
        scaling_factor=1/5
    )
    tiff_grand_bytes = convert_to_tiff_in_memory(
        image_file=io.BytesIO(file_bytes),
        pixel_size=pixel_size * 2,
        utm_center=(utm_x, utm_y),
        utm_crs=utm_crs,
        rotation_angle=-flight_angle,
        scaling_factor=1/3
    )
    
    # Conversion JPEG avec métadonnées (scaling_factor = 1)
    scaling_factor = 1
    new_width = int(img_width * scaling_factor)
    new_height = int(img_height * scaling_factor)
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    effective_pixel_size = pixel_size / scaling_factor
    center_x, center_y = (utm_x, utm_y)
    T1 = Affine.translation(-new_width/2, -new_height/2)
    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
    T3 = Affine.rotation(-flight_angle)
    T4 = Affine.translation(center_x, center_y)
    transform = T4 * T3 * T2 * T1

    corners = [
        (-new_width/2, -new_height/2),
        (new_width/2, -new_height/2),
        (new_width/2, new_height/2),
        (-new_width/2, new_height/2)
    ]
    corner_coords = [transform * corner for corner in corners]
    metadata_str = f"Frame Coordinates: {corner_coords}"
    
    try:
        import piexif
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
        else:
            exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
        user_comment = metadata_str
        try:
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.dump(user_comment, encoding="unicode")
        except AttributeError:
            prefix = b"UNICODE\0"
            encoded_comment = user_comment.encode("utf-16")
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + encoded_comment
        exif_bytes = piexif.dump(exif_dict)
    except ImportError:
        exif_bytes = None
    
    jpeg_buffer = io.BytesIO()
    if exif_bytes:
        img_resized.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
    else:
        img_resized.save(jpeg_buffer, format="JPEG")
    jpeg_image_bytes = jpeg_buffer.getvalue()
    
    return {
        "filename": uploaded_file.name,
        "data_original": file_bytes,
        "lat": lat,
        "lon": lon,
        "altitude": altitude,
        "focal_length": focal_length,
        "sensor_width": sensor_width_mm,
        "utm": (utm_x, utm_y),
        "utm_crs": utm_crs,
        "img_width": img_width,
        "img_height": img_height,
        "flight_angle": flight_angle,  # sera mis à jour ensuite
        "tiff_petit": tiff_petit_bytes,
        "tiff_grand": tiff_grand_bytes,
        "jpeg_image": jpeg_image_bytes
    }

def update_conversions(processed, flight_angle, pixel_size):
    file_bytes = processed["data_original"]
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    img_width, img_height = img.size
    utm = processed["utm"]
    utm_crs = processed["utm_crs"]
    
    tiff_petit = convert_to_tiff_in_memory(
         image_file=io.BytesIO(file_bytes),
         pixel_size=pixel_size,
         utm_center=utm,
         utm_crs=utm_crs,
         rotation_angle=-flight_angle,
         scaling_factor=1/5
    )
    tiff_grand = convert_to_tiff_in_memory(
         image_file=io.BytesIO(file_bytes),
         pixel_size=pixel_size * 2,
         utm_center=utm,
         utm_crs=utm_crs,
         rotation_angle=-flight_angle,
         scaling_factor=1/3
    )
    scaling_factor = 1
    new_width = int(img_width * scaling_factor)
    new_height = int(img_height * scaling_factor)
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    effective_pixel_size = pixel_size / scaling_factor
    center_x, center_y = utm
    T1 = Affine.translation(-new_width/2, -new_height/2)
    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
    T3 = Affine.rotation(-flight_angle)
    T4 = Affine.translation(center_x, center_y)
    transform = T4 * T3 * T2 * T1

    corners = [
         (-new_width/2, -new_height/2),
         (new_width/2, -new_height/2),
         (new_width/2, new_height/2),
         (-new_width/2, new_height/2)
    ]
    corner_coords = [transform * corner for corner in corners]
    metadata_str = f"Frame Coordinates: {corner_coords}"
    
    try:
         import piexif
         if "exif" in img.info:
             exif_dict = piexif.load(img.info["exif"])
         else:
             exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
         user_comment = metadata_str
         try:
             exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.dump(user_comment, encoding="unicode")
         except AttributeError:
             prefix = b"UNICODE\0"
             encoded_comment = user_comment.encode("utf-16")
             exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + encoded_comment
         exif_bytes = piexif.dump(exif_dict)
    except ImportError:
         exif_bytes = None
    jpeg_buffer = io.BytesIO()
    if exif_bytes:
         img_resized.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
    else:
         img_resized.save(jpeg_buffer, format="JPEG")
    jpeg_image = jpeg_buffer.getvalue()
    return tiff_petit, tiff_grand, jpeg_image

def calculate_flight_angle(index, images_info):
    if len(images_info) >= 2:
        if index == 0:
            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
        elif index == len(images_info) - 1:
            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
        else:
            dx = images_info[index+1]["utm"][0] - images_info[index-1]["utm"][0]
            dy = images_info[index+1]["utm"][1] - images_info[index-1]["utm"][1]
        return math.degrees(math.atan2(dx, dy))
    else:
        return 0

##############################################
# Reprojection (pour la détection manuelle)
##############################################
def reproject_tiff(input_tiff, target_crs="EPSG:4326"):
    with rasterio.open(input_tiff) as src:
        transform_, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform_,
            "width": width,
            "height": height,
        })
        unique_id = str(uuid.uuid4())[:8]
        output_tiff = f"reprojected_{unique_id}.tif"
        with rasterio.open(output_tiff, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform_,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return output_tiff

def get_reprojected_and_center_from_bytes(tiff_bytes, group):
    unique_id = str(uuid.uuid4())[:8]
    temp_path = f"temp_{group}_{unique_id}.tif"
    with open(temp_path, "wb") as f:
        f.write(tiff_bytes)
    with rasterio.open(temp_path) as src:
        crs_str = src.crs.to_string()
    if crs_str != "EPSG:4326":
        reproj_path = reproject_tiff(temp_path, "EPSG:4326")
    else:
        reproj_path = temp_path
    with rasterio.open(reproj_path) as src:
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
    return {"path": reproj_path, "center": (center_lon, center_lat), "bounds": bounds, "temp_original": temp_path}

##############################################
# Stockage dans le session_state
##############################################
if "converted_images" not in st.session_state:
    st.session_state.converted_images = []
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}

##############################################
# Interface globale
##############################################
st.title("Application de conversion et détection d'anomalies")

# --- 1. Téléversement unique des JPEG et traitement initial ---
uploaded_jpeg_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploaded_jpegs"
)
pixel_size = st.number_input(
    "Choisissez la résolution spatiale (m/pixel) :", 
    min_value=0.001, 
    value=0.03, 
    step=0.001, 
    format="%.3f"
)

if uploaded_jpeg_files:
    new_converted = []
    for up_file in uploaded_jpeg_files:
        processed = process_uploaded_image(up_file, pixel_size)
        if processed:
            new_converted.append(processed)
        else:
            st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
    if new_converted:
        # --- 2. Calcul du flight_angle pour chaque image ---
        for idx, processed in enumerate(new_converted):
            flight_angle = calculate_flight_angle(idx, new_converted)
            processed["flight_angle"] = flight_angle
            # Mise à jour des conversions avec l'angle calculé
            tiff_petit, tiff_grand, jpeg_image = update_conversions(processed, flight_angle, pixel_size)
            processed["tiff_petit"] = tiff_petit
            processed["tiff_grand"] = tiff_grand
            processed["jpeg_image"] = jpeg_image
        st.session_state.converted_images = new_converted
        st.success("Les images ont été traitées et converties avec succès.")
    else:
        st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")

##############################################
# Onglets de détection
##############################################
tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])

with tab_auto:
    st.header("Détection Automatique")
    if "converted_images" in st.session_state and st.session_state.converted_images:
        st.success("Les images converties pour la détection automatique ont été récupérées.")
        auto_images = [item["jpeg_image"] for item in st.session_state.converted_images]
        for idx, jpeg_bytes in enumerate(auto_images):
            st.image(jpeg_bytes, caption=f"Image convertie {idx+1}", use_column_width=True)
        # Insérez ici la logique de détection automatique en utilisant auto_images
    else:
        st.info("Aucune image convertie n'est disponible pour la détection automatique.")

with tab_manuel:
    st.header("Détection Manuelle")
    if st.session_state.converted_images:
        # Création de paires pour la détection manuelle via reprojection
        grand_list = []
        petit_list = []
        for item in st.session_state.converted_images:
            grand_list.append(get_reprojected_and_center_from_bytes(item["tiff_grand"], "grand"))
            petit_list.append(get_reprojected_and_center_from_bytes(item["tiff_petit"], "petit"))
        pair_count = min(len(grand_list), len(petit_list))
        pairs = []
        for i in range(pair_count):
            pairs.append({"grand": grand_list[i], "petit": petit_list[i]})
        st.session_state.pairs = pairs

        pair_count = len(st.session_state.pairs)
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        if col_nav1.button("← Précédent") and st.session_state.get("current_pair_index", 0) > 0:
            st.session_state.current_pair_index -= 1
        if col_nav3.button("Suivant →") and st.session_state.get("current_pair_index", 0) < pair_count - 1:
            st.session_state.current_pair_index += 1
        if "current_pair_index" not in st.session_state:
            st.session_state.current_pair_index = 0
        st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {pair_count}")
        current_index = st.session_state.current_pair_index
        current_pair = st.session_state.pairs[current_index]

        # Affichage du TIFF GRAND converti en PNG pour la carte de dessin
        reproj_grand_path = current_pair["grand"]["path"]
        with rasterio.open(reproj_grand_path) as src:
            grand_bounds = src.bounds
            data = src.read()
            if data.shape[0] >= 3:
                # Normalisation des bandes pour affichage
                r = np.clip(255 * (data[0] - np.percentile(data[0], 2)) / (np.percentile(data[0], 98) - np.percentile(data[0], 2)), 0, 255).astype(np.uint8)
                g = np.clip(255 * (data[1] - np.percentile(data[1], 2)) / (np.percentile(data[1], 98) - np.percentile(data[1], 2)), 0, 255).astype(np.uint8)
                b = np.clip(255 * (data[2] - np.percentile(data[2], 2)) / (np.percentile(data[2], 98) - np.percentile(data[2], 2)), 0, 255).astype(np.uint8)
                rgb_norm = np.dstack((r, g, b))
                image_grand = Image.fromarray(rgb_norm)
            else:
                band = data[0]
                band_norm = np.clip(255 * (band - np.percentile(band, 2)) / (np.percentile(band, 98) - np.percentile(band, 2)), 0, 255).astype(np.uint8)
                image_grand = Image.fromarray(band_norm, mode="L")
        unique_id = str(uuid.uuid4())[:8]
        temp_png_grand = f"converted_grand_{unique_id}.png"
        image_grand.save(temp_png_grand)
        display_path_grand = temp_png_grand

        # Affichage du TIFF PETIT converti en PNG pour la carte de dessin
        reproj_petit_path = current_pair["petit"]["path"]
        with rasterio.open(reproj_petit_path) as src:
            petit_bounds = src.bounds
            data = src.read()
            if data.shape[0] >= 3:
                r = np.clip(255 * (data[0] - np.percentile(data[0], 2)) / (np.percentile(data[0], 98) - np.percentile(data[0], 2)), 0, 255).astype(np.uint8)
                g = np.clip(255 * (data[1] - np.percentile(data[1], 2)) / (np.percentile(data[1], 98) - np.percentile(data[1], 2)), 0, 255).astype(np.uint8)
                b = np.clip(255 * (data[2] - np.percentile(data[2], 2)) / (np.percentile(data[2], 98) - np.percentile(data[2], 2)), 0, 255).astype(np.uint8)
                rgb_norm = np.dstack((r, g, b))
                image_petit = Image.fromarray(rgb_norm)
            else:
                band = data[0]
                band_norm = np.clip(255 * (band - np.percentile(band, 2)) / (np.percentile(band, 98) - np.percentile(band, 2)), 0, 255).astype(np.uint8)
                image_petit = Image.fromarray(band_norm, mode="L")
        temp_png_petit = f"converted_{unique_id}.png"
        image_petit.save(temp_png_petit)
        display_path_petit = temp_png_petit

        # Création de la carte de dessin avec routes et outil de dessin activé
        center_lat_grand = (grand_bounds.bottom + grand_bounds.top) / 2
        center_lon_grand = (grand_bounds.left + grand_bounds.right) / 2
        m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand,
                             marker_data=None, hide_osm=True, draw_routes=True, add_draw_tool=True)
        result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")

        # Gestion des marqueurs dessinés sur la carte
        features = []
        all_drawings = result_grand.get("all_drawings")
        if all_drawings:
            if isinstance(all_drawings, dict) and "features" in all_drawings:
                features = all_drawings.get("features", [])
            elif isinstance(all_drawings, list):
                features = all_drawings

        global_count = sum(len(markers) for markers in st.session_state.markers_by_pair.values()) if st.session_state.markers_by_pair else 0
        new_markers = []
        if features:
            st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
            for i, feature in enumerate(features):
                if feature.get("geometry", {}).get("type") == "Point":
                    coords = feature.get("geometry", {}).get("coordinates")
                    if coords and isinstance(coords, list) and len(coords) >= 2:
                        # Notez que dans le mode manuel, les coordonnées dessinées sont en EPSG:4326
                        lon_pt, lat_pt = coords[0], coords[1]
                    else:
                        lat_pt = lon_pt = None
                    st.markdown(f"**Marqueur {global_count + i + 1}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class = st.selectbox("Classe", list(class_color.keys()), key=f"class_{current_index}_{i}")
                    with col2:
                        selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{i}")
                    new_markers.append({
                        "ID": global_count + i + 1,
                        "classe": selected_class,
                        "gravite": selected_gravity,
                        "lat": lat_pt,
                        "long": lon_pt,
                        "detection": "Manuelle",
                        "couleur": class_color.get(selected_class, "#000000"),
                        "radius": gravity_sizes.get(selected_gravity, 5)
                    })
            st.session_state.markers_by_pair[current_index] = new_markers
        else:
            st.write("Aucun marqueur n'a été dessiné.")
    else:
        st.info("Aucune paire d'images converties n'est disponible pour la détection manuelle.")

##############################################
# Carte de suivi globale
##############################################
st.subheader("Carte de suivi")
global_markers = []
for markers in st.session_state.markers_by_pair.values():
    global_markers.extend(markers)

if st.session_state.pairs:
    try:
        with rasterio.open(st.session_state.pairs[0]["petit"]["path"]) as src:
            petit_bounds = src.bounds
    except Exception as e:
        st.error("Erreur lors de l'ouverture du TIFF PETIT pour la carte de suivi.")
        petit_bounds = None

    if petit_bounds:
        center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
        center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
        m_petit = create_map(center_lat_petit, center_lon_petit, petit_bounds,
                             display_path_petit,  # chemin vers l'image overlay pour le TIFF PETIT
                             marker_data=global_markers,
                             hide_osm=False,
                             tiff_opacity=0,
                             tiff_show=True,
                             tiff_control=False,
                             draw_routes=True,
                             add_draw_tool=False)
        st_folium(m_petit, width=700, height=500, key="folium_map_petit")
    else:
        st.info("Impossible d'afficher la carte de suivi.")
else:
    st.info("Aucune paire d'images n'est disponible pour la carte de suivi.")

st.markdown("### Récapitulatif global des défauts")
if global_markers:
    st.table(global_markers)
else:
    st.write("Aucun marqueur global n'a été enregistré.")
