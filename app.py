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
from folium.plugins import Draw  # Plugin pour dessiner sur la carte
from streamlit_folium import st_folium
import base64
import uuid
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString

# ---------------------------
# FONCTIONS DE LA PARTIE CONVERSION
# ---------------------------
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

def compute_gsd(altitude, focal_length_mm, sensor_width_mm, image_width_px):
    focal_length_m = focal_length_mm / 1000.0
    sensor_width_m = sensor_width_mm / 1000.0
    gsd = (altitude * sensor_width_m) / (focal_length_m * image_width_px)
    return gsd

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs, rotation_angle=0, scaling_factor=1):
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
    transform_affine = T4 * T3 * T2 * T1

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3 if len(img_array.shape) == 3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform_affine
    ) as dst:
        if len(img_array.shape) == 3:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            dst.write(img_array, 1)

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
    transform_affine = T4 * T3 * T2 * T1

    memfile = MemoryFile()
    with memfile.open(
        driver='GTiff',
        height=height,
        width=width,
        count=3 if len(img_array.shape) == 3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform_affine
    ) as dst:
        if len(img_array.shape) == 3:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            dst.write(img_array, 1)
    return memfile.read()

def convert_to_jpeg_with_frame(image_bytes, pixel_size, utm_center, utm_crs, rotation_angle=0):
    try:
        import piexif
    except ImportError:
        st.error("La librairie piexif est requise pour ajouter des métadonnées JPEG. Veuillez l'installer.")
        return None
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    orig_width, orig_height = img.size
    scaling_factor = 1  # Aucun redimensionnement
    new_width = int(orig_width * scaling_factor)
    new_height = int(orig_height * scaling_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    effective_pixel_size = pixel_size / scaling_factor
    center_x, center_y = utm_center
    T1 = Affine.translation(-new_width/2, -new_height/2)
    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
    T3 = Affine.rotation(rotation_angle)
    T4 = Affine.translation(center_x, center_y)
    transform_affine = T4 * T3 * T2 * T1

    # Calcul des coordonnées des 4 coins du cadre
    corners = [
        (-new_width/2, -new_height/2),
        (new_width/2, -new_height/2),
        (new_width/2, new_height/2),
        (-new_width/2, new_height/2)
    ]
    corner_coords = []
    for corner in corners:
        x, y = transform_affine * corner
        corner_coords.append((x, y))
    
    metadata_str = f"Frame Coordinates: {corner_coords}"
    
    try:
        if "exif" in img.info:
            exif_dict = piexif.load(img.info["exif"])
        else:
            exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
        try:
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.dump(metadata_str, encoding="unicode")
        except AttributeError:
            prefix = b"UNICODE\0"
            encoded_comment = metadata_str.encode("utf-16")
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + encoded_comment
        exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        st.error("Erreur lors de l'ajout des métadonnées : " + str(e))
        exif_bytes = None
    
    jpeg_buffer = io.BytesIO()
    if exif_bytes:
        img.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
    else:
        img.save(jpeg_buffer, format="JPEG")
    return jpeg_buffer.getvalue()

# ---------------------------
# SECTION 1 – CONVERSION
# ---------------------------
st.title("Application Intégrée: Conversion et Détection d'anomalies")
st.header("1. Conversion JPEG → GeoTIFF & Export JPEG avec métadonnées")

# Téléversement des images JPEG
uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True,
    key="conversion_upload"
)

if uploaded_files:
    images_info = []
    
    for up_file in uploaded_files:
        file_bytes = up_file.read()
        file_buffer = io.BytesIO(file_bytes)
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
        if lat is None or lon is None:
            st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
            continue
        
        img = Image.open(io.BytesIO(file_bytes))
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
        
        images_info.append({
            "filename": up_file.name,
            "data": file_bytes,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "focal_length": focal_length,
            "sensor_width": sensor_width_mm,
            "utm": (utm_x, utm_y),
            "utm_crs": utm_crs,
            "img_width": img_width,
            "img_height": img_height
        })
    
    if len(images_info) == 0:
        st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")
    else:
        st.session_state["images_info"] = images_info  # Conserver pour la détection ultérieure
        pixel_size = st.number_input(
            "Choisissez la résolution spatiale (m/pixel) :", 
            min_value=0.001, 
            value=0.03, 
            step=0.001, 
            format="%.3f"
        )
        st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
        
        # -------------------------------
        # Configuration 1 : Export groupé en GeoTIFF (facteur de redimensionnement fixé à 1/5)
        if st.button("Configuration 1 (Carte de suivi)"):
            conv_config1 = []  # Stockera les GeoTIFF en mémoire
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for info in images_info:
                    # Si plusieurs images, calcul d'un angle de vol (simplifié ici)
                    flight_angle = 0  
                    tiff_bytes = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle,
                        scaling_factor=1/5
                    )
                    conv_config1.append({
                        "filename": info["filename"].rsplit(".", 1)[0] + "_geotiff.tif",
                        "bytes": tiff_bytes
                    })
                    zip_file.writestr(info["filename"].rsplit(".", 1)[0] + "_geotiff.tif", tiff_bytes)
            zip_buffer.seek(0)
            st.download_button(
                label="Télécharger le ZIP Configuration 1",
                data=zip_buffer,
                file_name="images_geotiff_config1.zip",
                mime="application/zip"
            )
            st.session_state["conv_config1"] = conv_config1

        # -------------------------------
        # Configuration 2 : Export groupé en GeoTIFF x2 (facteur de redimensionnement 1/3 et résolution multipliée par 2)
        if st.button("Configuration 2 (Carte de dessin)"):
            conv_config2 = []
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for info in images_info:
                    flight_angle = 0  
                    tiff_bytes = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size * 2,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle,
                        scaling_factor=1/3
                    )
                    conv_config2.append({
                        "filename": info["filename"].rsplit(".", 1)[0] + "_geotiff_x2.tif",
                        "bytes": tiff_bytes
                    })
                    zip_file.writestr(info["filename"].rsplit(".", 1)[0] + "_geotiff_x2.tif", tiff_bytes)
            zip_buffer.seek(0)
            st.download_button(
                label="Télécharger le ZIP Configuration 2",
                data=zip_buffer,
                file_name="images_geotiff_config2.zip",
                mime="application/zip"
            )
            st.session_state["conv_config2"] = conv_config2

        # -------------------------------
        # Configuration images : Export groupé en JPEG avec métadonnées de cadre
        if st.button("Configuration images (Détection automatique)"):
            conv_images = []
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for info in images_info:
                    flight_angle = 0  
                    jpeg_bytes = convert_to_jpeg_with_frame(
                        image_bytes=info["data"],
                        pixel_size=pixel_size,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle
                    )
                    if jpeg_bytes is not None:
                        conv_images.append({
                            "filename": info["filename"].rsplit(".", 1)[0] + "_with_frame_coords.jpg",
                            "bytes": jpeg_bytes
                        })
                        zip_file.writestr(info["filename"].rsplit(".", 1)[0] + "_with_frame_coords.jpg", jpeg_bytes)
            zip_buffer.seek(0)
            st.download_button(
                label="Télécharger le ZIP Configuration images",
                data=zip_buffer,
                file_name="images_with_frame_coords.zip",
                mime="application/zip"
            )
            st.session_state["conv_images"] = conv_images

# ---------------------------
# FONCTIONS DE LA PARTIE DÉTECTION
# ---------------------------
def apply_color_gradient(tiff_path, output_png_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        colored_image = cmap(norm(data))
        plt.imsave(output_png_path, colored_image)
        plt.close()

def assign_route_to_marker(lat, lon, routes):
    """
    Pour un point (lat, lon) en EPSG:4326, retourne le nom de la route la plus proche,
    si celle-ci se trouve dans un seuil défini (sinon "Route inconnue").
    """
    marker_point = Point(lon, lat)
    min_distance = float('inf')
    closest_route = "Route inconnue"
    for route in routes:
        line = LineString(route["coords"])
        distance = marker_point.distance(line)
        if distance < min_distance:
            min_distance = distance
            closest_route = route["nom"]
    threshold = 0.01  # Seuil en degrés (à ajuster selon vos données)
    return closest_route if min_distance <= threshold else "Route inconnue"

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

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

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
    m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
    folium.LayerControl().add_to(m)
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
    return m

def get_reprojected_and_center(uploaded_file, group):
    unique_id = str(uuid.uuid4())[:8]
    temp_path = f"uploaded_{group}_{unique_id}.tif"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
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

# Chargement des routes depuis routeQSD.txt
if os.path.exists("routeQSD.txt"):
    with open("routeQSD.txt", "r") as f:
        routes_data = json.load(f)
else:
    routes_data = {"features": []}
routes_ci = []
for feature in routes_data.get("features", []):
    if feature["geometry"]["type"] == "LineString":
        routes_ci.append({
            "coords": feature["geometry"]["coordinates"],
            "nom": feature["properties"].get("ID", "Route inconnue")
        })

# Paramètres pour l'affichage des classes et gravités
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

# ---------------------------
# SECTION 2 – DÉTECTION D'ANOMALIES
# ---------------------------
st.header("2. Détection d'anomalies")

# Initialisation de la session state pour la détection
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []  # Liste des paires { "grand":..., "petit":... }
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}  # Marqueurs par indice de paire

# Interface en onglets pour détection automatique et manuelle
tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])

with tab_auto:
    st.header("Détection Automatique")
    # Si des images issues de la configuration images existent, on les utilise sinon on propose un uploader
    if "conv_images" in st.session_state and st.session_state.conv_images:
        auto_uploaded_images = []
        for img in st.session_state.conv_images:
            auto_uploaded_images.append(io.BytesIO(img["bytes"]))
        st.success("Images issues de la conversion disponibles pour la détection automatique.")
    else:
        auto_uploaded_images = st.file_uploader("Téléversez vos images JPEG ou PNG", type=["jpeg", "jpg", "png"], accept_multiple_files=True, key="auto_images")
        if auto_uploaded_images:
            st.success("Les images ont été bien téléversées.")
        else:
            st.info("Aucune image téléversée pour la détection automatique.")
    # (La logique de détection automatique pourra être implémentée ultérieurement)

with tab_manuel:
    st.header("Détection Manuelle")
    uploaded_files_grand = st.file_uploader("Téléversez vos fichiers TIFF GRAND", type=["tif", "tiff"], accept_multiple_files=True, key="tiff_grand")
    uploaded_files_petit = st.file_uploader("Téléversez vos fichiers TIFF PETIT", type=["tif", "tiff"], accept_multiple_files=True, key="tiff_petit")
    
    if uploaded_files_grand and uploaded_files_petit:
        if len(uploaded_files_grand) != len(uploaded_files_petit):
            st.error("Le nombre de fichiers TIFF GRAND et TIFF PETIT doit être identique.")
        else:
            grand_list = []
            petit_list = []
            for file in uploaded_files_grand:
                file.seek(0)
                grand_list.append(get_reprojected_and_center(file, "grand"))
            for file in uploaded_files_petit:
                file.seek(0)
                petit_list.append(get_reprojected_and_center(file, "petit"))
            # On trie les listes pour tenter de jumeler les images
            grand_list = sorted(grand_list, key=lambda d: d["center"])
            petit_list = sorted(petit_list, key=lambda d: d["center"])
            pair_count = len(grand_list)
            pairs = []
            for i in range(pair_count):
                pairs.append({"grand": grand_list[i], "petit": petit_list[i]})
            st.session_state.pairs = pairs

            # Navigation entre paires
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            if col_nav1.button("← Précédent") and st.session_state.current_pair_index > 0:
                st.session_state.current_pair_index -= 1
            if col_nav3.button("Suivant →") and st.session_state.current_pair_index < pair_count - 1:
                st.session_state.current_pair_index += 1
            st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {pair_count}")
            current_index = st.session_state.current_pair_index
            current_pair = st.session_state.pairs[current_index]

            # Traitement de la paire courante : génération des PNG pour affichage
            reproj_grand_path = current_pair["grand"]["path"]
            with rasterio.open(reproj_grand_path) as src:
                grand_bounds = src.bounds
                data = src.read()
                if data.shape[0] >= 3:
                    r = normalize_data(data[0])
                    g = normalize_data(data[1])
                    b = normalize_data(data[2])
                    rgb_norm = np.dstack((r, g, b))
                    image_grand = Image.fromarray(rgb_norm)
                else:
                    band = data[0]
                    band_norm = normalize_data(band)
                    image_grand = Image.fromarray(band_norm, mode="L")
            unique_id = str(uuid.uuid4())[:8]
            temp_png_grand = f"converted_grand_{unique_id}.png"
            image_grand.save(temp_png_grand)
            display_path_grand = temp_png_grand

            reproj_petit_path = current_pair["petit"]["path"]
            with rasterio.open(reproj_petit_path) as src:
                petit_bounds = src.bounds
                data = src.read()
                if data.shape[0] >= 3:
                    r = normalize_data(data[0])
                    g = normalize_data(data[1])
                    b = normalize_data(data[2])
                    rgb_norm = np.dstack((r, g, b))
                    image_petit = Image.fromarray(rgb_norm)
                else:
                    band = data[0]
                    band_norm = normalize_data(band)
                    image_petit = Image.fromarray(band_norm, mode="L")
            temp_png_petit = f"converted_{unique_id}.png"
            image_petit.save(temp_png_petit)
            display_path_petit = temp_png_petit

            # Carte de dessin : TIFF GRAND (OSM masqué) pour dessiner les marqueurs
            st.subheader("Carte de dessin")
            map_placeholder_grand = st.empty()
            m_grand = create_map(
                center_lat=(grand_bounds.bottom + grand_bounds.top)/2,
                center_lon=(grand_bounds.left + grand_bounds.right)/2,
                bounds=grand_bounds,
                display_path=display_path_grand,
                marker_data=None,
                hide_osm=True,
                draw_routes=False,
                add_draw_tool=True
            )
            result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")

            # Extraction et classification des marqueurs dessinés
            features = []
            all_drawings = result_grand.get("all_drawings")
            if all_drawings:
                if isinstance(all_drawings, dict) and "features" in all_drawings:
                    features = all_drawings.get("features", [])
                elif isinstance(all_drawings, list):
                    features = all_drawings

            global_count = sum(len(markers) for markers in st.session_state.markers_by_pair.values())
            new_markers = []
            if features:
                st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
                for i, feature in enumerate(features):
                    if feature.get("geometry", {}).get("type") == "Point":
                        coords = feature.get("geometry", {}).get("coordinates")
                        if coords and isinstance(coords, list) and len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            # Correspondance des coordonnées entre les TIFF
                            percent_x = (lon - grand_bounds.left) / (grand_bounds.right - grand_bounds.left)
                            percent_y = (lat - grand_bounds.bottom) / (grand_bounds.top - grand_bounds.bottom)
                            new_lon = petit_bounds.left + percent_x * (petit_bounds.right - petit_bounds.left)
                            new_lat = petit_bounds.bottom + percent_y * (petit_bounds.top - petit_bounds.bottom)
                        else:
                            new_lon = new_lat = None
                        assigned_route = assign_route_to_marker(new_lat, new_lon, routes_ci) if new_lat and new_lon else "Route inconnue"
                        st.markdown(f"**ID {global_count + i + 1}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_class = st.selectbox("Classe", list(class_color.keys()), key=f"class_{current_index}_{i}")
                        with col2:
                            selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{i}")
                        new_markers.append({
                            "ID": global_count + i + 1,
                            "classe": selected_class,
                            "gravite": selected_gravity,
                            "lat": new_lat,
                            "long": new_lon,
                            "routes": assigned_route,
                            "detection": "Manuelle",
                            "couleur": class_color.get(selected_class, "#000000"),
                            "radius": gravity_sizes.get(selected_gravity, 5)
                        })
                st.session_state.markers_by_pair[current_index] = new_markers
            else:
                st.write("Aucun marqueur n'a été détecté.")
    else:
        st.info("Veuillez téléverser les fichiers TIFF pour lancer la détection manuelle.")

# ---------------------------
# Carte de suivi et récapitulatif global
# ---------------------------
st.subheader("Carte de suivi")
global_markers = []
for markers in st.session_state.markers_by_pair.values():
    global_markers.extend(markers)

if st.session_state.pairs:
    # Utilisation de la première paire comme référence pour la carte PETIT
    first_pair = st.session_state.pairs[0]
    try:
        with rasterio.open(first_pair["petit"]["path"]) as src:
            petit_bounds = src.bounds
    except Exception as e:
        st.error("Erreur lors de l'ouverture du TIFF PETIT pour la carte de suivi.")
        st.error(e)
        petit_bounds = None

    if petit_bounds:
        center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
        center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
        # Pour la carte de suivi, on conserve l'affichage des routes (draw_routes=True)
        m_petit = create_map(
            center_lat=center_lat_petit,
            center_lon=center_lon_petit,
            bounds=petit_bounds,
            display_path=display_path_petit if 'display_path_petit' in locals() else "",
            marker_data=global_markers,
            tiff_opacity=0,
            tiff_show=True,
            tiff_control=False,
            draw_routes=True,
            add_draw_tool=False
        )
        st_folium(m_petit, width=700, height=500, key="folium_map_petit")
    else:
        st.info("Impossible d'afficher la carte de suivi à cause d'un problème avec le TIFF PETIT.")
else:
    # Aucune donnée TIFF téléversée : affichage d'une carte par défaut avec les routes
    all_lons = []
    all_lats = []
    for route in routes_ci:
        for lon, lat in route["coords"]:
            all_lons.append(lon)
            all_lats.append(lat)
    if all_lons and all_lats:
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        # Création d'un objet "bounds" simple
        class Bounds:
            pass
        route_bounds = Bounds()
        route_bounds.left = min_lon
        route_bounds.right = max_lon
        route_bounds.bottom = min_lat
        route_bounds.top = max_lat
        center_lat_default = (min_lat + max_lat) / 2
        center_lon_default = (min_lon + max_lon) / 2
        m_default = create_map(
            center_lat_default, center_lon_default, route_bounds, display_path="",
            marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False, draw_routes=True,
            add_draw_tool=False
        )
        st_folium(m_default, width=700, height=500, key="folium_map_default")
    else:
        st.info("Aucune donnée de route disponible pour afficher la carte de suivi.")

st.markdown("### Récapitulatif global des défauts")
if global_markers:
    st.table(global_markers)
else:
    st.write("Aucun marqueur global n'a été enregistré.")
