import streamlit as st
import folium
from folium.plugins import Draw  # Plugin pour dessiner sur la carte
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from PIL import Image, ImageOps
import numpy as np
import base64
import uuid
import os
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString
import io
import math
import exifread
from pyproj import Transformer
from affine import Affine
import zipfile

###############################################
# FONCTIONS & VARIABLES – Détection d’anomalies
###############################################

# Dictionnaire de couleurs et tailles pour les classes de défauts
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

# Chargement et gestion des routes depuis un fichier JSON (routeQSD.txt)
with open("routeQSD.txt", "r") as f:
    routes_data = json.load(f)
routes_ci = []
for feature in routes_data["features"]:
    if feature["geometry"]["type"] == "LineString":
        routes_ci.append({
            "coords": feature["geometry"]["coordinates"],
            "nom": feature["properties"].get("ID", "Route inconnue")
        })

def assign_route_to_marker(lat, lon, routes):
    """
    Pour un point (lat, lon) en EPSG:4326, retourne le nom de la route la plus proche,
    si celle‑ci se trouve dans un seuil défini (sinon "Route inconnue").
    """
    marker_point = Point(lon, lat)  # Note : Point(longitude, latitude)
    min_distance = float('inf')
    closest_route = "Route inconnue"
    for route in routes:
        line = LineString(route["coords"])  # Coordonnées au format [lon, lat]
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

def apply_color_gradient(tiff_path, output_png_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        colored_image = cmap(norm(data))
        plt.imsave(output_png_path, colored_image)
        plt.close()

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

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

def create_map(center_lat, center_lon, bounds, display_path, marker_data=None,
               hide_osm=False, tiff_opacity=1, tiff_show=True, tiff_control=True,
               draw_routes=True, add_draw_tool=True):
    """
    Création de la carte avec option d'afficher ou non les routes.
    Le paramètre add_draw_tool contrôle l'ajout de l'outil de dessin (pour placer des marqueurs).
    """
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

###############################################
# FONCTIONS – Conversion & Extraction EXIF (position, trajectoire, etc.)
###############################################
def extract_exif_info(image_file):
    """
    Extrait les informations EXIF (coordonnées GPS, altitude, focale, etc.)
    à partir d'un objet fichier.
    """
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
    """
    Convertit des coordonnées latitude/longitude en coordonnées UTM.
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

def compute_gsd(altitude, focal_length_mm, sensor_width_mm, image_width_px):
    """
    Calcule la Ground Sample Distance (GSD) en mètre par pixel.
    """
    focal_length_m = focal_length_mm / 1000.0
    sensor_width_m = sensor_width_mm / 1000.0
    return (altitude * sensor_width_m) / (focal_length_m * image_width_px)

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
    transform = T4 * T3 * T2 * T1

    with rasterio.open(
        output_path, 'w',
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

###############################################
# INITIALISATION DE LA SESSION STATE
###############################################
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []  # Liste des paires { "grand":..., "petit":... }
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}  # Marqueurs par indice de paire

# Variables de stockage pour les conversions avancées
# Elles seront alimentées par les boutons de configuration
if "config_images" not in st.session_state:
    st.session_state.config_images = None
if "tiff_petit" not in st.session_state:
    st.session_state.tiff_petit = None
if "tiff_grand" not in st.session_state:
    st.session_state.tiff_grand = None

###############################################
# INTERFACE STREAMLIT – Affichage global
###############################################

st.title("Traitements")

###############################################
# SECTION – Conversion avancée (avec logique de trajectoire & position)
###############################################
st.header("Conversion avancée d'images (JPEG → GeoTIFF & JPEG enrichis)")

uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True,
    key="conv_avancee_images"
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
        
        # Calcul de la largeur du capteur (en mm) si possible
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
        pixel_size = st.number_input(
            "Choisissez la résolution spatiale (m/pixel) :", 
            min_value=0.001, 
            value=0.03, 
            step=0.001, 
            format="%.3f"
        )
        st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
        
        # -------------------------------
        # Configuration 1 : Export groupé en GeoTIFF (scaling_factor = 1/5)
        if st.button("configuration 1"):
            tiff_petit_list = []
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    if len(images_info) >= 2:
                        if i == 0:
                            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                        elif i == len(images_info) - 1:
                            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                        else:
                            dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                            dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                        flight_angle_i = math.degrees(math.atan2(dx, dy))
                    else:
                        flight_angle_i = 0
                    tiff_bytes = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/5
                    )
                    output_filename = info["filename"].rsplit(".", 1)[0] + "_geotiff.tif"
                    zip_file.writestr(output_filename, tiff_bytes)
                    # Sauvegarde individuelle pour usage en détection manuelle (TIFF PETIT)
                    tiff_petit_list.append({"filename": output_filename, "data": tiff_bytes})
            zip_buffer.seek(0)
            st.download_button(
                label="Télécharger toutes les images GeoTIFF (configuration 1)",
                data=zip_buffer,
                file_name="images_geotiff.zip",
                mime="application/zip"
            )
            st.session_state.tiff_petit = tiff_petit_list
        
        # -------------------------------
        # Configuration 2 : Export groupé en GeoTIFF x2 (scaling_factor = 1/3, pixel_size multiplié par 2)
        if st.button("configuration 2"):
            tiff_grand_list = []
            zip_buffer_geotiff_x2 = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_geotiff_x2, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    if len(images_info) >= 2:
                        if i == 0:
                            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                        elif i == len(images_info) - 1:
                            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                        else:
                            dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                            dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                        flight_angle_i = math.degrees(math.atan2(dx, dy))
                    else:
                        flight_angle_i = 0
                    tiff_bytes_x2 = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size * 2,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/3
                    )
                    output_filename = info["filename"].rsplit(".", 1)[0] + "_geotiff_x2.tif"
                    zip_file.writestr(output_filename, tiff_bytes_x2)
                    tiff_grand_list.append({"filename": output_filename, "data": tiff_bytes_x2})
            zip_buffer_geotiff_x2.seek(0)
            st.download_button(
                label="Télécharger toutes les images GeoTIFF x2 (configuration 2)",
                data=zip_buffer_geotiff_x2,
                file_name="images_geotiff_x2.zip",
                mime="application/zip"
            )
            st.session_state.tiff_grand = tiff_grand_list
        
        # -------------------------------
        # Configuration images : Export groupé en JPEG avec métadonnées de cadre
        if st.button("configuration images"):
            config_images_list = []
            zip_buffer_jpeg = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_jpeg, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    if len(images_info) >= 2:
                        if i == 0:
                            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                        elif i == len(images_info) - 1:
                            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                        else:
                            dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                            dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                        flight_angle_i = math.degrees(math.atan2(dx, dy))
                    else:
                        flight_angle_i = 0
                    rotation_angle_i = -flight_angle_i
                    
                    # Pas de redimensionnement (scaling_factor = 1) pour configuration images
                    scaling_factor = 1
                    img = Image.open(io.BytesIO(info["data"]))
                    img = ImageOps.exif_transpose(img)
                    orig_width, orig_height = img.size
                    new_width = int(orig_width * scaling_factor)
                    new_height = int(orig_height * scaling_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    effective_pixel_size = pixel_size / scaling_factor
                    center_x, center_y = info["utm"]
                    T1 = Affine.translation(-new_width/2, -new_height/2)
                    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
                    T3 = Affine.rotation(rotation_angle_i)
                    T4 = Affine.translation(center_x, center_y)
                    transform = T4 * T3 * T2 * T1
                    
                    # Calcul des coordonnées des 4 coins du cadre
                    corners = [
                        (-new_width/2, -new_height/2),
                        (new_width/2, -new_height/2),
                        (new_width/2, new_height/2),
                        (-new_width/2, new_height/2)
                    ]
                    corner_coords = []
                    for corner in corners:
                        x, y = transform * corner
                        corner_coords.append((x, y))
                    
                    metadata_str = f"Frame Coordinates: {corner_coords}"
                    
                    try:
                        import piexif
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
                    except ImportError:
                        st.error("La librairie piexif est requise pour ajouter des métadonnées JPEG. Veuillez l'installer.")
                        exif_bytes = None
                    
                    jpeg_buffer = io.BytesIO()
                    if exif_bytes:
                        img.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
                    else:
                        img.save(jpeg_buffer, format="JPEG")
                    jpeg_bytes = jpeg_buffer.getvalue()
                    output_filename = info["filename"].rsplit(".", 1)[0] + "_with_frame_coords.jpg"
                    zip_file.writestr(output_filename, jpeg_bytes)
                    config_images_list.append({"filename": output_filename, "data": jpeg_bytes})
            zip_buffer_jpeg.seek(0)
            st.download_button(
                label="Télécharger toutes les images JPEG avec métadonnées de cadre (configuration images)",
                data=zip_buffer_jpeg,
                file_name="images_with_frame_coords.zip",
                mime="application/zip"
            )
            st.session_state.config_images = config_images_list
else:
    st.info("Veuillez téléverser au moins une image pour la conversion avancée.")

###############################################
# ONGLETS – Détection Automatique et Manuelle
###############################################
tabs = st.tabs(["Détection Automatique", "Détection Manuelle"])

with tabs[0]:
    st.header("Détection Automatique")
    if st.button("Utiliser configuration images"):
        if st.session_state.config_images:
            auto_images = st.session_state.config_images
            st.success(f"{len(auto_images)} images chargées depuis configuration images.")
            for img_info in auto_images:
                st.image(img_info["data"], caption=img_info["filename"])
        else:
            st.error("Aucune image disponible dans configuration images. Veuillez utiliser la configuration images dans la section Traitements.")
    else:
        st.info("Cliquez sur 'Utiliser configuration images' pour charger les images converties.")

with tabs[1]:
    st.header("Détection Manuelle")
    col1, col2 = st.columns(2)
    if col1.button("Utiliser configuration 2 (TIFF GRAND)"):
        if st.session_state.tiff_grand:
            tiff_grand_files = st.session_state.tiff_grand
            st.success(f"{len(tiff_grand_files)} fichiers TIFF GRAND chargés.")
        else:
            st.error("Aucun fichier TIFF GRAND disponible. Veuillez utiliser la configuration 2 dans la section Traitements.")
    else:
        tiff_grand_files = None
    if col2.button("Utiliser configuration 1 (TIFF PETIT)"):
        if st.session_state.tiff_petit:
            tiff_petit_files = st.session_state.tiff_petit
            st.success(f"{len(tiff_petit_files)} fichiers TIFF PETIT chargés.")
        else:
            st.error("Aucun fichier TIFF PETIT disponible. Veuillez utiliser la configuration 1 dans la section Traitements.")
    else:
        tiff_petit_files = None

    # Si les deux listes sont disponibles et de même longueur, on crée les paires
    if tiff_grand_files and tiff_petit_files:
        if len(tiff_grand_files) != len(tiff_petit_files):
            st.error("Le nombre de fichiers TIFF GRAND et TIFF PETIT n'est pas identique.")
        else:
            pairs = []
            # Pour chaque paire, on sauvegarde temporairement les fichiers TIFF dans le disque pour créer la structure attendue
            for i in range(len(tiff_grand_files)):
                # Sauvegarde temporaire
                grand_path = f"temp_grand_{i}_{str(uuid.uuid4())[:8]}.tif"
                petit_path = f"temp_petit_{i}_{str(uuid.uuid4())[:8]}.tif"
                with open(grand_path, "wb") as f:
                    f.write(tiff_grand_files[i]["data"])
                with open(petit_path, "wb") as f:
                    f.write(tiff_petit_files[i]["data"])
                # Utilisation de get_reprojected_and_center pour obtenir centre et bounds
                grand_info = get_reprojected_and_center(open(grand_path, "rb"), "grand")
                petit_info = get_reprojected_and_center(open(petit_path, "rb"), "petit")
                pairs.append({"grand": grand_info, "petit": petit_info})
            st.session_state.pairs = pairs
            st.write(f"Création de {len(pairs)} paires de TIFF pour la détection manuelle.")
            # Poursuite du traitement (navigation entre paires, affichage de carte, etc.)
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            prev_pressed = col_nav1.button("← Précédent")
            next_pressed = col_nav3.button("Suivant →")
            if prev_pressed and st.session_state.current_pair_index > 0:
                st.session_state.current_pair_index -= 1
            if next_pressed and st.session_state.current_pair_index < len(pairs) - 1:
                st.session_state.current_pair_index += 1
            st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {len(pairs)}")
            current_index = st.session_state.current_pair_index
            current_pair = st.session_state.pairs[current_index]

            # Traitement de la paire courante pour affichage sur la carte
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

            center_lat_grand = (grand_bounds.bottom + grand_bounds.top) / 2
            center_lon_grand = (grand_bounds.left + grand_bounds.right) / 2
            center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
            center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2

            st.subheader("Carte de dessin")
            map_placeholder_grand = st.empty()
            m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand,
                                 marker_data=None, hide_osm=True, draw_routes=False, add_draw_tool=True)
            result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")
            # Extraction et classification des marqueurs reste inchangé...
    else:
        st.info("Utilisez les boutons ci‑dessus pour charger les fichiers issus des configurations avancées.")

