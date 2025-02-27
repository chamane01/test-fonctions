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
import csv

#########################################
# Fonctions et définitions communes
#########################################
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
        count=3 if len(img_array.shape)==3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform
    ) as dst:
        if len(img_array.shape)==3:
            for i in range(3):
                dst.write(img_array[:, :, i], i+1)
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
        count=3 if len(img_array.shape)==3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform
    ) as dst:
        if len(img_array.shape)==3:
            for i in range(3):
                dst.write(img_array[:, :, i], i+1)
        else:
            dst.write(img_array, 1)
    return memfile.read()

def assign_route_to_marker(lat, lon, routes):
    marker_point = Point(lon, lat)
    min_distance = float('inf')
    closest_route = "Route inconnue"
    for route in routes:
        line = LineString(route["coords"])
        distance = marker_point.distance(line)
        if distance < min_distance:
            min_distance = distance
            closest_route = route["nom"]
    threshold = 0.01
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
            for i in range(1, src.count+1):
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
    return (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)

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
                tooltip=f"{marker['ID']} - {marker['classe']} (Gravité {marker['gravite']}) - Route : {marker.get('routes', 'Route inconnue')} - Détection: {marker.get('detection', 'Inconnue')} - Mission: {marker.get('mission', 'N/A')}"
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

#########################################
# Variables globales
#########################################
# Pour la détection, on conserve les marqueurs par paire
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}
# Gestion du compteur global pour les marqueurs par mission
if "mission_marker_counter" not in st.session_state:
    st.session_state.mission_marker_counter = {}

# Dictionnaire des 14 classes et des tailles de gravité
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
gravity_sizes = {1: 5, 2: 7, 3: 9}

# Chargement des routes
with open("routeQSD.txt", "r") as f:
    routes_data = json.load(f)
routes_ci = []
for feature in routes_data["features"]:
    if feature["geometry"]["type"] == "LineString":
        routes_ci.append({
            "coords": feature["geometry"]["coordinates"],
            "nom": feature["properties"].get("ID", "Route inconnue")
        })

#########################################
# Gestionnaire de missions (dans la sidebar)
#########################################
st.sidebar.header("Gestionnaire de missions")
if "missions" not in st.session_state:
    st.session_state.missions = {}

if st.sidebar.button("Créer une nouvelle mission"):
    new_mission_id = str(uuid.uuid4())[:8]
    st.session_state.missions[new_mission_id] = {"id": new_mission_id}
    st.session_state.current_mission = new_mission_id

if st.session_state.missions:
    mission_list = list(st.session_state.missions.keys())
    if "current_mission" not in st.session_state or st.session_state.current_mission not in mission_list:
        st.session_state.current_mission = mission_list[0]
    current_mission = st.sidebar.selectbox("Sélectionnez la mission", mission_list, index=mission_list.index(st.session_state.current_mission))
    st.session_state.current_mission = current_mission
else:
    st.sidebar.info("Aucune mission disponible. Créez-en une.")

#########################################
# Affichage sur une seule page
#########################################
st.title("traitements")

#########################
# Section Post-traitements des images
#########################
st.header("post-traitements des images")
uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True
)
images_info = []
if uploaded_files:
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
            if fp_unit == 2:
                sensor_width_mm = (img_width / fp_x_res) * 25.4
            elif fp_unit == 3:
                sensor_width_mm = (img_width / fp_x_res) * 10
            elif fp_unit == 4:
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
        if st.button("Générer les images prétraitées"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    # Calcul de l'angle de vol
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
                    # Conversion Configuration 1 (TIFF PETIT) avec scaling_factor = 1/5
                    tiff_bytes = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/5
                    )
                    output_filename_tiff1 = info["filename"].rsplit(".", 1)[0] + "_geotiff.tif"
                    zip_file.writestr(output_filename_tiff1, tiff_bytes)
                    # Conversion Configuration 2 (TIFF GRAND) avec scaling_factor = 1/3
                    tiff_bytes_x2 = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size * 2,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/3
                    )
                    output_filename_tiff2 = info["filename"].rsplit(".", 1)[0] + "_geotiff_x2.tif"
                    zip_file.writestr(output_filename_tiff2, tiff_bytes_x2)
                    # Conversion Configuration images (JPEG avec métadonnées)
                    rotation_angle_i = -flight_angle_i
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
                    transform_affine = T4 * T3 * T2 * T1
                    corners = [(-new_width/2, -new_height/2),
                               (new_width/2, -new_height/2),
                               (new_width/2, new_height/2),
                               (-new_width/2, new_height/2)]
                    corner_coords = []
                    for corner in corners:
                        x, y = transform_affine * corner
                        corner_coords.append((x, y))
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
                        st.error("La librairie piexif est requise pour ajouter des métadonnées JPEG. Veuillez l'installer.")
                        exif_bytes = None
                    jpeg_buffer = io.BytesIO()
                    if exif_bytes:
                        img.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
                    else:
                        img.save(jpeg_buffer, format="JPEG")
                    jpeg_bytes = jpeg_buffer.getvalue()
                    output_filename_jpeg = info["filename"].rsplit(".", 1)[0] + "_with_frame_coords.jpg"
                    zip_file.writestr(output_filename_jpeg, jpeg_bytes)
            zip_buffer.seek(0)
            st.session_state["preprocessed_zip"] = zip_buffer.getvalue()
            st.download_button(
                label="Télécharger les images prétraitées (ZIP)",
                data=zip_buffer,
                file_name="images_pretraitees.zip",
                mime="application/zip"
            )
            st.success("Vos images ont été post-traitées, vous pouvez les utilisés.")
else:
    st.info("Veuillez téléverser des images JPEG pour lancer le post-traitement.")

#####################
# Section Detections
#####################
st.markdown("---")
st.header("detections")
# Onglets pour détection automatique et détection manuelle
tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])

with tab_auto:
    st.subheader("Détection Automatique")
    if st.button("Utiliser les images converties (configuration images)"):
        if "preprocessed_zip" in st.session_state:
            zip_bytes = st.session_state["preprocessed_zip"]
            auto_converted_files = []
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
                for filename in zip_file.namelist():
                    if filename.endswith("_with_frame_coords.jpg"):
                        file_data = zip_file.read(filename)
                        file_obj = io.BytesIO(file_data)
                        file_obj.name = filename
                        auto_converted_files.append(file_obj)
            st.session_state["auto_converted_images"] = auto_converted_files
            st.success(f"{len(auto_converted_files)} images converties chargées.")
        else:
            st.error("Aucun résultat de conversion prétraitée n'est disponible.")
    # Le téléversement manuel est supprimé dans cette interface.

with tab_manuel:
    st.subheader("Détection Manuelle")
    # Bouton unique pour charger simultanément les fichiers TIFF GRAND et TIFF PETIT depuis la conversion
    if st.button("Utiliser résultats conversion TIFF (les deux configurations)"):
        if "preprocessed_zip" in st.session_state:
            zip_bytes = st.session_state["preprocessed_zip"]
            manual_grand_files = []
            manual_petit_files = []
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
                for filename in zip_file.namelist():
                    if filename.endswith("_geotiff_x2.tif"):
                        file_data = zip_file.read(filename)
                        file_obj = io.BytesIO(file_data)
                        file_obj.name = filename
                        manual_grand_files.append(file_obj)
                    elif filename.endswith("_geotiff.tif"):
                        file_data = zip_file.read(filename)
                        file_obj = io.BytesIO(file_data)
                        file_obj.name = filename
                        manual_petit_files.append(file_obj)
            if manual_grand_files and manual_petit_files:
                st.session_state["manual_grand_files"] = manual_grand_files
                st.session_state["manual_petit_files"] = manual_petit_files
                st.success(f"{len(manual_grand_files)} fichiers TIFF GRAND et {len(manual_petit_files)} fichiers TIFF PETIT chargés depuis conversion.")
            else:
                st.error("Aucun résultat de conversion prétraitée n'est disponible pour l'une ou l'autre configuration.")
        else:
            st.error("Aucun résultat de conversion prétraitée n'est disponible.")
    
    if st.session_state.get("manual_grand_files") and st.session_state.get("manual_petit_files"):
        grand_list = []
        petit_list = []
        for file in st.session_state["manual_grand_files"]:
            file.seek(0)
            grand_list.append(get_reprojected_and_center(file, "grand"))
        for file in st.session_state["manual_petit_files"]:
            file.seek(0)
            petit_list.append(get_reprojected_and_center(file, "petit"))
        grand_list = sorted(grand_list, key=lambda d: d["center"])
        petit_list = sorted(petit_list, key=lambda d: d["center"])
        pair_count = len(grand_list)
        pairs = []
        for i in range(pair_count):
            pairs.append({"grand": grand_list[i], "petit": petit_list[i]})
        st.session_state.pairs = pairs
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        prev_pressed = col_nav1.button("← Précédent")
        next_pressed = col_nav3.button("Suivant →")
        if prev_pressed and st.session_state.current_pair_index > 0:
            st.session_state.current_pair_index -= 1
        if next_pressed and st.session_state.current_pair_index < pair_count - 1:
            st.session_state.current_pair_index += 1
        st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {pair_count}")
        current_index = st.session_state.current_pair_index
        current_pair = st.session_state.pairs[current_index]
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
        utm_zone_petit = int((center_lon_petit + 180) / 6) + 1
        utm_crs_petit = f"EPSG:326{utm_zone_petit:02d}"
        st.subheader("Carte de dessin")
        m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand,
                             marker_data=None, hide_osm=True, draw_routes=False, add_draw_tool=True)
        result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")
        features = []
        all_drawings = result_grand.get("all_drawings")
        if all_drawings:
            if isinstance(all_drawings, dict) and "features" in all_drawings:
                features = all_drawings.get("features", [])
            elif isinstance(all_drawings, list):
                features = all_drawings
        # Gestion des IDs des marqueurs avec référence de mission
        current_mission = st.session_state.get("current_mission", "N/A")
        if current_mission not in st.session_state.mission_marker_counter:
            st.session_state.mission_marker_counter[current_mission] = 1
        existing_markers = st.session_state.markers_by_pair.get(current_index, [])
        updated_markers = []
        if features:
            st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
            for i, feature in enumerate(features):
                if feature.get("geometry", {}).get("type") == "Point":
                    coords = feature.get("geometry", {}).get("coordinates")
                    if coords and isinstance(coords, list) and len(coords) >= 2:
                        lon_pt, lat_pt = coords[0], coords[1]
                        percent_x = (lon_pt - grand_bounds.left) / (grand_bounds.right - grand_bounds.left)
                        percent_y = (lat_pt - grand_bounds.bottom) / (grand_bounds.top - grand_bounds.bottom)
                        new_lon = petit_bounds.left + percent_x * (petit_bounds.right - petit_bounds.left)
                        new_lat = petit_bounds.bottom + percent_y * (petit_bounds.top - petit_bounds.bottom)
                        utm_x_petit, utm_y_petit = transform("EPSG:4326", utm_crs_petit, [new_lon], [new_lat])
                        utm_coords_petit = (round(utm_x_petit[0], 2), round(utm_y_petit[0], 2))
                    else:
                        new_lon = new_lat = None
                        utm_coords_petit = "Inconnues"
                    assigned_route = assign_route_to_marker(new_lat, new_lon, routes_ci) if new_lat and new_lon else "Route inconnue"
                    if i < len(existing_markers):
                        marker_id = existing_markers[i]["ID"]
                    else:
                        marker_id = f"{current_mission}-{st.session_state.mission_marker_counter[current_mission]}"
                        st.session_state.mission_marker_counter[current_mission] += 1
                    st.markdown(f"**ID {marker_id}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class = st.selectbox("Classe", list(class_color.keys()), key=f"class_{current_index}_{marker_id}")
                    with col2:
                        selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{marker_id}")
                    updated_markers.append({
                        "ID": marker_id,
                        "classe": selected_class,
                        "gravite": selected_gravity,
                        "coordonnees UTM": utm_coords_petit,
                        "lat": new_lat,
                        "long": new_lon,
                        "routes": assigned_route,
                        "detection": "Manuelle",
                        "mission": current_mission,
                        "couleur": class_color.get(selected_class, "#000000"),
                        "radius": gravity_sizes.get(selected_gravity, 5)
                    })
            st.session_state.markers_by_pair[current_index] = updated_markers
        else:
            st.write("Aucun marqueur n'a été détecté.")
    else:
        st.info("Aucun fichier TIFF converti n'est disponible pour lancer la détection manuelle.")

st.subheader("Carte de suivi")
global_markers = []
for markers in st.session_state.markers_by_pair.values():
    global_markers.extend(markers)
if st.session_state.pairs:
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
        m_petit = create_map(center_lat_petit, center_lon_petit, petit_bounds,
                             display_path_petit if 'display_path_petit' in locals() else "",
                             marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False, draw_routes=True,
                             add_draw_tool=False)
        st_folium(m_petit, width=700, height=500, key="folium_map_petit")
    else:
        st.info("Impossible d'afficher la carte de suivi à cause d'un problème avec le TIFF PETIT.")
else:
    all_lons = []
    all_lats = []
    for route in routes_ci:
        for lon, lat in route["coords"]:
            all_lons.append(lon)
            all_lats.append(lat)
    if all_lons and all_lats:
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        class Bounds:
            pass
        route_bounds = Bounds()
        route_bounds.left = min_lon
        route_bounds.right = max_lon
        route_bounds.bottom = min_lat
        route_bounds.top = max_lat
        center_lat_default = (min_lat + max_lat) / 2
        center_lon_default = (min_lon + max_lon) / 2
        m_default = create_map(center_lat_default, center_lon_default, route_bounds, display_path="",
                               marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False, draw_routes=True,
                               add_draw_tool=False)
        st_folium(m_default, width=700, height=500, key="folium_map_default")
    else:
        st.info("Aucune donnée de route disponible pour afficher la carte de suivi.")
st.markdown("### Récapitulatif global des défauts")
if global_markers:
    st.table(global_markers)
else:
    st.write("Aucun marqueur global n'a été enregistré.")

#########################################
# Gestionnaire de missions : Export CSV
#########################################
st.markdown("---")
st.subheader("Export des résultats de la mission")
if st.button("Exporter les résultats de la mission en CSV"):
    current_mission = st.session_state.get("current_mission", None)
    if current_mission:
        # Collecter tous les marqueurs appartenant à la mission courante
        mission_markers = []
        for markers in st.session_state.markers_by_pair.values():
            for marker in markers:
                if marker.get("mission") == current_mission:
                    mission_markers.append(marker)
        if mission_markers:
            output = io.StringIO()
            writer = csv.writer(output, delimiter=';')
            # Ecrire l'en-tête
            writer.writerow(["ID", "Classe", "Gravité", "Coordonnées UTM", "Latitude", "Longitude", "Route", "Détection", "Mission"])
            # Ecrire les lignes
            for marker in mission_markers:
                writer.writerow([
                    marker.get("ID"),
                    marker.get("classe"),
                    marker.get("gravite"),
                    marker.get("coordonnees UTM"),
                    marker.get("lat"),
                    marker.get("long"),
                    marker.get("routes"),
                    marker.get("detection"),
                    marker.get("mission")
                ])
            csv_data = output.getvalue().encode('utf-8')
            st.download_button(
                label="Télécharger CSV de la mission",
                data=csv_data,
                file_name=f"mission_{current_mission}_resultats.csv",
                mime="text/csv"
            )
        else:
            st.info("Aucun marqueur n'est associé à la mission courante.")
    else:
        st.info("Aucune mission sélectionnée.")
