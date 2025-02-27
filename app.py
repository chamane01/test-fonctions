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

# Variables globales pour la détection
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}

# Exemple de dictionnaires pour les classes et tailles (à adapter selon vos besoins)
class_color = {"Classe1": "#FF0000", "Classe2": "#00FF00", "Classe3": "#0000FF"}
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
# Application principale : choix du mode
#########################################
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choisissez l'application", 
                            ("Conversion JPEG → GeoTIFF & Export JPEG", "Détection d'anomalies"))

#########################################
# Mode Conversion JPEG
#########################################
if app_mode == "Conversion JPEG → GeoTIFF & Export JPEG":
    st.title("Conversion JPEG → GeoTIFF & Export JPEG avec métadonnées de cadre")
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
            # Bouton unique pour générer et télécharger le ZIP de prétraitement
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
                        # Conversion Configuration 1 (TIFF PETIT)
                        # Facteur de redimensionnement pour configuration 1 : -5
                        tiff_bytes = convert_to_tiff_in_memory(
                            image_file=io.BytesIO(info["data"]),
                            pixel_size=pixel_size,
                            utm_center=info["utm"],
                            utm_crs=info["utm_crs"],
                            rotation_angle=-flight_angle_i,
                            scaling_factor=-5
                        )
                        output_filename_tiff1 = info["filename"].rsplit(".", 1)[0] + "_geotiff.tif"
                        zip_file.writestr(output_filename_tiff1, tiff_bytes)
                        # Conversion Configuration 2 (TIFF GRAND)
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
    else:
        st.info("Veuillez téléverser des images JPEG pour lancer la conversion.")

#########################################
# Mode Détection d'anomalies
#########################################
else:
    st.title("Détection d'anomalies")
    tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])
    with tab_auto:
        st.header("Détection Automatique")
        if st.button("Utiliser les images converties (configuration images)"):
            if "preprocessed_zip" in st.session_state:
                st.success("Images converties disponibles.")
            else:
                st.error("Aucun résultat de conversion prétraitée n'est disponible.")
    with tab_manuel:
        st.header("Détection Manuelle")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Utiliser résultats conversion TIFF GRAND (configuration 2)"):
                if "preprocessed_zip" in st.session_state:
                    st.success("Résultats TIFF GRAND disponibles.")
                else:
                    st.error("Aucun résultat de conversion prétraitée n'est disponible.")
        with col2:
            if st.button("Utiliser résultats conversion TIFF PETIT (configuration 1)"):
                if "preprocessed_zip" in st.session_state:
                    st.success("Résultats TIFF PETIT disponibles.")
                else:
                    st.error("Aucun résultat de conversion prétraitée n'est disponible.")
