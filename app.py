import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import json
from shapely.geometry import Point, LineString
from PIL import Image, ImageOps
import numpy as np
import base64
import uuid
import os
import matplotlib.pyplot as plt
import exifread
from pyproj import Transformer
import io
import math
from affine import Affine
import zipfile

###############################################
# FONCTIONS UTILES POUR LES TRAITEMENTS
###############################################

# Extraction EXIF (pour JPEG uniquement)
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

# Conversion lat/lon en UTM
def latlon_to_utm(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

# Conversion en GeoTIFF (retourne des bytes) – utilisé pour les deux configurations manuelles
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

    memfile = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(
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
        tiff_bytes = mem.read()
    return tiff_bytes

# Fonction pour extraire la métadonnée depuis des bytes TIFF
def get_tiff_metadata(tiff_bytes):
    with rasterio.MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            bounds = src.bounds
            center = ((bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2)
            transform = src.transform
            width = src.width
            height = src.height
    return {"bounds": bounds, "center": center, "transform": transform, "width": width, "height": height}

# Normalisation d'une image (pour affichage via folium)
def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

# Conversion d'un TIFF (en bytes) en PNG (en bytes) pour affichage
def tiff_to_png(tiff_bytes):
    with rasterio.MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            data = src.read()
            if data.shape[0] >= 3:
                r = normalize_data(data[0])
                g = normalize_data(data[1])
                b = normalize_data(data[2])
                rgb = np.dstack((r, g, b))
                img = Image.fromarray(rgb)
            else:
                band = data[0]
                band_norm = normalize_data(band)
                img = Image.fromarray(band_norm, mode="L")
            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG")
            return png_buffer.getvalue()

# Modification de add_image_overlay pour accepter directement des bytes PNG
def add_image_overlay(map_object, image_data, bounds, layer_name, opacity=1, show=True, control=True):
    # Si image_data est de type bytes, on l'encode directement
    if isinstance(image_data, bytes):
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        img_data_url = f"data:image/png;base64,{image_base64}"
    else:
        with open(image_data, "rb") as f:
            content = f.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
        img_data_url = f"data:image/png;base64,{image_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_data_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=opacity,
        show=show,
        control=control
    ).add_to(map_object)

###############################################
# FONCTIONS POUR LA CONVERSION CENTRALISÉE
###############################################
# Cette fonction parcourt les images uploadées et crée :
# - Une version "configuration images" (JPEG avec métadonnées de cadre) pour la détection automatique
# - Une version TIFF PETIT (config 1) et une version TIFF GRAND (config 2) pour la détection manuelle
def process_images(uploaded_files, pixel_size):
    auto_images = []       # Pour la détection automatique
    tiff_petit_list = []   # Configuration 1 (TIFF PETIT)
    tiff_grand_list = []   # Configuration 2 (TIFF GRAND)
    images_info = []       # Métadonnées de chaque image

    # Lecture et extraction des informations pour chaque image
    for file in uploaded_files:
        file_bytes = file.read()
        file_buffer = io.BytesIO(file_bytes)
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
        if lat is None or lon is None:
            continue  # On ignore les images sans GPS
        img = Image.open(io.BytesIO(file_bytes))
        img_width, img_height = img.size
        utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
        images_info.append({
            "filename": file.name,
            "data": file_bytes,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "focal_length": focal_length,
            "img_width": img_width,
            "img_height": img_height,
            "utm": (utm_x, utm_y),
            "utm_crs": utm_crs
        })
    if len(images_info) == 0:
        st.warning("Aucune image avec coordonnées GPS n'a été trouvée.")
        return

    # Calcul de l'angle de vol pour chaque image (si plusieurs images sont fournies)
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
            flight_angle = math.degrees(math.atan2(dx, dy))
        else:
            flight_angle = 0
        info["flight_angle"] = flight_angle

        # --- Conversion "configuration images" (détection automatique) ---
        # Pas de redimensionnement, rotation en fonction de l'angle de vol
        scaling_factor = 1
        rotation_angle = -info["flight_angle"]
        pil_img = Image.open(io.BytesIO(info["data"]))
        pil_img = ImageOps.exif_transpose(pil_img)
        orig_width, orig_height = pil_img.size
        new_width = int(orig_width * scaling_factor)
        new_height = int(orig_height * scaling_factor)
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        effective_pixel_size = pixel_size / scaling_factor
        center_x, center_y = info["utm"]
        T1 = Affine.translation(-new_width/2, -new_height/2)
        T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
        T3 = Affine.rotation(rotation_angle)
        T4 = Affine.translation(center_x, center_y)
        transform = T4 * T3 * T2 * T1
        # Calcul des 4 coins pour ajouter des métadonnées
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
            if "exif" in pil_img.info:
                exif_dict = piexif.load(pil_img.info["exif"])
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
            pil_img.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
        else:
            pil_img.save(jpeg_buffer, format="JPEG")
        auto_images.append({
            "filename": info["filename"],
            "image_bytes": jpeg_buffer.getvalue()
        })

        # --- Conversion configuration 1 : TIFF PETIT ---
        tiff_petit_bytes = convert_to_tiff_in_memory(
            image_file=io.BytesIO(info["data"]),
            pixel_size=pixel_size,
            utm_center=info["utm"],
            utm_crs=info["utm_crs"],
            rotation_angle=-info["flight_angle"],
            scaling_factor=1/5
        )
        tiff_petit_list.append(tiff_petit_bytes)

        # --- Conversion configuration 2 : TIFF GRAND ---
        tiff_grand_bytes = convert_to_tiff_in_memory(
            image_file=io.BytesIO(info["data"]),
            pixel_size=pixel_size * 2,
            utm_center=info["utm"],
            utm_crs=info["utm_crs"],
            rotation_angle=-info["flight_angle"],
            scaling_factor=1/3
        )
        tiff_grand_list.append(tiff_grand_bytes)
    
    # Stockage dans le session_state
    st.session_state.auto_images = auto_images
    st.session_state.tiff_petit = tiff_petit_list
    st.session_state.tiff_grand = tiff_grand_list
    st.session_state.images_info = images_info

###############################################
# INITIALISATION DU SESSION_STATE
###############################################
if "auto_images" not in st.session_state:
    st.session_state.auto_images = None
if "tiff_petit" not in st.session_state:
    st.session_state.tiff_petit = None
if "tiff_grand" not in st.session_state:
    st.session_state.tiff_grand = None
if "pairs" not in st.session_state:
    st.session_state.pairs = None
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0

###############################################
# INTERFACE : UPLOAD CENTRALISÉ & DÉMARRAGE
###############################################
st.title("Détection d'anomalies & Conversion d'images")
st.markdown("### Téléversez vos images JPEG (uniquement)")
uploaded_files = st.file_uploader("Sélectionnez vos images JPEG", type=["jpg", "jpeg"], accept_multiple_files=True)

# Choix de la résolution spatiale appliquée à toutes les conversions
pixel_size = st.number_input("Résolution spatiale (m/pixel) :", min_value=0.001, value=0.03, step=0.001, format="%.3f")

# Bouton pour lancer la détection (les transformations s'appliquent et les résultats sont stockés en mémoire)
if uploaded_files and st.button("Commencer la détection"):
    process_images(uploaded_files, pixel_size)
    # Pour la détection manuelle, on crée des paires (on suppose ici qu'il y a autant d'images pour chaque config)
    pairs = []
    if st.session_state.tiff_grand and st.session_state.tiff_petit:
        for i in range(len(st.session_state.tiff_grand)):
            meta_grand = get_tiff_metadata(st.session_state.tiff_grand[i])
            meta_petit = get_tiff_metadata(st.session_state.tiff_petit[i])
            # Convertit les TIFF en PNG pour affichage dans la carte
            png_grand = tiff_to_png(st.session_state.tiff_grand[i])
            png_petit = tiff_to_png(st.session_state.tiff_petit[i])
            pairs.append({
                "grand": {"tiff": st.session_state.tiff_grand[i], "metadata": meta_grand, "png": png_grand},
                "petit": {"tiff": st.session_state.tiff_petit[i], "metadata": meta_petit, "png": png_petit}
            })
    st.session_state.pairs = pairs

###############################################
# ONGLETS : DÉTECTION AUTOMATIQUE & MANUELLE
###############################################
tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])

###############################################
# Détection Automatique : Affichage direct des images converties (configuration images)
###############################################
with tab_auto:
    st.header("Détection Automatique")
    if st.session_state.auto_images:
        st.markdown("Les images converties (JPEG avec métadonnées de cadre) sont affichées ci-dessous :")
        for img_info in st.session_state.auto_images:
            st.image(img_info["image_bytes"], caption=img_info["filename"], use_column_width=True)
    else:
        st.info("Les images converties s'afficheront ici après avoir cliqué sur 'Commencer la détection'.")

###############################################
# Détection Manuelle : Utilisation des TIFF PETIT et GRAND pour la saisie manuelle sur carte
###############################################
with tab_manuel:
    st.header("Détection Manuelle")
    if st.session_state.pairs:
        # Navigation entre paires (si plusieurs images)
        pair_count = len(st.session_state.pairs)
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
        
        # Extraction des métadonnées depuis le TIFF GRAND (pour affichage de la carte de dessin)
        meta_grand = current_pair["grand"]["metadata"]
        # La carte de dessin utilise le PNG issu du TIFF GRAND
        st.subheader("Carte de dessin (à partir du TIFF GRAND)")
        m_grand = folium.Map(
            location=[(meta_grand["bounds"].bottom + meta_grand["bounds"].top)/2,
                      (meta_grand["bounds"].left + meta_grand["bounds"].right)/2],
            zoom_start=18, tiles=None
        )
        # Ajout de l'overlay du TIFF GRAND converti en PNG
        add_image_overlay(m_grand, current_pair["grand"]["png"], meta_grand["bounds"], "TIFF GRAND Overlay", opacity=1)
        # Ajout de l'outil de dessin
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
        draw.add_to(m_grand)
        folium.LayerControl().add_to(m_grand)
        result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")
        
        # Extraction et classification des marqueurs dessinés
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
            # Dictionnaire de couleurs pour les classes (peut être adapté)
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
            for i, feature in enumerate(features):
                if feature.get("geometry", {}).get("type") == "Point":
                    coords = feature.get("geometry", {}).get("coordinates")
                    if coords and isinstance(coords, list) and len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        # Pour simplifier, on réinterprète les coordonnées sur le TIFF GRAND et on
                        # estime leur position relative au TIFF PETIT via un simple proportionnement
                        bounds_grand = meta_grand["bounds"]
                        meta_petit = current_pair["petit"]["metadata"]
                        bounds_petit = meta_petit["bounds"]
                        percent_x = (lon - bounds_grand.left) / (bounds_grand.right - bounds_grand.left)
                        percent_y = (lat - bounds_grand.bottom) / (bounds_grand.top - bounds_grand.bottom)
                        new_lon = bounds_petit.left + percent_x * (bounds_petit.right - bounds_petit.left)
                        new_lat = bounds_petit.bottom + percent_y * (bounds_petit.top - bounds_petit.bottom)
                    else:
                        new_lon = new_lat = None
                    # Ici, on peut ajouter une fonction d'association de route si nécessaire
                    assigned_route = "Route inconnue"
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
            st.write("Aucun marqueur n'a été détecté sur la carte de dessin.")

        # Carte de suivi avec les marqueurs globaux
        st.subheader("Carte de suivi")
        global_markers = []
        for markers in st.session_state.markers_by_pair.values():
            global_markers.extend(markers)
        if current_pair["petit"]:
            meta_petit = current_pair["petit"]["metadata"]
            center_lat_petit = (meta_petit["bounds"].bottom + meta_petit["bounds"].top) / 2
            center_lon_petit = (meta_petit["bounds"].left + meta_petit["bounds"].right) / 2
            m_petit = folium.Map(location=[center_lat_petit, center_lon_petit], zoom_start=18)
            add_image_overlay(m_petit, current_pair["petit"]["png"], meta_petit["bounds"], "TIFF PETIT Overlay", opacity=0.7)
            # Ajout des marqueurs
            for marker in global_markers:
                folium.CircleMarker(
                    location=[marker["lat"], marker["long"]],
                    radius=marker["radius"],
                    color=marker["couleur"],
                    fill=True,
                    fill_color=marker["couleur"],
                    fill_opacity=0.7,
                    tooltip=f'{marker["classe"]} (Gravité {marker["gravite"]})'
                ).add_to(m_petit)
            folium.LayerControl().add_to(m_petit)
            st_folium(m_petit, width=700, height=500, key="folium_map_petit")
        else:
            st.info("Impossible d'afficher la carte de suivi.")
        
        st.markdown("### Récapitulatif global des défauts")
        if global_markers:
            st.table(global_markers)
        else:
            st.write("Aucun marqueur global n'a été enregistré.")
    else:
        st.info("Les images converties s'afficheront ici après avoir cliqué sur 'Commencer la détection'.")
