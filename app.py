import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine, array_bounds
from PIL import Image, ImageOps
import numpy as np
import base64
import uuid
import os
import math
import io
import json
import exifread
from pyproj import Transformer
import zipfile
import matplotlib.pyplot as plt

###############################################
# FONCTIONS UTILITAIRES
###############################################

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

def get_tiff_metadata(tiff_bytes):
    with rasterio.MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            bounds = src.bounds
            center = ((bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2)
            transform = src.transform
            width = src.width
            height = src.height
    return {"bounds": bounds, "center": center, "transform": transform, "width": width, "height": height}

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

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

def add_image_overlay(map_object, image_data, bounds, layer_name, opacity=1, show=True, control=True):
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

def reproject_tiff(input_path, target_crs="EPSG:4326"):
    with rasterio.open(input_path) as src:
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
    return {"path": reproj_path, "center": (center_lat, center_lon), "bounds": bounds, "temp_original": temp_path}

###############################################
# FONCTION DE TRAITEMENT UNIQUE
###############################################
def process_images(uploaded_files, pixel_size):
    auto_images = []       # Pour détection automatique (JPEG avec métadonnées de cadre)
    tiff_petit_list = []   # Pour configuration manuelle (TIFF PETIT)
    tiff_grand_list = []   # Pour configuration manuelle (TIFF GRAND)
    images_info = []       # Informations et métadonnées pour chaque image

    for file in uploaded_files:
        file_bytes = file.read()
        file_buffer = io.BytesIO(file_bytes)
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
        if lat is None or lon is None:
            st.warning(f"{file.name} n'a pas de coordonnées GPS et sera ignoré.")
            continue
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
        st.error("Aucune image exploitable n'a été trouvée.")
        return

    # Calcul de l'angle de vol pour chaque image
    for i, info in enumerate(images_info):
        if len(images_info) >= 2:
            if i == 0:
                dx = images_info[1]["utm"][0] - info["utm"][0]
                dy = images_info[1]["utm"][1] - info["utm"][1]
            elif i == len(images_info) - 1:
                dx = info["utm"][0] - images_info[i-1]["utm"][0]
                dy = info["utm"][1] - images_info[i-1]["utm"][1]
            else:
                dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
            flight_angle = math.degrees(math.atan2(dx, dy))
        else:
            flight_angle = 0
        info["flight_angle"] = flight_angle

        # --- Conversion pour détection automatique (JPEG avec métadonnées de cadre) ---
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
        # Calcul des coins du cadre
        corners = [(-new_width/2, -new_height/2),
                   (new_width/2, -new_height/2),
                   (new_width/2, new_height/2),
                   (-new_width/2, new_height/2)]
        corner_coords = [transform * corner for corner in corners]
        metadata_str = f"Frame Coordinates: {corner_coords}"
        try:
            import piexif
            if "exif" in pil_img.info:
                exif_dict = piexif.load(pil_img.info["exif"])
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

        # --- Conversion pour TIFF PETIT (scaling 1/5) ---
        tiff_petit_bytes = convert_to_tiff_in_memory(
            image_file=io.BytesIO(info["data"]),
            pixel_size=pixel_size,
            utm_center=info["utm"],
            utm_crs=info["utm_crs"],
            rotation_angle=-info["flight_angle"],
            scaling_factor=1/5
        )
        tiff_petit_list.append(tiff_petit_bytes)

        # --- Conversion pour TIFF GRAND (scaling 1/3, résolution multipliée par 2) ---
        tiff_grand_bytes = convert_to_tiff_in_memory(
            image_file=io.BytesIO(info["data"]),
            pixel_size=pixel_size * 2,
            utm_center=info["utm"],
            utm_crs=info["utm_crs"],
            rotation_angle=-info["flight_angle"],
            scaling_factor=1/3
        )
        tiff_grand_list.append(tiff_grand_bytes)
    
    st.session_state.auto_images = auto_images
    st.session_state.images_info = images_info
    st.session_state.tiff_petit = tiff_petit_list
    st.session_state.tiff_grand = tiff_grand_list

    # Création des paires pour la détection manuelle (conversion des TIFF en PNG)
    pairs = []
    if st.session_state.tiff_grand and st.session_state.tiff_petit:
        for i in range(len(st.session_state.tiff_grand)):
            meta_grand = get_tiff_metadata(st.session_state.tiff_grand[i])
            meta_petit = get_tiff_metadata(st.session_state.tiff_petit[i])
            png_grand = tiff_to_png(st.session_state.tiff_grand[i])
            png_petit = tiff_to_png(st.session_state.tiff_petit[i])
            pairs.append({
                "grand": {"info": meta_grand, "png": png_grand},
                "petit": {"info": meta_petit, "png": png_petit}
            })
    st.session_state.pairs = pairs

###############################################
# INITIALISATION DU SESSION_STATE
###############################################
if "auto_images" not in st.session_state:
    st.session_state.auto_images = None
if "pairs" not in st.session_state:
    st.session_state.pairs = None
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}

###############################################
# INTERFACE UTILISATEUR
###############################################

st.title("Détection d'anomalies & Conversion d'images")
st.markdown("### Téléversez vos images JPEG (uniquement)")
uploaded_files = st.file_uploader("Sélectionnez vos images JPEG", type=["jpg", "jpeg"], accept_multiple_files=True)
pixel_size = st.number_input("Résolution spatiale (m/pixel) :", min_value=0.001, value=0.03, step=0.001, format="%.3f")

if uploaded_files and st.button("Commencer la détection"):
    process_images(uploaded_files, pixel_size)

# Création des onglets pour affichage des résultats
tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])

with tab_auto:
    st.header("Détection Automatique")
    if st.session_state.auto_images:
        st.markdown("### Images converties pour détection automatique")
        for img_info in st.session_state.auto_images:
            st.image(img_info["image_bytes"], caption=img_info["filename"], use_column_width=True)
    else:
        st.info("Les images converties s'afficheront ici après le traitement.")

with tab_manuel:
    st.header("Détection Manuelle")
    if st.session_state.pairs:
        pair_count = len(st.session_state.pairs)
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        if col_nav1.button("← Précédent", key="prev_pair"):
            if st.session_state.current_pair_index > 0:
                st.session_state.current_pair_index -= 1
        if col_nav3.button("Suivant →", key="next_pair"):
            if st.session_state.current_pair_index < pair_count - 1:
                st.session_state.current_pair_index += 1
        st.write(f"Affichage de la paire {st.session_state.current_pair_index+1} sur {pair_count}")
        current_pair = st.session_state.pairs[st.session_state.current_pair_index]
        
        # Affichage de la carte de dessin à partir du TIFF GRAND
        st.subheader("Carte de dessin (à partir du TIFF GRAND)")
        meta_grand = current_pair["grand"]["info"]["bounds"]
        m_grand = folium.Map(
            location=[(meta_grand.bottom + meta_grand.top)/2, (meta_grand.left + meta_grand.right)/2],
            zoom_start=18, tiles=None
        )
        add_image_overlay(m_grand, current_pair["grand"]["png"], meta_grand, "TIFF GRAND Overlay", opacity=1)
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
        
        # Extraction des marqueurs dessinés
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
                        # Conversion des coordonnées du TIFF GRAND vers celles du TIFF PETIT
                        bounds_grand = meta_grand
                        petit_bounds = st.session_state.pairs[st.session_state.current_pair_index]["petit"]["info"]["bounds"]
                        percent_x = (lon - bounds_grand.left) / (bounds_grand.right - bounds_grand.left)
                        percent_y = (lat - bounds_grand.bottom) / (bounds_grand.top - bounds_grand.bottom)
                        new_lon = petit_bounds.left + percent_x * (petit_bounds.right - petit_bounds.left)
                        new_lat = petit_bounds.bottom + percent_y * (petit_bounds.top - petit_bounds.bottom)
                    else:
                        new_lon = new_lat = None
                    st.markdown(f"**Marqueur {global_count + i + 1}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class = st.selectbox("Classe", list(class_color.keys()), key=f"class_{st.session_state.current_pair_index}_{i}")
                    with col2:
                        selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{st.session_state.current_pair_index}_{i}")
                    new_markers.append({
                        "ID": global_count + i + 1,
                        "classe": selected_class,
                        "gravite": selected_gravity,
                        "lat": new_lat,
                        "long": new_lon,
                        "detection": "Manuelle",
                        "couleur": class_color.get(selected_class, "#000000"),
                        "radius": gravity_sizes.get(selected_gravity, 5)
                    })
            st.session_state.markers_by_pair[st.session_state.current_pair_index] = new_markers
        else:
            st.write("Aucun marqueur dessiné détecté.")

        # Carte de suivi (affichage du TIFF PETIT en fond)
        st.subheader("Carte de suivi")
        global_markers = []
        for markers in st.session_state.markers_by_pair.values():
            global_markers.extend(markers)
        petit_bounds = st.session_state.pairs[st.session_state.current_pair_index]["petit"]["info"]["bounds"]
        center_lat = (petit_bounds.bottom + petit_bounds.top) / 2
        center_lon = (petit_bounds.left + petit_bounds.right) / 2
        m_petit = folium.Map(location=[center_lat, center_lon], zoom_start=18)
        add_image_overlay(m_petit, st.session_state.pairs[st.session_state.current_pair_index]["petit"]["png"], petit_bounds, "TIFF PETIT Overlay", opacity=0.7)
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
        st.markdown("### Récapitulatif global des défauts")
        if global_markers:
            st.table(global_markers)
        else:
            st.write("Aucun marqueur global enregistré.")
    else:
        st.info("Les images converties s'afficheront ici après le traitement manuel.")
