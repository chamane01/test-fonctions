import streamlit as st
import folium
from folium.plugins import Draw  # Plugin pour dessiner sur la carte
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from PIL import Image
import numpy as np
import base64
import uuid
import os
import matplotlib.pyplot as plt

# Dictionnaire associant une couleur à chaque classe
class_color = {
    "Classe 1": "#FF0000",
    "Classe 2": "#00FF00",
    "Classe 3": "#0000FF",
    "Classe 4": "#FFFF00",
    "Classe 5": "#FF00FF",
    "Classe 6": "#00FFFF",
    "Classe 7": "#FFA500",
    "Classe 8": "#800080",
    "Classe 9": "#008000",
    "Classe 10": "#000080",
    "Classe 11": "#FFC0CB",
    "Classe 12": "#A52A2A",
    "Classe 13": "#808080"
}
# Définition de la taille (rayon) pour chaque niveau de gravité
gravity_sizes = {1: 5, 2: 10, 3: 15}

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

def add_image_overlay(map_object, image_path, bounds, layer_name, opacity=1):
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    img_data_url = f"data:image/png;base64,{image_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_data_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=opacity,
    ).add_to(map_object)

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

def enlarge_image(image_path, factor=4):
    """Retourne une version de l'image dont la taille en pixels est multipliée par 'factor'."""
    image = Image.open(image_path)
    new_size = (image.width * factor, image.height * factor)
    enlarged_image = image.resize(new_size, Image.NEAREST)
    unique_id = str(uuid.uuid4())[:8]
    enlarged_path = f"enlarged_{unique_id}.png"
    enlarged_image.save(enlarged_path)
    return enlarged_path

def create_map(center_lat, center_lon, bounds, display_path, enlarged_path, marker_data=None):
    m = folium.Map(location=[center_lat, center_lon])
    # Calque d'origine
    add_image_overlay(m, display_path, bounds, "TIFF Overlay", opacity=1)
    # Calque avec image grossie (upscalée) : même emprise géographique
    add_image_overlay(m, enlarged_path, bounds, "TIFF Grossi 4x", opacity=0.7)
    
    # Ajout du plugin de dessin
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
    # Ajout des marqueurs classifiés si fournis
    if marker_data:
        for marker in marker_data:
            lat = marker["lat"]
            lon = marker["lon"]
            selected_class = marker["Classe"]
            selected_gravity = marker["Gravité"]
            color = class_color.get(selected_class, "#000000")
            radius = gravity_sizes.get(selected_gravity, 5)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(m)
    return m

st.title("Affichage de TIFF avec classification dynamique des marqueurs")

uploaded_file = st.file_uploader("Téléversez votre fichier TIFF", type=["tif", "tiff"])
if uploaded_file is not None:
    unique_file_id = str(uuid.uuid4())[:8]
    temp_tiff_path = f"uploaded_{unique_file_id}.tif"
    with open(temp_tiff_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write("Fichier TIFF uploadé.")

    with rasterio.open(temp_tiff_path) as src:
        st.write("CRS du TIFF :", src.crs)
        bounds = src.bounds

    if src.crs.to_string() != "EPSG:4326":
        st.write("Reprojection vers EPSG:4326...")
        reprojected_path = reproject_tiff(temp_tiff_path, "EPSG:4326")
    else:
        reprojected_path = temp_tiff_path

    apply_gradient = st.checkbox("Appliquer un gradient de couleur (pour MNS/MNT)", value=False)
    if apply_gradient:
        unique_png_id = str(uuid.uuid4())[:8]
        temp_png_path = f"colored_{unique_png_id}.png"
        apply_color_gradient(reprojected_path, temp_png_path)
        display_path = temp_png_path
    else:
        with rasterio.open(reprojected_path) as src:
            data = src.read()
            if data.shape[0] >= 3:
                r = normalize_data(data[0])
                g = normalize_data(data[1])
                b = normalize_data(data[2])
                rgb_norm = np.dstack((r, g, b))
                image = Image.fromarray(rgb_norm)
            else:
                band = data[0]
                band_norm = normalize_data(band)
                image = Image.fromarray(band_norm, mode="L")
        temp_png_path = f"converted_{unique_file_id}.png"
        image.save(temp_png_path)
        display_path = temp_png_path

    # Création d'une version "grossie" 4x (upscalée) de l'image PNG
    enlarged_path = enlarge_image(display_path, factor=4)

    with rasterio.open(reprojected_path) as src:
        bounds = src.bounds
    st.write("Bornes (EPSG:4326) :", bounds)

    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2
    # Détermination de la zone UTM (pour l'affichage des coordonnées)
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone:02d}"  # Supposons l'hémisphère nord

    # Affichage initial de la carte
    map_placeholder = st.empty()
    m = create_map(center_lat, center_lon, bounds, display_path, enlarged_path, marker_data=None)
    result = st_folium(m, width=700, height=500, key="folium_map")

    st.subheader("Classification des marqueurs")
    marker_data = []
    features = []
    all_drawings = result.get("all_drawings")
    if all_drawings:
        if isinstance(all_drawings, dict) and "features" in all_drawings:
            features = all_drawings.get("features", [])
        elif isinstance(all_drawings, list):
            features = all_drawings

    if features:
        st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
        for idx, feature in enumerate(features):
            if feature.get("geometry", {}).get("type") == "Point":
                coords = feature.get("geometry", {}).get("coordinates")
                if coords and isinstance(coords, list) and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    utm_x, utm_y = transform("EPSG:4326", utm_crs, [lon], [lat])
                    utm_coords = (round(utm_x[0], 2), round(utm_y[0], 2))
                else:
                    lat, lon, utm_coords = None, None, "Inconnues"
                st.markdown(f"**Marqueur {idx+1} (Coordonnées UTM : {utm_coords})**")
                col1, col2 = st.columns(2)
                with col1:
                    selected_class = st.selectbox("Classe", [f"Classe {i}" for i in range(1, 14)], key=f"class_{idx}")
                with col2:
                    selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{idx}")
                marker_data.append({
                    "Marqueur": idx+1,
                    "Coordonnées UTM": utm_coords,
                    "lat": lat,
                    "lon": lon,
                    "Classe": selected_class,
                    "Gravité": selected_gravity
                })
    else:
        st.write("Aucun marqueur n'a été détecté.")

    if marker_data:
        st.markdown("### Récapitulatif des marqueurs")
        st.table(marker_data)
        # Mise à jour de la carte avec les cercles classifiés
        m_updated = create_map(center_lat, center_lon, bounds, display_path, enlarged_path, marker_data=marker_data)
        map_placeholder.write(st_folium(m_updated, width=700, height=500, key="updated_map"))
    
    # Nettoyage des fichiers temporaires
    for path in [temp_tiff_path, reprojected_path, temp_png_path, enlarged_path]:
        if os.path.exists(path):
            os.remove(path)
