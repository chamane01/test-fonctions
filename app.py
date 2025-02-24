import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from PIL import Image
import numpy as np
import base64
import uuid
import os
import subprocess
import matplotlib.pyplot as plt

# Dictionnaire associant une couleur à chacune des 13 classes
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

# Définition de la taille (rayon en pixels) pour chaque niveau de gravité
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

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

def generate_tiles(tiff_path, output_dir, zoom_range="0-22"):
    """
    Génère des tuiles à partir du TIFF en utilisant gdal2tiles.py.
    """
    cmd = f"gdal2tiles.py -z {zoom_range} {tiff_path} {output_dir}"
    subprocess.run(cmd, shell=True)

def create_map(center_lat, center_lon, bounds, tiles_dir, marker_data=None):
    # Création de la carte avec un zoom initial élevé et max_zoom fixé à 22
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, max_zoom=22)
    # Ajout des tuiles générées par gdal2tiles
    folium.TileLayer(
        tiles=f"{tiles_dir}/{{z}}/{{x}}/{{y}}.png",
        attr="Tuiles générées par gdal2tiles",
        name="TIFF Tiles",
        overlay=True,
        control=True,
        max_zoom=22
    ).add_to(m)
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

st.title("Affichage de TIFF avec tuiles haute résolution et classification des marqueurs")

# Téléversement du TIFF
uploaded_file = st.file_uploader("Téléversez votre fichier TIFF", type=["tif", "tiff"])
if uploaded_file is not None:
    unique_file_id = str(uuid.uuid4())[:8]
    temp_tiff_path = f"uploaded_{unique_file_id}.tif"
    with open(temp_tiff_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write("Fichier TIFF uploadé.")

    # Lecture du TIFF et récupération des bornes
    with rasterio.open(temp_tiff_path) as src:
        st.write("CRS du TIFF :", src.crs)
        bounds = src.bounds

    # Reprojection si nécessaire
    if src.crs.to_string() != "EPSG:4326":
        st.write("Reprojection vers EPSG:4326...")
        reprojected_path = reproject_tiff(temp_tiff_path, "EPSG:4326")
    else:
        reprojected_path = temp_tiff_path

    # Option de gradient de couleur
    apply_gradient = st.checkbox("Appliquer un gradient de couleur (pour MNS/MNT)", value=False)
    if apply_gradient:
        unique_png_id = str(uuid.uuid4())[:8]
        temp_png_path = f"colored_{unique_png_id}.png"
        apply_color_gradient(reprojected_path, temp_png_path)
        # Pour le tiling, on travaille directement sur le TIFF reprojeté
    else:
        # Si pas de gradient, on peut laisser le TIFF tel quel pour la génération des tuiles
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
        unique_png_id = str(uuid.uuid4())[:8]
        temp_png_path = f"converted_{unique_png_id}.png"
        image.save(temp_png_path)
        # Nous n'utilisons pas ce PNG pour le tiling, nous utilisons le TIFF reprojeté.

    # Génération des tuiles à partir du TIFF reprojeté
    tiles_dir = f"tiles_{unique_file_id}"
    if not os.path.exists(tiles_dir):
        st.info("Génération des tuiles en cours (cela peut prendre quelques instants)...")
        generate_tiles(reprojected_path, tiles_dir, zoom_range="0-22")
        st.success("Tuiles générées.")

    # Récupération des bornes (à nouveau, au cas où)
    with rasterio.open(reprojected_path) as src:
        bounds = src.bounds
    st.write("Bornes (EPSG:4326) :", bounds)

    # Calcul du centre et détermination de la zone UTM (pour affichage dans le bilan)
    center_lat = (bounds.bottom + bounds.top) / 2
    center_lon = (bounds.left + bounds.right) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone:02d}"  # Pour l'hémisphère nord

    # Affichage initial de la carte avec les tuiles
    map_placeholder = st.empty()
    m = create_map(center_lat, center_lon, bounds, tiles_dir, marker_data=None)
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
        m_updated = create_map(center_lat, center_lon, bounds, tiles_dir, marker_data=marker_data)
        map_placeholder.write(st_folium(m_updated, width=700, height=500, key="updated_map"))
    
    # Nettoyage des fichiers temporaires
    if os.path.exists(temp_tiff_path):
        os.remove(temp_tiff_path)
    if reprojected_path != temp_tiff_path and os.path.exists(reprojected_path):
        os.remove(reprojected_path)
    # Vous pouvez aussi choisir de conserver le dossier de tuiles pour une utilisation ultérieure
