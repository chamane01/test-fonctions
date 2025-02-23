import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import rasterio.warp
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from rasterio.plot import reshape_as_image
from PIL import Image
import numpy as np
import base64
import uuid
import os
import matplotlib.pyplot as plt

# --- Fonctions utilitaires ---

def reproject_tiff(input_tiff, target_crs="EPSG:4326"):
    """Reprojette un fichier TIFF vers le CRS cible et renvoie le chemin du fichier reprojeté."""
    with rasterio.open(input_tiff) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
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
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return output_tiff

def apply_color_gradient(tiff_path, output_png_path):
    """Applique un gradient de couleur sur une image TIFF (par exemple MNS/MNT) et sauvegarde le résultat en PNG."""
    with rasterio.open(tiff_path) as src:
        # Lecture de la première bande (pour un MNS/MNT)
        data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        colored_image = cmap(norm(data))
        plt.imsave(output_png_path, colored_image)
        plt.close()

def add_image_overlay(map_object, image_path, bounds, layer_name, opacity=1):
    """Ajoute une image (fichier PNG) en overlay sur une carte Folium."""
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

# --- Application principale ---

st.title("Affichage de TIFF sur une carte dynamique")

# Téléversement du fichier TIFF
uploaded_file = st.file_uploader("Téléversez votre fichier TIFF", type=["tif", "tiff"])
if uploaded_file is not None:
    # Sauvegarde temporaire du fichier téléversé
    unique_file_id = str(uuid.uuid4())[:8]
    temp_tiff_path = f"uploaded_{unique_file_id}.tif"
    with open(temp_tiff_path, "wb") as f:
        f.write(uploaded_file.read())
    st.write("Fichier TIFF uploadé.")

    # Lecture du TIFF pour vérifier le CRS et récupérer ses bornes
    with rasterio.open(temp_tiff_path) as src:
        st.write("CRS du TIFF :", src.crs)
        bounds = src.bounds

    # Reprojection si le CRS n'est pas déjà EPSG:4326
    if src.crs.to_string() != "EPSG:4326":
        st.write("Reprojection vers EPSG:4326...")
        reprojected_path = reproject_tiff(temp_tiff_path, "EPSG:4326")
    else:
        reprojected_path = temp_tiff_path

    # Optionnel : appliquer un gradient de couleur pour un rendu "coloré" (utile pour des MNS/MNT)
    apply_gradient = st.checkbox("Appliquer un gradient de couleur (pour MNS/MNT)", value=False)
    if apply_gradient:
        unique_png_id = str(uuid.uuid4())[:8]
        temp_png_path = f"colored_{unique_png_id}.png"
        apply_color_gradient(reprojected_path, temp_png_path)
        display_path = temp_png_path
    else:
        # Si aucun gradient n'est appliqué, convertir le TIFF en image (RGB ou niveaux de gris)
        with rasterio.open(reprojected_path) as src:
            data = src.read()
            if data.shape[0] >= 3:
                rgb = np.dstack((data[0], data[1], data[2]))
                rgb_norm = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(np.uint8)
                image = Image.fromarray(rgb_norm)
            else:
                band = data[0]
                band_norm = (255 * (band - band.min()) / (band.max() - band.min())).astype(np.uint8)
                image = Image.fromarray(band_norm, mode="L")
        temp_png_path = f"converted_{unique_file_id}.png"
        image.save(temp_png_path)
        display_path = temp_png_path

    # Lecture des bornes du TIFF reprojeté
    with rasterio.open(reprojected_path) as src:
        bounds = src.bounds
    st.write("Bornes (EPSG:4326) :", bounds)

    # Création de la carte centrée sur le TIFF
    m = folium.Map(location=[(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2], zoom_start=10)

    # Ajout de l'overlay de l'image
    add_image_overlay(m, display_path, bounds, "TIFF Overlay", opacity=1)
    
    # Ajustement de la vue pour zoomer sur le TIFF
    m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
    
    # Ajout d'un LayerControl pour basculer l'affichage de l'overlay
    folium.LayerControl().add_to(m)
    
    st_folium(m, width=700, height=500)

    # Nettoyage des fichiers temporaires
    if os.path.exists(temp_tiff_path):
        os.remove(temp_tiff_path)
    if reprojected_path != temp_tiff_path and os.path.exists(reprojected_path):
        os.remove(reprojected_path)
    if os.path.exists(temp_png_path):
        os.remove(temp_png_path)
