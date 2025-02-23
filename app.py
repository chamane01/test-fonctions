import streamlit as st
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
from PIL import Image
import folium
from streamlit_folium import st_folium
import io
import base64

st.title("Affichage d'un TIFF sur une carte dynamique")

# Téléversement du fichier TIFF
uploaded_file = st.file_uploader("Choisissez un fichier TIFF", type=["tif", "tiff"])

if uploaded_file is not None:
    # Ouverture du TIFF avec rasterio
    with rasterio.open(uploaded_file) as src:
        st.write("Système de référence (CRS) du TIFF :", src.crs)
        st.write("Dimensions :", src.width, "x", src.height)
        
        # Récupération et transformation des bornes en WGS84 si nécessaire
        bounds = src.bounds
        if src.crs.to_string() != 'EPSG:4326':
            bounds_wgs84 = transform_bounds(src.crs, 'EPSG:4326',
                                            bounds.left, bounds.bottom,
                                            bounds.right, bounds.top)
        else:
            bounds_wgs84 = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        st.write("Bornes (WGS84) :", bounds_wgs84)
        
        # Lecture des données du TIFF
        data = src.read()
        if data.shape[0] >= 3:
            # Supposons qu'il s'agisse d'un TIFF RGB
            rgb = np.dstack((data[0], data[1], data[2]))
            rgb_norm = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(np.uint8)
            image = Image.fromarray(rgb_norm)
        else:
            # Cas d'une seule bande (niveaux de gris)
            band = data[0]
            band_norm = (255 * (band - band.min()) / (band.max() - band.min())).astype(np.uint8)
            image = Image.fromarray(band_norm, mode="L")
        
        # Sauvegarde de l'image en mémoire au format PNG
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Conversion de l'image en data URL (base64)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
    
    # Calcul du centre de la zone pour centrer la carte
    center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
    center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2
    st.write("Centre de la carte :", center_lat, center_lon)
    
    # Création de la carte avec un zoom initial adapté
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Possibilité de choisir le paramètre 'origin' pour tester l'orientation de l'image
    origin = st.selectbox("Choisir l'origine de l'image", options=["upper", "lower"], index=0)
    
    # Ajout de l'image en overlay sur la carte avec opacité 100%
    folium.raster_layers.ImageOverlay(
        image=img_data_url,
        bounds=[[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]],
        opacity=1,  # opacité à 100%
        origin=origin,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    
    # Affichage de la carte dans Streamlit
    st_folium(m, width=700, height=500)
