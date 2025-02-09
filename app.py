import streamlit as st
import rasterio
import numpy as np
import folium
from streamlit_folium import folium_static
from rasterio.plot import show
from pyproj import Transformer
import json
from shapely.geometry import Point, LineString

# Fonction pour charger une image GeoTIFF
def load_tiff(file):
    dataset = rasterio.open(file)
    return dataset

# Fonction pour reprojeter en UTM
def reproject_to_utm(dataset):
    transformer = Transformer.from_crs(dataset.crs, "EPSG:32633", always_xy=True)  # Exemple UTM Zone 33N
    return transformer

# Interface utilisateur Streamlit
st.title("Éditeur de TIFF Géoréférencé")
uploaded_file = st.file_uploader("Charger une image TIFF", type=["tif", "tiff"], accept_multiple_files=False)

if uploaded_file:
    dataset = load_tiff(uploaded_file)
    transformer = reproject_to_utm(dataset)
    bounds = dataset.bounds
    
    # Carte interactive
    m = folium.Map(location=[(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2], zoom_start=12)
    folium_static(m)
    
    # Dessin interactif (points, lignes, polylignes)
    st.sidebar.header("Dessiner sur l'image")
    draw_mode = st.sidebar.radio("Sélectionnez un mode de dessin", ["Point", "Ligne", "Polyligne"])
    coordinates = st.sidebar.text_area("Entrez les coordonnées (lat, lon) séparées par des virgules")
    
    if st.sidebar.button("Ajouter l'annotation"):
        coords = [tuple(map(float, c.split(','))) for c in coordinates.split('\n') if c]
        if draw_mode == "Point":
            geom = Point(coords[0])
        elif draw_mode == "Ligne":
            geom = LineString(coords)
        else:
            geom = LineString(coords)  # Pour simplifier, une polyligne est une ligne
        
        # Reprojection vers UTM
        utm_coords = [transformer.transform(*c) for c in coords]
        
        # Export en GeoJSON
        geojson_data = json.dumps(geom.__geo_interface__, indent=2)
        st.sidebar.download_button("Télécharger GeoJSON", data=geojson_data, file_name="annotations.geojson", mime="application/json")
        
        st.success("Annotation ajoutée et exportable en GeoJSON !")
