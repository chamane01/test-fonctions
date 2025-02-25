import streamlit as st
import laspy
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np

st.title("Visualisation 2D du Nuage de Points Classifié")

# Chargement du fichier classifié
uploaded_classified = st.file_uploader("Téléchargez le fichier LAS classifié", type=["las", "laz"])
if uploaded_classified is not None:
    # Sauvegarde temporaire
    with open("classified_output.las", "wb") as f:
        f.write(uploaded_classified.read())

    # Lecture du fichier LAS
    las = laspy.read("classified_output.las")
    
    # Création d'un DataFrame avec les colonnes x, y, z et classification
    # Attention : pour st.map() ou folium, il faut des coordonnées en lat/lon.
    # Ici, on suppose que vos données sont déjà en EPSG:4326.
    df = pd.DataFrame({
        "lon": las.x,
        "lat": las.y,
        "elevation": las.z,
        "classification": las.classification
    })
    
    st.write("Nombre de points :", df.shape[0])
    
    # Définition d'une fonction pour attribuer une couleur selon la classification
    def get_color(cl):
        if cl == 2:
            return "brown"   # Sol
        elif cl == 3:
            return "lightgreen"  # Végétation faible
        elif cl == 4:
            return "green"   # Végétation moyenne
        elif cl == 5:
            return "darkgreen"   # Végétation haute/arbres
        elif cl == 6:
            return "gray"    # Bâtiments
        elif cl == 18:
            return "red"     # Lignes électriques
        else:
            return "blue"    # Autres / non classifiés

    # Création de la carte centrée sur la moyenne des coordonnées
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Ajout de chaque point dans la carte avec un cercle coloré selon la classification
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=1,
            color=get_color(row["classification"]),
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Classe : {row['classification']}, Z : {row['elevation']:.2f}"
        ).add_to(m)
    
    st_folium(m, width=700, height=500)
    
    st.write("Utilisez la souris pour zoomer/dézoomer et explorer la carte.")
