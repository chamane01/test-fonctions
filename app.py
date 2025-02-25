import streamlit as st
import laspy
import pandas as pd
import folium
from streamlit_folium import st_folium

st.title("Visualisation 2D du Nuage de Points Classifié")

# Chargement du fichier classifié
uploaded_classified = st.file_uploader("Téléchargez le fichier LAS classifié", type=["las", "laz"])
if uploaded_classified is not None:
    # Sauvegarde temporaire
    with open("classified_output.las", "wb") as f:
        f.write(uploaded_classified.read())

    # Lecture du fichier LAS
    las = laspy.read("classified_output.las")

    # Fonction pour s'assurer que la donnée est une liste
    def ensure_list(val):
        if hasattr(val, "ndim"):
            return val.tolist() if val.ndim > 0 else [val]
        return [val]

    lon = ensure_list(las.x)
    lat = ensure_list(las.y)
    elevation = ensure_list(las.z)
    classification = ensure_list(las.classification)

    # Création du DataFrame
    df = pd.DataFrame({
        "lon": lon,
        "lat": lat,
        "elevation": elevation,
        "classification": classification
    })

    st.write("Nombre de points :", df.shape[0])

    # Fonction d'attribution des couleurs selon la classification
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

    # Vérification des coordonnées : elles doivent être en EPSG:4326 (lat, lon)
    # Si elles ne le sont pas, pensez à les reprojeter.
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Ajout de chaque point sur la carte
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
    st.write("Utilisez la souris pour zoomer et explorer la carte.")
