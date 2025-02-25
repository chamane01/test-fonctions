import streamlit as st
import numpy as np
import pandas as pd
import laspy
from sklearn.cluster import DBSCAN
import plotly.express as px

st.title("Clusterisation de nuages de points LAS/LAZ avec SMRF")

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléversez votre fichier LAS/LAZ", type=["las", "laz"])

if uploaded_file is not None:
    try:
        # Lecture du fichier LAS/LAZ
        las = laspy.read(uploaded_file)
        # Extraction des coordonnées
        points = np.vstack((las.x, las.y, las.z)).T
        st.success("Fichier chargé avec succès !")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    # Sélection de l'objet à détecter et choix des paramètres SMRF recommandés
    st.sidebar.header("Paramètres SMRF")
    object_type = st.sidebar.selectbox("Sélectionnez l'objet à détecter", 
                                       ["Bâtiments", "Basse végétation", "Arbustes", "Arbres", "Lignes électriques", "Cours d’eau"])

    if object_type == "Bâtiments":
        cell_size = st.sidebar.slider("Cell Size (m)", 1, 3, 2)
        window_size = st.sidebar.slider("Window Size (m)", 20, 30, 25)
        slope_threshold = st.sidebar.slider("Slope Threshold (°)", 5, 15, 10)
        elevation_threshold = st.sidebar.slider("Elevation Threshold (m)", 2, 5, 3)
        iterations = st.sidebar.slider("Iterations", 1, 2, 1)
    elif object_type == "Basse végétation":
        cell_size = st.sidebar.slider("Cell Size (m)", 0.5, 1, 0.75, step=0.1)
        window_size = st.sidebar.slider("Window Size (m)", 5, 10, 7)
        slope_threshold = st.sidebar.slider("Slope Threshold (°)", 2, 7, 4)
        elevation_threshold = st.sidebar.slider("Elevation Threshold (m)", 0.2, 1, 0.6, step=0.1)
        iterations = 1
    elif object_type == "Arbustes":
        cell_size = st.sidebar.slider("Cell Size (m)", 1, 2, 1.5, step=0.1)
        window_size = st.sidebar.slider("Window Size (m)", 8, 15, 11)
        slope_threshold = st.sidebar.slider("Slope Threshold (°)", 5, 10, 7)
        elevation_threshold = st.sidebar.slider("Elevation Threshold (m)", 1, 3, 2)
        iterations = st.sidebar.slider("Iterations", 1, 2, 1)
    elif object_type == "Arbres":
        cell_size = st.sidebar.slider("Cell Size (m)", 2, 5, 3)
        window_size = st.sidebar.slider("Window Size (m)", 20, 40, 30)
        slope_threshold = st.sidebar.slider("Slope Threshold (°)", 10, 20, 15)
        elevation_threshold = st.sidebar.slider("Elevation Threshold (m)", 5, 20, 10)
        iterations = st.sidebar.slider("Iterations", 2, 3, 2)
    elif object_type == "Lignes électriques":
        cell_size = st.sidebar.slider("Cell Size (m)", 0.5, 1, 0.75, step=0.1)
        window_size = st.sidebar.slider("Window Size (m)", 10, 15, 12)
        slope_threshold = st.sidebar.slider("Slope Threshold (°)", 15, 30, 20)
        elevation_threshold = st.sidebar.slider("Elevation Threshold (m)", 10, 50, 20)
        iterations = st.sidebar.slider("Iterations", 1, 2, 1)
    elif object_type == "Cours d’eau":
        cell_size = st.sidebar.slider("Cell Size (m)", 1, 3, 2)
        window_size = st.sidebar.slider("Window Size (m)", 10, 20, 15)
        slope_threshold = st.sidebar.slider("Slope Threshold (°)", 2, 5, 3)
        elevation_threshold = st.sidebar.slider("Elevation Threshold (m)", -2, 1, -1)
        iterations = 1

    st.write(f"### Détection de : {object_type}")
    st.write("Paramètres utilisés :", 
             f"Cell Size = {cell_size} m, Window Size = {window_size} m,",
             f"Slope Threshold = {slope_threshold}°, Elevation Threshold = {elevation_threshold} m,",
             f"Iterations = {iterations}")

    # Filtrage de base : suppression des points proches du sol
    # Ici, on estime le sol à partir du 5ème percentile des altitudes
    z_ground = np.percentile(points[:, 2], 5)
    # Garder uniquement les points dont l'élévation (par rapport au sol) dépasse le seuil choisi
    mask = (points[:, 2] - z_ground) > elevation_threshold
    filtered_points = points[mask]
    
    if filtered_points.shape[0] == 0:
        st.warning("Aucun point n'a été détecté après filtrage par seuil d'élévation.")
    else:
        st.write(f"{filtered_points.shape[0]} points sélectionnés pour la clusterisation.")

        # Pour simplifier, on effectue une clusterisation DBSCAN sur les coordonnées X et Y
        # eps est défini par la taille de maille et min_samples par la taille de la fenêtre
        coords = filtered_points[:, :2]
        db = DBSCAN(eps=cell_size, min_samples=int(window_size)).fit(coords)
        labels = db.labels_

        # Création d'un DataFrame pour la visualisation
        df = pd.DataFrame({
            "X": filtered_points[:, 0],
            "Y": filtered_points[:, 1],
            "Cluster": labels.astype(str)
        })

        st.write("### Résultat de la clusterisation")
        st.write("Chaque cluster est affiché avec une couleur différente.")

        # Affichage sur une carte 2D avec Plotly
        fig = px.scatter(df, x="X", y="Y", color="Cluster",
                         title=f"Clusters détectés pour {object_type}",
                         labels={"X": "Coordonnée X", "Y": "Coordonnée Y"})
        st.plotly_chart(fig, use_container_width=True)
