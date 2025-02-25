import streamlit as st
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd

# Paramètres SMRF par type d'objet
SMRF_PARAMS = {
    "Bâtiments": {"cell_size": (1, 3), "window_size": (20, 30), "slope_threshold": (5, 15), "elevation_threshold": (2, 5), "iterations": (1, 2)},
    "Basse végétation": {"cell_size": (0.5, 1), "window_size": (5, 10), "slope_threshold": (2, 7), "elevation_threshold": (0.2, 1), "iterations": (1, 1)},
    "Arbustes": {"cell_size": (1, 2), "window_size": (8, 15), "slope_threshold": (5, 10), "elevation_threshold": (1, 3), "iterations": (1, 2)},
    "Arbres": {"cell_size": (2, 5), "window_size": (20, 40), "slope_threshold": (10, 20), "elevation_threshold": (5, 20), "iterations": (2, 3)},
    "Lignes électriques": {"cell_size": (0.5, 1), "window_size": (10, 15), "slope_threshold": (15, 30), "elevation_threshold": (10, 50), "iterations": (1, 2)},
    "Cours d’eau": {"cell_size": (1, 3), "window_size": (10, 20), "slope_threshold": (2, 5), "elevation_threshold": (-2, 1), "iterations": (1, 1)}
}

# Interface Streamlit
st.title("📌 Clustering et Affichage 2D des Points LAS/LAZ")

# Upload du fichier LAS/LAZ
uploaded_file = st.file_uploader("Téléverser un fichier LAS ou LAZ", type=["las", "laz"])

if uploaded_file is not None:
    with st.spinner("Chargement des données..."):
        # Lecture du fichier LAS/LAZ
        las = laspy.read(uploaded_file)
        points = np.vstack((las.x, las.y, las.z)).T
        df = pd.DataFrame(points, columns=["x", "y", "z"])
        
        # Sélection de l'objet à classifier
        object_type = st.selectbox("Sélectionnez le type d'objet à détecter", list(SMRF_PARAMS.keys()))
        params = SMRF_PARAMS[object_type]
        
        # Clustering avec DBSCAN
        clustering = DBSCAN(eps=np.mean(params["cell_size"]), min_samples=5).fit(df)
        df["cluster"] = clustering.labels_
        
        # Affichage des résultats
        unique_clusters = df["cluster"].unique()
        colors = plt.cm.get_cmap("tab10", len(unique_clusters))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cluster in enumerate(unique_clusters):
            if cluster != -1:  # -1 correspond au bruit
                cluster_points = df[df["cluster"] == cluster]
                ax.scatter(cluster_points["x"], cluster_points["y"], s=1, color=colors(i), label=f"Cluster {cluster}")
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Clustering des {object_type} sur une carte fixe 2D")
        ax.legend()
        
        st.pyplot(fig)
