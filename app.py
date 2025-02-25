import streamlit as st
import numpy as np
import pandas as pd
import pylas
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
import random

st.set_page_config(layout="wide")
st.title("Extraction des contours des objets (2D) à partir de fichiers LAS/LAZ")

# ---------------------------
# Fonctions utilitaires
# ---------------------------

def apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold):
    """
    Dummy CSF : sépare le sol des objets en considérant le sol comme les points en dessous de la médiane de Z.
    Remplacer par un algorithme CSF complet si nécessaire.
    """
    z_median = np.median(points[:, 2])
    ground_mask = points[:, 2] < z_median
    non_ground_mask = ~ground_mask
    return ground_mask, non_ground_mask

def apply_dbscan(points, eps, min_samples):
    """
    Applique DBSCAN sur les points (en utilisant uniquement x,y pour la segmentation spatiale).
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :2])
    return labels

def extract_contour(points):
    """
    Extrait le contour (enveloppe convexe) d’un ensemble de points 2D.
    Renvoie une liste de coordonnées ou None si non applicable.
    """
    if len(points) < 3:
        return None
    mp = MultiPoint(points[:, :2])
    hull = mp.convex_hull
    if hull.geom_type == 'Polygon':
        return list(hull.exterior.coords)
    else:
        return None

def classify_cluster(points):
    """
    Classification dummy qui attribue aléatoirement une classe parmi plusieurs.
    On peut étendre cette fonction pour inclure :
    ligne electrique, batiments, vegetations, routes, infrastructures, eau, etc.
    """
    classes = [
        "ligne electrique", "batiments", "vegetations", "routes",
        "infrastructure", "eau", "terrain industriel",
        "zones agricoles", "zones forestieres", "autre"
    ]
    return random.choice(classes)

# ---------------------------
# Paramètres dans la sidebar
# ---------------------------
st.sidebar.header("Paramètres CSF")
cloth_resolution = st.sidebar.slider("Cloth Resolution", 0.1, 5.0, 1.0, step=0.1)
rigidness = st.sidebar.slider("Rigidness", 1, 10, 3)
iterations = st.sidebar.slider("Iterations", 1, 100, 50)
class_threshold = st.sidebar.slider("Class Threshold", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.header("Paramètres DBSCAN")
eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 1.0, step=0.1)
min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 50, 5)

# ---------------------------
# Téléversement du fichier LAZ/LAS
# ---------------------------
uploaded_file = st.file_uploader("Téléverser un fichier LAZ ou LAS", type=["laz", "las"])

if uploaded_file is not None:
    try:
        # Lecture du fichier LAS/LAZ
        las = pylas.read(uploaded_file)
        # Extraction des coordonnées x, y, z
        points = np.vstack((las.x, las.y, las.z)).transpose()
        st.write(f"Nombre total de points : {points.shape[0]}")
        
        # ---------------------------
        # Étape 1 : Séparation sol / objets via CSF
        # ---------------------------
        st.write("Application du filtre CSF pour séparer le sol des objets...")
        ground_mask, non_ground_mask = apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold)
        # On ne retient que les points d'objets pour la suite
        object_points = points[non_ground_mask]
        st.write(f"Points d'objets : {object_points.shape[0]}")
        
        # ---------------------------
        # Étape 2 : Détection de clusters avec DBSCAN
        # ---------------------------
        st.write("Application de DBSCAN pour détecter les clusters d’objets...")
        labels = apply_dbscan(object_points, eps, min_samples)
        df_objects = pd.DataFrame(object_points, columns=["x", "y", "z"])
        df_objects["cluster"] = labels
        
        clusters = {}
        for label in np.unique(labels):
            if label == -1:  # Bruit
                continue
            cluster_pts = df_objects[df_objects["cluster"] == label][["x", "y", "z"]].values
            clusters[label] = cluster_pts
        
        # Extraction des contours et classification pour chaque cluster
        contours = {}
        classifications = {}
        for label, pts in clusters.items():
            contour = extract_contour(pts)
            if contour is not None:
                contours[label] = contour
                classifications[label] = classify_cluster(pts)
        
        # ---------------------------
        # Affichage 2D des contours
        # ---------------------------
        st.write("Affichage 2D : uniquement les contours (polylignes/polygones)")
        fig = go.Figure()
        color_map = px.colors.qualitative.Safe
        
        for i, (label, contour_coords) in enumerate(contours.items()):
            xs, ys = zip(*contour_coords)
            # Fermer le polygone en ajoutant le premier point à la fin
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
            classe = classifications[label]
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                line=dict(color=color_map[i % len(color_map)], width=3),
                name=f'Cluster {label} ({classe})'
            ))
        
        fig.update_layout(
            title="Contours des objets détectés",
            xaxis_title="X",
            yaxis_title="Y"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage d'un résumé des classifications
        st.subheader("Classification des objets détectés")
        df_class = pd.DataFrame({
            "Cluster": list(classifications.keys()),
            "Classification": list(classifications.values())
        })
        st.dataframe(df_class)
        
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("Veuillez téléverser un fichier LAZ ou LAS pour commencer le traitement.")
