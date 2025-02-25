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
st.title("Segmentation de fichier LAZ avec CSF & DBSCAN")

# ---------------------------
# Fonctions utilitaires
# ---------------------------

def apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold):
    """
    Dummy CSF : sépare le sol des objets en considérant le sol comme les points en dessous de la médiane de Z.
    Dans une application réelle, il faut remplacer cette fonction par l’algorithme de simulation de tissu (CSF).
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
    Extrait le contour (convex hull) d’un ensemble de points 2D.
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
    Fonction de classification dummy qui attribue aléatoirement une classe parmi plusieurs.
    Vous pourrez intégrer ici des règles ou un modèle d’apprentissage pour détecter par exemple :
    ligne électrique, bâtiments, végétations, routes, infrastructures, eau, zones industrielles, agricoles, forestières, etc.
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
# Téléversement du fichier LAZ
# ---------------------------
uploaded_file = st.file_uploader("Téléverser un fichier LAZ et LAS", type=["laz","las"])

if uploaded_file is not None:
    try:
        # Lecture du fichier LAZ
        las = pylas.read(uploaded_file)
        # Extraction des coordonnées x, y, z sous forme de tableau numpy
        points = np.vstack((las.x, las.y, las.z)).transpose()
        st.write(f"Nombre total de points : {points.shape[0]}")
        
        # ---------------------------
        # Étape 1 : CSF pour séparer sol et objets
        # ---------------------------
        st.write("Application du filtre CSF pour séparer le sol des objets...")
        ground_mask, non_ground_mask = apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold)
        ground_points = points[ground_mask]
        object_points = points[non_ground_mask]
        st.write(f"Points de sol : {ground_points.shape[0]}, Points d'objets : {object_points.shape[0]}")
        
        # ---------------------------
        # Étape 2 : DBSCAN pour regrouper les objets
        # ---------------------------
        st.write("Application de DBSCAN pour détecter les clusters d’objets...")
        labels = apply_dbscan(object_points, eps, min_samples)
        # Création d'un DataFrame pour gérer les clusters
        df_objects = pd.DataFrame(object_points, columns=["x", "y", "z"])
        df_objects["cluster"] = labels
        
        # Regroupement par cluster (on ignore le bruit identifié par DBSCAN : label = -1)
        clusters = {}
        for label in np.unique(labels):
            if label == -1:
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
        # Affichage sur une carte 2D avec Plotly
        # ---------------------------
        st.write("Affichage de la segmentation sur la carte 2D")
        fig = go.Figure()
        
        # Afficher les points du sol en gris
        fig.add_trace(go.Scatter(
            x=ground_points[:, 0],
            y=ground_points[:, 1],
            mode='markers',
            marker=dict(color='lightgray', size=1),
            name='Sol'
        ))
        
        # Couleurs pour les clusters
        color_map = px.colors.qualitative.Safe
        
        # Afficher les contours des clusters
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
        
        # Optionnel : Afficher également les points des clusters
        for label, pts in clusters.items():
            fig.add_trace(go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode='markers',
                marker=dict(color=color_map[label % len(color_map)], size=3),
                name=f'Points cluster {label}',
                opacity=0.5
            ))
        
        fig.update_layout(
            title="Segmentation et détection d’objets",
            xaxis_title="X",
            yaxis_title="Y",
            legend=dict(itemsizing='constant')
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
    st.info("Veuillez téléverser un fichier LAZ pour commencer le traitement.")
