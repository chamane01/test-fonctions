import streamlit as st
import numpy as np
import pandas as pd
import pylas
import plotly.graph_objects as go
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN

st.set_page_config(layout="wide")
st.title("Extraction des contours des objets (2D)")

# ---------------------------
# Fonctions utilitaires
# ---------------------------
def apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold):
    """
    Filtrage CSF simplifié : sépare les points de sol et d'objets.
    Ici, nous considérons comme sol les points en dessous de la médiane de Z.
    """
    z_median = np.median(points[:, 2])
    ground_mask = points[:, 2] < z_median
    non_ground_mask = ~ground_mask
    return ground_mask, non_ground_mask

def apply_dbscan(points, eps, min_samples):
    """
    Applique DBSCAN sur les coordonnées X, Y.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :2])
    return labels

def extract_contour(points, simplify_tolerance=0.1):
    """
    Calcule l'enveloppe convexe des points 2D et la simplifie.
    Renvoie une liste de coordonnées (x, y) représentant le contour.
    """
    if len(points) < 3:
        return None
    mp = MultiPoint(points[:, :2])
    hull = mp.convex_hull
    # Simplifier le polygone pour réduire le nombre de sommets (ajuster simplify_tolerance si besoin)
    simplified_hull = hull.simplify(simplify_tolerance, preserve_topology=True)
    if simplified_hull.geom_type == 'Polygon':
        return list(simplified_hull.exterior.coords)
    else:
        return None

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
        # Lecture du fichier LAZ/LAS
        las = pylas.read(uploaded_file)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        st.write(f"Nombre total de points : {points.shape[0]}")

        # ---------------------------
        # Séparation sol / objets via CSF
        # ---------------------------
        st.write("Application du filtre CSF...")
        _, non_ground_mask = apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold)
        object_points = points[non_ground_mask]
        st.write(f"Points d'objets : {object_points.shape[0]}")

        # ---------------------------
        # Détection des clusters avec DBSCAN
        # ---------------------------
        st.write("Application de DBSCAN pour détecter les clusters...")
        labels = apply_dbscan(object_points, eps, min_samples)
        df_objects = pd.DataFrame(object_points, columns=["x", "y", "z"])
        df_objects["cluster"] = labels

        clusters = {}
        for label in np.unique(labels):
            # Ignore le bruit (label = -1) et les clusters trop petits
            if label == -1:
                continue
            pts = df_objects[df_objects["cluster"] == label][["x", "y", "z"]].values
            if pts.shape[0] < 3:
                continue
            clusters[label] = pts

        # Extraction des contours pour chaque cluster
        contours = {}
        for label, pts in clusters.items():
            contour = extract_contour(pts, simplify_tolerance=0.1)
            if contour is not None:
                contours[label] = contour

        # ---------------------------
        # Affichage 2D des contours
        # ---------------------------
        st.write("Affichage 2D : uniquement les contours")
        fig = go.Figure()

        # Utilisation d'une palette de couleurs pour différencier les clusters
        color_map = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                     "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        for i, (label, contour_coords) in enumerate(contours.items()):
            xs, ys = zip(*contour_coords)
            # Fermer le polygone en ajoutant le premier point à la fin
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color_map[i % len(color_map)], width=3),
                    name=f"Cluster {label}"
                )
            )

        fig.update_layout(
            title="Contours des objets détectés",
            xaxis_title="X",
            yaxis_title="Y"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("Veuillez téléverser un fichier LAZ ou LAS pour commencer le traitement.")
