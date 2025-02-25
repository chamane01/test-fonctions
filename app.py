import streamlit as st
import numpy as np

import pycsf
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon

# ---------------------------
# Fonctions utilitaires
# ---------------------------

def load_laz(file_obj):
    """
    Charge un fichier LAZ et renvoie un tableau numpy de points (x, y, z)
    """
    try:
        las = laspy.read(file_obj)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        return points
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier LAZ : {e}")
        return None

def apply_csf(points, cloth_resolution, rigidness, iterations):
    """
    Applique le filtre CSF pour séparer le sol des objets.
    Renvoie deux tableaux numpy : ground_points et non_ground_points.
    """
    csf = pycsf.CSF()
    csf.setPointCloud(points)
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.iterations = iterations
    csf.do_filtering()
    ground = csf.get_ground()
    non_ground = csf.get_non_ground()
    return ground, non_ground

def cluster_dbscan(points, eps, min_samples):
    """
    Applique DBSCAN sur les points non-sol et renvoie le tableau des labels.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return db.labels_

def extract_contour(points):
    """
    Extrait le contour 2D d'un ensemble de points (en utilisant alphashape)
    et renvoie un objet shapely.Polygon.
    """
    if points.shape[0] < 4:
        return None
    # On travaille en 2D (x,y)
    pts_2d = points[:, :2]
    try:
        alpha = alphashape.optimizealpha(pts_2d)
        hull = alphashape.alphashape(pts_2d, alpha)
        if isinstance(hull, Polygon):
            return hull
        elif hull.geom_type == 'MultiPolygon':
            # On prend le polygone de plus grande aire
            hull = max(hull, key=lambda p: p.area)
            return hull
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du contour : {e}")
        return None

def classify_cluster(polygon):
    """
    Classe le polygone (extraction du contour) selon quelques critères heuristiques.
    Renvoie une chaîne de caractères parmi : 
    "ligne electrique", "batiment", "vegetation", "route", "inconnu".
    """
    if polygon is None:
        return "inconnu"
    
    area = polygon.area
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    ratio = width / height if height > 0 else 0
    
    # Heuristiques simples (à adapter selon vos données)
    if area < 5:
        return "ligne electrique"
    elif area < 50 and (ratio > 3 or ratio < 0.33):
        return "ligne electrique"
    elif 50 <= area < 200 and 0.8 < ratio < 1.2:
        return "batiment"
    elif area >= 200 and (ratio > 3 or ratio < 0.33):
        return "route"
    elif area >= 200:
        return "vegetation"
    else:
        return "inconnu"

# Dictionnaire de couleurs pour l'affichage selon la classification
classification_colors = {
    "ligne electrique": "red",
    "batiment": "blue",
    "vegetation": "green",
    "route": "orange",
    "inconnu": "gray"
}

# ---------------------------
# Interface Streamlit
# ---------------------------

st.title("Traitement de fichier LAZ : Segmentation & Classification")
st.markdown("""
Cette application permet de :
1. Téléverser un fichier LAZ  
2. Appliquer un filtrage du sol (CSF)  
3. Segmenter les objets par DBSCAN  
4. Extraire les contours et les regrouper en polylignes/polygones  
5. Afficher les objets en 2D avec classification (ex. : ligne électrique, bâtiment, végétation, route, etc.)
""")

# Téléversement du fichier LAZ
uploaded_file = st.file_uploader("Choisissez votre fichier LAZ", type=["laz"])

# Paramètres ajustables dans la sidebar
st.sidebar.header("Paramètres CSF")
csf_cloth_res = st.sidebar.slider("Cloth Resolution", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
csf_rigidness = st.sidebar.slider("Rigidness", min_value=0, max_value=10, value=2, step=1)
csf_iterations = st.sidebar.slider("Nombre d'itérations", min_value=100, max_value=1000, value=500, step=50)

st.sidebar.header("Paramètres DBSCAN")
dbscan_eps = st.sidebar.slider("Epsilon", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
dbscan_min_samples = st.sidebar.slider("Min Samples", min_value=1, max_value=20, value=5, step=1)

if uploaded_file is not None:
    st.write("**Chargement du fichier LAZ...**")
    points = load_laz(uploaded_file)
    if points is not None:
        st.write(f"Nombre de points chargés : {points.shape[0]}")

        # Application du filtrage CSF
        st.write("**Application du filtre CSF pour séparer le sol des objets...**")
        try:
            ground_points, non_ground_points = apply_csf(points, csf_cloth_res, csf_rigidness, csf_iterations)
            st.write(f"Nombre de points de sol : {ground_points.shape[0]}")
            st.write(f"Nombre de points d'objets : {non_ground_points.shape[0]}")
        except Exception as e:
            st.error(f"Erreur lors du filtrage CSF : {e}")
        
        # Segmentation par DBSCAN sur les points non-sol
        st.write("**Segmentation des objets par DBSCAN...**")
        labels = cluster_dbscan(non_ground_points, dbscan_eps, dbscan_min_samples)
        unique_labels = set(labels)
        nb_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        st.write(f"Nombre de clusters détectés (hors bruit) : {nb_clusters}")

        # Création de la figure d'affichage
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(ground_points[:, 0], ground_points[:, 1], s=1, c="lightgray", label="Sol")
        
        # Pour chaque cluster, extraction du contour et classification
        for label in unique_labels:
            if label == -1:
                # Points bruit
                continue
            cluster_pts = non_ground_points[labels == label]
            # Extraction du contour en 2D
            polygon = extract_contour(cluster_pts)
            classification = classify_cluster(polygon)
            color = classification_colors.get(classification, "black")
            
            # Affichage des points du cluster
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=1, c=color,
                       label=f"Cluster {label} - {classification}")
            
            # Si un contour a été extrait, on l'affiche
            if polygon is not None:
                x, y = polygon.exterior.xy
                ax.plot(x, y, color=color, linewidth=2)
        
        ax.set_title("Segmentation et Classification des Objets")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # Pour éviter les doublons dans la légende
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")
        
        st.pyplot(fig)
