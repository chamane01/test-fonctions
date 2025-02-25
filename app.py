import streamlit as st
import numpy as np
import laspy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def apply_ground_filter(points, grid_size, height_threshold):
    """
    Filtre le sol avec une approche simple basée sur une grille.
    Pour chaque cellule, on calcule le minimum de z et on considère comme sol
    les points dont z <= (min + height_threshold).
    """
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    
    num_bins_x = int(np.ceil((x_max - x_min) / grid_size))
    num_bins_y = int(np.ceil((y_max - y_min) / grid_size))
    
    # Définir les bords de la grille
    x_edges = np.linspace(x_min, x_max, num_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, num_bins_y + 1)
    
    # Assigner chaque point à une cellule
    bins_x = np.digitize(points[:,0], x_edges) - 1
    bins_y = np.digitize(points[:,1], y_edges) - 1
    
    # Initialiser la DEM (Digital Elevation Model)
    DEM = np.full((num_bins_x, num_bins_y), np.inf)
    
    # Pour chaque point, mettre à jour le minimum de z de la cellule correspondante
    for i in range(points.shape[0]):
        bx = bins_x[i]
        by = bins_y[i]
        if points[i,2] < DEM[bx, by]:
            DEM[bx, by] = points[i,2]
            
    # Création du masque de points sol
    ground_mask = np.zeros(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        bx = bins_x[i]
        by = bins_y[i]
        if points[i,2] <= DEM[bx, by] + height_threshold:
            ground_mask[i] = True
            
    ground_points = points[ground_mask]
    non_ground_points = points[~ground_mask]
    
    return ground_points, non_ground_points

def cluster_dbscan(points, eps, min_samples):
    """
    Applique DBSCAN sur les points et renvoie les labels de cluster.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return db.labels_

def extract_contour(points):
    """
    Extrait le contour 2D d'un ensemble de points via alphashape
    et renvoie un objet shapely.Polygon.
    """
    if points.shape[0] < 4:
        return None
    pts_2d = points[:, :2]
    try:
        alpha = alphashape.optimizealpha(pts_2d)
        hull = alphashape.alphashape(pts_2d, alpha)
        if isinstance(hull, Polygon):
            return hull
        elif hull.geom_type == 'MultiPolygon':
            # On choisit le polygone de plus grande aire
            hull = max(hull, key=lambda p: p.area)
            return hull
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du contour : {e}")
        return None

def classify_cluster(polygon):
    """
    Classe le polygone selon des critères heuristiques basés sur l'aire et la forme.
    Les classes proposées incluent : 
    "ligne electrique", "batiment", "vegetation", "route", "inconnu".
    """
    if polygon is None:
        return "inconnu"
    
    area = polygon.area
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    ratio = width / height if height > 0 else 0
    
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
2. Appliquer un filtrage du sol (filtre basé sur une grille simple)  
3. Segmenter les objets par DBSCAN  
4. Extraire les contours et les regrouper en polylignes/polygones  
5. Afficher les objets en 2D avec classification (ex. : ligne électrique, bâtiment, végétation, route, etc.)
""")

# Téléversement du fichier LAZ
uploaded_file = st.file_uploader("Choisissez votre fichier LAZ, las", type=["laz","las"])

# Paramètres ajustables dans la sidebar
st.sidebar.header("Paramètres du filtre de sol")
grid_size = st.sidebar.slider("Taille de la grille (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
height_threshold = st.sidebar.slider("Seuil de hauteur (m)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

st.sidebar.header("Paramètres DBSCAN")
dbscan_eps = st.sidebar.slider("Epsilon", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
dbscan_min_samples = st.sidebar.slider("Min Samples", min_value=1, max_value=20, value=5, step=1)

if uploaded_file is not None:
    st.write("**Chargement du fichier LAZ...**")
    points = load_laz(uploaded_file)
    if points is not None:
        st.write(f"Nombre de points chargés : {points.shape[0]}")

        # Application du filtre de sol
        st.write("**Application du filtre de sol...**")
        try:
            ground_points, non_ground_points = apply_ground_filter(points, grid_size, height_threshold)
            st.write(f"Nombre de points de sol : {ground_points.shape[0]}")
            st.write(f"Nombre de points d'objets : {non_ground_points.shape[0]}")
        except Exception as e:
            st.error(f"Erreur lors du filtrage du sol : {e}")
        
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
                continue  # Ignorer le bruit
            cluster_pts = non_ground_points[labels == label]
            polygon = extract_contour(cluster_pts)
            classification = classify_cluster(polygon)
            color = classification_colors.get(classification, "black")
            
            # Affichage des points du cluster
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=1, c=color,
                       label=f"Cluster {label} - {classification}")
            
            # Affichage du contour s'il a été extrait
            if polygon is not None:
                x, y = polygon.exterior.xy
                ax.plot(x, y, color=color, linewidth=2)
        
        ax.set_title("Segmentation et Classification des Objets")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")
        
        st.pyplot(fig)
