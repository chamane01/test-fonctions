import streamlit as st
import numpy as np
import laspy
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import pour la recherche de voisins (clustering) et visualisation 3D
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    bins_x = np.clip(np.digitize(points[:,0], x_edges) - 1, 0, num_bins_x - 1)
    bins_y = np.clip(np.digitize(points[:,1], y_edges) - 1, 0, num_bins_y - 1)
    
    # Initialiser le Digital Elevation Model (DEM)
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

def cluster_objects(points, distance_threshold):
    """
    Regroupe les points (en 2D : x et y) en clusters en utilisant une approche
    de composantes connexes et un arbre KD pour accélérer la recherche de voisins.
    """
    pts_2d = points[:, :2]
    tree = cKDTree(pts_2d)
    n = pts_2d.shape[0]
    # Initialisation pour union-find
    uf = list(range(n))
    
    def find(i):
        while uf[i] != i:
            uf[i] = uf[uf[i]]
            i = uf[i]
        return i
    
    def union(i, j):
        ri = find(i)
        rj = find(j)
        uf[ri] = rj
    
    # Pour chaque point, fusionner avec ses voisins dans le seuil donné
    for i in range(n):
        voisins = tree.query_ball_point(pts_2d[i], distance_threshold)
        for j in voisins:
            union(i, j)
    
    # Attribution des labels de cluster
    rep_to_label = {}
    labels = np.empty(n, dtype=int)
    label_counter = 0
    for i in range(n):
        rep = find(i)
        if rep not in rep_to_label:
            rep_to_label[rep] = label_counter
            label_counter += 1
        labels[i] = rep_to_label[rep]
    
    return labels

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
            # Choisir le polygone de plus grande aire
            hull = max(hull, key=lambda p: p.area)
            return hull
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du contour : {e}")
        return None

def classify_cluster(polygon):
    """
    Classe le polygone selon des critères heuristiques (aire, forme).
    Les classes proposées : "ligne electrique", "batiment", "vegetation", "route", "inconnu".
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

def compute_3d_bounding_box(points):
    """
    Calcule la boîte englobante 3D d'un ensemble de points.
    Renvoie les valeurs minimales et maximales (min_vals, max_vals).
    """
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    return min_vals, max_vals

def get_box_faces(min_vals, max_vals):
    """
    Construit les 6 faces (sous forme de listes de sommets) de la boîte englobante.
    """
    minx, miny, minz = min_vals
    maxx, maxy, maxz = max_vals
    # Sommets du cube
    vertices = np.array([
        [minx, miny, minz],
        [maxx, miny, minz],
        [maxx, maxy, minz],
        [minx, maxy, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, maxy, maxz],
        [minx, maxy, maxz]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # base
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # face avant
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # face arrière
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # face droite
        [vertices[4], vertices[7], vertices[3], vertices[0]]   # face gauche
    ]
    return faces

# ---------------------------
# Interface Streamlit
# ---------------------------

st.title("Traitement de fichier LAZ : Filtrage par Grille et Modélisation 3D")
st.markdown("""
Cette application permet de :
1. Téléverser un fichier LAZ  
2. Appliquer un filtrage du sol (basé sur une grille simple)  
3. Segmenter les objets via un clustering par proximité  
4. Extraire les contours 2D et modéliser les objets en 3D (lignes pour les lignes électriques, cubes pour les autres objets)  
5. Afficher les objets avec leur classification (ex. : ligne électrique, bâtiment, végétation, route, etc.)
""")

# Téléversement du fichier LAZ
uploaded_file = st.file_uploader("Choisissez votre fichier LAZ", type=["laz", "las"])

# Paramètres du filtrage du sol (par grille) dans la sidebar
st.sidebar.header("Paramètres du Filtrage du Sol")
grid_size = st.sidebar.slider("Taille de la grille (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
height_threshold = st.sidebar.slider("Seuil de hauteur (m)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

# Paramètres du clustering
st.sidebar.header("Paramètres du Clustering")
clustering_distance = st.sidebar.slider("Distance de clustering (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

if uploaded_file is not None:
    st.write("**Chargement du fichier LAZ...**")
    points = load_laz(uploaded_file)
    if points is not None:
        st.write(f"Nombre de points chargés : {points.shape[0]}")
        
        # Application du filtrage du sol par grille
        st.write("**Application du filtrage du sol...**")
        try:
            ground_points, non_ground_points = apply_ground_filter(points, grid_size, height_threshold)
            st.write(f"Nombre de points de sol : {ground_points.shape[0]}")
            st.write(f"Nombre de points d'objets : {non_ground_points.shape[0]}")
        except Exception as e:
            st.error(f"Erreur lors du filtrage du sol : {e}")
        
        # Clustering des points d'objets (via connectivité en 2D)
        st.write("**Clustering des objets...**")
        try:
            labels = cluster_objects(non_ground_points, clustering_distance)
            unique_labels = np.unique(labels)
            st.write(f"Nombre d'objets détectés : {len(unique_labels)}")
        except Exception as e:
            st.error(f"Erreur lors du clustering : {e}")
        
        # Création de la figure d'affichage en 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Affichage des points du sol en gris clair
        ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2],
                   s=1, c="lightgray", label="Sol")
        
        # Pour chaque objet (cluster) : extraction du contour, classification et modélisation 3D
        for label in unique_labels:
            cluster_pts = non_ground_points[labels == label]
            # Extraction du contour 2D pour classification
            polygon = extract_contour(cluster_pts)
            classification = classify_cluster(polygon)
            color = classification_colors.get(classification, "black")
            
            if classification == "ligne electrique":
                # Pour une ligne électrique, tracer une ligne verticale passant par le centre du cluster
                mean_xy = np.mean(cluster_pts[:, :2], axis=0)
                min_z = np.min(cluster_pts[:, 2])
                max_z = np.max(cluster_pts[:, 2])
                ax.plot([mean_xy[0], mean_xy[0]], [mean_xy[1], mean_xy[1]],
                        [min_z, max_z], color=color, linewidth=2,
                        label=f"Objet {label} - {classification}")
            else:
                # Pour les autres objets, tracer la boîte englobante (cube)
                min_vals, max_vals = compute_3d_bounding_box(cluster_pts)
                faces = get_box_faces(min_vals, max_vals)
                box = Poly3DCollection(faces, facecolors=color, alpha=0.3, edgecolor=color)
                ax.add_collection3d(box)
                # Optionnel : tracer également le contour 2D sur le plan sol
                if polygon is not None:
                    x, y = polygon.exterior.xy
                    z = np.full_like(x, min_vals[2])
                    ax.plot(x, y, z, color=color, linewidth=2,
                            label=f"Objet {label} - {classification}")
        
        ax.set_title("Filtrage par Grille et Modélisation 3D des Objets")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        st.pyplot(fig)
