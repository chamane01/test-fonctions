import streamlit as st
import numpy as np
import laspy
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Import pour CSF et clustering ---
import pycsf
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

def apply_csf_filter(points, cloth_resolution, rigidness, iterations):
    """
    Applique le filtrage CSF pour séparer les points du sol des points d'objets.
    Les paramètres CSF sont définis par la résolution du tissu, la rigidité et le nombre d’itérations.
    """
    csf = pycsf.CSF()
    csf.setPointCloud(points)
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.iterations = iterations
    csf.do_filtering()
    ground_points = csf.get_ground_points()
    non_ground_points = csf.get_non_ground_points()
    return ground_points, non_ground_points

def cluster_objects(points, distance_threshold):
    """
    Regroupe les points (2D, en x et y) en clusters en utilisant une approche de composantes connexes.
    Cette fonction implémente un union-find basé sur une recherche dans un arbre KD.
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
    # Pour chaque point, on fusionne avec tous ses voisins dans le seuil donné
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

st.title("Traitement de fichier LAZ : Filtrage CSF et Modélisation 3D")
st.markdown("""
Cette application permet de :
1. Téléverser un fichier LAZ  
2. Appliquer le filtrage CSF pour séparer les points du sol des points d'objets  
3. Segmenter les objets via un clustering basé sur la connectivité  
4. Extraire les contours 2D et modéliser les objets en 3D (lignes, cubes, etc.)  
5. Afficher les objets avec leur classification (ex. : ligne electrique, batiment, vegetation, route, etc.)
""")

# Téléversement du fichier LAZ
uploaded_file = st.file_uploader("Choisissez votre fichier LAZ", type=["laz", "las"])

# Paramètres CSF dans la sidebar
st.sidebar.header("Paramètres du Filtrage CSF")
cloth_resolution = st.sidebar.slider("Résolution du tissu (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
rigidness = st.sidebar.slider("Rigidité", min_value=1, max_value=10, value=3, step=1)
iterations = st.sidebar.slider("Nombre d'itérations", min_value=50, max_value=1000, value=500, step=50)

# Paramètres du clustering (remplaçant DBSCAN)
st.sidebar.header("Paramètres du Clustering")
clustering_distance = st.sidebar.slider("Distance de clustering (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

if uploaded_file is not None:
    st.write("**Chargement du fichier LAZ...**")
    points = load_laz(uploaded_file)
    if points is not None:
        st.write(f"Nombre de points chargés : {points.shape[0]}")
        
        # Application du filtrage CSF
        st.write("**Application du filtrage CSF...**")
        try:
            ground_points, non_ground_points = apply_csf_filter(points, cloth_resolution, rigidness, iterations)
            st.write(f"Nombre de points de sol : {ground_points.shape[0]}")
            st.write(f"Nombre de points d'objets : {non_ground_points.shape[0]}")
        except Exception as e:
            st.error(f"Erreur lors du filtrage CSF : {e}")
        
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
        
        ax.set_title("Filtrage CSF et Modélisation 3D des Objets")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        st.pyplot(fig)
