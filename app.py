import streamlit as st
import laspy
import io
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon

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

st.title("Plan de masse de segmentation LAZ avec CSF & DBSCAN")

# ---------------------------
# Fonctions utilitaires
# ---------------------------
def apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold):
    """
    Dummy CSF : considère comme sol les points dont la coordonnée Z est inférieure
    à la médiane. (Pour une vraie application, intégrer un algorithme CSF.)
    """
    z_median = np.median(points[:, 2])
    ground_mask = points[:, 2] < z_median
    non_ground_mask = ~ground_mask
    return ground_mask, non_ground_mask

def apply_dbscan(points, eps, min_samples):
    """
    Applique DBSCAN sur les coordonnées XY.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :2])
    return labels

def get_convex_hull(points):
    """
    Calcule l'enveloppe convexe en 2D (XY) d'un ensemble de points.
    """
    mp = MultiPoint(points[:, :2])
    if mp.is_empty:
        return None
    hull = mp.convex_hull
    if isinstance(hull, Polygon):
        return hull
    return None

def points_to_polyline(points):
    """
    Transforme un ensemble de points en une polyligne ordonnée (algorithme glouton).
    """
    if len(points) < 2:
        return points
    points_copy = points.copy()
    start_index = np.argmin(points_copy[:, 0])
    polyline = [points_copy[start_index]]
    points_copy = np.delete(points_copy, start_index, axis=0)
    while len(points_copy) > 0:
        last_point = polyline[-1]
        distances = np.linalg.norm(points_copy - last_point, axis=1)
        next_index = np.argmin(distances)
        polyline.append(points_copy[next_index])
        points_copy = np.delete(points_copy, next_index, axis=0)
    return np.array(polyline)

def classify_cluster(points):
    """
    Classification heuristique des clusters.
    
    - **Pylône** : extension verticale (z_range) > 10 m et dispersion en XY < 5 m.
    - **Ligne électrique** : cluster très allongé (min(x_range,y_range) < 2 et max(x_range,y_range) > 10)
      avec un nombre modéré de points (n < 300).  
    - **Batiment**, **Route** ou **Végétation** selon d'autres critères.
    """
    n = len(points)
    if n < 5:
        return "inconnu"
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Pylône : très haut (>10 m) et compact en XY
    if z_range > 10 and x_range < 5 and y_range < 5:
        return "pylone"
    # Ligne électrique : forme très allongée
    if min(x_range, y_range) < 2 and max(x_range, y_range) > 10 and n < 300:
        return "ligne electrique"
    # Batiment : structure relativement haute et compacte
    if z_range > 3 and n > 100 and x_range < 30 and y_range < 30:
        return "batiment"
    # Route : presque plat et avec une grande étendue horizontale
    if z_range < 2 and (x_range > 20 or y_range > 20):
        return "route"
    # Végétation : cluster volumineux avec une certaine hauteur
    if n > 200 and z_range > 2:
        if np.mean(points[:, 2]) < (z_min + 2):
            return "herbe"
        else:
            return "arbres"
    return "inconnu"

# ---------------------------
# Application principale
# ---------------------------
uploaded_file = st.file_uploader("Téléverser un fichier LAZ ou LAS", type=["laz", "las"])
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)
        las = laspy.read(file_buffer)
        # Extraction des coordonnées X, Y, Z
        points = np.vstack((las.x, las.y, las.z)).T
        st.write(f"Nombre total de points : {points.shape[0]}")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()
    
    # --- Segmentation CSF ---
    st.write("Application du CSF (dummy) pour séparer sol et objets...")
    ground_mask, non_ground_mask = apply_csf(points, cloth_resolution, rigidness, iterations, class_threshold)
    ground_points = points[ground_mask]
    object_points = points[non_ground_mask]
    st.write(f"Points de sol : {ground_points.shape[0]}, Points d'objets : {object_points.shape[0]}")
    
    # --- Clustering DBSCAN ---
    st.write("Application de DBSCAN sur les points d'objets...")
    labels = apply_dbscan(object_points, eps, min_samples)
    unique_labels = set(labels)
    nb_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    st.write(f"Nombre de clusters détectés (hors bruit) : {nb_clusters}")
    
    # --- Regroupement et classification des clusters ---
    clusters = {}
    classified_clusters = {}
    for label in unique_labels:
        if label == -1:
            continue  # Ignorer le bruit
        cluster_points = object_points[labels == label]
        clusters[label] = cluster_points
        classification = classify_cluster(cluster_points)
        classified_clusters[label] = {"points": cluster_points, "class": classification}
    
    # --- Affichage global sur un plan de masse en 2D ---
    st.write("Affichage du plan de masse global")
    fig_global, ax_global = plt.subplots(figsize=(10, 10))
    ax_global.scatter(ground_points[:, 0], ground_points[:, 1],
                      c="lightgray", s=0.5, label="Sol")
    
    # Dictionnaire de couleurs pour les classes
    color_map = {
        "batiment": "red",
        "route": "blue",
        "arbres": "green",
        "herbe": "lightgreen",
        "ligne electrique": "orange",
        "pylone": "purple",
        "inconnu": "gray"
    }
    
    legend_categories = set()
    for label, data in classified_clusters.items():
        pts = data["points"]
        cls = data["class"]
        color = color_map.get(cls, "gray")
        if cls == "ligne electrique":
            # Transformation des points en polyligne et accentuation (linewidth=4)
            polyline = points_to_polyline(pts)
            ax_global.plot(polyline[:, 0], polyline[:, 1],
                           color=color, linewidth=4,
                           label=cls if cls not in legend_categories else "")
        else:
            hull = get_convex_hull(pts)
            if hull is not None:
                xh, yh = hull.exterior.xy
                ax_global.fill(xh, yh, alpha=0.5, fc=color, ec="black",
                               label=cls if cls not in legend_categories else "")
            else:
                ax_global.scatter(pts[:, 0], pts[:, 1],
                                  c=color, s=5,
                                  label=cls if cls not in legend_categories else "")
        legend_categories.add(cls)
    
    ax_global.set_xlabel("X")
    ax_global.set_ylabel("Y")
    ax_global.set_title("Plan de masse des objets détectés")
    ax_global.legend(loc="upper right", fontsize="small")
    st.pyplot(fig_global)
    
    # --- Affichage par zones (découpage en 10 parties le long de X) ---
    st.write("Affichage par zones pour zoom détaillé")
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    zones = np.linspace(min_x, max_x, 11)  # 11 bornes pour 10 zones
    zone_tabs = st.tabs([f"Zone {i+1}" for i in range(10)])
    for i, tab in enumerate(zone_tabs):
        with tab:
            x_min_zone = zones[i]
            x_max_zone = zones[i+1]
            zone_ground = ground_points[(ground_points[:, 0] >= x_min_zone) & (ground_points[:, 0] < x_max_zone)]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(zone_ground[:, 0], zone_ground[:, 1],
                       c="lightgray", s=0.5, label="Sol")
            for label, data in classified_clusters.items():
                pts = data["points"]
                zone_pts = pts[(pts[:, 0] >= x_min_zone) & (pts[:, 0] < x_max_zone)]
                if len(zone_pts) == 0:
                    continue
                cls = data["class"]
                color = color_map.get(cls, "gray")
                if cls == "ligne electrique":
                    polyline = points_to_polyline(zone_pts)
                    ax.plot(polyline[:, 0], polyline[:, 1],
                            color=color, linewidth=4, label=cls)
                else:
                    hull = get_convex_hull(zone_pts)
                    if hull is not None:
                        xh, yh = hull.exterior.xy
                        ax.fill(xh, yh, alpha=0.5, fc=color, ec="black", label=cls)
                    else:
                        ax.scatter(zone_pts[:, 0], zone_pts[:, 1],
                                   c=color, s=5, label=cls)
            ax.set_xlim(x_min_zone, x_max_zone)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Zone {i+1} (X: {x_min_zone:.2f} - {x_max_zone:.2f})")
            ax.legend(loc="upper right", fontsize="small")
            st.pyplot(fig)
    
    st.success("Traitement terminé.")

else:
    st.info("Veuillez téléverser un fichier LAZ ou LAS pour commencer le traitement.")
