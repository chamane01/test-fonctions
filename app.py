import streamlit as st
import laspy
import io
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon

# --- Fonction de segmentation (simulation CSF) ---
def csf_segmentation(points, ground_percentile=10, height_threshold=1.0):
    """
    Sépare les points de sol des points non-sol.
    Ici, on considère comme sol les points dont la coordonnée Z est inférieure
    au percentile 'ground_percentile' augmenté d'un seuil.
    (Ceci est une approximation et non la véritable méthode CSF.)
    """
    z_vals = points[:, 2]
    ground_limit = np.percentile(z_vals, ground_percentile) + height_threshold
    ground_mask = z_vals <= ground_limit
    return points[ground_mask], points[~ground_mask]

# --- Fonction de classification des clusters ---
def classify_cluster(points):
    """
    Applique des règles heuristiques pour classifier un cluster.
    Ces règles sont purement indicatives :
      - 'batiment' : cluster avec une hauteur (z_range) relativement élevée et une étendue horizontale modérée.
      - 'route' : cluster plat avec une grande étendue.
      - 'arbres' ou 'herbe' : végétation (on peut distinguer grossièrement selon la hauteur moyenne).
      - 'ligne electrique' : cluster étroit et allongé.
      - 'pylone' : petit cluster vertical.
    """
    n = len(points)
    if n < 30:
        return "inconnu"
    
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    
    # Règle pour pylône : très petit en XY mais étendu en Z
    if z_range > 5 and x_range < 5 and y_range < 5:
        return "pylone"
    # Règle pour bâtiment : hauteur importante et cluster compact en XY
    if z_range > 3 and n > 100 and x_range < 30 and y_range < 30:
        return "batiment"
    # Règle pour route : faible variation en altitude et grande étendue en XY
    if z_range < 2 and (x_range > 20 or y_range > 20):
        return "route"
    # Règle pour ligne électrique : cluster étroit et allongé
    if n < 100 and max(x_range, y_range) > 30 and min(x_range, y_range) < 5:
        return "ligne electrique"
    # Règle pour végétation : cluster volumineux avec une certaine irrégularité
    if n > 200 and z_range > 2:
        # On peut distinguer herbe et arbres par la hauteur moyenne (heuristique)
        if np.mean(points[:, 2]) < (np.min(points[:, 2]) + 2):
            return "herbe"
        else:
            return "arbres"
    return "inconnu"

# --- Fonction pour générer une enveloppe convexe ---
def get_convex_hull(points):
    """
    Calcule l'enveloppe convexe (convex hull) en 2D (XY) d'un ensemble de points.
    """
    multipoint = MultiPoint(points[:, :2])
    if multipoint.is_empty:
        return None
    hull = multipoint.convex_hull
    if isinstance(hull, Polygon):
        return hull
    return None

# --- Nouvelle fonction : RANSAC pour la détection de ligne ---
def ransac_line(points, threshold=0.5, min_inliers_ratio=0.7, iterations=100):
    """
    Détecte une ligne dans un ensemble de points 2D via RANSAC.
    Renvoie les paramètres de la ligne (a, b, c) dans l'équation ax + by + c = 0
    ainsi qu'un masque booléen des inliers, ou (None, None) si aucune ligne satisfaisante n'est trouvée.
    """
    best_line = None
    best_inlier_count = 0
    best_inliers = None
    n_points = points.shape[0]
    if n_points < 2:
        return None, None
    for _ in range(iterations):
        # Échantillonnage aléatoire de deux points
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        # Calcul des paramètres de la ligne
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        norm = np.sqrt(a*a + b*b)
        if norm == 0:
            continue
        a /= norm
        b /= norm
        c = -(a*p1[0] + b*p1[1])
        # Calcul de la distance de tous les points à la ligne
        distances = np.abs(a*points[:, 0] + b*points[:, 1] + c)
        inliers = distances < threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line = (a, b, c)
            best_inliers = inliers
    if best_line is not None and best_inlier_count / n_points >= min_inliers_ratio:
        return best_line, best_inliers
    else:
        return None, None

# --- Application principale ---
def main():
    st.title("Traitement de fichiers LAZ/LAS")
    st.write("Cette application permet d'uploader un fichier LAZ ou LAS, "
             "d'appliquer une segmentation (simulation CSF) puis un clustering (DBSCAN) afin de détecter :")
    st.markdown("""
    - Bâtiments  
    - Routes  
    - Végétation (herbe, arbres)  
    - Lignes électriques (HT, MT)  
    - Pylônes  
    """)
    
    uploaded_file = st.file_uploader("Choisissez un fichier LAZ ou LAS", type=["laz", "las"])
    if uploaded_file is not None:
        try:
            # Lecture du fichier dans un buffer
            file_bytes = uploaded_file.read()
            file_buffer = io.BytesIO(file_bytes)
            las = laspy.read(file_buffer)
            # Extraction des coordonnées X, Y, Z
            points = np.vstack((las.x, las.y, las.z)).T
            st.success("Fichier chargé avec succès!")
        except Exception as e:
            st.error("Erreur lors de la lecture du fichier : " + str(e))
            return
        
        # --- Segmentation CSF simulée ---
        st.write("**Segmentation CSF (simulation)**")
        with st.spinner("Séparation des points de sol et non-sol..."):
            ground_points, non_ground_points = csf_segmentation(points)
        st.write(f"Nombre de points de sol : {len(ground_points)}")
        st.write(f"Nombre de points non-sol : {len(non_ground_points)}")
        
        # --- Clustering DBSCAN ---
        st.write("**Clustering DBSCAN sur les points non-sol**")
        with st.spinner("Clustering en cours..."):
            # On applique DBSCAN sur les coordonnées XY
            dbscan = DBSCAN(eps=1.0, min_samples=10)
            labels = dbscan.fit_predict(non_ground_points[:, :2])
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        st.write(f"Nombre de clusters détectés (hors bruit) : {n_clusters}")
        
        # --- Regroupement et classification des clusters ---
        clusters = {}
        for label in unique_labels:
            if label == -1:
                continue  # on ignore le bruit
            clusters[label] = non_ground_points[labels == label]
            
        st.write("**Classification des clusters**")
        classified_clusters = {}
        for label, pts in clusters.items():
            category = classify_cluster(pts)
            extra_data = {}
            # Pour les lignes électriques, on applique RANSAC afin de valider la linéarité
            if category == "ligne electrique":
                line_params, inliers = ransac_line(pts[:, :2])
                if line_params is not None:
                    category = "ligne electrique (RANSAC validé)"
                    extra_data["line_params"] = line_params
                    extra_data["inliers"] = inliers
            classified_clusters[label] = {"points": pts, "category": category}
            classified_clusters[label].update(extra_data)
        
        # Dictionnaire de couleurs pour chaque catégorie
        color_map = {
            "batiment": "red",
            "route": "blue",
            "arbres": "green",
            "herbe": "lightgreen",
            "ligne electrique": "orange",
            "ligne electrique (RANSAC validé)": "orange",
            "pylone": "purple",
            "inconnu": "gray"
        }
        
        # --- Affichage global sur une carte 2D ---
        st.write("**Plan de masse global**")
        fig_global, ax_global = plt.subplots(figsize=(10, 10))
        ax_global.scatter(ground_points[:, 0], ground_points[:, 1],
                          c="saddlebrown", s=0.5, label="Sol")
        
        legend_categories = set()
        for label, data in classified_clusters.items():
            pts = data["points"]
            category = data["category"]
            color = color_map.get(category, "gray")
            
            # Pour les clusters de lignes électriques validés, afficher la ligne ajustée
            if category.startswith("ligne electrique") and data.get("line_params") is not None:
                a, b, c = data["line_params"]
                # Utiliser les inliers pour définir la portion de ligne à afficher
                if "inliers" in data and data["inliers"] is not None:
                    inlier_pts = pts[data["inliers"]]
                else:
                    inlier_pts = pts
                if inlier_pts.shape[0] > 0:
                    # Calculer un vecteur de direction perpendiculaire au vecteur normal (a, b)
                    direction = np.array([-b, a])
                    # Projeter les points sur ce vecteur pour obtenir une mesure d'étendue
                    projections = inlier_pts[:, :2].dot(direction)
                    pt_min = inlier_pts[:, :2][np.argmin(projections)]
                    pt_max = inlier_pts[:, :2][np.argmax(projections)]
                    ax_global.plot([pt_min[0], pt_max[0]], [pt_min[1], pt_max[1]], 
                                   color=color, linewidth=2)
                else:
                    ax_global.scatter(pts[:, 0], pts[:, 1], c=color, s=5)
            else:
                # Pour les autres catégories, afficher l'enveloppe convexe si possible
                hull = get_convex_hull(pts)
                if hull is not None:
                    x, y = hull.exterior.xy
                    ax_global.fill(x, y, alpha=0.5, fc=color, ec="black")
                else:
                    ax_global.scatter(pts[:, 0], pts[:, 1], c=color, s=5)
            
            if category not in legend_categories:
                ax_global.scatter([], [], c=color, label=category)
                legend_categories.add(category)
        
        ax_global.set_xlabel("X")
        ax_global.set_ylabel("Y")
        ax_global.set_title("Plan de masse des objets détectés")
        ax_global.legend(loc="upper right", fontsize="small")
        st.pyplot(fig_global)
        
        # --- Affichage par zones pour un meilleur zoom ---
        st.write("**Affichage par zones**")
        # Déterminer l'étendue en X et découper en 10 zones
        min_x = points[:, 0].min()
        max_x = points[:, 0].max()
        zones = np.linspace(min_x, max_x, num=11)  # 11 bornes => 10 zones
        
        # Création de 10 onglets pour afficher chaque zone
        zone_tabs = st.tabs([f"Zone {i+1}" for i in range(10)])
        for i, tab in enumerate(zone_tabs):
            with tab:
                x_min_zone = zones[i]
                x_max_zone = zones[i+1]
                # Filtrer les points de sol pour la zone
                zone_ground_mask = (ground_points[:, 0] >= x_min_zone) & (ground_points[:, 0] < x_max_zone)
                zone_ground = ground_points[zone_ground_mask]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(zone_ground[:, 0], zone_ground[:, 1],
                           c="saddlebrown", s=0.5, label="Sol")
                
                for label, data in classified_clusters.items():
                    pts = data["points"]
                    zone_pts = pts[(pts[:, 0] >= x_min_zone) & (pts[:, 0] < x_max_zone)]
                    if len(zone_pts) == 0:
                        continue
                    category = data["category"]
                    color = color_map.get(category, "gray")
                    # Pour les lignes électriques, tenter d'afficher la ligne ajustée
                    if category.startswith("ligne electrique") and data.get("line_params") is not None:
                        a, b, c = data["line_params"]
                        if "inliers" in data and data["inliers"] is not None:
                            inlier_pts = pts[data["inliers"]]
                        else:
                            inlier_pts = pts
                        if inlier_pts.shape[0] > 0:
                            direction = np.array([-b, a])
                            projections = inlier_pts[:, :2].dot(direction)
                            pt_min = inlier_pts[:, :2][np.argmin(projections)]
                            pt_max = inlier_pts[:, :2][np.argmax(projections)]
                            ax.plot([pt_min[0], pt_max[0]], [pt_min[1], pt_max[1]], 
                                    color=color, linewidth=2, label=category)
                        else:
                            ax.scatter(zone_pts[:, 0], zone_pts[:, 1], c=color, s=5)
                    else:
                        hull = get_convex_hull(zone_pts)
                        if hull is not None:
                            xh, yh = hull.exterior.xy
                            ax.fill(xh, yh, alpha=0.5, fc=color, ec="black")
                        else:
                            ax.scatter(zone_pts[:, 0], zone_pts[:, 1], c=color, s=5)
                
                ax.set_xlim(x_min_zone, x_max_zone)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"Zone {i+1} (X: {x_min_zone:.2f} - {x_max_zone:.2f})")
                ax.legend(loc="upper right", fontsize="small")
                st.pyplot(fig)
        
        st.success("Traitement terminé.")

if __name__ == "__main__":
    main()
