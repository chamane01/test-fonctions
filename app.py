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
            classified_clusters[label] = {"points": pts, "category": category}
        
        # --- Affichage global sur une carte 2D ---
        st.write("**Plan de masse global**")
        fig_global, ax_global = plt.subplots(figsize=(10, 10))
        ax_global.scatter(ground_points[:, 0], ground_points[:, 1],
                          c="saddlebrown", s=0.5, label="Sol")
        
        # Dictionnaire de couleurs pour chaque catégorie
        color_map = {
            "batiment": "red",
            "route": "blue",
            "arbres": "green",
            "herbe": "lightgreen",
            "ligne electrique": "orange",
            "pylone": "purple",
            "inconnu": "gray"
        }
        
        # Pour éviter des doublons dans la légende
        legend_categories = set()
        for label, data in classified_clusters.items():
            pts = data["points"]
            category = data["category"]
            hull = get_convex_hull(pts)
            color = color_map.get(category, "gray")
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
                
                # Afficher les clusters présents dans la zone
                for label, data in classified_clusters.items():
                    pts = data["points"]
                    zone_pts = pts[(pts[:, 0] >= x_min_zone) & (pts[:, 0] < x_max_zone)]
                    if len(zone_pts) == 0:
                        continue
                    category = data["category"]
                    hull = get_convex_hull(zone_pts)
                    color = color_map.get(category, "gray")
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
