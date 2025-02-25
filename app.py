import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# --- Fonction d'extraction d'une droite par PCA ---
def fit_line_pca(points):
    """
    Ajuste une droite aux points via PCA et retourne les extrémités du segment obtenu.
    """
    pca = PCA(n_components=2)
    pca.fit(points)
    pc1 = pca.components_[0]  # direction principale
    # Projection de chaque point sur le premier axe principal
    proj = np.dot(points - pca.mean_, pc1)
    t_min, t_max = proj.min(), proj.max()
    point_min = pca.mean_ + t_min * pc1
    point_max = pca.mean_ + t_max * pc1
    return np.array([point_min, point_max])

# --- Extraction et filtrage des segments ---
def extract_lines(points_xy, eps=1.0, min_samples=5, cluster_min_points=10,
                  min_length=10.0, max_angle=20.0):
    """
    Applique DBSCAN pour regrouper les points (en 2D) et pour chaque cluster suffisamment
    grand, ajuste une droite via PCA. Un segment est validé uniquement s'il a une longueur
    supérieure à 'min_length' et un angle (par rapport à l'horizontale) inférieur à 'max_angle' degrés.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_xy)
    unique_labels = set(labels)
    valid_lines = []
    for label in unique_labels:
        if label == -1:
            continue  # ignorer le bruit
        cluster_points = points_xy[labels == label]
        if len(cluster_points) < cluster_min_points:
            continue
        # Extraction du segment par PCA
        line_segment = fit_line_pca(cluster_points)
        # Filtre sur la longueur du segment
        seg_length = np.linalg.norm(line_segment[1] - line_segment[0])
        if seg_length < min_length:
            continue
        # Calcul de l'angle par rapport à l'horizontale
        dx = line_segment[1][0] - line_segment[0][0]
        dy = line_segment[1][1] - line_segment[0][1]
        if dx == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan(abs(dy/dx)))
        if angle > max_angle:
            continue
        valid_lines.append(line_segment)
    return valid_lines

# --- Application Streamlit ---
st.title("Extraction de lignes électriques depuis un nuage LAS/LAZ")

st.markdown("""
Ce prototype téléverse un fichier LAS/LAZ, extrait les points situés au-dessus d'un seuil (pour isoler les structures surélevées)  
puis applique un clustering (DBSCAN) suivi de filtres géométriques (longueur minimale et angle par rapport à l'horizontale)  
afin de ne retenir que des segments pouvant correspondre à des lignes électriques.  
Les segments validés sont affichés sous forme de polylignes (sans points) et la zone globale est découpée en 10 vues zoomées.
""")

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléversez un fichier LAS ou LAZ", type=["las", "laz"])

if uploaded_file is not None:
    with st.spinner("Lecture du fichier et traitement du nuage de points..."):
        try:
            las = laspy.read(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
        else:
            # Extraction des coordonnées
            x = las.x
            y = las.y
            z = las.z
            points = np.vstack((x, y, z)).T
            st.write(f"Nombre total de points : **{points.shape[0]}**")
            
            # Estimation du niveau de sol (10ème percentile)
            ground_level = np.percentile(z, 10)
            st.write(f"Niveau du sol estimé (10ème percentile) : **{ground_level:.2f}**")
            
            # Filtrage par hauteur (points avec z > seuil)
            seuil_defaut = ground_level + 5.0
            seuil = st.number_input("Seuil de hauteur (points avec z > seuil)", 
                                    value=seuil_defaut, step=0.5)
            mask = z > seuil
            candidate_points = points[mask]
            st.write(f"Nombre de points candidats (z > {seuil}) : **{candidate_points.shape[0]}**")
            
            if candidate_points.shape[0] == 0:
                st.error("Aucun point au-dessus du seuil sélectionné.")
            else:
                candidate_points_xy = candidate_points[:, :2]
                
                st.markdown("### Paramètres du clustering et filtrage géométrique")
                eps = st.slider("DBSCAN - eps", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                min_samples = st.slider("DBSCAN - min_samples", min_value=1, max_value=20, value=5)
                cluster_min_points = st.slider("Nombre minimal de points par cluster", 
                                               min_value=2, max_value=100, value=10)
                min_length = st.number_input("Longueur minimale du segment", value=10.0, step=1.0)
                max_angle = st.slider("Angle maximum par rapport à l'horizontale (°)", 
                                      min_value=0.0, max_value=90.0, value=20.0, step=1.0)
                
                # Extraction des segments validés
                lines = extract_lines(candidate_points_xy, eps=eps, min_samples=min_samples, 
                                        cluster_min_points=cluster_min_points, 
                                        min_length=min_length, max_angle=max_angle)
                st.write(f"Nombre de segments linéaires validés : **{len(lines)}**")
                
                if len(lines) == 0:
                    st.error("Aucun segment validé après application des filtres géométriques et de distance.")
                else:
                    # --- Affichage global en 2D : polylignes seulement ---
                    fig_main, ax_main = plt.subplots(figsize=(8, 6))
                    for seg in lines:
                        ax_main.plot(seg[:, 0], seg[:, 1], color='blue', linewidth=2)
                    ax_main.set_xlabel("X")
                    ax_main.set_ylabel("Y")
                    ax_main.set_title("Lignes électriques extraites (vue globale)")
                    # Légende commune pour toutes les lignes
                    ax_main.plot([], [], color='blue', linewidth=2, label="Lignes électriques")
                    ax_main.legend(loc='best')
                    st.pyplot(fig_main)
                    
                    # --- Découpage de la zone en 10 parties pour zoom ---
                    x_min, x_max = candidate_points_xy[:, 0].min(), candidate_points_xy[:, 0].max()
                    y_min, y_max = candidate_points_xy[:, 1].min(), candidate_points_xy[:, 1].max()
                    delta = (x_max - x_min) / 10
                    
                    fig_zoom, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
                    axes = axes.flatten()
                    for i, ax in enumerate(axes):
                        sub_xmin = x_min + i * delta
                        sub_xmax = sub_xmin + delta
                        ax.set_xlim(sub_xmin, sub_xmax)
                        ax.set_ylim(y_min, y_max)
                        ax.set_title(f"Zoom {i+1}: X [{sub_xmin:.1f}, {sub_xmax:.1f}]")
                        for seg in lines:
                            ax.plot(seg[:, 0], seg[:, 1], color='blue', linewidth=2)
                    fig_zoom.tight_layout()
                    st.pyplot(fig_zoom)
                    
    st.success("Traitement terminé avec segmentation géométrique et affichages zoomés.")
