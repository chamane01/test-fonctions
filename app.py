import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# --- Fonctions utilitaires ---

def fit_line_pca(points):
    """
    Calcule une droite (segment) par analyse en composantes principales (PCA).
    Les extrémités du segment sont obtenues en projetant les points sur le premier axe principal.
    """
    pca = PCA(n_components=2)
    pca.fit(points)
    pc1 = pca.components_[0]  # direction principale
    # Projeter chaque point sur la direction principale
    proj = np.dot(points - pca.mean_, pc1)
    t_min, t_max = proj.min(), proj.max()
    point_min = pca.mean_ + t_min * pc1
    point_max = pca.mean_ + t_max * pc1
    return np.array([point_min, point_max])

def extract_lines(points_xy, eps=1.0, min_samples=5, cluster_min_points=10):
    """
    Applique DBSCAN pour regrouper les points (en 2D) et, pour chaque cluster suffisamment grand,
    extrait une droite via une analyse en composantes principales.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_xy)
    unique_labels = set(labels)
    lines = []
    for label in unique_labels:
        if label == -1:
            continue  # bruit
        cluster_points = points_xy[labels == label]
        if len(cluster_points) < cluster_min_points:
            continue
        line_segment = fit_line_pca(cluster_points)
        lines.append(line_segment)
    return lines

# --- Application Streamlit ---

st.title("Extraction de lignes électriques depuis un nuage LAS/LAZ")

st.markdown("""
Ce prototype permet de téléverser un fichier LAS/LAZ, d'extraire les points situés au-dessus d'un seuil (en hauteur)
puis d'appliquer un clustering (DBSCAN) afin d'isoler des structures linéaires potentielles (comme des lignes électriques).
Les segments extraits sont ensuite affichés en plan 2D.
""")

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléversez un fichier LAS ou LAZ", type=["las", "laz"])

if uploaded_file is not None:
    with st.spinner("Lecture du fichier et traitement du nuage de points..."):
        # Lecture du fichier avec laspy
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
            
            # Estimation du niveau de sol (par exemple le 10ème percentile)
            ground_level = np.percentile(z, 10)
            st.write(f"Estimation du niveau du sol (10ème percentile) : **{ground_level:.2f}**")
            
            # Paramétrage du seuil en hauteur (modulable par l'utilisateur)
            seuil_defaut = ground_level + 5.0
            seuil = st.number_input("Seuil de hauteur (points avec z > seuil seront considérés)", 
                                    value=seuil_defaut, step=0.5)
            
            # Filtrage des points candidats (potentiellement des câbles ou structures surélevées)
            mask = z > seuil
            candidate_points = points[mask]
            st.write(f"Nombre de points candidats (z > {seuil}) : **{candidate_points.shape[0]}**")
            
            if candidate_points.shape[0] == 0:
                st.error("Aucun point au-dessus du seuil sélectionné.")
            else:
                # Utilisation uniquement des coordonnées XY pour l'extraction de droites
                candidate_points_xy = candidate_points[:, :2]
                
                st.markdown("### Paramètres du clustering DBSCAN")
                eps = st.slider("DBSCAN - eps", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                min_samples = st.slider("DBSCAN - min_samples", min_value=1, max_value=20, value=5)
                cluster_min_points = st.slider("Nombre minimal de points par cluster", 
                                               min_value=2, max_value=100, value=10)
                
                # Extraction des lignes
                lines = extract_lines(candidate_points_xy, eps=eps, 
                                        min_samples=min_samples, cluster_min_points=cluster_min_points)
                st.write(f"Nombre de segments linéaires extraits : **{len(lines)}**")
                
                # Affichage sur une carte 2D
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(candidate_points_xy[:, 0], candidate_points_xy[:, 1],
                           s=1, color='grey', label='Points candidats')
                
                for i, line in enumerate(lines):
                    ax.plot(line[:, 0], line[:, 1], linewidth=2, label=f"Ligne {i+1}")
                
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title("Lignes électriques extraites")
                ax.legend(loc='best', fontsize='small')
                st.pyplot(fig)
                
    st.success("Traitement terminé. (Raisonnement terminé en 13 secondes)")
