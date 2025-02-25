import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def csf_segmentation(points, ground_percentile=10, height_threshold=1.0):
    """
    Sépare les points de sol des points non-sol.
    On considère comme sol les points dont la coordonnée Z est inférieure
    au percentile 'ground_percentile' des Z augmenté d'un seuil.
    (Il s'agit ici d'une approximation de la méthode CSF.)
    """
    z_vals = points[:, 2]
    ground_limit = np.percentile(z_vals, ground_percentile) + height_threshold
    ground_mask = z_vals <= ground_limit
    return points[ground_mask], points[~ground_mask]

def main():
    st.title("Extraction de lignes haute tension avec CSF + DBSCAN")
    
    uploaded_file = st.file_uploader("Fichier LAS/LAZ", type=["las", "laz"])
    if uploaded_file is not None:
        try:
            las = laspy.read(uploaded_file)
            points = np.vstack((las.x, las.y, las.z)).T
            st.write(f"Nombre total de points : {len(points)}")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            return

        # Paramètres pour la segmentation CSF simulée
        ground_percentile = st.slider("Percentile pour segmentation du sol", 1, 50, 10, 1)
        ground_height_threshold = st.slider("Seuil de hauteur pour le sol", 0.1, 5.0, 1.0, 0.1)

        with st.spinner("Segmentation CSF (simulation)..."):
            ground_points, non_ground_points = csf_segmentation(points, ground_percentile, ground_height_threshold)
        st.write(f"Nombre de points de sol : {len(ground_points)}")
        st.write(f"Nombre de points non-sol : {len(non_ground_points)}")

        # Paramètres de hauteur pour extraire les points candidats (câbles)
        z_min = st.slider("Hauteur minimum pour les câbles", 0.0, 100.0, 20.0, 1.0)
        z_max = st.slider("Hauteur maximum pour les câbles", 0.0, 200.0, 80.0, 1.0)

        # Filtrer les points non-sol selon la plage de hauteur
        mask_height = (non_ground_points[:, 2] > z_min) & (non_ground_points[:, 2] < z_max)
        candidate_points = non_ground_points[mask_height]
        candidate_points_xy = candidate_points[:, :2]
        st.write(f"Nombre de points candidats pour lignes : {len(candidate_points_xy)}")

        if candidate_points_xy.shape[0] == 0:
            st.warning("Aucun point candidat trouvé pour le clustering. Vérifiez les paramètres de hauteur.")
            return

        # Paramètres DBSCAN
        eps = st.slider("DBSCAN eps", 0.1, 10.0, 1.0, 0.1)
        min_samples = st.slider("DBSCAN min_samples", 1, 20, 5, 1)
        cluster_min_points = st.slider("Nombre minimum de points par cluster", 100, 20000, 10000, 100)

        with st.spinner("Clustering DBSCAN..."):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(candidate_points_xy)

        unique_labels = set(labels)
        lines = []
        for label in unique_labels:
            if label == -1:
                continue  # Ignorer le bruit
            cluster_pts = candidate_points_xy[labels == label]
            if len(cluster_pts) >= cluster_min_points:
                lines.append(cluster_pts)

        st.write(f"Nombre de clusters retenus : {len(lines)}")

        # Visualisation des clusters
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(candidate_points_xy[:, 0], candidate_points_xy[:, 1], s=1, color='grey', label="Points candidats")
        for i, cluster_pts in enumerate(lines):
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=1, label=f"Cluster {i+1}")
        ax.set_title("Clusters de câbles potentiels")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
