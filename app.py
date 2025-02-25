import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
# from smrf import SMRF  # Hypothétique import si on a un module SMRF Python

def apply_smrf(points, cell_size=1.0, slope=0.2, window=18, elevation_threshold=0.5):
    """
    Pseudo-code d'un filtre SMRF :
    1. Créer une grille 2D à partir de points (X, Y).
    2. Calculer la hauteur min par cellule -> surface initiale.
    3. Faire une opération morphologique (opening) avec 'window'.
    4. Marquer en hors-sol les points dont la hauteur est > (surface + elevation_threshold),
       en tenant compte du slope pour les terrains en pente.
    5. Retourne un masque (True = sol, False = non-sol).
    """
    # ... Implémentation à adapter ou à remplacer par une librairie existante ...
    # Ici, pour l'exemple, on considère que tous les points sont hors-sol.
    ground_mask = np.full(len(points), False, dtype=bool)
    return ground_mask

def main():
    st.title("Extraction de lignes haute tension avec SMRF + DBSCAN")

    uploaded_file = st.file_uploader("Fichier LAS/LAZ", type=["las", "laz"])
    if uploaded_file is not None:
        las = laspy.read(uploaded_file)
        x, y, z = las.x, las.y, las.z
        points = np.vstack((x, y, z)).T

        st.write(f"Total points: {len(points)}")

        # Paramètres SMRF
        cell_size = st.slider("Taille de cellule SMRF (m)", 0.5, 5.0, 1.0, 0.5)
        slope = st.slider("Slope (tolérance de pente)", 0.0, 1.0, 0.2, 0.05)
        window = st.slider("Window (morphological opening)", 5, 30, 18, 1)
        elevation_threshold = st.slider("Elevation threshold", 0.1, 2.0, 0.5, 0.1)

        # Paramètres de hauteur pour les câbles
        z_min = st.slider("Hauteur minimum câbles", 0.0, 100.0, 20.0, 1.0)
        z_max = st.slider("Hauteur maximum câbles", 0.0, 200.0, 80.0, 1.0)

        # Paramètres DBSCAN
        eps = st.slider("DBSCAN eps", 0.1, 10.0, 1.0, 0.1)
        min_samples = st.slider("DBSCAN min_samples", 1, 20, 5, 1)
        cluster_min_points = st.slider("Nombre min de points par cluster", 100, 20000, 10000, 100)

        if st.button("Lancer le traitement"):
            with st.spinner("Classification SMRF..."):
                ground_mask = apply_smrf(points, cell_size, slope, window, elevation_threshold)

            # On ne garde que les points hors-sol
            non_ground_points = points[~ground_mask]

            # Double seuil en hauteur
            mask_height = (non_ground_points[:, 2] > z_min) & (non_ground_points[:, 2] < z_max)
            candidate_points = non_ground_points[mask_height]
            candidate_points_xy = candidate_points[:, :2]

            st.write(f"Points candidats pour lignes: {len(candidate_points_xy)}")
            
            # Vérifications de debug
            st.write("Shape de candidate_points_xy :", candidate_points_xy.shape)
            st.write("Présence de NaN :", np.isnan(candidate_points_xy).any())
            st.write("Présence d'inf :", np.isinf(candidate_points_xy).any())

            # Clustering DBSCAN
            with st.spinner("Clustering DBSCAN..."):
                if candidate_points_xy.size == 0:
                    st.error("Aucun point candidat pour le clustering. Veuillez vérifier vos paramètres ou vos données.")
                else:
                    db = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = db.fit_predict(candidate_points_xy)
                    
                    # Filtrage des clusters trop petits
                    unique_labels = set(labels)
                    lines = []
                    for label in unique_labels:
                        if label == -1:
                            continue
                        cluster_pts = candidate_points_xy[labels == label]
                        if len(cluster_pts) >= cluster_min_points:
                            # Ici, vous pouvez extraire le segment par PCA ou autre méthode
                            lines.append(cluster_pts)
                    
                    st.write(f"Clusters retenus : {len(lines)}")

                    # Visualisation des résultats
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(candidate_points_xy[:, 0], candidate_points_xy[:, 1], s=1, color='grey', label='Points candidats')
                    for i, cluster_pts in enumerate(lines):
                        ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=1, label=f"Cluster {i+1}")
                    ax.set_title("Clusters de câbles potentiels")
                    ax.legend()
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
