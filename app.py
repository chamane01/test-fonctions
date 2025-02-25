import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def apply_smrf(points, cell_size=1.0, slope=0.2, window=18, elevation_threshold=0.5):
    """
    Implémentation simplifiée du filtre SMRF pour séparer sol et hors-sol.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Création d'une grille 2D pour le filtrage
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    grid_x = np.arange(min_x, max_x, cell_size)
    grid_y = np.arange(min_y, max_y, cell_size)
    
    ground_z = np.zeros((len(grid_x), len(grid_y))) + np.inf
    
    for i in range(len(x)):
        ix = np.searchsorted(grid_x, x[i]) - 1
        iy = np.searchsorted(grid_y, y[i]) - 1
        if 0 <= ix < len(grid_x) and 0 <= iy < len(grid_y):
            ground_z[ix, iy] = min(ground_z[ix, iy], z[i])
    
    # Lissage de la surface de référence
    ground_z = gaussian_filter(ground_z, sigma=window / 2.0)
    
    # Détection des points hors-sol
    ground_mask = np.zeros(len(z), dtype=bool)
    for i in range(len(x)):
        ix = np.searchsorted(grid_x, x[i]) - 1
        iy = np.searchsorted(grid_y, y[i]) - 1
        if 0 <= ix < len(grid_x) and 0 <= iy < len(grid_y):
            if z[i] <= ground_z[ix, iy] + elevation_threshold + slope * (z[i] - ground_z[ix, iy]):
                ground_mask[i] = True
    
    return ground_mask

def classify_objects(points, ground_mask):
    """
    Classification des objets hors-sol en fonction de la hauteur et d'autres critères.
    """
    z = points[:, 2]
    
    classes = {
        "Bâtiments": (z > 2) & (z < 20),  # Hauteur typique des bâtiments
        "Basse végétation": (z > 0.2) & (z <= 1.0),
        "Arbustes": (z > 1.0) & (z <= 3.0),
        "Arbres": (z > 3.0),
        "Lignes électriques": (z > 5.0) & (np.std(z) < 1.5),  # Objets en hauteur et isolés
        "Cours d’eau": (z <= 0.5),  # Proche du sol
    }

    classified_points = {key: points[~ground_mask & mask] for key, mask in classes.items()}
    
    return classified_points

def main():
    st.title("Détection SMRF et classification des objets hors-sol")

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

        if st.button("Lancer le traitement"):
            with st.spinner("Application du filtre SMRF..."):
                ground_mask = apply_smrf(points, cell_size, slope, window, elevation_threshold)
            
            # Classification des objets
            classified_points = classify_objects(points, ground_mask)

            st.write(f"Points classifiés comme sol : {np.sum(ground_mask)}")
            st.write(f"Points classifiés comme hors-sol : {np.sum(~ground_mask)}")

            # Visualisation des résultats
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colors = {
                "Bâtiments": "blue",
                "Basse végétation": "green",
                "Arbustes": "yellow",
                "Arbres": "darkgreen",
                "Lignes électriques": "red",
                "Cours d’eau": "cyan",
            }
            
            for label, points in classified_points.items():
                if len(points) > 0:
                    ax.scatter(points[:, 0], points[:, 1], s=1, color=colors[label], label=label)

            ax.set_title("Classification des objets hors-sol")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
