import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def apply_smrf(points, cell_size, slope, window, elevation_threshold):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
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
    
    ground_z = gaussian_filter(ground_z, sigma=window / 2.0)
    
    ground_mask = np.zeros(len(z), dtype=bool)
    for i in range(len(x)):
        ix = np.searchsorted(grid_x, x[i]) - 1
        iy = np.searchsorted(grid_y, y[i]) - 1
        if 0 <= ix < len(grid_x) and 0 <= iy < len(grid_y):
            if z[i] <= ground_z[ix, iy] + elevation_threshold + slope * (z[i] - ground_z[ix, iy]):
                ground_mask[i] = True
    
    return ground_mask

def classify_objects(points, ground_mask, params):
    z = points[:, 2]
    classes = {}
    
    for obj, (min_z, max_z) in params.items():
        classes[obj] = (z > min_z) & (z <= max_z)
    
    classified_points = {key: points[~ground_mask & mask] for key, mask in classes.items()}
    
    return classified_points

def main():
    st.title("Classification des objets LiDAR avec paramètres spécifiques")

    uploaded_file = st.file_uploader("Fichier LAS/LAZ", type=["las", "laz"])
    if uploaded_file is not None:
        las = laspy.read(uploaded_file)
        points = np.vstack((las.x, las.y, las.z)).T
        st.write(f"Total points: {len(points)}")

        object_params = {}
        for obj in ["Bâtiments", "Basse végétation", "Arbustes", "Arbres", "Lignes électriques", "Cours d’eau"]:
            with st.expander(f"Paramètres pour {obj}"):
                cell_size = st.slider(f"Taille de cellule ({obj})", 0.5, 5.0, 1.0, 0.5)
                slope = st.slider(f"Tolérance pente ({obj})", 0.0, 1.0, 0.2, 0.05)
                window = st.slider(f"Fenêtre morphologique ({obj})", 5, 30, 18, 1)
                elevation_threshold = st.slider(f"Seuil d'élévation ({obj})", 0.1, 2.0, 0.5, 0.1)
                min_height = st.slider(f"Hauteur min ({obj})", 0.0, 50.0, 0.5, 0.1)
                max_height = st.slider(f"Hauteur max ({obj})", 0.0, 50.0, 10.0, 0.5)
                object_params[obj] = (min_height, max_height)

        if st.button("Lancer le traitement"):
            with st.spinner("Application du filtre SMRF..."):
                ground_mask = apply_smrf(points, 1.0, 0.2, 18, 0.5)  # Paramètres génériques pour le sol
            
            classified_points = classify_objects(points, ground_mask, object_params)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = {"Bâtiments": "blue", "Basse végétation": "green", "Arbustes": "yellow", 
                      "Arbres": "darkgreen", "Lignes électriques": "red", "Cours d’eau": "cyan"}
            
            for label, pts in classified_points.items():
                if len(pts) > 0:
                    ax.scatter(pts[:, 0], pts[:, 1], s=1, color=colors[label], label=label)
            
            ax.set_title("Classification des objets hors-sol")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
