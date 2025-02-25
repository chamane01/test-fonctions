import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.ndimage import morphology

# --- Nouvelle fonction SMRF pour la filtration du sol ---
def smrf_filter(points, cell_size=2.0, slope_threshold=0.2, window_size=3):
    """
    Implémentation simplifiée du Simple Morphological Filter (SMRF)
    pour l'extraction des points de sol.
    """
    coords = points[:, :2]
    z = points[:, 2]
    
    # Création d'une grille régulière
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    grid_x = np.arange(x_min, x_max, cell_size)
    grid_y = np.arange(y_min, y_max, cell_size)
    
    grid_ground = np.full((len(grid_y), len(grid_x)), np.nan)
    
    # Remplissage de la grille avec l'altitude minimale
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            mask = (coords[:, 0] >= x) & (coords[:, 0] < x + cell_size) & \
                   (coords[:, 1] >= y) & (coords[:, 1] < y + cell_size)
            if np.any(mask):
                grid_ground[j, i] = np.min(z[mask])
    
    # Filtration morphologique
    opened = morphology.grey_opening(grid_ground, size=window_size)
    
    # Interpolation pour obtenir le modèle de sol
    from scipy.interpolate import RectBivariateSpline
    x_idx = np.arange(grid_x.shape[0])
    y_idx = np.arange(grid_y.shape[0])
    interp_spline = RectBivariateSpline(y_idx, x_idx, opened)
    return interp_spline
# --- Fonctions utilitaires modifiées ---

def fit_line_pca(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # Vérification de la linéarité
    if pca.explained_variance_ratio_[0] < 0.9:  # Seuil de linéarité
        return None
    
    pc1 = pca.components_[0]
    proj = np.dot(points - pca.mean_, pc1)
    t_min, t_max = proj.min(), proj.max()
    return np.array([pca.mean_ + t_min * pc1, pca.mean_ + t_max * pc1])

def extract_lines(points_xy, eps=0.5, min_samples=3, cluster_min_points=5000, linearity_threshold=0.9):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_xy)
    
    lines = []
    for label in set(labels):
        if label == -1:
            continue
            
        cluster_points = points_xy[labels == label]
        if len(cluster_points) < cluster_min_points:
            continue
            
        line_segment = fit_line_pca(cluster_points)
        if line_segment is not None:
            lines.append(line_segment)
            
    return lines

# --- Application Streamlit modifiée ---

st.title("Extraction de lignes électriques depuis un nuage LAS/LAZ")

# ... (le reste du code d'interface reste similaire jusqu'au traitement du fichier)

            else:
                # --- Nouvelle section SMRF ---
                st.markdown("### Paramètres SMRF pour la détection du sol")
                cell_size = st.slider("Taille de cellule SMRF (m)", 0.5, 5.0, 2.0)
                slope_threshold = st.slider("Seuil de pente SMRF", 0.1, 1.0, 0.2)
                
                with st.spinner("Calcul du modèle de terrain avec SMRF..."):
                    smrf_model = smrf_filter(points, cell_size, slope_threshold)
                    
                    # Calcul de la hauteur normalisée
                    grid_coords = np.floor((points[:, :2] - np.min(points[:, :2], axis=0)) / cell_size).astype(int)
                    ground_z = smrf_model(grid_coords[:, 1], grid_coords[:, 0], grid=True)
                    normalized_z = points[:, 2] - ground_z
                    
                # --- Nouveaux paramètres de hauteur relative ---
                st.markdown("### Paramètres de hauteur des lignes")
                min_height = st.slider("Hauteur minimale au-dessus du sol (m)", 5.0, 50.0, 15.0)
                max_height = st.slider("Hauteur maximale au-dessus du sol (m)", 20.0, 100.0, 40.0)
                
                # Filtrage amélioré
                mask = (normalized_z > min_height) & (normalized_z < max_height)
                candidate_points = points[mask]
                candidate_points_xy = candidate_points[:, :2]
                
                # --- Paramètres DBSCAN adaptés ---
                st.markdown("### Paramètres du clustering adaptés")
                eps = st.slider("DBSCAN - eps", 0.1, 5.0, 0.5, 0.1)
                min_samples = st.slider("DBSCAN - min_samples", 1, 20, 3)
                cluster_min_points = st.slider("Points minimum par cluster", 1000, 15000, 5000, 100)
                linearity_threshold = st.slider("Seuil de linéarité PCA", 0.7, 1.0, 0.9)
                
                # ... (le reste du code reste similaire avec ajout du paramètre linearity_threshold)
