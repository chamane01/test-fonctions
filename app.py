import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.interpolate import griddata

# --- Filtre SMRF pour extraire le sol ---
def smrf_filter(points, grid_size, window_size, height_threshold):
    """
    Implémente un filtre SMRF simplifié.
    1. Il construit une grille (MNT) en assignant à chaque cellule la valeur minimale de z.
    2. Une ouverture morphologique (érosion puis dilatation) est appliquée pour lisser le MNT.
    3. Le modèle est ensuite interpolé sur l'ensemble des points pour déterminer la différence.
    Les points dont la différence (z - modèle) est inférieure à height_threshold sont considérés comme sol.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Création de la grille
    xi = np.arange(x.min(), x.max(), grid_size)
    yi = np.arange(y.min(), y.max(), grid_size)
    X, Y = np.meshgrid(xi, yi)
    grid_z = np.full(X.shape, np.nan)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mask = (x >= X[i, j]) & (x < X[i, j] + grid_size) & (y >= Y[i, j]) & (y < Y[i, j] + grid_size)
            if np.any(mask):
                grid_z[i, j] = np.min(z[mask])
    # Remplacer les NaN par la valeur minimale du nuage
    grid_z = np.where(np.isnan(grid_z), np.nanmin(z), grid_z)
    
    # Application d'une ouverture morphologique (érosion puis dilatation)
    eroded = minimum_filter(grid_z, size=window_size)
    opened = maximum_filter(eroded, size=window_size)
    
    # Interpolation du MNT filtré sur tous les points
    points_grid = np.column_stack((X.ravel(), Y.ravel()))
    ground_model = griddata(points_grid, opened.ravel(), (x, y), method='linear')
    ground_model = np.where(np.isnan(ground_model), np.nanmin(z), ground_model)
    
    # Détermination des points sol : différence inférieure au seuil
    diff = z - ground_model
    ground_mask = diff < height_threshold
    return ground_mask, ground_model

# --- Classification des clusters non-sol ---
def classify_cluster(cluster_points, cluster_ground):
    """
    Heuristique de classification basée sur :
      - La linéarité (via PCA sur XY) : si le cluster est très linéaire, il est considéré comme une ligne électrique.
      - La différence moyenne de hauteur par rapport au sol.
      - La taille (nombre de points) du cluster.
    """
    # Calcul de la hauteur moyenne relative au sol
    mean_diff = np.mean(cluster_points[:, 2] - cluster_ground)
    # Analyse de la forme (PCA sur XY)
    pca = PCA(n_components=2)
    pca.fit(cluster_points[:, :2])
    eigenvalues = pca.explained_variance_
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] != 0 else 0

    # Heuristiques de classification (les seuils ont été ajustés pour une détection améliorée)
    if linearity > 0.85 and len(cluster_points) < 5000:
        return "Ligne électrique"
    elif mean_diff > 3.0 and len(cluster_points) > 500:
        return "Bâtiment"
    elif 1.0 < mean_diff <= 3.0:
        return "Végétation"
    else:
        return "Autres"

# --- Extraction d'un segment linéaire par PCA (pour visualiser les lignes électriques) ---
def fit_line_pca(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    pc1 = pca.components_[0]
    proj = np.dot(points - pca.mean_, pc1)
    t_min, t_max = proj.min(), proj.max()
    point_min = pca.mean_ + t_min * pc1
    point_max = pca.mean_ + t_max * pc1
    return np.array([point_min, point_max])

# --- Application Streamlit ---
st.title("Classification et extraction dans un nuage LAS/LAZ")

st.markdown("""
Ce prototype permet de téléverser un fichier LAS/LAZ, d'appliquer un filtre SMRF pour estimer le sol, 
puis de classifier les points non-sol en différentes catégories (végétation, bâtiment, ligne électrique, autres)
via un clustering DBSCAN et des critères heuristiques. Chaque classe est affichée avec une couleur dédiée.
""")

# Paramètres réglables par l'utilisateur via la sidebar
st.sidebar.header("Paramètres SMRF")
grid_size = st.sidebar.number_input("Taille de la cellule (m)", value=1.0, step=0.1)
window_size = st.sidebar.slider("Taille de la fenêtre (pour filtrage morphologique)", min_value=1, max_value=10, value=3)
height_threshold = st.sidebar.number_input("Seuil de différence pour le sol (m)", value=0.5, step=0.1)

st.sidebar.header("Paramètres DBSCAN")
eps = st.sidebar.slider("DBSCAN - eps", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
min_samples = st.sidebar.slider("DBSCAN - min_samples", min_value=1, max_value=20, value=5)
cluster_min_points = st.sidebar.slider("Nombre minimal de points par cluster", min_value=50, max_value=20000, value=1000, step=50)

# Téléversement du fichier LAS/LAZ
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
            st.write(f"Nombre total de points : **{points.shape[0]}**")
            
            # Application du filtre SMRF pour estimer le sol
            ground_mask, ground_model = smrf_filter(points, grid_size, window_size, height_threshold)
            n_ground = np.sum(ground_mask)
            st.write(f"Points classifiés comme sol : **{n_ground}**")
            st.write(f"Points non-sol : **{points.shape[0] - n_ground}**")
            
            # Séparation des points sol et non-sol
            ground_points = points[ground_mask]
            non_ground_points = points[~ground_mask]
            # Récupérer le modèle du sol pour les points non-sol (pour le calcul des différences)
            non_ground_ground = ground_model[~ground_mask]
            
            # Extraction des clusters sur les points non-sol (en utilisant uniquement les coordonnées XY)
            non_ground_xy = non_ground_points[:, :2]
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(non_ground_xy)
            unique_labels = set(labels)
            
            clusters = {}
            # Pour stocker la classification par cluster
            cluster_class = {}
            # Pour les segments linéaires (potentielles lignes électriques)
            line_segments = []
            for label in unique_labels:
                if label == -1:
                    continue  # bruit
                idx = labels == label
                if np.sum(idx) < cluster_min_points:
                    continue
                cluster_pts = non_ground_points[idx]
                cluster_ground_vals = non_ground_ground[idx]
                cls = classify_cluster(cluster_pts, cluster_ground_vals)
                cluster_class[label] = cls
                clusters[label] = cluster_pts
                # Pour les lignes électriques, extraire un segment par PCA
                if cls == "Ligne électrique":
                    line_segments.append(fit_line_pca(cluster_pts))
            
            st.write(f"Nombre de clusters retenus : **{len(clusters)}**")
            
            # Dictionnaire de couleurs pour chaque classe
            colors = {
                "Ligne électrique": "red",
                "Bâtiment": "grey",
                "Végétation": "green",
                "Autres": "blue"
            }
            
            # Affichage de la classification sur une carte 2D
            fig, ax = plt.subplots(figsize=(10, 8))
            # Affichage des points sol
            ax.scatter(ground_points[:, 0], ground_points[:, 1], s=1, color="saddlebrown", label="Sol")
            # Affichage des clusters non-sol
            for label, pts in clusters.items():
                cls = cluster_class[label]
                ax.scatter(pts[:, 0], pts[:, 1], s=1, color=colors.get(cls, "black"), label=cls)
            # Affichage des segments de lignes électriques
            for seg in line_segments:
                ax.plot(seg[:, 0], seg[:, 1], linewidth=2, color="red", label="Segment ligne électrique")
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Classification du nuage de points")
            # Pour éviter les doublons dans la légende
            handles, labels_legend = ax.get_legend_handles_labels()
            by_label = dict(zip(labels_legend, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
            st.pyplot(fig)
            
            st.success("Traitement terminé.")
