import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN

# Configuration de la page
st.set_page_config(page_title="LiDAR Clustering", layout="wide")
st.title("🛩️ Classification d'objets LiDAR avec SMRF")

# Fonction de traitement SMRF (simplifiée)
def apply_smrf(points, params):
    """
    Applique un filtre morphologique simplifié.
    À remplacer par une implémentation PDAL réelle.
    """
    # Ici, vous devriez implémenter la logique SMRF avec PDAL
    # Ceci est une simulation de classification
    z = points[:, 2]
    return (z > params['elevation_threshold']).astype(int)

# Téléversement de fichier
las_file = st.file_uploader("Téléversez un fichier LAS/LAZ", type=['las', 'laz'])

if las_file:
    # Lecture du fichier LAS/LAZ
    with las_file.open() as f:
        las = laspy.read(f)
        points = np.vstack((las.x, las.y, las.z)).transpose()

    # Sélection de l'objet à détecter
    object_type = st.selectbox("Sélectionnez l'objet à détecter", [
        "Bâtiments 🏢", 
        "Basse végétation 🌱",
        "Arbustes 🌿",
        "Arbres 🌳",
        "Lignes électriques ⚡",
        "Cours d’eau 🌊"
    ])

    # Paramètres SMRF par défaut
    params = {
        'Bâtiments 🏢': (1.5, 25, 10, 3.5, 1),
        'Basse végétation 🌱': (0.75, 7.5, 4.5, 0.6, 1),
        'Arbustes 🌿': (1.5, 11.5, 7.5, 2, 1),
        'Arbres 🌳': (3.5, 30, 15, 12.5, 2),
        'Lignes électriques ⚡': (0.75, 12.5, 22.5, 30, 1),
        'Cours d’eau 🌊': (2, 15, 3.5, -0.5, 1)
    }[object_type]

    # Widgets de paramétrage
    with st.expander("Paramètres SMRF"):
        col1, col2, col3 = st.columns(3)
        with col1:
            cell_size = st.slider("Taille de maille (m)", 0.5, 5.0, params[0], 0.5)
            window_size = st.slider("Taille de fenêtre (m)", 5, 40, params[1], 5)
        with col2:
            slope_threshold = st.slider("Seuil de pente (°)", 0, 30, params[2], 1)
            elevation_threshold = st.slider("Seuil d'élévation (m)", -2.0, 50.0, params[3], 0.5)
        with col3:
            iterations = st.slider("Itérations", 1, 3, params[4], 1)

    # Bouton de traitement
    if st.button("Lancer la classification"):
        # Application du "filtre" (simulation)
        smrf_params = {
            'elevation_threshold': elevation_threshold,
            'slope_threshold': slope_threshold
        }
        
        classified = apply_smrf(points, smrf_params)
        object_points = points[classified == 1]

        # Clustering avec DBSCAN
        clustering = DBSCAN(eps=cell_size, min_samples=10).fit(object_points[:, :2])
        
        # Création du DataFrame
        df = pd.DataFrame({
            'X': object_points[:, 0],
            'Y': object_points[:, 1],
            'Cluster': clustering.labels_
        })

        # Visualisation
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            df['X'], 
            df['Y'], 
            c=df['Cluster'], 
            cmap='tab20', 
            s=1, 
            alpha=0.6
        )
        
        ax.set_title(f"Classification des {object_type.split()[0]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(scatter, ax=ax, label='Clusters')
        
        st.pyplot(fig)
        
        # Statistiques
        st.subheader("📊 Statistiques de classification")
        col1, col2, col3 = st.columns(3)
        col1.metric("Points totaux", len(points))
        col2.metric("Points classifiés", len(object_points))
        col3.metric("Clusters détectés", df['Cluster'].nunique()))

# Instructions d'installation
st.sidebar.markdown("### 📦 Dépendances requises")
st.sidebar.code("""
pip install streamlit laspy numpy pandas matplotlib scikit-learn
""")
st.sidebar.markdown("""
**Note** : Pour une implémentation réelle du SMRF :
1. Installer PDAL (`conda install -c conda-forge pdal`)
2. Implémenter le pipeline SMRF
3. Remplacer la fonction `apply_smrf` simulée
""")
