import streamlit as st
import pandas as pd
import numpy as np
import laspy
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.cluster import DBSCAN
from streamlit_folium import folium_static

# Fonction pour lire un fichier LAS/LAZ
def read_las_file(file_path):
    """Lit un fichier LAS/LAZ et retourne un DataFrame contenant les points."""
    try:
        las = laspy.read(file_path)
        if len(las.x) == 0:
            raise ValueError("Le fichier LAS/LAZ est vide ou mal formaté.")

        df = pd.DataFrame({
            'x': las.x,
            'y': las.y,
            'z': las.z,
            'intensity': las.intensity if hasattr(las, 'intensity') else 0,
            'classification': las.classification if hasattr(las, 'classification') else -1
        })
        
        return df

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier LAS/LAZ : {e}")
        return pd.DataFrame()

# Fonction de clustering avec DBSCAN
def cluster_points(df, eps=3, min_samples=10):
    """Applique DBSCAN sur les points pour regrouper les objets."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df[['x', 'y', 'z']])
    df['cluster'] = clustering.labels_  # Assigner les clusters
    return df

# Fonction pour afficher les clusters en 2D
def plot_2d_clusters(df):
    """Affiche les clusters en 2D avec matplotlib."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['x'], y=df['y'], hue=df['cluster'], palette="viridis", s=10)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Visualisation 2D des clusters")
    plt.legend(title="Cluster", loc='upper right', bbox_to_anchor=(1.15, 1))
    st.pyplot(plt)

# Fonction pour afficher les points sur une carte
def plot_on_map(df):
    """Affiche les points clusterisés sur une carte avec Folium."""
    if df.empty:
        st.warning("Aucun point à afficher sur la carte.")
        return
    
    # Déterminer le centre de la carte
    center_x, center_y = df['x'].mean(), df['y'].mean()
    
    # Créer une carte Folium
    m = folium.Map(location=[center_y, center_x], zoom_start=15)
    
    # Assigner une couleur unique à chaque cluster
    clusters = df['cluster'].unique()
    colors = sns.color_palette("hsv", len(clusters)).as_hex()
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(clusters)}

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['y'], row['x']], 
            radius=2,
            color=cluster_colors[row['cluster']],
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    folium_static(m)

# Interface Streamlit
st.title("📌 Téléversement et Clustering de Fichiers LAS/LAZ")

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléverse un fichier LAS/LAZ", type=["las", "laz"])

if uploaded_file:
    # Lire le fichier et afficher les premières lignes
    st.write("📊 **Aperçu des données**")
    df = read_las_file(uploaded_file)
    st.dataframe(df.head())

    if not df.empty:
        # Paramètres DBSCAN
        eps = st.slider("Échelle de regroupement (eps)", 1, 10, 3)
        min_samples = st.slider("Nombre minimal de points par cluster", 5, 50, 10)

        # Appliquer le clustering
        df = cluster_points(df, eps, min_samples)

        # Afficher la visualisation 2D
        st.subheader("📍 Visualisation 2D des Clusters")
        plot_2d_clusters(df)

        # Afficher la carte interactive
        st.subheader("🗺️ Carte des Clusters")
        plot_on_map(df)
