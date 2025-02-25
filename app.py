import streamlit as st
import pandas as pd
import numpy as np
import laspy
import tempfile
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

def read_las_file(file_path):
    with laspy.open(file_path) as las:
        points = las.read()
        df = pd.DataFrame({
            'X': points.x,
            'Y': points.y,
            'Z': points.z
        })
    return df

def cluster_points(df, eps=2, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df[['X', 'Y', 'Z']])
    df['Cluster'] = clustering.labels_
    return df

def plot_clusters(df):
    unique_clusters = df['Cluster'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in unique_clusters:
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(cluster_data['X'], cluster_data['Y'], s=1, label=f'Cluster {cluster_id}', color=colors(cluster_id % 10))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Clustering des points LAS')
    ax.legend()
    st.pyplot(fig)

st.title('Téléverser et Clusteriser un fichier LAS/LAZ')

uploaded_file = st.file_uploader("Téléverser un fichier LAS/LAZ", type=["las", "laz"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(uploaded_file.getbuffer())
        file_path = tmpfile.name
    
    df = read_las_file(file_path)
    
    st.write("Aperçu des données :", df.head())
    
    eps = st.slider("Distance max entre points (eps)", 0.1, 10.0, 2.0, 0.1)
    min_samples = st.slider("Nombre minimum de points par cluster", 1, 50, 10, 1)
    
    df_clustered = cluster_points(df, eps, min_samples)
    plot_clusters(df_clustered)
    
    os.remove(file_path)
