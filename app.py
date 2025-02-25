import streamlit as st
import pandas as pd
import numpy as np
import laspy
from sklearn.cluster import DBSCAN

# Fonction pour lire un fichier LAS et extraire les points
def read_las_file(file_path):
    """Lit un fichier .las et retourne un DataFrame avec les coordonnées des points."""
    try:
        with laspy.open(file_path) as las_file:
            las = las_file.read()
            df = pd.DataFrame({
                'x': las.x,
                'y': las.y,
                'z': las.z
            })
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier LAS : {e}")
        return pd.DataFrame()

# Fonction de clustering avec DBSCAN
def cluster_points(df, eps=3, min_samples=10):
    """Applique DBSCAN sur les points pour regrouper les objets."""
    if df.empty:
        st.warning("Le DataFrame est vide, impossible d'appliquer DBSCAN.")
        return df

    # Vérifier les valeurs manquantes et les supprimer
    df = df.dropna(subset=['x', 'y', 'z'])

    # Vérifier si les colonnes existent et sont bien numériques
    try:
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(float)
    except ValueError:
        st.error("Les colonnes x, y, z contiennent des valeurs non numériques.")
        return df

    # Vérifier qu'il y a assez de points pour DBSCAN
    if len(df) < min_samples:
        st.warning("Pas assez de points valides pour appliquer DBSCAN.")
        df['cluster'] = -1  # Mettre tout dans un seul cluster par défaut
        return df

    # Appliquer DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(df[['x', 'y', 'z']])
    df['cluster'] = clustering.labels_
    return df

# Interface Streamlit
st.title("Analyse et Clustering de Fichiers LAS")

uploaded_file = st.file_uploader("Chargez un fichier .las", type=["las"])

if uploaded_file is not None:
    st.success("Fichier chargé avec succès !")

    # Sauvegarde temporaire du fichier
    file_path = "/tmp/uploaded_file.las"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Lecture et affichage des données
    df = read_las_file(file_path)

    if not df.empty:
        st.write("Aperçu des données :")
        st.dataframe(df.head())

        # Paramètres pour DBSCAN
        eps = st.slider("Distance maximale entre points (eps)", 1, 10, 3)
        min_samples = st.slider("Nombre minimal de points par cluster", 1, 20, 10)

        if st.button("Appliquer DBSCAN"):
            df = cluster_points(df, eps, min_samples)
            st.write("Résultats du clustering :")
            st.dataframe(df.head())

            # Affichage du nombre de clusters trouvés
            n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].unique() else 0)
            st.write(f"Nombre de clusters détectés : {n_clusters}")

            # Option de téléchargement du fichier avec les clusters
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger les résultats", data=csv, file_name="clustering_results.csv", mime="text/csv")

