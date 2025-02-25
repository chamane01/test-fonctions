import streamlit as st
import laspy
import numpy as np
import pandas as pd
import pydeck as pdk
from tempfile import NamedTemporaryFile
import pdal
import json

st.title("Détection de lignes électriques LiDAR")

# Paramètres SMRF ajustables
st.sidebar.header("Paramètres SMRF")
cell_size = st.sidebar.slider("Taille de cellule (m)", 0.5, 1.0, 0.5)
window_size = st.sidebar.slider("Taille de fenêtre (m)", 10.0, 20.0, 15.0)
elevation_threshold = st.sidebar.slider("Seuil d'élévation (m)", 0.5, 1.0, 0.5)
slope_threshold = st.sidebar.slider("Seuil de pente (°)", 15, 30, 20)

# Téléversement du fichier LAS/LAZ
uploaded_file = st.file_uploader("Téléverser un fichier LAS/LAZ", type=["las", "laz"])

if uploaded_file:
    with NamedTemporaryFile(suffix=".las", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        input_path = tmp.name

    # Pipeline PDAL avec SMRF
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_path
            },
            {
                "type": "filters.smrf",
                "cell": cell_size,
                "window": window_size,
                "threshold": elevation_threshold,
                "slope": slope_threshold
            },
            {
                "type": "writers.las",
                "filename": "filtered.las"
            }
        ]
    }

    try:
        pdal.Pipeline(json.dumps(pipeline)).execute()
        
        # Lecture des résultats
        las = laspy.read("filtered.las")
        z = las.z
        classification = las.classification

        # Filtrage des lignes électriques
        non_ground_mask = (classification != 2)  # Classe 2 = sol
        power_lines_mask = (z > 5) & (z < 30)    # Hauteur typique des lignes
        
        filtered_points = non_ground_mask & power_lines_mask

        # Création d'un DataFrame pour la visualisation
        df = pd.DataFrame({
            "x": las.x[filtered_points],
            "y": las.y[filtered_points],
            "z": las.z[filtered_points]
        })

        # Visualisation 3D avec PyDeck
        st.subheader("Visualisation des lignes détectées")
        view_state = pdk.ViewState(
            longitude=np.mean(df["x"]),
            latitude=np.mean(df["y"]),
            zoom=14,
            pitch=50,
            bearing=0
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["x", "y", "z"],
            get_color=[255, 0, 0, 160],
            get_radius=1,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Altitude: {z} m"}
        ))

        # Statistiques
        st.write(f"Points détectés comme lignes électriques : {len(df)}")
        st.write("Distribution des hauteurs :")
        st.bar_chart(df["z"].value_counts())

    except Exception as e:
        st.error(f"Erreur de traitement : {str(e)}")

st.markdown("""
**Instructions :**
1. Téléversez un fichier LAS/LAZ
2. Ajustez les paramètres SMRF dans la barre latérale
3. Visualisez les résultats en 3D
""")
