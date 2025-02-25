import streamlit as st
import pdal
import json
import tempfile
import os

st.title("Classification de Nuage de Points Lidar")

# Téléchargement du fichier LAS/LAZ
uploaded_file = st.file_uploader("Téléchargez votre fichier LAS/LAZ", type=["las", "laz"])
if uploaded_file is not None:
    # Sauvegarde temporaire du fichier téléchargé
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp:
        tmp.write(uploaded_file.read())
        input_filename = tmp.name

    st.write("Fichier téléchargé:", input_filename)

    # Définition de la pipeline PDAL avec le filtre SMRF
    pipeline_definition = {
        "pipeline": [
            input_filename,
            {
                "type": "filters.smrf",
                "scalar": 1.25,      # Facteur d'échelle pour le filtrage
                "slope": 0.15,       # Pente maximale pour considérer un point comme sol
                "threshold": 0.5,    # Seuil de hauteur pour la classification
                "window": 16.0       # Taille de la fenêtre en mètres
            },
            {
                "type": "writers.las",
                "filename": "classified_output.las"
            }
        ]
    }

    # Exécution de la pipeline PDAL
    pipeline = pdal.Pipeline(json.dumps(pipeline_definition))
    try:
        count = pipeline.execute()
        st.write(f"Nombre de points traités : {count}")
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la pipeline PDAL : {e}")

    # Mise à disposition du fichier classifié en téléchargement
    output_file = "classified_output.las"
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            st.download_button("Télécharger le fichier classifié", f, file_name="classified_output.las")
