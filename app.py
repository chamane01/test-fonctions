import streamlit as st
from st_canvas import st_canvas
import json

# Titre de l'application
st.title("Application de Dessin Avancée")

# Configuration du canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage
    stroke_width=3,  # Épaisseur du trait
    stroke_color="#000000",  # Couleur du trait
    background_color="#ffffff",  # Couleur de fond
    height=600,  # Hauteur du canvas
    width=800,  # Largeur du canvas
    drawing_mode="freedraw",  # Mode de dessin (ici, dessin libre)
    key="canvas",
)

# Barre latérale pour les outils
with st.sidebar:
    st.header("Outils de Dessin")
    
    # Sélection de l'outil
    tool = st.selectbox(
        "Outil :",
        ["Point", "Ligne", "Polyligne", "Polygone", "Carré", "Rectangle"],
        key="tool_select",
    )
    
    # Choix de la couleur
    color = st.color_picker("Couleur :", "#000000", key="color_picker")
    
    # Choix de l'épaisseur/taille
    brush_size = st.slider("Épaisseur/Taille :", 1, 50, 5, key="brush_size")
    
    # Bouton pour effacer le canvas
    if st.button("Effacer le dessin", key="clear_canvas"):
        canvas_result.json_data = None
    
    # Téléversement d'image
    uploaded_image = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"], key="image_loader")
    
    # Bouton pour exporter les dessins en GeoJSON
    if st.button("Exporter en GeoJSON", key="export_geojson"):
        if canvas_result.json_data:
            st.download_button(
                label="Télécharger GeoJSON",
                data=json.dumps(canvas_result.json_data, indent=2),
                file_name="dessin.geojson",
                mime="application/json",
            )
    
    # Bouton pour exporter les dessins en JSON
    if st.button("Exporter en JSON", key="export_json"):
        if canvas_result.json_data:
            st.download_button(
                label="Télécharger JSON",
                data=json.dumps(canvas_result.json_data, indent=2),
                file_name="dessin.json",
                mime="application/json",
            )

# Affichage des mesures
st.header("Mesures")
if canvas_result.json_data:
    st.write(canvas_result.json_data)
else:
    st.write("Aucune mesure disponible.")

# Affichage de l'image téléversée
if uploaded_image is not None:
    st.image(uploaded_image, caption="Image téléversée", use_column_width=True)
