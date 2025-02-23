import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Application de Dessin", layout="wide")
st.title("Application de Dessin Basique")

# Paramètres du pinceau pointillé
dot_size = st.sidebar.radio("Taille du point", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")

# Téléversement d'une image comme fond
uploaded_file = st.sidebar.file_uploader("Téléverser une image de fond", type=["png", "jpg", "jpeg"])
background_image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGBA")
    image = image.resize((600, 400))  # Adapter la taille au canevas
    background_image = image  # Passage direct de l'objet PIL

# Création du canevas
canvas_result = st_canvas(
    stroke_width=dot_size,
    stroke_color=stroke_color,
    background_image=background_image,  # Utilisation de l'objet PIL directement
    height=400,
    width=600,
    drawing_mode="point",
    key="canvas",
)

# Sauvegarde du dessin si l'utilisateur le souhaite
if canvas_result.image_data is not None:
    # Les valeurs de canvas_result.image_data sont entre 0 et 1, on les met à l'échelle sur 0-255.
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
