import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Application de Dessin", layout="wide")

st.title("Application de Dessin Basique")

# Paramètres du canevas
drawing_mode = st.sidebar.selectbox("Mode de dessin", ["freedraw", "line", "rect", "circle", "transform"])
stroke_width = st.sidebar.slider("Taille du pinceau", 1, 25, 5)
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")

# Création du canevas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage par défaut
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=background_color,
    height=400,
    width=600,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Sauvegarde du dessin si l'utilisateur le souhaite
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
