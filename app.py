import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Application de Dessin", layout="wide")

st.title("Application de Dessin Basique")

# Sélection de la taille des points
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")

# Création du canevas
canvas_result = st_canvas(
    fill_color=stroke_color,  # La couleur du point
    stroke_width=point_size,  # Taille du point
    stroke_color=stroke_color,
    background_color=background_color,
    height=400,
    width=600,
    drawing_mode="freedraw",  # Mode dessin libre pour placer les points
    key="canvas",
)

# Sauvegarde du dessin si l'utilisateur le souhaite
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
