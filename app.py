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

# Dimensions du canevas
canvas_width, canvas_height = 600, 400

# Téléversement d'une image comme fond
uploaded_file = st.sidebar.file_uploader("Téléverser une image de fond", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGBA")
    image = image.resize((canvas_width, canvas_height))
    background_image = np.array(image)
else:
    # Image blanche par défaut pour éviter le problème de "None"
    background_image = np.ones((canvas_height, canvas_width, 4), dtype=np.uint8) * 255

# Création du canevas
canvas_result = st_canvas(
    stroke_width=dot_size,
    stroke_color=stroke_color,
    background_image=background_image,  # Image toujours définie
    height=canvas_height,
    width=canvas_width,
    drawing_mode="point",
    key="canvas",
)

# Sauvegarde du dessin si l'utilisateur le souhaite
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
