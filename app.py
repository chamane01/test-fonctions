import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

st.set_page_config(page_title="Application de Dessin", layout="wide")
st.title("Application de Dessin Basique avec Image de Fond")

# Téléversement de l'image de fond
uploaded_file = st.file_uploader("Téléversez une image pour le fond", type=["png", "jpg", "jpeg"])
bg_image = None
if uploaded_file:
    bg_image = Image.open(uploaded_file)
    st.image(bg_image, caption="Image de fond", use_container_width=True)
    canvas_width, canvas_height = bg_image.size
else:
    canvas_height = 400
    canvas_width = 600

# Paramètres du dessin
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF") if not bg_image else None

# Création du canevas
canvas_result = st_canvas(
    fill_color=stroke_color,
    stroke_width=point_size,
    stroke_color=stroke_color,
    background_color=background_color,
    background_image=bg_image,  # Utilisation directe de l'image PIL
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",
    key="canvas",
)

# Sauvegarde du dessin
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
