import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Wrapper pour forcer l'évaluation booléenne à True
class AlwaysTrue:
    def __init__(self, arr):
        self.arr = arr
    def __bool__(self):
        return True
    def __getattr__(self, attr):
        return getattr(self.arr, attr)
    def __array__(self, *args, **kwargs):
        return np.array(self.arr, *args, **kwargs)

# Configuration de la page
st.set_page_config(page_title="Application de Dessin", layout="wide")
st.title("Application de Dessin Basique avec Image de Fond")

# 1. Téléversement de l'image de fond
uploaded_file = st.file_uploader("Téléversez une image pour le fond", type=["png", "jpg", "jpeg"])
if uploaded_file:
    bg_image = Image.open(uploaded_file)
    st.image(bg_image, caption="Image de fond", use_column_width=True)
    # Conversion de l'image en tableau NumPy et encapsulation dans AlwaysTrue
    image_np = np.array(bg_image)
    wrapped_bg_image = AlwaysTrue(image_np)
else:
    wrapped_bg_image = None

# 2. Paramètres du dessin dans la barre latérale
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
# Si aucune image n'est chargée, possibilité de choisir une couleur de fond
if wrapped_bg_image is None:
    background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")
else:
    background_color = None

# Définition des dimensions du canevas
if wrapped_bg_image is not None:
    canvas_height = bg_image.height
    canvas_width = bg_image.width
else:
    canvas_height = 400
    canvas_width = 600

# 3. Création du canevas avec l'image de fond (si disponible)
canvas_result = st_canvas(
    fill_color=stroke_color,             # Couleur de remplissage pour le dessin
    stroke_width=point_size,              # Taille du pinceau
    stroke_color=stroke_color,
    background_color=background_color,    # Utilisé uniquement si aucune image n'est chargée
    background_image=wrapped_bg_image,    # Image de fond chargée par l'utilisateur
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",              # Mode dessin libre
    key="canvas",
)

# 4. Sauvegarde du dessin
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
