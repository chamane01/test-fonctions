import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Wrapper modifié pour renvoyer les attributs height et width depuis le tableau NumPy
class AlwaysTrue:
    def __init__(self, arr):
        self.arr = arr
    def __bool__(self):
        return True
    def __getattr__(self, attr):
        # Renvoie la hauteur ou la largeur à partir de la forme du tableau
        if attr == "height":
            return self.arr.shape[0]
        if attr == "width":
            return self.arr.shape[1]
        return getattr(self.arr, attr)
    def __array__(self, *args, **kwargs):
        return np.array(self.arr, *args, **kwargs)

st.set_page_config(page_title="Application de Dessin", layout="wide")
st.title("Application de Dessin Basique avec Image de Fond")

# Téléversement de l'image de fond
uploaded_file = st.file_uploader("Téléversez une image pour le fond", type=["png", "jpg", "jpeg"])
if uploaded_file:
    bg_image = Image.open(uploaded_file)
    st.image(bg_image, caption="Image de fond", use_container_width=True)
    
    # Récupération des dimensions réelles de l'image
    image_width, image_height = bg_image.size
    image_np = np.array(bg_image)
    wrapped_bg_image = AlwaysTrue(image_np)
else:
    wrapped_bg_image = None

# Paramètres du dessin
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
if wrapped_bg_image is None:
    background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")
else:
    background_color = None

# Si l'image est chargée, ajuster les dimensions du canevas en fonction de l'image
if wrapped_bg_image is not None:
    canvas_height = image_height  # Hauteur de l'image de fond
    canvas_width = image_width    # Largeur de l'image de fond
else:
    canvas_height = 400
    canvas_width = 600

# Création du canevas
canvas_result = st_canvas(
    fill_color=stroke_color,             # Couleur de remplissage du dessin
    stroke_width=point_size,              # Taille du pinceau
    stroke_color=stroke_color,
    background_color=background_color,    # Utilisé uniquement si aucune image n'est chargée
    background_image=wrapped_bg_image,    # Image de fond uploadée
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",              # Mode dessin libre
    key="canvas",
)

# Sauvegarde du dessin
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin.png'")
