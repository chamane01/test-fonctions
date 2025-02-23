import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Wrapper permettant de renvoyer les attributs et méthodes du PIL Image
class AlwaysTrue:
    def __init__(self, pil_img):
        self.pil_img = pil_img
    def __bool__(self):
        return True
    def __getattr__(self, attr):
        return getattr(self.pil_img, attr)
    def __array__(self, *args, **kwargs):
        return np.array(self.pil_img, *args, **kwargs)

st.set_page_config(page_title="Dessin sur TIFF", layout="wide")
st.title("Application de Dessin sur TIFF")

# Téléversement du fichier TIFF
uploaded_file = st.file_uploader("Téléversez un fichier TIFF", type=["tiff", "tif"])
if uploaded_file:
    tiff_image = Image.open(uploaded_file)
    # Si le TIFF est multi-pages, on se place sur la première
    try:
        tiff_image.seek(0)
    except EOFError:
        pass

    # Affichage du TIFF avec la largeur du conteneur
    st.image(tiff_image, caption="TIFF téléversé", use_container_width=True)

    # Récupération des dimensions réelles du TIFF
    image_width, image_height = tiff_image.size

    # Définir des dimensions maximales pour le canevas (exemple : 800x600)
    max_canvas_width = 800
    max_canvas_height = 600

    # Calcul du ratio pour réduire l'image si elle est trop grande
    ratio = min(max_canvas_width / image_width, max_canvas_height / image_height, 1)
    canvas_width = int(image_width * ratio)
    canvas_height = int(image_height * ratio)

    # Création du wrapper autour du PIL Image pour le fond du canevas
    wrapped_tiff = AlwaysTrue(tiff_image)
else:
    wrapped_tiff = None
    canvas_width = 600
    canvas_height = 400

# Paramètres du dessin dans la barre latérale
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
if wrapped_tiff is None:
    background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")
else:
    background_color = None

# Création du canevas avec l'image TIFF en fond (si téléversée)
canvas_result = st_canvas(
    fill_color=stroke_color,             # Couleur de remplissage pour le dessin
    stroke_width=point_size,              # Taille du pinceau
    stroke_color=stroke_color,
    background_color=background_color,    # Utilisé uniquement si aucun TIFF n'est téléversé
    background_image=wrapped_tiff,        # TIFF téléversé
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",              # Mode dessin libre
    key="canvas",
)

# Sauvegarde du dessin réalisé sur le canevas
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    if st.sidebar.button("Sauvegarder le dessin"):
        img.save("dessin_tiff.png")
        st.sidebar.success("Dessin sauvegardé sous 'dessin_tiff.png'")
