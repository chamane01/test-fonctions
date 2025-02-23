import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Wrapper modifié pour renvoyer les attributs "height" et "width" à partir du tableau NumPy
class AlwaysTrue:
    def __init__(self, arr):
        self.arr = arr
    def __bool__(self):
        return True
    def __getattr__(self, attr):
        if attr == "height":
            return self.arr.shape[0]
        if attr == "width":
            return self.arr.shape[1]
        return getattr(self.arr, attr)
    def __array__(self, *args, **kwargs):
        return np.array(self.arr, *args, **kwargs)

st.set_page_config(page_title="Dessin sur TIFF", layout="wide")
st.title("Application de Dessin sur TIFF")

# Téléversement du fichier TIFF
uploaded_file = st.file_uploader("Téléversez un fichier TIFF", type=["tiff", "tif"])
if uploaded_file:
    tiff_image = Image.open(uploaded_file)
    # Pour gérer les TIFF multi-pages, on se positionne sur la première page
    try:
        tiff_image.seek(0)
    except EOFError:
        pass
    st.image(tiff_image, caption="TIFF téléversé", use_container_width=True)
    
    # Récupération des dimensions réelles du TIFF
    image_width, image_height = tiff_image.size
    # Conversion en RGB pour assurer une bonne compatibilité
    image_np = np.array(tiff_image.convert("RGB"))
    wrapped_tiff = AlwaysTrue(image_np)
else:
    wrapped_tiff = None

# Paramètres du dessin dans la barre latérale
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
if wrapped_tiff is None:
    background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")
else:
    background_color = None

# Ajustement des dimensions du canevas en fonction du TIFF téléversé
if wrapped_tiff is not None:
    canvas_height = image_height
    canvas_width = image_width
else:
    canvas_height = 400
    canvas_width = 600

# Création du canevas avec l'image TIFF en fond (si disponible)
canvas_result = st_canvas(
    fill_color=stroke_color,             # Couleur de remplissage pour le dessin
    stroke_width=point_size,              # Taille du pinceau
    stroke_color=stroke_color,
    background_color=background_color,    # Utilisé uniquement si aucun TIFF n'est téléversé
    background_image=wrapped_tiff,        # Fichier TIFF téléversé
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
