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

st.title("Application de Marquage d'Images")

# 1. Téléversement des images
uploaded_files = st.file_uploader("Téléversez vos images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Chargement des images en utilisant PIL
    images = [Image.open(file) for file in uploaded_files]
    
    # Navigation entre les images avec un slider
    img_index = st.slider("Choisissez une image", 0, len(images) - 1, 0)
    current_image = images[img_index]
    st.image(current_image, caption=f"Image {img_index + 1}/{len(images)}", use_column_width=True)
    
    # 2. Sélection de la classe et de la gravité dans la barre latérale
    st.sidebar.header("Paramètres du marqueur")
    classes = [f"Classe {i}" for i in range(1, 14)]  # 13 classes
    selected_class = st.sidebar.selectbox("Sélectionnez la classe", classes)
    selected_severity = st.sidebar.selectbox("Sélectionnez la gravité", [1, 2, 3])
    
    st.write("Cliquez sur l'image pour placer des marqueurs.")
    
    # Conversion de l'image en tableau NumPy
    image_np = np.array(current_image)
    # On encapsule le tableau dans le wrapper pour éviter l'erreur du test booléen
    wrapped_image = AlwaysTrue(image_np)
    
    # 3. Création d'un canvas interactif avec l'image en fond
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage semi-transparente
        stroke_width=5,
        stroke_color="red",
        background_image=wrapped_image,
        update_streamlit=True,
        height=current_image.height,
        width=current_image.width,
        drawing_mode="point",  # Mode point pour placer des marqueurs
        point_display_radius=10,
        key="canvas",
    )
    
    # Récupération et affichage des marqueurs avec leurs coordonnées
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            st.subheader("Marqueurs placés")
            for i, obj in enumerate(objects):
                # Coordonnées locales du marqueur
                x = obj.get("left", 0)
                y = obj.get("top", 0)
                st.write(f"Marqueur {i+1} : Coordonnées ({x:.2f}, {y:.2f}) | {selected_class} - Gravité {selected_severity}")
