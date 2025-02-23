import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Application de Marquage d'Images")

# 1. Téléversement des images
uploaded_files = st.file_uploader("Téléversez vos images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Charger les images
    images = [Image.open(file) for file in uploaded_files]
    
    # 2. Navigation entre les images : slider
    img_index = st.slider("Choisissez une image", 0, len(images)-1, 0)
    current_image = images[img_index]
    st.image(current_image, caption=f"Image {img_index+1}/{len(images)}", use_column_width=True)
    
    # 3. Sélection de la classe et de la gravité avant le placement du marqueur
    st.sidebar.header("Paramètres du marqueur")
    classes = [f"Classe {i}" for i in range(1, 14)]  # 13 classes
    selected_class = st.sidebar.selectbox("Sélectionnez la classe", classes)
    selected_severity = st.sidebar.selectbox("Sélectionnez la gravité", [1, 2, 3])
    
    st.write("Cliquez sur l'image pour placer des marqueurs.")
    
    # Création d'un canvas interactif sur l'image
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage semi-transparente pour les points
        stroke_width=5,
        stroke_color="red",
        background_image=current_image,
        update_streamlit=True,
        height=current_image.height,
        width=current_image.width,
        drawing_mode="point",  # Mode point pour placer des marqueurs
        point_display_radius=10,
        key="canvas",
    )
    
    # Traitement des données de dessin pour récupérer les marqueurs et leurs coordonnées
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            st.subheader("Marqueurs placés")
            for i, obj in enumerate(objects):
                # Les coordonnées locales du marqueur (issue d'une grille basée sur la taille de l'image)
                x = obj.get("left", 0)
                y = obj.get("top", 0)
                st.write(f"Marqueur {i+1} : Coordonnées ({x:.2f}, {y:.2f}) | {selected_class} - Gravité {selected_severity}")
