import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.title("Correction d'image interactive")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléversez une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    # Si l'image possède un canal alpha, on le convertit en RGB
    if image_np.ndim == 3 and image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    st.subheader("Image originale")
    st.image(image_np, channels="RGB", use_column_width=True)
    
    st.sidebar.title("Paramètres globaux")
    red_factor = st.sidebar.slider("Intensité Rouge", 0.0, 2.0, 1.0, step=0.01)
    green_factor = st.sidebar.slider("Intensité Vert", 0.0, 2.0, 1.0, step=0.01)
    blue_factor = st.sidebar.slider("Intensité Bleu", 0.0, 2.0, 1.0, step=0.01)
    
    brightness = st.sidebar.slider("Luminosité", 0.5, 2.0, 1.0, step=0.01)
    contrast = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0, step=0.01)
    
    black_adj = st.sidebar.slider("Ajustement Noir", -100, 100, 0, step=1)
    white_adj = st.sidebar.slider("Ajustement Blanc", -100, 100, 0, step=1)
    
    st.sidebar.title("Correction sélective")
    selected_color = st.sidebar.color_picker("Couleur sélective", "#ff0000")
    selective_factor = st.sidebar.slider("Facteur de correction", 0.0, 2.0, 1.0, step=0.01)
    threshold = st.sidebar.slider("Seuil de proximité", 0, 255, 50, step=1)
    
    # Traitement de l'image
    img_mod = image_np.astype(np.float32)
    
    # Ajustement des canaux de couleur
    img_mod[..., 0] *= red_factor   # canal rouge
    img_mod[..., 1] *= green_factor # canal vert
    img_mod[..., 2] *= blue_factor  # canal bleu
    
    # Application du contraste (en recentrant autour de 127.5) et de la luminosité
    img_mod = (img_mod - 127.5) * contrast + 127.5
    img_mod *= brightness
    
    # Ajustement global pour les zones sombres et claires
    img_mod += (black_adj + white_adj)
    
    # Correction sélective sur la couleur choisie
    # Conversion de la couleur hexadécimale en valeurs RGB
    sel_hex = selected_color.lstrip('#')
    sel_rgb = np.array([int(sel_hex[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
    # Calcul de la distance euclidienne entre chaque pixel et la couleur sélectionnée
    diff = np.linalg.norm(img_mod - sel_rgb, axis=-1)
    mask = diff < threshold
    # Pour les pixels proches de la couleur cible, on applique le facteur de correction
    img_mod[mask] *= selective_factor
    
    # Contrainte des valeurs entre 0 et 255
    img_mod = np.clip(img_mod, 0, 255).astype(np.uint8)
    
    st.subheader("Image modifiée")
    st.image(img_mod, channels="RGB", use_column_width=True)
    
else:
    st.write("Veuillez téléverser une image pour commencer.")
