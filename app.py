import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(image):
    # 1. Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Correction de l'éclairage non uniforme
    # Utilisation d'une opération morphologique (fermeture) pour estimer le fond
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg)
    # Soustraction du fond pour corriger les variations d'illumination
    corrected = cv2.absdiff(gray, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Réduction du bruit par flou gaussien
    blurred = cv2.GaussianBlur(corrected, (5, 5), 0)
    
    # 4. Détection des contours avec l'algorithme Canny
    # Les valeurs des seuils (50 et 150) peuvent être ajustées selon le type d'image
    edges = cv2.Canny(blurred, 50, 150)
    
    # 5. Opération morphologique pour "fermer" les contours détectés et éliminer les petits bruits
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 6. Filtrage médian pour réduire encore le bruit résiduel
    final = cv2.medianBlur(closed, 3)
    
    return final

st.title("Détection Automatique de Dégradations (Noir & Blanc)")
st.write("Ce système, inspiré de la méthodologie décrite dans le document de Barrile et al. (2020) :contentReference[oaicite:1]{index=1}, permet d'extraire les dégradations sous forme de traces blanches sur fond noir.")

uploaded_files = st.file_uploader("Choisir une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Lecture de l'image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            processed = process_image(image)
            processed_pil = Image.fromarray(processed)
            
            st.subheader(f"Résultat pour {uploaded_file.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), caption="Image Originale", use_column_width=True)
            with col2:
                st.image(processed_pil, caption="Dégradations en Blanc sur Fond Noir", use_column_width=True)
        else:
            st.error(f"Erreur lors de la lecture de {uploaded_file.name}")
