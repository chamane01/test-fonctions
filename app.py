import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(image):
    # 1. Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Amélioration du contraste avec égalisation d'histogramme
    equalized = cv2.equalizeHist(gray)
    
    # 3. Lissage pour réduire le bruit
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # 4. Extraction des gradients pour simuler une carte de profondeur
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 5. Seuillage adaptatif pour isoler les dégradations
    thresh = cv2.adaptiveThreshold(gradient_norm, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 6. Opérations morphologiques pour éliminer les petits bruits
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned

st.title("Détection Automatique de Dégradations")
st.write("Téléversez une ou plusieurs images pour détecter automatiquement les dégradations.")

uploaded_files = st.file_uploader("Choisir une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Lecture de l'image via OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            processed = process_image(image)
            # Conversion de l'image traitée pour affichage
            processed_pil = Image.fromarray(processed)
            
            st.subheader(f"Résultat pour {uploaded_file.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), caption="Image Originale", use_column_width=True)
            with col2:
                st.image(processed_pil, caption="Dégradations détectées", use_column_width=True)
        else:
            st.error(f"Erreur lors de la lecture de {uploaded_file.name}")
