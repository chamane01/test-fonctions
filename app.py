import streamlit as st 
import cv2
import numpy as np
from PIL import Image

def process_image(img, canny_thresh1, canny_thresh2, clahe_clip, clahe_tile, morph_kernel_size,
                  extra_canny_enabled, canny_thresh3, canny_thresh4, white_filter_enabled, white_filter_size):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(gray)
    
    # Optionnel : flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Détection des contours avec Canny (seuils 1 et 2)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    
    # Si activé, détection avec seuils supplémentaires
    if extra_canny_enabled:
        edges2 = cv2.Canny(blurred, canny_thresh3, canny_thresh4)
        # Combinaison des deux détections par opération OR bit à bit
        edges = cv2.bitwise_or(edges, edges2)
    
    # Opération morphologique pour fermer les petites interruptions et réduire le bruit
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Si activé, appliquer un filtre supplémentaire pour regrouper les zones blanches
    if white_filter_enabled and white_filter_size is not None:
        kernel_white = np.ones((white_filter_size, white_filter_size), np.uint8)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_white)
    
    return closed

def main():
    st.title("Détection Automatique de Dégradations sur Images de Voirie")
    st.write("Téléversez une ou plusieurs images pour obtenir une image traitée "
             "où les dégradations (ex. fissures) sont affichées en blanc sur fond noir.")
    
    # Paramètres de contrôle dans la barre latérale
    st.sidebar.header("Paramètres de traitement")
    canny_thresh1 = st.sidebar.slider("Seuil 1 Canny", 0, 255, 50)
    canny_thresh2 = st.sidebar.slider("Seuil 2 Canny", 0, 255, 150)
    clahe_clip = st.sidebar.slider("Clip Limit CLAHE", 1.0, 10.0, 2.0, step=0.5)
    clahe_tile = st.sidebar.slider("Tile Grid Size CLAHE", 2, 20, 8)
    morph_kernel_size = st.sidebar.slider("Taille du noyau morphologique", 1, 10, 3)
    
    # Paramètres supplémentaires pour Canny
    extra_canny_enabled = st.sidebar.checkbox("Utiliser seuils Canny supplémentaires", value=False)
    if extra_canny_enabled:
        canny_thresh3 = st.sidebar.slider("Seuil 3 Canny", 0, 255, 30)
        canny_thresh4 = st.sidebar.slider("Seuil 4 Canny", 0, 255, 100)
    else:
        canny_thresh3 = 0
        canny_thresh4 = 0
    
    # Paramètre supplémentaire pour le filtre blanc
    white_filter_enabled = st.sidebar.checkbox("Activer filtre de regroupement blanc", value=False)
    if white_filter_enabled:
        white_filter_size = st.sidebar.slider("Taille du filtre blanc", 1, 10, 3)
    else:
        white_filter_size = None
    
    # Téléversement des images
    uploaded_files = st.file_uploader("Téléversez une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            # Lecture de l'image avec OpenCV
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Traitement de l'image
            processed = process_image(image, canny_thresh1, canny_thresh2, clahe_clip, clahe_tile, morph_kernel_size,
                                      extra_canny_enabled, canny_thresh3, canny_thresh4, white_filter_enabled, white_filter_size)
            
            # Affichage
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale", use_column_width=True)
            st.image(processed, caption="Image traitée (dégradations détectées)", use_column_width=True)

if __name__ == '__main__':
    main()
