import streamlit as st
import cv2
import numpy as np
from PIL import Image

def adjust_gamma(image, gamma=1.0):
    # Correction gamma pour ajuster l'illumination
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_image(img, canny_thresh1, canny_thresh2, canny_thresh3, canny_thresh4,
                  clahe_clip, clahe_tile, illumination_gamma, morph_kernel_size, white_filter_kernel_size):
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Correction d'illumination via gamma correction
    if illumination_gamma != 1.0:
        gray = adjust_gamma(gray, illumination_gamma)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(gray)
    
    # Flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Détection des contours avec Canny à partir de deux jeux de seuils
    edges1 = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    edges2 = cv2.Canny(blurred, canny_thresh3, canny_thresh4)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Opération morphologique pour fermer les petites interruptions et réduire le bruit
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Filtre supplémentaire pour regrouper les zones blanches et réduire le bruit
    if white_filter_kernel_size > 1:
        kernel2 = np.ones((white_filter_kernel_size, white_filter_kernel_size), np.uint8)
        # Dilation puis érosion pour consolider les zones blanches
        grouped = cv2.dilate(closed, kernel2, iterations=1)
        filtered = cv2.erode(grouped, kernel2, iterations=1)
    else:
        filtered = closed
    
    return filtered

def main():
    st.title("Détection Automatique de Dégradations sur Images de Voirie")
    st.write("Téléversez une ou plusieurs images pour obtenir une image traitée "
             "où les dégradations (ex. fissures) sont affichées en blanc sur fond noir.")
    
    # Paramètres de contrôle dans la barre latérale
    st.sidebar.header("Paramètres de traitement")
    canny_thresh1 = st.sidebar.slider("Seuil 1 Canny", 0, 255, 50)
    canny_thresh2 = st.sidebar.slider("Seuil 2 Canny", 0, 255, 150)
    canny_thresh3 = st.sidebar.slider("Seuil 3 Canny", 0, 255, 50)
    canny_thresh4 = st.sidebar.slider("Seuil 4 Canny", 0, 255, 150)
    clahe_clip = st.sidebar.slider("Clip Limit CLAHE", 1.0, 10.0, 2.0, step=0.5)
    clahe_tile = st.sidebar.slider("Tile Grid Size CLAHE", 2, 20, 8)
    illumination_gamma = st.sidebar.slider("Correction d'illumination (gamma)", 0.1, 3.0, 1.0, step=0.1)
    morph_kernel_size = st.sidebar.slider("Taille du noyau morphologique", 1, 10, 3)
    white_filter_kernel_size = st.sidebar.slider("Taille du filtre de regroupement", 1, 10, 1)
    
    # Téléversement des images
    uploaded_files = st.file_uploader("Téléversez une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            # Lecture de l'image avec OpenCV
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Traitement de l'image
            processed = process_image(image, canny_thresh1, canny_thresh2, canny_thresh3, canny_thresh4,
                                      clahe_clip, clahe_tile, illumination_gamma, morph_kernel_size, white_filter_kernel_size)
            
            # Affichage
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale", use_column_width=True)
            st.image(processed, caption="Image traitée (dégradations détectées)", use_column_width=True)

if __name__ == '__main__':
    main()
