import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(img, illum_kernel, clahe_clip, threshold_val, median_ksize, max_area, closing_kernel_size):
    # Étape 1 : Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Étape 2 : Correction d'illumination
    # On utilise une ouverture morphologique pour estimer le fond lumineux
    kernel_illum = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (illum_kernel, illum_kernel))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_illum)
    illum_corrected = cv2.subtract(gray, background)
    
    # Optionnel : Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
    enhanced = clahe.apply(illum_corrected)
    
    # Optionnel : Flou gaussien pour réduire le bruit résiduel
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Étape 3 : Segmentation par seuillage (thresholding)
    ret, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Étape 4 : Filtrage médian pour éliminer le bruit (par exemple, filtre 3x3)
    median = cv2.medianBlur(thresh, median_ksize)
    
    # Étape 5 : Suppression des grandes régions (par exemple, celles dont l'aire > max_area)
    contours, _ = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(median)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # On conserve uniquement les régions dont l'aire est inférieure ou égale au seuil
        if area <= max_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # Étape 6 : Fermeture morphologique pour améliorer la cohérence des fissures
    # Création d'un noyau en forme de diamant
    size = closing_kernel_size
    diamond = np.zeros((2*size+1, 2*size+1), dtype=np.uint8)
    for i in range(2*size+1):
        for j in range(2*size+1):
            if abs(i - size) + abs(j - size) <= size:
                diamond[i, j] = 1
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, diamond)
    
    return closed

def main():
    st.title("Détection Automatique de Dégradations sur Voirie")
    st.write("Cette application applique une série de traitements (correction d'illumination, segmentation par seuillage, post-traitement) "
             "pour détecter et améliorer les dégradations sur les images, conformément à la Fig. 7 de votre PDF.")
    
    # Paramètres de contrôle dans la barre latérale
    st.sidebar.header("Paramètres de traitement")
    illum_kernel = st.sidebar.slider("Taille du noyau pour correction d'illumination", 3, 31, 15, step=2)
    clahe_clip = st.sidebar.slider("Clip Limit CLAHE", 1.0, 10.0, 2.0, step=0.5)
    threshold_val = st.sidebar.slider("Seuil de segmentation", 0, 255, 100)
    median_ksize = st.sidebar.slider("Taille du filtre médian (impair)", 3, 11, 3, step=2)
    max_area = st.sidebar.slider("Aire maximale pour conserver une région", 100, 2000, 500)
    closing_kernel_size = st.sidebar.slider("Taille du noyau de fermeture (diamant)", 1, 10, 5)
    
    # Téléversement des images
    uploaded_files = st.file_uploader("Téléversez une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Application du traitement complet
            processed = process_image(image, illum_kernel, clahe_clip, threshold_val, median_ksize, max_area, closing_kernel_size)
            
            # Affichage de l'image originale et de l'image traitée
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale", use_column_width=True)
            st.image(processed, caption="Image traitée (dégradations détectées)", use_column_width=True)

if __name__ == '__main__':
    main()
