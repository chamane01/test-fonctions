import streamlit as st
import cv2
import numpy as np
from PIL import Image

def remove_small_components(binary_image, min_size):
    """
    Supprime les petites composantes blanches de taille < min_size.
    binary_image : image binaire (0 ou 255)
    min_size : aire minimale pour conserver la composante
    """
    # Labeling des composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # stats[i, cv2.CC_STAT_AREA] = aire de la composante i
    # On crée un masque de sortie initialement vide
    output = np.zeros(binary_image.shape, dtype=np.uint8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            # On garde les composantes suffisamment grandes
            output[labels == i] = 255
    
    return output

def process_image(img,
                  canny_thresh1, canny_thresh2,
                  clahe_clip, clahe_tile,
                  morph_kernel_size,
                  open_kernel_size,
                  noise_filter_enabled, noise_filter_size,
                  remove_small_comp_enabled, min_component_size):
    """
    Chaîne de traitement de l'image pour détecter et affiner l'apparence des fissures.
    """

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(gray)

    # Réduction du bruit par flou gaussien
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Filtre de bruit supplémentaire (filtre médian) si activé
    if noise_filter_enabled and noise_filter_size is not None:
        blurred = cv2.medianBlur(blurred, noise_filter_size)

    # Détection des contours (Canny)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # Fermeture morphologique pour combler les petites interruptions
    kernel_close = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    # Ouverture morphologique pour enlever de petits bruits isolés
    if open_kernel_size > 0:
        kernel_open = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    else:
        opened = closed

    # Option pour supprimer les très petites composantes blanches
    if remove_small_comp_enabled and min_component_size > 0:
        final = remove_small_components(opened, min_component_size)
    else:
        final = opened

    return final

def main():
    st.title("Détection Automatique de Dégradations sur Images de Voirie")
    st.write("Téléversez une ou plusieurs images pour obtenir une image traitée "
             "où les dégradations (ex. fissures) sont affichées en blanc sur fond noir.")

    # Paramètres de traitement dans la barre latérale
    st.sidebar.header("Paramètres de traitement")

    # Paramètres Canny conseillés : (50, 150) ou (100, 200) selon le contraste
    canny_thresh1 = st.sidebar.slider("Seuil 1 Canny", 0, 255, 50)
    canny_thresh2 = st.sidebar.slider("Seuil 2 Canny", 0, 255, 150)

    # Paramètres CLAHE conseillés : clip limit ~1.0, tile grid ~6
    clahe_clip = st.sidebar.slider("Clip Limit CLAHE", 0.0, 10.0, 1.0, step=0.5)
    clahe_tile = st.sidebar.slider("Tile Grid Size CLAHE", 2, 20, 6)

    # Paramètre pour la fermeture morphologique (pour relier les fissures)
    morph_kernel_size = st.sidebar.slider("Taille du noyau (close)", 1, 15, 5)

    # Paramètre pour l'ouverture morphologique (pour enlever les bruits)
    open_kernel_size = st.sidebar.slider("Taille du noyau (open)", 0, 15, 3)

    # Filtre médian supplémentaire
    noise_filter_enabled = st.sidebar.checkbox("Activer filtre de bruit (médian)", value=False)
    if noise_filter_enabled:
        noise_filter_size = st.sidebar.slider("Taille du filtre médian (impair)", 3, 11, 3, step=2)
    else:
        noise_filter_size = None

    # Suppression des petites composantes
    remove_small_comp_enabled = st.sidebar.checkbox("Supprimer petites composantes", value=False)
    if remove_small_comp_enabled:
        min_component_size = st.sidebar.slider("Taille min composante", 1, 500, 50)
    else:
        min_component_size = 0

    # Téléversement des images
    uploaded_files = st.file_uploader("Téléversez une ou plusieurs images", 
                                      type=["png", "jpg", "jpeg"], 
                                      accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Traitement de l'image
            processed = process_image(
                image,
                canny_thresh1, canny_thresh2,
                clahe_clip, clahe_tile,
                morph_kernel_size,
                open_kernel_size,
                noise_filter_enabled, noise_filter_size,
                remove_small_comp_enabled, min_component_size
            )

            # Affichage
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                     caption="Image originale", use_column_width=True)
            st.image(processed,
                     caption="Image traitée (dégradations détectées)",
                     use_column_width=True)

if __name__ == '__main__':
    main()
