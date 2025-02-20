import cv2
import numpy as np
import streamlit as st

def get_specific_color_mask(img_bgr, target_color_rgb, tolerance):
    # Convertir la couleur cible de RGB à BGR puis HSV
    target_color_bgr = np.uint8([[target_color_rgb[::-1]]])  # RGB->BGR
    target_hsv = cv2.cvtColor(target_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    # Convertir l'image en HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Définir les bornes de tolérance
    lower = np.array([
        max(0, target_hsv[0] - tolerance),
        max(0, target_hsv[1] - tolerance),
        max(0, target_hsv[2] - tolerance)
    ])
    upper = np.array([
        min(179, target_hsv[0] + tolerance),
        min(255, target_hsv[1] + tolerance),
        min(255, target_hsv[2] + tolerance)
    ])
    
    # Créer le masque
    return cv2.inRange(img_hsv, lower, upper)

# Interface Streamlit
st.title("Sélecteur de Couleur Intelligent")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Widgets de sélection
    selected_color = st.color_picker("Choisir une couleur", '#FF0000')
    tolerance = st.slider("Tolérance", 0, 100, 20)
    
    # Conversion de la couleur sélectionnée
    rgb_tuple = tuple(int(selected_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Génération du masque
    mask = get_specific_color_mask(original_bgr, rgb_tuple, tolerance)
    result = cv2.bitwise_and(original_bgr, original_bgr, mask=mask)
    
    # Affichage
    st.image([original_rgb, mask, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)], 
             caption=["Original", "Masque", "Résultat"], width=300)
