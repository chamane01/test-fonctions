import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Fonctions de conversion RGB <-> CMYK ---

def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100
    r_, g_, b_ = r/255.0, g/255.0, b/255.0
    k = 1 - max(r_, g_, b_)
    c = (1 - r_ - k) / (1 - k + 1e-8)
    m = (1 - g_ - k) / (1 - k + 1e-8)
    y = (1 - b_ - k) / (1 - k + 1e-8)
    return c * 100, m * 100, y * 100, k * 100

def cmyk_to_rgb(c, m, y, k):
    r = 1 - min(1, c/100.0 + k/100.0)
    g = 1 - min(1, m/100.0 + k/100.0)
    b = 1 - min(1, y/100.0 + k/100.0)
    return int(r * 255), int(g * 255), int(b * 255)

# --- Fonction de création du masque pour une couleur spécifique ---
def get_specific_color_mask(img_bgr, target_hex, tolerance):
    """
    Génère un masque pour les pixels dont la teinte est proche de la couleur cible.
    La comparaison se fait sur la composante H (teinte) en tenant compte du caractère circulaire.
    """
    # Conversion du code hex en tuple RGB, puis en BGR
    target_rgb = tuple(int(target_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Comparaison sur la composante H
    diff = cv2.absdiff(img_hsv[:, :, 0], np.uint8(target_hsv[0]))
    diff = np.minimum(diff, 180 - diff)
    mask = cv2.inRange(diff, 0, tolerance)
    return mask

# --- Fonction d'application de la correction sélective ---
def apply_selective_color(img_bgr, mask, c_adj, m_adj, y_adj, k_adj, method="Relative"):
    """
    Applique une correction de couleur sur l'image (conversion RGB -> CMJN, ajustement, reconversion).
    Seules les zones définies par le masque sont modifiées.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_img = img_rgb.copy()
    h, w = out_img.shape[:2]
    for row in range(h):
        for col in range(w):
            if mask[row, col] != 0:
                r, g, b = out_img[row, col]
                c, m, y, k = rgb_to_cmyk(r, g, b)
                if method == "Relative":
                    c += (c_adj / 100.0) * c
                    m += (m_adj / 100.0) * m
                    y += (y_adj / 100.0) * y
                    k += (k_adj / 100.0) * k
                else:  # Méthode Absolute
                    c += c_adj
                    m += m_adj
                    y += y_adj
                    k += k_adj
                c = max(0, min(100, c))
                m = max(0, min(100, m))
                y = max(0, min(100, y))
                k = max(0, min(100, k))
                out_img[row, col] = cmyk_to_rgb(c, m, y, k)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

# --- Interface Streamlit pour l'outil Pupette ---
st.title("Outil Pupette – Sélection de Couleur Personnalisée")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléversez une image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    st.image(pil_img, caption="Image originale", use_column_width=True)

    st.sidebar.subheader("Outil Pupette")
    # Choix de la méthode de sélection
    selection_method = st.sidebar.radio("Méthode de sélection", ["Color Picker", "Pipette"], key="pupette_method")

    if selection_method == "Color Picker":
        selected_color = st.sidebar.color_picker("Sélectionnez la couleur", "#FF0000", key="pupette_color")
    else:
        st.markdown("**Cliquez sur l'image pour sélectionner une couleur**")
        # Conversion de l'image PIL en bytes pour streamlit_image_coordinates
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        coords = streamlit_image_coordinates(
            "Cliquez sur l'image pour sélectionner un pixel",
            img_bytes,
            key="pupette_pipette"
        )
        if coords is not None:
            x = int(coords["x"])
            y = int(coords["y"])
            if 0 <= x < original_bgr.shape[1] and 0 <= y < original_bgr.shape[0]:
                pixel = original_bgr[y, x]  # Rappel : l'indexation se fait en [y, x]
                pixel_rgb = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2RGB)[0, 0]
                selected_color = '#%02x%02x%02x' % tuple(pixel_rgb)
                st.write("Couleur sélectionnée via pipette :", selected_color)
        else:
            selected_color = "#FF0000"  # Valeur par défaut

    # Paramètres de l'outil Pupette
    tolerance = st.sidebar.slider("Tolérance", 0, 100, 30, key="pupette_tolerance")
    c_adj = st.sidebar.slider("Ajustement Cyan", -100, 100, 0, key="pupette_c")
    m_adj = st.sidebar.slider("Ajustement Magenta", -100, 100, 0, key="pupette_m")
    y_adj = st.sidebar.slider("Ajustement Jaune", -100, 100, 0, key="pupette_y")
    k_adj = st.sidebar.slider("Ajustement Noir", -100, 100, 0, key="pupette_k")
    method_adj = st.sidebar.radio("Méthode d'ajustement", ["Relative", "Absolute"], key="pupette_method_adj")

    st.write("**Couleur Pupette sélectionnée :**", selected_color)

    # Génération du masque et application de la correction sur l'image
    mask = get_specific_color_mask(original_bgr, selected_color, tolerance)
    result = apply_selective_color(original_bgr, mask, c_adj, m_adj, y_adj, k_adj, method_adj)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
             caption="Image après application de l'outil Pupette",
             use_column_width=True)
else:
    st.write("Veuillez téléverser une image pour commencer.")
