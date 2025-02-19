import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ----------------------------------------------------------------------------
# 1) Définir les gammes de couleurs en HSV (approximation) pour détecter
# ----------------------------------------------------------------------------
color_ranges = {
    "Rouges":   [((0,   50,  50), (10,  255, 255)),  # Rouge clair
                 ((170, 50,  50), (180, 255, 255))], # Rouge foncé
    "Jaunes":   [((20,  50,  50), (35,  255, 255))],
    "Verts":    [((35,  50,  50), (85,  255, 255))],
    "Cyans":    [((85,  50,  50), (100, 255, 255))],
    "Bleus":    [((100, 50,  50), (130, 255, 255))],
    "Magentas": [((130, 50,  50), (170, 255, 255))],
    # Pour Blancs, Neutres, Noirs, on gère via la luminosité (V) et saturation (S)
    "Blancs":   "whites",
    "Neutres":  "neutrals",
    "Noirs":    "blacks"
}

# ----------------------------------------------------------------------------
# 2) Conversion simplifiée RGB <-> CMYK (approche naïve, non gérée par ICC)
# ----------------------------------------------------------------------------
def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100
    r_ = r / 255.0
    g_ = g / 255.0
    b_ = b / 255.0
    k = 1 - max(r_, g_, b_)
    c = (1 - r_ - k) / (1 - k + 1e-8)
    m = (1 - g_ - k) / (1 - k + 1e-8)
    y = (1 - b_ - k) / (1 - k + 1e-8)
    return (c * 100, m * 100, y * 100, k * 100)

def cmyk_to_rgb(c, m, y, k):
    C = c / 100.0
    M = m / 100.0
    Y = y / 100.0
    K = k / 100.0
    r_ = 1 - min(1, C + K)
    g_ = 1 - min(1, M + K)
    b_ = 1 - min(1, Y + K)
    return (int(r_ * 255), int(g_ * 255), int(b_ * 255))

# ----------------------------------------------------------------------------
# 3) Création d'un masque pour Blancs / Neutres / Noirs
# ----------------------------------------------------------------------------
def mask_special_zones(img_hsv, zone):
    """
    Crée un masque pour "Blancs", "Neutres", "Noirs" via la luminosité (V) et la saturation (S).
    - Blancs  : V>200 et S<50
    - Noirs   : V<50
    - Neutres : S<50, V moyen (ex: 50..200)
    """
    H, S, V = cv2.split(img_hsv)
    mask = np.zeros_like(H, dtype=np.uint8)

    if zone == "Blancs":
        mask[(V > 200) & (S < 50)] = 255
    elif zone == "Noirs":
        mask[(V < 50)] = 255
    elif zone == "Neutres":
        mask[(S < 50) & (V >= 50) & (V <= 200)] = 255

    return mask

# ----------------------------------------------------------------------------
# 4) Extraire le masque correspondant à la gamme de couleur choisie
# ----------------------------------------------------------------------------
def get_color_mask(img_bgr, target_color):
    """
    Retourne un masque (0 ou 255) pour la gamme de couleur sélectionnée.
    - img_bgr : image BGR (OpenCV)
    - target_color : ex. "Rouges", "Jaunes", "Blancs", etc.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if target_color in ["Blancs", "Neutres", "Noirs"]:
        # Gammes particulières
        mask = mask_special_zones(img_hsv, target_color)
    else:
        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        for (low, high) in color_ranges[target_color]:
            lower = np.array(low, dtype=np.uint8)
            upper = np.array(high, dtype=np.uint8)
            temp_mask = cv2.inRange(img_hsv, lower, upper)
            mask = cv2.bitwise_or(mask, temp_mask)

    return mask

# ----------------------------------------------------------------------------
# 5) Appliquer la correction sélective sur la zone masquée
# ----------------------------------------------------------------------------
def apply_selective_color(img_bgr, mask, c_adj, m_adj, y_adj, k_adj, method="Relative"):
    """
    - img_bgr: image BGR (uint8)
    - mask: masque 0/255 indiquant la zone à corriger
    - c_adj, m_adj, y_adj, k_adj : ajustements CMJN (-100..+100)
    - method: "Relative" ou "Absolute"
    Retourne l'image BGR modifiée.
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
                else:  # "Absolute"
                    c += c_adj
                    m += m_adj
                    y += y_adj
                    k += k_adj

                # Bornage [0..100]
                c = max(0, min(100, c))
                m = max(0, min(100, m))
                y = max(0, min(100, y))
                k = max(0, min(100, k))

                # Conversion inverse en RGB
                r2, g2, b2 = cmyk_to_rgb(c, m, y, k)
                out_img[row, col] = (r2, g2, b2)

    # Retour en BGR
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

# ----------------------------------------------------------------------------
# 6) Interface Streamlit
# ----------------------------------------------------------------------------
st.title("Correction sélective - Mode dynamique + deuxième image (fond blanc)")

uploaded_file = st.file_uploader("Téléversez une image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Lecture de l'image en BGR
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    st.subheader("Image originale")
    st.image(pil_img, use_column_width=True)

    # Paramètres dans la barre latérale
    st.sidebar.title("Paramètres de correction sélective")
    selected_range = st.sidebar.selectbox(
        "Couleurs",
        list(color_ranges.keys())  # ["Rouges","Jaunes",...,"Blancs","Neutres","Noirs"]
    )
    c_adj = st.sidebar.slider("Cyan",  -100, 100, 0, step=1)
    m_adj = st.sidebar.slider("Magenta", -100, 100, 0, step=1)
    y_adj = st.sidebar.slider("Jaune",   -100, 100, 0, step=1)
    k_adj = st.sidebar.slider("Noir",    -100, 100, 0, step=1)
    method = st.sidebar.radio("Méthode", ("Relative", "Absolute"))

    # Checkbox pour la couche colorée sur la deuxième image
    show_layer = st.sidebar.checkbox("Afficher la couche de couleur sur l'image du bas", value=True)

    # Calcul dynamique du masque
    mask = get_color_mask(img_bgr, selected_range)

    # Application de la correction sélective
    out_bgr = apply_selective_color(img_bgr, mask, c_adj, m_adj, y_adj, k_adj, method)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Image modifiée")
    st.image(out_rgb, use_column_width=True)

    # ------------------------------------------------------------------------
    # Génération de la deuxième image (fond blanc + zones modifiées)
    # ------------------------------------------------------------------------
    h, w = img_bgr.shape[:2]
    # On part d'une image blanche
    layer_bgr = np.full((h, w, 3), 255, dtype=np.uint8)

    # Si l'option est cochée, on copie les pixels modifiés sur fond blanc
    if show_layer:
        layer_bgr[mask != 0] = out_bgr[mask != 0]

    layer_rgb = cv2.cvtColor(layer_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Couche de couleur (fond blanc)")
    st.image(layer_rgb, use_column_width=True)

else:
    st.write("Veuillez téléverser une image pour commencer.")
