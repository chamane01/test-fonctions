import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------
# 1) Définir les gammes de couleurs en HSV pour l'identification
# ---------------------------------------------------------------------
color_ranges = {
    "Rouges":   [((0,  50,  50), (10, 255, 255)),  # Rouge clair
                 ((170,50,  50), (180,255,255))], # Rouge foncé
    "Jaunes":   [((20, 50,  50), (35, 255, 255))],
    "Verts":    [((35, 50,  50), (85, 255, 255))],
    "Cyans":    [((85, 50,  50), (100,255, 255))],
    "Bleus":    [((100,50,  50), (130,255, 255))],
    "Magentas": [((130,50,  50), (170,255, 255))],
    "Blancs":   "whites",   # zones très claires et peu saturées
    "Neutres":  "neutrals", # zones saturation faible, luminosité moyenne
    "Noirs":    "blacks"    # zones très sombres
}

# ---------------------------------------------------------------------
# 2) Conversion simplifiée RGB <-> CMYK
# ---------------------------------------------------------------------
def rgb_to_cmyk(r, g, b):
    """Convertit (r,g,b) [0..255] en (c,m,y,k) [0..100], méthode simplifiée."""
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
    """Convertit (c,m,y,k) [0..100] en (r,g,b) [0..255], méthode simplifiée."""
    C = c / 100.0
    M = m / 100.0
    Y = y / 100.0
    K = k / 100.0
    r_ = 1 - min(1, C + K)
    g_ = 1 - min(1, M + K)
    b_ = 1 - min(1, Y + K)
    return (int(r_ * 255), int(g_ * 255), int(b_ * 255))

# ---------------------------------------------------------------------
# 3) Création du masque pour Blancs / Neutres / Noirs
# ---------------------------------------------------------------------
def mask_special_zones(img_hsv, zone):
    """
    Crée un masque pour "Blancs", "Neutres" ou "Noirs" selon la luminosité (V) et la saturation (S).
    - Blancs  : V > 200, S < 50
    - Noirs   : V < 50
    - Neutres : S < 50 et V moyen
    """
    h, w = img_hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    H, S, V = cv2.split(img_hsv)

    if zone == "Blancs":
        # Ex: V>200 et S<50
        mask[(V > 200) & (S < 50)] = 255
    elif zone == "Noirs":
        # Ex: V<50
        mask[(V < 50)] = 255
    elif zone == "Neutres":
        # Ex: S<50, V entre 50 et 200 (à ajuster selon vos besoins)
        mask[(S < 50) & (V >= 50) & (V <= 200)] = 255

    return mask

# ---------------------------------------------------------------------
# 4) Fonction de correction sélective (approximation)
# ---------------------------------------------------------------------
def apply_selective_color(img_bgr, target_color, c_adj, m_adj, y_adj, k_adj, method="Relative"):
    """
    - img_bgr: image OpenCV en BGR (uint8)
    - target_color: "Rouges", "Jaunes", etc. (clé du dictionnaire color_ranges)
    - c_adj, m_adj, y_adj, k_adj: ajustements CMJN (-100..+100)
    - method: "Relative" ou "Absolute"
    Retourne une nouvelle image BGR modifiée.
    """
    # Convertir en HSV pour détecter la zone ciblée
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Construire un masque global pour la gamme de couleur
    if target_color in ["Blancs", "Neutres", "Noirs"]:
        mask = mask_special_zones(img_hsv, target_color)
    else:
        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        for (low, high) in color_ranges[target_color]:
            lower = np.array(low, dtype=np.uint8)
            upper = np.array(high, dtype=np.uint8)
            temp_mask = cv2.inRange(img_hsv, lower, upper)
            mask = cv2.bitwise_or(mask, temp_mask)

    # Convertir l'image en RGB pour la conversion CMYK
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_img = img_rgb.copy()
    h, w = out_img.shape[:2]

    # Parcourir l'image et appliquer la transformation
    for row in range(h):
        for col in range(w):
            if mask[row, col] != 0:
                r, g, b = out_img[row, col]
                c, m, y, k = rgb_to_cmyk(r, g, b)

                if method == "Relative":
                    # Ex : c += c_adj% de c
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

                # Reconvertir en RGB
                r2, g2, b2 = cmyk_to_rgb(c, m, y, k)
                out_img[row, col] = (r2, g2, b2)

    # Reconversion en BGR pour OpenCV
    final_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return final_bgr

# ---------------------------------------------------------------------
# 5) Interface Streamlit
# ---------------------------------------------------------------------
st.title("Correction sélective (approx. Photoshop) - Mode dynamique")

# Téléversement
uploaded_file = st.file_uploader("Téléversez une image (JPEG ou PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Lecture de l'image
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Affichage de l'image originale
    st.subheader("Image originale")
    st.image(pil_img, use_column_width=True)

    # Paramètres dans la barre latérale
    st.sidebar.title("Paramètres de correction sélective")
    selected_range = st.sidebar.selectbox(
        "Couleurs",
        ["Rouges", "Jaunes", "Verts", "Cyans", "Bleus", "Magentas", "Blancs", "Neutres", "Noirs"]
    )
    c_adj = st.sidebar.slider("Cyan",  -100, 100, 0, step=1)
    m_adj = st.sidebar.slider("Magenta", -100, 100, 0, step=1)
    y_adj = st.sidebar.slider("Jaune",   -100, 100, 0, step=1)
    k_adj = st.sidebar.slider("Noir",    -100, 100, 0, step=1)
    method = st.sidebar.radio("Méthode", ("Relative", "Absolute"))

    # --- Application dynamique ---
    # Dès que l'utilisateur change un paramètre, Streamlit relance le script.
    # Nous recalculons donc l'image modifiée à chaque itération.
    out_bgr = apply_selective_color(img_bgr, selected_range, c_adj, m_adj, y_adj, k_adj, method)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # Affichage de l'image modifiée
    st.subheader("Image modifiée")
    st.image(out_rgb, use_column_width=True)

else:
    st.write("Veuillez téléverser une image pour commencer.")
