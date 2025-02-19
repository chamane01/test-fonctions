import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ----------------------------------------------------------------------------
# 1) Définir les gammes de couleurs (approximation HSV ou via V/S)
# ----------------------------------------------------------------------------
color_ranges = {
    "Rouges":   [((0,   50,  50), (10,  255, 255)),
                 ((170, 50,  50), (180, 255, 255))],
    "Jaunes":   [((20,  50,  50), (35,  255, 255))],
    "Verts":    [((35,  50,  50), (85,  255, 255))],
    "Cyans":    [((85,  50,  50), (100, 255, 255))],
    "Bleus":    [((100, 50,  50), (130, 255, 255))],
    "Magentas": [((130, 50,  50), (170, 255, 255))],
    # Pour Blancs, Neutres, Noirs, on se base sur la luminosité et la saturation
    "Blancs":   "whites",
    "Neutres":  "neutrals",
    "Noirs":    "blacks"
}

layer_names = ["Rouges", "Jaunes", "Verts", "Cyans", "Bleus", "Magentas", "Blancs", "Neutres", "Noirs"]

# ----------------------------------------------------------------------------
# 2) Fonctions de conversion RGB <-> CMYK (approche simplifiée)
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
# 3) Masque pour les zones spéciales (Blancs, Neutres, Noirs)
# ----------------------------------------------------------------------------
def mask_special_zones(img_hsv, zone):
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
# 4) Récupérer le masque pour une gamme de couleur donnée
# ----------------------------------------------------------------------------
def get_color_mask(img_bgr, target_color):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if target_color in ["Blancs", "Neutres", "Noirs"]:
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
# 5) Appliquer la correction sélective sur une zone (masque)
# ----------------------------------------------------------------------------
def apply_selective_color(img_bgr, mask, c_adj, m_adj, y_adj, k_adj, method="Relative"):
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
                else:  # Absolute
                    c += c_adj
                    m += m_adj
                    y += y_adj
                    k += k_adj
                c = max(0, min(100, c))
                m = max(0, min(100, m))
                y = max(0, min(100, y))
                k = max(0, min(100, k))
                r2, g2, b2 = cmyk_to_rgb(c, m, y, k)
                out_img[row, col] = (r2, g2, b2)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

# ----------------------------------------------------------------------------
# 6) Interface Streamlit – Paramètres multicouches
# ----------------------------------------------------------------------------
st.sidebar.title("Paramètres de Correction Sélective - Multicouche")

# Pour chaque couche, on stocke dans un dictionnaire ses paramètres
layer_params = {}
for layer in layer_names:
    with st.sidebar.expander(f"Couche {layer}"):
        active = st.checkbox("Activer", value=False, key=f"active_{layer}")
        if active:
            c_adj = st.slider("Cyan", -100, 100, 0, key=f"c_{layer}")
            m_adj = st.slider("Magenta", -100, 100, 0, key=f"m_{layer}")
            y_adj = st.slider("Jaune", -100, 100, 0, key=f"y_{layer}")
            k_adj = st.slider("Noir", -100, 100, 0, key=f"k_{layer}")
            method = st.radio("Méthode", options=["Relative", "Absolute"], index=0, key=f"method_{layer}")
        else:
            c_adj, m_adj, y_adj, k_adj, method = 0, 0, 0, 0, "Relative"
        layer_params[layer] = {
            "active": active,
            "c_adj": c_adj,
            "m_adj": m_adj,
            "y_adj": y_adj,
            "k_adj": k_adj,
            "method": method
        }

st.sidebar.markdown("---")
st.sidebar.title("Mode d'affichage")
main_display_mode = st.sidebar.radio("Image modifiée", options=["Combinaison", "Couche active"])
color_layer_display_mode = st.sidebar.radio("Couche de couleur (fond blanc)", options=["Combinaison", "Couche active"])

active_layers = [layer for layer in layer_names if layer_params[layer]["active"]]
if main_display_mode == "Couche active" and active_layers:
    selected_main_layer = st.sidebar.selectbox("Sélectionnez la couche pour l'image modifiée", options=active_layers)
else:
    selected_main_layer = None

if color_layer_display_mode == "Couche active" and active_layers:
    selected_color_layer = st.sidebar.selectbox("Sélectionnez la couche pour la couche de couleur", options=active_layers)
else:
    selected_color_layer = None

# ----------------------------------------------------------------------------
# 7) Chargement de l'image et calcul des résultats par couche
# ----------------------------------------------------------------------------
st.title("Correction Sélective – Mode Multicouche Dynamique")
uploaded_file = st.file_uploader("Téléversez une image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    st.subheader("Image originale")
    st.image(pil_img, use_column_width=True)

    # Dictionnaire pour stocker pour chaque couche : masque et image corrigée
    layer_results = {}
    for layer in active_layers:
        mask = get_color_mask(original_bgr, layer)
        params = layer_params[layer]
        result = apply_selective_color(original_bgr, mask, params["c_adj"], params["m_adj"],
                                       params["y_adj"], params["k_adj"], params["method"])
        layer_results[layer] = {"mask": mask, "result": result}

    # Image modifiée – mode combiné
    combined_main = original_bgr.copy()
    for layer in active_layers:
        mask = layer_results[layer]["mask"]
        combined_main[mask != 0] = layer_results[layer]["result"][mask != 0]

    # Image modifiée – mode couche unique
    if main_display_mode == "Couche active" and selected_main_layer in layer_results:
        main_display_img = layer_results[selected_main_layer]["result"]
    else:
        main_display_img = combined_main

    # Couche de couleur sur fond blanc – mode combiné
    h, w = original_bgr.shape[:2]
    combined_color_layer = np.full((h, w, 3), 255, dtype=np.uint8)
    for layer in active_layers:
        mask = layer_results[layer]["mask"]
        combined_color_layer[mask != 0] = layer_results[layer]["result"][mask != 0]

    # Couche de couleur – mode couche unique
    if color_layer_display_mode == "Couche active" and selected_color_layer in layer_results:
        single_color = np.full((h, w, 3), 255, dtype=np.uint8)
        mask = layer_results[selected_color_layer]["mask"]
        single_color[mask != 0] = layer_results[selected_color_layer]["result"][mask != 0]
        color_display_img = single_color
    else:
        color_display_img = combined_color_layer

    # Conversion pour affichage (BGR -> RGB)
    main_display_rgb = cv2.cvtColor(main_display_img, cv2.COLOR_BGR2RGB)
    color_display_rgb = cv2.cvtColor(color_display_img, cv2.COLOR_BGR2RGB)

    st.subheader("Image modifiée")
    st.image(main_display_rgb, use_column_width=True)

    st.subheader("Couche de couleur (fond blanc)")
    st.image(color_display_rgb, use_column_width=True)
else:
    st.write("Veuillez téléverser une image pour commencer.")
