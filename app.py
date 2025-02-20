import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import tempfile
import os

# ----------------------------------------------------------------------------
# 1) Définir les gammes de couleurs (approximation HSV ou via V/S)
# ----------------------------------------------------------------------------
color_ranges = {
    "Rouges":   [((0, 50, 50), (10, 255, 255)),
                 ((170, 50, 50), (180, 255, 255))],
    "Jaunes":   [((20, 50, 50), (35, 255, 255))],
    "Verts":    [((35, 50, 50), (85, 255, 255))],
    "Cyans":    [((85, 50, 50), (100, 255, 255))],
    "Bleus":    [((100, 50, 50), (130, 255, 255))],
    "Magentas": [((130, 50, 50), (170, 255, 255))],
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
# 5) Suppression des petites composantes connectées (filtrage du bruit)
# ----------------------------------------------------------------------------
def remove_small_components(mask, min_area):
    """
    Supprime les composantes connectées dont l'aire est inférieure à 'min_area'.
    """
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            new_mask[labels == i] = 255
    return new_mask

# ----------------------------------------------------------------------------
# 6) Appliquer la correction sélective sur une zone (masque)
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
# 7) Appliquer les modifications classiques
# ----------------------------------------------------------------------------
def apply_classic_modifications(img, brightness=0, contrast=1.0, saturation=1.0, gamma=1.0):
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        img = cv2.LUT(img, table)
    return img

# ----------------------------------------------------------------------------
# 8) Génération d'un PDF cumulatif
# ----------------------------------------------------------------------------
def generate_pdf_cumulative(export_text, image_list):
    """
    Génère un PDF avec en première page le texte d'export,
    puis une page par image présente dans image_list.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, export_text)
    for img in image_list:
        pdf.add_page()
        temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_img_file.name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pdf.image(temp_img_file.name, x=10, y=10, w=pdf.w - 20)
        os.unlink(temp_img_file.name)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes

# ----------------------------------------------------------------------------
# 9) Barre latérale : paramètres et filtres
# ----------------------------------------------------------------------------
st.sidebar.title("Paramètres de Correction")

# Séquence de corrections
correction_sequence = st.sidebar.radio("Séquence de corrections",
                                         options=["Correction 1 seule", "Correction 2 (en chaîne)", "Correction 3 (en chaîne)"])

# ----- Pour chaque correction, dans chaque couche, on ajoute le contrôle du filtrage bruit -----

# ----- Correction 1 -----
st.sidebar.subheader("Correction 1")
with st.sidebar.expander("Paramètres par Couche (Corr 1)"):
    layer_params_corr1 = {}
    for layer in layer_names:
        with st.sidebar.expander(f"Couche {layer}"):
            active = st.checkbox("Activer", value=False, key=f"active_corr1_{layer}")
            if active:
                c_adj = st.slider("Cyan", -100, 100, 0, key=f"c_corr1_{layer}")
                m_adj = st.slider("Magenta", -100, 100, 0, key=f"m_corr1_{layer}")
                y_adj = st.slider("Jaune", -100, 100, 0, key=f"y_corr1_{layer}")
                k_adj = st.slider("Noir", -100, 100, 0, key=f"k_corr1_{layer}")
                method = st.radio("Méthode", options=["Relative", "Absolute"], index=0, key=f"method_corr1_{layer}")
                # --- Contrôle de filtrage du bruit pour cette couche
                noise_active = st.checkbox("Activer filtrage bruit", value=False, key=f"noise_active_corr1_{layer}")
                if noise_active:
                    noise_min_area = st.slider("Taille minimale (pixels)", 0, 5000, 50, key=f"noise_min_corr1_{layer}")
                else:
                    noise_min_area = 0
            else:
                c_adj, m_adj, y_adj, k_adj, method = 0, 0, 0, 0, "Relative"
                noise_active = False
                noise_min_area = 0
            layer_params_corr1[layer] = {
                "active": active,
                "c_adj": c_adj,
                "m_adj": m_adj,
                "y_adj": y_adj,
                "k_adj": k_adj,
                "method": method,
                "noise_active": noise_active,
                "noise_min_area": noise_min_area
            }
with st.sidebar.expander("Modifications Classiques (Corr 1)"):
    classic_active_corr1 = st.checkbox("Activer modifs classiques", key="classic_active_corr1")
    if classic_active_corr1:
        brightness_corr1 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr1")
        contrast_corr1 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr1")
        saturation_corr1 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr1")
        gamma_corr1 = st.slider("Gamma", 50, 150, 100, key="gamma_corr1")
    else:
        brightness_corr1, contrast_corr1, saturation_corr1, gamma_corr1 = 0, 100, 100, 100

# ----- Correction 2 (si applicable) -----
if correction_sequence in ["Correction 2 (en chaîne)", "Correction 3 (en chaîne)"]:
    st.sidebar.subheader("Correction 2")
    with st.sidebar.expander("Paramètres par Couche (Corr 2)"):
        layer_params_corr2 = {}
        for layer in layer_names:
            with st.sidebar.expander(f"Couche {layer}"):
                active = st.checkbox("Activer", value=False, key=f"active_corr2_{layer}")
                if active:
                    c_adj = st.slider("Cyan", -100, 100, 0, key=f"c_corr2_{layer}")
                    m_adj = st.slider("Magenta", -100, 100, 0, key=f"m_corr2_{layer}")
                    y_adj = st.slider("Jaune", -100, 100, 0, key=f"y_corr2_{layer}")
                    k_adj = st.slider("Noir", -100, 100, 0, key=f"k_corr2_{layer}")
                    method = st.radio("Méthode", options=["Relative", "Absolute"], index=0, key=f"method_corr2_{layer}")
                    noise_active = st.checkbox("Activer filtrage bruit", value=False, key=f"noise_active_corr2_{layer}")
                    if noise_active:
                        noise_min_area = st.slider("Taille minimale (pixels)", 0, 5000, 50, key=f"noise_min_corr2_{layer}")
                    else:
                        noise_min_area = 0
                else:
                    c_adj, m_adj, y_adj, k_adj, method = 0, 0, 0, 0, "Relative"
                    noise_active = False
                    noise_min_area = 0
                layer_params_corr2[layer] = {
                    "active": active,
                    "c_adj": c_adj,
                    "m_adj": m_adj,
                    "y_adj": y_adj,
                    "k_adj": k_adj,
                    "method": method,
                    "noise_active": noise_active,
                    "noise_min_area": noise_min_area
                }
    with st.sidebar.expander("Modifications Classiques (Corr 2)"):
        classic_active_corr2 = st.checkbox("Activer modifs classiques", key="classic_active_corr2")
        if classic_active_corr2:
            brightness_corr2 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr2")
            contrast_corr2 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr2")
            saturation_corr2 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr2")
            gamma_corr2 = st.slider("Gamma", 50, 150, 100, key="gamma_corr2")
        else:
            brightness_corr2, contrast_corr2, saturation_corr2, gamma_corr2 = 0, 100, 100, 100

# ----- Correction 3 (si applicable) -----
if correction_sequence == "Correction 3 (en chaîne)":
    st.sidebar.subheader("Correction 3")
    with st.sidebar.expander("Paramètres par Couche (Corr 3)"):
        layer_params_corr3 = {}
        for layer in layer_names:
            with st.sidebar.expander(f"Couche {layer}"):
                active = st.checkbox("Activer", value=False, key=f"active_corr3_{layer}")
                if active:
                    c_adj = st.slider("Cyan", -100, 100, 0, key=f"c_corr3_{layer}")
                    m_adj = st.slider("Magenta", -100, 100, 0, key=f"m_corr3_{layer}")
                    y_adj = st.slider("Jaune", -100, 100, 0, key=f"y_corr3_{layer}")
                    k_adj = st.slider("Noir", -100, 100, 0, key=f"k_corr3_{layer}")
                    method = st.radio("Méthode", options=["Relative", "Absolute"], index=0, key=f"method_corr3_{layer}")
                    noise_active = st.checkbox("Activer filtrage bruit", value=False, key=f"noise_active_corr3_{layer}")
                    if noise_active:
                        noise_min_area = st.slider("Taille minimale (pixels)", 0, 5000, 50, key=f"noise_min_corr3_{layer}")
                    else:
                        noise_min_area = 0
                else:
                    c_adj, m_adj, y_adj, k_adj, method = 0, 0, 0, 0, "Relative"
                    noise_active = False
                    noise_min_area = 0
                layer_params_corr3[layer] = {
                    "active": active,
                    "c_adj": c_adj,
                    "m_adj": m_adj,
                    "y_adj": y_adj,
                    "k_adj": k_adj,
                    "method": method,
                    "noise_active": noise_active,
                    "noise_min_area": noise_min_area
                }
    with st.sidebar.expander("Modifications Classiques (Corr 3)"):
        classic_active_corr3 = st.checkbox("Activer modifs classiques", key="classic_active_corr3")
        if classic_active_corr3:
            brightness_corr3 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr3")
            contrast_corr3 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr3")
            saturation_corr3 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr3")
            gamma_corr3 = st.slider("Gamma", 50, 150, 100, key="gamma_corr3")
        else:
            brightness_corr3, contrast_corr3, saturation_corr3, gamma_corr3 = 0, 100, 100, 100

# ----- Mode d'affichage -----
st.sidebar.markdown("---")
st.sidebar.subheader("Mode d'affichage")
main_display_mode = st.sidebar.radio("Image modifiée", options=["Combinaison", "Couche active"], key="main_display_mode")
color_layer_display_mode = st.sidebar.radio("Couche de couleur (fond blanc)", options=["Combinaison", "Couche active"], key="color_layer_display_mode")

if main_display_mode == "Couche active":
    if correction_sequence == "Correction 1 seule":
        active_layers = [layer for layer in layer_names if layer_params_corr1[layer]["active"]]
        selected_main_layer = st.sidebar.selectbox("Sélectionnez la couche (Corr 1)", options=active_layers, key="selected_main_corr1") if active_layers else None
    elif correction_sequence == "Correction 2 (en chaîne)":
        active_layers = [layer for layer in layer_names if layer_params_corr2[layer]["active"]]
        selected_main_layer = st.sidebar.selectbox("Sélectionnez la couche (Corr 2)", options=active_layers, key="selected_main_corr2") if active_layers else None
    else:
        active_layers = [layer for layer in layer_names if layer_params_corr3[layer]["active"]]
        selected_main_layer = st.sidebar.selectbox("Sélectionnez la couche (Corr 3)", options=active_layers, key="selected_main_corr3") if active_layers else None
else:
    selected_main_layer = None

if color_layer_display_mode == "Couche active":
    if correction_sequence == "Correction 1 seule":
        active_layers = [layer for layer in layer_names if layer_params_corr1[layer]["active"]]
        selected_color_layer = st.sidebar.selectbox("Sélectionnez la couche (Corr 1)", options=active_layers, key="selected_color_corr1") if active_layers else None
    elif correction_sequence == "Correction 2 (en chaîne)":
        active_layers = [layer for layer in layer_names if layer_params_corr2[layer]["active"]]
        selected_color_layer = st.sidebar.selectbox("Sélectionnez la couche (Corr 2)", options=active_layers, key="selected_color_corr2") if active_layers else None
    else:
        active_layers = [layer for layer in layer_names if layer_params_corr3[layer]["active"]]
        selected_color_layer = st.sidebar.selectbox("Sélectionnez la couche (Corr 3)", options=active_layers, key="selected_color_corr3") if active_layers else None
else:
    selected_color_layer = None

# ----------------------------------------------------------------------------
# Fonction d'export texte (rapport)
# ----------------------------------------------------------------------------
def generate_export_text(correction_label, layer_params, classic_active, brightness, contrast, saturation, gamma):
    text = f"Export pour {correction_label}\n\n"
    text += "Paramètres par Couche:\n"
    for layer in layer_names:
        params = layer_params.get(layer, None)
        if params and params["active"]:
            text += f" - Couche {layer}: Cyan: {params['c_adj']}, Magenta: {params['m_adj']}, Jaune: {params['y_adj']}, Noir: {params['k_adj']}, Méthode: {params['method']}"
            text += f", Filtrage bruit: {params['noise_active']} (min: {params['noise_min_area']})\n"
    text += "\nModifications Classiques:\n"
    text += f" - Actif: {classic_active}\n"
    if classic_active:
        text += f"   Luminosité: {brightness}\n"
        text += f"   Contraste: {contrast}\n"
        text += f"   Saturation: {saturation}\n"
        text += f"   Gamma: {gamma}\n"
    return text

# ----------------------------------------------------------------------------
# Traitement de l'image et application des corrections
# ----------------------------------------------------------------------------
st.title("Correction Sélective – Mode Multicouche Dynamique")
uploaded_file = st.file_uploader("Téléversez une image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    st.subheader("Image originale")
    st.image(pil_img, use_container_width=True)

    # --- Correction 1 ---
    layer_results_corr1 = {}
    for layer in layer_names:
        if layer_params_corr1[layer]["active"]:
            mask = get_color_mask(original_bgr, layer)
            if layer_params_corr1[layer]["noise_active"]:
                mask = remove_small_components(mask, layer_params_corr1[layer]["noise_min_area"])
            params = layer_params_corr1[layer]
            result = apply_selective_color(original_bgr, mask,
                                           params["c_adj"], params["m_adj"],
                                           params["y_adj"], params["k_adj"],
                                           params["method"])
            layer_results_corr1[layer] = {"mask": mask, "result": result}
    combined_main_corr1 = original_bgr.copy()
    for layer in layer_results_corr1:
        mask = layer_results_corr1[layer]["mask"]
        combined_main_corr1[mask != 0] = layer_results_corr1[layer]["result"][mask != 0]
    h, w = original_bgr.shape[:2]
    combined_color_corr1 = np.full((h, w, 3), 255, dtype=np.uint8)
    for layer in layer_results_corr1:
        mask = layer_results_corr1[layer]["mask"]
        combined_color_corr1[mask != 0] = layer_results_corr1[layer]["result"][mask != 0]
    if main_display_mode == "Couche active" and selected_main_layer in layer_results_corr1:
        main_display_corr1 = layer_results_corr1[selected_main_layer]["result"]
    else:
        main_display_corr1 = combined_main_corr1
    if color_layer_display_mode == "Couche active" and selected_color_layer in layer_results_corr1:
        single_color = np.full((h, w, 3), 255, dtype=np.uint8)
        mask = layer_results_corr1[selected_color_layer]["mask"]
        single_color[mask != 0] = layer_results_corr1[selected_color_layer]["result"][mask != 0]
        color_display_corr1 = single_color
    else:
        color_display_corr1 = combined_color_corr1
    if classic_active_corr1:
        contrast_factor_corr1 = contrast_corr1 / 100.0
        saturation_factor_corr1 = saturation_corr1 / 100.0
        gamma_factor_corr1 = gamma_corr1 / 100.0
        main_display_corr1 = apply_classic_modifications(main_display_corr1, brightness=brightness_corr1,
                                                         contrast=contrast_factor_corr1,
                                                         saturation=saturation_factor_corr1,
                                                         gamma=gamma_factor_corr1)
        color_display_corr1 = apply_classic_modifications(color_display_corr1, brightness=brightness_corr1,
                                                          contrast=contrast_factor_corr1,
                                                          saturation=saturation_factor_corr1,
                                                          gamma=gamma_factor_corr1)
    if correction_sequence == "Correction 1 seule":
        st.subheader("Image modifiée (Correction 1)")
        st.image(cv2.cvtColor(main_display_corr1, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.subheader("Couche de couleur (fond blanc) - Correction 1")
        st.image(cv2.cvtColor(color_display_corr1, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Export Corr 1
        export_text_corr1 = generate_export_text("Correction 1", layer_params_corr1, classic_active_corr1,
                                                 brightness_corr1, contrast_corr1, saturation_corr1, gamma_corr1)
        pdf_bytes_corr1 = generate_pdf_cumulative(export_text_corr1, [original_bgr, main_display_corr1, color_display_corr1])
        st.download_button("Télécharger les paramètres (TXT) - Corr 1",
                           data=export_text_corr1,
                           file_name="export_correction1.txt",
                           mime="text/plain")
        st.download_button("Télécharger le rapport (PDF) - Corr 1",
                           data=pdf_bytes_corr1,
                           file_name="export_correction1.pdf",
                           mime="application/pdf")
    # --- Correction 2 en chaîne (à partir de Corr 1) ---
    elif correction_sequence in ["Correction 2 (en chaîne)", "Correction 3 (en chaîne)"]:
        corr1_source = main_display_corr1.copy()
        layer_results_corr2 = {}
        for layer in layer_names:
            if layer_params_corr2[layer]["active"]:
                mask = get_color_mask(corr1_source, layer)
                if layer_params_corr2[layer]["noise_active"]:
                    mask = remove_small_components(mask, layer_params_corr2[layer]["noise_min_area"])
                params = layer_params_corr2[layer]
                result = apply_selective_color(corr1_source, mask,
                                               params["c_adj"], params["m_adj"],
                                               params["y_adj"], params["k_adj"],
                                               params["method"])
                layer_results_corr2[layer] = {"mask": mask, "result": result}
        combined_main_corr2 = corr1_source.copy()
        for layer in layer_results_corr2:
            mask = layer_results_corr2[layer]["mask"]
            combined_main_corr2[mask != 0] = layer_results_corr2[layer]["result"][mask != 0]
        combined_color_corr2 = np.full((h, w, 3), 255, dtype=np.uint8)
        for layer in layer_results_corr2:
            mask = layer_results_corr2[layer]["mask"]
            combined_color_corr2[mask != 0] = layer_results_corr2[layer]["result"][mask != 0]
        if main_display_mode == "Couche active" and selected_main_layer in layer_results_corr2:
            main_display_corr2 = layer_results_corr2[selected_main_layer]["result"]
        else:
            main_display_corr2 = combined_main_corr2
        if color_layer_display_mode == "Couche active" and selected_color_layer in layer_results_corr2:
            single_color_corr2 = np.full((h, w, 3), 255, dtype=np.uint8)
            mask = layer_results_corr2[selected_color_layer]["mask"]
            single_color_corr2[mask != 0] = layer_results_corr2[selected_color_layer]["result"][mask != 0]
            color_display_corr2 = single_color_corr2
        else:
            color_display_corr2 = combined_color_corr2
        if classic_active_corr2:
            contrast_factor_corr2 = contrast_corr2 / 100.0
            saturation_factor_corr2 = saturation_corr2 / 100.0
            gamma_factor_corr2 = gamma_corr2 / 100.0
            main_display_corr2 = apply_classic_modifications(main_display_corr2, brightness=brightness_corr2,
                                                             contrast=contrast_factor_corr2,
                                                             saturation=saturation_factor_corr2,
                                                             gamma=gamma_factor_corr2)
            color_display_corr2 = apply_classic_modifications(color_display_corr2, brightness=brightness_corr2,
                                                              contrast=contrast_factor_corr2,
                                                              saturation=saturation_factor_corr2,
                                                              gamma=gamma_factor_corr2)
        st.subheader("Image modifiée (Correction 2)")
        st.image(cv2.cvtColor(main_display_corr2, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.subheader("Couche de couleur (fond blanc) - Correction 2")
        st.image(cv2.cvtColor(color_display_corr2, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Export Corr 2 : rapport cumulatif (Corr 1 + Corr 2)
        export_text_corr2 = "Export cumulatif pour Correction 2\n\n"
        export_text_corr2 += generate_export_text("Correction 1", layer_params_corr1, classic_active_corr1,
                                                  brightness_corr1, contrast_corr1, saturation_corr1, gamma_corr1)
        export_text_corr2 += "\n" + generate_export_text("Correction 2", layer_params_corr2, classic_active_corr2,
                                                        brightness_corr2, contrast_corr2, saturation_corr2, gamma_corr2)
        pdf_bytes_corr2 = generate_pdf_cumulative(export_text_corr2,
                                                  [original_bgr, main_display_corr1, color_display_corr1, main_display_corr2, color_display_corr2])
        st.download_button("Télécharger les paramètres (TXT) - Corr 2",
                           data=export_text_corr2,
                           file_name="export_correction2.txt",
                           mime="text/plain")
        st.download_button("Télécharger le rapport (PDF) - Corr 2",
                           data=pdf_bytes_corr2,
                           file_name="export_correction2.pdf",
                           mime="application/pdf")
        
        # --- Correction 3 en chaîne (à partir de Corr 2) ---
        if correction_sequence == "Correction 3 (en chaîne)":
            corr2_source = main_display_corr2.copy()
            layer_results_corr3 = {}
            for layer in layer_names:
                if layer_params_corr3[layer]["active"]:
                    mask = get_color_mask(corr2_source, layer)
                    if layer_params_corr3[layer]["noise_active"]:
                        mask = remove_small_components(mask, layer_params_corr3[layer]["noise_min_area"])
                    params = layer_params_corr3[layer]
                    result = apply_selective_color(corr2_source, mask,
                                                   params["c_adj"], params["m_adj"],
                                                   params["y_adj"], params["k_adj"],
                                                   params["method"])
                    layer_results_corr3[layer] = {"mask": mask, "result": result}
            combined_main_corr3 = corr2_source.copy()
            for layer in layer_results_corr3:
                mask = layer_results_corr3[layer]["mask"]
                combined_main_corr3[mask != 0] = layer_results_corr3[layer]["result"][mask != 0]
            combined_color_corr3 = np.full((h, w, 3), 255, dtype=np.uint8)
            for layer in layer_results_corr3:
                mask = layer_results_corr3[layer]["mask"]
                combined_color_corr3[mask != 0] = layer_results_corr3[layer]["result"][mask != 0]
            if main_display_mode == "Couche active" and selected_main_layer in layer_results_corr3:
                main_display_corr3 = layer_results_corr3[selected_main_layer]["result"]
            else:
                main_display_corr3 = combined_main_corr3
            if color_layer_display_mode == "Couche active" and selected_color_layer in layer_results_corr3:
                single_color_corr3 = np.full((h, w, 3), 255, dtype=np.uint8)
                mask = layer_results_corr3[selected_color_layer]["mask"]
                single_color_corr3[mask != 0] = layer_results_corr3[selected_color_layer]["result"][mask != 0]
                color_display_corr3 = single_color_corr3
            else:
                color_display_corr3 = combined_color_corr3
            if classic_active_corr3:
                contrast_factor_corr3 = contrast_corr3 / 100.0
                saturation_factor_corr3 = saturation_corr3 / 100.0
                gamma_factor_corr3 = gamma_corr3 / 100.0
                main_display_corr3 = apply_classic_modifications(main_display_corr3, brightness=brightness_corr3,
                                                                 contrast=contrast_factor_corr3,
                                                                 saturation=saturation_factor_corr3,
                                                                 gamma=gamma_factor_corr3)
                color_display_corr3 = apply_classic_modifications(color_display_corr3, brightness=brightness_corr3,
                                                                  contrast=contrast_factor_corr3,
                                                                  saturation=saturation_factor_corr3,
                                                                  gamma=gamma_factor_corr3)
            st.subheader("Image modifiée (Correction 3)")
            st.image(cv2.cvtColor(main_display_corr3, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.subheader("Couche de couleur (fond blanc) - Correction 3")
            st.image(cv2.cvtColor(color_display_corr3, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Export Corr 3 : rapport cumulatif (Corr 1 + Corr 2 + Corr 3)
            export_text_corr3 = "Export cumulatif pour Correction 3\n\n"
            export_text_corr3 += generate_export_text("Correction 1", layer_params_corr1, classic_active_corr1,
                                                      brightness_corr1, contrast_corr1, saturation_corr1, gamma_corr1)
            export_text_corr3 += "\n" + generate_export_text("Correction 2", layer_params_corr2, classic_active_corr2,
                                                            brightness_corr2, contrast_corr2, saturation_corr2, gamma_corr2)
            export_text_corr3 += "\n" + generate_export_text("Correction 3", layer_params_corr3, classic_active_corr3,
                                                            brightness_corr3, contrast_corr3, saturation_corr3, gamma_corr3)
            pdf_bytes_corr3 = generate_pdf_cumulative(export_text_corr3,
                                                      [original_bgr, main_display_corr1, color_display_corr1,
                                                       main_display_corr2, color_display_corr2,
                                                       main_display_corr3, color_display_corr3])
            st.download_button("Télécharger les paramètres (TXT) - Corr 3",
                               data=export_text_corr3,
                               file_name="export_correction3.txt",
                               mime="text/plain")
            st.download_button("Télécharger le rapport (PDF) - Corr 3",
                               data=pdf_bytes_corr3,
                               file_name="export_correction3.pdf",
                               mime="application/pdf")
else:
    st.write("Veuillez téléverser une image pour commencer.")
