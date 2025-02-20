import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import tempfile, os
from streamlit_drawable_canvas import st_canvas

# ----------------------------------------------------------------------------
# 1) Définir les gammes de couleurs (approximation HSV)
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
    r_ = r / 255.0; g_ = g / 255.0; b_ = b / 255.0
    k = 1 - max(r_, g_, b_)
    c = (1 - r_ - k) / (1 - k + 1e-8)
    m = (1 - g_ - k) / (1 - k + 1e-8)
    y = (1 - b_ - k) / (1 - k + 1e-8)
    return (c * 100, m * 100, y * 100, k * 100)

def cmyk_to_rgb(c, m, y, k):
    C = c / 100.0; M = m / 100.0; Y = y / 100.0; K = k / 100.0
    r_ = 1 - min(1, C + K)
    g_ = 1 - min(1, M + K)
    b_ = 1 - min(1, Y + K)
    return (int(r_ * 255), int(g_ * 255), int(b_ * 255))

# ----------------------------------------------------------------------------
# 3) Masques pour zones spéciales et sélection de couleur spécifique
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

def get_color_mask(img_bgr, target_color):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if target_color in ["Blancs", "Neutres", "Noirs"]:
        return mask_special_zones(img_hsv, target_color)
    mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for (low, high) in color_ranges[target_color]:
        lower = np.array(low, dtype=np.uint8)
        upper = np.array(high, dtype=np.uint8)
        temp_mask = cv2.inRange(img_hsv, lower, upper)
        mask = cv2.bitwise_or(mask, temp_mask)
    return mask

def get_specific_color_mask(img_bgr, target_hex, tolerance):
    # Convertir la couleur hexadécimale en RGB puis en BGR
    target_rgb = tuple(int(target_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    target_array = np.full(img_hsv[:,:,0].shape, target_hsv[0], dtype=np.uint8)
    diff = cv2.absdiff(img_hsv[:,:,0], target_array)
    diff = np.minimum(diff, 180 - diff)
    mask = cv2.inRange(diff, 0, tolerance)
    return mask

# ----------------------------------------------------------------------------
# 4) Application de la correction sélective sur une zone (masque)
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
                else:
                    c += c_adj; m += m_adj; y += y_adj; k += k_adj
                c = max(0, min(100, c))
                m = max(0, min(100, m))
                y = max(0, min(100, y))
                k = max(0, min(100, k))
                r2, g2, b2 = cmyk_to_rgb(c, m, y, k)
                out_img[row, col] = (r2, g2, b2)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

# ----------------------------------------------------------------------------
# 5) Application des modifications classiques
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
# 6) Barre latérale : paramètres de correction
# ----------------------------------------------------------------------------
st.sidebar.title("Paramètres de Correction")

# Séquence de corrections
correction_sequence = st.sidebar.radio("Séquence de corrections",
    options=["Correction 1 seule", "Correction 2 (en chaîne)", "Correction 3 (en chaîne)"])

# Pour chaque correction, on définit des paramètres par couche et classiques
# -- Correction 1 --
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
            else:
                c_adj, m_adj, y_adj, k_adj, method = 0,0,0,0,"Relative"
            layer_params_corr1[layer] = {"active": active, "c_adj": c_adj, "m_adj": m_adj,
                                         "y_adj": y_adj, "k_adj": k_adj, "method": method}
with st.sidebar.expander("Modifications Classiques (Corr 1)"):
    classic_active_corr1 = st.checkbox("Activer modifs classiques", key="classic_active_corr1")
    if classic_active_corr1:
        brightness_corr1 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr1")
        contrast_corr1 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr1")
        saturation_corr1 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr1")
        gamma_corr1 = st.slider("Gamma", 50, 150, 100, key="gamma_corr1")
    else:
        brightness_corr1, contrast_corr1, saturation_corr1, gamma_corr1 = 0, 100, 100, 100

# -- Correction 2 --
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
                else:
                    c_adj, m_adj, y_adj, k_adj, method = 0,0,0,0,"Relative"
                layer_params_corr2[layer] = {"active": active, "c_adj": c_adj, "m_adj": m_adj,
                                             "y_adj": y_adj, "k_adj": k_adj, "method": method}
    with st.sidebar.expander("Modifications Classiques (Corr 2)"):
        classic_active_corr2 = st.checkbox("Activer modifs classiques", key="classic_active_corr2")
        if classic_active_corr2:
            brightness_corr2 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr2")
            contrast_corr2 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr2")
            saturation_corr2 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr2")
            gamma_corr2 = st.slider("Gamma", 50, 150, 100, key="gamma_corr2")
        else:
            brightness_corr2, contrast_corr2, saturation_corr2, gamma_corr2 = 0, 100, 100, 100

# -- Correction 3 --
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
                else:
                    c_adj, m_adj, y_adj, k_adj, method = 0,0,0,0,"Relative"
                layer_params_corr3[layer] = {"active": active, "c_adj": c_adj, "m_adj": m_adj,
                                             "y_adj": y_adj, "k_adj": k_adj, "method": method}
    with st.sidebar.expander("Modifications Classiques (Corr 3)"):
        classic_active_corr3 = st.checkbox("Activer modifs classiques", key="classic_active_corr3")
        if classic_active_corr3:
            brightness_corr3 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr3")
            contrast_corr3 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr3")
            saturation_corr3 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr3")
            gamma_corr3 = st.slider("Gamma", 50, 150, 100, key="gamma_corr3")
        else:
            brightness_corr3, contrast_corr3, saturation_corr3, gamma_corr3 = 0, 100, 100, 100

# ----------------------------------------------------------------------------
# 7bis) Outil Pupette intégré dans le flow : il sera appliqué sur le résultat de chaque correction
# (On ne propose pas de section globale dans la sidebar ; l'outil pupette apparaîtra sous chaque correction.)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 8) Traitement de l'image et application des corrections
# ----------------------------------------------------------------------------
st.title("Correction Sélective – Mode Multicouche Dynamique")
uploaded_file = st.file_uploader("Téléversez une image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    st.subheader("Image originale")
    st.image(pil_img, use_container_width=True)

    ############################
    # CORRECTION 1
    ############################
    # Calcul de Correction 1
    layer_results_corr1 = {}
    for layer in layer_names:
        if layer_params_corr1[layer]["active"]:
            mask = get_color_mask(original_bgr, layer)
            params = layer_params_corr1[layer]
            result = apply_selective_color(original_bgr, mask, params["c_adj"], params["m_adj"],
                                           params["y_adj"], params["k_adj"], params["method"])
            layer_results_corr1[layer] = {"mask": mask, "result": result}
    combined_main_corr1 = original_bgr.copy()
    for layer in layer_results_corr1:
        msk = layer_results_corr1[layer]["mask"]
        combined_main_corr1[msk != 0] = layer_results_corr1[layer]["result"][msk != 0]
    h, w = original_bgr.shape[:2]
    # Application des modifs classiques si activées
    if classic_active_corr1:
        cf = contrast_corr1 / 100.0; sf = saturation_corr1 / 100.0; gf = gamma_corr1 / 100.0
        combined_main_corr1 = apply_classic_modifications(combined_main_corr1, brightness=brightness_corr1,
                                                          contrast=cf, saturation=sf, gamma=gf)
    st.header("Correction 1")
    st.image(cv2.cvtColor(combined_main_corr1, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Outil Pupette pour Correction 1
    st.markdown("### Outil Pupette - Correction 1")
    use_pupette_corr1 = st.checkbox("Activer l'outil Pupette pour Correction 1", key="use_pupette_corr1")
    pupette_result_corr1 = None
    if use_pupette_corr1:
        canvas_result1 = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=Image.fromarray(cv2.cvtColor(combined_main_corr1, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="point",
            key="canvas_corr1"
        )
        if canvas_result1.json_data is not None and "objects" in canvas_result1.json_data and len(canvas_result1.json_data["objects"]) > 0:
            last_obj = canvas_result1.json_data["objects"][-1]
            x = int(last_obj.get("left", 0))
            y = int(last_obj.get("top", 0))
            # Extraire la couleur du pixel sélectionné
            pixel_color = combined_main_corr1[y, x]  # en BGR
            pixel_color_rgb = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2RGB)[0,0]
            selected_color_hex = '#%02x%02x%02x' % tuple(pixel_color_rgb)
            st.write("Couleur sélectionnée :", selected_color_hex)
            # Paramètres pupette pour Correction 1
            pupette_tol1 = st.slider("Tolérance Pupette (Corr 1)", 0, 100, 30, key="pupette_tol_corr1")
            pupette_c1 = st.slider("Cyan Pupette (Corr 1)", -100, 100, 0, key="pupette_c_corr1")
            pupette_m1 = st.slider("Magenta Pupette (Corr 1)", -100, 100, 0, key="pupette_m_corr1")
            pupette_y1 = st.slider("Jaune Pupette (Corr 1)", -100, 100, 0, key="pupette_y_corr1")
            pupette_k1 = st.slider("Noir Pupette (Corr 1)", -100, 100, 0, key="pupette_k_corr1")
            pupette_method1 = st.radio("Méthode Pupette (Corr 1)", options=["Relative", "Absolute"], index=0, key="pupette_method_corr1")
            mask_pupette1 = get_specific_color_mask(combined_main_corr1, selected_color_hex, pupette_tol1)
            pupette_result_corr1 = apply_selective_color(combined_main_corr1, mask_pupette1,
                                                         pupette_c1, pupette_m1, pupette_y1, pupette_k1,
                                                         pupette_method1)
            st.subheader("Résultat Pupette - Correction 1")
            st.image(cv2.cvtColor(pupette_result_corr1, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.write("Cliquez sur l'image pour sélectionner une couleur.")

    # Export pour Correction 1 (exporté directement sous la section)
    st.markdown("#### Export Correction 1")
    export_text_corr1 = "Export pour Correction 1\n\n"
    export_text_corr1 += "Paramètres par Couche (Corr 1):\n"
    for layer in layer_names:
        p = layer_params_corr1.get(layer)
        if p and p["active"]:
            export_text_corr1 += f"  - Couche {layer}: Cyan: {p['c_adj']}, Magenta: {p['m_adj']}, Jaune: {p['y_adj']}, Noir: {p['k_adj']}, Méthode: {p['method']}\n"
    if classic_active_corr1:
        export_text_corr1 += f"\nModifs Classiques: Luminosité: {brightness_corr1}, Contraste: {contrast_corr1}, Saturation: {saturation_corr1}, Gamma: {gamma_corr1}\n"
    if use_pupette_corr1 and pupette_result_corr1 is not None:
        export_text_corr1 += f"\nOutil Pupette:\n  Couleur sélectionnée: {selected_color_hex}, Tolérance: {pupette_tol1}\n"
        export_text_corr1 += f"  Cyan: {pupette_c1}, Magenta: {pupette_m1}, Jaune: {pupette_y1}, Noir: {pupette_k1}, Méthode: {pupette_method1}\n"
    pdf1 = FPDF()
    pdf1.add_page()
    pdf1.set_font("Arial", size=12)
    pdf1.multi_cell(0, 10, export_text_corr1)
    temp_img_file1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_img_file1.name, combined_main_corr1)
    pdf1.image(temp_img_file1.name, x=10, y=pdf1.get_y() + 10, w=pdf1.w - 20)
    os.unlink(temp_img_file1.name)
    if use_pupette_corr1 and pupette_result_corr1 is not None:
        pdf1.add_page()
        pdf1.set_font("Arial", size=12)
        pdf1.cell(0, 10, "Résultat Pupette - Correction 1", ln=True)
        temp_img_file1b = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_img_file1b.name, pupette_result_corr1)
        pdf1.image(temp_img_file1b.name, x=10, y=pdf1.get_y() + 10, w=pdf1.w - 20)
        os.unlink(temp_img_file1b.name)
    pdf1_bytes = pdf1.output(dest="S").encode("latin-1")
    st.download_button("Télécharger Rapport Correction 1 (PDF)", data=pdf1_bytes, file_name="export_corr1.pdf", mime="application/pdf")
    st.download_button("Télécharger Paramètres Correction 1 (TXT)", data=export_text_corr1, file_name="export_corr1.txt", mime="text/plain")

    ############################
    # CORRECTION 2 (si applicable)
    ############################
    if correction_sequence in ["Correction 2 (en chaîne)", "Correction 3 (en chaîne)"]:
        # On part du résultat de Correction 1
        corr1_source = combined_main_corr1.copy()
        layer_results_corr2 = {}
        for layer in layer_names:
            if layer_params_corr2[layer]["active"]:
                mask = get_color_mask(corr1_source, layer)
                params = layer_params_corr2[layer]
                result = apply_selective_color(corr1_source, mask, params["c_adj"], params["m_adj"],
                                               params["y_adj"], params["k_adj"], params["method"])
                layer_results_corr2[layer] = {"mask": mask, "result": result}
        combined_main_corr2 = corr1_source.copy()
        for layer in layer_results_corr2:
            msk = layer_results_corr2[layer]["mask"]
            combined_main_corr2[msk != 0] = layer_results_corr2[layer]["result"][msk != 0]
        if classic_active_corr2:
            cf2 = contrast_corr2 / 100.0; sf2 = saturation_corr2 / 100.0; gf2 = gamma_corr2 / 100.0
            combined_main_corr2 = apply_classic_modifications(combined_main_corr2, brightness=brightness_corr2,
                                                              contrast=cf2, saturation=sf2, gamma=gf2)
        st.header("Correction 2")
        st.image(cv2.cvtColor(combined_main_corr2, cv2.COLOR_BGR2RGB), use_container_width=True)
        # Outil Pupette pour Correction 2
        st.markdown("### Outil Pupette - Correction 2")
        use_pupette_corr2 = st.checkbox("Activer l'outil Pupette pour Correction 2", key="use_pupette_corr2")
        pupette_result_corr2 = None
        if use_pupette_corr2:
            canvas_result2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#ff0000",
                background_image=Image.fromarray(cv2.cvtColor(combined_main_corr2, cv2.COLOR_BGR2RGB)),
                update_streamlit=True,
                height=combined_main_corr2.shape[0],
                width=combined_main_corr2.shape[1],
                drawing_mode="point",
                key="canvas_corr2"
            )
            if canvas_result2.json_data is not None and "objects" in canvas_result2.json_data and len(canvas_result2.json_data["objects"]) > 0:
                last_obj = canvas_result2.json_data["objects"][-1]
                x2 = int(last_obj.get("left", 0))
                y2 = int(last_obj.get("top", 0))
                pixel_color2 = combined_main_corr2[y2, x2]
                pixel_color2_rgb = cv2.cvtColor(np.uint8([[pixel_color2]]), cv2.COLOR_BGR2RGB)[0,0]
                selected_color_hex2 = '#%02x%02x%02x' % tuple(pixel_color2_rgb)
                st.write("Couleur sélectionnée :", selected_color_hex2)
                pupette_tol2 = st.slider("Tolérance Pupette (Corr 2)", 0, 100, 30, key="pupette_tol_corr2")
                pupette_c2 = st.slider("Cyan Pupette (Corr 2)", -100, 100, 0, key="pupette_c_corr2")
                pupette_m2 = st.slider("Magenta Pupette (Corr 2)", -100, 100, 0, key="pupette_m_corr2")
                pupette_y2 = st.slider("Jaune Pupette (Corr 2)", -100, 100, 0, key="pupette_y_corr2")
                pupette_k2 = st.slider("Noir Pupette (Corr 2)", -100, 100, 0, key="pupette_k_corr2")
                pupette_method2 = st.radio("Méthode Pupette (Corr 2)", options=["Relative", "Absolute"], index=0, key="pupette_method_corr2")
                mask_pupette2 = get_specific_color_mask(combined_main_corr2, selected_color_hex2, pupette_tol2)
                pupette_result_corr2 = apply_selective_color(combined_main_corr2, mask_pupette2,
                                                             pupette_c2, pupette_m2, pupette_y2, pupette_k2,
                                                             pupette_method2)
                st.subheader("Résultat Pupette - Correction 2")
                st.image(cv2.cvtColor(pupette_result_corr2, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.write("Cliquez sur l'image pour sélectionner une couleur.")
        st.markdown("#### Export Correction 2")
        export_text_corr2 = "Export pour Correction 2\n\n"
        export_text_corr2 += "Paramètres par Couche (Corr 2):\n"
        for layer in layer_names:
            p = layer_params_corr2.get(layer)
            if p and p["active"]:
                export_text_corr2 += f"  - Couche {layer}: Cyan: {p['c_adj']}, Magenta: {p['m_adj']}, Jaune: {p['y_adj']}, Noir: {p['k_adj']}, Méthode: {p['method']}\n"
        if classic_active_corr2:
            export_text_corr2 += f"\nModifs Classiques: Luminosité: {brightness_corr2}, Contraste: {contrast_corr2}, Saturation: {saturation_corr2}, Gamma: {gamma_corr2}\n"
        if use_pupette_corr2 and pupette_result_corr2 is not None:
            export_text_corr2 += f"\nOutil Pupette:\n  Couleur sélectionnée: {selected_color_hex2}, Tolérance: {pupette_tol2}\n"
            export_text_corr2 += f"  Cyan: {pupette_c2}, Magenta: {pupette_m2}, Jaune: {pupette_y2}, Noir: {pupette_k2}, Méthode: {pupette_method2}\n"
        pdf2 = FPDF()
        pdf2.add_page()
        pdf2.set_font("Arial", size=12)
        pdf2.multi_cell(0, 10, export_text_corr2)
        temp_img_file2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_img_file2.name, combined_main_corr2)
        pdf2.image(temp_img_file2.name, x=10, y=pdf2.get_y() + 10, w=pdf2.w - 20)
        os.unlink(temp_img_file2.name)
        if use_pupette_corr2 and pupette_result_corr2 is not None:
            pdf2.add_page()
            pdf2.set_font("Arial", size=12)
            pdf2.cell(0, 10, "Résultat Pupette - Correction 2", ln=True)
            temp_img_file2b = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_img_file2b.name, pupette_result_corr2)
            pdf2.image(temp_img_file2b.name, x=10, y=pdf2.get_y() + 10, w=pdf2.w - 20)
            os.unlink(temp_img_file2b.name)
        pdf2_bytes = pdf2.output(dest="S").encode("latin-1")
        st.download_button("Télécharger Rapport Correction 2 (PDF)", data=pdf2_bytes, file_name="export_corr2.pdf", mime="application/pdf")
        st.download_button("Télécharger Paramètres Correction 2 (TXT)", data=export_text_corr2, file_name="export_corr2.txt", mime="text/plain")

    ############################
    # CORRECTION 3 (si applicable)
    ############################
    if correction_sequence == "Correction 3 (en chaîne)":
        corr2_source = combined_main_corr2.copy()
        layer_results_corr3 = {}
        for layer in layer_names:
            if layer_params_corr3[layer]["active"]:
                mask = get_color_mask(corr2_source, layer)
                params = layer_params_corr3[layer]
                result = apply_selective_color(corr2_source, mask, params["c_adj"], params["m_adj"],
                                               params["y_adj"], params["k_adj"], params["method"])
                layer_results_corr3[layer] = {"mask": mask, "result": result}
        combined_main_corr3 = corr2_source.copy()
        for layer in layer_results_corr3:
            msk = layer_results_corr3[layer]["mask"]
            combined_main_corr3[msk != 0] = layer_results_corr3[layer]["result"][msk != 0]
        if classic_active_corr3:
            cf3 = contrast_corr3 / 100.0; sf3 = saturation_corr3 / 100.0; gf3 = gamma_corr3 / 100.0
            combined_main_corr3 = apply_classic_modifications(combined_main_corr3, brightness=brightness_corr3,
                                                              contrast=cf3, saturation=sf3, gamma=gf3)
        st.header("Correction 3")
        st.image(cv2.cvtColor(combined_main_corr3, cv2.COLOR_BGR2RGB), use_container_width=True)
        # Outil Pupette pour Correction 3
        st.markdown("### Outil Pupette - Correction 3")
        use_pupette_corr3 = st.checkbox("Activer l'outil Pupette pour Correction 3", key="use_pupette_corr3")
        pupette_result_corr3 = None
        if use_pupette_corr3:
            canvas_result3 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#ff0000",
                background_image=Image.fromarray(cv2.cvtColor(combined_main_corr3, cv2.COLOR_BGR2RGB)),
                update_streamlit=True,
                height=combined_main_corr3.shape[0],
                width=combined_main_corr3.shape[1],
                drawing_mode="point",
                key="canvas_corr3"
            )
            if canvas_result3.json_data is not None and "objects" in canvas_result3.json_data and len(canvas_result3.json_data["objects"]) > 0:
                last_obj = canvas_result3.json_data["objects"][-1]
                x3 = int(last_obj.get("left", 0))
                y3 = int(last_obj.get("top", 0))
                pixel_color3 = combined_main_corr3[y3, x3]
                pixel_color3_rgb = cv2.cvtColor(np.uint8([[pixel_color3]]), cv2.COLOR_BGR2RGB)[0,0]
                selected_color_hex3 = '#%02x%02x%02x' % tuple(pixel_color3_rgb)
                st.write("Couleur sélectionnée :", selected_color_hex3)
                pupette_tol3 = st.slider("Tolérance Pupette (Corr 3)", 0, 100, 30, key="pupette_tol_corr3")
                pupette_c3 = st.slider("Cyan Pupette (Corr 3)", -100, 100, 0, key="pupette_c_corr3")
                pupette_m3 = st.slider("Magenta Pupette (Corr 3)", -100, 100, 0, key="pupette_m_corr3")
                pupette_y3 = st.slider("Jaune Pupette (Corr 3)", -100, 100, 0, key="pupette_y_corr3")
                pupette_k3 = st.slider("Noir Pupette (Corr 3)", -100, 100, 0, key="pupette_k_corr3")
                pupette_method3 = st.radio("Méthode Pupette (Corr 3)", options=["Relative", "Absolute"], index=0, key="pupette_method_corr3")
                mask_pupette3 = get_specific_color_mask(combined_main_corr3, selected_color_hex3, pupette_tol3)
                pupette_result_corr3 = apply_selective_color(combined_main_corr3, mask_pupette3,
                                                             pupette_c3, pupette_m3, pupette_y3, pupette_k3,
                                                             pupette_method3)
                st.subheader("Résultat Pupette - Correction 3")
                st.image(cv2.cvtColor(pupette_result_corr3, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.write("Cliquez sur l'image pour sélectionner une couleur.")
        st.markdown("#### Export Correction 3")
        export_text_corr3 = "Export pour Correction 3\n\n"
        export_text_corr3 += "Paramètres par Couche (Corr 3):\n"
        for layer in layer_names:
            p = layer_params_corr3.get(layer)
            if p and p["active"]:
                export_text_corr3 += f"  - Couche {layer}: Cyan: {p['c_adj']}, Magenta: {p['m_adj']}, Jaune: {p['y_adj']}, Noir: {p['k_adj']}, Méthode: {p['method']}\n"
        if classic_active_corr3:
            export_text_corr3 += f"\nModifs Classiques: Luminosité: {brightness_corr3}, Contraste: {contrast_corr3}, Saturation: {saturation_corr3}, Gamma: {gamma_corr3}\n"
        if use_pupette_corr3 and pupette_result_corr3 is not None:
            export_text_corr3 += f"\nOutil Pupette:\n  Couleur sélectionnée: {selected_color_hex3}, Tolérance: {pupette_tol3}\n"
            export_text_corr3 += f"  Cyan: {pupette_c3}, Magenta: {pupette_m3}, Jaune: {pupette_y3}, Noir: {pupette_k3}, Méthode: {pupette_method3}\n"
        pdf3 = FPDF()
        pdf3.add_page()
        pdf3.set_font("Arial", size=12)
        pdf3.multi_cell(0, 10, export_text_corr3)
        temp_img_file3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_img_file3.name, combined_main_corr3)
        pdf3.image(temp_img_file3.name, x=10, y=pdf3.get_y() + 10, w=pdf3.w - 20)
        os.unlink(temp_img_file3.name)
        if use_pupette_corr3 and pupette_result_corr3 is not None:
            pdf3.add_page()
            pdf3.set_font("Arial", size=12)
            pdf3.cell(0, 10, "Résultat Pupette - Correction 3", ln=True)
            temp_img_file3b = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_img_file3b.name, pupette_result_corr3)
            pdf3.image(temp_img_file3b.name, x=10, y=pdf3.get_y() + 10, w=pdf3.w - 20)
            os.unlink(temp_img_file3b.name)
        pdf3_bytes = pdf3.output(dest="S").encode("latin-1")
        st.download_button("Télécharger Rapport Correction 3 (PDF)", data=pdf3_bytes, file_name="export_corr3.pdf", mime="application/pdf")
        st.download_button("Télécharger Paramètres Correction 3 (TXT)", data=export_text_corr3, file_name="export_corr3.txt", mime="text/plain")
else:
    st.write("Veuillez téléverser une image pour commencer.")
