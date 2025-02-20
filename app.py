import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import tempfile, os
# >>> On n'utilise plus streamlit_drawable_canvas
# from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------------------------------------------------------
# 1) Définir les couches prédéfinies (approximation HSV)
# ----------------------------------------------------------------------------
predef_layers = ["Rouges", "Jaunes", "Verts", "Cyans", "Bleus", "Magentas", "Blancs", "Neutres", "Noirs"]

# Pour chaque couche prédéfinie, on définit des plages HSV (sauf pour les zones spéciales)
color_ranges = {
    "Rouges":   [((0, 50, 50), (10, 255, 255)),
                 ((170, 50, 50), (180, 255, 255))],
    "Jaunes":   [((20, 50, 50), (35, 255, 255))],
    "Verts":    [((35, 50, 50), (85, 255, 255))],
    "Cyans":    [((85, 50, 50), (100, 255, 255))],
    "Bleus":    [((100, 50, 50), (130, 255, 255))],
    "Magentas": [((130, 50, 50), (170, 255, 255))],
    # Pour Blancs, Neutres, Noirs, on se base sur la luminosité/saturation
    "Blancs":   "whites",
    "Neutres":  "neutrals",
    "Noirs":    "blacks"
}

# ----------------------------------------------------------------------------
# 2) Fonctions de conversion RGB <-> CMYK (simplifiées)
# ----------------------------------------------------------------------------
def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100
    r_, g_, b_ = r/255.0, g/255.0, b/255.0
    k = 1 - max(r_, g_, b_)
    c = (1 - r_ - k) / (1 - k + 1e-8)
    m = (1 - g_ - k) / (1 - k + 1e-8)
    y = (1 - b_ - k) / (1 - k + 1e-8)
    return (c*100, m*100, y*100, k*100)

def cmyk_to_rgb(c, m, y, k):
    C, M, Y, K = c/100.0, m/100.0, y/100.0, k/100.0
    r = 1 - min(1, C+K)
    g = 1 - min(1, M+K)
    b = 1 - min(1, Y+K)
    return (int(r*255), int(g*255), int(b*255))

# ----------------------------------------------------------------------------
# 3) Masques pour zones spéciales et pour la couche pupette (couleur personnalisée)
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
        temp = cv2.inRange(img_hsv, lower, upper)
        mask = cv2.bitwise_or(mask, temp)
    return mask

def get_specific_color_mask(img_bgr, target_hex, tolerance):
    # Convertir le code hex en RGB puis en BGR
    target_rgb = tuple(int(target_hex.lstrip('#')[i:i+2], 16) for i in (0,2,4))
    target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    target_array = np.full(img_hsv[:,:,0].shape, target_hsv[0], dtype=np.uint8)
    diff = cv2.absdiff(img_hsv[:,:,0], target_array)
    diff = np.minimum(diff, 180 - diff)
    mask = cv2.inRange(diff, 0, tolerance)
    return mask

# ----------------------------------------------------------------------------
# 4) Application de la correction sélective sur un masque
# ----------------------------------------------------------------------------
def apply_selective_color(img_bgr, mask, c_adj, m_adj, y_adj, k_adj, method="Relative"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_img = img_rgb.copy()
    h, w = out_img.shape[:2]
    for i in range(h):
        for j in range(w):
            if mask[i, j] != 0:
                r, g, b = out_img[i, j]
                c, m, y, k = rgb_to_cmyk(r, g, b)
                if method=="Relative":
                    c += (c_adj/100.0)*c
                    m += (m_adj/100.0)*m
                    y += (y_adj/100.0)*y
                    k += (k_adj/100.0)*k
                else:
                    c += c_adj; m += m_adj; y += y_adj; k += k_adj
                c = max(0, min(100, c))
                m = max(0, min(100, m))
                y = max(0, min(100, y))
                k = max(0, min(100, k))
                out_img[i, j] = cmyk_to_rgb(c, m, y, k)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

# ----------------------------------------------------------------------------
# 5) Application des modifications classiques (ajustement global)
# ----------------------------------------------------------------------------
def apply_classic_modifications(img, brightness=0, contrast=1.0, saturation=1.0, gamma=1.0):
    img = img.astype(np.float32)
    # Luminosité & contraste
    img = img*contrast + brightness
    img = np.clip(img,0,255).astype(np.uint8)
    # Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= saturation
    hsv[...,1] = np.clip(hsv[...,1],0,255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Gamma
    if gamma != 1.0:
        invGamma = 1.0/gamma
        table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(256)]).astype("uint8")
        img = cv2.LUT(img, table)
    return img

# ----------------------------------------------------------------------------
# 6) Barre latérale – Paramètres de correction
# ----------------------------------------------------------------------------
st.sidebar.title("Paramètres de Correction")

# Choix de la séquence de corrections
correction_sequence = st.sidebar.radio(
    "Séquence de corrections",
    options=["Correction 1 seule", "Correction 2 (en chaîne)", "Correction 3 (en chaîne)"]
)

# -- Correction 1 --
st.sidebar.subheader("Correction 1")
with st.sidebar.expander("Paramètres par Couche (Corr 1)"):
    layer_params_corr1 = {}
    for layer in predef_layers:
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
            layer_params_corr1[layer] = {
                "active": active, 
                "c_adj": c_adj, 
                "m_adj": m_adj,
                "y_adj": y_adj, 
                "k_adj": k_adj, 
                "method": method
            }

with st.sidebar.expander("Modifications Classiques (Corr 1)"):
    classic_active_corr1 = st.checkbox("Activer modifs classiques", key="classic_active_corr1")
    if classic_active_corr1:
        brightness_corr1 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr1")
        contrast_corr1 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr1")
        saturation_corr1 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr1")
        gamma_corr1 = st.slider("Gamma", 50, 150, 100, key="gamma_corr1")
    else:
        brightness_corr1, contrast_corr1, saturation_corr1, gamma_corr1 = 0,100,100,100

# -- Correction 2 --
if correction_sequence in ["Correction 2 (en chaîne)", "Correction 3 (en chaîne)"]:
    st.sidebar.subheader("Correction 2")
    with st.sidebar.expander("Paramètres par Couche (Corr 2)"):
        layer_params_corr2 = {}
        for layer in predef_layers:
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
                layer_params_corr2[layer] = {
                    "active": active, 
                    "c_adj": c_adj, 
                    "m_adj": m_adj,
                    "y_adj": y_adj, 
                    "k_adj": k_adj, 
                    "method": method
                }
    with st.sidebar.expander("Modifications Classiques (Corr 2)"):
        classic_active_corr2 = st.checkbox("Activer modifs classiques", key="classic_active_corr2")
        if classic_active_corr2:
            brightness_corr2 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr2")
            contrast_corr2 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr2")
            saturation_corr2 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr2")
            gamma_corr2 = st.slider("Gamma", 50, 150, 100, key="gamma_corr2")
        else:
            brightness_corr2, contrast_corr2, saturation_corr2, gamma_corr2 = 0,100,100,100

# -- Correction 3 --
if correction_sequence == "Correction 3 (en chaîne)":
    st.sidebar.subheader("Correction 3")
    with st.sidebar.expander("Paramètres par Couche (Corr 3)"):
        layer_params_corr3 = {}
        for layer in predef_layers:
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
                layer_params_corr3[layer] = {
                    "active": active, 
                    "c_adj": c_adj, 
                    "m_adj": m_adj,
                    "y_adj": y_adj, 
                    "k_adj": k_adj, 
                    "method": method
                }
    with st.sidebar.expander("Modifications Classiques (Corr 3)"):
        classic_active_corr3 = st.checkbox("Activer modifs classiques", key="classic_active_corr3")
        if classic_active_corr3:
            brightness_corr3 = st.slider("Luminosité", -100, 100, 0, key="brightness_corr3")
            contrast_corr3 = st.slider("Contraste (%)", 50, 150, 100, key="contrast_corr3")
            saturation_corr3 = st.slider("Saturation (%)", 50, 150, 100, key="saturation_corr3")
            gamma_corr3 = st.slider("Gamma", 50, 150, 100, key="gamma_corr3")
        else:
            brightness_corr3, contrast_corr3, saturation_corr3, gamma_corr3 = 0,100,100,100

# ----------------------------------------------------------------------------
# 6bis) Couche Pupette – Couche personnalisée via sélection sur l'image
# ----------------------------------------------------------------------------
custom_layer_enabled = st.sidebar.checkbox("Activer Couche Pupette (personnalisée)", key="active_custom")
if custom_layer_enabled:
    custom_tolerance = st.sidebar.slider("Tolérance Couche Pupette", 0, 100, 30, key="custom_tolerance")
    custom_c_adj = st.sidebar.slider("Cyan (Pupette)", -100, 100, 0, key="custom_c")
    custom_m_adj = st.sidebar.slider("Magenta (Pupette)", -100, 100, 0, key="custom_m")
    custom_y_adj = st.sidebar.slider("Jaune (Pupette)", -100, 100, 0, key="custom_y")
    custom_k_adj = st.sidebar.slider("Noir (Pupette)", -100, 100, 0, key="custom_k")
    custom_method = st.sidebar.radio("Méthode (Pupette)", options=["Relative", "Absolute"], index=0, key="custom_method")

# ----------------------------------------------------------------------------
# 7) Téléversement et affichage de l'image originale
# ----------------------------------------------------------------------------
st.title("Correction Sélective – Mode Multicouche Dynamique")
uploaded_file = st.file_uploader("Téléversez une image (JPEG/PNG)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    st.subheader("Image originale")
    st.image(pil_img, use_container_width=True)

    # Si la couche pupette est activée, on propose la sélection de couleur via streamlit_image_coordinates
    custom_color = None
    if custom_layer_enabled:
        st.markdown("### Outil Pipette – Couche personnalisée")
        # On affiche l'image cliquable pour récupérer les coordonnées
        coords = streamlit_image_coordinates(
            "Cliquez sur l'image pour sélectionner un pixel",
            pil_img,
            key="custom_pipette"
        )
        if coords is not None:
            x_custom = int(coords["x"])
            y_custom = int(coords["y"])
            # Sécuriser si jamais on clique en dehors
            if 0 <= x_custom < original_bgr.shape[1] and 0 <= y_custom < original_bgr.shape[0]:
                pixel = original_bgr[y_custom, x_custom]  # BGR
                pixel_rgb = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2RGB)[0,0]
                custom_color = '#%02x%02x%02x' % tuple(pixel_rgb)
                st.write("Couleur pupette sélectionnée :", custom_color)

    # ----------------------------------------------------------------------------
    # 8) Traitement des corrections (chaîne)
    # ----------------------------------------------------------------------------
    def process_correction(source_img, layer_params):
        """Applique les corrections par couche sur source_img à partir du dictionnaire layer_params."""
        combined = source_img.copy()
        for layer in predef_layers:
            if layer_params.get(layer, {}).get("active", False):
                msk = get_color_mask(source_img, layer)
                p = layer_params[layer]
                res = apply_selective_color(
                    source_img, msk,
                    p["c_adj"], p["m_adj"], p["y_adj"], p["k_adj"],
                    p["method"]
                )
                combined[msk != 0] = res[msk != 0]
        return combined

    def add_custom_layer(img):
        """Applique la couche 'pupette' si activée et une couleur sélectionnée."""
        if custom_layer_enabled and custom_color is not None:
            msk = get_specific_color_mask(img, custom_color, custom_tolerance)
            res = apply_selective_color(img, msk, custom_c_adj, custom_m_adj, custom_y_adj, custom_k_adj, custom_method)
            img[msk != 0] = res[msk != 0]
        return img

    # --- Correction 1 ---
    corr1 = process_correction(original_bgr, layer_params_corr1)
    if classic_active_corr1:
        corr1 = apply_classic_modifications(
            corr1,
            brightness=brightness_corr1,
            contrast=contrast_corr1/100.0,
            saturation=saturation_corr1/100.0,
            gamma=gamma_corr1/100.0
        )
    # Ajouter la couche pupette
    corr1 = add_custom_layer(corr1)
    st.header("Correction 1")
    st.image(cv2.cvtColor(corr1, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Export Correction 1
    export_text1 = "Export Correction 1\n\nParamètres par Couche:\n"
    for layer in predef_layers:
        p = layer_params_corr1.get(layer)
        if p and p["active"]:
            export_text1 += f" - Couche {layer}: Cyan {p['c_adj']}, Magenta {p['m_adj']}, Jaune {p['y_adj']}, Noir {p['k_adj']}, Méthode {p['method']}\n"
    if classic_active_corr1:
        export_text1 += f"\nModifs Classiques: Luminosité {brightness_corr1}, Contraste {contrast_corr1}, Saturation {saturation_corr1}, Gamma {gamma_corr1}\n"
    if custom_layer_enabled and custom_color is not None:
        export_text1 += f"\nCouche Pupette: Couleur {custom_color}, Tolérance {custom_tolerance}, "
        export_text1 += f"Cyan {custom_c_adj}, Magenta {custom_m_adj}, Jaune {custom_y_adj}, Noir {custom_k_adj}, Méthode {custom_method}\n"

    pdf1 = FPDF()
    pdf1.add_page()
    pdf1.set_font("Arial", size=12)
    pdf1.multi_cell(0, 10, export_text1)
    temp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file1.name, corr1)
    pdf1.image(temp_file1.name, x=10, y=pdf1.get_y()+10, w=pdf1.w-20)
    os.unlink(temp_file1.name)
    pdf1_bytes = pdf1.output(dest="S").encode("latin-1")
    st.download_button("Exporter Correction 1 (PDF)", data=pdf1_bytes, file_name="corr1.pdf", mime="application/pdf")
    st.download_button("Exporter Correction 1 (TXT)", data=export_text1, file_name="corr1.txt", mime="text/plain")

    # --- Correction 2 ---
    if correction_sequence in ["Correction 2 (en chaîne)", "Correction 3 (en chaîne)"]:
        corr2 = process_correction(corr1, layer_params_corr2)
        if classic_active_corr2:
            corr2 = apply_classic_modifications(
                corr2,
                brightness=brightness_corr2,
                contrast=contrast_corr2/100.0,
                saturation=saturation_corr2/100.0,
                gamma=gamma_corr2/100.0
            )
        corr2 = add_custom_layer(corr2)
        st.header("Correction 2")
        st.image(cv2.cvtColor(corr2, cv2.COLOR_BGR2RGB), use_container_width=True)

        export_text2 = "Export Correction 2\n\nParamètres par Couche:\n"
        for layer in predef_layers:
            p = layer_params_corr2.get(layer)
            if p and p["active"]:
                export_text2 += f" - Couche {layer}: Cyan {p['c_adj']}, Magenta {p['m_adj']}, Jaune {p['y_adj']}, Noir {p['k_adj']}, Méthode {p['method']}\n"
        if classic_active_corr2:
            export_text2 += f"\nModifs Classiques: Luminosité {brightness_corr2}, Contraste {contrast_corr2}, Saturation {saturation_corr2}, Gamma {gamma_corr2}\n"
        if custom_layer_enabled and custom_color is not None:
            export_text2 += f"\nCouche Pupette: Couleur {custom_color}, Tolérance {custom_tolerance}, "
            export_text2 += f"Cyan {custom_c_adj}, Magenta {custom_m_adj}, Jaune {custom_y_adj}, Noir {custom_k_adj}, Méthode {custom_method}\n"

        pdf2 = FPDF()
        pdf2.add_page()
        pdf2.set_font("Arial", size=12)
        pdf2.multi_cell(0, 10, export_text2)
        temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_file2.name, corr2)
        pdf2.image(temp_file2.name, x=10, y=pdf2.get_y()+10, w=pdf2.w-20)
        os.unlink(temp_file2.name)
        pdf2_bytes = pdf2.output(dest="S").encode("latin-1")
        st.download_button("Exporter Correction 2 (PDF)", data=pdf2_bytes, file_name="corr2.pdf", mime="application/pdf")
        st.download_button("Exporter Correction 2 (TXT)", data=export_text2, file_name="corr2.txt", mime="text/plain")

        # --- Correction 3 ---
        if correction_sequence == "Correction 3 (en chaîne)":
            corr3 = process_correction(corr2, layer_params_corr3)
            if classic_active_corr3:
                corr3 = apply_classic_modifications(
                    corr3,
                    brightness=brightness_corr3,
                    contrast=contrast_corr3/100.0,
                    saturation=saturation_corr3/100.0,
                    gamma=gamma_corr3/100.0
                )
            corr3 = add_custom_layer(corr3)
            st.header("Correction 3")
            st.image(cv2.cvtColor(corr3, cv2.COLOR_BGR2RGB), use_container_width=True)

            export_text3 = "Export Correction 3\n\nParamètres par Couche:\n"
            for layer in predef_layers:
                p = layer_params_corr3.get(layer)
                if p and p["active"]:
                    export_text3 += f" - Couche {layer}: Cyan {p['c_adj']}, Magenta {p['m_adj']}, Jaune {p['y_adj']}, Noir {p['k_adj']}, Méthode {p['method']}\n"
            if classic_active_corr3:
                export_text3 += f"\nModifs Classiques: Luminosité {brightness_corr3}, Contraste {contrast_corr3}, Saturation {saturation_corr3}, Gamma {gamma_corr3}\n"
            if custom_layer_enabled and custom_color is not None:
                export_text3 += f"\nCouche Pupette: Couleur {custom_color}, Tolérance {custom_tolerance}, "
                export_text3 += f"Cyan {custom_c_adj}, Magenta {custom_m_adj}, Jaune {custom_y_adj}, Noir {custom_k_adj}, Méthode {custom_method}\n"

            pdf3 = FPDF()
            pdf3.add_page()
            pdf3.set_font("Arial", size=12)
            pdf3.multi_cell(0, 10, export_text3)
            temp_file3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file3.name, corr3)
            pdf3.image(temp_file3.name, x=10, y=pdf3.get_y()+10, w=pdf3.w-20)
            os.unlink(temp_file3.name)
            pdf3_bytes = pdf3.output(dest="S").encode("latin-1")
            st.download_button("Exporter Correction 3 (PDF)", data=pdf3_bytes, file_name="corr3.pdf", mime="application/pdf")
            st.download_button("Exporter Correction 3 (TXT)", data=export_text3, file_name="corr3.txt", mime="text/plain")
else:
    st.write("Veuillez téléverser une image pour commencer.")
