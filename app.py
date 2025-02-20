import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors

def main():
    st.title("Remplissage de Polygones – Couleur variable")
    st.write(
        """
        Cet outil permet de segmenter puis de remplir des blocs de n'importe quelle couleur 
        (rouge, jaune, noir, etc.) en les transformant en polygones pleins.
        """
    )

    # --- Chargement de l'image ---
    uploaded_file = st.file_uploader("Téléversez votre image (JPEG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        st.subheader("Aperçu de l'image originale")
        st.image(pil_img, use_container_width=True)

        # --- Sélecteur de couleur + tolérances ---
        st.sidebar.title("Sélection de la couleur cible")
        picked_color = st.sidebar.color_picker("Choisissez la couleur principale à segmenter", "#00FFFF")
        
        # Convertir la couleur hex en (R,G,B)
        rgb_float = mcolors.to_rgb(picked_color)  # Ex. "#ff00ff" => (1.0, 0.0, 1.0)
        r, g, b = [int(x*255) for x in rgb_float]
        
        # Conversion de (B, G, R) en HSV (OpenCV attend BGR)
        test_patch = np.uint8([[[b, g, r]]])  # patch 1x1
        hsv_patch = cv2.cvtColor(test_patch, cv2.COLOR_BGR2HSV)[0][0]
        hue_center, sat_center, val_center = hsv_patch
        
        st.sidebar.write("Couleur sélectionnée en HSV : ", (hue_center, sat_center, val_center))
        
        # Tolérances sur H, S, V
        st.sidebar.title("Tolérances HSV")
        h_tolerance = st.sidebar.slider("Tolérance Hue", 0, 179, 10)
        s_tolerance = st.sidebar.slider("Tolérance Saturation", 0, 255, 50)
        v_tolerance = st.sidebar.slider("Tolérance Value", 0, 255, 50)

        # Calcul des bornes HSV (en tenant compte des limites [0,179] pour H et [0,255] pour S,V)
        lower_h = max(hue_center - h_tolerance, 0)
        upper_h = min(hue_center + h_tolerance, 179)
        lower_s = max(sat_center - s_tolerance, 0)
        upper_s = min(sat_center + s_tolerance, 255)
        lower_v = max(val_center - v_tolerance, 0)
        upper_v = min(val_center + v_tolerance, 255)

        lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

        # --- Paramètres morphologiques (optionnel) ---
        st.sidebar.title("Paramètres Morphologiques")
        use_morph = st.sidebar.checkbox("Activer un closing morphologique", value=False)
        morph_kernel = st.sidebar.slider("Taille du kernel (closing)", 1, 20, 5)

        # Conversion en HSV et seuillage
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Optionnel : opération morpho "closing" pour boucher les petits trous
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- Détection et remplissage des contours externes ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # On crée un masque "rempli"
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, color=255, thickness=-1)

        # On reconstruit une image BGR finale : fond blanc
        result_bgr = np.ones_like(img_bgr) * 255

        # Couleur cible en BGR (mêmes R, G, B inversés)
        # picked_color était en #RRGGBB, on l'a en (r,g,b)
        # => BGR = (b, g, r)
        color_bgr = (b, g, r)

        # Là où filled_mask == 255, on applique la couleur sélectionnée
        result_bgr[filled_mask == 255] = color_bgr

        # --- Affichage du résultat ---
        st.subheader("Image avec polygones remplis")
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        # --- Option de téléchargement du résultat ---
        download_col, _ = st.columns([1,3])
        with download_col:
            # Convertir l'image finale en PNG
            out_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            pil_out = Image.fromarray(out_rgb)
            st.download_button(
                label="Télécharger l'image traitée",
                data=pil_to_bytes(pil_out),
                file_name="polygones_remplis.png",
                mime="image/png"
            )
    else:
        st.write("Veuillez téléverser une image pour commencer.")

def pil_to_bytes(img_pil):
    """Convertit un objet PIL en bytes PNG."""
    import io
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

if __name__ == "__main__":
    main()
