import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

def main():
    st.title("Remplissage de Polygones – Couleur Automatique ou Manuelle")

    # 1) Téléversement de l'image
    uploaded_file = st.file_uploader("Téléversez votre image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        st.subheader("Aperçu de l'image originale")
        st.image(pil_img, use_container_width=True)

        # 2) Mode automatique ou manuel ?
        st.sidebar.title("Choix de la couleur")
        auto_mode = st.sidebar.checkbox("Détection automatique de la couleur dominante (hors blanc)", value=True)

        # 3) Détection automatique ou color picker
        if auto_mode:
            # On détecte la couleur dominante (hors zones blanches)
            dominant_bgr = get_dominant_color(img_bgr)
            # Convertit BGR -> RGB -> hex (pour l'afficher à titre indicatif)
            b, g, r = dominant_bgr
            hex_auto = "#{:02x}{:02x}{:02x}".format(r, g, b)
            st.sidebar.write(f"Couleur dominante détectée : {hex_auto}")
            
            # Conversion BGR->HSV pour en extraire (hue_center, sat_center, val_center)
            hsv_patch = cv2.cvtColor(np.uint8([[[b,g,r]]]), cv2.COLOR_BGR2HSV)[0][0]
            hue_center, sat_center, val_center = hsv_patch
        else:
            # Mode manuel : color picker
            picked_color = st.sidebar.color_picker("Couleur cible", "#FFFF00")  # par défaut jaune
            # Convertit hex -> (r,g,b)
            rgb_float = mcolors.to_rgb(picked_color)  # ex : "#FF00FF" => (1.0, 0.0, 1.0)
            r, g, b = [int(x*255) for x in rgb_float]
            # Conversion BGR->HSV
            hsv_patch = cv2.cvtColor(np.uint8([[[b,g,r]]]), cv2.COLOR_BGR2HSV)[0][0]
            hue_center, sat_center, val_center = hsv_patch

        # 4) Tolérances HSV
        st.sidebar.title("Tolérances HSV")
        h_tolerance = st.sidebar.slider("Tolérance Hue", 0, 179, 10)
        s_tolerance = st.sidebar.slider("Tolérance Saturation", 0, 255, 50)
        v_tolerance = st.sidebar.slider("Tolérance Value", 0, 255, 50)

        # Calcul des bornes
        lower_h = max(hue_center - h_tolerance, 0)
        upper_h = min(hue_center + h_tolerance, 179)
        lower_s = max(sat_center - s_tolerance, 0)
        upper_s = min(sat_center + s_tolerance, 255)
        lower_v = max(val_center - v_tolerance, 0)
        upper_v = min(val_center + v_tolerance, 255)

        lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

        # 5) Paramètres morpho (pour boucher les petits trous)
        st.sidebar.title("Paramètres Morphologiques")
        use_morph = st.sidebar.checkbox("Activer un closing morphologique", value=False)
        morph_kernel = st.sidebar.slider("Taille du kernel (closing)", 1, 20, 5)

        # Conversion en HSV et seuillage
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Optionnel : morpho
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 6) Détection & remplissage des contours externes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, color=255, thickness=-1)

        # 7) Construction de l'image résultat
        result_bgr = np.full_like(img_bgr, 255)  # fond blanc

        # Couleur de remplissage : si auto_mode => dominant_bgr, sinon => la couleur manuelle (b,g,r)
        if auto_mode:
            fill_color_bgr = dominant_bgr
        else:
            fill_color_bgr = (b, g, r)

        result_bgr[filled_mask == 255] = fill_color_bgr

        # 8) Affichage
        st.subheader("Image avec polygones remplis")
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, use_container_width=True)

        # 9) Téléchargement
        st.download_button(
            label="Télécharger l'image traitée",
            data=pil_to_bytes(Image.fromarray(result_rgb)),
            file_name="polygones_remplis.png",
            mime="image/png"
        )
    else:
        st.write("Veuillez téléverser une image pour commencer.")

def pil_to_bytes(img_pil):
    """Convertit un objet PIL en bytes PNG."""
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def get_dominant_color(img_bgr):
    """
    Renvoie la couleur dominante (en BGR) hors zones blanches.
    Stratégie :
      - On redimensionne l'image pour accélérer.
      - On retire les pixels proches du blanc (>= 240,240,240).
      - On applique un KMeans (k=1) sur le reste.
      - S'il n'y a pas assez de pixels, on renvoie (0,0,0).
    """
    # Redimensionne l'image pour accélérer (ex: 200px max)
    h, w = img_bgr.shape[:2]
    scale = 200 / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    # Convertir en array Nx3
    pixels = img_small.reshape((-1, 3))
    # Filtrer les pixels quasi blancs
    mask_non_white = np.where(np.all(pixels < [240,240,240], axis=1))[0]
    if len(mask_non_white) < 10:
        # S'il n'y a pas assez de pixels hors blanc, on renvoie un noir par défaut
        return (0,0,0)

    # On garde les pixels non-blancs
    data = pixels[mask_non_white].astype(np.float32)

    # KMeans (k=1) => un seul cluster
    kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto')
    kmeans.fit(data)
    center = kmeans.cluster_centers_[0]
    # Convertit en BGR int
    bgr = tuple([int(c) for c in center])
    return bgr

if __name__ == "__main__":
    main()
