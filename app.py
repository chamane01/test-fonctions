import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detect_dominant_color(img_bgr):
    """
    Détecte la couleur dominante de l'image en excluant les pixels quasiment blancs.
    Retourne la couleur dominante en BGR et en HSV.
    """
    # On exclut les pixels presque blancs (fond)
    seuil_blanc = 240
    mask_nonblanc = ~np.all(img_bgr >= seuil_blanc, axis=2)
    nonwhite_pixels = img_bgr[mask_nonblanc].reshape(-1, 3)
    
    if nonwhite_pixels.size == 0:
        # Si aucun pixel n'est détecté (cas extrême), on renvoie magenta par défaut
        dominant_bgr = np.array([255, 0, 255], dtype=np.uint8)
    else:
        # Appliquer k-means pour k=1 sur les pixels non-blancs
        pixels = np.float32(nonwhite_pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_bgr = center[0].astype(np.uint8)
    
    # Conversion en HSV
    dominant_bgr_reshaped = np.uint8([[dominant_bgr]])
    dominant_hsv = cv2.cvtColor(dominant_bgr_reshaped, cv2.COLOR_BGR2HSV)[0][0]
    return dominant_bgr, dominant_hsv

def main():
    st.title("Remplissage de Polygones – Détection de Couleur Dominante")
    st.write(
        """
        Cet outil permet de prendre une image où des objets sont d'une certaine couleur sur fond blanc 
        et de produire une image où chaque bloc est rempli, sans trous.
        La couleur utilisée est détectée automatiquement en tant que couleur dominante (hors fond blanc).
        """
    )

    # --- Chargement de l'image ---
    uploaded_file = st.file_uploader("Téléversez votre image (JPEG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        st.subheader("Aperçu de l'image originale")
        st.image(pil_img, use_container_width=True)
        
        # Détection de la couleur dominante (hors blanc)
        dominant_bgr, dominant_hsv = detect_dominant_color(img_bgr)
        h, s, v = int(dominant_hsv[0]), int(dominant_hsv[1]), int(dominant_hsv[2])
        
        # Définition d'une marge autour de la couleur dominante
        delta_h = 10
        delta_s = 50
        delta_v = 50
        lower = np.array([max(0, h - delta_h), max(0, s - delta_s), max(0, v - delta_v)], dtype=np.uint8)
        upper = np.array([min(179, h + delta_h), min(255, s + delta_s), min(255, v + delta_v)], dtype=np.uint8)
        
        # Affichage des informations sur la couleur dominante
        st.sidebar.title("Couleur dominante détectée")
        st.sidebar.write(f"Couleur dominante (BGR) : {tuple(dominant_bgr)}")
        st.sidebar.write(f"Couleur dominante (HSV) : {tuple(dominant_hsv)}")
        st.sidebar.write("Plage de seuillage automatique (HSV) :")
        st.sidebar.write(f"H: {lower[0]} - {upper[0]}")
        st.sidebar.write(f"S: {lower[1]} - {upper[1]}")
        st.sidebar.write(f"V: {lower[2]} - {upper[2]}")
        
        # --- Paramètres morphologiques (optionnel) ---
        st.sidebar.title("Paramètres Morphologiques")
        use_morph = st.sidebar.checkbox("Activer un closing morphologique (pour boucher les petits trous)", value=False)
        morph_kernel = st.sidebar.slider("Taille du kernel", 1, 20, 5)
        
        # Conversion en HSV et seuillage basé sur la couleur dominante détectée
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Optionnel : opération morpho "closing" pour réduire les petits trous
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # --- Détection et remplissage des contours externes ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Création d'un masque "rempli"
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, color=255, thickness=-1)
        
        # Reconstruction d'une image BGR finale : fond blanc
        result_bgr = np.ones_like(img_bgr) * 255
        # Là où filled_mask == 255, on applique la couleur dominante détectée
        result_bgr[filled_mask == 255] = tuple(int(c) for c in dominant_bgr)
        
        # --- Affichage du résultat ---
        st.subheader("Image avec polygones remplis")
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # --- Option de téléchargement du résultat ---
        download_col, _ = st.columns([1,3])
        with download_col:
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
    return buf.getvalue()

if __name__ == "__main__":
    main()
