import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Remplissage de Polygones – Image Magenta sur Fond Blanc")
    st.write(
        """
        Cet outil permet de prendre une image où les objets (ex. bâtiments) sont en magenta 
        sur fond blanc et de produire une image où chaque bloc est rempli, sans trous.
        """
    )

    # --- Chargement de l'image ---
    uploaded_file = st.file_uploader("Téléversez votre image (JPEG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        st.subheader("Aperçu de l'image originale")
        st.image(pil_img, use_container_width=True)

        # --- Paramètres de seuillage HSV ---
        st.sidebar.title("Paramètres de seuillage (HSV)")
        st.sidebar.write("Ajustez pour extraire la zone magenta")
        h_lower = st.sidebar.slider("H min", 0, 179, 140)
        s_lower = st.sidebar.slider("S min", 0, 255, 50)
        v_lower = st.sidebar.slider("V min", 0, 255, 50)
        h_upper = st.sidebar.slider("H max", 0, 179, 160)
        s_upper = st.sidebar.slider("S max", 0, 255, 255)
        v_upper = st.sidebar.slider("V max", 0, 255, 255)

        # --- Paramètres morphologiques (optionnel) ---
        st.sidebar.title("Paramètres Morphologiques")
        use_morph = st.sidebar.checkbox("Activer un closing morphologique (pour boucher les petits trous)", value=False)
        morph_kernel = st.sidebar.slider("Taille du kernel", 1, 20, 5)

        # Conversion en HSV et seuillage
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
        upper = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Optionnel : opération morpho "closing" pour réduire les petits trous
        # (si vos blocs présentent encore des discontinuités)
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
        # Là où filled_mask == 255, on applique la couleur magenta (BGR = 255,0,255)
        result_bgr[filled_mask == 255] = (255, 0, 255)

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
