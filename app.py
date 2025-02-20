import streamlit as st
import cv2
import numpy as np
import tempfile

def main():
    st.title("Photogrammétrie - Génération d'Orthophoto")
    st.write("Téléversez au moins deux images pour générer une orthophoto (mosaïque).")

    # Téléversement multiple de fichiers image
    uploaded_files = st.file_uploader("Choisissez vos images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        images = []
        for file in uploaded_files:
            # Lecture des données de l'image
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        
        if len(images) < 2:
            st.error("Veuillez téléverser au moins deux images.")
        else:
            st.info("Traitement en cours...")
            # Utilisation du module Stitcher d'OpenCV pour assembler les images
            if hasattr(cv2, 'Stitcher_create'):
                stitcher = cv2.Stitcher_create()
            else:
                stitcher = cv2.createStitcher(False)
            
            status, stitched = stitcher.stitch(images)
            
            if status == cv2.Stitcher_OK:
                st.success("Orthophoto générée avec succès!")
                # Conversion en RGB pour affichage avec Streamlit
                stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
                st.image(stitched_rgb, caption="Orthophoto", use_column_width=True)
                
                # Sauvegarde de l'image en TIFF dans un fichier temporaire
                tiff_file = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
                cv2.imwrite(tiff_file.name, stitched)
                
                with open(tiff_file.name, "rb") as file:
                    st.download_button("Télécharger l'Orthophoto TIFF",
                                       data=file,
                                       file_name="orthophoto.tiff",
                                       mime="image/tiff")
            else:
                st.error(f"Erreur lors de la génération de l'orthophoto (code : {status}).")

if __name__ == "__main__":
    main()
