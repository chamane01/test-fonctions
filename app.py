import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

st.title("Sélecteur de couleur avec pipette")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Ouvrir l'image avec PIL
    image = Image.open(uploaded_file)
    # Assurez-vous que l'image possède un format (certains fichiers uploadés n'ont pas l'attribut "format")
    if image.format is None:
        image.format = "PNG"
    
    st.image(image, caption="Image téléversée", use_column_width=True)
    
    # Conversion de l'image en tableau numpy pour extraire la couleur plus tard
    image_np = np.array(image)
    
    # Création du canvas avec l'image en arrière-plan
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Couleur de remplissage (transparent ici)
        stroke_width=1,
        stroke_color="#ffffff",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="point",  # Mode point pour capturer le clic
        point_display_radius=3,
        key="canvas",
    )
    
    # Si l'utilisateur a cliqué, récupérer les coordonnées
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            # On récupère le dernier point cliqué
            last_point = objects[-1]
            x = int(last_point["left"])
            y = int(last_point["top"])
            st.write(f"Coordonnées sélectionnées : x = {x}, y = {y}")
            
            # Vérifier que les coordonnées sont bien dans les dimensions de l'image
            if x < image_np.shape[1] and y < image_np.shape[0]:
                # Extraction de la couleur (RGB) au point cliqué
                color = image_np[y, x]
                st.write("Couleur sélectionnée (RGB) :", color)
                # Conversion de la couleur en code hexadécimal
                hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
                st.markdown(
                    f"<div style='width:100px; height:100px; background-color:{hex_color};'></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error("Les coordonnées sont hors de l'image.")
