import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import base64
from io import BytesIO

st.set_page_config(page_title="Application de Dessin", layout="wide")
st.title("Application de Dessin - Version Base64")

def decode_base64_image(base64_str):
    """Convertit une chaîne base64 en image PIL"""
    try:
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
            
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        st.error(f"Erreur de décodage : {str(e)}")
        return None

# Choix de la source de l'image
source_type = st.radio("Source de l'image", ["Téléversement", "Coller en base64"])

bg_image = None
canvas_width = 600
canvas_height = 400

if source_type == "Téléversement":
    uploaded_file = st.file_uploader("Téléversez une image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        bg_image = Image.open(uploaded_file)
        canvas_width, canvas_height = bg_image.size
        
else:
    base64_input = st.text_area("Collez votre image en base64 (commence par 'data:image/...')")
    if base64_input:
        bg_image = decode_base64_image(base64_input)
        if bg_image:
            canvas_width, canvas_height = bg_image.size
            st.image(bg_image, caption="Image décodée", use_container_width=True)

# Paramètres du dessin
point_size = st.sidebar.radio("Taille des points", [5, 10, 20])
stroke_color = st.sidebar.color_picker("Couleur du pinceau", "#000000")
background_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF") if not bg_image else None

# Création du canevas
canvas_result = st_canvas(
    fill_color=stroke_color,
    stroke_width=point_size,
    stroke_color=stroke_color,
    background_color=background_color,
    background_image=bg_image,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="freedraw",
    key="canvas",
)

# Sauvegarde du résultat
if canvas_result.image_data is not None:
    output_image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
    
    if st.sidebar.button("Exporter le dessin"):
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Création d'un lien de téléchargement
        href = f'<a href="data:image/png;base64,{img_b64}" download="dessin.png">Télécharger l\'image</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
