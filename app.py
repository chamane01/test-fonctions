import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS
import math

def get_exif_data(image):
    """Récupère les métadonnées EXIF de l'image."""
    exif_data = {}
    try:
        exif_raw = image._getexif()
        if exif_raw:
            for tag, value in exif_raw.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
    except AttributeError:
        pass
    return exif_data

def calculate_gsd(height, sensor_width, focal_length, img_width):
    """Calcule le GSD (Ground Sample Distance)."""
    return (height * sensor_width) / (focal_length * img_width)

# Interface Streamlit
st.title("Calcul de la Surface Couverte par une Image Aérienne")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléverser une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image Téléversée", use_column_width=True)
    
    # Extraction des métadonnées
    exif_data = get_exif_data(image)
    focal_length = exif_data.get("FocalLength", "Non disponible")
    
    # Saisie des paramètres
    height = st.number_input("Hauteur de vol (m)", min_value=1.0, value=100.0)
    sensor_width = st.number_input("Largeur du capteur (mm)", min_value=1.0, value=36.0)
    img_width = image.width  # Largeur en pixels
    img_height = image.height  # Hauteur en pixels
    
    # Utilisation de la focale détectée si disponible
    if isinstance(focal_length, tuple):
        focal_length = focal_length[0] / focal_length[1]  # Cas où la focale est une fraction
    elif isinstance(focal_length, int) or isinstance(focal_length, float):
        pass  # Valeur correcte
    else:
        focal_length = st.number_input("Longueur focale (mm)", min_value=1.0, value=35.0)
    
    # Calcul du GSD et dimensions au sol
    gsd = calculate_gsd(height, sensor_width, focal_length, img_width)
    width_ground = gsd * img_width
    height_ground = gsd * img_height
    
    # Affichage des résultats
    st.subheader("Résultats")
    st.write(f"GSD (Taille d'un pixel au sol) : {gsd:.4f} m/pixel")
    st.write(f"Largeur couverte au sol : {width_ground:.2f} m")
    st.write(f"Hauteur couverte au sol : {height_ground:.2f} m")
