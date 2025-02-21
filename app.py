import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image, ExifTags
import numpy as np
import math

st.title("Affichage d'une image orthorectifiée sur une carte")

# --- Fonctions d'extraction EXIF et GPS ---

def get_exif_data(image):
    """Extrait les données EXIF de l'image."""
    exif_data = {}
    try:
        raw_exif = image._getexif()
        if raw_exif:
            for tag, value in raw_exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des EXIF : {e}")
    return exif_data

def get_gps_info(exif_data):
    """Extrait le dictionnaire GPS des données EXIF."""
    if "GPSInfo" not in exif_data:
        return None
    gps_info = {}
    for key in exif_data["GPSInfo"].keys():
        decode = ExifTags.GPSTAGS.get(key, key)
        gps_info[decode] = exif_data["GPSInfo"][key]
    return gps_info

def convert_to_degress(value):
    """
    Convertit la valeur GPS (exprimée en fraction) en degrés décimaux.
    Ex: ((30, 1), (15, 1), (3000, 100)) -> 30 + 15/60 + 30/3600
    """
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + (m / 60.0) + (s / 3600.0)

# --- Téléversement de l'image ---
uploaded_file = st.file_uploader("Téléverser une image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image originale", use_column_width=True)

    # Extraction des EXIF
    exif_data = get_exif_data(image)
    gps_info = get_gps_info(exif_data)
    if gps_info is None:
        st.error("Aucune donnée GPS trouvée dans l'image.")
        st.stop()

    # Récupération de la latitude
    if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
        lat = convert_to_degress(gps_info["GPSLatitude"])
        if gps_info["GPSLatitudeRef"] != "N":
            lat = -lat
    else:
        st.error("Données GPS incomplètes pour la latitude.")
        st.stop()

    # Récupération de la longitude
    if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
        lon = convert_to_degress(gps_info["GPSLongitude"])
        if gps_info["GPSLongitudeRef"] != "E":
            lon = -lon
    else:
        st.error("Données GPS incomplètes pour la longitude.")
        st.stop()

    # Récupération de l'orientation (GPSImgDirection), si disponible
    angle = 0
    if "GPSImgDirection" in gps_info:
        direction = gps_info["GPSImgDirection"]
        # La donnée peut être un tuple (numérateur, dénominateur)
        if isinstance(direction, tuple):
            angle = direction[0] / direction[1]
        else:
            angle = float(direction)
    st.write(f"Coordonnées GPS extraites : **Latitude** {lat:.6f}, **Longitude** {lon:.6f}")
    st.write(f"Orientation (GPSImgDirection) : **{angle:.1f}°** (si fournie)")

    # --- Orthorectification simple : rotation de l'image pour l'aligner au nord ---
    # Pour obtenir un image « north-up », on effectue une rotation de -angle (en degrés)
    rotated_image = image.rotate(-angle, expand=True)
    st.image(rotated_image, caption="Image orthorectifiée (rotation appliquée)", use_column_width=True)

    # --- Application de l'échelle fixe : 3 cm par pixel (0,03 m/pixel) ---
    img_width, img_height = rotated_image.size
    width_m = img_width * 0.03
    height_m = img_height * 0.03
    st.write(f"Dimensions de l'image après rotation : {img_width}px x {img_height}px")
    st.write(f"Dimensions réelles (échelle 3 cm/pixel) : environ {width_m:.2f} m x {height_m:.2f} m")

    # --- Calcul des bornes géographiques de l'image ---
    # On considère le point GPS extrait comme le centre de l'image.
    # Conversion approximative : 1° de latitude ~ 111320 m, 1° de longitude ~ 111320 * cos(lat) m
    dlat = (height_m / 2) / 111320
    dlon = (width_m / 2) / (111320 * math.cos(math.radians(lat)))
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]

    # --- Création de la carte Folium ---
    m = folium.Map(location=[lat, lon], zoom_start=18)

    # Conversion de l'image (après rotation) en tableau numpy pour l'overlay
    image_np = np.array(rotated_image)

    # Ajout de l'overlay de l'image sur la carte
    folium.raster_layers.ImageOverlay(
        image=image_np,
        bounds=bounds,
        opacity=1,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Marqueur au centre de l'image
    folium.Marker([lat, lon], tooltip="Centre de l'image").add_to(m)
    folium.LayerControl().add_to(m)

    st.subheader("Carte avec l'image positionnée et orthorectifiée")
    st_folium(m, width=700, height=500)
else:
    st.info("Veuillez téléverser une image contenant des données GPS.")
