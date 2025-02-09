import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import json
from streamlit_drawable_canvas import st_canvas
from pyproj import Transformer
from shapely.geometry import mapping, Point, LineString, Polygon

st.set_page_config(layout="wide")
st.title("Éditeur d'images Drone Géolocalisées")

###########################
# Fonctions EXIF et GPS
###########################

def get_exif_data(image):
    """Extrait les données EXIF de l'image."""
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif_data[decoded] = value
    except Exception as e:
        st.write("Erreur lors de l'extraction des EXIF:", e)
    return exif_data

def get_decimal_from_dms(dms, ref):
    """Convertit une coordonnée GPS du format DMS en degrés décimaux."""
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_gps_coords(exif_data):
    """Extrait latitude et longitude depuis les données EXIF."""
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None, None
    gps_data = {}
    for key in gps_info.keys():
        decode = ExifTags.GPSTAGS.get(key, key)
        gps_data[decode] = gps_info[key]
    try:
        lat = get_decimal_from_dms(gps_data["GPSLatitude"], gps_data["GPSLatitudeRef"])
        lon = get_decimal_from_dms(gps_data["GPSLongitude"], gps_data["GPSLongitudeRef"])
        return lat, lon
    except Exception as e:
        st.write("Erreur d'extraction des coordonnées GPS:", e)
        return None, None

###########################
# Projection UTM et conversion pixel → coordonnées réelles
###########################

def latlon_to_utm(lat, lon):
    """
    Convertit une coordonnée lat/lon (EPSG:4326) en UTM.
    La zone UTM est déterminée automatiquement en fonction de la longitude.
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_code = 32600 + zone  # hémisphère nord
    else:
        epsg_code = 32700 + zone  # hémisphère sud
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{epsg_code}", always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, epsg_code

def pixel_to_utm(x, y, top_left_x, top_left_y, resolution):
    """
    Convertit des coordonnées pixel (x,y) en coordonnées UTM.
    On suppose que l'image est orientée avec le nord en haut.
    """
    utm_x = top_left_x + x * resolution
    utm_y = top_left_y - y * resolution  # y croît vers le bas dans l'image
    return utm_x, utm_y

###########################
# Interface utilisateur
###########################

st.sidebar.header("Charger une image")
uploaded_file = st.sidebar.file_uploader("Sélectionnez une image (JPEG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Chargement de l'image via PIL
    image = Image.open(uploaded_file)
    width, height = image.size
    st.image(image, caption="Image chargée", use_column_width=True)

    # Extraction des données EXIF pour récupérer la géolocalisation
    exif_data = get_exif_data(image)
    lat, lon = get_gps_coords(exif_data)
    if lat is not None and lon is not None:
        st.sidebar.success(f"Coordonnées GPS extraites : {lat:.6f}, {lon:.6f}")
    else:
        st.sidebar.warning("Aucune donnée GPS trouvée dans les métadonnées EXIF.")
        lat = st.sidebar.number_input("Entrez la latitude", value=0.0, format="%.6f")
        lon = st.sidebar.number_input("Entrez la longitude", value=0.0, format="%.6f")

    # Conversion en UTM
    center_utm_x, center_utm_y, epsg_code = latlon_to_utm(lat, lon)
    st.sidebar.info(f"Projection UTM utilisée : EPSG:{epsg_code}")

    # Paramétrage de la résolution au sol (m/px)
    resolution = st.sidebar.number_input("Résolution au sol (m/px)", value=0.05, step=0.01, format="%.3f")

    # Calcul des coordonnées UTM du coin supérieur gauche de l'image.
    # On considère que le centre de l'image correspond aux coordonnées GPS extraites.
    top_left_x = center_utm_x - (width / 2) * resolution
    top_left_y = center_utm_y + (height / 2) * resolution

    # Choix du mode de dessin
    drawing_mode = st.sidebar.selectbox("Mode de dessin", ("point", "line", "polygon"))

    # Détermination du mode de dessin pour st_canvas
    if drawing_mode == "point":
        canvas_drawing_mode = "circle"
    elif drawing_mode == "line":
        canvas_drawing_mode = "line"
    elif drawing_mode == "polygon":
        canvas_drawing_mode = "polygon"
    else:
        canvas_drawing_mode = "freedraw"

    st.sidebar.markdown("**Dessinez directement sur l'image ci-dessous.**")

    # Conversion de l'image en tableau NumPy pour st_canvas
    image_array = np.array(image)

    # Zone de dessin interactive avec st_canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # couleur de remplissage par défaut
        stroke_width=2,
        stroke_color="#000000",
        background_image=image_array,
        background_image_mode="fit",  # mode en minuscules
        # Si nécessaire, retirez 'update_streamlit' ou mettez à jour streamlit-drawable-canvas\n        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=canvas_drawing_mode,
        key="canvas",
    )

    ###########################
    # Traitement des annotations
    ###########################

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        features = []
        for obj in objects:
            # Pour le mode "point", récupérer le centre du cercle dessiné
            if obj.get("type") == "circle" and drawing_mode == "point":
                x = obj.get("left", 0) + obj.get("width", 0) / 2
                y = obj.get("top", 0) + obj.get("height", 0) / 2
                utm_coord = pixel_to_utm(x, y, top_left_x, top_left_y, resolution)
                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [utm_coord[0], utm_coord[1]]},
                    "properties": {}
                }
                features.append(feature)
            # Pour les lignes et polygones, extraire la liste des points
            elif obj.get("type") in ["path", "line", "polygon"]:
                pts = obj.get("points")
                if pts and len(pts) > 0:
                    utm_points = [pixel_to_utm(pt.get("x", 0), pt.get("y", 0), top_left_x, top_left_y, resolution) for pt in pts]
                    if drawing_mode == "polygon" or obj.get("type") == "polygon":
                        # Pour un polygone, assurer que le premier et le dernier point soient identiques
                        if utm_points[0] != utm_points[-1]:
                            utm_points.append(utm_points[0])
                        geom = Polygon(utm_points)
                    else:
                        geom = LineString(utm_points)
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {}
                    }
                    features.append(feature)
        if features:
            geojson = {"type": "FeatureCollection", "features": features}
            geojson_str = json.dumps(geojson, indent=2)
            st.download_button("Télécharger les annotations (GeoJSON)", data=geojson_str, file_name="annotations.geojson", mime="application/json")
            st.subheader("Aperçu du GeoJSON")
            st.code(geojson_str, language='json')
        else:
            st.info("Aucune annotation n'a été dessinée.")
else:
    st.info("Veuillez charger une image pour commencer.")
