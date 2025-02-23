import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
import os

st.title("Calcul de l'Empreinte au Sol et Conversion en GeoTIFF")

# --- Fonction pour convertir une valeur GPS en degrés décimaux ---
def convert_to_degrees(value):
    # Si l'élément est déjà un float, le renvoyer directement, sinon on suppose qu'il s'agit d'un tuple (num, den)
    d = float(value[0]) if not isinstance(value[0], tuple) else float(value[0][0]) / float(value[0][1])
    m = float(value[1]) if not isinstance(value[1], tuple) else float(value[1][0]) / float(value[1][1])
    s = float(value[2]) if not isinstance(value[2], tuple) else float(value[2][0]) / float(value[2][1])
    return d + m / 60 + s / 3600

# --- Fonction de conversion de lat/lon en UTM ---
def latlon_to_utm(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        crs_utm = CRS.from_epsg(32600 + zone)
    else:
        crs_utm = CRS.from_epsg(32700 + zone)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs_utm, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, crs_utm

# --- Téléversement de l'image ---
uploaded_file = st.file_uploader("Téléverser une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # --- Extraction des métadonnées EXIF avec PIL ---
    exif_data = {}
    if hasattr(image, '_getexif'):
        exif_raw = image._getexif()
        if exif_raw is not None:
            for tag, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value

    # --- Extraction des coordonnées GPS ---
    gps_lat, gps_lon = None, None
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        # Les clés standards de GPSInfo dans PIL :
        # 1: GPSLatitudeRef, 2: GPSLatitude, 3: GPSLongitudeRef, 4: GPSLongitude, 6: GPSAltitude
        gps_latitude = gps_info.get(2)
        gps_latitude_ref = gps_info.get(1)
        gps_longitude = gps_info.get(4)
        gps_longitude_ref = gps_info.get(3)
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            gps_lat = convert_to_degrees(gps_latitude)
            if gps_latitude_ref != 'N':
                gps_lat = -gps_lat
            gps_lon = convert_to_degrees(gps_longitude)
            if gps_longitude_ref != 'E':
                gps_lon = -gps_lon

    # --- Récupération de la longueur focale depuis EXIF ---
    focal_length_exif = None
    if 'FocalLength' in exif_data:
        focal = exif_data['FocalLength']
        if isinstance(focal, tuple) and len(focal) == 2:
            focal_length_exif = focal[0] / focal[1]
        else:
            focal_length_exif = float(focal)

    # --- Récupération de l'altitude GPS depuis EXIF ---
    gps_altitude = None
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        alt = gps_info.get(6)  # clé 6 correspond généralement à GPSAltitude
        if alt:
            if isinstance(alt, tuple) and len(alt) == 2:
                gps_altitude = alt[0] / alt[1]
            else:
                gps_altitude = float(alt)

    st.subheader("Métadonnées extraites")
    st.write("Longueur focale (EXIF) :", focal_length_exif if focal_length_exif is not None else "Non disponible")
    st.write("Altitude GPS (EXIF) :", gps_altitude if gps_altitude is not None else "Non disponible")
    st.write("Coordonnées GPS :", f"Lat: {gps_lat}, Lon: {gps_lon}" if (gps_lat is not None and gps_lon is not None) else "Non disponibles")
    st.write("Dimensions de l'image (pixels) :", image.size)

    st.subheader("Entrer ou vérifier les paramètres nécessaires")
    # Paramètres utilisateur avec valeurs par défaut préremplies si disponibles
    hauteur = st.number_input("Hauteur de vol (m)", value=(gps_altitude if gps_altitude is not None else 100.0))
    focale = st.number_input("Longueur focale (mm)", value=(focal_length_exif if focal_length_exif is not None else 50.0))
    largeur_capteur = st.number_input("Largeur du capteur (mm)", value=36.0)
    
    # --- Calcul de l'empreinte au sol et du GSD ---
    if st.button("Calculer"):
        # Empreinte au sol en m
        empreinte_sol = (hauteur * largeur_capteur) / focale  
        # Nombre de pixels en largeur de l'image
        resolution_pixels = image.width  
        # Ground Sampling Distance (m/pixel)
        gsd = empreinte_sol / resolution_pixels  
        
        st.markdown("### Résultats")
        st.write(f"**Empreinte au sol :** {empreinte_sol:.2f} m")
        st.write(f"**Résolution au sol (GSD) :** {gsd*100:.2f} cm/pixel")
        
        # --- Bouton pour conversion et téléchargement en GeoTIFF ---
        if st.button("Convertir et Télécharger en GeoTIFF"):
            # Conversion de l'image en tableau numpy
            image_np = np.array(image)
            
            if gps_lat is None or gps_lon is None:
                st.error("Les coordonnées GPS ne sont pas disponibles pour géoréférencer l'image.")
            else:
                # Conversion des coordonnées GPS en UTM
                utm_x, utm_y, utm_crs = latlon_to_utm(gps_lat, gps_lon)
                
                # On suppose ici que le point GPS correspond au centre de l'image.
                # Calcul du coin supérieur gauche en UTM à partir du GSD calculé
                img_width, img_height = image.size
                top_left_x = utm_x - (img_width / 2) * gsd
                top_left_y = utm_y + (img_height / 2) * gsd  # en UTM, Y augmente vers le nord
                transform = from_origin(top_left_x, top_left_y, gsd, gsd)
                
                output_path = "output_georef.tif"
                height_img, width_img = image_np.shape[:2]
                count = 3 if image_np.ndim == 3 and image_np.shape[2] == 3 else 1
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height_img,
                    width=width_img,
                    count=count,
                    dtype=image_np.dtype,
                    crs=utm_crs.to_wkt(),
                    transform=transform
                ) as dst:
                    if count == 3:
                        for i in range(3):
                            dst.write(image_np[:, :, i], i + 1)
                    else:
                        dst.write(image_np, 1)
                
                with open(output_path, "rb") as f:
                    tiff_bytes = f.read()
                st.download_button(
                    label="Télécharger le GeoTIFF",
                    data=tiff_bytes,
                    file_name="image_georef.tif",
                    mime="image/tiff"
                )
                os.remove(output_path)
