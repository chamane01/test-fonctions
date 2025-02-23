import streamlit as st
from PIL import Image, ExifTags, ImageOps
import io
import numpy as np
import os
import rasterio
from rasterio.transform import from_origin
import pyproj

st.title("Calcul de l'Empreinte au Sol et Conversion en GeoTIFF")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléverser une image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Ouvrir et afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # Extraction des métadonnées EXIF via PIL
    exif_data = {}
    if hasattr(image, '_getexif'):
        exif_raw = image._getexif()
        if exif_raw is not None:
            exif_data = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_raw.items()}
    
    # Extraction de la longueur focale
    focal_length_exif = None
    if 'FocalLength' in exif_data:
        focal = exif_data['FocalLength']
        if isinstance(focal, tuple) and len(focal) == 2:
            focal_length_exif = focal[0] / focal[1]
        else:
            focal_length_exif = float(focal)
    
    # Extraction de l'altitude GPS
    gps_altitude = None
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        for key in gps_info:
            tag = ExifTags.GPSTAGS.get(key, key)
            if tag == 'GPSAltitude':
                alt_val = gps_info[key]
                if isinstance(alt_val, tuple) and len(alt_val) == 2:
                    gps_altitude = alt_val[0] / alt_val[1]
                else:
                    gps_altitude = float(alt_val)
    
    # Extraction des coordonnées GPS (Latitude et Longitude)
    gps_lat = gps_lon = None
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        gps_data = {}
        for key in gps_info:
            sub_tag = ExifTags.GPSTAGS.get(key, key)
            gps_data[sub_tag] = gps_info[key]
        if all(k in gps_data for k in ('GPSLatitude', 'GPSLatitudeRef', 'GPSLongitude', 'GPSLongitudeRef')):
            def dms_to_dd(dms):
                return dms[0] + dms[1]/60 + dms[2]/3600
            lat = dms_to_dd([val[0]/val[1] for val in gps_data['GPSLatitude']])
            if gps_data['GPSLatitudeRef'] != 'N':
                lat = -lat
            lon = dms_to_dd([val[0]/val[1] for val in gps_data['GPSLongitude']])
            if gps_data['GPSLongitudeRef'] != 'E':
                lon = -lon
            gps_lat = lat
            gps_lon = lon

    st.subheader("Métadonnées extraites")
    st.write("Longueur focale (EXIF) :", focal_length_exif if focal_length_exif is not None else "Non disponible")
    st.write("Altitude GPS (EXIF) :", gps_altitude if gps_altitude is not None else "Non disponible")
    st.write("Coordonnées GPS :", f"{gps_lat}, {gps_lon}" if gps_lat and gps_lon else "Non disponibles")
    st.write("Dimensions de l'image (pixels) :", image.size)
    
    st.subheader("Paramètres de calcul")
    # Permettre la saisie ou la vérification des paramètres
    hauteur = st.number_input("Hauteur de vol (m)", value=(gps_altitude if gps_altitude is not None else 100.0))
    focale = st.number_input("Longueur focale (mm)", value=(focal_length_exif if focal_length_exif is not None else 50.0))
    largeur_capteur = st.number_input("Largeur du capteur (mm)", value=36.0)
    
    # Calcul de l'empreinte au sol et du GSD
    if st.button("Calculer"):
        empreinte_sol = (hauteur * largeur_capteur) / focale  # en m
        resolution_pixels = image.width  # résolution horizontale en pixels
        gsd = empreinte_sol / resolution_pixels  # en m/pixel
        st.markdown("### Résultats")
        st.write(f"**Empreinte au sol :** {empreinte_sol:.2f} m")
        st.write(f"**Résolution au sol (GSD) :** {gsd*100:.2f} cm/pixel")
        # Stocker la valeur calculée dans la session pour la conversion
        st.session_state['gsd'] = gsd

    # Fonction de conversion de lat/lon en UTM
    def latlon_to_utm(lat, lon):
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            utm_crs = f"EPSG:326{zone:02d}"
        else:
            utm_crs = f"EPSG:327{zone:02d}"
        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return utm_x, utm_y, utm_crs

    # Bouton de conversion en GeoTIFF et téléchargement (disponible si le calcul a été effectué)
    if 'gsd' in st.session_state:
        if st.button("Convertir et Télécharger en GeoTIFF"):
            if gps_lat is None or gps_lon is None:
                st.error("Les coordonnées GPS ne sont pas disponibles pour géoréférencer l'image.")
            else:
                # Conversion des coordonnées en UTM
                utm_x, utm_y, utm_crs = latlon_to_utm(gps_lat, gps_lon)
                
                # Correction d'orientation si nécessaire
                image_corr = ImageOps.exif_transpose(image)
                img_array = np.array(image_corr)
                height, width = img_array.shape[:2]
                
                # Utiliser la résolution au sol (GSD) calculée pour définir la taille d'un pixel (m/pixel)
                pixel_size = st.session_state['gsd']
                
                # Déterminer le coin supérieur gauche en UTM
                x_min = utm_x - (width / 2) * pixel_size
                y_max = utm_y + (height / 2) * pixel_size  # Note : y décroît vers le bas de l'image
                transform = from_origin(x_min, y_max, pixel_size, pixel_size)
                
                output_path = "output.tif"
                count = 3 if len(img_array.shape) == 3 else 1
                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=count,
                    dtype=img_array.dtype,
                    crs=utm_crs,
                    transform=transform
                ) as dst:
                    if count == 3:
                        for i in range(3):
                            dst.write(img_array[:, :, i], i + 1)
                    else:
                        dst.write(img_array, 1)
                
                st.success("Image convertie en GeoTIFF avec la résolution au sol calculée.")
                # Proposer le téléchargement du fichier GeoTIFF
                with open(output_path, "rb") as f:
                    tiff_bytes = f.read()
                st.download_button(
                    label="Télécharger le GeoTIFF",
                    data=tiff_bytes,
                    file_name="image_geotiff.tif",
                    mime="image/tiff"
                )
                os.remove(output_path)
else:
    st.info("Veuillez téléverser une image.")
