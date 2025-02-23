import streamlit as st
from PIL import Image, ExifTags, ImageOps
import exifread
import rasterio
from rasterio.transform import from_origin
import numpy as np
import os
from pyproj import Transformer
import io
import math

# === Fonctions issues du second code ===
def extract_exif_info(image_file):
    """
    Extrait les informations EXIF d'une image :
    - GPS (lat, lon, altitude)
    - EXIF FocalLength
    - FocalPlaneXResolution et FocalPlaneResolutionUnit (pour estimer la largeur du capteur)
    Renvoie (lat, lon, altitude, focal_length, fp_x_res, fp_unit) ou None si indisponible.
    """
    image_file.seek(0)
    tags = exifread.process_file(image_file, details=False)
    
    lat = lon = altitude = focal_length = None
    fp_x_res = fp_unit = None
    
    # --- GPS Latitude / Longitude ---
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_ref = tags.get('GPS GPSLongitudeRef')
        
        if lat_vals and lon_vals and lat_ref and lon_ref:
            lat = (float(lat_vals[0].num) / lat_vals[0].den +
                   float(lat_vals[1].num) / lat_vals[1].den / 60 +
                   float(lat_vals[2].num) / lat_vals[2].den / 3600)
            lon = (float(lon_vals[0].num) / lon_vals[0].den +
                   float(lon_vals[1].num) / lon_vals[1].den / 60 +
                   float(lon_vals[2].num) / lon_vals[2].den / 3600)
            
            if lat_ref.printable.strip().upper() == 'S':
                lat = -lat
            if lon_ref.printable.strip().upper() == 'W':
                lon = -lon
    
    # --- GPS Altitude ---
    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den
        
    # --- Focal Length ---
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den
    
    # --- Focal Plane Resolution (X) + Unit ---
    if 'EXIF FocalPlaneXResolution' in tags and 'EXIF FocalPlaneResolutionUnit' in tags:
        fp_res_tag = tags['EXIF FocalPlaneXResolution']
        fp_unit_tag = tags['EXIF FocalPlaneResolutionUnit']
        
        fp_x_res = float(fp_res_tag.values[0].num) / fp_res_tag.values[0].den
        fp_unit = int(fp_unit_tag.values[0])
    
    return lat, lon, altitude, focal_length, fp_x_res, fp_unit

def latlon_to_utm(lat, lon):
    """
    Convertit des coordonnées lat/lon en coordonnées UTM.
    Renvoie (utm_x, utm_y, utm_crs).
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs):
    """
    Convertit une image JPEG en GeoTIFF géoréférencé en UTM.
    - utm_center : (x, y) du centre en coordonnées UTM
    - pixel_size : taille d'un pixel en mètres (m/pixel)
    - utm_crs    : code EPSG (ex: 'EPSG:32632')
    """
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Coordonnées du coin supérieur gauche
    x_min = utm_center[0] - (width / 2) * pixel_size
    y_max = utm_center[1] + (height / 2) * pixel_size  # y diminue vers le bas
    
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3 if len(img_array.shape) == 3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform
    ) as dst:
        if len(img_array.shape) == 3:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            dst.write(img_array, 1)

# === Application Streamlit ===
st.title("Calcul de l'Empreinte au Sol et Conversion JPEG → GeoTIFF")

uploaded_file = st.file_uploader("Téléverser une image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture des octets de l'image pour usage multiple
    file_bytes = uploaded_file.read()
    image_buffer = io.BytesIO(file_bytes)
    image = Image.open(image_buffer)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # --- Extraction des métadonnées via PIL ---
    exif_data = {}
    if hasattr(image, '_getexif'):
        exif_raw = image._getexif()
        if exif_raw is not None:
            for tag, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value

    # Longueur focale (EXIF)
    focal_length_exif = None
    if 'FocalLength' in exif_data:
        focal = exif_data['FocalLength']
        if isinstance(focal, tuple) and len(focal) == 2:
            focal_length_exif = focal[0] / focal[1]
        else:
            focal_length_exif = float(focal)
    
    # Altitude GPS (EXIF)
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
    
    st.subheader("Métadonnées extraites (via PIL)")
    st.write("Longueur focale (EXIF) : ", focal_length_exif if focal_length_exif is not None else "Non disponible")
    st.write("Altitude GPS (EXIF) : ", gps_altitude if gps_altitude is not None else "Non disponible")
    st.write("Dimensions de l'image (pixels) : ", image.size)
    
    st.subheader("Paramètres pour le calcul de l'empreinte au sol")
    # L'utilisateur peut ajuster les valeurs, préremplies si disponibles
    hauteur = st.number_input("Hauteur de vol (m)", value=(gps_altitude if gps_altitude is not None else 100.0))
    focale = st.number_input("Longueur focale (mm)", value=(focal_length_exif if focal_length_exif is not None else 50.0))
    largeur_capteur = st.number_input("Largeur du capteur (mm)", value=36.0)
    
    if st.button("Calculer et Convertir"):
        # --- Calcul de l'empreinte au sol et du GSD ---
        empreinte_sol = (hauteur * largeur_capteur) / focale
        resolution_pixels = image.size[0]  # largeur en pixels
        gsd = empreinte_sol / resolution_pixels  # en m/pixel
        
        st.markdown("### Résultats du calcul")
        st.write(f"**Empreinte au sol :** {empreinte_sol:.2f} m")
        st.write(f"**Résolution au sol (GSD) :** {gsd*100:.2f} cm/pixel")
        
        # --- Extraction complémentaire des métadonnées avec exifread pour la géoréférenciation ---
        image_buffer_exif = io.BytesIO(file_bytes)
        lat, lon, altitude_exif, focal_length_exif2, fp_x_res, fp_unit = extract_exif_info(image_buffer_exif)
        if lat is None or lon is None:
            st.warning("Pas de coordonnées GPS dans l'image, conversion impossible.")
        else:
            img_width, img_height = image.size
            # Calcul de la largeur du capteur à partir de FocalPlaneXResolution si disponible
            sensor_width_mm = None
            if fp_x_res and fp_unit:
                if fp_unit == 2:   # pouces
                    sensor_width_mm = (img_width / fp_x_res) * 25.4
                elif fp_unit == 3: # cm
                    sensor_width_mm = (img_width / fp_x_res) * 10
                elif fp_unit == 4: # mm
                    sensor_width_mm = (img_width / fp_x_res)
            # Si non disponible, on utilise la valeur saisie par l'utilisateur
            if sensor_width_mm is None:
                sensor_width_mm = largeur_capteur
            
            # Conversion des coordonnées lat/lon en UTM
            utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
            
            # Ici, on utilise le GSD calculé comme taille de pixel pour la conversion
            pixel_size = gsd  # en m/pixel
            st.info(f"Résolution spatiale appliquée : {pixel_size*100:.2f} cm/pixel")
            
            output_path = "output.tif"
            convert_to_tiff(
                image_file=io.BytesIO(file_bytes),
                output_path=output_path,
                utm_center=(utm_x, utm_y),
                pixel_size=pixel_size,
                utm_crs=utm_crs
            )
            
            st.success("Image convertie en GeoTIFF.")
            # Affichage des métadonnées du GeoTIFF créé
            with rasterio.open(output_path) as src:
                st.write("**Méta-données GeoTIFF**")
                st.write("CRS :", src.crs)
                st.write("Transform :", src.transform)
            
            # Proposition de téléchargement du GeoTIFF
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Télécharger le GeoTIFF",
                    data=f,
                    file_name="image_geotiff.tif",
                    mime="image/tiff"
                )
            
            os.remove(output_path)
