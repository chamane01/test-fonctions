import streamlit as st
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import exifread
import numpy as np
import os
from pyproj import Transformer
import io
import math

def extract_exif_info(image_file):
    """
    Extrait les informations EXIF d'une image : GPS (lat, lon, altitude) et FocalLength.
    Renvoie (lat, lon, altitude, focal_length) ou None si non disponibles.
    """
    image_file.seek(0)
    tags = exifread.process_file(image_file)
    
    lat = None
    lon = None
    altitude = None
    focal_length = None
    
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags['GPS GPSLatitudeRef'].printable.strip()
        lon_ref = tags['GPS GPSLongitudeRef'].printable.strip()
        lat = (float(lat_vals[0].num) / lat_vals[0].den +
               float(lat_vals[1].num) / lat_vals[1].den / 60 +
               float(lat_vals[2].num) / lat_vals[2].den / 3600)
        lon = (float(lon_vals[0].num) / lon_vals[0].den +
               float(lon_vals[1].num) / lon_vals[1].den / 60 +
               float(lon_vals[2].num) / lon_vals[2].den / 3600)
        if lat_ref.upper() == 'S':
            lat = -lat
        if lon_ref.upper() == 'W':
            lon = -lon
    
    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den
        
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den
    
    return lat, lon, altitude, focal_length

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

def compute_gsd(altitude, focal_length, sensor_width, image_width):
    """
    Calcule le Ground Sampling Distance (GSD) en mètres par pixel.
    sensor_width est en mm et focal_length en mm, altitude en m.
    """
    # Conversion de la largeur du capteur de mm en m
    sensor_width_m = sensor_width / 1000.0
    gsd = (altitude * sensor_width_m) / (focal_length * image_width)
    return gsd

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs):
    """
    Convertit une image JPEG en GeoTIFF géoréférencé en UTM.
    utm_center est le centre de l'image en coordonnées UTM (x, y).
    pixel_size est la taille réelle d'un pixel en mètres.
    """
    img = Image.open(image_file)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Calcul du coin supérieur gauche à partir du centre
    top_left_x = utm_center[0] - (width / 2) * pixel_size
    top_left_y = utm_center[1] + (height / 2) * pixel_size  # y augmente vers le nord

    transform = from_origin(top_left_x, top_left_y, pixel_size, pixel_size)
    
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

st.title("Conversion JPEG → GeoTIFF avec échelle réelle")

# Demander à l'utilisateur de saisir la largeur du capteur en mm (valeur par défaut indicative)
sensor_width = st.number_input("Largeur du capteur (mm)", value=6.17, step=0.01)

uploaded_files = st.file_uploader("Choisissez une ou plusieurs images JPEG", type=['jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    images_info = []  # Liste des informations extraites pour chaque image
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)
        lat, lon, altitude, focal_length = extract_exif_info(file_buffer)
        
        if lat is None or lon is None:
            st.warning(f"Aucune info GPS pour {uploaded_file.name}")
            continue
        
        utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
        images_info.append({
            'name': uploaded_file.name,
            'data': file_bytes,
            'lat': lat,
            'lon': lon,
            'altitude': altitude,
            'focal_length': focal_length,
            'utm': (utm_x, utm_y),
            'utm_crs': utm_crs
        })
    
    if len(images_info) == 0:
        st.error("Aucune image avec informations GPS valides n'a été trouvée.")
    else:
        # Utilisation de la première image comme référence
        ref_image_info = images_info[0]
        ref_img = Image.open(io.BytesIO(ref_image_info['data']))
        ref_width, ref_height = ref_img.size
        
        # Si les informations photogrammétriques (altitude et focale) sont disponibles
        if ref_image_info['altitude'] is not None and ref_image_info['focal_length'] is not None:
            pixel_size = compute_gsd(ref_image_info['altitude'], ref_image_info['focal_length'], sensor_width, ref_width)
            st.success(f"Échelle calculée via photogrammétrie : {pixel_size:.4f} m/pixel")
        else:
            st.warning("Informations photogrammétriques incomplètes pour la première image.")
            # En cas de données insuffisantes, on peut tenter une estimation via la distance entre images
            if len(images_info) > 1:
                xs = [img['utm'][0] for img in images_info]
                xs_sorted = sorted(xs)
                avg_distance = (xs_sorted[-1] - xs_sorted[0]) / (len(xs_sorted) - 1)
                pixel_size = avg_distance / ref_width
                st.info(f"Échelle estimée via recouvrement : {pixel_size:.4f} m/pixel")
            else:
                pixel_size = 1.0
                st.info("Utilisation d'une échelle par défaut de 1.0 m/pixel")
        
        # Conversion de l'image de référence en GeoTIFF
        tiff_path = "output.tif"
        ref_utm = ref_image_info['utm']
        utm_crs = ref_image_info['utm_crs']
        
        convert_to_tiff(io.BytesIO(ref_image_info['data']), tiff_path, ref_utm, pixel_size, utm_crs)
        
        with open(tiff_path, "rb") as f:
            st.download_button("Télécharger le fichier GeoTIFF", f, file_name="image_geotiff.tif")
        
        os.remove(tiff_path)
