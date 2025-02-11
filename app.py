import streamlit as st
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import exifread
import numpy as np
import os

def extract_gps_info(image_path):
    """Extrait les coordonnées GPS d'une image si disponibles"""
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat = tags['GPS GPSLatitude'].values
        lon = tags['GPS GPSLongitude'].values
        lat_ref = tags['GPS GPSLatitudeRef'].values
        lon_ref = tags['GPS GPSLongitudeRef'].values
        
        lat = (lat[0] + lat[1]/60 + lat[2]/3600) * (-1 if lat_ref == 'S' else 1)
        lon = (lon[0] + lon[1]/60 + lon[2]/3600) * (-1 if lon_ref == 'W' else 1)
        
        return lat, lon
    return None, None

def convert_to_tiff(image_path, output_path, lat, lon):
    """Convertit une image JPEG en GeoTIFF avec des coordonnées GPS"""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Définition d'une transformation basique (origine en lat/lon)
    transform = from_origin(lon, lat, 0.0001, 0.0001)  # Approximation de la résolution
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3 if len(img_array.shape) == 3 else 1,
        dtype=img_array.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        if len(img_array.shape) == 3:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            dst.write(img_array, 1)

st.title("Convertisseur JPEG → GeoTIFF avec métadonnées GPS")

uploaded_file = st.file_uploader("Choisissez une image JPEG", type=['jpg', 'jpeg'])

if uploaded_file:
    image_path = "temp.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())
    
    lat, lon = extract_gps_info(image_path)
    
    if lat and lon:
        st.success(f"Coordonnées GPS détectées : Latitude {lat}, Longitude {lon}")
    else:
        st.warning("Aucune information GPS trouvée. Le TIFF ne sera pas géoréférencé.")
        lat, lon = 0, 0  # Valeurs par défaut
    
    tiff_path = "output.tif"
    convert_to_tiff(image_path, tiff_path, lat, lon)
    
    with open(tiff_path, "rb") as f:
        st.download_button("Télécharger le fichier GeoTIFF", f, file_name="image_geotiff.tif")
    
    os.remove(image_path)
    os.remove(tiff_path)
