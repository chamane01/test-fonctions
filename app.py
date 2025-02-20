import streamlit as st
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import exifread
import numpy as np
import os
from pyproj import Transformer
import io

def extract_gps_info(image_file):
    """Extrait les coordonnées GPS d'une image si disponibles."""
    image_file.seek(0)
    tags = exifread.process_file(image_file)
    
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags['GPS GPSLatitudeRef'].printable.strip()
        lon_ref = tags['GPS GPSLongitudeRef'].printable.strip()
        
        lat = (lat_vals[0].num / lat_vals[0].den +
               lat_vals[1].num / lat_vals[1].den / 60 +
               lat_vals[2].num / lat_vals[2].den / 3600)
        lon = (lon_vals[0].num / lon_vals[0].den +
               lon_vals[1].num / lon_vals[1].den / 60 +
               lon_vals[2].num / lon_vals[2].den / 3600)
        
        if lat_ref.upper() == 'S': lat = -lat
        if lon_ref.upper() == 'W': lon = -lon
            
        return lat, lon
    return None, None

def latlon_to_utm(lat, lon):
    """Convertit des coordonnées lat/lon en coordonnées UTM."""
    zone = int((lon + 180) / 6) + 1
    utm_crs = f"EPSG:326{zone}" if lat >= 0 else f"EPSG:327{zone}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

def calculate_distance_utm(coord1, coord2):
    """Calcule la distance en mètres entre deux points en UTM."""
    return np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs):
    """Convertit une image en GeoTIFF avec échelle correcte."""
    img = Image.open(image_file)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    top_left_x = utm_center[0] - (width / 2) * pixel_size
    top_left_y = utm_center[1] + (height / 2) * pixel_size

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

st.title("Convertisseur JPEG → GeoTIFF avec échelle correcte")

uploaded_files = st.file_uploader("Choisissez plusieurs images JPEG", type=['jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    utm_coords = []
    gps_data = []
    images_data = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        images_data.append(file_bytes)
        file_buffer = io.BytesIO(file_bytes)
        lat, lon = extract_gps_info(file_buffer)
        
        if lat is not None and lon is not None:
            utm_x, utm_y, _ = latlon_to_utm(lat, lon)
            utm_coords.append((utm_x, utm_y))
            gps_data.append((lat, lon))
        else:
            st.warning(f"Aucune info GPS trouvée pour {uploaded_file.name}")

    if len(utm_coords) < 2:
        st.error("Il faut au moins 2 images avec GPS pour estimer l'échelle.")
    else:
        ref_image_bytes = images_data[0]
        ref_file = io.BytesIO(ref_image_bytes)
        ref_img = Image.open(ref_file)
        ref_width, ref_height = ref_img.size

        # Calcul de la distance entre la première et la dernière image
        total_distance = calculate_distance_utm(utm_coords[0], utm_coords[-1])
        pixel_distance = ref_width * (len(utm_coords) - 1)

        # Calcul du pixel_size en mètre
        pixel_size = total_distance / pixel_distance

        # Création du GeoTIFF avec échelle correcte
        tiff_path = "output.tif"
        convert_to_tiff(io.BytesIO(ref_image_bytes), tiff_path, utm_coords[0], pixel_size, latlon_to_utm(gps_data[0][0], gps_data[0][1])[2])

        with open(tiff_path, "rb") as f:
            st.download_button("Télécharger le GeoTIFF", f, file_name="image_geotiff.tif")

        os.remove(tiff_path)
