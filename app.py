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
    """Extrait les coordonnées GPS d'une image si disponibles.
    image_file doit être un objet BytesIO."""
    image_file.seek(0)
    tags = exifread.process_file(image_file)
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags['GPS GPSLatitudeRef'].printable.strip()
        lon_ref = tags['GPS GPSLongitudeRef'].printable.strip()
        
        # Conversion en degrés décimaux
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
            
        return lat, lon
    return None, None

def latlon_to_utm(lat, lon):
    """Convertit des coordonnées lat/lon en coordonnées UTM.
    Retourne (utm_x, utm_y, utm_crs)."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs):
    """Convertit une image JPEG en GeoTIFF géoréférencé en UTM.
    utm_center est le centre de l'image en coordonnées UTM (x, y).
    pixel_size est la taille réelle d'un pixel en mètres."""
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

st.title("Convertisseur JPEG → GeoTIFF avec métadonnées GPS et échelle réelle")

uploaded_files = st.file_uploader("Choisissez une ou plusieurs images JPEG", type=['jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    utm_coords = []
    gps_data = []   # (lat, lon) pour chaque image
    images_data = []  # enregistrement des données binaires pour chaque image

    # Parcours de chaque image téléversée
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
    
    if len(utm_coords) == 0:
        st.error("Aucune image avec des informations GPS valides n'a été trouvée.")
    else:
        # Pour la conversion, nous utilisons la première image comme référence
        ref_image_bytes = images_data[0]
        ref_file = io.BytesIO(ref_image_bytes)
        ref_img = Image.open(ref_file)
        ref_width, ref_height = ref_img.size
        
        # Récupération des coordonnées UTM pour l'image de référence
        ref_utm = utm_coords[0]
        # On récupère le CRS UTM de la première image (les images doivent être dans le même fuseau)
        utm_crs = latlon_to_utm(gps_data[0][0], gps_data[0][1])[2]
        
        # Si plusieurs images sont téléversées, on estime la taille réelle d'un pixel
        if len(utm_coords) > 1:
            # On suppose ici que les images sont disposées horizontalement
            xs = [coord[0] for coord in utm_coords]
            xs_sorted = sorted(xs)
            # La distance moyenne entre centres d'images (en mètres)
            avg_distance = (xs_sorted[-1] - xs_sorted[0]) / (len(xs_sorted) - 1)
            # La largeur d'une image en mètres est alors estimée par avg_distance
            pixel_size = avg_distance / ref_width
        else:
            # Valeur par défaut en cas d'image unique (ex: 1 mètre par pixel)
            pixel_size = 1.0
        
        tiff_path = "output.tif"
        # Conversion de l'image de référence en GeoTIFF
        convert_to_tiff(io.BytesIO(ref_image_bytes), tiff_path, ref_utm, pixel_size, utm_crs)
        
        with open(tiff_path, "rb") as f:
            st.download_button("Télécharger le fichier GeoTIFF", f, file_name="image_geotiff.tif")
        
        os.remove(tiff_path)
