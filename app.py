import streamlit as st
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from PIL import Image, ImageOps
import exifread
import numpy as np
import os
from pyproj import Transformer
import io
import math
from affine import Affine
import zipfile

# Fonction d'extraction des métadonnées EXIF
def extract_exif_info(image_file):
    """
    Extrait les informations EXIF d'une image : 
    - GPS (lat, lon, altitude)
    - EXIF FocalLength
    - FocalPlaneXResolution et FocalPlaneResolutionUnit (pour calculer la largeur du capteur)
    Renvoie (lat, lon, altitude, focal_length, fp_x_res, fp_unit).
    """
    image_file.seek(0)
    tags = exifread.process_file(image_file)
    
    lat = lon = altitude = focal_length = None
    fp_x_res = fp_unit = None
    
    # --- GPS Latitude / Longitude ---
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags.get('GPS GPSLatitudeRef', None)
        lon_ref = tags.get('GPS GPSLongitudeRef', None)
        
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

# Conversion lat/lon en UTM
def latlon_to_utm(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

# Calcul du GSD (m/pixel)
def compute_gsd(altitude, focal_length_mm, sensor_width_mm, image_width_px):
    focal_length_m = focal_length_mm / 1000.0
    sensor_width_m = sensor_width_mm / 1000.0
    gsd = (altitude * sensor_width_m) / (focal_length_m * image_width_px)
    return gsd

# Conversion en GeoTIFF sur disque (utilisé dans la conversion individuelle si besoin)
def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs, rotation_angle=0, scaling_factor=1):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    orig_width, orig_height = img.size
    new_width = int(orig_width * scaling_factor)
    new_height = int(orig_height * scaling_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    effective_pixel_size = pixel_size / scaling_factor

    center_x, center_y = utm_center
    T1 = Affine.translation(-width/2, -height/2)
    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
    T3 = Affine.rotation(rotation_angle)
    T4 = Affine.translation(center_x, center_y)
    transform = T4 * T3 * T2 * T1

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

# Conversion en GeoTIFF en mémoire (retourne des bytes)
def convert_to_tiff_in_memory(image_file, pixel_size, utm_center, utm_crs, rotation_angle=0, scaling_factor=1):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    orig_width, orig_height = img.size
    new_width = int(orig_width * scaling_factor)
    new_height = int(orig_height * scaling_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    effective_pixel_size = pixel_size / scaling_factor

    center_x, center_y = utm_center
    T1 = Affine.translation(-width/2, -height/2)
    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
    T3 = Affine.rotation(rotation_angle)
    T4 = Affine.translation(center_x, center_y)
    transform = T4 * T3 * T2 * T1

    memfile = MemoryFile()
    with memfile.open(
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
    return memfile.read()

st.title("Conversion JPEG → GeoTIFF & Export JPEG avec métadonnées de cadre")

uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    images_info = []
    
    for up_file in uploaded_files:
        file_bytes = up_file.read()
        file_buffer = io.BytesIO(file_bytes)
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
        if lat is None or lon is None:
            st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
            continue
        
        img = Image.open(io.BytesIO(file_bytes))
        img_width, img_height = img.size
        
        sensor_width_mm = None
        if fp_x_res and fp_unit:
            if fp_unit == 2:   # pouces
                sensor_width_mm = (img_width / fp_x_res) * 25.4
            elif fp_unit == 3: # cm
                sensor_width_mm = (img_width / fp_x_res) * 10
            elif fp_unit == 4: # mm
                sensor_width_mm = (img_width / fp_x_res)
        
        utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
        
        images_info.append({
            "filename": up_file.name,
            "data": file_bytes,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "focal_length": focal_length,
            "sensor_width": sensor_width_mm,
            "utm": (utm_x, utm_y),
            "utm_crs": utm_crs,
            "img_width": img_width,
            "img_height": img_height
        })
    
    if len(images_info) == 0:
        st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")
    else:
        # Saisie de la résolution spatiale
        pixel_size = st.number_input(
            "Choisissez la résolution spatiale (m/pixel) :", 
            min_value=0.001, 
            value=0.03, 
            step=0.001, 
            format="%.3f"
        )
        st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
        
        # -------------------------------
        # Conversion groupée en GeoTIFF (Configuration 1) avec scaling_factor fixe = 1/5 (correspondant à -5)
        if st.button("Convertir et télécharger toutes les images en GeoTIFF"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    # Calcul de l'angle de trajectoire pour chaque image
                    if len(images_info) >= 2:
                        if i == 0:
                            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                        elif i == len(images_info) - 1:
                            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                        else:
                            dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                            dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                        flight_angle_i = math.degrees(math.atan2(dx, dy))
                    else:
                        flight_angle_i = 0
                    tiff_bytes = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/5  # Facteur fixe -5
                    )
                    output_filename = info["filename"].rsplit(".", 1)[0] + "_geotiff.tif"
                    zip_file.writestr(output_filename, tiff_bytes)
            zip_buffer.seek(0)
            st.download_button(
                label="Télécharger toutes les images GeoTIFF (ZIP)",
                data=zip_buffer,
                file_name="images_geotiff.zip",
                mime="application/zip"
            )
        
        # -------------------------------
        # Conversion groupée en GeoTIFF x2 (Configuration 2) avec scaling_factor fixe = 1/3 et résolution spatiale multipliée par 2
        if st.button("Convertir et télécharger toutes les images en GeoTIFF x2"):
            zip_buffer_geotiff_x2 = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_geotiff_x2, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    if len(images_info) >= 2:
                        if i == 0:
                            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                        elif i == len(images_info) - 1:
                            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                        else:
                            dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                            dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                        flight_angle_i = math.degrees(math.atan2(dx, dy))
                    else:
                        flight_angle_i = 0
                    tiff_bytes_x2 = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size * 2,  # Résolution spatiale doublée
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/3  # Facteur fixe -3
                    )
                    output_filename = info["filename"].rsplit(".", 1)[0] + "_geotiff_x2.tif"
                    zip_file.writestr(output_filename, tiff_bytes_x2)
            zip_buffer_geotiff_x2.seek(0)
            st.download_button(
                label="Télécharger toutes les images GeoTIFF x2 (ZIP)",
                data=zip_buffer_geotiff_x2,
                file_name="images_geotiff_x2.zip",
                mime="application/zip"
            )
        
        # -------------------------------
        # Conversion groupée en JPEG avec métadonnées (Configuration JPEG) avec scaling_factor fixe = 1 (pas de redimensionnement)
        if st.button("Convertir et télécharger toutes les images en JPEG avec métadonnées de cadre"):
            zip_buffer_jpeg = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_jpeg, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    if len(images_info) >= 2:
                        if i == 0:
                            dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                            dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                        elif i == len(images_info) - 1:
                            dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                            dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                        else:
                            dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                            dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                        flight_angle_i = math.degrees(math.atan2(dx, dy))
                    else:
                        flight_angle_i = 0
                    rotation_angle_i = -flight_angle_i
                    
                    # Pour cette configuration, aucun redimensionnement (scaling_factor = 1)
                    scaling_factor = 1
                    img = Image.open(io.BytesIO(info["data"]))
                    img = ImageOps.exif_transpose(img)
                    orig_width, orig_height = img.size
                    new_width = int(orig_width * scaling_factor)
                    new_height = int(orig_height * scaling_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    effective_pixel_size = pixel_size / scaling_factor
                    center_x, center_y = info["utm"]
                    T1 = Affine.translation(-new_width/2, -new_height/2)
                    T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
                    T3 = Affine.rotation(rotation_angle_i)
                    T4 = Affine.translation(center_x, center_y)
                    transform = T4 * T3 * T2 * T1
                    
                    # Calcul des coordonnées des 4 coins du cadre
                    corners = [
                        (-new_width/2, -new_height/2),
                        (new_width/2, -new_height/2),
                        (new_width/2, new_height/2),
                        (-new_width/2, new_height/2)
                    ]
                    corner_coords = []
                    for corner in corners:
                        x, y = transform * corner
                        corner_coords.append((x, y))
                    
                    metadata_str = f"Frame Coordinates: {corner_coords}"
                    
                    # Injection des métadonnées dans l'EXIF via piexif
                    try:
                        import piexif
                        if "exif" in img.info:
                            exif_dict = piexif.load(img.info["exif"])
                        else:
                            exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
                        user_comment = metadata_str
                        try:
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.dump(user_comment, encoding="unicode")
                        except AttributeError:
                            prefix = b"UNICODE\0"
                            encoded_comment = user_comment.encode("utf-16")
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + encoded_comment
                        exif_bytes = piexif.dump(exif_dict)
                    except ImportError:
                        st.error("La librairie piexif est requise pour ajouter des métadonnées JPEG. Veuillez l'installer.")
                        exif_bytes = None
                    
                    jpeg_buffer = io.BytesIO()
                    if exif_bytes:
                        img.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
                    else:
                        img.save(jpeg_buffer, format="JPEG")
                    jpeg_bytes = jpeg_buffer.getvalue()
                    output_filename = info["filename"].rsplit(".", 1)[0] + "_with_frame_coords.jpg"
                    zip_file.writestr(output_filename, jpeg_bytes)
            zip_buffer_jpeg.seek(0)
            st.download_button(
                label="Télécharger toutes les images JPEG avec métadonnées de cadre (ZIP)",
                data=zip_buffer_jpeg,
                file_name="images_with_frame_coords.zip",
                mime="application/zip"
            )
