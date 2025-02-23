import streamlit as st
import rasterio
from rasterio.transform import from_origin
from PIL import Image, ImageOps
import exifread
import numpy as np
import os
from pyproj import Transformer
import io
import math

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
                   float(lat_vals[2].num) / lon_vals[2].den / 3600)
            
            # Correction N/S et E/W
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

def compute_gsd(altitude, focal_length_mm, sensor_width_mm, image_width_px):
    """
    Calcule le GSD (m/pixel) à partir de :
    - altitude (m)
    - focal_length_mm (mm)
    - sensor_width_mm (mm)
    - image_width_px (pixels)
    """
    focal_length_m = focal_length_mm / 1000.0
    sensor_width_m = sensor_width_mm / 1000.0
    gsd = (altitude * sensor_width_m) / (focal_length_m * image_width_px)
    return gsd

def convert_to_tiff_axis_aligned(image_file, output_path, utm_center, pixel_size, utm_crs, rotation_angle=0):
    """
    Convertit une image JPEG en GeoTIFF orienté avec le nord en haut.
    La rotation (en degrés) est appliquée sur l'image avec expand=True afin d'obtenir
    un rendu recadré sans le cadre issu de la position d'affichage d'origine.
    
    La transformation géoréférencée est recalculée pour que le centre de l'image (après rotation)
    corresponde à la coordonnée UTM fournie.
    """
    # Ouvrir et corriger l'orientation EXIF
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    
    # Appliquer la rotation si nécessaire (PIL tourne dans le sens anti-horaire)
    if rotation_angle != 0:
        img_rot = img.rotate(rotation_angle, expand=True)
    else:
        img_rot = img
        
    # Nouvelle taille de l'image après rotation
    new_width, new_height = img_rot.size
    img_array = np.array(img_rot)
    
    # Recalcule de la transformation pour un TIFF axis-aligné
    center_x, center_y = utm_center
    x_min = center_x - (new_width / 2) * pixel_size
    y_max = center_y + (new_height / 2) * pixel_size
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=new_height,
        width=new_width,
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

st.title("Conversion JPEG → GeoTIFF sans cadre, avec sélection et résolution ajustable")

uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    images_info = []
    
    # Parcours des fichiers téléchargés
    for up_file in uploaded_files:
        file_bytes = up_file.read()
        file_buffer = io.BytesIO(file_bytes)
        
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
        
        if lat is None or lon is None:
            st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
            continue
        
        img = Image.open(io.BytesIO(file_bytes))
        img_width, img_height = img.size
        
        # Calcul de la largeur du capteur (en mm) si disponible
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
            "altitude": altitude,           # en m
            "focal_length": focal_length,     # en mm
            "sensor_width": sensor_width_mm,  # en mm
            "utm": (utm_x, utm_y),
            "utm_crs": utm_crs,
            "img_width": img_width,
            "img_height": img_height
        })
    
    if len(images_info) == 0:
        st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")
    else:
        # Sélecteur pour choisir l'image à exporter
        filenames = [info["filename"] for info in images_info]
        selected_filename = st.selectbox("Sélectionnez l'image à exporter en GeoTIFF", filenames)
        selected_image = next(item for item in images_info if item["filename"] == selected_filename)
        
        # Affichage d'une information sur le GSD calculé (si possible)
        if (selected_image["altitude"] is not None and 
            selected_image["focal_length"] is not None and 
            selected_image["sensor_width"] is not None):
            pixel_size_calc = compute_gsd(
                altitude=selected_image["altitude"],
                focal_length_mm=selected_image["focal_length"],
                sensor_width_mm=selected_image["sensor_width"],
                image_width_px=selected_image["img_width"]
            )
            st.success(f"Image de référence : {selected_image['filename']}  |  GSD calculé = {pixel_size_calc:.4f} m/pixel")
        else:
            st.warning("Les métadonnées nécessaires au calcul du GSD ne sont pas complètes pour cette image.")
        
        # Curseur pour ajuster la résolution d'export (en m/pixel)
        pixel_size = st.slider(
            "Ajustez la résolution spatiale (m/pixel) :",
            min_value=0.001,
            max_value=0.1,
            value=0.03,
            step=0.001,
            format="%.3f"
        )
        st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
        
        # Calcul de l'angle de trajectoire (si plusieurs images sont disponibles)
        if len(images_info) >= 2:
            first_img = images_info[0]
            last_img = images_info[-1]
            dx = last_img["utm"][0] - first_img["utm"][0]
            dy = last_img["utm"][1] - first_img["utm"][1]
            flight_angle = math.degrees(math.atan2(dx, dy))
            st.info(f"Angle de trajectoire calculé : {flight_angle:.1f}° (0° = nord)")
        else:
            flight_angle = 0
            st.info("Angle de trajectoire non calculable (une seule image) → 0°")
        
        # Correction pour que le nord soit en haut (on applique -flight_angle)
        rotation_correction = -flight_angle
        st.info(f"Correction d'orientation appliquée : {rotation_correction:.1f}°")
        
        # Export du TIFF à partir de l'image sélectionnée et de la résolution choisie
        output_path = "output.tif"
        convert_to_tiff_axis_aligned(
            image_file=io.BytesIO(selected_image["data"]),
            output_path=output_path,
            utm_center=selected_image["utm"],
            pixel_size=pixel_size,
            utm_crs=selected_image["utm_crs"],
            rotation_angle=rotation_correction
        )
        
        st.success(f"Image {selected_image['filename']} convertie en GeoTIFF.")
        
        # Affichage des métadonnées du GeoTIFF créé
        with rasterio.open(output_path) as src:
            st.write("**Méta-données GeoTIFF**")
            st.write("CRS :", src.crs)
            st.write("Transform :", src.transform)
        
        # Bouton de téléchargement du GeoTIFF
        with open(output_path, "rb") as f:
            st.download_button(
                label="Télécharger le GeoTIFF",
                data=f,
                file_name="image_geotiff.tif",
                mime="image/tiff"
            )
        
        os.remove(output_path)
