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
from affine import Affine

def extract_exif_info(image_file):
    """
    Extrait les informations EXIF d'une image : 
    - GPS (lat, lon, altitude)
    - EXIF FocalLength
    - FocalPlaneXResolution et FocalPlaneResolutionUnit (pour calculer la largeur du capteur)
    Renvoie (lat, lon, altitude, focal_length, fp_x_res, fp_unit) ou None si non disponibles.
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
            
            # Gestion N/S et E/W
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
        # fp_unit : 2 = pouces, 3 = cm, 4 = mm, etc.
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
    # Convertir mm -> m
    focal_length_m = focal_length_mm / 1000.0
    sensor_width_m = sensor_width_mm / 1000.0
    
    # Formule GSD = (Altitude * LargeurCapteur) / (Focale * NbPixels)
    gsd = (altitude * sensor_width_m) / (focal_length_m * image_width_px)
    return gsd

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs, rotation_angle=0):
    """
    Convertit une image JPEG en GeoTIFF géoréférencé en UTM avec correction d'orientation.
    La transformation affine est construite de façon à :
    - Centrer l'image sur son centre (dimensions en pixels)
    - Appliquer l'échelle (taille d'un pixel)
    - Appliquer une rotation (pour aligner la trajectoire avec le nord)
    - Positionner le centre sur les coordonnées UTM calculées
    """
    # Correction d'orientation grâce aux métadonnées EXIF
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Construction de la transformation affine
    # On souhaite : T = Translation(utm_center) * Rotation(rotation_angle) * Scale(pixel_size, -pixel_size) * Translation(-width/2, -height/2)
    center_x, center_y = utm_center
    T1 = Affine.translation(-width/2, -height/2)
    T2 = Affine.scale(pixel_size, -pixel_size)  # y négatif pour que le haut de l'image corresponde au nord
    T3 = Affine.rotation(rotation_angle)         # rotation en degrés (sens trigonométrique)
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

st.title("Conversion JPEG → GeoTIFF avec orientation corrigée")

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
        
        # Vérifier la présence de coordonnées GPS
        if lat is None or lon is None:
            st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
            continue
        
        # Ouvrir l'image pour connaître sa taille en pixels
        img = Image.open(io.BytesIO(file_bytes))
        img_width, img_height = img.size
        
        # Calcul de la largeur du capteur (mm) si FocalPlaneXResolution disponible
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
            "altitude": altitude,        # en m
            "focal_length": focal_length, # en mm
            "sensor_width": sensor_width_mm, # en mm
            "utm": (utm_x, utm_y),
            "utm_crs": utm_crs,
            "img_width": img_width,
            "img_height": img_height
        })
    
    if len(images_info) == 0:
        st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")
    else:
        # Choisir une image de référence possédant les métadonnées nécessaires
        ref_image_info = None
        for info in images_info:
            if (info["altitude"] is not None and 
                info["focal_length"] is not None and
                info["sensor_width"] is not None):
                ref_image_info = info
                break
        
        if ref_image_info:
            # Calcul du GSD (m/pixel) à partir de l'image de référence
            pixel_size_calc = compute_gsd(
                altitude=ref_image_info["altitude"],
                focal_length_mm=ref_image_info["focal_length"],
                sensor_width_mm=ref_image_info["sensor_width"],
                image_width_px=ref_image_info["img_width"]
            )
            st.success(
                f"Image de référence : {ref_image_info['filename']} \n\n"
                f"GSD calculé = {pixel_size_calc:.4f} m/pixel"
            )
        else:
            st.warning(
                "Aucune image ne possède toutes les métadonnées nécessaires (Altitude, Focale, "
                "Largeur de capteur)."
            )
        
        # Permettre à l'utilisateur de choisir la résolution spatiale (en m/pixel)
        pixel_size = st.number_input(
            "Choisissez la résolution spatiale (m/pixel) :", 
            min_value=0.001, 
            value=0.03, 
            step=0.001, 
            format="%.3f"
        )
        st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
        
        # Calcul de l'angle de trajectoire si plusieurs images sont fournies
        if len(images_info) >= 2:
            first_img = images_info[0]
            last_img = images_info[-1]
            dx = last_img["utm"][0] - first_img["utm"][0]
            dy = last_img["utm"][1] - first_img["utm"][1]
            # L'angle mesuré à partir du nord (0° = nord, 90° = est, -90° = ouest)
            flight_angle = math.degrees(math.atan2(dx, dy))
            st.info(f"Angle de trajectoire calculé : {flight_angle:.1f}° (0° = nord)")
        else:
            flight_angle = 0
            st.info("Angle de trajectoire non calculable (une seule image) → 0°")
        
        # Pour orienter l'image de façon que le nord soit en haut, on applique une rotation de -flight_angle
        rotation_correction = -flight_angle
        st.info(f"Correction d'orientation appliquée : {rotation_correction:.1f}°")
        
        # On utilise l'image de référence si disponible, sinon la première image exploitable
        final_ref = ref_image_info if ref_image_info else images_info[0]
        
        output_path = "output.tif"
        convert_to_tiff(
            image_file=io.BytesIO(final_ref["data"]),
            output_path=output_path,
            utm_center=final_ref["utm"],
            pixel_size=pixel_size,
            utm_crs=final_ref["utm_crs"],
            rotation_angle=rotation_correction
        )
        
        st.success(f"Image {final_ref['filename']} convertie en GeoTIFF.")
        
        # Vérification et affichage des métadonnées du GeoTIFF créé
        with rasterio.open(output_path) as src:
            st.write("**Méta-données GeoTIFF**")
            st.write("CRS :", src.crs)
            st.write("Transform :", src.transform)
        
        # Proposer le téléchargement
        with open(output_path, "rb") as f:
            st.download_button(
                label="Télécharger le GeoTIFF",
                data=f,
                file_name="image_geotiff.tif",
                mime="image/tiff"
            )
        
        os.remove(output_path)
