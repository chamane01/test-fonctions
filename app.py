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

# Conversion en GeoTIFF sur disque
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
        # Sélection d'une image pour conversion GeoTIFF individuelle
        selected_filename = st.selectbox(
            "Sélectionnez l'image à convertir en GeoTIFF",
            options=[info["filename"] for info in images_info]
        )
        selected_image_info = next(info for info in images_info if info["filename"] == selected_filename)
        
        # Calcul de l'angle de trajectoire (si plusieurs images)
        if len(images_info) >= 2:
            idx = next(i for i, info in enumerate(images_info) if info["filename"] == selected_filename)
            if idx == 0:
                dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
            elif idx == len(images_info) - 1:
                dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
            else:
                dx = images_info[idx+1]["utm"][0] - images_info[idx-1]["utm"][0]
                dy = images_info[idx+1]["utm"][1] - images_info[idx-1]["utm"][1]
            flight_angle = math.degrees(math.atan2(dx, dy))
            st.info(f"Angle de trajectoire local calculé : {flight_angle:.1f}° (0° = nord)")
        else:
            flight_angle = 0
            st.info("Angle de trajectoire non calculable (une seule image) → 0°")
        
        # Calcul du GSD si possible
        if (selected_image_info["altitude"] is not None and 
            selected_image_info["focal_length"] is not None and 
            selected_image_info["sensor_width"] is not None):
            pixel_size_calc = compute_gsd(
                altitude=selected_image_info["altitude"],
                focal_length_mm=selected_image_info["focal_length"],
                sensor_width_mm=selected_image_info["sensor_width"],
                image_width_px=selected_image_info["img_width"]
            )
            st.success(
                f"Image sélectionnée : {selected_image_info['filename']}\n\n"
                f"GSD calculé = {pixel_size_calc:.4f} m/pixel"
            )
        else:
            st.warning("L'image sélectionnée ne possède pas toutes les métadonnées nécessaires pour calculer automatiquement le GSD.")
        
        pixel_size = st.number_input(
            "Choisissez la résolution spatiale (m/pixel) :", 
            min_value=0.001, 
            value=0.03, 
            step=0.001, 
            format="%.3f"
        )
        st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
        
        # Sélecteur élargi pour le facteur de redimensionnement (utilisé pour la conversion standard)
        scale_options = ["/50", "/20", "/10", "/5", "/2", "0", "*2", "*5", "*10", "*20", "*50"]
        selected_scale = st.select_slider("Choisissez le facteur de redimensionnement de l'image", options=scale_options, value="0")
        mapping = {
            "/50": 1/50,
            "/20": 1/20,
            "/10": 1/10,
            "/5": 1/5,
            "/2": 1/2,
            "0": 1,
            "*2": 2,
            "*5": 5,
            "*10": 10,
            "*20": 20,
            "*50": 50
        }
        scaling_factor = mapping[selected_scale]
        st.info(f"Facteur de redimensionnement sélectionné : {scaling_factor}")
        
        effective_pixel_size = pixel_size / scaling_factor
        st.info(f"Résolution spatiale effective après redimensionnement : {effective_pixel_size*100:.1f} cm/pixel")
        
        rotation_correction = -flight_angle
        st.info(f"Correction d'orientation appliquée : {rotation_correction:.1f}°")
        
        # Conversion et téléchargement de l'image GeoTIFF sélectionnée
        output_path = "output.tif"
        convert_to_tiff(
            image_file=io.BytesIO(selected_image_info["data"]),
            output_path=output_path,
            utm_center=selected_image_info["utm"],
            pixel_size=pixel_size,
            utm_crs=selected_image_info["utm_crs"],
            rotation_angle=rotation_correction,
            scaling_factor=scaling_factor
        )
        
        st.success(f"Image {selected_image_info['filename']} convertie en GeoTIFF.")
        with rasterio.open(output_path) as src:
            st.write("**Méta-données GeoTIFF**")
            st.write("CRS :", src.crs)
            st.write("Transform :", src.transform)
        
        with open(output_path, "rb") as f:
            st.download_button(
                label="Télécharger le GeoTIFF",
                data=f,
                file_name="image_geotiff.tif",
                mime="image/tiff"
            )
        os.remove(output_path)
        
        # Téléchargement groupé de toutes les images en GeoTIFF
        if st.button("Convertir et télécharger **toutes** les images en GeoTIFF"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
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
                    tiff_bytes = convert_to_tiff_in_memory(
                        image_file=io.BytesIO(info["data"]),
                        pixel_size=pixel_size,
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=scaling_factor
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
        # Option supplémentaire GeoTIFF x2 :
        # Pour cet export, le redimensionnement n'est plus soumis au sélecteur,
        # le facteur de redimensionnement est fixé à 1/3 et la résolution spatiale est multipliée par 2.
        if st.button("Convertir et télécharger **toutes** les images en GeoTIFF x2"):
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
                        pixel_size=pixel_size * 2,  # Application d'un facteur 2 sur la résolution spatiale
                        utm_center=info["utm"],
                        utm_crs=info["utm_crs"],
                        rotation_angle=-flight_angle_i,
                        scaling_factor=1/3        # Facteur constant : division par 3
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
        
        # --- Export groupé de toutes les images en JPEG avec métadonnées de cadre ---
        if st.button("Convertir et télécharger **toutes** les images en JPEG avec métadonnées de cadre"):
            zip_buffer_jpeg = io.BytesIO()
            with zipfile.ZipFile(zip_buffer_jpeg, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, info in enumerate(images_info):
                    # Calcul de l'angle de trajectoire pour cette image
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
                    
                    # Traitement de l'image avec redimensionnement et rotation
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
                            # On essaie d'utiliser piexif.helper.dump
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.dump(user_comment, encoding="unicode")
                        except AttributeError:
                            # Fonction de repli si helper.dump n'est pas disponible
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
