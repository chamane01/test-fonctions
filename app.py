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

def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs):
    """
    Convertit une image JPEG en GeoTIFF géoréférencé en UTM.
    - utm_center : (x, y) du centre en coordonnées UTM
    - pixel_size : taille d'un pixel en mètres
    - utm_crs    : code EPSG (ex: 'EPSG:32632')
    """
    img = Image.open(image_file)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Coordonnées du coin supérieur gauche (x_min, y_max)
    x_min = utm_center[0] - (width / 2) * pixel_size
    y_max = utm_center[1] + (height / 2) * pixel_size  # car y décroit vers le bas dans l'image
    
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

st.title("Conversion JPEG → GeoTIFF sans saisie manuelle")

uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images (JPG/JPEG) avec métadonnées EXIF",
    type=["jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # Liste des informations utiles pour chaque image
    images_info = []
    
    for up_file in uploaded_files:
        file_bytes = up_file.read()
        file_buffer = io.BytesIO(file_bytes)
        
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
        
        # On doit au moins avoir une position GPS pour géoréférencer
        if lat is None or lon is None:
            st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
            continue
        
        # Ouvrir l'image pour connaître sa taille en pixels
        img = Image.open(io.BytesIO(file_bytes))
        img_width, img_height = img.size
        
        # Calculer la largeur du capteur (mm) s’il y a FocalPlaneXResolution
        sensor_width_mm = None
        if fp_x_res and fp_unit:
            # Selon l'unité
            if fp_unit == 2:   # pouces
                sensor_width_mm = (img_width / fp_x_res) * 25.4
            elif fp_unit == 3: # cm
                sensor_width_mm = (img_width / fp_x_res) * 10
            elif fp_unit == 4: # mm
                sensor_width_mm = (img_width / fp_x_res)
            # Sinon, on ne sait pas trop, on laisse None
        
        # Conversion en UTM
        utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
        
        images_info.append({
            "filename": up_file.name,
            "data": file_bytes,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,        # m
            "focal_length": focal_length, # mm
            "sensor_width": sensor_width_mm, # mm
            "utm": (utm_x, utm_y),
            "utm_crs": utm_crs,
            "img_width": img_width,
            "img_height": img_height
        })
    
    # On ne garde que les images avec GPS
    if len(images_info) == 0:
        st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")
    else:
        # On va essayer de trouver la première image qui possède toutes les métadonnées nécessaires
        # (altitude, focal_length, sensor_width).
        ref_image_info = None
        for info in images_info:
            if (info["altitude"] is not None and 
                info["focal_length"] is not None and
                info["sensor_width"] is not None):
                ref_image_info = info
                break
        
        if ref_image_info:
            # On calcule le GSD à partir de cette image "complète"
            pixel_size = compute_gsd(
                altitude=ref_image_info["altitude"],
                focal_length_mm=ref_image_info["focal_length"],
                sensor_width_mm=ref_image_info["sensor_width"],
                image_width_px=ref_image_info["img_width"]
            )
            st.success(
                f"Image de référence : {ref_image_info['filename']} \n\n"
                f"GSD calculé = {pixel_size:.4f} m/pixel"
            )
        else:
            # Personne n'a toutes les métadonnées
            st.warning(
                "Aucune image ne possède toutes les métadonnées nécessaires (Altitude, Focale, "
                "Largeur de capteur). On tente un calcul de secours."
            )
            
            # Fallback : si on a au moins 2 images, on essaye d'estimer l'échelle
            if len(images_info) > 1:
                # On va prendre toutes les images, calculer la distance en UTM
                # entre la première et la dernière (pour simplifier).
                
                # Tri par X UTM (ou Y, c’est arbitraire)
                images_sorted = sorted(images_info, key=lambda x: x["utm"][0])
                first_img = images_sorted[0]
                last_img  = images_sorted[-1]
                
                # Distance en UTM (2D)
                dx = last_img["utm"][0] - first_img["utm"][0]
                dy = last_img["utm"][1] - first_img["utm"][1]
                dist_m = math.sqrt(dx*dx + dy*dy)
                
                # Nombre d'images dans la série
                nb_images = len(images_sorted)
                
                # Hypothèse : (nb_images - 1) * largeur_image_pixels correspond à dist_m
                # => GSD = dist_m / [ (nb_images - 1) * largeur_image_pixels ]
                # On suppose que toutes les images ont la même largeur en pixels
                # (c’est grossier, mais c’est un fallback)
                
                # On prend la largeur de la première image pour référence
                if nb_images > 1:
                    ref_width_px = images_sorted[0]["img_width"]
                    if nb_images == 1:
                        # (cas impossible car nb_images > 1)
                        pixel_size = 1.0
                    else:
                        pixel_size = dist_m / ((nb_images - 1) * ref_width_px)
                        
                    st.info(
                        "Estimation de l'échelle par la distance UTM entre la première et la dernière image.\n"
                        f"GSD estimé = {pixel_size:.4f} m/pixel"
                    )
                else:
                    # 1 seule image => impossible d'estimer
                    pixel_size = 1.0
            else:
                # 1 seule image sans métadonnées suffisantes => on fixe un GSD arbitraire
                pixel_size = 1.0
                st.info("Une seule image, métadonnées incomplètes : on fixe un GSD = 1.0 m/pixel.")
        
        # À ce stade, on a un pixel_size défini soit par le calcul principal, soit par le fallback
        # On va convertir la première image de la liste (ou la ref_image si on veut)
        # en GeoTIFF comme démonstration.
        
        # Si on a utilisé ref_image_info, on s'en sert. Sinon, on prend la première
        final_ref = ref_image_info if ref_image_info else images_info[0]
        
        # Sortie en GeoTIFF
        output_path = "output.tif"
        convert_to_tiff(
            image_file=io.BytesIO(final_ref["data"]),
            output_path=output_path,
            utm_center=final_ref["utm"],
            pixel_size=pixel_size,
            utm_crs=final_ref["utm_crs"]
        )
        
        st.success(f"Image {final_ref['filename']} convertie en GeoTIFF.")
        
        # Proposer le téléchargement
        with open(output_path, "rb") as f:
            st.download_button(
                label="Télécharger le GeoTIFF",
                data=f,
                file_name="image_geotiff.tif",
                mime="image/tiff"
            )
        
        # Nettoyage
        os.remove(output_path)
