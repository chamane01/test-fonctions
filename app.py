import streamlit as st
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import exifread
import numpy as np
import cv2
import os
from pyproj import Transformer
import io
import math

#############################
# 1. Extraction des métadonnées EXIF
#############################
def extract_exif_info(image_file):
    """
    Extrait des informations EXIF utiles :
      - GPS : Latitude, Longitude, Altitude (supposée AGL)
      - Focale (EXIF FocalLength)
      - Résolution focale (FocalPlaneXResolution et FocalPlaneResolutionUnit) pour estimer la largeur du capteur
      - Direction de l'image (GPSImgDirection) pour déterminer le yaw
    Renvoie un dictionnaire.
    """
    image_file.seek(0)
    tags = exifread.process_file(image_file, details=False)
    
    # GPS Latitude / Longitude
    lat = lon = None
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_ref = tags.get('GPS GPSLongitudeRef')
        if lat_vals and lon_vals and lat_ref and lon_ref:
            lat = (float(lat_vals[0].num)/lat_vals[0].den +
                   float(lat_vals[1].num)/lat_vals[1].den/60 +
                   float(lat_vals[2].num)/lat_vals[2].den/3600)
            lon = (float(lon_vals[0].num)/lon_vals[0].den +
                   float(lon_vals[1].num)/lon_vals[1].den/60 +
                   float(lon_vals[2].num)/lon_vals[2].den/3600)
            if lat_ref.printable.strip().upper() == 'S':
                lat = -lat
            if lon_ref.printable.strip().upper() == 'W':
                lon = -lon

    # Altitude
    altitude = None
    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den

    # Focale
    focal_length = None
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den

    # Résolution focale (pour estimer largeur capteur)
    fp_x_res = None
    fp_unit = None
    if 'EXIF FocalPlaneXResolution' in tags and 'EXIF FocalPlaneResolutionUnit' in tags:
        fp_res_tag = tags['EXIF FocalPlaneXResolution']
        fp_unit_tag = tags['EXIF FocalPlaneResolutionUnit']
        fp_x_res = float(fp_res_tag.values[0].num) / fp_res_tag.values[0].den
        fp_unit = int(fp_unit_tag.values[0])
    
    # Direction de l'image (yaw) en degrés
    gps_img_direction = None
    if 'GPS GPSImgDirection' in tags:
        dir_tag = tags['GPS GPSImgDirection']
        gps_img_direction = float(dir_tag.values[0].num) / dir_tag.values[0].den

    return {
        'lat': lat,
        'lon': lon,
        'altitude': altitude,           # en m (supposée AGL)
        'focal_length': focal_length,     # en mm
        'fp_x_res': fp_x_res,
        'fp_unit': fp_unit,
        'gps_img_direction': gps_img_direction
    }

#############################
# 2. Conversion GPS -> UTM
#############################
def latlon_to_utm(lat, lon):
    """
    Convertit lat/lon (WGS84) en coordonnées UTM.
    Renvoie (utm_x, utm_y, utm_crs)
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        utm_crs = f"EPSG:326{zone:02d}"
    else:
        utm_crs = f"EPSG:327{zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, utm_crs

#############################
# 3. Modèle de la caméra
#############################
def compute_camera_matrix(focal_length, sensor_width, image_width, image_height):
    """
    Calcule la matrice intrinsèque K.
    focal_length et sensor_width en mm, image_width en pixels.
    On suppose que le centre optique est au centre de l'image.
    """
    # Conversion focale en pixels
    f_pixels = (focal_length / sensor_width) * image_width
    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array([[f_pixels, 0, cx],
                  [0, f_pixels, cy],
                  [0, 0, 1]])
    return K

def rotation_matrix_from_yaw(yaw_deg):
    """
    Construit la matrice de rotation à partir du yaw en degrés.
    Pour un drone orienté vers le sol, on suppose pitch=roll=0.
    On applique ensuite une correction pour que l'axe optique pointe vers le sol.
    """
    yaw = np.deg2rad(yaw_deg)
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0, 0, 1]])
    # Correction : on souhaite que l'axe optique, initialement (0,0,1), devienne (0,0,-1)
    R_fixed = np.diag([1, 1, -1])
    R = R_yaw @ R_fixed
    return R

#############################
# 4. Projection de pixels sur le sol (terrain plat, Z=0)
#############################
def image_to_ground(u, v, K, R, T):
    """
    Pour un pixel (u,v) de l'image, calcule son intersection avec le plan sol (Z=0).
    - K : matrice intrinsèque
    - R : matrice de rotation (de la caméra vers le monde)
    - T : position de la caméra en coordonnées monde (X, Y, Z), avec Z = altitude (AGL)
    Renvoie le point (X, Y, 0)
    """
    invK = np.linalg.inv(K)
    pixel_homog = np.array([u, v, 1])
    d_cam = invK @ pixel_homog            # vecteur direction dans le repère caméra
    d_world = R @ d_cam                   # direction dans le repère monde
    # Calcul de lambda tel que : T_z + lambda*d_world_z = 0 (sol : Z=0)
    if d_world[2] == 0:
        lambda_val = 0
    else:
        lambda_val = -T[2] / d_world[2]
    ground_point = T + lambda_val * d_world
    return ground_point

#############################
# 5. Orthorectification de l'image
#############################
def orthorectify_image(image_bytes, exif_data, fallback_sensor_width=6.17):
    """
    Réalise l'orthorectification d'une image à partir de ses métadonnées.
    On considère un terrain plat (Z=0) et on suppose que l'altitude est donnée en AGL.
    Retourne l'image orthorectifiée, sa transformation géoréférencée et le CRS.
    """
    # Chargement de l'image
    pil_img = Image.open(io.BytesIO(image_bytes))
    image = np.array(pil_img)
    img_height, img_width = image.shape[0:2]

    # Vérification des métadonnées essentielles
    lat = exif_data.get('lat')
    lon = exif_data.get('lon')
    altitude = exif_data.get('altitude')  # en m (AGL)
    focal_length = exif_data.get('focal_length')  # en mm
    sensor_width = None
    if exif_data.get('fp_x_res') and exif_data.get('fp_unit'):
        # fp_unit : 2=pouces, 3=cm, 4=mm
        if exif_data['fp_unit'] == 2:
            sensor_width = (img_width / exif_data['fp_x_res']) * 25.4
        elif exif_data['fp_unit'] == 3:
            sensor_width = (img_width / exif_data['fp_x_res']) * 10
        elif exif_data['fp_unit'] == 4:
            sensor_width = (img_width / exif_data['fp_x_res'])
    if sensor_width is None:
        sensor_width = fallback_sensor_width  # valeur par défaut (en mm)

    if None in [lat, lon, altitude, focal_length]:
        st.error("Les métadonnées EXIF essentielles (GPS, altitude, focale) sont manquantes.")
        return None, None, None

    # Conversion en UTM
    utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
    # Position de la caméra en monde (on considère que l'altitude est AGL, donc le sol est à Z=0)
    T = np.array([utm_x, utm_y, altitude])  # altitude en m

    # Matrice intrinsèque de la caméra
    K = compute_camera_matrix(focal_length, sensor_width, img_width, img_height)

    # Rotation
    yaw = exif_data.get('gps_img_direction')
    if yaw is None:
        yaw = 0  # par défaut, aucune rotation horizontale
    R = rotation_matrix_from_yaw(yaw)

    # Calcul des projections sur le sol des 4 coins de l'image
    pts_img = np.array([
        [0, 0],
        [img_width, 0],
        [img_width, img_height],
        [0, img_height]
    ], dtype=np.float32)
    pts_ground = []
    for pt in pts_img:
        gp = image_to_ground(pt[0], pt[1], K, R, T)
        pts_ground.append([gp[0], gp[1]])
    pts_ground = np.array(pts_ground, dtype=np.float32)

    # Calcul du homographie H qui mappe l'image (points source) vers le sol (points destination)
    # H vérifie : pts_ground ~ H * pts_img_homog
    H, status = cv2.findHomography(pts_img, pts_ground)

    # Estimation de la résolution au sol (m/pixel) par différentiel au coin supérieur gauche
    gp_00 = image_to_ground(0, 0, K, R, T)
    gp_10 = image_to_ground(1, 0, K, R, T)
    res_x = np.linalg.norm(np.array(gp_10[:2]) - np.array(gp_00[:2]))
    gp_01 = image_to_ground(0, 1, K, R, T)
    res_y = np.linalg.norm(np.array(gp_01[:2]) - np.array(gp_00[:2]))
    desired_resolution = (res_x + res_y) / 2.0  # en m/pixel

    # Détermination de l'étendue au sol (bounding box) à partir des coins
    min_x, min_y = np.min(pts_ground, axis=0)
    max_x, max_y = np.max(pts_ground, axis=0)
    # Pour la géotransformation, le coin supérieur gauche est (min_x, max_y)
    out_width = int(np.ceil((max_x - min_x) / desired_resolution))
    out_height = int(np.ceil((max_y - min_y) / desired_resolution))

    # Construction de la matrice "destination" D : passage de pixels de l'image orthorectifiée aux coordonnées sol
    # D : (u,v) -> (min_x + u*res, max_y - v*res)
    D = np.array([
        [desired_resolution, 0, min_x],
        [0, -desired_resolution, max_y],
        [0, 0, 1]
    ])

    # La transformation totale T_total qui mappe l'image orthorectifiée (destination) vers l'image source est :
    # T_total = H_inv * D, donc pour passer de destination vers source
    H_inv = np.linalg.inv(H)
    T_total = H_inv @ D

    # Warp l'image avec cv2.warpPerspective en utilisant T_total
    ortho_img = cv2.warpPerspective(image, T_total, (out_width, out_height))

    # Définition de la géotransformation pour le GeoTIFF :
    # Pixel (0,0) correspond à (min_x, max_y) et la résolution est desired_resolution
    geotransform = (min_x, desired_resolution, 0, max_y, 0, -desired_resolution)

    return ortho_img, geotransform, utm_crs

#############################
# 6. Sauvegarde en GeoTIFF
#############################
def save_geotiff(filename, image_array, geotransform, crs):
    """
    Sauvegarde image_array en GeoTIFF avec la géotransformation et le CRS donnés.
    """
    height, width = image_array.shape[:2]
    count = 3 if image_array.ndim == 3 and image_array.shape[2] == 3 else 1
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=image_array.dtype,
        crs=crs,
        transform=geotransform
    ) as dst:
        if count == 3:
            for i in range(3):
                dst.write(image_array[:, :, i], i + 1)
        else:
            dst.write(image_array, 1)

#############################
# 7. Interface Streamlit
#############################
st.title("Orthorectification d'image (terrain plat) à partir des métadonnées EXIF")

uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images JPEG (avec EXIF)",
    type=["jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for up_file in uploaded_files:
        file_bytes = up_file.read()
        exif_data = extract_exif_info(io.BytesIO(file_bytes))
        # Affichage des métadonnées extraites
        st.write(f"**{up_file.name}** - Métadonnées extraites :")
        st.write(exif_data)
        
        result = orthorectify_image(file_bytes, exif_data)
        if result[0] is None:
            continue  # passe à l'image suivante si problème
        ortho_img, geotransform, utm_crs = result
        
        st.image(ortho_img, caption=f"Image orthorectifiée : {up_file.name}", use_column_width=True)
        
        # Sauvegarde temporaire en GeoTIFF
        output_path = "ortho_output.tif"
        save_geotiff(output_path, ortho_img, geotransform, utm_crs)
        
        st.write("**Informations GeoTIFF :**")
        st.write(f"CRS : {utm_crs}")
        st.write(f"Géotransform : {geotransform}")
        
        with open(output_path, "rb") as f:
            st.download_button("Télécharger le GeoTIFF orthorectifié", f, file_name=f"ortho_{up_file.name}.tif")
        
        os.remove(output_path)
