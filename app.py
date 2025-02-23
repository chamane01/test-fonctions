import streamlit as st
import rasterio
from rasterio.transform import from_origin
from affine import Affine
from PIL import Image
import exifread
import numpy as np
import cv2
import os
from pyproj import CRS, Transformer
import io
import math

#####################################
# 1. Extraction des métadonnées EXIF
#####################################
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

    altitude = None
    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den

    focal_length = None
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den

    fp_x_res = None
    fp_unit = None
    if 'EXIF FocalPlaneXResolution' in tags and 'EXIF FocalPlaneResolutionUnit' in tags:
        fp_res_tag = tags['EXIF FocalPlaneXResolution']
        fp_unit_tag = tags['EXIF FocalPlaneResolutionUnit']
        fp_x_res = float(fp_res_tag.values[0].num) / fp_res_tag.values[0].den
        fp_unit = int(fp_unit_tag.values[0])
    
    gps_img_direction = None
    if 'GPS GPSImgDirection' in tags:
        dir_tag = tags['GPS GPSImgDirection']
        gps_img_direction = float(dir_tag.values[0].num) / dir_tag.values[0].den

    return {
        'lat': lat,
        'lon': lon,
        'altitude': altitude,           # en m (AGL)
        'focal_length': focal_length,     # en mm
        'fp_x_res': fp_x_res,
        'fp_unit': fp_unit,
        'gps_img_direction': gps_img_direction
    }

#####################################
# 2. Conversion GPS -> UTM (renvoie un objet CRS)
#####################################
def latlon_to_utm(lat, lon):
    """
    Convertit lat/lon (WGS84) en coordonnées UTM.
    Renvoie (utm_x, utm_y, crs_utm) avec crs_utm un objet pyproj.CRS.
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        crs_utm = CRS.from_epsg(32600 + zone)
    else:
        crs_utm = CRS.from_epsg(32700 + zone)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs_utm, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y, crs_utm

#####################################
# 3. Modèle de la caméra
#####################################
def compute_camera_matrix(focal_length, sensor_width, image_width, image_height):
    """
    Calcule la matrice intrinsèque K.
    focal_length et sensor_width en mm, image_width en pixels.
    On suppose que le centre optique est au centre de l'image.
    """
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
    On suppose pitch=roll=0, et on applique une correction pour que l'axe optique pointe vers le sol.
    """
    yaw = np.deg2rad(yaw_deg)
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0, 0, 1]])
    R_fixed = np.diag([1, 1, -1])
    return R_yaw @ R_fixed

#####################################
# 4. Projection de pixels sur le sol (terrain plat, Z=0)
#####################################
def image_to_ground(u, v, K, R, T):
    """
    Pour un pixel (u,v) de l'image, calcule son intersection avec le plan sol (Z=0).
    - K : matrice intrinsèque
    - R : matrice de rotation (de la caméra vers le monde)
    - T : position de la caméra en coordonnées monde (X, Y, Z)
    Renvoie le point (X, Y, 0)
    """
    invK = np.linalg.inv(K)
    pixel_homog = np.array([u, v, 1])
    d_cam = invK @ pixel_homog
    d_world = R @ d_cam
    if d_world[2] == 0:
        lambda_val = 0
    else:
        lambda_val = -T[2] / d_world[2]
    return T + lambda_val * d_world

#####################################
# 5. Orthorectification de l'image
#####################################
def orthorectify_image(image_bytes, exif_data, fallback_sensor_width=6.17):
    """
    Réalise l'orthorectification d'une image à partir de ses métadonnées.
    On considère un terrain plat (Z=0) et on suppose que l'altitude est donnée en AGL.
    Retourne l'image orthorectifiée, sa géotransformation et le CRS.
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    image = np.array(pil_img)
    img_height, img_width = image.shape[:2]

    lat = exif_data.get('lat')
    lon = exif_data.get('lon')
    altitude = exif_data.get('altitude')
    focal_length = exif_data.get('focal_length')
    sensor_width = None
    if exif_data.get('fp_x_res') and exif_data.get('fp_unit'):
        if exif_data['fp_unit'] == 2:
            sensor_width = (img_width / exif_data['fp_x_res']) * 25.4
        elif exif_data['fp_unit'] == 3:
            sensor_width = (img_width / exif_data['fp_x_res']) * 10
        elif exif_data['fp_unit'] == 4:
            sensor_width = (img_width / exif_data['fp_x_res'])
    if sensor_width is None:
        sensor_width = fallback_sensor_width

    if None in [lat, lon, altitude, focal_length]:
        st.error("Les métadonnées EXIF essentielles (GPS, altitude, focale) sont manquantes.")
        return None, None, None

    utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
    T = np.array([utm_x, utm_y, altitude])

    K = compute_camera_matrix(focal_length, sensor_width, img_width, img_height)
    yaw = exif_data.get('gps_img_direction') or 0
    R = rotation_matrix_from_yaw(yaw)

    pts_img = np.array([[0, 0],
                        [img_width, 0],
                        [img_width, img_height],
                        [0, img_height]], dtype=np.float32)
    pts_ground = []
    for pt in pts_img:
        gp = image_to_ground(pt[0], pt[1], K, R, T)
        pts_ground.append([gp[0], gp[1]])
    pts_ground = np.array(pts_ground, dtype=np.float32)

    H, status = cv2.findHomography(pts_img, pts_ground)

    gp_00 = image_to_ground(0, 0, K, R, T)
    gp_10 = image_to_ground(1, 0, K, R, T)
    res_x = np.linalg.norm(np.array(gp_10[:2]) - np.array(gp_00[:2]))
    gp_01 = image_to_ground(0, 1, K, R, T)
    res_y = np.linalg.norm(np.array(gp_01[:2]) - np.array(gp_00[:2]))
    desired_resolution = (res_x + res_y) / 2.0

    min_x, min_y = np.min(pts_ground, axis=0)
    max_x, max_y = np.max(pts_ground, axis=0)
    out_width = int(np.ceil((max_x - min_x) / desired_resolution))
    out_height = int(np.ceil((max_y - min_y) / desired_resolution))

    D = np.array([[desired_resolution, 0, min_x],
                  [0, -desired_resolution, max_y],
                  [0, 0, 1]])
    H_inv = np.linalg.inv(H)
    T_total = H_inv @ D

    ortho_img = cv2.warpPerspective(image, T_total, (out_width, out_height))
    geotransform = (min_x, desired_resolution, 0, max_y, 0, -desired_resolution)
    return ortho_img, geotransform, utm_crs

#####################################
# 6. Sauvegarde en GeoTIFF
#####################################
def save_geotiff(filename, image_array, geotransform, crs_obj):
    """
    Sauvegarde image_array en GeoTIFF avec la géotransformation et le CRS donnés.
    Le CRS est passé sous forme d'objet pyproj.CRS et converti en WKT.
    """
    affine_transform = Affine(*[float(val) for val in geotransform])
    height, width = image_array.shape[:2]
    count = 3 if image_array.ndim == 3 and image_array.shape[2] == 3 else 1
    crs_wkt = crs_obj.to_wkt()
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=image_array.dtype,
        crs=crs_wkt,
        transform=affine_transform
    ) as dst:
        if count == 3:
            for i in range(3):
                dst.write(image_array[:, :, i], i + 1)
        else:
            dst.write(image_array, 1)

#####################################
# 7. Interface Streamlit
#####################################
st.title("Orthorectification et conversion en GeoTIFF")

uploaded_files = st.file_uploader(
    "Téléversez une ou plusieurs images JPEG (avec EXIF)",
    type=["jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for up_file in uploaded_files:
        file_bytes = up_file.read()
        exif_data = extract_exif_info(io.BytesIO(file_bytes))
        st.write(f"**{up_file.name}** - Métadonnées extraites :")
        st.write(exif_data)
        
        result = orthorectify_image(file_bytes, exif_data)
        if result[0] is None:
            continue
        ortho_img, geotransform, utm_crs = result
        
        st.image(ortho_img, caption=f"Image orthorectifiée : {up_file.name}", use_column_width=True)
        
        output_path = "ortho_output.tif"
        save_geotiff(output_path, ortho_img, geotransform, utm_crs)
        
        st.write("**Informations GeoTIFF :**")
        st.write(f"CRS : {utm_crs.to_wkt()}")
        st.write(f"Géotransform : {geotransform}")
        
        with open(output_path, "rb") as f:
            st.download_button(
                label="Télécharger le GeoTIFF orthorectifié",
                data=f,
                file_name=f"ortho_{up_file.name}.tif",
                mime="image/tiff"
            )
        os.remove(output_path)
