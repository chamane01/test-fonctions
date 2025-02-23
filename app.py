import streamlit as st
import exifread
import io, os, math, datetime
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image, ImageOps, ExifTags
import rasterio
from affine import Affine
from pyproj import Transformer

# --- Fonctions d'extraction des métadonnées ---
def extract_exif_info(image_file):
    """
    Extrait depuis une image (via exifread) :
      - Les coordonnées GPS (latitude, longitude)
      - L'altitude (en m)
      - La longueur focale (en mm)
      - La résolution du plan focal (pour estimer la largeur du capteur)
      - La date/heure de prise de vue
    Renvoie un tuple : (lat, lon, altitude, focal_length, fp_x_res, fp_unit, dt)
    """
    image_file.seek(0)
    tags = exifread.process_file(image_file, details=False)
    
    # Date/heure
    dt = None
    if 'EXIF DateTimeOriginal' in tags:
        dt = str(tags['EXIF DateTimeOriginal'])
    elif 'Image DateTime' in tags:
        dt = str(tags['Image DateTime'])
    
    # GPS Latitude/Longitude
    lat = None
    lon = None
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_vals = tags['GPS GPSLatitude'].values
        lon_vals = tags['GPS GPSLongitude'].values
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_ref = tags.get('GPS GPSLongitudeRef')
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
                
    # Altitude
    altitude = None
    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den
        
    # Longueur focale
    focal_length = None
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den
        
    # FocalPlaneXResolution et unité
    fp_x_res = None
    fp_unit = None
    if 'EXIF FocalPlaneXResolution' in tags and 'EXIF FocalPlaneResolutionUnit' in tags:
        fp_res_tag = tags['EXIF FocalPlaneXResolution']
        fp_unit_tag = tags['EXIF FocalPlaneResolutionUnit']
        fp_x_res = float(fp_res_tag.values[0].num) / fp_res_tag.values[0].den
        fp_unit = int(fp_unit_tag.values[0])
        
    return lat, lon, altitude, focal_length, fp_x_res, fp_unit, dt

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

# --- Fonction de conversion en GeoTIFF avec rotation ---
def convert_to_tiff(image_file, output_path, utm_center, pixel_size, utm_crs, rotation_angle):
    """
    Convertit une image en GeoTIFF géoréférencé.
    L'orientation du GeoTIFF est fixée de sorte que la largeur de l'image soit perpendiculaire
    à la trajectoire (rotation_angle en radians).
    
    L'UTM_center correspond au centre géoréférencé souhaité (x, y).
    """
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Calcul de la matrice affine avec rotation
    a = pixel_size * math.cos(rotation_angle)
    b = -pixel_size * math.sin(rotation_angle)
    d = pixel_size * math.sin(rotation_angle)
    e = pixel_size * math.cos(rotation_angle)
    # On positionne le centre de l'image sur utm_center
    c = utm_center[0] - (width / 2) * a - (height / 2) * b
    f = utm_center[1] - (width / 2) * d - (height / 2) * e
    transform = Affine(a, b, c, d, e, f)
    
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

# --- Application principale Streamlit ---
st.title("Trajectoire et Conversion GeoTIFF avec Orientation Déduite")

st.markdown("""
Cette application vous permet de :
- Téléverser plusieurs images (JPG/JPEG/PNG).
- Extraire les positions (et la date/heure si disponibles) pour tracer la trajectoire de prise de vue.
- Sélectionner l'image à traiter.  
- Calculer l’orientation de la trajectoire afin que la largeur du TIFF exporté soit perpendiculaire à l’axe de la trajectoire.
- Convertir l'image sélectionnée en GeoTIFF en appliquant la rotation appropriée.
""")

uploaded_files = st.file_uploader("Téléverser plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    image_infos = []
    for file in uploaded_files:
        file_bytes = file.read()
        file_buffer = io.BytesIO(file_bytes)
        # Extraction des métadonnées avec exifread
        lat, lon, altitude, focal_length, fp_x_res, fp_unit, dt = extract_exif_info(file_buffer)
        # Essayer d'ouvrir l'image pour connaître ses dimensions
        try:
            img = Image.open(io.BytesIO(file_bytes))
            width, height = img.size
        except Exception as e:
            st.error(f"Erreur d'ouverture de {file.name}")
            continue
        
        if lat is None or lon is None:
            st.warning(f"{file.name} : Coordonnées GPS non trouvées, image ignorée.")
            continue
        
        # Conversion de la date/heure en objet datetime (si possible)
        dt_obj = None
        if dt:
            try:
                dt_obj = datetime.datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
            except Exception:
                dt_obj = None
        
        image_infos.append({
            "filename": file.name,
            "data": file_bytes,
            "lat": lat,
            "lon": lon,
            "dt": dt,
            "dt_obj": dt_obj,
            "altitude": altitude,
            "focal_length": focal_length,
            "fp_x_res": fp_x_res,
            "fp_unit": fp_unit,
            "width": width,
            "height": height
        })
    
    if len(image_infos) == 0:
        st.error("Aucune image exploitable n'a été trouvée.")
    else:
        # Tri des images par date/heure (si disponible) ou par nom
        image_infos_sorted = sorted(image_infos, key=lambda x: x["dt_obj"] if x["dt_obj"] is not None else x["filename"])
        
        # Calcul et ajout des coordonnées UTM pour chaque image
        for info in image_infos_sorted:
            utm_x, utm_y, utm_crs = latlon_to_utm(info["lat"], info["lon"])
            info["utm_x"] = utm_x
            info["utm_y"] = utm_y
            info["utm_crs"] = utm_crs
        
        # Construction d'une DataFrame pour tracer la trajectoire en coordonnées UTM
        df = pd.DataFrame({
            "utm_x": [info["utm_x"] for info in image_infos_sorted],
            "utm_y": [info["utm_y"] for info in image_infos_sorted],
            "label": [f"{i+1}: {info['filename']}" for i, info in enumerate(image_infos_sorted)]
        })
        
        st.subheader("Trajectoire des prises de vue (en coordonnées UTM)")
        chart = alt.Chart(df).mark_line(point=True).encode(
            x='utm_x:Q',
            y='utm_y:Q',
            tooltip=["label"]
        ).properties(width=600, height=400)
        st.altair_chart(chart, use_container_width=True)
        
        # Proposition de choisir l'image à traiter (ordre chronologique)
        options = [f"{i+1}: {info['filename']} ({info['dt'] if info['dt'] else 'Date inconnue'})" 
                   for i, info in enumerate(image_infos_sorted)]
        selected_option = st.selectbox("Sélectionnez l'image à traiter", options, index=0)
        selected_idx = int(selected_option.split(":")[0]) - 1
        selected_info = image_infos_sorted[selected_idx]
        
        # Calcul de la tangente de la trajectoire au point sélectionné
        if len(image_infos_sorted) == 1:
            tangent_angle = 0.0
        elif selected_idx == 0:
            dx = image_infos_sorted[1]["utm_x"] - image_infos_sorted[0]["utm_x"]
            dy = image_infos_sorted[1]["utm_y"] - image_infos_sorted[0]["utm_y"]
            tangent_angle = math.atan2(dy, dx)
        elif selected_idx == len(image_infos_sorted) - 1:
            dx = image_infos_sorted[-1]["utm_x"] - image_infos_sorted[-2]["utm_x"]
            dy = image_infos_sorted[-1]["utm_y"] - image_infos_sorted[-2]["utm_y"]
            tangent_angle = math.atan2(dy, dx)
        else:
            dx1 = image_infos_sorted[selected_idx]["utm_x"] - image_infos_sorted[selected_idx - 1]["utm_x"]
            dy1 = image_infos_sorted[selected_idx]["utm_y"] - image_infos_sorted[selected_idx - 1]["utm_y"]
            dx2 = image_infos_sorted[selected_idx + 1]["utm_x"] - image_infos_sorted[selected_idx]["utm_x"]
            dy2 = image_infos_sorted[selected_idx + 1]["utm_y"] - image_infos_sorted[selected_idx]["utm_y"]
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            tangent_angle = (angle1 + angle2) / 2.0
        
        # Pour que la largeur du TIFF soit perpendiculaire à la trajectoire,
        # on définit l'angle de rotation comme tangent_angle + 90° (en radians)
        rotation_angle = tangent_angle + math.pi / 2
        st.write(f"Angle de la tangente (radians) : {tangent_angle:.2f}")
        st.write(f"Rotation appliquée pour le TIFF (radians) : {rotation_angle:.2f}")
        
        st.subheader("Paramètres pour le calcul de l'empreinte au sol")
        # Paramètres : si disponibles, préremplis depuis les métadonnées
        hauteur = st.number_input("Hauteur de vol (m)", value=(selected_info["altitude"] if selected_info["altitude"] is not None else 100.0))
        focale = st.number_input("Longueur focale (mm)", value=(selected_info["focal_length"] if selected_info["focal_length"] is not None else 50.0))
        largeur_capteur = st.number_input("Largeur du capteur (mm)", value=36.0)
        
        if st.button("Calculer et convertir l'image sélectionnée"):
            # Calcul de l'empreinte au sol et du GSD
            empreinte_sol = (hauteur * largeur_capteur) / focale
            resolution_pixels = selected_info["width"]
            gsd = empreinte_sol / resolution_pixels  # en m/pixel
            st.write(f"Empreinte au sol : {empreinte_sol:.2f} m")
            st.write(f"GSD : {gsd*100:.2f} cm/pixel")
            
            pixel_size = gsd  # on utilise le GSD comme taille de pixel
            output_path = "output.tif"
            convert_to_tiff(
                image_file=io.BytesIO(selected_info["data"]),
                output_path=output_path,
                utm_center=(selected_info["utm_x"], selected_info["utm_y"]),
                pixel_size=pixel_size,
                utm_crs=selected_info["utm_crs"],
                rotation_angle=rotation_angle
            )
            st.success("Image convertie en GeoTIFF.")
            with rasterio.open(output_path) as src:
                st.write("**Méta-données du GeoTIFF**")
                st.write("CRS :", src.crs)
                st.write("Transform :", src.transform)
            with open(output_path, "rb") as f:
                st.download_button("Télécharger le GeoTIFF", data=f, file_name="image_geotiff.tif", mime="image/tiff")
            os.remove(output_path)
