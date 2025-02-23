import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from affine import Affine
from pyproj import Transformer
import io

st.title("Empreinte au Sol et Géoréférencement d'une Photo Aérienne")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléverser une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # Extraction des métadonnées EXIF
    exif_data = {}
    if hasattr(image, '_getexif'):
        exif_raw = image._getexif()
        if exif_raw is not None:
            for tag, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value

    # Extraction de la longueur focale
    focal_length_exif = None
    if 'FocalLength' in exif_data:
        focal = exif_data['FocalLength']
        if isinstance(focal, tuple) and len(focal) == 2:
            focal_length_exif = focal[0] / focal[1]
        else:
            focal_length_exif = float(focal)

    # Extraction de l'altitude GPS
    gps_altitude = None
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        for key in gps_info:
            tag = ExifTags.GPSTAGS.get(key, key)
            if tag == 'GPSAltitude':
                alt_val = gps_info[key]
                if isinstance(alt_val, tuple) and len(alt_val) == 2:
                    gps_altitude = alt_val[0] / alt_val[1]
                else:
                    gps_altitude = float(alt_val)

    # Fonction modifiée pour convertir une valeur GPS en degrés décimaux
    def convert_to_degrees(value):
        # Vérifie si la valeur est un tuple de 3 éléments avec des sous-tuples
        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], tuple):
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
            return d + (m / 60.0) + (s / 3600.0)
        else:
            # Sinon, tente de convertir directement en float
            try:
                return float(value)
            except Exception as e:
                st.error("Erreur lors de la conversion des coordonnées GPS: " + str(e))
                return None

    # Extraction des coordonnées GPS (latitude et longitude)
    gps_lat = None
    gps_lon = None
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        if 2 in gps_info and 4 in gps_info:
            gps_lat = convert_to_degrees(gps_info[2])
            gps_lon = convert_to_degrees(gps_info[4])
            # Vérification des références (N/S, E/W)
            if 1 in gps_info and gps_info[1] == 'S':
                gps_lat = -gps_lat
            if 3 in gps_info and gps_info[3] == 'W':
                gps_lon = -gps_lon

    st.subheader("Métadonnées extraites")
    st.write("Longueur focale (EXIF) :", focal_length_exif if focal_length_exif is not None else "Non disponible")
    st.write("Altitude GPS (EXIF) :", gps_altitude if gps_altitude is not None else "Non disponible")
    st.write("Coordonnées GPS (EXIF) :", (gps_lat, gps_lon) if gps_lat is not None and gps_lon is not None else "Non disponibles")
    st.write("Dimensions de l'image (pixels) :", image.size)

    st.subheader("Paramètres nécessaires")
    # Saisie des paramètres avec préremplissage si possible
    hauteur = st.number_input("Hauteur de vol (m)", value=(gps_altitude if gps_altitude is not None else 100.0))
    focale = st.number_input("Longueur focale (mm)", value=(focal_length_exif if focal_length_exif is not None else 50.0))
    largeur_capteur = st.number_input("Largeur du capteur (mm)", value=36.0)
    
    # Estimation de la hauteur du capteur en fonction de l'image (proportionnel à l'image)
    image_width, image_height = image.size
    sensor_height = largeur_capteur * (image_height / image_width)
    
    # Calcul de l'empreinte au sol
    empreinte_horiz = (hauteur * largeur_capteur) / focale   # en mètres (largeur au sol)
    empreinte_vert  = (hauteur * sensor_height) / focale      # en mètres (hauteur au sol)
    
    # Calcul de la résolution au sol (GSD) en m/pixel basé sur la largeur
    gsd = empreinte_horiz / image_width
    
    st.markdown("### Résultats")
    st.write(f"**Empreinte au sol (largeur) :** {empreinte_horiz:.2f} m")
    st.write(f"**Empreinte au sol (hauteur) :** {empreinte_vert:.2f} m")
    st.write(f"**Résolution au sol (GSD) :** {gsd*100:.2f} cm/pixel")
    
    st.markdown("---")
    st.subheader("Génération du GeoTIFF géoréférencé")
    st.write("Le GeoTIFF sera créé avec une taille de pixel de **{:.2f} m** (correspondant à la résolution au sol).".format(gsd))
    
    if st.button("Générer et Télécharger le GeoTIFF"):
        if gps_lat is None or gps_lon is None:
            st.error("Les coordonnées GPS ne sont pas disponibles dans les métadonnées, impossible de géoréférencer l'image.")
        else:
            # Fonction pour obtenir le CRS UTM à partir des coordonnées (WGS84)
            def get_utm_crs(lon, lat):
                zone = int((lon + 180) / 6) + 1
                if lat >= 0:
                    epsg = 32600 + zone
                else:
                    epsg = 32700 + zone
                return rasterio.crs.CRS.from_epsg(epsg)
            
            utm_crs = get_utm_crs(gps_lon, gps_lat)
            
            # Transformer le centre GPS en coordonnées UTM (unités en mètres)
            transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            center_x, center_y = transformer.transform(gps_lon, gps_lat)
            
            # On considère que le centre de l'image correspond au centre de l'empreinte au sol
            top_left_x = center_x - empreinte_horiz / 2
            top_left_y = center_y + empreinte_vert / 2  # y décroît vers le bas
            
            # Création de la transformation affine
            transform_affine = Affine.translation(top_left_x, top_left_y) * Affine.scale(gsd, -gsd)
            
            # Conversion de l'image en tableau numpy
            img_array = np.array(image)
            if img_array.ndim == 2:
                count = 1
            else:
                count = img_array.shape[2]
                # Réorganisation pour rasterio (band, row, col)
                img_array = np.transpose(img_array, (2, 0, 1))
            
            # Création du GeoTIFF en mémoire avec la résolution au sol trouvée
            memfile = MemoryFile()
            with memfile.open(driver='GTiff',
                              height=img_array.shape[1],
                              width=img_array.shape[2],
                              count=count,
                              dtype=img_array.dtype,
                              crs=utm_crs,
                              transform=transform_affine) as dataset:
                dataset.write(img_array)
            
            geotiff_bytes = memfile.read()
            
            st.download_button(
                label="Télécharger le GeoTIFF",
                data=geotiff_bytes,
                file_name="image_georeferencee.tif",
                mime="image/tiff"
            )
