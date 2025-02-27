import streamlit as st
import folium
from folium.plugins import Draw  # Plugin pour dessiner sur la carte
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from rasterio.io import MemoryFile
from PIL import Image, ImageOps
import numpy as np
import base64
import uuid
import os
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString
import io
import math
import exifread
from pyproj import Transformer
from affine import Affine

###############################################
# Fonctions et variables existantes
###############################################
class_color = {
    "deformations ornierage": "#FF0000",
    "fissurations": "#00FF00",
    "Faiençage": "#0000FF",
    "fissure de retrait": "#FFFF00",
    "fissure anarchique": "#FF00FF",
    "reparations": "#00FFFF",
    "nid de poule": "#FFA500",
    "arrachements": "#800080",
    "fluage": "#008000",
    "denivellement accotement": "#000080",
    "chaussée detruite": "#FFC0CB",
    "envahissement vegetations": "#A52A2A",
    "assainissements": "#808080",
    "depot de terre": "#8B4513"
}
gravity_sizes = {1: 5, 2: 10, 3: 15}

###############################################
# Chargement et gestion des routes
###############################################
with open("routeQSD.txt", "r") as f:
    routes_data = json.load(f)
routes_ci = []
for feature in routes_data["features"]:
    if feature["geometry"]["type"] == "LineString":
        routes_ci.append({
            "coords": feature["geometry"]["coordinates"],
            "nom": feature["properties"].get("ID", "Route inconnue")
        })

def assign_route_to_marker(lat, lon, routes):
    """
    Pour un point (lat, lon) en EPSG:4326, retourne le nom de la route la plus proche,
    si celle-ci se trouve dans un seuil défini (sinon "Route inconnue").
    """
    marker_point = Point(lon, lat)  # Point(longitude, latitude)
    min_distance = float('inf')
    closest_route = "Route inconnue"
    for route in routes:
        line = LineString(route["coords"])  # Coordonnées au format [lon, lat]
        distance = marker_point.distance(line)
        if distance < min_distance:
            min_distance = distance
            closest_route = route["nom"]
    threshold = 0.01  # Seuil en degrés (à ajuster selon vos données)
    return closest_route if min_distance <= threshold else "Route inconnue"

###############################################
# Fonctions de reprojection et d'affichage
###############################################
def reproject_tiff(input_tiff, target_crs="EPSG:4326"):
    with rasterio.open(input_tiff) as src:
        transform_, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform_,
            "width": width,
            "height": height,
        })
        unique_id = str(uuid.uuid4())[:8]
        output_tiff = f"reprojected_{unique_id}.tif"
        with rasterio.open(output_tiff, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform_,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return output_tiff

def apply_color_gradient(tiff_path, output_png_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        colored_image = cmap(norm(data))
        plt.imsave(output_png_path, colored_image)
        plt.close()

def add_image_overlay(map_object, image_path, bounds, layer_name, opacity=1, show=True, control=True):
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    img_data_url = f"data:image/png;base64,{image_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_data_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=opacity,
        show=show,
        control=control
    ).add_to(map_object)

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

def create_map(center_lat, center_lon, bounds, display_path, marker_data=None,
               hide_osm=False, tiff_opacity=1, tiff_show=True, tiff_control=True,
               draw_routes=True, add_draw_tool=True):
    """
    Création de la carte avec option d'afficher ou non les routes.
    Le paramètre add_draw_tool contrôle l'ajout de l'outil de dessin (marqueurs).
    """
    if hide_osm:
        m = folium.Map(location=[center_lat, center_lon], tiles=None)
    else:
        m = folium.Map(location=[center_lat, center_lon])
    if display_path:
        add_image_overlay(m, display_path, bounds, "TIFF Overlay", opacity=tiff_opacity,
                          show=tiff_show, control=tiff_control)
    if draw_routes:
        for route in routes_ci:
            poly_coords = [(lat, lon) for lon, lat in route["coords"]]
            folium.PolyLine(
                locations=poly_coords,
                color="blue",
                weight=3,
                opacity=0.7,
                tooltip=route["nom"]
            ).add_to(m)
    if add_draw_tool:
        draw = Draw(
            draw_options={
                'marker': True,
                'polyline': False,
                'polygon': False,
                'rectangle': False,
                'circle': False,
                'circlemarker': False,
            },
            edit_options={'edit': True}
        )
        draw.add_to(m)
    m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
    folium.LayerControl().add_to(m)
    if marker_data:
        for marker in marker_data:
            lat = marker["lat"]
            lon = marker["long"]
            color = class_color.get(marker["classe"], "#000000")
            radius = gravity_sizes.get(marker["gravite"], 5)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"{marker['classe']} (Gravité {marker['gravite']}) - Route : {marker.get('routes', 'Route inconnue')} - Détection: {marker.get('detection', 'Inconnue')}"
            ).add_to(m)
    return m

###############################################
# Fonctions utilitaires pour le jumelage par centre
###############################################
def get_reprojected_and_center(uploaded_file, group):
    unique_id = str(uuid.uuid4())[:8]
    temp_path = f"uploaded_{group}_{unique_id}.tif"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    with rasterio.open(temp_path) as src:
        crs_str = src.crs.to_string()
    if crs_str != "EPSG:4326":
        reproj_path = reproject_tiff(temp_path, "EPSG:4326")
    else:
        reproj_path = temp_path
    with rasterio.open(reproj_path) as src:
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
    return {"path": reproj_path, "center": (center_lon, center_lat), "bounds": bounds, "temp_original": temp_path}

###############################################
# Fonctions pour la conversion d'image en GeoTIFF
###############################################
def extract_exif_info(image_file):
    """
    Extrait les informations EXIF (coordonnées GPS, altitude, focale et résolution du capteur)
    à partir d'un objet fichier.
    """
    image_file.seek(0)
    tags = exifread.process_file(image_file, details=False)
    
    lat = lon = altitude = focal_length = None
    fp_x_res = fp_unit = None
    
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

    if 'GPS GPSAltitude' in tags:
        alt_tag = tags['GPS GPSAltitude']
        altitude = float(alt_tag.values[0].num) / alt_tag.values[0].den
        
    if 'EXIF FocalLength' in tags:
        focal_tag = tags['EXIF FocalLength']
        focal_length = float(focal_tag.values[0].num) / focal_tag.values[0].den
        
    if 'EXIF FocalPlaneXResolution' in tags and 'EXIF FocalPlaneResolutionUnit' in tags:
        fp_res_tag = tags['EXIF FocalPlaneXResolution']
        fp_unit_tag = tags['EXIF FocalPlaneResolutionUnit']
        fp_x_res = float(fp_res_tag.values[0].num) / fp_res_tag.values[0].den
        fp_unit = int(fp_unit_tag.values[0])
    
    return lat, lon, altitude, focal_length, fp_x_res, fp_unit

def latlon_to_utm(lat, lon):
    """
    Convertit des coordonnées latitude/longitude en coordonnées UTM et retourne
    le centre UTM (x, y) ainsi que la référence CRS.
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
    Calcule la Ground Sample Distance (GSD) en mètre par pixel.
    """
    focal_length_m = focal_length_mm / 1000.0
    sensor_width_m = sensor_width_mm / 1000.0
    return (altitude * sensor_width_m) / (focal_length_m * image_width_px)

def convert_to_tiff_in_memory(image_file, pixel_size, utm_center, utm_crs, rotation_angle=0, scaling_factor=1):
    """
    Convertit une image en GeoTIFF et retourne les octets correspondants.
    """
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
        count=3 if img_array.ndim == 3 else 1,
        dtype=img_array.dtype,
        crs=utm_crs,
        transform=transform
    ) as dst:
        if img_array.ndim == 3:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            dst.write(img_array, 1)
    return memfile.read()

###############################################
# INITIALISATION DE LA SESSION STATE
###############################################
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []  # Liste des paires { "grand":..., "petit":... }
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}  # Marqueurs par indice de paire

###############################################
# Interface en onglets
###############################################
st.title("Détection d'anomalies et conversion d'image")

# Création de trois onglets : Détection Automatique, Détection Manuelle, Conversion d'image
tab_auto, tab_manuel, tab_conversion = st.tabs(["Détection Automatique", "Détection Manuelle", "Conversion d'image"])

###############################################
# Onglet 1 : Détection Automatique
###############################################
with tab_auto:
    st.header("Détection Automatique")
    auto_uploaded_images = st.file_uploader("Téléversez vos images JPEG ou PNG", type=["jpeg", "jpg", "png"], accept_multiple_files=True, key="auto_images")
    if auto_uploaded_images:
        st.success("Les images ont été bien téléversées.")
        # La logique de détection automatique sera implémentée ultérieurement.
    else:
        st.info("Aucune image téléversée pour la détection automatique.")

###############################################
# Onglet 2 : Détection Manuelle
###############################################
with tab_manuel:
    st.header("Détection Manuelle")
    uploaded_files_grand = st.file_uploader("Téléversez vos fichiers TIFF GRAND", type=["tif", "tiff"], accept_multiple_files=True, key="tiff_grand")
    uploaded_files_petit = st.file_uploader("Téléversez vos fichiers TIFF PETIT", type=["tif", "tiff"], accept_multiple_files=True, key="tiff_petit")
    
    if uploaded_files_grand and uploaded_files_petit:
        if len(uploaded_files_grand) != len(uploaded_files_petit):
            st.error("Le nombre de fichiers TIFF GRAND et TIFF PETIT doit être identique.")
        else:
            grand_list = []
            petit_list = []
            for file in uploaded_files_grand:
                file.seek(0)
                grand_list.append(get_reprojected_and_center(file, "grand"))
            for file in uploaded_files_petit:
                file.seek(0)
                petit_list.append(get_reprojected_and_center(file, "petit"))
            grand_list = sorted(grand_list, key=lambda d: d["center"])
            petit_list = sorted(petit_list, key=lambda d: d["center"])
            pair_count = len(grand_list)
            pairs = []
            for i in range(pair_count):
                pairs.append({"grand": grand_list[i], "petit": petit_list[i]})
            st.session_state.pairs = pairs

            ###############################################
            # Navigation entre paires
            ###############################################
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            prev_pressed = col_nav1.button("← Précédent")
            next_pressed = col_nav3.button("Suivant →")
            if prev_pressed and st.session_state.current_pair_index > 0:
                st.session_state.current_pair_index -= 1
            if next_pressed and st.session_state.current_pair_index < pair_count - 1:
                st.session_state.current_pair_index += 1
            st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {pair_count}")
            current_index = st.session_state.current_pair_index
            current_pair = st.session_state.pairs[current_index]

            ###############################################
            # Traitement de la paire courante : génération des PNG
            ###############################################
            reproj_grand_path = current_pair["grand"]["path"]
            with rasterio.open(reproj_grand_path) as src:
                grand_bounds = src.bounds
                data = src.read()
                if data.shape[0] >= 3:
                    r = normalize_data(data[0])
                    g = normalize_data(data[1])
                    b = normalize_data(data[2])
                    rgb_norm = np.dstack((r, g, b))
                    image_grand = Image.fromarray(rgb_norm)
                else:
                    band = data[0]
                    band_norm = normalize_data(band)
                    image_grand = Image.fromarray(band_norm, mode="L")
            unique_id = str(uuid.uuid4())[:8]
            temp_png_grand = f"converted_grand_{unique_id}.png"
            image_grand.save(temp_png_grand)
            display_path_grand = temp_png_grand

            reproj_petit_path = current_pair["petit"]["path"]
            with rasterio.open(reproj_petit_path) as src:
                petit_bounds = src.bounds
                data = src.read()
                if data.shape[0] >= 3:
                    r = normalize_data(data[0])
                    g = normalize_data(data[1])
                    b = normalize_data(data[2])
                    rgb_norm = np.dstack((r, g, b))
                    image_petit = Image.fromarray(rgb_norm)
                else:
                    band = data[0]
                    band_norm = normalize_data(band)
                    image_petit = Image.fromarray(band_norm, mode="L")
            temp_png_petit = f"converted_{unique_id}.png"
            image_petit.save(temp_png_petit)
            display_path_petit = temp_png_petit

            center_lat_grand = (grand_bounds.bottom + grand_bounds.top) / 2
            center_lon_grand = (grand_bounds.left + grand_bounds.right) / 2
            utm_zone_grand = int((center_lon_grand + 180) / 6) + 1
            center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
            center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
            utm_zone_petit = int((center_lon_petit + 180) / 6) + 1
            utm_crs_petit = f"EPSG:326{utm_zone_petit:02d}"

            ###############################################
            # Carte de dessin : TIFF GRAND (OSM masqué) pour dessin des marqueurs (routes non affichées)
            ###############################################
            st.subheader("Carte de dessin")
            map_placeholder_grand = st.empty()
            m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand,
                                 marker_data=None, hide_osm=True, draw_routes=False, add_draw_tool=True)
            result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")

            ###############################################
            # Extraction et classification des marqueurs pour la paire courante
            ###############################################
            features = []
            all_drawings = result_grand.get("all_drawings")
            if all_drawings:
                if isinstance(all_drawings, dict) and "features" in all_drawings:
                    features = all_drawings.get("features", [])
                elif isinstance(all_drawings, list):
                    features = all_drawings

            global_count = sum(len(markers) for markers in st.session_state.markers_by_pair.values())
            new_markers = []
            if features:
                st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
                for i, feature in enumerate(features):
                    if feature.get("geometry", {}).get("type") == "Point":
                        coords = feature.get("geometry", {}).get("coordinates")
                        if coords and isinstance(coords, list) and len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            percent_x = (lon - grand_bounds.left) / (grand_bounds.right - grand_bounds.left)
                            percent_y = (lat - grand_bounds.bottom) / (grand_bounds.top - grand_bounds.bottom)
                            new_lon = petit_bounds.left + percent_x * (petit_bounds.right - petit_bounds.left)
                            new_lat = petit_bounds.bottom + percent_y * (petit_bounds.top - petit_bounds.bottom)
                            utm_x_petit, utm_y_petit = transform("EPSG:4326", utm_crs_petit, [new_lon], [new_lat])
                            utm_coords_petit = (round(utm_x_petit[0], 2), round(utm_y_petit[0], 2))
                        else:
                            new_lon = new_lat = None
                            utm_coords_petit = "Inconnues"
                        assigned_route = assign_route_to_marker(new_lat, new_lon, routes_ci) if new_lat and new_lon else "Route inconnue"
                        st.markdown(f"**ID {global_count + i + 1}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_class = st.selectbox("Classe", list(class_color.keys()), key=f"class_{current_index}_{i}")
                        with col2:
                            selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{i}")
                        new_markers.append({
                            "ID": global_count + i + 1,
                            "classe": selected_class,
                            "gravite": selected_gravity,
                            "coordonnees UTM": utm_coords_petit,
                            "lat": new_lat,
                            "long": new_lon,
                            "routes": assigned_route,
                            "detection": "Manuelle",  # Marqueur placé manuellement
                            "couleur": class_color.get(selected_class, "#000000"),
                            "radius": gravity_sizes.get(selected_gravity, 5)
                        })
                st.session_state.markers_by_pair[current_index] = new_markers
            else:
                st.write("Aucun marqueur n'a été détecté.")

            ###############################################
            # Nettoyage des fichiers temporaires (pour cette paire)
            ###############################################
            # for file_path in [current_pair["grand"]["temp_original"], current_pair["petit"]["temp_original"],
            #                   reproj_grand_path, reproj_petit_path, temp_png_grand, temp_png_petit]:
            #     if os.path.exists(file_path):
            #         os.remove(file_path)
    else:
        st.info("Veuillez téléverser les fichiers TIFF pour lancer la détection manuelle.")

###############################################
# Onglet 3 : Conversion d'image en GeoTIFF
###############################################
with tab_conversion:
    st.header("Conversion d'image en GeoTIFF")
    uploaded_image = st.file_uploader("Téléversez une image JPEG ou PNG", type=["jpeg", "jpg", "png"], key="conversion_image")
    if uploaded_image is not None:
        # Extraction des informations EXIF
        lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(uploaded_image)
        if lat is None or lon is None:
            st.error("Aucune donnée GPS trouvée dans l'image. Veuillez fournir une image avec des informations GPS.")
        else:
            st.write("**Informations EXIF extraites :**")
            st.write(f"Latitude : {lat}, Longitude : {lon}")
            st.write(f"Altitude : {altitude} m, Focale : {focal_length} mm")
            
            # Conversion en coordonnées UTM
            utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
            st.write(f"Coordonnées UTM : X = {utm_x}, Y = {utm_y} ({utm_crs})")
            utm_center = (utm_x, utm_y)
            
            # Paramètres de conversion (modifiable par l'utilisateur)
            pixel_size = st.number_input("Taille du pixel (m/pixel)", value=0.03, min_value=0.001, step=0.001, format="%.3f")
            rotation_angle = st.number_input("Angle de rotation (en degrés)", value=0.0, step=1.0)
            scaling_factor = st.number_input("Facteur de redimensionnement", value=1.0, min_value=0.1, step=0.1)
            
            # Remise à zéro du flux du fichier avant conversion
            uploaded_image.seek(0)
            tiff_bytes = convert_to_tiff_in_memory(
                image_file=uploaded_image,
                pixel_size=pixel_size,
                utm_center=utm_center,
                utm_crs=utm_crs,
                rotation_angle=rotation_angle,
                scaling_factor=scaling_factor
            )
            
            st.download_button(
                label="Télécharger le GeoTIFF",
                data=tiff_bytes,
                file_name="converted_image.tif",
                mime="image/tiff"
            )
    else:
        st.info("Veuillez téléverser une image pour la conversion.")

###############################################
# Section commune : Carte de suivi et récapitulatif global
###############################################
st.subheader("Carte de suivi")
global_markers = []
for markers in st.session_state.markers_by_pair.values():
    global_markers.extend(markers)

if st.session_state.pairs:
    # Utilisation de la première paire comme référence pour la carte PETIT
    first_pair = st.session_state.pairs[0]
    try:
        with rasterio.open(first_pair["petit"]["path"]) as src:
            petit_bounds = src.bounds
    except Exception as e:
        st.error("Erreur lors de l'ouverture du TIFF PETIT pour la carte de suivi.")
        st.error(e)
        petit_bounds = None

    if petit_bounds:
        center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
        center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
        # Pour la carte de suivi, on n'ajoute pas l'outil de dessin (add_draw_tool=False)
        m_petit = create_map(center_lat_petit, center_lon_petit, petit_bounds,
                             display_path_petit if 'display_path_petit' in locals() else "",
                             marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False, draw_routes=True,
                             add_draw_tool=False)
        st_folium(m_petit, width=700, height=500, key="folium_map_petit")
    else:
        st.info("Impossible d'afficher la carte de suivi à cause d'un problème avec le TIFF PETIT.")
else:
    # Aucune donnée TIFF téléversée : affichage d'une carte par défaut avec les routes
    all_lons = []
    all_lats = []
    for route in routes_ci:
        for lon, lat in route["coords"]:
            all_lons.append(lon)
            all_lats.append(lat)
    if all_lons and all_lats:
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        # Création d'un objet "bounds" simple
        class Bounds:
            pass
        route_bounds = Bounds()
        route_bounds.left = min_lon
        route_bounds.right = max_lon
        route_bounds.bottom = min_lat
        route_bounds.top = max_lat
        center_lat_default = (min_lat + max_lat) / 2
        center_lon_default = (min_lon + max_lon) / 2
        m_default = create_map(center_lat_default, center_lon_default, route_bounds, display_path="",
                               marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False, draw_routes=True,
                               add_draw_tool=False)
        st_folium(m_default, width=700, height=500, key="folium_map_default")
    else:
        st.info("Aucune donnée de route disponible pour afficher la carte de suivi.")

st.markdown("### Récapitulatif global des défauts")
if global_markers:
    st.table(global_markers)
else:
    st.write("Aucun marqueur global n'a été enregistré.")
