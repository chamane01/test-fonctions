import streamlit as st
import folium
from folium.plugins import Draw  # Plugin pour dessiner sur la carte
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from PIL import Image
import numpy as np
import base64
import uuid
import os
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, LineString

###############################################
# Paramètres et dictionnaires
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

##############################################
# Chargement et gestion des routes
##############################################
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

##############################################
# Fonctions de reprojection et d'affichage
##############################################
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

def add_image_overlay(map_object, image_data, bounds, layer_name, opacity=1, show=True, control=True):
    # Si image_data est déjà des bytes, on les utilise directement
    if isinstance(image_data, bytes):
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        img_data_url = f"data:image/png;base64,{image_base64}"
    else:
        with open(image_data, "rb") as f:
            content = f.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
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

##############################################
# Fonctions utilitaires pour le jumelage par centre
##############################################
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

##############################################
# INITIALISATION DU SESSION STATE
##############################################
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []  # Liste des paires { "grand":..., "petit":... }
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}  # Marqueurs par indice de paire

##############################################
# Interface en onglets pour détection automatique et manuelle
##############################################
st.title("Détection d'anomalies")

# Inversion des onglets : Détection Automatique en premier, Détection Manuelle en deuxième
tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])

with tab_auto:
    st.header("Détection Automatique")
    auto_uploaded_images = st.file_uploader("Téléversez vos images JPEG ou PNG", type=["jpeg", "jpg", "png"], accept_multiple_files=True, key="auto_images")
    if auto_uploaded_images:
        st.success("Les images ont été bien téléversées.")
        # La logique de détection automatique sera implémentée ultérieurement.
    else:
        st.info("Aucune image téléversée pour la détection automatique.")

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
            # Tri par centre (attention, ici le centre est un tuple (lon, lat))
            grand_list = sorted(grand_list, key=lambda d: d["center"])
            petit_list = sorted(petit_list, key=lambda d: d["center"])
            pair_count = len(grand_list)
            pairs = []
            for i in range(pair_count):
                pairs.append({"grand": grand_list[i], "petit": petit_list[i]})
            st.session_state.pairs = pairs

            ##############################################
            # Navigation entre paires
            ##############################################
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            if col_nav1.button("← Précédent"):
                if st.session_state.current_pair_index > 0:
                    st.session_state.current_pair_index -= 1
            if col_nav3.button("Suivant →"):
                if st.session_state.current_pair_index < pair_count - 1:
                    st.session_state.current_pair_index += 1
            st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {pair_count}")
            current_index = st.session_state.current_pair_index
            current_pair = st.session_state.pairs[current_index]

            ##############################################
            # Traitement de la paire courante : génération des PNG en mémoire
            ##############################################
            # Pour le TIFF GRAND
            reproj_grand_path = current_pair["grand"]["path"]
            with open(reproj_grand_path, "rb") as f:
                tiff_bytes_grand = f.read()
            png_grand = tiff_to_png(tiff_bytes_grand)
            with rasterio.open(reproj_grand_path) as src:
                grand_bounds = src.bounds

            # Pour le TIFF PETIT
            reproj_petit_path = current_pair["petit"]["path"]
            with open(reproj_petit_path, "rb") as f:
                tiff_bytes_petit = f.read()
            png_petit = tiff_to_png(tiff_bytes_petit)
            with rasterio.open(reproj_petit_path) as src:
                petit_bounds = src.bounds

            ##############################################
            # Affichage de la carte de dessin (à partir du TIFF GRAND)
            ##############################################
            st.subheader("Carte de dessin (TIFF GRAND)")
            m_grand = folium.Map(
                location=[(grand_bounds.bottom + grand_bounds.top)/2, (grand_bounds.left + grand_bounds.right)/2],
                zoom_start=18, tiles=None
            )
            # On utilise ici le PNG en mémoire (bytes) pour l'overlay
            add_image_overlay(m_grand, png_grand, grand_bounds, "TIFF GRAND Overlay", opacity=1)
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
            draw.add_to(m_grand)
            folium.LayerControl().add_to(m_grand)
            result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")

            ##############################################
            # Extraction et classification des marqueurs pour la paire courante
            ##############################################
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
                        else:
                            new_lon = new_lat = None
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
                            "coordonnees UTM": "Calculées",  # Vous pouvez calculer et afficher ces coordonnées
                            "lat": new_lat,
                            "long": new_lon,
                            "routes": assigned_route,
                            "detection": "Manuelle",
                            "couleur": class_color.get(selected_class, "#000000"),
                            "radius": gravity_sizes.get(selected_gravity, 5)
                        })
                st.session_state.markers_by_pair[current_index] = new_markers
            else:
                st.write("Aucun marqueur n'a été détecté.")

            ##############################################
            # Carte de suivi (fond basé sur le TIFF PETIT)
            ##############################################
            st.subheader("Carte de suivi")
            global_markers = []
            for markers in st.session_state.markers_by_pair.values():
                global_markers.extend(markers)
            m_petit = folium.Map(
                location=[(petit_bounds.bottom + petit_bounds.top)/2, (petit_bounds.left + petit_bounds.right)/2],
                zoom_start=18
            )
            add_image_overlay(m_petit, png_petit, petit_bounds, "TIFF PETIT Overlay", opacity=0.7)
            for marker in global_markers:
                folium.CircleMarker(
                    location=[marker["lat"], marker["long"]],
                    radius=marker["radius"],
                    color=marker["couleur"],
                    fill=True,
                    fill_color=marker["couleur"],
                    fill_opacity=0.7,
                    tooltip=f'{marker["classe"]} (Gravité {marker["gravite"]})'
                ).add_to(m_petit)
            folium.LayerControl().add_to(m_petit)
            st_folium(m_petit, width=700, height=500, key="folium_map_petit")
            st.markdown("### Récapitulatif global des défauts")
            if global_markers:
                st.table(global_markers)
            else:
                st.write("Aucun marqueur global n'a été enregistré.")
    else:
        st.info("Veuillez téléverser les fichiers TIFF pour lancer la détection manuelle.")

##############################################
# Section commune : Carte de suivi par défaut si aucun TIFF n'est téléversé
##############################################
st.subheader("Carte de suivi")
global_markers = []
for markers in st.session_state.markers_by_pair.values():
    global_markers.extend(markers)

if st.session_state.pairs:
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
        m_default = create_map(center_lat_petit, center_lon_petit, petit_bounds,
                               display_path="", marker_data=global_markers,
                               tiff_opacity=0, tiff_show=True, tiff_control=False,
                               draw_routes=True, add_draw_tool=False)
        st_folium(m_default, width=700, height=500, key="folium_map_default")
    else:
        st.info("Impossible d'afficher la carte de suivi à cause d'un problème avec le TIFF PETIT.")
else:
    st.info("Aucune donnée TIFF téléversée pour afficher la carte de suivi.")

st.markdown("### Récapitulatif global des défauts")
if global_markers:
    st.table(global_markers)
else:
    st.write("Aucun marqueur global n'a été enregistré.")
