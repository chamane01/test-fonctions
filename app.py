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

###############################
# Paramètres et dictionnaires
###############################
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

# Modification : ajout du paramètre "show" et "control" pour gérer l'affichage dans le LayerControl
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

# Modification : ajout des paramètres "tiff_opacity", "tiff_show" et "tiff_control" pour contrôler l'overlay TIFF
def create_map(center_lat, center_lon, bounds, display_path, marker_data=None, hide_osm=False, tiff_opacity=1, tiff_show=True, tiff_control=True):
    if hide_osm:
        m = folium.Map(location=[center_lat, center_lon], tiles=None)
    else:
        m = folium.Map(location=[center_lat, center_lon])
    add_image_overlay(m, display_path, bounds, "TIFF Overlay", opacity=tiff_opacity, show=tiff_show, control=tiff_control)
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
# INITIALISATION DE LA SESSION STATE
###############################################
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = []  # Liste des paires { "grand":..., "petit":... }
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}  # Marqueurs par indice de paire

###############################################
# UPLOAD MULTIPLE DES FICHIERS
###############################################
st.title("Affichage par paire de TIFF avec jumelage par centre, navigation et récapitulatif global")

uploaded_files_grand = st.file_uploader("TIFF GRAND", type=["tif", "tiff"], accept_multiple_files=True)
uploaded_files_petit = st.file_uploader("TIFF PETIT", type=["tif", "tiff"], accept_multiple_files=True)

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
        # Navigation entre paires sans appel explicite de experimental_rerun
        ###############################################
        col_nav1, col_nav2, col_nav3 = st.columns([1,2,1])
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

        # Masquage de l'affichage des bornes TIFF GRAND et TIFF PETIT
        # (Les lignes d'affichage des bornes ont été supprimées)

        center_lat_grand = (grand_bounds.bottom + grand_bounds.top) / 2
        center_lon_grand = (grand_bounds.left + grand_bounds.right) / 2
        utm_zone_grand = int((center_lon_grand + 180) / 6) + 1
        utm_crs_grand = f"EPSG:326{utm_zone_grand:02d}"
        center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
        center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
        utm_zone_petit = int((center_lon_petit + 180) / 6) + 1
        utm_crs_petit = f"EPSG:326{utm_zone_petit:02d}"

        ###############################################
        # Carte 1 : TIFF GRAND (OSM masqué) pour dessin des marqueurs
        ###############################################
        st.subheader("Carte de dessin")
        map_placeholder_grand = st.empty()
        m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand, marker_data=None, hide_osm=True)
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
                    st.markdown(f"**ID {global_count + i + 1}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class = st.selectbox("Classe", [
                            "deformations ornierage",
                            "fissurations",
                            "Faiençage",
                            "fissure de retrait",
                            "fissure anarchique",
                            "reparations",
                            "nid de poule",
                            "arrachements",
                            "fluage",
                            "denivellement accotement",
                            "chaussée detruite",
                            "envahissement vegetations",
                            "assainissements",
                            "depot de terre"
                        ], key=f"class_{current_index}_{i}")
                    with col2:
                        selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{i}")
                    new_markers.append({
                        "ID": global_count + i + 1,
                        "classe": selected_class,
                        "gravite": selected_gravity,
                        "coordonnees UTM": utm_coords_petit,
                        "lat": new_lat,
                        "long": new_lon
                    })
            st.session_state.markers_by_pair[current_index] = new_markers
        else:
            st.write("Aucun marqueur n'a été détecté.")

        ###############################################
        # Carte 2 : TIFF PETIT avec TOUS les marqueurs
        ###############################################
        # On agrège tous les marqueurs enregistrés pour les afficher sur la carte PETIT.
        global_markers = []
        for markers in st.session_state.markers_by_pair.values():
            global_markers.extend(markers)

        st.subheader("Carte de suivie")
        map_placeholder_petit = st.empty()
        # Pour la carte PETIT, l'overlay TIFF est rendu transparent (tiff_opacity=0) et est retiré du gestionnaire de couche (tiff_control=False)
        m_petit = create_map(center_lat_petit, center_lon_petit, petit_bounds, display_path_petit,
                             marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False)
        st_folium(m_petit, width=700, height=500, key="folium_map_petit")

        ###############################################
        # Récapitulatif global unique des marqueurs
        ###############################################
        global_markers_table = []
        for idx in sorted(st.session_state.markers_by_pair.keys()):
            global_markers_table.extend(st.session_state.markers_by_pair[idx])
        if global_markers_table:
            st.markdown("### Récapitulatif global des défauts")
            st.table(global_markers_table)
        else:
            st.write("Aucun marqueur global n'a été enregistré.")

        ###############################################
        # Nettoyage des fichiers temporaires (pour cette paire)
        ###############################################
        for file_path in [current_pair["grand"]["temp_original"], current_pair["petit"]["temp_original"],
                          reproj_grand_path, reproj_petit_path, temp_png_grand, temp_png_petit]:
            if os.path.exists(file_path):
                os.remove(file_path)
