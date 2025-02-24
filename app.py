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

# Dictionnaire associant une couleur à chaque classe
class_color = {
    "Classe 1": "#FF0000",
    "Classe 2": "#00FF00",
    "Classe 3": "#0000FF",
    "Classe 4": "#FFFF00",
    "Classe 5": "#FF00FF",
    "Classe 6": "#00FFFF",
    "Classe 7": "#FFA500",
    "Classe 8": "#800080",
    "Classe 9": "#008000",
    "Classe 10": "#000080",
    "Classe 11": "#FFC0CB",
    "Classe 12": "#A52A2A",
    "Classe 13": "#808080"
}
# Taille (rayon) pour chaque niveau de gravité
gravity_sizes = {1: 5, 2: 10, 3: 15}

# --- Fonction de reprojection (d'après le code de référence) ---
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

# --- Fonctions d'affichage et de traitement des images ---
def apply_color_gradient(tiff_path, output_png_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        cmap = plt.get_cmap("terrain")
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())
        colored_image = cmap(norm(data))
        plt.imsave(output_png_path, colored_image)
        plt.close()

def add_image_overlay(map_object, image_path, bounds, layer_name, opacity=1):
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    img_data_url = f"data:image/png;base64,{image_base64}"
    folium.raster_layers.ImageOverlay(
        image=img_data_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=opacity,
    ).add_to(map_object)

def normalize_data(data):
    lower = np.percentile(data, 2)
    upper = np.percentile(data, 98)
    norm_data = np.clip(data, lower, upper)
    norm_data = (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)
    return norm_data

# --- Création de la carte ---
def create_map(center_lat, center_lon, bounds, display_path, marker_data=None, hide_osm=False):
    # Si hide_osm est True, aucun fond de carte n'est affiché
    if hide_osm:
        m = folium.Map(location=[center_lat, center_lon], tiles=None)
    else:
        m = folium.Map(location=[center_lat, center_lon])
    add_image_overlay(m, display_path, bounds, "TIFF Overlay", opacity=1)
    # Plugin de dessin
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
    # Affichage des marqueurs (pour la paire courante)
    if marker_data:
        for marker in marker_data:
            lat = marker["lat"]
            lon = marker["lon"]
            color = marker.get("color", "#000000")
            radius = marker.get("radius", 5)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(m)
    return m

###########################################
# INITIALISATION DE LA SESSION STATE
###########################################
if "current_pair_index" not in st.session_state:
    st.session_state.current_pair_index = 0
if "markers_by_pair" not in st.session_state:
    st.session_state.markers_by_pair = {}  # stocke les marqueurs par indice de paire

###########################################
# UPLOAD MULTIPLE DES FICHIERS
###########################################
st.title("Affichage par paire de TIFF avec navigation et récapitulatif global")
uploaded_files_grand = st.file_uploader("Téléversez vos fichiers TIFF GRAND", type=["tif", "tiff"], accept_multiple_files=True)
uploaded_files_petit = st.file_uploader("Téléversez vos fichiers TIFF PETIT", type=["tif", "tiff"], accept_multiple_files=True)

if uploaded_files_grand and uploaded_files_petit:
    if len(uploaded_files_grand) != len(uploaded_files_petit):
        st.error("Le nombre de fichiers TIFF GRAND et TIFF PETIT doit être identique.")
    else:
        pair_count = len(uploaded_files_grand)
        # Navigation entre les paires
        col_nav1, col_nav2, col_nav3 = st.columns([1,2,1])
        with col_nav1:
            if st.button("← Précédent"):
                if st.session_state.current_pair_index > 0:
                    st.session_state.current_pair_index -= 1
                    st.experimental_rerun()
        with col_nav3:
            if st.button("Suivant →"):
                if st.session_state.current_pair_index < pair_count - 1:
                    st.session_state.current_pair_index += 1
                    st.experimental_rerun()
        st.write(f"Affichage de la paire {st.session_state.current_pair_index + 1} sur {pair_count}")
        current_index = st.session_state.current_pair_index

        ###########################################
        # TRAITEMENT DE LA PAIRE COURANTE
        ###########################################
        # Sauvegarde temporaire des fichiers (TIFF GRAND et TIFF PETIT)
        unique_file_id = str(uuid.uuid4())[:8]
        temp_tiff_grand_path = f"uploaded_grand_{unique_file_id}.tif"
        temp_tiff_petit_path = f"uploaded_petit_{unique_file_id}.tif"
        with open(temp_tiff_grand_path, "wb") as f:
            f.write(uploaded_files_grand[current_index].read())
        with open(temp_tiff_petit_path, "wb") as f:
            f.write(uploaded_files_petit[current_index].read())

        # Reprojection des TIFF en EPSG:4326 (si nécessaire)
        with rasterio.open(temp_tiff_grand_path) as src:
            st.write("CRS du TIFF GRAND :", src.crs)
        if src.crs.to_string() != "EPSG:4326":
            st.write("Reprojection du TIFF GRAND vers EPSG:4326...")
            reprojected_grand_path = reproject_tiff(temp_tiff_grand_path, "EPSG:4326")
        else:
            reprojected_grand_path = temp_tiff_grand_path
        with rasterio.open(reprojected_grand_path) as src:
            grand_bounds = src.bounds

        with rasterio.open(temp_tiff_petit_path) as src:
            st.write("CRS du TIFF PETIT :", src.crs)
        if src.crs.to_string() != "EPSG:4326":
            st.write("Reprojection du TIFF PETIT vers EPSG:4326...")
            reprojected_petit_path = reproject_tiff(temp_tiff_petit_path, "EPSG:4326")
        else:
            reprojected_petit_path = temp_tiff_petit_path
        with rasterio.open(reprojected_petit_path) as src:
            petit_bounds = src.bounds

        # Option gradient pour le TIFF PETIT
        apply_gradient = st.checkbox("Appliquer un gradient de couleur pour le TIFF PETIT", value=False)

        # Création de l'image PNG pour le TIFF GRAND
        with rasterio.open(reprojected_grand_path) as src:
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
        temp_png_grand = f"converted_grand_{unique_file_id}.png"
        image_grand.save(temp_png_grand)
        display_path_grand = temp_png_grand

        # Création de l'image PNG pour le TIFF PETIT
        if apply_gradient:
            temp_png_petit = f"colored_{unique_file_id}.png"
            apply_color_gradient(reprojected_petit_path, temp_png_petit)
        else:
            with rasterio.open(reprojected_petit_path) as src:
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
            temp_png_petit = f"converted_{unique_file_id}.png"
            image_petit.save(temp_png_petit)
        display_path_petit = temp_png_petit

        st.write("Bornes TIFF GRAND (EPSG:4326) :", grand_bounds)
        st.write("Bornes TIFF PETIT (EPSG:4326) :", petit_bounds)

        # Détermination des centres et des zones UTM pour chaque TIFF
        center_lat_grand = (grand_bounds.bottom + grand_bounds.top) / 2
        center_lon_grand = (grand_bounds.left + grand_bounds.right) / 2
        utm_zone_grand = int((center_lon_grand + 180) / 6) + 1
        utm_crs_grand = f"EPSG:326{utm_zone_grand:02d}"

        center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
        center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
        utm_zone_petit = int((center_lon_petit + 180) / 6) + 1
        utm_crs_petit = f"EPSG:326{utm_zone_petit:02d}"

        ###########################################
        # Carte 1 : TIFF GRAND pour le dessin (OSM masqué)
        ###########################################
        st.subheader("Carte 1 : TIFF GRAND (pour dessin des marqueurs)")
        map_placeholder_grand = st.empty()
        m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand, marker_data=None, hide_osm=True)
        result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")

        ###########################################
        # Extraction et classification des marqueurs pour la paire courante
        ###########################################
        # Récupération des dessins (marqueurs) sur la carte 1
        features = []
        all_drawings = result_grand.get("all_drawings")
        if all_drawings:
            if isinstance(all_drawings, dict) and "features" in all_drawings:
                features = all_drawings.get("features", [])
            elif isinstance(all_drawings, list):
                features = all_drawings

        # Calcul global du nombre de marqueurs déjà enregistrés (toutes paires confondues)
        global_count = sum(len(markers) for markers in st.session_state.markers_by_pair.values())

        # Pour la paire courante, on ajoute (ou met à jour) les marqueurs si de nouveaux dessins sont présents
        current_markers = st.session_state.markers_by_pair.get(current_index, [])
        if features:
            st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
            # On part du décompte global pour numéroter correctement
            new_markers = []
            for i, feature in enumerate(features):
                if feature.get("geometry", {}).get("type") == "Point":
                    coords = feature.get("geometry", {}).get("coordinates")
                    if coords and isinstance(coords, list) and len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        # Calcul des pourcentages dans le TIFF GRAND
                        percent_x = (lon - grand_bounds.left) / (grand_bounds.right - grand_bounds.left)
                        percent_y = (lat - grand_bounds.bottom) / (grand_bounds.top - grand_bounds.bottom)
                        # Projection dans le système du TIFF PETIT (EPSG:4326)
                        new_lon = petit_bounds.left + percent_x * (petit_bounds.right - petit_bounds.left)
                        new_lat = petit_bounds.bottom + percent_y * (petit_bounds.top - petit_bounds.bottom)
                        # Conversion en UTM pour le TIFF PETIT (optionnel ici)
                        utm_x_petit, utm_y_petit = transform("EPSG:4326", utm_crs_petit, [new_lon], [new_lat])
                        utm_coords_petit = (round(utm_x_petit[0], 2), round(utm_y_petit[0], 2))
                    else:
                        new_lon = new_lat = None
                        utm_coords_petit = "Inconnues"
                    st.markdown(f"**Marqueur {global_count + i + 1}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class = st.selectbox("Classe", [f"Classe {j}" for j in range(1, 14)], key=f"class_{current_index}_{i}")
                    with col2:
                        selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{i}")
                    new_markers.append({
                        "Marqueur": global_count + i + 1,
                        "Coordonnées (EPSG:4326)": (round(new_lon, 4), round(new_lat, 4)) if new_lon and new_lat else "Inconnues",
                        "Coordonnées UTM": utm_coords_petit,
                        "Classe": selected_class,
                        "Gravité": selected_gravity,
                        # Pour la carte 2, on utilise ces coordonnées reprojetées
                        "lat": new_lat,
                        "lon": new_lon,
                        "color": class_color.get(selected_class, "#000000"),
                        "radius": gravity_sizes.get(selected_gravity, 5)
                    })
            # On stocke (ou cumule) les marqueurs pour cette paire
            st.session_state.markers_by_pair[current_index] = new_markers
        else:
            st.write("Aucun marqueur n'a été détecté.")

        ###########################################
        # Carte 2 : TIFF PETIT avec les marqueurs de la paire courante
        ###########################################
        current_markers = st.session_state.markers_by_pair.get(current_index, [])
        st.subheader("Carte 2 : TIFF PETIT (avec marqueurs reprojetés)")
        map_placeholder_petit = st.empty()
        m_petit = create_map(center_lat_petit, center_lon_petit, petit_bounds, display_path_petit, marker_data=current_markers)
        st_folium(m_petit, width=700, height=500, key="folium_map_petit")

        ###########################################
        # Récapitulatif global (unique) des marqueurs de toutes les paires
        ###########################################
        global_markers = []
        # On cumule les marqueurs de chaque paire dans l'ordre
        for idx in sorted(st.session_state.markers_by_pair.keys()):
            global_markers.extend(st.session_state.markers_by_pair[idx])
        if global_markers:
            st.markdown("### Récapitulatif global des marqueurs")
            st.table(global_markers)
        else:
            st.write("Aucun marqueur global n'a été enregistré.")

        ###########################################
        # Nettoyage des fichiers temporaires
        ###########################################
        for file_path in [temp_tiff_grand_path, temp_tiff_petit_path, reprojected_grand_path, reprojected_petit_path, temp_png_grand, temp_png_petit]:
            if os.path.exists(file_path):
                os.remove(file_path)
