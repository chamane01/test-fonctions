import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import rasterio
import numpy as np
from PIL import Image
import io, base64
from pyproj import Transformer  # Pour la conversion en UTM

st.set_page_config(layout="wide")
st.title("Annotation d'images TIFF sur carte")

# Fonction de conversion lat/lon -> UTM
def latlon_to_utm(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
        hemisphere = 'N'
    else:
        epsg = 32700 + zone
        hemisphere = 'S'
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_easting, utm_northing = transformer.transform(lon, lat)
    return utm_easting, utm_northing, zone, hemisphere

# 1. Téléversement des fichiers TIFF
uploaded_files = st.file_uploader("Téléversez vos fichiers TIFF", type=["tiff", "tif"], accept_multiple_files=True)

if uploaded_files:
    # Initialisation des variables dans session_state
    if "current_image" not in st.session_state:
        st.session_state["current_image"] = 0
    if "markers" not in st.session_state:
        st.session_state["markers"] = []  # Stockera la liste des marqueurs et leurs informations

    # Navigation entre les images : boutons et slider
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Image précédente"):
            st.session_state["current_image"] = max(0, st.session_state["current_image"] - 1)
    with col3:
        if st.button("Image suivante"):
            st.session_state["current_image"] = min(len(uploaded_files) - 1, st.session_state["current_image"] + 1)

    num_files = len(uploaded_files)
    # Utilisation du slider avec la clé "current_image" qui mettra automatiquement à jour st.session_state["current_image"]
    st.slider("Sélectionnez l'image", min_value=0, max_value=num_files - 1,
              value=st.session_state["current_image"], key="current_image")
    st.write(f"Affichage de l'image {st.session_state['current_image'] + 1} sur {num_files}")

    # 2. Affichage de l'image sur une carte
    current_file = uploaded_files[st.session_state["current_image"]]
    current_bytes = current_file.read()
    with rasterio.MemoryFile(current_bytes) as memfile:
        with memfile.open() as dataset:
            bounds = dataset.bounds  # (left, bottom, right, top)
            center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
            arr = dataset.read()
            if arr.shape[0] >= 3:
                arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)
            else:
                arr = arr[0]
            pil_img = Image.fromarray(arr.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_b64}"

    m = folium.Map(location=center, zoom_start=18)
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'rectangle': False,
            'circlemarker': False,
            'marker': True,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)
    st.write("Utilisez l'outil de dessin (icône en haut à gauche de la carte) pour ajouter un marqueur.")

    output = st_folium(m, width=700, height=500, returned_objects=["all_drawings"])

    # 3. Attribution d'une classe et d'une gravité après placement du marqueur
    if output and output.get("all_drawings"):
        for drawing in output["all_drawings"]:
            geometry = drawing.get("geometry", {})
            if geometry.get("type") == "Point":
                coords = geometry.get("coordinates", [])
                if coords:
                    lon, lat = coords
                    if not any(np.isclose(marker["lat"], lat) and np.isclose(marker["lon"], lon)
                               for marker in st.session_state["markers"]):
                        st.write("Nouveau marqueur détecté :")
                        st.write(f"• Coordonnées : Latitude {lat:.6f}, Longitude {lon:.6f}")
                        st.write("Attribuez-lui une classe et une gravité :")
                        with st.form(key=f"marker_form_{lat}_{lon}"):
                            selected_class = st.selectbox("Sélectionnez la classe", [f"Classe {i+1}" for i in range(13)])
                            selected_severity = st.radio("Sélectionnez la gravité", [1, 2, 3])
                            submitted = st.form_submit_button("Enregistrer le marqueur")
                            if submitted:
                                st.session_state["markers"].append({
                                    "lat": lat,
                                    "lon": lon,
                                    "class": selected_class,
                                    "severity": selected_severity,
                                    "image_index": st.session_state["current_image"],
                                })
                                st.success("Marqueur enregistré!")

    st.subheader("Liste des marqueurs enregistrés")
    markers_current = [marker for marker in st.session_state["markers"]
                       if marker["image_index"] == st.session_state["current_image"]]
    if markers_current:
        for idx, marker in enumerate(markers_current):
            st.write(f"Marqueur {idx+1} : {marker}")
    else:
        st.write("Aucun marqueur enregistré pour cette image pour le moment.")

    # 4. Génération et export du rapport
    if st.session_state["markers"]:
        st.subheader("Génération du rapport")
        report_text = "Rapport des marqueurs\n\n"
        class_counts = {}
        for marker in st.session_state["markers"]:
            class_counts[marker["class"]] = class_counts.get(marker["class"], 0) + 1
        report_text += "Nombre de marqueurs par classe:\n"
        for cl, count in class_counts.items():
            report_text += f"{cl} : {count}\n"
        report_text += "\nPositions en UTM des marqueurs:\n"
        for marker in st.session_state["markers"]:
            utm_e, utm_n, zone, hemi = latlon_to_utm(marker["lat"], marker["lon"])
            report_text += (f"{marker['class']} (Image {marker['image_index']+1}) : "
                            f"Easting {utm_e:.2f}, Northing {utm_n:.2f} (Zone {zone}{hemi})\n")
        st.download_button(label="Télécharger le rapport",
                           data=report_text,
                           file_name="rapport_marqueurs.txt",
                           mime="text/plain")
