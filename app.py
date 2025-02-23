import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import rasterio
import numpy as np
from PIL import Image
import io, base64

st.set_page_config(layout="wide")
st.title("Annotation d'images TIFF sur carte")

# 1. Téléversement des fichiers TIFF
uploaded_files = st.file_uploader("Téléversez vos fichiers TIFF", type=["tiff", "tif"], accept_multiple_files=True)

if uploaded_files:
    # Initialisation des variables dans session_state
    if "current_image" not in st.session_state:
        st.session_state.current_image = 0
    if "markers" not in st.session_state:
        st.session_state.markers = []  # Stockera la liste des marqueurs et leurs informations

    # Navigation entre les images : boutons et slider
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Image précédente"):
            st.session_state.current_image = max(0, st.session_state.current_image - 1)
    with col3:
        if st.button("Image suivante"):
            st.session_state.current_image = min(len(uploaded_files) - 1, st.session_state.current_image + 1)

    # S'assurer que la valeur par défaut du slider est dans l'intervalle
    num_files = len(uploaded_files)
    default_image = st.session_state.current_image if st.session_state.current_image < num_files else 0
    st.session_state.current_image = st.slider(
        "Sélectionnez l'image",
        min_value=0,
        max_value=num_files - 1,
        value=default_image,
        key="current_image_slider"
    )
    st.write(f"Affichage de l'image {st.session_state.current_image + 1} sur {num_files}")

    # 2. Affichage de l'image sur une carte
    current_file = uploaded_files[st.session_state.current_image]
    # Lecture du fichier TIFF depuis la mémoire
    current_bytes = current_file.read()
    with rasterio.MemoryFile(current_bytes) as memfile:
        with memfile.open() as dataset:
            # Récupération des métadonnées : bounds (on suppose ici que l'image est en coordonnées géographiques)
            bounds = dataset.bounds  # (left, bottom, right, top)
            center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
            
            # Lecture de l'image sous forme de tableau numpy
            arr = dataset.read()
            # Si l'image possède au moins 3 canaux, on prend les 3 premiers pour un affichage RGB
            if arr.shape[0] >= 3:
                arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)
            else:
                # Pour un canal unique, affichage en niveaux de gris
                arr = arr[0]
            
            # Conversion du tableau en image PIL puis en PNG encodé en base64
            pil_img = Image.fromarray(arr.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_b64}"

    # Création de la carte Folium centrée sur l'image
    m = folium.Map(location=center, zoom_start=18)
    
    # Ajout de l'image en overlay (semi-transparent)
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    
    # Ajout de l'outil de dessin pour placer des marqueurs (points uniquement)
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
    
    # Affichage de la carte interactive et récupération des dessins
    output = st_folium(m, width=700, height=500, returned_objects=["all_drawings"])
    
    # 3. Attribution d'une classe et d'une gravité après placement du marqueur
    if output and output.get("all_drawings"):
        for drawing in output["all_drawings"]:
            # On s'intéresse aux marqueurs (type "Point")
            geometry = drawing.get("geometry", {})
            if geometry.get("type") == "Point":
                coords = geometry.get("coordinates", [])
                if coords:
                    lon, lat = coords
                    # Vérification pour éviter d'ajouter des doublons
                    if not any(np.isclose(marker["lat"], lat) and np.isclose(marker["lon"], lon)
                               for marker in st.session_state.markers):
                        st.write("Nouveau marqueur détecté :")
                        st.write(f"• Coordonnées : Latitude {lat:.6f}, Longitude {lon:.6f}")
                        st.write("Attribuez-lui une classe et une gravité :")
                        with st.form(key=f"marker_form_{lat}_{lon}"):
                            selected_class = st.selectbox("Sélectionnez la classe", [f"Classe {i+1}" for i in range(13)])
                            selected_severity = st.radio("Sélectionnez la gravité", [1, 2, 3])
                            submitted = st.form_submit_button("Enregistrer le marqueur")
                            if submitted:
                                st.session_state.markers.append({
                                    "lat": lat,
                                    "lon": lon,
                                    "class": selected_class,
                                    "severity": selected_severity,
                                    "image_index": st.session_state.current_image,
                                })
                                st.success("Marqueur enregistré!")
    
    # Affichage de la liste des marqueurs enregistrés pour l'image courante
    st.subheader("Liste des marqueurs enregistrés")
    markers_current = [marker for marker in st.session_state.markers if marker["image_index"] == st.session_state.current_image]
    if markers_current:
        for idx, marker in enumerate(markers_current):
            st.write(f"Marqueur {idx+1} : {marker}")
    else:
        st.write("Aucun marqueur enregistré pour cette image pour le moment.")
