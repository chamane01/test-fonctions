import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import rasterio
import numpy as np
from PIL import Image
import io, base64
import utm
import pandas as pd

st.set_page_config(layout="wide")
st.title("Annotation d'images TIFF sur carte")

# 0. Choix du mode d'annotation dans la sidebar
mode = st.sidebar.radio("Mode de classification", 
                         ["Sélectionner avant placement", "Sélectionner après placement"])

if mode == "Sélectionner avant placement":
    default_class = st.sidebar.selectbox("Sélectionnez la classe par défaut", [f"Classe {i+1}" for i in range(13)])
    default_severity = st.sidebar.radio("Sélectionnez la gravité par défaut", [1, 2, 3])

# 1. Téléversement des fichiers TIFF
uploaded_files = st.file_uploader("Téléversez vos fichiers TIFF", type=["tiff", "tif"], accept_multiple_files=True)

if uploaded_files:
    # Initialisation des variables dans session_state
    if "current_image" not in st.session_state:
        st.session_state.current_image = 0
    if "markers" not in st.session_state:
        st.session_state.markers = []  # Contiendra tous les marqueurs avec leurs infos

    # Navigation entre les images : boutons et slider
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Image précédente"):
            st.session_state.current_image = max(0, st.session_state.current_image - 1)
    with col3:
        if st.button("Image suivante"):
            st.session_state.current_image = min(len(uploaded_files) - 1, st.session_state.current_image + 1)
    
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

    # 2. Lecture du fichier TIFF et affichage sur une carte
    current_file = uploaded_files[st.session_state.current_image]
    current_bytes = current_file.read()
    with rasterio.MemoryFile(current_bytes) as memfile:
        with memfile.open() as dataset:
            # Récupération des métadonnées : bounds (ici supposés en coordonnées géographiques)
            bounds = dataset.bounds  # (left, bottom, right, top)
            center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
            
            # Lecture de l'image sous forme de tableau numpy
            arr = dataset.read()
            if arr.shape[0] >= 3:
                arr = np.stack([arr[0], arr[1], arr[2]], axis=-1)
            else:
                arr = arr[0]
            
            # Conversion en image PIL puis en PNG encodé en base64
            pil_img = Image.fromarray(arr.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_url = f"data:image/png;base64,{img_b64}"

    # Création de la carte Folium avec un zoom maximum élevé
    m = folium.Map(location=center, zoom_start=18, max_zoom=22)
    
    # Ajout de l'image en overlay
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
    
    # 3. Traitement des marqueurs ajoutés et attribution de classe et gravité
    if output and output.get("all_drawings"):
        for drawing in output["all_drawings"]:
            geometry = drawing.get("geometry", {})
            if geometry.get("type") == "Point":
                coords = geometry.get("coordinates", [])
                if coords:
                    lon, lat = coords
                    # Vérifier que le marqueur n'est pas déjà enregistré
                    if not any(np.isclose(marker["lat"], lat) and np.isclose(marker["lon"], lon)
                               for marker in st.session_state.markers):
                        # Conversion en coordonnées UTM
                        try:
                            utm_coords = utm.from_latlon(lat, lon)
                            utm_dict = {
                                "easting": utm_coords[0],
                                "northing": utm_coords[1],
                                "zone_number": utm_coords[2],
                                "zone_letter": utm_coords[3]
                            }
                        except Exception as e:
                            utm_dict = {"error": str(e)}
                        
                        st.write("Nouveau marqueur détecté :")
                        st.write(f"• Coordonnées : Latitude {lat:.6f}, Longitude {lon:.6f}")
                        st.write(f"• Coordonnées UTM : {utm_dict}")
                        
                        if mode == "Sélectionner après placement":
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
                                        "utm": utm_dict
                                    })
                                    st.success("Marqueur enregistré!")
                        else:
                            # Mode "Sélectionner avant placement" : on utilise les valeurs par défaut choisies
                            st.session_state.markers.append({
                                "lat": lat,
                                "lon": lon,
                                "class": default_class,
                                "severity": default_severity,
                                "image_index": st.session_state.current_image,
                                "utm": utm_dict
                            })
                            st.success(f"Marqueur enregistré automatiquement (Classe : {default_class}, Gravité : {default_severity})")
    
    # Affichage de la liste des marqueurs enregistrés pour l'image courante
    st.subheader("Liste des marqueurs enregistrés")
    markers_current = [marker for marker in st.session_state.markers if marker["image_index"] == st.session_state.current_image]
    if markers_current:
        for idx, marker in enumerate(markers_current):
            st.write(f"Marqueur {idx+1} : Classe = {marker['class']}, Gravité = {marker['severity']}, Coordonnées UTM = {marker['utm']}")
    else:
        st.write("Aucun marqueur enregistré pour cette image pour le moment.")
    
    # 4. Bouton pour exporter le rapport des marqueurs au format CSV
    st.write("---")
    if st.button("Exporter rapport"):
        if st.session_state.markers:
            df = pd.DataFrame(st.session_state.markers)
            # Ne garder que les colonnes d'intérêt
            df = df[['class', 'severity', 'utm']]
            df = df.rename(columns={"class": "Classe", "severity": "Gravité", "utm": "PositionUTM"})
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le rapport en CSV",
                data=csv,
                file_name='rapport_markers.csv',
                mime='text/csv'
            )
        else:
            st.info("Aucun marqueur à exporter.")
