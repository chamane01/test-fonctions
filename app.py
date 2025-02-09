import streamlit as st
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from pyproj import CRS, Transformer
import json
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import Point, LineString
import tempfile
import os
import base64
from io import BytesIO

# --- Patch pour corriger l'erreur de st_canvas liée à image_to_url ---
def image_to_url(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

if not hasattr(st, "image_to_url"):
    st.image_to_url = image_to_url
# --- Fin du patch ---

def get_utm_crs(lat, lon):
    """
    Calcule le CRS UTM à partir du centre (lat, lon) en degrés.
    """
    zone = int((lon + 180) // 6 + 1)
    return CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': lat < 0})

def get_center_lon_lat(bounds, crs):
    """
    Calcule le centre de l'image et le convertit en EPSG:4326 si nécessaire.
    Si le CRS est géographique, on s'assure que le centre soit dans une plage réaliste.
    """
    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.top + bounds.bottom) / 2
    if crs.is_geographic:
        # Si le CRS est géographique, les coordonnées doivent être en degrés
        if not (-180 <= center_x <= 180 and -90 <= center_y <= 90):
            raise ValueError("Les coordonnées du centre ne semblent pas être en degrés. "
                             "L'image semble ne pas être géoréférencée correctement.")
        return center_x, center_y
    else:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("La transformation du centre de l'image a produit des coordonnées invalides.")
        return lon, lat

def process_image(uploaded_file):
    # Sauvegarder le fichier dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with rasterio.open(tmp_path) as src:
        # Si le CRS n'est pas défini, on utilise EPSG:4326 par défaut
        if src.crs is None:
            src_crs = CRS.from_epsg(4326)
        else:
            src_crs = src.crs

        # Calcul du centre de l'image (converti en EPSG:4326 si nécessaire)
        lon, lat = get_center_lon_lat(src.bounds, src_crs)

        # Calcul du CRS UTM à partir du centre en degrés
        utm_crs = get_utm_crs(lat, lon)

        # Calcul de la transformation pour reprojeter l'image
        utm_transform, width, height = calculate_default_transform(
            src_crs, utm_crs, src.width, src.height, *src.bounds
        )

        # Création du tableau de destination pour l'image reprojetée
        reproj_array = np.empty((src.count, height, width), dtype=src.dtypes[0])
        reproject(
            source=src.read(),
            destination=reproj_array,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=utm_transform,
            dst_crs=utm_crs
        )

        original_size = (src.width, src.height)

    # Supprimer le fichier temporaire
    os.unlink(tmp_path)

    # Conversion du tableau reprojeté en image PIL en fonction du nombre de bandes
    if src.count >= 3:
        # Image RGB (on prend les 3 premières bandes)
        rgb_array = np.transpose(reproj_array[:3], (1, 2, 0)).astype(np.uint8)
        pil_img = Image.fromarray(rgb_array, mode='RGB')
    elif src.count == 1:
        # Image en niveaux de gris
        gray_array = reproj_array[0].astype(np.uint8)
        pil_img = Image.fromarray(gray_array, mode='L')
    else:
        raise ValueError(f"Nombre de bandes non supporté : {src.count}")

    return {
        'image': pil_img,
        'transform': utm_transform,
        'crs': utm_crs,
        'original_size': original_size,
        'reprojected_size': (width, height)
    }

def canvas_to_geo(canvas_obj, transform, original_size, display_size):
    """
    Transforme les coordonnées du canvas vers celles de l'image d'origine.
    """
    scale_x = original_size[0] / display_size[0]
    scale_y = original_size[1] / display_size[1]

    if canvas_obj['type'] == 'rect':
        x = canvas_obj['left'] * scale_x
        y = canvas_obj['top'] * scale_y
        return transform * (x, y)
    
    elif canvas_obj['type'] == 'path':
        # Extraction des points de la trajectoire
        points = [(cmd[1], cmd[2]) for cmd in canvas_obj['path'] if cmd[0] in ['M', 'L']]
        scaled_points = [(x * scale_x, y * scale_y) for x, y in points]
        return [transform * (x, y) for x, y in scaled_points]
    
    return None

st.title("Éditeur d'Images Géoréférencées")

uploaded_files = st.file_uploader(
    "Télécharger des images", 
    type=['tif', 'tiff', 'jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    processed_images = []
    for uploaded_file in uploaded_files:
        try:
            processed = process_image(uploaded_file)
            processed['name'] = uploaded_file.name
            processed_images.append(processed)
        except Exception as e:
            st.error(f"Erreur de traitement de {uploaded_file.name}: {str(e)}")

    if processed_images:
        selected_image = st.selectbox(
            "Sélectionner une image", 
            processed_images, 
            format_func=lambda x: x['name']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            display_width = st.slider("Largeur d'affichage", 400, 1200, 800)
            # Redimensionnement de l'image pour l'affichage
            display_img = selected_image['image'].resize(
                (display_width, int(display_width * selected_image['image'].height / selected_image['image'].width))
            )
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_image=display_img,
                height=display_img.height,
                width=display_width,
                drawing_mode="freedraw",
                key="canvas"
            )

        with col2:
            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                features = []
                for obj in canvas_result.json_data['objects']:
                    geo = canvas_to_geo(
                        obj, 
                        selected_image['transform'], 
                        selected_image['original_size'], 
                        (display_width, display_img.height)
                    )
                    if geo:
                        if obj['type'] == 'rect':
                            geometry = Point(geo)
                        elif obj['type'] == 'path':
                            if isinstance(geo, list) and len(geo) > 1:
                                geometry = LineString(geo)
                            else:
                                geometry = Point(geo[0] if isinstance(geo, list) else geo)
                        features.append({
                            'type': 'Feature',
                            'geometry': geometry.__geo_interface__,
                            'properties': {}
                        })

                if features:
                    geojson = {
                        'type': 'FeatureCollection',
                        'features': features,
                        'crs': {
                            'type': 'name',
                            'properties': {'name': str(selected_image['crs'].to_authority())}
                        }
                    }
                    
                    st.download_button(
                        label="Télécharger GeoJSON",
                        data=json.dumps(geojson, indent=2),
                        file_name="annotations.geojson",
                        mime="application/json"
                    )
                    st.json(geojson)
