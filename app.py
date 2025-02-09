import streamlit as st
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from pyproj import Transformer, CRS
import json
from streamlit_drawable_canvas import st_canvas
from shapely.geometry import shape, Point, LineString, Polygon
import geopandas as gpd
import tempfile
import os

def get_utm_crs(lat, lon):
    zone = int((lon + 180) // 6 + 1)
    return CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': lat < 0})

def process_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    with rasterio.open(tmp_path) as src:
        img_array = src.read()
        transform = src.transform
        crs = src.crs

        # Calculate UTM CRS
        lon, lat = src.lnglat()
        utm_crs = get_utm_crs(lat, lon)

        # Reproject transform
        utm_transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        
        # Create reprojected image array
        reproj_array = np.empty((src.count, height, width), dtype=src.dtypes[0])
        reproject(
            source=src.read(),
            destination=reproj_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=utm_transform,
            dst_crs=utm_crs
        )

    os.unlink(tmp_path)
    return {
        'image': Image.fromarray(np.transpose(reproj_array[:3], [1,2,0]).astype(np.uint8)),
        'transform': utm_transform,
        'crs': utm_crs,
        'original_size': (src.width, src.height),
        'reprojected_size': (width, height)
    }

def canvas_to_geo(canvas_obj, transform, original_size, display_size):
    scale_x = original_size[0] / display_size[0]
    scale_y = original_size[1] / display_size[1]
    
    if canvas_obj['type'] == 'rect':
        x = canvas_obj['left'] * scale_x
        y = canvas_obj['top'] * scale_y
        return transform * (x, y)
    
    elif canvas_obj['type'] == 'path':
        points = [(cmd[1], cmd[2]) for cmd in canvas_obj['path'] if cmd[0] in ['M', 'L']]
        scaled_points = [(x * scale_x, y * scale_y) for x, y in points]
        return [transform * (x, y) for x, y in scaled_points]
    
    return None

st.title("Éditeur d'Images Géoréférencées")

uploaded_files = st.file_uploader("Télécharger des images", 
                                type=['tif', 'tiff', 'jpg', 'jpeg', 'png'], 
                                accept_multiple_files=True)

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
        selected_image = st.selectbox("Sélectionner une image", processed_images, format_func=lambda x: x['name'])
        
        col1, col2 = st.columns(2)
        with col1:
            display_width = st.slider("Largeur d'affichage", 400, 1200, 800)
            display_img = selected_image['image'].resize((display_width, int(display_width * selected_image['image'].height / selected_image['image'].width)))

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
            if canvas_result.json_data is not None:
                features = []
                for obj in canvas_result.json_data['objects']:
                    geo = canvas_to_geo(obj, 
                                      selected_image['transform'], 
                                      selected_image['original_size'], 
                                      (display_width, display_img.height))
                    
                    if geo:
                        if obj['type'] == 'rect':
                            geometry = Point(geo)
                        elif obj['type'] == 'path':
                            if len(geo) > 1:
                                geometry = LineString(geo)
                            else:
                                geometry = Point(geo[0])
                        
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
