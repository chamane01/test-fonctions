import streamlit as st
import pandas as pd
import json
import altair as alt
import plotly.express as px
import pydeck as pdk
from datetime import datetime, date, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from io import BytesIO
import calendar
import matplotlib.pyplot as plt

# Imports pour la partie Missions
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as rio_transform
from PIL import Image, ImageOps, ImageDraw
import exifread
import numpy as np
import os
from pyproj import Transformer
import io
import math
from affine import Affine
import zipfile
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
import base64
import uuid
import csv
import random, string
from shapely.geometry import Point, LineString

#############################################
# Configuration générale et CSS
#############################################
st.set_page_config(page_title="Suivi des Dégradations sur Routes Ivoiriennes", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
        color: #333;
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stMetric {
        background: linear-gradient(90deg, #8e44ad, #3498db);
        color: #fff;
        padding: 10px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True
)

#############################################
# Menu principal : Tableau de bord, Missions, Rapport
#############################################
menu_option = st.sidebar.radio("Menu", ["Tableau de bord", "Missions", "Rapport"])

#############################################
# Données et fonctions communes pour Tableau de bord et Rapport
#############################################
# Chargement des données du suivi (JSON)
with open("jeu_donnees_missions (1).json", "r", encoding="utf-8") as f:
    data = json.load(f)
missions_df = pd.DataFrame(data)
missions_df['date'] = pd.to_datetime(missions_df['date'])

# Extraction des défauts depuis le JSON
defects = []
for mission in data:
    for defect in mission.get("Données Défauts", []):
        defects.append(defect)
df_defects = pd.DataFrame(defects)
if 'date' in df_defects.columns:
    df_defects['date'] = pd.to_datetime(df_defects['date'])

#############################################
# Fonction de génération du rapport PDF
#############################################
def generate_report_pdf(report_type, missions_df, df_defects, metadata, start_date, end_date):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    PAGE_WIDTH, PAGE_HEIGHT = A4
    margin = 40

    # En-tête avec le logo
    logo_path = "images (5).png"
    try:
        img = ImageReader(logo_path)
        c.drawImage(img, margin, PAGE_HEIGHT - margin - 50, width=50, height=50, preserveAspectRatio=True, mask='auto')
    except Exception as e:
        st.error(f"Erreur de chargement du logo: {str(e)}")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin + 60, PAGE_HEIGHT - margin - 20, metadata.get("titre", "Rapport de Suivi"))
    c.setFont("Helvetica", 12)
    c.drawString(margin + 60, PAGE_HEIGHT - margin - 40, f"Type: {report_type} | Période: {start_date} au {end_date}")
    c.drawString(margin + 60, PAGE_HEIGHT - margin - 60, f"Éditeur: {metadata.get('editor', 'Inconnu')}")
    
    # Calcul des indicateurs pour la période
    filtered_missions = missions_df[(missions_df['date'].dt.date >= start_date) & (missions_df['date'].dt.date <= end_date)]
    num_missions = len(filtered_missions)
    total_distance = filtered_missions["distance(km)"].sum() if "distance(km)" in filtered_missions.columns else 0
    avg_distance = filtered_missions["distance(km)"].mean() if ("distance(km)" in filtered_missions.columns and num_missions > 0) else 0

    # On utilise ici uniquement les défauts issus du JSON (pour le rapport)
    filtered_defects = df_defects[(df_defects['date'].dt.date >= start_date) & (df_defects['date'].dt.date <= end_date)] if 'date' in df_defects.columns else df_defects
    total_defects = len(filtered_defects)
    
    # Affichage des indicateurs
    y = PAGE_HEIGHT - margin - 100
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Métriques Principales")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Nombre de Missions: {num_missions}")
    y -= 20
    c.drawString(margin, y, f"Distance Totale (km): {total_distance:.1f}")
    y -= 20
    c.drawString(margin, y, f"Distance Moyenne (km): {avg_distance:.1f}")
    y -= 20
    c.drawString(margin, y, f"Nombre de Défauts: {total_defects}")
    
    # Zone pour les graphiques
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Graphiques et Analyses")
    y -= 20
    c.setFont("Helvetica", 10)
    
    # Premier graphique : Bar chart
    fig1, ax1 = plt.subplots(figsize=(4,3))
    categories = ['Missions', 'Défauts']
    values = [num_missions, total_defects]
    bars = ax1.bar(categories, values, color=['#3498db', '#e74c3c'])
    ax1.set_title("Comparaison Missions / Défauts", fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    buf1 = BytesIO()
    plt.savefig(buf1, format='PNG', dpi=100)
    plt.close(fig1)
    buf1.seek(0)
    img_chart1 = ImageReader(buf1)
    
    # Second graphique : Pie chart avec texte réduit et rayon augmenté
    fig2, ax2 = plt.subplots(figsize=(4,3))
    if not filtered_defects.empty and "classe" in filtered_defects.columns:
        defect_counts = filtered_defects['classe'].value_counts()
        ax2.pie(defect_counts, labels=defect_counts.index, autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.Pastel1.colors, textprops={'fontsize': 6}, radius=1.1)
        ax2.set_title("Répartition Défauts par Catégorie", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Pas de données", horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    buf2 = BytesIO()
    plt.savefig(buf2, format='PNG', dpi=100)
    plt.close(fig2)
    buf2.seek(0)
    img_chart2 = ImageReader(buf2)
    
    # Positionnement côte à côte des graphiques
    chart_width = (PAGE_WIDTH - 3 * margin) / 2
    chart_height = 200
    y_chart = y - chart_height - 20
    c.drawImage(img_chart1, margin, y_chart, width=chart_width, height=chart_height, preserveAspectRatio=True, mask='auto')
    c.drawImage(img_chart2, margin + chart_width + margin, y_chart, width=chart_width, height=chart_height, preserveAspectRatio=True, mask='auto')
    
    # Pied de page
    c.setFont("Helvetica", 8)
    c.drawRightString(PAGE_WIDTH - margin, margin, f"Rapport généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

#############################################
# Affichage pour Tableau de bord, Missions et Rapport
#############################################
if menu_option == "Tableau de bord":
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("images (5).png", width=200)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.title("Tableau de bord de Suivi des Dégradations sur les Routes Ivoiriennes")
    st.markdown("Parce que nous croyons que la route précède le développement")
    
    if "distance(km)" not in missions_df.columns:
        st.error("Le fichier ne contient pas la colonne 'distance(km)'.")
    else:
        # Récupération des missions historiques
        hist = st.session_state.get("mission_history", [])
        num_hist = len(hist)
        total_distance_hist = sum(float(m.get("distance(km)", 0)) for m in hist)
        # Indicateurs issus du JSON
        num_missions_json = len(missions_df)
        total_distance_json = missions_df["distance(km)"].sum()
        # Combinaison des missions
        total_missions = num_missions_json + num_hist
        total_distance_all = total_distance_json + total_distance_hist
        avg_distance_all = total_distance_all / total_missions if total_missions > 0 else 0
        
        # Fusion des défauts issus du JSON avec ceux des missions historiques
        mission_ids = [m["id"] for m in st.session_state.get("mission_history", [])]
        markers_list = []
        for markers in st.session_state.get("markers_by_pair", {}).values():
            for marker in markers:
                if marker.get("mission") in mission_ids:
                    markers_list.append(marker)
        if markers_list:
            df_markers = pd.DataFrame(markers_list)
            if "date" in df_markers.columns:
                df_markers['date'] = pd.to_datetime(df_markers['date'], errors='coerce')
            df_defects_combined = pd.concat([df_defects, df_markers], ignore_index=True)
        else:
            df_defects_combined = df_defects.copy()
        total_defects_all = len(df_defects_combined)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de Missions", total_missions)
        col2.metric("Nombre de Défauts", total_defects_all)
        col3.metric("Distance Totale (km)", f"{total_distance_all:.1f}")
        col4.metric("Distance Moyenne (km)", f"{avg_distance_all:.1f}")
        
        st.markdown("---")
        # Graphiques interactifs
        missions_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M")).size().reset_index(name="Missions")
        missions_over_time['date'] = missions_over_time['date'].dt.to_timestamp()
        chart_missions = alt.Chart(missions_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Missions:Q', title='Nombre de Missions'),
            tooltip=['date:T', 'Missions:Q']
        ).properties(width=350, height=300, title="Évolution des Missions")
        
        defects_over_time = df_defects_combined.groupby(df_defects_combined['date'].dt.to_period("M")).size().reset_index(name="Défauts")
        defects_over_time['date'] = defects_over_time['date'].dt.to_timestamp()
        chart_defects_time = alt.Chart(defects_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Défauts:Q', title='Nombre de Défauts'),
            tooltip=['date:T', 'Défauts:Q']
        ).properties(width=350, height=300, title="Évolution des Défauts")
        
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            st.altair_chart(chart_missions, use_container_width=True)
        with col_time2:
            st.altair_chart(chart_defects_time, use_container_width=True)
        
        st.markdown("---")
        distance_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M"))["distance(km)"].sum().reset_index(name="Distance Totale")
        distance_over_time['date'] = distance_over_time['date'].dt.to_timestamp()
        chart_distance_time = alt.Chart(distance_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Distance Totale:Q', title='Km Totaux', scale=alt.Scale(domain=[0, distance_over_time["Distance Totale"].max()*1.2])),
            tooltip=['date:T', 'Distance Totale:Q']
        ).properties(width=350, height=300, title="Évolution des Km")
        
        gravity_sizes = {1: 5, 2: 7, 3: 9}
        if 'gravite' in df_defects_combined.columns:
            df_defects_combined['severite'] = df_defects_combined['gravite'].map(gravity_sizes)
        severity_over_time = df_defects_combined.groupby(df_defects_combined['date'].dt.to_period("M"))["severite"].mean().reset_index(name="Score Moyen")
        severity_over_time['date'] = severity_over_time['date'].dt.to_timestamp()
        chart_severity_over_time = alt.Chart(severity_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Score Moyen:Q', title='Score de Sévérité Moyen'),
            tooltip=['date:T', 'Score Moyen:Q']
        ).properties(width=350, height=300, title="Score de Sévérité Moyen")
        
        col_time3, col_time4 = st.columns(2)
        with col_time3:
            st.altair_chart(chart_distance_time, use_container_width=True)
        with col_time4:
            st.altair_chart(chart_severity_over_time, use_container_width=True)
        
        st.markdown("---")
        if 'classe' in df_defects_combined.columns:
            defect_category_counts = df_defects_combined['classe'].value_counts().reset_index()
            defect_category_counts.columns = ["Catégorie", "Nombre de Défauts"]
            fig_pie = px.pie(defect_category_counts, values='Nombre de Défauts', names='Catégorie',
                             title="Répartition Globale des Défauts par Catégorie",
                             color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        if 'lat' in df_defects_combined.columns and 'long' in df_defects_combined.columns:
            def get_red_color(gravite):
                if gravite == 1:
                    return [255, 200, 200]
                elif gravite == 2:
                    return [255, 100, 100]
                elif gravite == 3:
                    return [255, 0, 0]
                else:
                    return [255, 0, 0]
            df_defects_combined['marker_color'] = df_defects_combined['gravite'].apply(get_red_color)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_defects_combined,
                get_position='[long, lat]',
                get_color="marker_color",
                get_radius="severite * 50",
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=df_defects_combined['lat'].mean(),
                longitude=df_defects_combined['long'].mean(),
                zoom=8,
                pitch=0,
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Route: {routes}\nClasse: {classe}\nGravité: {gravite}"}
            )
            st.pydeck_chart(deck)
        
        st.markdown("---")
        show_all = st.checkbox("Afficher tous les éléments", value=False)
        limit = None if show_all else 7
        route_defect_counts = df_defects_combined['routes'].value_counts().reset_index()
        route_defect_counts.columns = ["Route", "Nombre de Défauts"]
        display_routes = route_defect_counts if show_all else route_defect_counts.head(limit)
        chart_routes = alt.Chart(display_routes).mark_bar().encode(
            x=alt.X("Route:N", sort='-y', title="Route", axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            tooltip=["Route:N", "Nombre de Défauts:Q"],
            color=alt.Color("Route:N", scale=alt.Scale(scheme='tableau10'))
        ).properties(width=450, height=500, title="Nombre de Défauts par Route")
        route_severity = df_defects_combined.groupby('routes')['severite'].sum().reset_index().sort_values(by='severite', ascending=False)
        display_severity = route_severity if show_all else route_severity.head(limit)
        chart_severity = alt.Chart(display_severity).mark_bar().encode(
            x=alt.X("routes:N", sort='-y', title="Route", axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("severite:Q", title="Score de Sévérité Total"),
            tooltip=["routes:N", "severite:Q"],
            color=alt.Color("routes:N", scale=alt.Scale(scheme='tableau20'))
        ).properties(width=450, height=500, title="Score de Sévérité par Route")
        
        col_route, col_severity = st.columns(2)
        with col_route:
            st.altair_chart(chart_routes, use_container_width=True)
        with col_severity:
            st.altair_chart(chart_severity, use_container_width=True)
        
        st.markdown("### Analyse par Type de Défaut")
        defect_types = df_defects_combined['classe'].unique()
        selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
        filtered_defects_type = df_defects_combined[df_defects_combined['classe'] == selected_defect]
        if not filtered_defects_type.empty:
            route_count_selected = filtered_defects_type['routes'].value_counts().reset_index()
            route_count_selected.columns = ["Route", "Nombre de Défauts"]
            display_selected = route_count_selected if show_all else route_count_selected.head(limit)
            chart_defect_type = alt.Chart(display_selected).mark_bar().encode(
                x=alt.X("Route:N", sort='-y', title="Route", axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
                y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
                tooltip=["Route:N", "Nombre de Défauts:Q"],
                color=alt.Color("Route:N", scale=alt.Scale(scheme='category20b'))
            ).properties(width=900, height=500, title=f"Répartition des Défauts pour le Type : {selected_defect} (Top 7 par défaut)")
            st.altair_chart(chart_defect_type, use_container_width=True)
        else:
            st.write("Aucune donnée disponible pour ce type de défaut.")
        
        st.markdown("---")
        st.markdown("### Analyse par Route")
        selected_route = st.selectbox("Sélectionnez une route", sorted(df_defects_combined['routes'].unique()))
        inventory = df_defects_combined[df_defects_combined['routes'] == selected_route]['classe'].value_counts().reset_index()
        inventory.columns = ["Dégradation", "Nombre de Défauts"]
        chart_route_inventory = alt.Chart(inventory).mark_bar().encode(
            x=alt.X("Dégradation:N", sort='-y', title="Dégradation", axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            tooltip=["Dégradation:N", "Nombre de Défauts:Q"],
            color=alt.Color("Dégradation:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=900, height=500, title=f"Inventaire des Dégradations pour la Route : {selected_route}")
        st.altair_chart(chart_route_inventory, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif menu_option == "Missions":
    ######################################################################################
    # Section Missions : Création, gestion, post-traitement, détection, suivi et export
    ######################################################################################
    def extract_exif_info(image_file):
        image_file.seek(0)
        tags = exifread.process_file(image_file)
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
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            utm_crs = f"EPSG:326{zone:02d}"
        else:
            utm_crs = f"EPSG:327{zone:02d}"
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        utm_x, utm_y = transformer.transform(lon, lat)
        return utm_x, utm_y, utm_crs

    def compute_gsd(altitude, focal_length_mm, sensor_width_mm, image_width_px):
        focal_length_m = focal_length_mm / 1000.0
        sensor_width_m = sensor_width_mm / 1000.0
        return (altitude * sensor_width_m) / (focal_length_m * image_width_px)

    def convert_to_tiff_in_memory(image_file, pixel_size, utm_center, utm_crs, rotation_angle=0, scaling_factor=1):
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
            count=3 if len(img_array.shape)==3 else 1,
            dtype=img_array.dtype,
            crs=utm_crs,
            transform=transform
        ) as dst:
            if len(img_array.shape)==3:
                for i in range(3):
                    dst.write(img_array[:, :, i], i+1)
            else:
                dst.write(img_array, 1)
        return memfile.read()

    def assign_route_to_marker(lat, lon, routes):
        marker_point = Point(lon, lat)
        min_distance = float('inf')
        closest_route = "Route inconnue"
        for route in routes:
            line = LineString(route["coords"])
            distance = marker_point.distance(line)
            if distance < min_distance:
                min_distance = distance
                closest_route = route["nom"]
        threshold = 0.01
        return closest_route if min_distance <= threshold else "Route inconnue"

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
                for i in range(1, src.count+1):
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
        return (255 * (norm_data - lower) / (upper - lower)).astype(np.uint8)

    def create_map(center_lat, center_lon, bounds, display_path, marker_data=None,
                   hide_osm=False, tiff_opacity=1, tiff_show=True, tiff_control=True,
                   draw_routes=True, add_draw_tool=True):
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
            m.add_child(MeasureControl())
        m.fit_bounds([[bounds.bottom, bounds.left], [bounds.top, bounds.right]])
        folium.LayerControl().add_to(m)
        if marker_data:
            for marker in marker_data:
                if marker.get("lat") is not None and marker.get("long") is not None:
                    color = class_color.get(marker["classe"], "#000000")
                    radius = gravity_sizes.get(marker["gravite"], 5)
                    folium.CircleMarker(
                        location=[marker["lat"], marker["long"]],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        tooltip=f"{marker['ID']} - {marker['classe']} (Gravité {marker['gravite']}) - Route : {marker.get('routes', 'Route inconnue')} - Détection: {marker.get('detection', 'Inconnue')} - Mission: {marker.get('mission', 'N/A')}"
                    ).add_to(m)
        return m

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

    def generate_mission_id(date_mission, counter):
        random_letter = random.choice(string.ascii_uppercase)
        return f"{date_mission.strftime('%Y%m%d')}-{counter:03d}-{random_letter}"

    #########################################
    # Variables globales pour Missions
    #########################################
    if "current_pair_index" not in st.session_state:
        st.session_state.current_pair_index = 0
    if "pairs" not in st.session_state:
        st.session_state.pairs = []
    if "markers_by_pair" not in st.session_state:
        st.session_state.markers_by_pair = {}
    if "mission_marker_counter" not in st.session_state:
        st.session_state.mission_marker_counter = {}
    if "missions" not in st.session_state:
        st.session_state.missions = {}
    if "mission_counter" not in st.session_state:
        st.session_state.mission_counter = 1
    if "mission_history" not in st.session_state:
        st.session_state.mission_history = []

    # Dictionnaire des classes et niveaux de gravité
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
    gravity_sizes = {1: 5, 2: 7, 3: 9}

    # Chargement des routes
    with open("routeQSD.txt", "r") as f:
        routes_data = json.load(f)
    routes_ci = []
    for feature in routes_data["features"]:
        if feature["geometry"]["type"] == "LineString":
            routes_ci.append({
                "coords": feature["geometry"]["coordinates"],
                "nom": feature["properties"].get("ID", "Route inconnue")
            })

    #########################################
    # Création et gestion des missions dans la sidebar
    #########################################
    st.sidebar.header("Création de mission")
    with st.sidebar.form("mission_form"):
        operator = st.text_input("Nom de l'opérateur")
        appareil_type = st.selectbox("Type d'appareil", ["Drone", "Camera"])
        nom_appareil = st.text_input("Nom de l'appareil (Drone ou Camera)")
        date_mission = st.date_input("Date de la mission")
        troncon = st.text_input("Tronçon")
        mission_submit = st.form_submit_button("Créer la mission")
        if mission_submit:
            new_mission_id = generate_mission_id(date_mission, st.session_state.mission_counter)
            st.session_state.mission_counter += 1
            st.session_state.missions[new_mission_id] = {
                "id": new_mission_id,
                "operator": operator,
                "appareil_type": appareil_type,
                "nom_appareil": nom_appareil,
                "date": date_mission.strftime("%Y-%m-%d"),
                "troncon": troncon
            }
            st.session_state.current_mission = new_mission_id
            st.success(f"Mission {new_mission_id} créée.")

    st.sidebar.subheader("Sélection de mission")
    if st.session_state.missions:
        mission_list = list(st.session_state.missions.keys())
        if "current_mission" not in st.session_state or st.session_state.current_mission not in mission_list:
            st.session_state.current_mission = mission_list[0]
        current_mission = st.sidebar.selectbox("Sélectionnez la mission", mission_list, index=mission_list.index(st.session_state.current_mission))
        st.session_state.current_mission = current_mission
    else:
        st.sidebar.info("Aucune mission disponible.")

    # Historique des missions
    global_markers = []
    for markers in st.session_state.get("markers_by_pair", {}).values():
        global_markers.extend(markers)

    st.sidebar.subheader("Historique des missions sauvegardées")
    if st.session_state.mission_history:
        mission_markers_map = {}
        for marker in global_markers:
            mission_id = marker.get("mission", "N/A")
            if mission_id not in mission_markers_map:
                mission_markers_map[mission_id] = []
            mission_markers_map[mission_id].append(marker)
        mission_history_display = []
        for mission in st.session_state.mission_history:
            m = mission.copy()
            markers_for_mission = mission_markers_map.get(m["id"], [])
            m["Données Défauts"] = markers_for_mission  # Conserver la liste pour d'éventuels traitements
            mission_history_display.append(m)
        df_missions_hist = pd.DataFrame(mission_history_display)
        st.sidebar.table(df_missions_hist)
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_missions_hist.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
        
        csv_buffer = io.StringIO()
        df_missions_hist.to_csv(csv_buffer, index=False, sep=";")
        csv_data = csv_buffer.getvalue().encode("utf-8")
        
        txt_data = df_missions_hist.to_string(index=False)
        txt_data = txt_data.encode("utf-8")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("mission_history.xlsx", excel_data)
            zip_file.writestr("mission_history.csv", csv_data)
            zip_file.writestr("mission_history.txt", txt_data)
        zip_buffer.seek(0)
        st.sidebar.download_button(
            label="Télécharger l'historique des missions (ZIP)",
            data=zip_buffer,
            file_name="mission_history.zip",
            mime="application/zip"
        )
    else:
        st.sidebar.info("Aucune mission sauvegardée.")

    #########################################
    # Affichage sur la page Missions
    #########################################
    st.title("TRAITEMENTS")
    st.header("Post-traitements")
    uploaded_files = st.file_uploader(
        "Téléversez une ou plusieurs images (JPG/JPEG)",
        type=["jpg", "jpeg"],
        accept_multiple_files=True
    )
    images_info = []
    if uploaded_files:
        for up_file in uploaded_files:
            file_bytes = up_file.read()
            file_buffer = io.BytesIO(file_bytes)
            lat, lon, altitude, focal_length, fp_x_res, fp_unit = extract_exif_info(file_buffer)
            if lat is None or lon is None:
                st.warning(f"{up_file.name} : pas de coordonnées GPS, l'image sera ignorée.")
                continue
            img = Image.open(io.BytesIO(file_bytes))
            img_width, img_height = img.size
            sensor_width_mm = None
            if fp_x_res and fp_unit:
                if fp_unit == 2:
                    sensor_width_mm = (img_width / fp_x_res) * 25.4
                elif fp_unit == 3:
                    sensor_width_mm = (img_width / fp_x_res) * 10
                elif fp_unit == 4:
                    sensor_width_mm = (img_width / fp_x_res)
            utm_x, utm_y, utm_crs = latlon_to_utm(lat, lon)
            images_info.append({
                "filename": up_file.name,
                "data": file_bytes,
                "lat": lat,
                "lon": lon,
                "altitude": altitude,
                "focal_length": focal_length,
                "sensor_width": sensor_width_mm,
                "utm": (utm_x, utm_y),
                "utm_crs": utm_crs,
                "img_width": img_width,
                "img_height": img_height
            })
        st.session_state["images_info"] = images_info
        if len(images_info) > 1:
            total_distance = 0
            for i in range(1, len(images_info)):
                x1, y1 = images_info[i-1]["utm"]
                x2, y2 = images_info[i]["utm"]
                total_distance += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            st.session_state["trajet_distance_km"] = total_distance / 1000
        elif len(images_info) == 1:
            st.session_state["trajet_distance_km"] = 0

        if len(images_info) == 0:
            st.error("Aucune image exploitable (avec coordonnées GPS) n'a été trouvée.")
        else:
            pixel_size = st.number_input(
                "Choisissez la résolution spatiale (m/pixel) :", 
                min_value=0.001, 
                value=0.03, 
                step=0.001, 
                format="%.3f"
            )
            st.info(f"Résolution spatiale appliquée : {pixel_size*100:.1f} cm/pixel")
            if st.button("Générer les images prétraitées"):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for i, info in enumerate(images_info):
                        if len(images_info) >= 2:
                            if i == 0:
                                dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                                dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                            elif i == len(images_info) - 1:
                                dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                                dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                            else:
                                dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                                dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                            flight_angle_i = math.degrees(math.atan2(dx, dy))
                        else:
                            flight_angle_i = 0
                        tiff_bytes = convert_to_tiff_in_memory(
                            image_file=io.BytesIO(info["data"]),
                            pixel_size=pixel_size,
                            utm_center=info["utm"],
                            utm_crs=info["utm_crs"],
                            rotation_angle=-flight_angle_i,
                            scaling_factor=1/5
                        )
                        output_filename_tiff1 = info["filename"].rsplit(".", 1)[0] + "_geotiff.tif"
                        zip_file.writestr(output_filename_tiff1, tiff_bytes)
                        tiff_bytes_x2 = convert_to_tiff_in_memory(
                            image_file=io.BytesIO(info["data"]),
                            pixel_size=pixel_size * 2,
                            utm_center=info["utm"],
                            utm_crs=info["utm_crs"],
                            rotation_angle=-flight_angle_i,
                            scaling_factor=1/3
                        )
                        output_filename_tiff2 = info["filename"].rsplit(".", 1)[0] + "_geotiff_x2.tif"
                        zip_file.writestr(output_filename_tiff2, tiff_bytes_x2)
                        rotation_angle_i = -flight_angle_i
                        scaling_factor = 1
                        img = Image.open(io.BytesIO(info["data"]))
                        img = ImageOps.exif_transpose(img)
                        orig_width, orig_height = img.size
                        new_width = int(orig_width * scaling_factor)
                        new_height = int(orig_height * scaling_factor)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        effective_pixel_size = pixel_size / scaling_factor
                        center_x, center_y = info["utm"]
                        T1 = Affine.translation(-new_width/2, -new_height/2)
                        T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
                        T3 = Affine.rotation(rotation_angle_i)
                        T4 = Affine.translation(center_x, center_y)
                        transform_affine = T4 * T3 * T2 * T1
                        corners = [(-new_width/2, -new_height/2),
                                   (new_width/2, -new_height/2),
                                   (new_width/2, new_height/2),
                                   (-new_width/2, new_height/2)]
                        corner_coords = []
                        for corner in corners:
                            x, y = transform_affine * corner
                            corner_coords.append((x, y))
                        metadata_str = f"Frame Coordinates: {corner_coords}"
                        try:
                            import piexif
                            if "exif" in img.info:
                                exif_dict = piexif.load(img.info["exif"])
                            else:
                                exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
                            user_comment = metadata_str
                            try:
                                exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.dump(user_comment, encoding="unicode")
                            except AttributeError:
                                prefix = b"UNICODE\0"
                                encoded_comment = user_comment.encode("utf-16")
                                exif_dict["Exif"][piexif.ExifIFD.UserComment] = prefix + encoded_comment
                            exif_bytes = piexif.dump(exif_dict)
                        except ImportError:
                            st.error("La librairie piexif est requise pour ajouter des métadonnées JPEG. Veuillez l'installer.")
                            exif_bytes = None
                        jpeg_buffer = io.BytesIO()
                        if exif_bytes:
                            img.save(jpeg_buffer, format="JPEG", exif=exif_bytes)
                        else:
                            img.save(jpeg_buffer, format="JPEG")
                        jpeg_bytes = jpeg_buffer.getvalue()
                        output_filename_jpeg = info["filename"].rsplit(".", 1)[0] + "_with_frame_coords.jpg"
                        zip_file.writestr(output_filename_jpeg, jpeg_bytes)
                zip_buffer.seek(0)
                st.session_state["preprocessed_zip"] = zip_buffer.getvalue()
                st.download_button(
                    label="Télécharger les images prétraitées (ZIP)",
                    data=zip_buffer,
                    file_name="images_pretraitees.zip",
                    mime="application/zip"
                )
                st.success("Vos images ont été post-traitées, vous pouvez les utiliser.")
    else:
        st.info("Veuillez téléverser des images JPEG pour lancer le post-traitement.")

    st.markdown("---")
    st.header("Detections")
    tab_auto, tab_manuel = st.tabs(["Détection Automatique", "Détection Manuelle"])
    with tab_auto:
        if st.button("Lancer le traitement Automatique"):
            if "preprocessed_zip" in st.session_state:
                zip_bytes = st.session_state["preprocessed_zip"]
                auto_converted_files = []
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
                    for filename in zip_file.namelist():
                        if filename.endswith("_with_frame_coords.jpg"):
                            file_data = zip_file.read(filename)
                            file_obj = io.BytesIO(file_data)
                            file_obj.name = filename
                            auto_converted_files.append(file_obj)
                st.session_state["auto_converted_images"] = auto_converted_files
                st.success(f"{len(auto_converted_files)} images converties chargées.")
                if "images_info" in st.session_state:
                    images_info = st.session_state["images_info"]
                    auto_markers = []
                    for i, info in enumerate(images_info):
                        if len(images_info) >= 2:
                            if i == 0:
                                dx = images_info[1]["utm"][0] - images_info[0]["utm"][0]
                                dy = images_info[1]["utm"][1] - images_info[0]["utm"][1]
                            elif i == len(images_info) - 1:
                                dx = images_info[-1]["utm"][0] - images_info[-2]["utm"][0]
                                dy = images_info[-1]["utm"][1] - images_info[-2]["utm"][1]
                            else:
                                dx = images_info[i+1]["utm"][0] - images_info[i-1]["utm"][0]
                                dy = images_info[i+1]["utm"][1] - images_info[i-1]["utm"][1]
                            flight_angle_i = math.degrees(math.atan2(dx, dy))
                        else:
                            flight_angle_i = 0
                        rotation_angle_i = -flight_angle_i
                        scaling_factor = 1
                        new_width = info["img_width"]
                        new_height = info["img_height"]
                        effective_pixel_size = pixel_size
                        center_x, center_y = info["utm"]
                        T1 = Affine.translation(-new_width/2, -new_height/2)
                        T2 = Affine.scale(effective_pixel_size, -effective_pixel_size)
                        T3 = Affine.rotation(rotation_angle_i)
                        T4 = Affine.translation(center_x, center_y)
                        transform_affine = T4 * T3 * T2 * T1
                        marker_utm = transform_affine * (0.7 * new_width, new_height/2)
                        transformer = Transformer.from_crs(info["utm_crs"], "EPSG:4326", always_xy=True)
                        lon_conv, lat_conv = transformer.transform(marker_utm[0], marker_utm[1])
                        marker = {
                            "ID": f"{st.session_state.get('current_mission', 'auto')}-{i+1}",
                            "classe": "deformations ornierage",
                            "gravite": 1,
                            "coordonnees UTM": (round(marker_utm[0],2), round(marker_utm[1],2)),
                            "lat": lat_conv,
                            "long": lon_conv,
                            "routes": "Route inconnue",
                            "detection": "Automatique",
                            "mission": st.session_state.get("current_mission", "N/A"),
                            "couleur": class_color.get("deformations ornierage", "#FF0000"),
                            "radius": gravity_sizes.get(1, 5),
                            "date": st.session_state.missions.get(st.session_state.get("current_mission", "N/A"), {}).get("date", ""),
                            "appareil": st.session_state.missions.get(st.session_state.get("current_mission", "N/A"), {}).get("appareil_type", ""),
                            "nom_appareil": st.session_state.missions.get(st.session_state.get("current_mission", "N/A"), {}).get("nom_appareil", "")
                        }
                        auto_markers.append(marker)
                    st.session_state.markers_by_pair["auto"] = auto_markers
                else:
                    st.error("Aucune information d'images disponible pour le traitement automatique.")
                for i, file_obj in enumerate(auto_converted_files):
                    file_obj.seek(0)
                    img = Image.open(file_obj)
                    draw = ImageDraw.Draw(img)
                    x = 0.7 * img.width
                    y_img = 0.5 * img.height
                    r = 5
                    draw.ellipse((x-r, y_img-r, x+r, y_img+r), fill="red")
                    st.image(img, caption=file_obj.name)
            else:
                st.error("Aucun résultat de conversion prétraitée n'est disponible.")
    with tab_manuel:
        if st.button("Commencer le traitement manuel"):
            if "preprocessed_zip" in st.session_state:
                zip_bytes = st.session_state["preprocessed_zip"]
                manual_grand_files = []
                manual_petit_files = []
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_file:
                    for filename in zip_file.namelist():
                        if filename.endswith("_geotiff_x2.tif"):
                            file_data = zip_file.read(filename)
                            file_obj = io.BytesIO(file_data)
                            file_obj.name = filename
                            manual_grand_files.append(file_obj)
                        elif filename.endswith("_geotiff.tif"):
                            file_data = zip_file.read(filename)
                            file_obj = io.BytesIO(file_data)
                            file_obj.name = filename
                            manual_petit_files.append(file_obj)
                if manual_grand_files and manual_petit_files:
                    st.session_state["manual_grand_files"] = manual_grand_files
                    st.session_state["manual_petit_files"] = manual_petit_files
                    st.success("Fichiers chargés depuis conversion.")
                else:
                    st.error("Aucun résultat de conversion prétraitée n'est disponible pour l'une ou l'autre configuration.")
            else:
                st.error("Aucun résultat de conversion prétraitée n'est disponible.")
        if st.session_state.get("manual_grand_files") and st.session_state.get("manual_petit_files"):
            grand_list = []
            petit_list = []
            for file in st.session_state["manual_grand_files"]:
                file.seek(0)
                grand_list.append(get_reprojected_and_center(file, "grand"))
            for file in st.session_state["manual_petit_files"]:
                file.seek(0)
                petit_list.append(get_reprojected_and_center(file, "petit"))
            grand_list = sorted(grand_list, key=lambda d: d["center"])
            petit_list = sorted(petit_list, key=lambda d: d["center"])
            pair_count = len(grand_list)
            pairs = []
            for i in range(pair_count):
                pairs.append({"grand": grand_list[i], "petit": petit_list[i]})
            st.session_state.pairs = pairs
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
            center_lat_petit = (petit_bounds.bottom + petit_bounds.top) / 2
            center_lon_petit = (petit_bounds.left + petit_bounds.right) / 2
            utm_zone_petit = int((center_lon_petit + 180) / 6) + 1
            utm_crs_petit = f"EPSG:326{utm_zone_petit:02d}"
            st.subheader("Carte de dessin")
            m_grand = create_map(center_lat_grand, center_lon_grand, grand_bounds, display_path_grand,
                                 marker_data=None, hide_osm=True, draw_routes=False, add_draw_tool=True)
            result_grand = st_folium(m_grand, width=700, height=500, key="folium_map_grand")
            features = []
            all_drawings = result_grand.get("all_drawings")
            if all_drawings:
                if isinstance(all_drawings, dict) and "features" in all_drawings:
                    features = all_drawings.get("features", [])
                elif isinstance(all_drawings, list):
                    features = all_drawings
            current_mission = st.session_state.get("current_mission", "N/A")
            mission_details = st.session_state.missions.get(current_mission, {}) if current_mission != "N/A" else {}
            if current_mission not in st.session_state.mission_marker_counter:
                st.session_state.mission_marker_counter[current_mission] = 1
            existing_markers = st.session_state.markers_by_pair.get(current_index, [])
            updated_markers = []
            if features:
                st.markdown("Pour chaque marqueur dessiné, associez une classe et un niveau de gravité :")
                for i, feature in enumerate(features):
                    if feature.get("geometry", {}).get("type") == "Point":
                        coords = feature.get("geometry", {}).get("coordinates")
                        if coords and isinstance(coords, list) and len(coords) >= 2:
                            lon_pt, lat_pt = coords[0], coords[1]
                            percent_x = (lon_pt - grand_bounds.left) / (grand_bounds.right - grand_bounds.left)
                            percent_y = (lat_pt - grand_bounds.bottom) / (grand_bounds.top - grand_bounds.bottom)
                            new_lon = petit_bounds.left + percent_x * (petit_bounds.right - petit_bounds.left)
                            new_lat = petit_bounds.bottom + percent_y * (petit_bounds.top - petit_bounds.bottom)
                            utm_x_petit, utm_y_petit = rio_transform("EPSG:4326", utm_crs_petit, [new_lon], [new_lat])
                            utm_coords_petit = (round(utm_x_petit[0], 2), round(utm_y_petit[0], 2))
                        else:
                            new_lon = new_lat = None
                            utm_coords_petit = "Inconnues"
                        assigned_route = assign_route_to_marker(new_lat, new_lon, routes_ci) if new_lat and new_lon else "Route inconnue"
                        if i < len(existing_markers):
                            marker_id = existing_markers[i]["ID"]
                        else:
                            marker_id = f"{current_mission}-{st.session_state.mission_marker_counter[current_mission]}"
                            st.session_state.mission_marker_counter[current_mission] += 1
                        st.markdown(f"**ID {marker_id}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_class = st.selectbox("Classe", list(class_color.keys()), key=f"class_{current_index}_{marker_id}")
                        with col2:
                            selected_gravity = st.selectbox("Gravité", [1, 2, 3], key=f"gravity_{current_index}_{marker_id}")
                        updated_markers.append({
                            "ID": marker_id,
                            "classe": selected_class,
                            "gravite": selected_gravity,
                            "coordonnees UTM": utm_coords_petit,
                            "lat": new_lat,
                            "long": new_lon,
                            "routes": assigned_route,
                            "detection": "Manuelle",
                            "mission": current_mission,
                            "couleur": class_color.get(selected_class, "#000000"),
                            "radius": gravity_sizes.get(selected_gravity, 5),
                            "date": mission_details.get("date", ""),
                            "appareil": mission_details.get("appareil_type", ""),
                            "nom_appareil": mission_details.get("nom_appareil", "")
                        })
                st.session_state.markers_by_pair[current_index] = updated_markers
            else:
                st.write("Aucun marqueur n'a été détecté.")
        else:
            st.info("Aucun fichier TIFF converti n'est disponible pour lancer la détection manuelle.")

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
            m_petit = create_map(center_lat_petit, center_lon_petit, petit_bounds,
                                 display_path_petit if 'display_path_petit' in locals() else "",
                                 marker_data=global_markers, tiff_opacity=0, tiff_show=True, tiff_control=False, draw_routes=True,
                                 add_draw_tool=False)
            st_folium(m_petit, width=700, height=500, key="folium_map_petit")
        else:
            st.info("Impossible d'afficher la carte de suivi à cause d'un problème avec le TIFF PETIT.")
    else:
        all_lons = []
        all_lats = []
        for route in routes_ci:
            for lon, lat in route["coords"]:
                all_lons.append(lon)
                all_lats.append(lat)
        if all_lons and all_lats:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
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

    if st.button("Sauvegarder la mission"):
        current_mission = st.session_state.get("current_mission", None)
        if current_mission:
            mission_details = st.session_state.missions.get(current_mission, {})
            mission_details["distance(km)"] = st.session_state.get("trajet_distance_km", 0)
            if mission_details not in st.session_state.mission_history:
                st.session_state.mission_history.append(mission_details)
            st.success("Mission sauvegardée dans l'historique.")
        else:
            st.error("Aucune mission sélectionnée.")

    if st.button("Exporter les résultats de la mission en CSV"):
        current_mission = st.session_state.get("current_mission", None)
        if current_mission:
            mission_markers = []
            for markers in st.session_state.markers_by_pair.values():
                for marker in markers:
                    if marker.get("mission") == current_mission:
                        mission_markers.append(marker)
            if mission_markers:
                output = io.StringIO()
                writer = csv.writer(output, delimiter=';')
                writer.writerow(["ID", "Classe", "Gravité", "Coordonnées UTM", "Latitude", "Longitude", "Route", "Détection", "Mission", "Date", "Appareil", "Nom Appareil"])
                for marker in mission_markers:
                    writer.writerow([
                        marker.get("ID"),
                        marker.get("classe"),
                        marker.get("gravite"),
                        marker.get("coordonnees UTM"),
                        marker.get("lat"),
                        marker.get("long"),
                        marker.get("routes"),
                        marker.get("detection"),
                        marker.get("mission"),
                        marker.get("date", ""),
                        marker.get("appareil", ""),
                        marker.get("nom_appareil", "")
                    ])
                csv_data = output.getvalue().encode('utf-8')
                st.download_button(
                    label="Télécharger CSV de la mission",
                    data=csv_data,
                    file_name=f"mission_{current_mission}_resultats.csv",
                    mime="text/csv"
                )
            else:
                st.info("Aucun marqueur n'est associé à la mission courante.")
        else:
            st.info("Aucune mission sélectionnée.")

elif menu_option == "Rapport":
    st.title("Génération de Rapports")
    report_type = st.selectbox("Sélectionnez le type de rapport", 
                               ["Journalier", "Semaine", "Mensuel", "Annuel", "Général"])
    
    if report_type == "Journalier":
        selected_day = st.date_input("Sélectionnez le jour", value=date.today(), key="daily_date")
        start_date = selected_day
        end_date = selected_day
    elif report_type == "Semaine":
        selected_week = st.date_input("Sélectionnez une date de la semaine", value=date.today(), key="weekly_date")
        start_date = selected_week - timedelta(days=selected_week.weekday())
        end_date = start_date + timedelta(days=6)
    elif report_type == "Mensuel":
        selected_month = st.date_input("Sélectionnez un mois", value=date.today(), key="monthly_date")
        start_date = selected_month.replace(day=1)
        _, last_day = calendar.monthrange(selected_month.year, selected_month.month)
        end_date = selected_month.replace(day=last_day)
    elif report_type == "Annuel":
        selected_year = st.number_input("Sélectionnez l'année", value=date.today().year, step=1, key="yearly_year")
        start_date = date(int(selected_year), 1, 1)
        end_date = date(int(selected_year), 12, 31)
    else:
        start_date = missions_df['date'].min().date()
        end_date = missions_df['date'].max().date()
    
    st.sidebar.header("📝 Métadonnées pour le Rapport")
    titre = st.sidebar.text_input("Titre du rapport", "Rapport de Suivi")
    editor = st.sidebar.text_input("Éditeur", "Admin")
    metadata = {"titre": titre, "editor": editor}
    
    if st.button("Générer le Rapport PDF"):
        pdf_buffer = generate_report_pdf(report_type, missions_df, df_defects, metadata, start_date, end_date)
        st.success("✅ Rapport généré avec succès!")
        st.download_button("Télécharger le PDF", data=pdf_buffer, file_name=f"rapport_{report_type}.pdf", mime="application/pdf")
