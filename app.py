import streamlit as st
import pandas as pd
import json
import altair as alt
import plotly.express as px
from datetime import datetime

# Configuration de la page et CSS minimaliste et coloré
st.set_page_config(page_title="Dashboard Ultra Moderne", layout="wide")
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
    </style>
    """, unsafe_allow_html=True
)

st.title("Dashboard Ultra Moderne - Missions et Défauts")
st.markdown("Visualisez vos données avec des graphiques interactifs, des barres détaillées et des évolutions dans le temps.")

# Téléversement du fichier JSON via la sidebar
uploaded_file = st.sidebar.file_uploader("Téléverser un fichier JSON", type=["json"])

if uploaded_file is not None:
    # Chargement des données
    data = json.load(uploaded_file)
    
    # Création d'un DataFrame pour les missions
    missions_df = pd.DataFrame(data)
    missions_df['date'] = pd.to_datetime(missions_df['date'])
    
    # Vérification de la nouvelle colonne "distance(km)"
    if "distance(km)" not in missions_df.columns:
        st.error("Le fichier ne contient pas la colonne 'distance(km)'.")
    else:
        # Extraction et calcul des métriques principales
        num_missions = len(missions_df)
        
        # Distance totale et moyenne
        total_distance = missions_df["distance(km)"].sum()
        avg_distance = missions_df["distance(km)"].mean()
        
        # Extraction de tous les défauts dans une liste
        defects = []
        for mission in data:
            for defect in mission.get("Données Défauts", []):
                defects.append(defect)
        df_defects = pd.DataFrame(defects)
        total_defects = len(df_defects)
        
        # Conversion de la date pour les défauts
        if 'date' in df_defects.columns:
            df_defects['date'] = pd.to_datetime(df_defects['date'])
        
        # Mapping des niveaux de gravité (pour le score de sévérité)
        gravity_sizes = {1: 5, 2: 7, 3: 9}
        df_defects['severite'] = df_defects['gravite'].map(gravity_sizes)
        
        # Affichage des métriques principales en 4 colonnes
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de Missions", num_missions)
        col2.metric("Nombre de Défauts", total_defects)
        col3.metric("Distance Totale (km)", f"{total_distance:.1f}")
        col4.metric("Distance Moyenne (km)", f"{avg_distance:.1f}")
        
        st.markdown("---")
        
        # Option pour afficher tous les éléments dans les graphiques (sinon, on limite aux 7 premiers)
        show_all = st.checkbox("Afficher tous les éléments", value=False)
        
        #########################################
        # Graphique : Évolution des Missions
        missions_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M")).size().reset_index(name="Missions")
        missions_over_time['date'] = missions_over_time['date'].dt.to_timestamp()
        chart_missions = alt.Chart(missions_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Missions:Q', title='Nombre de Missions'),
            tooltip=['date:T', 'Missions:Q']
        ).properties(
            width=700,
            height=300,
            title="Évolution des Missions dans le Temps"
        )
        st.altair_chart(chart_missions, use_container_width=True)
        
        #########################################
        # Graphique : Évolution des Défauts dans le Temps
        defects_over_time = df_defects.groupby(df_defects['date'].dt.to_period("M")).size().reset_index(name="Défauts")
        defects_over_time['date'] = defects_over_time['date'].dt.to_timestamp()
        chart_defects_time = alt.Chart(defects_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Défauts:Q', title='Nombre de Défauts'),
            tooltip=['date:T', 'Défauts:Q']
        ).properties(
            width=700,
            height=300,
            title="Évolution des Défauts dans le Temps"
        )
        st.altair_chart(chart_defects_time, use_container_width=True)
        
        #########################################
        # Graphique : Évolution des Km dans le Temps
        distance_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M"))["distance(km)"].sum().reset_index(name="Distance Totale")
        distance_over_time['date'] = distance_over_time['date'].dt.to_timestamp()
        chart_distance_time = alt.Chart(distance_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Distance Totale:Q', title='Km Totaux', scale=alt.Scale(domain=[0, distance_over_time["Distance Totale"].max()*1.2])),
            tooltip=['date:T', 'Distance Totale:Q']
        ).properties(
            width=700,
            height=300,
            title="Évolution des Km par Mission dans le Temps"
        )
        st.altair_chart(chart_distance_time, use_container_width=True)
        
        st.markdown("---")
        
        #########################################
        # Graphique : Nombre de Défauts par Route
        route_defect_counts = df_defects['routes'].value_counts().reset_index()
        route_defect_counts.columns = ["Route", "Nombre de Défauts"]
        # Limiter aux 7 premiers si non affichage complet
        display_routes = route_defect_counts if show_all else route_defect_counts.head(7)
        max_count = display_routes["Nombre de Défauts"].max()
        chart_routes = alt.Chart(display_routes).mark_bar().encode(
            x=alt.X("Route:N", sort='-y', title="Route", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts", scale=alt.Scale(domain=[0, max_count*1.2])),
            tooltip=["Route:N", "Nombre de Défauts:Q"],
            color=alt.Color("Route:N", scale=alt.Scale(scheme='tableau10'))
        ).properties(
            width=700,
            height=300,
            title="Nombre de Défauts par Route (Top 7 par défaut)"
        )
        st.altair_chart(chart_routes, use_container_width=True)
        
        #########################################
        # Graphique : Routes avec le Score de Sévérité Total
        route_severity = df_defects.groupby('routes')['severite'].sum().reset_index().sort_values(by='severite', ascending=False)
        display_severity = route_severity if show_all else route_severity.head(7)
        max_severity = display_severity["severite"].max()
        chart_severity = alt.Chart(display_severity).mark_bar().encode(
            x=alt.X("routes:N", sort='-y', title="Route", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("severite:Q", title="Score de Sévérité Total", scale=alt.Scale(domain=[0, max_severity*1.2])),
            tooltip=["routes:N", "severite:Q"],
            color=alt.Color("routes:N", scale=alt.Scale(scheme='tableau20'))
        ).properties(
            width=700,
            height=300,
            title="Routes avec le Score de Sévérité le Plus Élevé (Top 7 par défaut)"
        )
        st.altair_chart(chart_severity, use_container_width=True)
        
        #########################################
        # Graphique : Répartition des Défauts par Catégorie (Classes)
        defect_category_counts = df_defects['classe'].value_counts().reset_index()
        defect_category_counts.columns = ["Catégorie", "Nombre de Défauts"]
        display_categories = defect_category_counts if show_all else defect_category_counts.head(7)
        max_cat = display_categories["Nombre de Défauts"].max()
        chart_categories = alt.Chart(display_categories).mark_bar().encode(
            x=alt.X("Catégorie:N", sort='-y', title="Catégorie", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts", scale=alt.Scale(domain=[0, max_cat*1.2])),
            tooltip=["Catégorie:N", "Nombre de Défauts:Q"],
            color=alt.Color("Catégorie:N", scale=alt.Scale(domain=list(display_categories["Catégorie"]), range=px.colors.qualitative.Set3))
        ).properties(
            width=700,
            height=300,
            title="Répartition des Défauts par Catégorie (Top 7 par défaut)"
        )
        st.altair_chart(chart_categories, use_container_width=True)
        
        st.markdown("---")
        
        #########################################
        # Analyse interactive par Type de Défaut
        st.markdown("### Analyse par Type de Défaut")
        defect_types = df_defects['classe'].unique()
        selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
        filtered_defects = df_defects[df_defects['classe'] == selected_defect]
        if not filtered_defects.empty:
            route_count_selected = filtered_defects['routes'].value_counts().reset_index()
            route_count_selected.columns = ["Route", "Nombre de Défauts"]
            # Limiter à 7 éléments par défaut
            display_selected = route_count_selected if show_all else route_count_selected.head(7)
            max_sel = display_selected["Nombre de Défauts"].max()
            chart_defect_type = alt.Chart(display_selected).mark_bar().encode(
                x=alt.X("Route:N", sort='-y', title="Route", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts", scale=alt.Scale(domain=[0, max_sel*1.2])),
                tooltip=["Route:N", "Nombre de Défauts:Q"],
                color=alt.Color("Route:N", scale=alt.Scale(scheme='category20b'))
            ).properties(
                width=700,
                height=300,
                title=f"Répartition des Défauts pour le Type : {selected_defect} (Top 7 par défaut)"
            )
            st.altair_chart(chart_defect_type, use_container_width=True)
        else:
            st.write("Aucune donnée disponible pour ce type de défaut.")
            
else:
    st.info("Veuillez téléverser un fichier JSON contenant vos données pour afficher le dashboard.")
