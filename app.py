import streamlit as st
import pandas as pd
import json
import altair as alt
import plotly.express as px
import pydeck as pdk
from datetime import datetime

# Configuration de la page et CSS pour centrer le contenu et limiter la largeur
st.set_page_config(page_title="Tableau de Suivie des Routes Nationales", layout="wide")
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

# Affichage du logo à partir d'un chemin direct (images (5).png)
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("images (5).png", width=200)
st.markdown("</div>", unsafe_allow_html=True)

# Début du conteneur principal (contenu centré et à largeur limitée)
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.title("tableau de suivie des routes nationales")
st.markdown("Visualisez vos données avec des graphiques interactifs et un design moderne.")

# Chargement direct du fichier JSON depuis le repository
with open("jeu_donnees_missions (1).json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
# DataFrame des missions
missions_df = pd.DataFrame(data)
missions_df['date'] = pd.to_datetime(missions_df['date'])

if "distance(km)" not in missions_df.columns:
    st.error("Le fichier ne contient pas la colonne 'distance(km)'.")
else:
    # Calcul des métriques principales
    num_missions = len(missions_df)
    total_distance = missions_df["distance(km)"].sum()
    avg_distance = missions_df["distance(km)"].mean()
    
    # Extraction de tous les défauts de chaque mission
    defects = []
    for mission in data:
        for defect in mission.get("Données Défauts", []):
            defects.append(defect)
    df_defects = pd.DataFrame(defects)
    total_defects = len(df_defects)
    
    if 'date' in df_defects.columns:
        df_defects['date'] = pd.to_datetime(df_defects['date'])
    
    # Mapping des niveaux de gravité pour la taille des marqueurs
    gravity_sizes = {1: 5, 2: 7, 3: 9}
    df_defects['severite'] = df_defects['gravite'].map(gravity_sizes)
    
    # Affichage des métriques principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de Missions", num_missions)
    col2.metric("Nombre de Défauts", total_defects)
    col3.metric("Distance Totale (km)", f"{total_distance:.1f}")
    col4.metric("Distance Moyenne (km)", f"{avg_distance:.1f}")
    
    st.markdown("---")
    
    # Graphique 1 : Évolution des Missions dans le Temps
    missions_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M")).size().reset_index(name="Missions")
    missions_over_time['date'] = missions_over_time['date'].dt.to_timestamp()
    chart_missions = alt.Chart(missions_over_time).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('Missions:Q', title='Nombre de Missions'),
        tooltip=['date:T', 'Missions:Q']
    ).properties(width=700, height=300, title="Évolution des Missions dans le Temps")
    st.altair_chart(chart_missions, use_container_width=True)
    
    # Graphique 2 : Évolution des Défauts dans le Temps
    defects_over_time = df_defects.groupby(df_defects['date'].dt.to_period("M")).size().reset_index(name="Défauts")
    defects_over_time['date'] = defects_over_time['date'].dt.to_timestamp()
    chart_defects_time = alt.Chart(defects_over_time).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('Défauts:Q', title='Nombre de Défauts'),
        tooltip=['date:T', 'Défauts:Q']
    ).properties(width=700, height=300, title="Évolution des Défauts dans le Temps")
    st.altair_chart(chart_defects_time, use_container_width=True)
    
    # Graphique 3 : Évolution des Km par Mission dans le Temps
    distance_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M"))["distance(km)"].sum().reset_index(name="Distance Totale")
    distance_over_time['date'] = distance_over_time['date'].dt.to_timestamp()
    chart_distance_time = alt.Chart(distance_over_time).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('Distance Totale:Q', title='Km Totaux', scale=alt.Scale(domain=[0, distance_over_time["Distance Totale"].max()*1.2])),
        tooltip=['date:T', 'Distance Totale:Q']
    ).properties(width=700, height=300, title="Évolution des Km par Mission dans le Temps")
    st.altair_chart(chart_distance_time, use_container_width=True)
    
    st.markdown("---")
    
    # Graphique 4 : Diagramme circulaire pour la Répartition Globale des Défauts par Catégorie
    defect_category_counts = df_defects['classe'].value_counts().reset_index()
    defect_category_counts.columns = ["Catégorie", "Nombre de Défauts"]
    fig_pie = px.pie(defect_category_counts, values='Nombre de Défauts', names='Catégorie',
                     title="Répartition Globale des Défauts par Catégorie",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Carte des Défauts : couleurs rouges variant selon la gravité
    def get_red_color(gravite):
        if gravite == 1:
            return [255, 200, 200]  # Rouge pâle
        elif gravite == 2:
            return [255, 100, 100]  # Rouge moyen
        elif gravite == 3:
            return [255, 0, 0]      # Rouge vif
        else:
            return [255, 0, 0]
            
    df_defects['marker_color'] = df_defects['gravite'].apply(get_red_color)
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_defects,
        get_position='[long, lat]',
        get_color="marker_color",
        get_radius="radius * 50",
        pickable=True,
    )
    
    view_state = pdk.ViewState(
        latitude=df_defects['lat'].mean(),
        longitude=df_defects['long'].mean(),
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
    
    # Option d'affichage complet pour les graphiques verticaux
    show_all = st.checkbox("Afficher tous les éléments", value=False)
    limit = None if show_all else 7
    
    # Graphique 5 : Nombre de Défauts par Route (diagramme vertical)
    route_defect_counts = df_defects['routes'].value_counts().reset_index()
    route_defect_counts.columns = ["Route", "Nombre de Défauts"]
    display_routes = route_defect_counts if show_all else route_defect_counts.head(limit)
    chart_routes = alt.Chart(display_routes).mark_bar().encode(
        x=alt.X("Route:N", sort='-y', title="Route",
                axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
        y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
        tooltip=["Route:N", "Nombre de Défauts:Q"],
        color=alt.Color("Route:N", scale=alt.Scale(scheme='tableau10'))
    ).properties(width=900, height=500, title="Nombre de Défauts par Route (Top 7 par défaut)")
    st.altair_chart(chart_routes, use_container_width=True)
    
    # Graphique 6 : Routes avec le Score de Sévérité Total (diagramme vertical)
    route_severity = df_defects.groupby('routes')['severite'].sum().reset_index().sort_values(by='severite', ascending=False)
    display_severity = route_severity if show_all else route_severity.head(limit)
    chart_severity = alt.Chart(display_severity).mark_bar().encode(
        x=alt.X("routes:N", sort='-y', title="Route",
                axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
        y=alt.Y("severite:Q", title="Score de Sévérité Total"),
        tooltip=["routes:N", "severite:Q"],
        color=alt.Color("routes:N", scale=alt.Scale(scheme='tableau20'))
    ).properties(width=900, height=500, title="Routes avec le Score de Sévérité le Plus Élevé (Top 7 par défaut)")
    st.altair_chart(chart_severity, use_container_width=True)
    
    # Graphique 7 : Analyse interactive par Type de Défaut (diagramme vertical)
    st.markdown("### Analyse par Type de Défaut")
    defect_types = df_defects['classe'].unique()
    selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
    filtered_defects = df_defects[df_defects['classe'] == selected_defect]
    if not filtered_defects.empty:
        route_count_selected = filtered_defects['routes'].value_counts().reset_index()
        route_count_selected.columns = ["Route", "Nombre de Défauts"]
        display_selected = route_count_selected if show_all else route_count_selected.head(limit)
        chart_defect_type = alt.Chart(display_selected).mark_bar().encode(
            x=alt.X("Route:N", sort='-y', title="Route",
                    axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            tooltip=["Route:N", "Nombre de Défauts:Q"],
            color=alt.Color("Route:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=900, height=500, title=f"Répartition des Défauts pour le Type : {selected_defect} (Top 7 par défaut)")
        st.altair_chart(chart_defect_type, use_container_width=True)
    else:
        st.write("Aucune donnée disponible pour ce type de défaut.")
    
    st.markdown("---")
    
    # Nouvelle section : Analyse par Route
    st.markdown("### Analyse par Route")
    selected_route = st.selectbox("Sélectionnez une route", sorted(df_defects['routes'].unique()))
    inventory = df_defects[df_defects['routes'] == selected_route]['classe'].value_counts().reset_index()
    inventory.columns = ["Dégradation", "Nombre de Défauts"]
    chart_route_inventory = alt.Chart(inventory).mark_bar().encode(
        x=alt.X("Dégradation:N", sort='-y', title="Dégradation",
                axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
        y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
        tooltip=["Dégradation:N", "Nombre de Défauts:Q"],
        color=alt.Color("Dégradation:N", scale=alt.Scale(scheme='category20b'))
    ).properties(width=900, height=500, title=f"Inventaire des Dégradations pour la Route : {selected_route}")
    st.altair_chart(chart_route_inventory, use_container_width=True)

# Fermeture du conteneur principal
st.markdown("</div>", unsafe_allow_html=True)
