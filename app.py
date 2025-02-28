import streamlit as st
import pandas as pd
import json
import altair as alt
import plotly.express as px
from collections import defaultdict

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
st.markdown("Un tableau de bord minimaliste, coloré et interactif pour visualiser vos données de missions et défauts.")

# Téléversement du fichier JSON via la sidebar
uploaded_file = st.sidebar.file_uploader("Téléverser un fichier JSON", type=["json"])

if uploaded_file is not None:
    # Chargement des données
    data = json.load(uploaded_file)
    
    # Nombre de missions
    num_missions = len(data)
    
    # Extraction des défauts et récupération des dates de mission
    defects = []
    missions_dates = []
    for mission in data:
        missions_dates.append(mission.get("date"))
        for defect in mission.get("Données Défauts", []):
            defects.append(defect)
    
    df_defects = pd.DataFrame(defects)
    total_defects = len(df_defects)
    
    # Conversion de la date si présente
    if 'date' in df_defects.columns:
        df_defects['date'] = pd.to_datetime(df_defects['date'])
    
    # Mapping des niveaux de gravité (pour le score de sévérité)
    gravity_sizes = {1: 5, 2: 7, 3: 9}
    df_defects['severite'] = df_defects['gravite'].map(gravity_sizes)
    
    # Affichage des métriques principales en 3 colonnes
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de Missions", num_missions)
    col2.metric("Nombre de Défauts", total_defects)
    avg_severity = df_defects['severite'].mean() if total_defects > 0 else 0
    col3.metric("Score de Sévérité Moyen", f"{avg_severity:.1f}")
    
    st.markdown("---")
    
    # Graphique 1 : Évolution des missions dans le temps
    missions_df = pd.DataFrame(data)
    missions_df['date'] = pd.to_datetime(missions_df['date'])
    missions_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M")).size().reset_index(name="count")
    missions_over_time['date'] = missions_over_time['date'].dt.to_timestamp()
    chart_missions = alt.Chart(missions_over_time).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('count:Q', title='Nombre de Missions'),
        tooltip=['date:T', 'count:Q']
    ).properties(
        width=700,
        height=300,
        title="Évolution des Missions dans le Temps"
    )
    st.altair_chart(chart_missions, use_container_width=True)
    
    st.markdown("---")
    
    # Graphique 2 : Défauts par Route (Bar Chart)
    route_defect_counts = df_defects['routes'].value_counts().reset_index()
    route_defect_counts.columns = ["Route", "Nombre de Défauts"]
    chart_routes = alt.Chart(route_defect_counts).mark_bar().encode(
        x=alt.X("Route:N", sort='-y', title="Route"),
        y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
        color=alt.Color("Route:N", scale=alt.Scale(scheme='tableau10'))
    ).properties(
        width=700,
        height=300,
        title="Nombre de Défauts par Route"
    )
    st.altair_chart(chart_routes, use_container_width=True)
    
    st.markdown("---")
    
    # Graphique 3 : Routes avec le Score de Sévérité (Bar Chart)
    route_severity = df_defects.groupby('routes')['severite'].sum().reset_index().sort_values(by='severite', ascending=False)
    chart_severity = alt.Chart(route_severity).mark_bar().encode(
        x=alt.X("routes:N", sort='-y', title="Route"),
        y=alt.Y("severite:Q", title="Score de Sévérité Total"),
        color=alt.Color("routes:N", scale=alt.Scale(scheme='tableau20'))
    ).properties(
        width=700,
        height=300,
        title="Routes avec le Score de Sévérité le Plus Élevé"
    )
    st.altair_chart(chart_severity, use_container_width=True)
    
    st.markdown("---")
    
    # Graphique 4 : Répartition des Défauts par Catégorie (Camembert avec Plotly)
    defect_category_counts = df_defects['classe'].value_counts().reset_index()
    defect_category_counts.columns = ["Catégorie", "Nombre de Défauts"]
    fig_pie = px.pie(defect_category_counts, values='Nombre de Défauts', names='Catégorie',
                     title="Répartition des Défauts par Catégorie", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Graphique 5 : Carte des Défauts
    st.markdown("### Carte des Défauts")
    if 'lat' in df_defects.columns and 'long' in df_defects.columns:
        df_map = df_defects[['lat', 'long']].dropna().rename(columns={'lat': 'latitude', 'long': 'longitude'})
        st.map(df_map)
    
    st.markdown("---")
    
    # Analyse interactive par type de défaut
    st.markdown("### Analyse par Type de Défaut")
    defect_types = df_defects['classe'].unique()
    selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
    filtered_defects = df_defects[df_defects['classe'] == selected_defect]
    if not filtered_defects.empty:
        route_count_selected = filtered_defects['routes'].value_counts().reset_index()
        route_count_selected.columns = ["Route", "Nombre de Défauts"]
        chart_defect_type = alt.Chart(route_count_selected).mark_bar().encode(
            x=alt.X("Route:N", sort='-y', title="Route"),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            color=alt.Color("Route:N", scale=alt.Scale(scheme='category20b'))
        ).properties(
            width=700,
            height=300,
            title=f"Répartition des Défauts pour le Type : {selected_defect}"
        )
        st.altair_chart(chart_defect_type, use_container_width=True)
    else:
        st.write("Aucune donnée disponible pour ce type de défaut.")
    
else:
    st.info("Veuillez téléverser un fichier JSON contenant vos données pour afficher le dashboard.")
