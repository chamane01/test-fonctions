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
st.markdown("Visualisez vos données avec des graphiques interactifs et un design moderne.")

# Téléversement du fichier JSON via la sidebar
uploaded_file = st.sidebar.file_uploader("Téléverser un fichier JSON", type=["json"])

if uploaded_file is not None:
    data = json.load(uploaded_file)
    
    # DataFrame des missions
    missions_df = pd.DataFrame(data)
    missions_df['date'] = pd.to_datetime(missions_df['date'])
    
    if "distance(km)" not in missions_df.columns:
        st.error("Le fichier ne contient pas la colonne 'distance(km)'.")
    else:
        num_missions = len(missions_df)
        total_distance = missions_df["distance(km)"].sum()
        avg_distance = missions_df["distance(km)"].mean()
        
        # Extraction des défauts de toutes les missions
        defects = []
        for mission in data:
            for defect in mission.get("Données Défauts", []):
                defects.append(defect)
        df_defects = pd.DataFrame(defects)
        total_defects = len(df_defects)
        
        if 'date' in df_defects.columns:
            df_defects['date'] = pd.to_datetime(df_defects['date'])
        
        gravity_sizes = {1: 5, 2: 7, 3: 9}
        df_defects['severite'] = df_defects['gravite'].map(gravity_sizes)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de Missions", num_missions)
        col2.metric("Nombre de Défauts", total_defects)
        col3.metric("Distance Totale (km)", f"{total_distance:.1f}")
        col4.metric("Distance Moyenne (km)", f"{avg_distance:.1f}")
        
        st.markdown("---")
        
        # Bouton pour afficher tous les éléments (placé ici pour impacter les diagrammes verticaux)
        show_all = st.checkbox("Afficher tous les éléments", value=False)
        
        # Limitation du nombre d'éléments affichés (7 par défaut)
        limit = None if show_all else 7

        # 📌 **Diagramme vertical : Nombre de Défauts par Route**
        route_defect_counts = df_defects['routes'].value_counts().reset_index()
        route_defect_counts.columns = ["Route", "Nombre de Défauts"]
        display_routes = route_defect_counts.head(limit)

        chart_routes = alt.Chart(display_routes).mark_bar().encode(
            x=alt.X("Route:N", sort='-y', title="Route", 
                    axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            tooltip=["Route:N", "Nombre de Défauts:Q"],
            color=alt.Color("Route:N", scale=alt.Scale(scheme='tableau10'))
        ).properties(
            width=900,
            height=500,
            title="Nombre de Défauts par Route (Top 7 par défaut)"
        )
        st.altair_chart(chart_routes, use_container_width=True)

        # 📌 **Diagramme vertical : Routes avec le Score de Sévérité Total**
        route_severity = df_defects.groupby('routes')['severite'].sum().reset_index().sort_values(by='severite', ascending=False)
        display_severity = route_severity.head(limit)

        chart_severity = alt.Chart(display_severity).mark_bar().encode(
            x=alt.X("routes:N", sort='-y', title="Route",
                    axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("severite:Q", title="Score de Sévérité Total"),
            tooltip=["routes:N", "severite:Q"],
            color=alt.Color("routes:N", scale=alt.Scale(scheme='tableau20'))
        ).properties(
            width=900,
            height=500,
            title="Routes avec le Score de Sévérité le Plus Élevé (Top 7 par défaut)"
        )
        st.altair_chart(chart_severity, use_container_width=True)

        # 📌 **Analyse interactive par Type de Défaut**
        st.markdown("### Analyse par Type de Défaut")
        defect_types = df_defects['classe'].unique()
        selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
        filtered_defects = df_defects[df_defects['classe'] == selected_defect]
        
        if not filtered_defects.empty:
            route_count_selected = filtered_defects['routes'].value_counts().reset_index()
            route_count_selected.columns = ["Route", "Nombre de Défauts"]
            display_selected = route_count_selected.head(limit)

            chart_defect_type = alt.Chart(display_selected).mark_bar().encode(
                x=alt.X("Route:N", sort='-y', title="Route",
                        axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
                y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
                tooltip=["Route:N", "Nombre de Défauts:Q"],
                color=alt.Color("Route:N", scale=alt.Scale(scheme='category20b'))
            ).properties(
                width=900,
                height=500,
                title=f"Répartition des Défauts pour le Type : {selected_defect} (Top 7 par défaut)"
            )
            st.altair_chart(chart_defect_type, use_container_width=True)
        else:
            st.write("Aucune donnée disponible pour ce type de défaut.")

else:
    st.info("Veuillez téléverser un fichier JSON contenant vos données pour afficher le dashboard.")
