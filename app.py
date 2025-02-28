import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# Connexion à la base de données SQLite
conn = sqlite3.connect("base_donnees_missions.db")
# Lecture des tables defects et missions
defects_df = pd.read_sql("SELECT * FROM defects", conn)
missions_df = pd.read_sql("SELECT * FROM missions", conn)
conn.close()

# Conversion de la colonne date en type datetime
defects_df['date'] = pd.to_datetime(defects_df['date'])

# Barre latérale – Filtres
st.sidebar.header("Filtres")
# Filtre sur les routes disponibles
routes_uniques = defects_df['routes'].unique().tolist()
routes_selectionnees = st.sidebar.multiselect("Sélectionnez les routes :", 
                                               options=routes_uniques, 
                                               default=routes_uniques)
# Filtre sur la plage de dates
min_date = defects_df['date'].min().date()
max_date = defects_df['date'].max().date()
date_range = st.sidebar.date_input("Plage de dates :", 
                                   value=(min_date, max_date), 
                                   min_value=min_date, 
                                   max_value=max_date)

# Filtrer les données en fonction des sélections
df_filtre = defects_df[
    (defects_df['routes'].isin(routes_selectionnees)) &
    (defects_df['date'] >= pd.to_datetime(date_range[0])) &
    (defects_df['date'] <= pd.to_datetime(date_range[1]))
]

# Titre du tableau de bord
st.title("Tableau de Bord des Défauts des Missions")
st.markdown("### Vue d'ensemble globale")

# Indicateurs clés
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nombre de missions", missions_df.shape[0])
with col2:
    st.metric("Nombre total de défauts", defects_df.shape[0])
with col3:
    st.metric("Défauts filtrés", df_filtre.shape[0])

# Graphique – Défauts par route
st.markdown("#### Défauts par Route")
defauts_route = df_filtre.groupby("routes").size().reset_index(name="count")
fig_routes = px.bar(defauts_route, x="routes", y="count", color="routes",
                    title="Répartition des défauts par route", 
                    labels={"count": "Nombre de défauts"})
st.plotly_chart(fig_routes, use_container_width=True)

# Graphique – Évolution dans le temps
st.markdown("#### Évolution des Défauts dans le Temps")
defauts_temps = df_filtre.groupby(df_filtre['date'].dt.date).size().reset_index(name="count")
fig_temps = px.line(defauts_temps, x="date", y="count", markers=True,
                    title="Nombre de défauts au fil du temps",
                    labels={"date": "Date", "count": "Nombre de défauts"})
st.plotly_chart(fig_temps, use_container_width=True)

# Carte interactive – Localisation des défauts
st.markdown("#### Carte des Défauts")
if not df_filtre.empty:
    # La fonction st.map attend des colonnes nommées 'lat' et 'lon'
    map_data = df_filtre[['lat', 'long']].rename(columns={"long": "lon"})
    st.map(map_data)
else:
    st.info("Aucune donnée à afficher sur la carte pour les filtres sélectionnés.")

# Graphique – Répartition par Gravité
st.markdown("#### Répartition de la Gravité des Défauts")
gravite_dist = df_filtre.groupby("gravite").size().reset_index(name="count")
fig_gravite = px.pie(gravite_dist, names="gravite", values="count", 
                     title="Distribution de la gravité",
                     color_discrete_sequence=px.colors.qualitative.Set3)
st.plotly_chart(fig_gravite, use_container_width=True)

# Affichage d'un aperçu tabulaire des données filtrées
st.markdown("### Aperçu des données filtrées")
st.dataframe(df_filtre.reset_index(drop=True))
