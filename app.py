import streamlit as st
import pandas as pd
import json
import altair as alt
from collections import defaultdict

# Dictionnaire des couleurs par classe de défaut
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

# Valeurs numériques associées aux niveaux de gravité
gravity_sizes = {1: 5, 2: 7, 3: 9}

st.set_page_config(page_title="Dashboard Missions & Défauts", layout="wide")
st.title("Dashboard Ultra Moderne - Missions et Défauts")
st.markdown("### Téléversez votre fichier JSON contenant les données")

# Zone de téléversement dans la sidebar
uploaded_file = st.sidebar.file_uploader("Téléverser le fichier JSON", type=["json"])

if uploaded_file is not None:
    # Charger les données JSON
    data = json.load(uploaded_file)
    
    # Nombre de missions
    num_missions = len(data)
    
    # Extraction de tous les défauts dans une liste
    defects = []
    for mission in data:
        for defect in mission.get("Données Défauts", []):
            defects.append(defect)
    
    # Création d'un DataFrame pour les défauts
    df_defects = pd.DataFrame(defects)
    total_defects = len(df_defects)
    
    # Affichage des métriques principales
    col1, col2 = st.columns(2)
    col1.metric("Nombre de Missions", num_missions)
    col2.metric("Nombre de Défauts", total_defects)
    
    # Affichage des routes concernées
    st.markdown("### Routes concernées")
    routes = df_defects['routes'].dropna()
    unique_routes = routes.unique()
    st.write(unique_routes)
    
    # Routes ayant le plus de défauts
    st.markdown("### Routes avec le plus de Défauts")
    route_defect_counts = routes.value_counts().reset_index()
    route_defect_counts.columns = ["Route", "Nombre de Défauts"]
    st.dataframe(route_defect_counts)
    
    # Calcul du score de sévérité par défaut selon gravity_sizes
    df_defects['severite'] = df_defects['gravite'].map(gravity_sizes)
    route_severity = df_defects.groupby('routes')['severite'].sum().reset_index()
    route_severity = route_severity.sort_values(by='severite', ascending=False)
    st.markdown("### Routes avec le plus de Défauts Graves (Score de Sévérité)")
    st.dataframe(route_severity)
    
    # Sélecteur interactif pour choisir un type de défaut
    defect_types = df_defects['classe'].unique()
    selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
    filtered = df_defects[df_defects['classe'] == selected_defect]
    route_count_selected = filtered['routes'].value_counts().reset_index()
    route_count_selected.columns = ["Route", "Nombre de Défauts"]
    st.markdown(f"### Route avec le plus grand nombre de défauts pour le type : **{selected_defect}**")
    if not route_count_selected.empty:
        top_route = route_count_selected.iloc[0]
        st.write(top_route)
    else:
        st.write("Aucune donnée pour ce type de défaut.")
    
    # Défauts par catégories (classes)
    st.markdown("### Défauts par Catégories")
    defect_category_counts = df_defects['classe'].value_counts().reset_index()
    defect_category_counts.columns = ["Catégorie", "Nombre de Défauts"]
    st.dataframe(defect_category_counts)
    
    # Graphique en barres avec couleurs personnalisées par catégorie
    chart = alt.Chart(defect_category_counts).mark_bar().encode(
        x=alt.X("Catégorie:N", sort='-y', title="Catégorie de Défaut"),
        y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
        color=alt.Color("Catégorie:N",
                        scale=alt.Scale(domain=list(class_color.keys()),
                                        range=list(class_color.values())))
    ).properties(
        width=700,
        height=400,
        title="Répartition des Défauts par Catégorie"
    )
    st.altair_chart(chart, use_container_width=True)
    
else:
    st.info("Veuillez téléverser un fichier JSON contenant les données pour afficher le dashboard.")
