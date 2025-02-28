import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.title("Dashboard d'Analyse des Données JSON")

# Chargement du fichier JSON
uploaded_file = st.file_uploader("Charger votre fichier JSON", type=["json"])

if uploaded_file is not None:
    # Lecture du JSON et conversion en DataFrame
    data = json.load(uploaded_file)
    df = pd.DataFrame(data)
    
    # Conversion de la colonne date en datetime si elle existe
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    
    st.sidebar.header("Aperçu des données")
    st.sidebar.write(df.head())

    # 1. Histogramme des 5 routes ayant le plus de défauts
    st.header("1. Top 5 Routes avec le Plus de Défauts")
    if 'routes' in df.columns:
        route_counts = df['routes'].value_counts().head(5)
        fig1 = px.bar(
            x=route_counts.index,
            y=route_counts.values,
            labels={'x': 'Route', 'y': 'Nombre de Défauts'},
            title="Les 5 routes avec le plus de défauts"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.error("La colonne 'routes' est introuvable dans les données.")

    # 2. Histogramme de la gravité
    st.header("2. Histogramme de la Gravité")
    if 'gravite' in df.columns:
        fig2 = px.histogram(
            df,
            x='gravite',
            nbins=10,
            labels={'gravite': 'Gravité'},
            title="Distribution de la Gravité"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("La colonne 'gravite' est introuvable dans les données.")

    # 3. Défauts par catégories
    st.header("3. Défauts par Catégories")
    if 'classe' in df.columns:
        category_counts = df['classe'].value_counts()
        fig3 = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={'x': 'Catégorie', 'y': 'Nombre de Défauts'},
            title="Nombre de Défauts par Catégorie"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.error("La colonne 'classe' est introuvable dans les données.")

    # 4. Courbes d'évolution des détections par jours, mois, et année
    st.header("4. Évolution des Détections")
    if 'date' in df.columns and not df['date'].isnull().all():
        # Évolution quotidienne
        df_daily = df.groupby('date').size().reset_index(name='detections')
        fig_daily = px.line(
            df_daily,
            x='date',
            y='detections',
            labels={'date': 'Date', 'detections': "Nombre de Détections"},
            title="Évolution Quotidienne"
        )
        st.subheader("Par Jour")
        st.plotly_chart(fig_daily, use_container_width=True)

        # Évolution mensuelle
        df_monthly = df.groupby(pd.Grouper(key='date', freq='M')).size().reset_index(name='detections')
        fig_monthly = px.line(
            df_monthly,
            x='date',
            y='detections',
            labels={'date': 'Date', 'detections': "Nombre de Détections"},
            title="Évolution Mensuelle"
        )
        st.subheader("Par Mois")
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Évolution annuelle
        df_yearly = df.groupby(pd.Grouper(key='date', freq='Y')).size().reset_index(name='detections')
        fig_yearly = px.line(
            df_yearly,
            x='date',
            y='detections',
            labels={'date': 'Date', 'detections': "Nombre de Détections"},
            title="Évolution Annuelle"
        )
        st.subheader("Par Année")
        st.plotly_chart(fig_yearly, use_container_width=True)
    else:
        st.error("La colonne 'date' est manquante ou mal formatée.")

else:
    st.info("Veuillez charger un fichier JSON pour afficher le tableau de bord.")
