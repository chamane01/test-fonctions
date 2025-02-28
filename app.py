import streamlit as st
import pandas as pd
import json
import plotly.express as px

# Configuration de la page
st.set_page_config(page_title="Dashboard d'Analyses JSON", layout="wide")
st.title("📊 Dashboard d'Analyses des Données JSON")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisissez une section", ["📄 Aperçu des Données", "📊 Tableau de Bord"])

# Chargement du fichier JSON
uploaded_file = st.file_uploader("Charger votre fichier JSON", type=["json"])

if uploaded_file is not None:
    # Lecture du fichier JSON et conversion en DataFrame
    data = json.load(uploaded_file)
    df = pd.DataFrame(data)
    
    # Conversion de la colonne 'date' en datetime si elle existe
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Section : Aperçu des Données
    if section == "📄 Aperçu des Données":
        st.subheader("Aperçu des Données")
        st.write(df.head())
    
    # Section : Tableau de Bord des Analyses
    elif section == "📊 Tableau de Bord":
        st.subheader("Tableau de Bord des Analyses")
        
        # Affichage de quelques métriques globales
        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre Total de Défauts", df.shape[0])
        col2.metric("Nombre de Routes", df["routes"].nunique() if "routes" in df.columns else "N/A")
        col3.metric("Nombre de Catégories", df["classe"].nunique() if "classe" in df.columns else "N/A")
        st.markdown("---")
        
        # 1. Histogramme des 5 routes ayant le plus de défauts
        st.subheader("1. Top 5 Routes avec le Plus de Défauts")
        if "routes" in df.columns:
            routes_counts = df["routes"].value_counts().head(5)
            fig1 = px.bar(
                x=routes_counts.index,
                y=routes_counts.values,
                labels={"x": "Route", "y": "Nombre de Défauts"},
                title="Les 5 routes avec le plus de défauts"
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.error("La colonne 'routes' est introuvable dans les données.")
        st.markdown("---")
        
        # 2. Histogramme de la gravité
        st.subheader("2. Distribution de la Gravité")
        if "gravite" in df.columns:
            fig2 = px.histogram(
                df,
                x="gravite",
                nbins=10,
                labels={"gravite": "Gravité"},
                title="Distribution des niveaux de gravité"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("La colonne 'gravite' est introuvable dans les données.")
        st.markdown("---")
        
        # 3. Défauts par Catégorie
        st.subheader("3. Défauts par Catégorie")
        if "classe" in df.columns:
            cat_counts = df["classe"].value_counts()
            fig3 = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                labels={"x": "Catégorie", "y": "Nombre de Défauts"},
                title="Nombre de défauts par catégorie"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error("La colonne 'classe' est introuvable dans les données.")
        st.markdown("---")
        
        # 4. Évolution des Détections par Jour, Mois et Année
        st.subheader("4. Évolution des Détections")
        if "date" in df.columns and not df["date"].isnull().all():
            # Évolution quotidienne
            df_daily = df.groupby("date").size().reset_index(name="detections")
            fig_daily = px.line(
                df_daily,
                x="date",
                y="detections",
                labels={"date": "Date", "detections": "Nombre de Détections"},
                title="Évolution Quotidienne"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Évolution mensuelle
            df_monthly = df.groupby(pd.Grouper(key="date", freq="M")).size().reset_index(name="detections")
            fig_monthly = px.line(
                df_monthly,
                x="date",
                y="detections",
                labels={"date": "Date", "detections": "Nombre de Détections"},
                title="Évolution Mensuelle"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Évolution annuelle
            df_yearly = df.groupby(pd.Grouper(key="date", freq="Y")).size().reset_index(name="detections")
            fig_yearly = px.line(
                df_yearly,
                x="date",
                y="detections",
                labels={"date": "Date", "detections": "Nombre de Détections"},
                title="Évolution Annuelle"
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
        else:
            st.error("La colonne 'date' est manquante ou mal formatée.")
else:
    st.info("Veuillez charger un fichier JSON pour afficher le tableau de bord.")
