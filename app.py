import streamlit as st
import pandas as pd
import json
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Configuration de la page
st.set_page_config(page_title="Dashboard de Défauts", layout="wide")
st.title("📊 Dashboard de Défauts - Analyse des Données JSON")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisissez une section", ["📄 Aperçu des Données", "📊 Tableau de Bord"])

# Chargement du fichier JSON
uploaded_file = st.file_uploader("Charger votre fichier JSON", type=["json"])

if uploaded_file is not None:
    # Lecture du JSON et conversion en DataFrame
    data = json.load(uploaded_file)
    df = pd.DataFrame(data)
    
    # Conversion de la colonne date en datetime (si présente)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Section Aperçu des Données
    if section == "📄 Aperçu des Données":
        st.subheader("Aperçu des Données")
        st.write(df.head())
    
    # Section Tableau de Bord
    elif section == "📊 Tableau de Bord":
        st.subheader("Tableau de Bord")
        
        # Affichage des métriques globales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Défauts", df.shape[0])
        col2.metric("Routes Uniques", df["routes"].nunique() if "routes" in df.columns else "N/A")
        col3.metric("Catégories", df["classe"].nunique() if "classe" in df.columns else "N/A")
        # Calcul total des défauts par catégorie sous forme de texte (affiché en dessous)
        if "classe" in df.columns:
            cat_totaux = df["classe"].value_counts().to_dict()
            cat_txt = " | ".join([f"{cat} : {nb}" for cat, nb in cat_totaux.items()])
            col4.metric("Défauts par Catégorie", cat_txt)
        else:
            col4.metric("Défauts par Catégorie", "N/A")
        st.markdown("---")
        
        # Carte Folium avec éléments dynamiques (cercles colorés)
        st.subheader("Carte des Défauts")
        if "lat" in df.columns and "long" in df.columns:
            avg_lat = df["lat"].mean()
            avg_lon = df["long"].mean()
        else:
            avg_lat, avg_lon = 0, 0
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=8)
        
        # Dictionnaire par défaut pour associer chaque catégorie à une couleur
        default_colors = {
            "assainissements": "green",
            "fissure anarchique": "orange",
            "Faiençage": "blue",
            "fissurations": "red",
            "fissure de retrait": "purple",
            "reparations": "pink",
            "arrachements": "gray",
            "denivellement accotement": "cyan",
            "fluage": "yellow"
        }
        
        # Ajouter un cercle pour chaque défaut (utilisation de la couleur et du rayon dynamiques)
        for idx, row in df.iterrows():
            lat = row["lat"] if "lat" in row else None
            lon = row["long"] if "long" in row else None
            if pd.notnull(lat) and pd.notnull(lon):
                # Utilisation de la couleur spécifiée dans la colonne 'couleur' si présente, sinon via le mapping par catégorie
                color = row["couleur"] if "couleur" in row and pd.notnull(row["couleur"]) else default_colors.get(row.get("classe", "").lower(), "black")
                # Utilisation de la valeur de 'radius' si présente, sinon une valeur par défaut
                radius = row["radius"] if "radius" in row and pd.notnull(row["radius"]) else 5
                popup_text = (
                    f"ID : {row.get('ID', 'N/A')}<br>"
                    f"Catégorie : {row.get('classe', 'N/A')}<br>"
                    f"Gravité : {row.get('gravite', 'N/A')}<br>"
                    f"Route : {row.get('routes', 'N/A')}<br>"
                    f"Date : {row.get('date', 'N/A')}"
                )
                folium.Circle(
                    location=[lat, lon],
                    radius=radius * 10,  # facteur d'agrandissement pour une meilleure visibilité
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=popup_text,
                    tooltip=f"{row.get('classe', 'N/A')} (Gravité {row.get('gravite', 'N/A')})"
                ).add_to(m)
        st_folium(m, width=700, height=500)
        st.markdown("---")
        
        # Statistiques par Catégorie
        st.subheader("Statistiques par Catégorie")
        if "classe" in df.columns:
            cat_counts = df["classe"].value_counts().reset_index()
            cat_counts.columns = ["Catégorie", "Nombre de Défauts"]
            st.dataframe(cat_counts)
            fig_cat = px.bar(cat_counts, x="Catégorie", y="Nombre de Défauts", title="Nombre de Défauts par Catégorie")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.error("La colonne 'classe' n'existe pas dans les données.")
        st.markdown("---")
        
        # 1. Top 5 Routes avec le Plus de Défauts
        st.subheader("Top 5 Routes avec le Plus de Défauts")
        if "routes" in df.columns:
            routes_counts = df["routes"].value_counts().head(5)
            fig_routes = px.bar(
                x=routes_counts.index,
                y=routes_counts.values,
                labels={"x": "Route", "y": "Nombre de Défauts"},
                title="Les 5 routes avec le plus de défauts"
            )
            st.plotly_chart(fig_routes, use_container_width=True)
        else:
            st.error("La colonne 'routes' n'existe pas dans les données.")
        st.markdown("---")
        
        # 2. Distribution de la Gravité
        st.subheader("Distribution de la Gravité")
        if "gravite" in df.columns:
            fig_gravite = px.histogram(
                df,
                x="gravite",
                nbins=10,
                labels={"gravite": "Gravité"},
                title="Distribution des Niveaux de Gravité"
            )
            st.plotly_chart(fig_gravite, use_container_width=True)
        else:
            st.error("La colonne 'gravite' n'existe pas dans les données.")
        st.markdown("---")
        
        # 3. Évolution des Détections par Jour, Mois et Année
        st.subheader("Évolution des Détections")
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
