import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta
import io

# --------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------
def get_data_from_db(db_path="missions.db"):
    """
    Récupère les données de la table 'mission_history' dans la base de données SQLite.
    La table est supposée contenir au moins les colonnes suivantes :
      - id (TEXT)
      - date (TEXT au format YYYY-MM-DD)
      - operator (TEXT)
      - appareil_type (TEXT)
      - nom_appareil (TEXT)
      - troncon (TEXT)    --> représentant le segment/route
      - defects (TEXT)     --> ou toute autre donnée pertinente
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM mission_history", conn)
        conn.close()
        if not df.empty:
            # Conversion de la colonne date au format datetime
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error("Erreur lors de la récupération des données depuis la base de données.")
        st.error(e)
        return pd.DataFrame()

def generate_fictitious_data():
    """
    Génère des données fictives pour 500 missions réparties sur une année (exemple : 2023)
    et retourne le contenu sous forme de chaîne de caractères au format texte (colonnes séparées par tabulations).
    """
    num_missions = 500
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    # Création d'une liste de dates possibles
    total_days = (end_date - start_date).days + 1
    date_range = [start_date + timedelta(days=i) for i in range(total_days)]
    
    operators = ["Alice", "Bob", "Charlie", "David", "Eva"]
    appareils = ["Drone", "Camera"]
    nom_appareils = ["Appareil A", "Appareil B", "Appareil C", "Appareil D"]
    troncons = ["Route 1", "Route 2", "Route 3", "Route 4", "Route 5"]
    
    data = []
    for i in range(num_missions):
        mission_date = np.random.choice(date_range)
        operator = np.random.choice(operators)
        appareil_type = np.random.choice(appareils)
        nom_appareil = np.random.choice(nom_appareils)
        troncon = np.random.choice(troncons)
        mission_id = f"{mission_date.strftime('%Y%m%d')}-{i+1:03d}"
        defects = f"Defaut_{np.random.randint(1, 5)}"
        data.append({
            "id": mission_id,
            "date": mission_date.strftime("%Y-%m-%d"),
            "operator": operator,
            "appareil_type": appareil_type,
            "nom_appareil": nom_appareil,
            "troncon": troncon,
            "defects": defects
        })
    df = pd.DataFrame(data)
    # Retourne le contenu au format texte (colonnes séparées par tabulations)
    return df.to_csv(sep="\t", index=False)

# --------------------------------------------------
# Fonctions d'affichage du tableau de bord
# --------------------------------------------------
def display_dashboard():
    st.title("Tableau de Bord des Missions")
    
    # Récupération des données depuis la base de données
    df = get_data_from_db()
    if df.empty:
        st.info("La base de données ne contient aucune donnée ou n'est pas accessible.")
        return
    
    # Filtrage par date dans la sidebar
    st.sidebar.subheader("Filtrer par période")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Date de début", min_value=min_date, value=min_date)
    end_date = st.sidebar.date_input("Date de fin", min_value=min_date, max_value=max_date, value=max_date)
    
    # Filtrage des données
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]
    
    st.subheader("Indicateurs clés")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre total de missions", len(filtered_df))
    # Nombre moyen de missions par mois
    if not filtered_df.empty:
        months = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 30.0
        col2.metric("Missions/mois (moyenne)", f"{len(filtered_df)/max(months, 1):.1f}")
    else:
        col2.metric("Missions/mois (moyenne)", "0")
    # Nombre de tronçons uniques
    col3.metric("Nombre de tronçons", filtered_df['troncon'].nunique())
    
    st.markdown("---")
    
    # Analyse temporelle : missions par mois
    st.subheader("Évolution des missions dans le temps")
    df_time = filtered_df.copy()
    df_time["month"] = df_time["date"].dt.to_period("M").astype(str)
    monthly_counts = df_time.groupby("month").size().reset_index(name="missions")
    fig_time = px.line(monthly_counts, x="month", y="missions", markers=True, title="Nombre de missions par mois")
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.markdown("---")
    
    # Analyse par tronçon (route)
    st.subheader("Répartition des missions par tronçon")
    route_counts = filtered_df.groupby("troncon").size().reset_index(name="missions")
    fig_route = px.bar(route_counts, x="troncon", y="missions", title="Nombre de missions par tronçon", text="missions")
    st.plotly_chart(fig_route, use_container_width=True)
    
    st.markdown("---")
    
    # Affichage de la table filtrée
    st.subheader("Détails des missions")
    st.dataframe(filtered_df)

def display_download_fictitious_data():
    st.sidebar.subheader("Données fictives")
    fictitious_txt = generate_fictitious_data()
    st.sidebar.download_button(
        label="Télécharger 500 missions fictives",
        data=fictitious_txt,
        file_name="fictitious_missions.txt",
        mime="text/plain"
    )

# --------------------------------------------------
# Programme principal
# --------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    menu_option = st.sidebar.selectbox("Choisissez une section", ["Tableau de bord"], key="dashboard_menu")
    
    # Affichage du bouton pour télécharger les données fictives
    display_download_fictitious_data()
    
    if menu_option == "Tableau de bord":
        display_dashboard()

if __name__ == "__main__":
    main()
