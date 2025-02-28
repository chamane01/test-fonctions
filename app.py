import streamlit as st
import pandas as pd
import sqlite3
import folium
import plotly.express as px
from streamlit_folium import st_folium
from datetime import datetime

# === Fonctions de gestion de la base de données ===

def connect_db(db_file='base_donnees.db'):
    """Se connecte à la base SQLite (créée si nécessaire)."""
    return sqlite3.connect(db_file)

def create_tables(conn):
    """Crée les tables missions et defects si elles n'existent pas déjà."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS missions (
            id TEXT PRIMARY KEY,
            operator TEXT,
            appareil_type TEXT,
            nom_appareil TEXT,
            date TEXT,
            troncon TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS defects (
            id TEXT PRIMARY KEY,
            mission_id TEXT,
            classe TEXT,
            gravite INTEGER,
            coordonnees_utm TEXT,
            lat REAL,
            longitude REAL,
            routes TEXT,
            detection TEXT,
            couleur TEXT,
            radius INTEGER,
            date TEXT,
            appareil TEXT,
            nom_appareil TEXT,
            FOREIGN KEY (mission_id) REFERENCES missions(id)
        )
    ''')
    conn.commit()

def insert_mission(conn, row):
    """Insère une mission dans la table missions."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO missions (id, operator, appareil_type, nom_appareil, date, troncon)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (row['id'], row['operator'], row['appareil_type'], row['nom_appareil'], row['date'], row['troncon']))
    conn.commit()

def insert_defect(conn, row):
    """Insère un défaut dans la table defects."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO defects 
        (id, mission_id, classe, gravite, coordonnees_utm, lat, longitude, routes, detection, couleur, radius, date, appareil, nom_appareil)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row['ID'], 
        row['mission'], 
        row['classe'], 
        row['gravite'], 
        row['coordonnees UTM'], 
        row['lat'], 
        row['long'],  # colonne "long" issue du fichier source
        row['routes'], 
        row['detection'], 
        row['couleur'], 
        row['radius'], 
        row['date'], 
        row['appareil'], 
        row['nom_appareil']
    ))
    conn.commit()

def process_uploaded_file(uploaded_file):
    """
    Charge le fichier TXT (séparateur tabulation), affiche un aperçu,
    crée les tables et insère les données dans la base SQLite.
    """
    df = pd.read_csv(uploaded_file, sep="\t")
    st.write("Aperçu des données importées :")
    st.dataframe(df.head())

    # Préparation d'une table missions à partir des missions uniques
    missions_df = df[['mission', 'appareil', 'nom_appareil', 'date']].drop_duplicates().copy()
    missions_df.rename(columns={'mission': 'id', 'appareil': 'appareil_type'}, inplace=True)
    missions_df['operator'] = ""
    missions_df['troncon'] = ""

    conn = connect_db()
    create_tables(conn)

    # Insertion des missions
    for _, row in missions_df.iterrows():
        insert_mission(conn, row)
    # Insertion des défauts
    for _, row in df.iterrows():
        insert_defect(conn, row)

    conn.close()
    st.success("Les données ont été insérées dans la base de données avec succès !")

def get_missions():
    """Récupère les missions depuis la base de données."""
    conn = connect_db()
    df = pd.read_sql("SELECT * FROM missions", conn)
    conn.close()
    return df

def get_defects():
    """Récupère les défauts depuis la base de données."""
    conn = connect_db()
    df = pd.read_sql("SELECT * FROM defects", conn)
    conn.close()
    return df

def add_mission_manual(id_mission, operator, appareil_type, nom_appareil, date_mission, troncon):
    """Ajoute une nouvelle mission dans la base de données."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO missions (id, operator, appareil_type, nom_appareil, date, troncon)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (id_mission, operator, appareil_type, nom_appareil, date_mission, troncon))
    conn.commit()
    conn.close()

# === Fonctions de visualisation ===

def show_map(defects_df):
    """Affiche une carte interactive avec les défauts."""
    if defects_df.empty:
        st.warning("Aucune donnée de défauts à afficher sur la carte.")
        return
    # Calcul du centre de la carte
    center_lat = defects_df['lat'].mean()
    center_lon = defects_df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    for _, row in defects_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['longitude']],
            radius=5 + row['gravite'],  # taille proportionnelle à la gravité
            color=row['couleur'] if row['couleur'] else 'blue',
            fill=True,
            fill_color=row['couleur'] if row['couleur'] else 'blue',
            popup=(
                f"<b>ID :</b> {row['id']}<br>"
                f"<b>Classe :</b> {row['classe']}<br>"
                f"<b>Gravité :</b> {row['gravite']}<br>"
                f"<b>Date :</b> {row['date']}"
            ),
            tooltip=row['classe']
        ).add_to(m)
    st_folium(m, width=700, height=500)

def show_charts(defects_df):
    """Affiche plusieurs graphiques interactifs basés sur les défauts."""
    st.subheader("Répartition des défauts par catégorie")
    if not defects_df.empty:
        fig1 = px.pie(defects_df, names='classe', title="Défauts par catégorie")
        st.plotly_chart(fig1)

        st.subheader("Distribution de la gravité")
        fig2 = px.histogram(defects_df, x="gravite", nbins=10, title="Histogramme de la gravité")
        st.plotly_chart(fig2)

        st.subheader("Évolution temporelle des défauts")
        # Conversion de la colonne date en datetime
        defects_df['date'] = pd.to_datetime(defects_df['date'], errors='coerce')
        df_time = defects_df.dropna(subset=['date'])
        if not df_time.empty:
            df_time_grouped = df_time.groupby(df_time['date'].dt.date).size().reset_index(name='Nombre de défauts')
            fig3 = px.line(df_time_grouped, x='date', y='Nombre de défauts', title="Évolution des défauts")
            st.plotly_chart(fig3)
        else:
            st.info("Les données temporelles ne sont pas au format attendu.")
    else:
        st.warning("Aucune donnée disponible pour les graphiques.")

# === Application principale Streamlit ===

def main():
    st.title("Tableau de Bord Dynamique – Gestion des Défauts")
    navigation = st.sidebar.radio("Navigation", ["Charger les données", "Tableau de Bord", "Gestion des Missions"])
    
    # Section 1 : Chargement des données
    if navigation == "Charger les données":
        st.header("Charger un fichier TXT")
        st.write("Importer le fichier TXT (séparateur tabulation) contenant la base de données des défauts.")
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["txt", "csv"])
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)

    # Section 2 : Tableau de Bord des Défauts
    elif navigation == "Tableau de Bord":
        st.header("Visualisation des Défauts")
        defects_df = get_defects()
        if defects_df.empty:
            st.warning("Aucune donnée de défauts disponible. Veuillez charger un fichier au préalable.")
        else:
            st.subheader("Carte Interactive")
            show_map(defects_df)
            st.subheader("Statistiques")
            show_charts(defects_df)

    # Section 3 : Gestion des Missions
    elif navigation == "Gestion des Missions":
        st.header("Gestion des Missions")
        missions_df = get_missions()
        st.subheader("Liste des Missions")
        st.dataframe(missions_df)
        
        st.subheader("Ajouter une nouvelle Mission")
        with st.form("ajouter_mission_form"):
            id_mission = st.text_input("ID Mission")
            operator = st.text_input("Opérateur")
            appareil_type = st.text_input("Type d'appareil")
            nom_appareil = st.text_input("Nom de l'appareil")
            date_mission = st.date_input("Date de la mission", datetime.today())
            troncon = st.text_input("Tronçon")
            submit = st.form_submit_button("Ajouter Mission")
            if submit:
                add_mission_manual(id_mission, operator, appareil_type, nom_appareil, str(date_mission), troncon)
                st.success("Mission ajoutée avec succès !")
    
    # Option de téléchargement de la base de données (accessible depuis la sidebar)
    st.sidebar.subheader("Télécharger la base de données")
    if st.sidebar.button("Télécharger DB"):
        try:
            with open('base_donnees.db', 'rb') as f:
                st.download_button("Télécharger base_donnees.db", f, file_name="base_donnees.db")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement : {e}")

if __name__ == '__main__':
    main()
