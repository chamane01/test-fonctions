import streamlit as st
import pandas as pd
import random
import json
from datetime import datetime, timedelta

# Dictionnaires de classes et tailles de gravité
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
gravity_sizes = {1: 5, 2: 7, 3: 9}

# Listes d'exemples pour opérateurs et appareils
operators = ["BINTA", "Magassouba", "KONE", "OUATTARA"]
appareil_types = ["Drone"]
nom_appareils = ["mavic", "phantom", "inspire", "spark"]

# Fonction pour générer une date aléatoire entre deux dates
def random_date(start, end):
    delta = end - start
    random_days = random.randrange(delta.days)
    return start + timedelta(days=random_days)

# Chargement du fichier GeoJSON (routeQSD.txt) contenant les routes
@st.cache_data
def load_routes(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    features = geojson.get("features", [])
    # On récupère la liste unique des IDs des routes
    route_ids = list({feat["properties"].get("ID") for feat in features if "ID" in feat["properties"]})
    return features, route_ids

route_features, route_ids = load_routes("routeQSD.txt")

# Fonction qui, à partir d'un identifiant de route, retourne une coordonnée (lat, long)
def get_random_coordinate(route_id):
    for feat in route_features:
        if feat["properties"].get("ID") == route_id:
            coords = feat["geometry"]["coordinates"]
            # Choix aléatoire d'un point dans la ligne
            coord = random.choice(coords)
            # GeoJSON stocke [longitude, latitude]
            return coord[1], coord[0]
    return round(random.uniform(5.2, 5.4), 6), round(random.uniform(-4.1, -3.9), 6)

# Fonction pour générer des coordonnées UTM aléatoires (en mètres)
def get_random_utm():
    easting = round(random.uniform(397000, 430000), 2)
    northing = round(random.uniform(578000, 594000), 2)
    return (easting, northing)

# Paramètres de génération
num_missions = 130
total_defauts = 1234
base_defauts = 9
extra_defauts = total_defauts - (base_defauts * num_missions)  # ici 64

# Répartition des défauts supplémentaires sur certaines missions
missions_extra = random.sample(range(num_missions), extra_defauts)

start_date = datetime(2023, 5, 1)
end_date = datetime(2025, 2, 28)

missions = []
mission_defaut_count = 0

# Génération des missions
for i in range(num_missions):
    mission_date = random_date(start_date, end_date)
    date_str = mission_date.strftime("%Y%m%d")
    mission_counter = f"{i+1:03d}"
    letter = random.choice(["F", "D"])
    mission_id = f"{date_str}-{mission_counter}-{letter}"
    
    operator = random.choice(operators)
    appareil_type = random.choice(appareil_types)
    nom_appareil = random.choice(nom_appareils)
    
    troncon = random.choice([f"pk{random.randint(1,20)}-pk{random.randint(20,100)}", "Route inconnue"])
    
    n_defauts = base_defauts + (1 if i in missions_extra else 0)
    defauts = []
    for j in range(n_defauts):
        defaut_id = f"{mission_id}-{j+1}"
        classe = random.choice(list(class_color.keys()))
        gravite = random.choice([1, 2, 3])
        utm = get_random_utm()
        if random.random() < 0.8 and route_ids:
            route_defaut = random.choice(route_ids)
            lat, lon = get_random_coordinate(route_defaut)
        else:
            lat = round(random.uniform(5.2, 5.4), 6)
            lon = round(random.uniform(-4.1, -3.9), 6)
            route_defaut = "Route inconnue"
        
        defaut = {
            "ID": defaut_id,
            "classe": classe,
            "gravite": gravite,
            "coordonnees UTM": utm,
            "lat": lat,
            "long": lon,
            "routes": route_defaut,
            "detection": "Manuelle",
            "mission": mission_id,
            "couleur": class_color[classe],
            "radius": gravity_sizes[gravite],
            "date": mission_date.strftime("%Y-%m-%d"),
            "appareil": appareil_type,
            "nom_appareil": nom_appareil
        }
        defauts.append(defaut)
        mission_defaut_count += 1

    mission = {
        "id": mission_id,
        "operator": operator,
        "appareil_type": appareil_type,
        "nom_appareil": nom_appareil,
        "date": mission_date.strftime("%Y-%m-%d"),
        "troncon": troncon,
        "Données Défauts": defauts
    }
    missions.append(mission)

st.write(f"Nombre total de missions générées : {len(missions)}")
st.write(f"Nombre total de défauts générés : {mission_defaut_count}")

# Affichage sous forme de DataFrame (les colonnes imbriquées s'affichent en 'object')
df = pd.DataFrame(missions)
st.dataframe(df)

# Téléchargement en CSV (tableau plat)
@st.cache_data
def convert_df(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv = convert_df(df)
st.download_button(
    label="Télécharger le jeu de données en CSV",
    data=csv,
    file_name='jeu_donnees_missions.csv',
    mime='text/csv'
)

# Conversion de la liste de missions en JSON (structure complète)
json_data = json.dumps(missions, indent=4, ensure_ascii=False)

st.download_button(
    label="Télécharger le jeu de données en JSON",
    data=json_data,
    file_name='jeu_donnees_missions.json',
    mime='application/json'
)

# Téléchargement en TXT (ici le contenu JSON est sauvegardé dans un fichier .txt)
st.download_button(
    label="Télécharger le jeu de données en TXT",
    data=json_data,
    file_name='jeu_donnees_missions.txt',
    mime='text/plain'
)
