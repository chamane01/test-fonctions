import streamlit as st
import pandas as pd
import random

# Liste de quelques routes de la Côte d'Ivoire
routes_list = ["la cotiere", "A1", "A2", "A3", "A100", "A4", "A5"]

# Liste de classes de défauts (exemples)
defect_classes = [
    "deformations ornierage", "fissurations", "Faiençage", "fissure de retrait",
    "fissure anarchique", "reparations", "nid de poule", "fluage", "arrachements",
    "depot de terre", "assainissements", "envahissement vegetations",
    "chaussée detruite", "denivellement accotement"
]

def format_road(road):
    # Pour les routes de type A, on ajoute un descriptif
    if road.startswith("A"):
        return f"{road}(port-bouet, bassam)"
    return road

def generate_data(num_missions, total_defects):
    """
    Génère une DataFrame avec les colonnes suivantes :
    ID, classe, gravite, coordonnees UTM, lat, long, routes, detection, mission,
    couleur, radius, date, appareil, nom_appareil.
    """
    rows = []
    defect_counter_global = 1

    # Répartition uniforme des défauts par mission
    defects_per_mission = total_defects // num_missions
    reste = total_defects % num_missions

    for m in range(1, num_missions + 1):
        # Génération d'un code mission (ex: "20250228-001-I")
        mission_code = f"20250228-{m:03d}-I"
        # Ajustement pour le reste des défauts
        nb_defects = defects_per_mission + (1 if m <= reste else 0)
        for d in range(1, nb_defects + 1):
            # ID du défaut (ex: "20250228-001-I-1")
            defect_id = f"{mission_code}-{d}"
            # Choix aléatoire de la classe de défaut
            defect_class = random.choice(defect_classes)
            # Gravité aléatoire entre 1 et 3
            gravite = random.choice([1, 2, 3])
            # Coordonnées UTM simulées (valeurs proches de l'exemple)
            utm_x = round(random.uniform(429600, 429750), 2)
            utm_y = round(random.uniform(578840, 579000), 2)
            coordonnees = f"[{utm_x}, {utm_y}]"
            # Latitude et longitude avec de petites variations
            lat = round(random.uniform(5.2365, 5.2380), 4)
            lon = round(random.uniform(-3.6355, -3.6340), 4)
            # Choix aléatoire d'une route et formatage
            route = format_road(random.choice(routes_list))
            # Détection toujours "Manuelle" dans l'exemple
            detection = "Manuelle"
            # Couleur générée aléatoirement en hexadécimal
            couleur = f"#{random.randint(0, 0xFFFFFF):06X}"
            # Rayon : choix parmi quelques valeurs (exemples : 5, 7 ou 9)
            radius = random.choice([5, 7, 9])
            # Date, appareil et nom d'appareil fixes d'après l'exemple
            date = "2025-02-28"
            appareil = "Drone"
            nom_appareil = "phantom"
            
            rows.append({
                "ID": defect_id,
                "classe": defect_class,
                "gravite": gravite,
                "coordonnees UTM": coordonnees,
                "lat": lat,
                "long": lon,
                "routes": route,
                "detection": detection,
                "mission": mission_code,
                "couleur": couleur,
                "radius": radius,
                "date": date,
                "appareil": appareil,
                "nom_appareil": nom_appareil
            })
            defect_counter_global += 1

    df = pd.DataFrame(rows)
    return df

st.title("Générateur de base de données de défauts")
st.write("Ce script génère une base de données (TXT) selon les scénarios proposés.")

# Sélection du type de base de données dans la barre latérale
option = st.sidebar.selectbox(
    "Choisissez le type de base de données",
    options=[
        "100 défauts dans une mission",
        "100 missions avec 1000 défauts détectés",
        "150 missions avec 1000 défauts détectés"
    ]
)

# Paramétrage en fonction de l'option sélectionnée
if option == "100 défauts dans une mission":
    num_missions = 1
    total_defects = 100
elif option == "100 missions avec 1000 défauts détectés":
    num_missions = 100
    total_defects = 1000
elif option == "150 missions avec 1000 défauts détectés":
    num_missions = 150
    total_defects = 1000

st.sidebar.write(f"Nombre de missions : **{num_missions}**")
st.sidebar.write(f"Nombre total de défauts : **{total_defects}**")

# Génération de la base de données
df = generate_data(num_missions, total_defects)

st.subheader("Aperçu de la base de données")
st.dataframe(df.head(10))

# Conversion de la DataFrame en chaîne de caractères au format CSV (séparateur tabulation)
data_str = df.to_csv(sep="\t", index=False)

# Bouton de téléchargement
st.download_button(
    label="Télécharger la base de données (TXT)",
    data=data_str,
    file_name="base_de_donnees.txt",
    mime="text/plain"
)
