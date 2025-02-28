import streamlit as st
import json
import random

# Listes de valeurs possibles pour certains champs
CLASSES = [
    "deformations ornierage", "fissurations", "Faiençage", "fissure de retrait",
    "fissure anarchique", "reparations", "nid de poule", "fluage",
    "arrachements", "depot de terre", "assainissements", "envahissement vegetations",
    "chaussée detruite", "denivellement accotement"
]
ROUTES = ["la cotiere", "A1", "A2", "A3", "A100", "Autre"]
HEX_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#008000", "#800080", "#8B4513", "#808080", "#A52A2A",
    "#FFC0CB", "#000080"
]

def generate_defect(mission_id, defect_index):
    """
    Génère un dictionnaire correspondant à un défaut, en se basant sur les exemples fournis.
    Les coordonnées UTM, lat et long sont légèrement randomisées autour de valeurs de base.
    """
    utm_x = 429640 + random.uniform(-50, 50)
    utm_y = 578928 + random.uniform(-50, 50)
    lat = 5.237 + random.uniform(-0.001, 0.001)
    lon = -3.6349 + random.uniform(-0.001, 0.001)
    
    defect = {
        "ID": f"{mission_id}-{defect_index+1}",
        "classe": random.choice(CLASSES),
        "gravite": random.choice([1, 2, 3]),
        "coordonnees UTM": (round(utm_x, 2), round(utm_y, 2)),
        "lat": round(lat, 6),
        "long": round(lon, 6),
        "routes": random.choice(ROUTES),
        "detection": "Manuelle",
        "mission": mission_id,
        "couleur": random.choice(HEX_COLORS),
        "radius": random.choice([5, 7, 9]),
        "date": "2025-02-28",
        "appareil": "Drone",
        "nom_appareil": "phantom"
    }
    return defect

def generate_mission(mission_index, num_defects):
    """
    Génère une mission au format configuration 2.
    Le champ 'troncon' est généré sous la forme pkX-pkY, avec X et Y aléatoires.
    La liste 'Données Défauts' contient 'num_defects' défauts générés.
    """
    mission_id = f"20250228-001-I-{mission_index:03d}"
    mission = {
        "id": mission_id,
        "operator": "magass",
        "appareil_type": "Drone",
        "nom_appareil": "phantom",
        "date": "2025-02-28",
        "troncon": f"pk{random.randint(1,20)}-pk{random.randint(21,100)}",
        "Données Défauts": [generate_defect(mission_id, i) for i in range(num_defects)]
    }
    return mission

def main():
    st.title("Téléchargement de la Base de Données")
    st.markdown(
        """
        Ce Streamlit génère une base de données de missions pour défauts détectés.
        Dans cet exemple, 150 missions sont créées de façon à totaliser exactement 1000 défauts.
        Les données sont générées de façon aléatoire en s’appuyant sur les configurations fournies.
        """
    )
    
    # Paramètres
    num_missions = 150
    total_defects = 1000
    
    # Distribuer 6 défauts par mission (6*150 = 900) et répartir les 100 défauts restants aléatoirement
    defects_distribution = [6] * num_missions
    extra_defects = total_defects - (6 * num_missions)  # ici 100
    indices_extra = random.sample(range(num_missions), extra_defects)
    for idx in indices_extra:
        defects_distribution[idx] += 1

    # Générer les missions
    missions = [
        generate_mission(i + 1, defects_distribution[i])
        for i in range(num_missions)
    ]
    
    # Conversion en texte (ici au format JSON avec indentations)
    data_txt = json.dumps(missions, indent=4, ensure_ascii=False)
    
    st.download_button(
        label="Télécharger la base de données (txt)",
        data=data_txt,
        file_name="base_donnees_missions.txt",
        mime="text/plain"
    )
    
    st.write("Nombre total de défauts générés :", sum(defects_distribution))
    st.write("Nombre de missions générées :", num_missions)

if __name__ == "__main__":
    main()
