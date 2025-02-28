import streamlit as st
import json
import ast

st.title("Transformation TXT en JSON")

# Permettre à l'utilisateur de charger le fichier txt
uploaded_file = st.file_uploader("Choisissez un fichier TXT", type="txt")

if uploaded_file is not None:
    # Lecture et décodage du contenu du fichier
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()

    if len(lines) < 2:
        st.error("Le fichier ne semble pas contenir suffisamment de données.")
    else:
        # La première ligne correspond à l'en-tête
        header = lines[0].split()
        # Si l'en-tête est par exemple "Données Défauts", on les fusionne en une seule clé
        if len(header) >= 7 and header[-2:] == ["Données", "Défauts"]:
            header = header[:6] + ["donnees_defauts"]

        data = []
        for line in lines[1:]:
            if not line.strip():
                continue  # ignorer les lignes vides
            # On découpe chaque ligne en autant de parties que de colonnes (maxsplit = nombre de colonnes - 1)
            parts = line.split(maxsplit=len(header)-1)
            record = {}
            for i, col in enumerate(header):
                # Pour la dernière colonne, tenter de convertir la chaîne en objet Python (liste de dict)
                if i == len(header)-1:
                    try:
                        record[col] = ast.literal_eval(parts[i])
                    except Exception as e:
                        record[col] = parts[i]
                else:
                    record[col] = parts[i]
            data.append(record)

        # Affichage du résultat sous forme JSON
        st.subheader("Données JSON")
        st.json(data)

        # Optionnel : possibilité de télécharger le JSON généré
        json_str = json.dumps(data, indent=4)
        st.download_button("Télécharger le JSON", json_str, file_name="data.json", mime="application/json")
