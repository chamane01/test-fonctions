import streamlit as st
import pandas as pd
import json

# Fonction pour charger et transformer le fichier en JSON
def txt_to_json(file):
    df = pd.read_csv(file, delimiter="\t")  # Lire le fichier avec tabulation comme séparateur
    json_data = df.to_dict(orient="records")  # Convertir en liste de dictionnaires
    return json.dumps(json_data, indent=4, ensure_ascii=False)

# Interface Streamlit
st.title("Convertisseur TXT vers JSON")

uploaded_file = st.file_uploader("Téléchargez votre fichier TXT", type=["txt"])

if uploaded_file is not None:
    json_result = txt_to_json(uploaded_file)
    st.subheader("Résultat JSON :")
    st.text_area("JSON Output", json_result, height=300)
    
    st.download_button(
        label="Télécharger le JSON",
        data=json_result,
        file_name="output.json",
        mime="application/json"
    )
