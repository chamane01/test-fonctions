import streamlit as st
from PIL import Image
import os

# Vérification et chargement des images
def load_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path)
    return Image.open("profil.jpg")  # Image par défaut

# Base de données virtuelle de 5 comptes avec image de profil
users_db = {
    "alice": {"password": "pass1", "role": "directions", "profile": "fille.jpeg"},
    "bob": {"password": "pass2", "role": "services", "profile": "garcon.jpeg"},
    "charlie": {"password": "pass3", "role": "directions", "profile": "garcon.jpeg"},
    "david": {"password": "pass4", "role": "services", "profile": "garcon.jpeg"},
    "eve": {"password": "pass5", "role": "directions", "profile": "fille.jpeg"}
}

# Image par défaut si aucun profil n'est défini
DEFAULT_PROFILE = "profil.jpg"

# Initialisation des variables de session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "page_option" not in st.session_state:
    st.session_state.page_option = None

def login(username, password):
    user = users_db.get(username)
    if user and user["password"] == password:
        st.session_state.logged_in = True
        st.session_state.current_user = username
        st.session_state.role = user["role"]
        return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.role = None
    st.session_state.page_option = None
    st.write("Déconnexion effectuée. Veuillez actualiser la page.")

# Si l'utilisateur n'est pas connecté, afficher la page de connexion
if not st.session_state.logged_in:
    st.title("Connexion à Ubuntu Détect")
    st.image("images (5).png", width=200)  # Affichage du logo sur la page de connexion
    st.write("Bienvenue dans **Ubuntu Détect : L'Esprit d'Humanité dans la Détection des Défauts**")
    
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if login(username, password):
            st.success("Connexion réussie!")
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

# Une fois connecté, afficher l'interface principale
else:
    # Récupération du chemin de l'image de profil de l'utilisateur
    user_data = users_db.get(st.session_state.current_user, {})
    profile_image_path = user_data.get("profile", DEFAULT_PROFILE)

    # Chargement de l'image
    profile_image = load_image(profile_image_path)

    # Affichage de l'image de profil avec une vignette circulaire
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; position: relative; display: inline-block;">
            <div style="
                width: 120px;
                height: 120px;
                border-radius: 50%;
                overflow: hidden;
                border: 5px solid #2E4053; /* Couleur du masque */
                display: flex;
                align-items: center;
                justify-content: center;">
                <img src="{profile_image_path}" style="width: 100%; height: 100%; object-fit: cover;" />
            </div>
            <p><strong>{st.session_state.current_user}</strong><br>({st.session_state.role.capitalize()})</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Bouton de déconnexion
    if st.sidebar.button("Déconnexion"):
        logout()

    # Définition du menu selon le rôle
    if st.session_state.role == "directions":
        options = ["Tableau de bord", "Missions", "Rapports"]
    elif st.session_state.role == "services":
        options = ["Missions", "Rapports"]
    else:
        options = []

    st.session_state.page_option = st.sidebar.radio("Menu", options)

    # Titre principal dynamique
    st.title("Ubuntu Détect : L'Esprit d'Humanité dans la Détection des Défauts")

    # Affichage du contenu selon l'option sélectionnée
    if st.session_state.page_option == "Tableau de bord":
        st.write("Contenu du Tableau de bord : indicateurs, graphiques interactifs, etc.")
    elif st.session_state.page_option == "Missions":
        st.write("Contenu de la Gestion des Missions : création, suivi, export, etc.")
    elif st.session_state.page_option == "Rapports":
        st.write("Contenu de la Génération de Rapports : choix de périodes, génération de PDF, etc.")
    else:
        st.write("Veuillez sélectionner une option dans le menu.")
