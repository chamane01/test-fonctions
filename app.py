import streamlit as st

# Base de données virtuelle de 5 comptes
users_db = {
    "alice": {"password": "pass1", "role": "directions"},
    "bob": {"password": "pass2", "role": "services"},
    "charlie": {"password": "pass3", "role": "directions"},
    "david": {"password": "pass4", "role": "services"},
    "eve": {"password": "pass5", "role": "directions"}
}

# Initialiser l'état de connexion et d'autres variables de session
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
    st.experimental_rerun()

# Si l'utilisateur n'est pas connecté, afficher la page de connexion
if not st.session_state.logged_in:
    st.title("Connexion à Ubuntu Détect")
    st.image("images (5).png", width=200)  # Logo affiché sur la page de connexion
    st.write("Bienvenue dans **Ubuntu Détect : L'Esprit d'Humanité dans la Détection des Défauts**")
    
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if login(username, password):
            st.success("Connexion réussie!")
            st.experimental_rerun()  # Rafraîchit la page pour afficher l'interface principale
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

# Une fois connecté, afficher l'interface principale
else:
    # Barre latérale affichant les informations de connexion et le menu
    st.sidebar.header(f"Connecté(e) : {st.session_state.current_user} ({st.session_state.role.capitalize()})")
    if st.sidebar.button("Déconnexion"):
        logout()
    
    # Menu dynamique en fonction du rôle de l'utilisateur
    if st.session_state.role == "directions":
        options = ["Tableau de bord", "Missions", "Rapports"]
    elif st.session_state.role == "services":
        options = ["Missions", "Rapports"]
    else:
        options = []
    
    st.session_state.page_option = st.sidebar.radio("Menu", options)
    
    # Titre principal dynamique de l'application
    st.title("Ubuntu Détect : L'Esprit d'Humanité dans la Détection des Défauts")
    
    # Contenu affiché selon l'option sélectionnée
    if st.session_state.page_option == "Tableau de bord":
        st.write("Contenu du Tableau de bord : indicateurs, graphiques interactifs, etc.")
        # Placez ici votre code spécifique pour le tableau de bord.
    elif st.session_state.page_option == "Missions":
        st.write("Contenu de la Gestion des Missions : création, suivi, export, etc.")
        # Placez ici votre code spécifique pour la gestion des missions.
    elif st.session_state.page_option == "Rapports":
        st.write("Contenu de la Génération de Rapports : choix de périodes, génération de PDF, etc.")
        # Placez ici votre code spécifique pour la génération de rapports.
    else:
        st.write("Veuillez sélectionner une option dans le menu.")
