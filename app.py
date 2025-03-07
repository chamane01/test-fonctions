import streamlit as st

# Base de données virtuelle de 5 comptes avec image de profil (chemins relatifs aux images)
users_db = {
    "alice": {"password": "pass1", "role": "directions", "profile": "fille.jpeg"},
    "bob": {"password": "pass2", "role": "services", "profile": "garcon.jpeg"},
    "charlie": {"password": "pass3", "role": "directions", "profile": "garcon.jpeg"},
    "david": {"password": "pass4", "role": "services", "profile": "garcon.jpeg"},
    "eve": {"password": "pass5", "role": "directions", "profile": "fille.jpeg"}
}

# Si aucun profil n'est défini pour un utilisateur, on utilisera "profil.jpg"
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
    st.experimental_rerun()

# Page de connexion (affichée si l'utilisateur n'est pas connecté)
if not st.session_state.logged_in:
    st.title("Connexion à Ubuntu Détect")
    st.image("images (5).png", width=200)  # Logo affiché sur la page de connexion
    st.write("Bienvenue dans **Ubuntu Détect : L'Esprit d'Humanité dans la Détection des Défauts**")
    
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if login(username, password):
            st.success("Connexion réussie!")
            # La mise à jour de st.session_state provoque une réexécution du script.
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

# Une fois connecté, affichage de l'interface principale
else:
    # Récupération de l'image de profil : si l'utilisateur a un profil défini, on l'utilise, sinon par défaut
    user_data = users_db.get(st.session_state.current_user, {})
    profile_image = user_data.get("profile", DEFAULT_PROFILE)
    
    # Affichage de la sidebar avec image de profil en forme circulaire
    sidebar_html = f"""
    <div style="text-align: center;">
        <img src="{profile_image}" style="width:100px; border-radius:50%; border:2px solid #ccc;" />
        <p><strong>{st.session_state.current_user}</strong><br>({st.session_state.role.capitalize()})</p>
    </div>
    """
    st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)
    if st.sidebar.button("Déconnexion"):
        logout()
        st.stop()
    
    # Définition du menu dynamique selon le rôle
    if st.session_state.role == "directions":
        options = ["Tableau de bord", "Missions", "Rapports"]
    elif st.session_state.role == "services":
        options = ["Missions", "Rapports"]
    else:
        options = []
    
    st.session_state.page_option = st.sidebar.radio("Menu", options)
    
    # Titre principal dynamique
    st.title("Ubuntu Détect : L'Esprit d'Humanité dans la Détection des Défauts")
    
    # Affichage du contenu en fonction de l'option choisie
    if st.session_state.page_option == "Tableau de bord":
        st.write("Contenu du Tableau de bord : indicateurs, graphiques interactifs, etc.")
        # ... insérez ici votre code du tableau de bord ...
    elif st.session_state.page_option == "Missions":
        st.write("Contenu de la Gestion des Missions : création, suivi, export, etc.")
        # ... insérez ici votre code de gestion des missions ...
    elif st.session_state.page_option == "Rapports":
        st.write("Contenu de la Génération de Rapports : choix de périodes, génération de PDF, etc.")
        # ... insérez ici votre code de génération de rapports ...
    else:
        st.write("Veuillez sélectionner une option dans le menu.")
