import streamlit as st

# Base de données virtuelle de 5 comptes
users_db = {
    "alice": {"password": "pass1", "role": "directions"},
    "bob": {"password": "pass2", "role": "services"},
    "charlie": {"password": "pass3", "role": "directions"},
    "david": {"password": "pass4", "role": "services"},
    "eve": {"password": "pass5", "role": "services"}
}

# Initialisation de l'état de connexion dans la session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "role" not in st.session_state:
    st.session_state.role = None

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

if not st.session_state.logged_in:
    st.title("Connexion à l'application de détection de défauts")
    st.write("Veuillez vous connecter avec vos identifiants.")
    
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if login(username, password):
            st.success("Connexion réussie!")
            st.experimental_rerun()  # Actualise pour afficher le contenu correspondant
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")

else:
    st.sidebar.write(f"Connecté en tant que : **{st.session_state.current_user}**")
    st.sidebar.write(f"Mode d'accès : **{st.session_state.role.capitalize()}**")
    
    st.sidebar.button("Déconnexion", on_click=logout)

    # Redirection vers l'interface selon le rôle de l'utilisateur
    if st.session_state.role == "directions":
        st.title("Interface - Directions")
        st.write("Bienvenue dans l'espace Directions. Ici, vous pouvez consulter et gérer les rapports et analyses pour la Direction.")
        # Placez ici le contenu spécifique à l'interface Directions
    elif st.session_state.role == "services":
        st.title("Interface - Services")
        st.write("Bienvenue dans l'espace Services. Ici, vous pouvez accéder aux outils de détection et aux fonctionnalités propres aux Services.")
        # Placez ici le contenu spécifique à l'interface Services
    else:
        st.write("Rôle non reconnu.")
