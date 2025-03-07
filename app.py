import streamlit as st

# Définir des identifiants d'exemple (à adapter et sécuriser pour la production)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"

# Initialiser l'état de connexion dans la session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login(username, password):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.logged_in = True
        return True
    else:
        return False

if not st.session_state.logged_in:
    st.title("Connexion Admin")
    st.write("Veuillez entrer vos identifiants pour accéder à l'interface administrateur.")
    
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    
    if st.button("Se connecter"):
        if login(username, password):
            st.success("Connexion réussie!")
            st.experimental_rerun()  # Actualise la page pour afficher l'interface admin
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
else:
    st.title("Bienvenue, Admin!")
    st.write("Vous êtes connecté à l'interface administrateur.")
    
    # Ici, vous pouvez ajouter votre contenu administrateur
    st.write("Contenu protégé de l'interface admin ...")
    
    if st.button("Déconnexion"):
        st.session_state.logged_in = False
        st.experimental_rerun()
