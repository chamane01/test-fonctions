import streamlit as st

# Configuration CSS personnalisée
st.markdown("""
    <style>
        .profile-frame {
            background: linear-gradient(45deg, #2C3E50, #3498DB);
            padding: 4px;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .profile-frame:hover {
            transform: scale(1.05);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2C3E50 60%, #3498DB 100%);
            color: white;
        }
        .stButton>button {
            background: #3498DB !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px 24px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: #2980B9 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .stTextInput>div>div>input {
            border-radius: 8px !important;
            padding: 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Base de données virtuelle
users_db = {
    "alice": {"password": "pass1", "role": "directions", "profile": "fille.jpeg"},
    "bob": {"password": "pass2", "role": "services", "profile": "garcon.jpeg"},
    "charlie": {"password": "pass3", "role": "directions", "profile": "garcon.jpeg"},
    "david": {"password": "pass4", "role": "services", "profile": "garcon.jpeg"},
    "eve": {"password": "pass5", "role": "directions", "profile": "fille.jpeg"}
}

DEFAULT_PROFILE = "profil.jpg"

# Initialisation session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.role = None
    st.session_state.page_option = None

def login(username, password):
    if user := users_db.get(username):
        if user["password"] == password:
            st.session_state.update({
                "logged_in": True,
                "current_user": username,
                "role": user["role"]
            })
            return True
    return False

def logout():
    st.session_state.clear()
    st.experimental_rerun()

# Page de connexion
if not st.session_state.logged_in:
    col = st.columns([1, 3, 1])
    with col[1]:
        st.image("images (5).png", width=300)
        st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Ubuntu Détect</h1>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; margin-bottom: 40px;'>L'Esprit d'Humanité dans la Détection des Défauts</div>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur", key="user_input")
            password = st.text_input("Mot de passe", type="password", key="pass_input")
            if st.form_submit_button("Se connecter 🔑"):
                if login(username, password):
                    st.success("Connexion réussie! ✅")
                else:
                    st.error("Identifiants incorrects ❌")

# Interface principale
else:
    user_data = users_db.get(st.session_state.current_user, {})
    profile_image = user_data.get("profile", DEFAULT_PROFILE)
    
    # Sidebar stylisée
    with st.sidebar:
        st.markdown(f"""
            <div style='text-align: center; padding: 20px 0;'>
                <div class="profile-frame">
                    <img src="{profile_image}" 
                         style="width:100px; height:100px; 
                                border-radius:50%; 
                                object-fit:cover;
                                border: 2px solid white;"/>
                </div>
                <h3 style='margin: 15px 0;'>{st.session_state.current_user}</h3>
                <div style='background: #3498DB; 
                            padding: 6px; 
                            border-radius: 8px;
                            margin-bottom: 30px;'>
                    {st.session_state.role.capitalize()}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Déconnexion 🚪"):
            logout()
        
        menu_options = ["Tableau de bord", "Missions", "Rapports"] if st.session_state.role == "directions" else ["Missions", "Rapports"]
        st.session_state.page_option = st.radio("Navigation", menu_options, label_visibility="collapsed")

    # Contenu principal
    st.markdown(f"<h1 style='color: #2C3E50;'>{st.session_state.page_option}</h1>", unsafe_allow_html=True)
    
    if st.session_state.page_option == "Tableau de bord":
        st.write("Statistiques en temps réel...")
        # Ajouter des composants visuels ici
    
    elif st.session_state.page_option == "Missions":
        st.write("Gestion des missions...")
    
    elif st.session_state.page_option == "Rapports":
        st.write("Génération de rapports...")
