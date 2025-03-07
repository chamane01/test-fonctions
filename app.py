import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Ubuntu Détect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base de données virtuelle de 5 comptes avec chemin d'image de profil
users_db = {
    "alice": {"password": "pass1", "role": "directions", "profile": "fille.jpeg"},
    "bob": {"password": "pass2", "role": "services", "profile": "garcon.jpeg"},
    "charlie": {"password": "pass3", "role": "directions", "profile": "garcon.jpeg"},
    "david": {"password": "pass4", "role": "services", "profile": "garcon.jpeg"},
    "eve": {"password": "pass5", "role": "directions", "profile": "fille.jpeg"}
}

DEFAULT_PROFILE = "profil.jpg"

# Initialisation de l'état de session
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

# Styles CSS personnalisés
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .profile-img {
            border-radius: 50%;
            border: 3px solid #3498db;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        .stMarkdown h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Page de connexion
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images (5).png", width=300)
        st.markdown("<h1 style='text-align: center;'>🔐 Connexion à Ubuntu Détect</h1>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            username = st.text_input("👤 Nom d'utilisateur", key="login_user")
            password = st.text_input("🔒 Mot de passe", type="password", key="login_pass")
            
            if st.button("🚀 Se connecter", use_container_width=True):
                if login(username, password):
                    st.success("✅ Connexion réussie!")
                else:
                    st.error("❌ Identifiants incorrects")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("""
            <div style='text-align: center; margin-top: 20px; color: #7f8c8d;'>
                Bienvenue dans <strong>Ubuntu Détect</strong><br>
                L'Esprit d'Humanité dans la Détection des Défauts
            </div>
        """, unsafe_allow_html=True)

# Interface principale
else:
    # Sidebar
    with st.sidebar:
        user_data = users_db.get(st.session_state.current_user, {})
        profile_image = user_data.get("profile", DEFAULT_PROFILE)
        st.markdown(f"<img src='{profile_image}' class='profile-img' width='100'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: #3498db; text-align: center;'>{st.session_state.current_user}</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; margin-bottom: 30px;'>Rôle: <strong>{st.session_state.role.capitalize()}</strong></div>", unsafe_allow_html=True)
        
        if st.button("🚪 Déconnexion", use_container_width=True):
            logout()
        
        st.markdown("---")
        menu_title = st.session_state.role.capitalize() + " Menu"
        st.markdown(f"<h4 style='color: #3498db;'>{menu_title}</h4>", unsafe_allow_html=True)
        
        # Options du menu
        if st.session_state.role == "directions":
            options = ["📊 Tableau de bord", "📋 Missions", "📈 Rapports"]
        else:
            options = ["📋 Missions", "📈 Rapports"]
        
        st.session_state.page_option = st.radio("", options, label_visibility="collapsed")

    # Contenu principal
    st.markdown(f"<h1>Ubuntu Détect 🔍</h1><div style='height: 3px; background: linear-gradient(90deg, #3498db 0%, #2c3e50 100%); margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Contenu dynamique
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.session_state.page_option == "📊 Tableau de bord":
            st.header("📊 Tableau de bord interactif")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Missions actives", 12, "+3 vs hier")
            with cols[1]:
                st.metric("Taux de résolution", "84%", "-2%")
            with cols[2]:
                st.metric("Satisfaction", "⭐ 4.8/5", "stable")
            st.line_chart({"Données": [10, 20, 15, 25, 30]}, height=300)
            
        elif st.session_state.page_option == "📋 Missions":
            st.header("📋 Gestion des Missions")
            with st.expander("➕ Créer une nouvelle mission"):
                st.text_input("Titre de la mission")
                st.date_input("Date limite")
                st.form_submit_button("Créer")
            
            with st.expander("📂 Missions en cours (5)"):
                st.write("Liste des missions...")
                
        elif st.session_state.page_option == "📈 Rapports":
            st.header("📈 Analyse des Rapports")
            st.selectbox("Période", ["7 derniers jours", "30 derniers jours", "Personnalisé"])
            cols = st.columns(2)
            cols[0].bar_chart({"Résolus": [45], "En attente": [15]})
            cols[1].area_chart({"Évolution": [10, 20, 35, 40, 45]})
            
        st.markdown("</div>", unsafe_allow_html=True)
