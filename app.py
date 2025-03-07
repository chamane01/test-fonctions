import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Ubuntu Détect",
    page_icon="🔍",
    layout="wide"
)

# Base de données virtuelle
users_db = {
    "alice": {"password": "pass1", "role": "directions", "profile": "fille.jpeg"},
    "bob": {"password": "pass2", "role": "services", "profile": "garcon.jpeg"},
    "charlie": {"password": "pass3", "role": "directions", "profile": "garcon.jpeg"},
    "david": {"password": "pass4", "role": "services", "profile": "garcon.jpeg"},
    "eve": {"password": "pass5", "role": "directions", "profile": "fille.jpeg"}
}

DEFAULT_PROFILE = "profil.jpg"

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main {
        background: url('https://i.pinimg.com/originals/7d/9f/53/7d9f5307f0b74e9c5e8b5a7c5e5e8b5a.jpg');
        background-size: cover;
    }
    
    .login-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 3rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        width: 100%;
        background: #4CAF50 !important;
        color: white !important;
    }
    
    .sidebar .profile-box {
        padding: 1.5rem;
        background: #f0f2f6;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Gestion de session
def init_session():
    session_defaults = {
        "logged_in": False,
        "current_user": None,
        "role": None,
        "page_option": None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

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
    st.rerun()

# Page de connexion
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.title("🔍 Ubuntu Détect")
            st.image("ubuntu_logo.png", width=200)
            st.write("""
                **L'Esprit d'Humanité dans la Détection des Défauts**  
                *Plateforme collaborative de gestion des inspections techniques*
            """)
            
            with st.form("Login"):
                username = st.text_input("Nom d'utilisateur")
                password = st.text_input("Mot de passe", type="password")
                if st.form_submit_button("Se connecter"):
                    if not login(username, password):
                        st.error("Identifiants incorrects")
            st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Interface principale
user_data = users_db.get(st.session_state.current_user, {})
profile_image = user_data.get("profile", DEFAULT_PROFILE)

# Sidebar
with st.sidebar:
    st.markdown('<div class="profile-box">', unsafe_allow_html=True)
    st.image(profile_image, width=100)
    st.subheader(st.session_state.current_user)
    st.caption(f"Rôle: {st.session_state.role.capitalize()}")
    if st.button("🚪 Déconnexion"):
        logout()
    st.markdown('</div>', unsafe_allow_html=True)

    menu_options = ["Tableau de bord", "Missions", "Rapports"] if st.session_state.role == "directions" else ["Missions", "Rapports"]
    st.session_state.page_option = st.radio("Navigation", menu_options)

# Contenu principal
st.title(f"🔍 {st.session_state.page_option}")

if st.session_state.page_option == "Tableau de bord":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">📊 **Statistiques**<br>15 Nouvelles missions<br>32% Progrès global</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">✅ **Réalisations**<br>98% Taux de résolution<br>4.8★ Satisfaction</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">📅 **Calendrier**<br>3 Échéances<br>2 Réunions</div>', unsafe_allow_html=True)
    
    with st.expander("📈 Graphiques analytiques"):
        st.line_chart({"Données": [1, 3, 2, 4, 5, 2, 4]})

elif st.session_state.page_option == "Missions":
    tab1, tab2 = st.tabs(["📋 Liste des missions", "➕ Nouvelle mission"])
    
    with tab1:
        st.dataframe({
            "Mission": ["Inspection électrique", "Contrôle sécurité", "Audit réseau"],
            "Statut": ["En cours", "Terminé", "En attente"],
            "Progrès": [45, 100, 10]
        }, use_container_width=True)
    
    with tab2:
        with st.form("Nouvelle mission"):
            st.text_input("Titre de la mission")
            st.date_input("Date d'échéance")
            st.text_area("Description")
            st.form_submit_button("Créer mission")

elif st.session_state.page_option == "Rapports":
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.date_input("Date de début")
            st.date_input("Date de fin")
            st.selectbox("Format", ["PDF", "Excel", "HTML"])
        with col2:
            st.write("Aperçu du rapport")
            st.code("Données du rapport généré...", language="markdown")
            
    if st.button("🖨 Générer le rapport"):
        st.success("Rapport généré avec succès!")
