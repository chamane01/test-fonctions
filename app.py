import streamlit as st

# Base de données virtuelle de 5 comptes avec chemin d'image de profil
users_db = {
    "alice": {"password": "pass1", "role": "directions", "profile": "fille.jpeg"},
    "bob": {"password": "pass2", "role": "services", "profile": "garcon.jpeg"},
    "charlie": {"password": "pass3", "role": "directions", "profile": "garcon.jpeg"},
    "david": {"password": "pass4", "role": "services", "profile": "garcon.jpeg"},
    "eve": {"password": "pass5", "role": "directions", "profile": "fille.jpeg"}
}

# Image par défaut si aucun profil n'est défini
DEFAULT_PROFILE = "profil.jpg"

# Initialisation de l'état de session
if "logged_in" not in st.session_state:
    st.session_state.update({
        "logged_in": False,
        "current_user": None,
        "role": None,
        "page_option": None
    })

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

# Styles CSS globaux
st.markdown("""
<style>
    :root {
        --primary: #2C3E50;
        --secondary: #3498DB;
        --accent: #E74C3C;
        --background: #F8F9FA;
        --success: #2ECC71;
    }

    .stApp {
        background: var(--background);
    }

    .login-container {
        background: white;
        padding: 3rem 4rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 2rem auto;
        max-width: 500px;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        min-height: 180px;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary) 0%, #1a2835 100%);
        padding: 1rem;
    }

    .stButton>button {
        background: var(--accent) !important;
        border: 2px solid var(--accent) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(0.98);
    }

    .stRadio div[role="radiogroup"] {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 8px;
    }

    .stRadio [role="radio"] {
        padding: 12px 20px !important;
        margin: 4px 0 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }

    .stRadio [aria-checked="true"] {
        background: var(--secondary) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2) !important;
    }

    .section-header {
        position: relative;
        padding-bottom: 0.8rem;
        margin-bottom: 2rem !important;
    }

    .section-header::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 3px;
        background: var(--secondary);
        border-radius: 2px;
    }

    .stForm {
        border: 1px solid #e0e0e0 !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        background: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Page de connexion
if not st.session_state.logged_in:
    col = st.columns([1, 3, 1])
    with col[1]:
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.image("images (5).png", width=300)
            st.markdown(
                "<h1 style='text-align: center; color: var(--primary); margin-bottom: 0.5rem;'>Ubuntu Détect</h1>", 
                unsafe_allow_html=True
            )
            st.markdown(
                "<div style='text-align: center; color: var(--secondary); font-size: 1.2rem; margin-bottom: 2rem;'>L'Esprit d'Humanité dans la Détection des Défauts</div>", 
                unsafe_allow_html=True
            )
            
            with st.form("login_form"):
                username = st.text_input("Nom d'utilisateur", key="user_input")
                password = st.text_input("Mot de passe", type="password", key="pass_input")
                if st.form_submit_button("Se connecter 🔑"):
                    if login(username, password):
                        st.success("Connexion réussie! ✅")
                    else:
                        st.error("Identifiants incorrects ❌")
            st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Interface principale
else:
    user_data = users_db.get(st.session_state.current_user, {})
    profile_image = user_data.get("profile", DEFAULT_PROFILE)
    
    # Sidebar stylisée
    with st.sidebar:
        st.markdown('<div style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
        st.markdown('<div style="border: 4px solid var(--secondary); border-radius: 50%; padding: 4px; display: inline-block;">', 
                    unsafe_allow_html=True)
        st.image(profile_image, width=120)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='margin: 1.5rem 0;'>
                <h3 style='color: white; margin: 0;'>{st.session_state.current_user}</h3>
                <div style='color: rgba(255,255,255,0.7); font-size: 0.9em;'>{st.session_state.role.capitalize()}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.1); margin: 1.5rem 0;">', unsafe_allow_html=True)
        
        menu_options = ["📊 Tableau de bord", "📋 Missions", "📈 Rapports"] if st.session_state.role == "directions" else ["📋 Missions", "📈 Rapports"]
        st.session_state.page_option = st.radio("Navigation", menu_options, label_visibility="collapsed")
        
        st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
        if st.button("🚪 Déconnexion", use_container_width=True):
            logout()
        st.markdown('</div>', unsafe_allow_html=True)

    # Contenu principal
    st.markdown(
        f"<h1 class='section-header' style='color: var(--primary);'>{st.session_state.page_option.split()[-1]}</h1>", 
        unsafe_allow_html=True
    )
    
    if st.session_state.page_option == "📊 Tableau de bord":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>📊 Statistiques</h3>
                    <p>15 Nouvelles missions</p>
                    <div style="background: #f0f0f0; border-radius: 8px; height: 8px
