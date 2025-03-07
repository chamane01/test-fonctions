import streamlit as st

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
    st.session_state.update({
        "logged_in": False,
        "current_user": None,
        "role": None,
        "page_option": None,
        "dark_mode": False  # Ajout du state pour le mode sombre
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
    # Toggle mode clair/sombre
    with st.sidebar:
        if st.button("🌙" if st.session_state.dark_mode else "☀️", help="Basculer le mode clair/sombre"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    user_data = users_db.get(st.session_state.current_user, {})
    profile_image = user_data.get("profile", DEFAULT_PROFILE)
    
    # Sidebar stylisée
    with st.sidebar:
        st.markdown('<div class="profile-box">', unsafe_allow_html=True)
        st.image(profile_image, width=150)
        st.markdown(
            f"<h3 style='color: var(--text-color); margin: 15px 0;'>{st.session_state.current_user}</h3>", 
            unsafe_allow_html=True
        )
        st.markdown(f"""
            <div style='background: var(--accent); 
                        padding: 6px; 
                        border-radius: 8px;
                        margin-bottom: 1rem;
                        color: white;'>
                {st.session_state.role.capitalize()}
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Déconnexion 🚪", use_container_width=True):
            logout()
        
        menu_options = ["Tableau de bord", "Missions", "Rapports"]
        st.session_state.page_option = st.radio("Navigation", menu_options, label_visibility="collapsed")
    
    # Définition des variables CSS selon le mode
    theme_vars = """
        :root {{
            --primary: {primary};
            --secondary: {secondary};
            --accent: {accent};
            --background: {background};
            --card-bg: {card_bg};
            --text-color: {text_color};
            --sidebar-bg: {sidebar_bg};
            --progress-bg: {progress_bg};
        }}
    """.format(
        primary="#FFFFFF" if st.session_state.dark_mode else "#2C3E50",
        secondary="#3498DB",
        accent="#2C3E50" if st.session_state.dark_mode else "#3498DB",
        background="#1A1A1A" if st.session_state.dark_mode else "#FFFFFF",
        card_bg="#2C3E50" if st.session_state.dark_mode else "#FFFFFF",
        text_color="#FFFFFF" if st.session_state.dark_mode else "#2C3E50",
        sidebar_bg="linear-gradient(180deg, #2C3E50 60%, #1A1A1A 100%)" if st.session_state.dark_mode else "linear-gradient(180deg, #FFFFFF 60%, #F0F0F0 100%)",
        progress_bg="#555555" if st.session_state.dark_mode else "#F0F0F0"
    )

    # Injection de CSS personnalisé
    st.markdown(f"""
    <style>
        {theme_vars}
        
        body {{
            background-color: var(--background);
            color: var(--text-color);
        }}
        
        .sidebar .sidebar-content {{
            background: var(--sidebar-bg);
            color: var(--text-color);
        }}
        
        .metric-card {{
            background: var(--card-bg);
            color: var(--text-color);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}
        
        .stButton>button {{
            background: var(--secondary) !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px 24px;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            background: var(--accent) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
    </style>
    """, unsafe_allow_html=True)

    # Affichage du titre dynamique
    if st.session_state.page_option == "Tableau de bord":
        title_text = "Tableau de bord général" if st.session_state.role == "directions" else "Tableau de bord personnel"
    else:
        title_text = st.session_state.page_option

    st.markdown(
        f"<h1 style='color: var(--primary); border-bottom: 2px solid var(--secondary); padding-bottom: 0.5rem;'>{title_text}</h1>", 
        unsafe_allow_html=True
    )
    
    # Contenu de la page sélectionnée
    if st.session_state.page_option == "Tableau de bord":
        if st.session_state.role == "directions":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <h3>📊 Statistiques</h3>
                        <p>15 Nouvelles missions</p>
                        <div style="background: var(--progress-bg); border-radius: 8px; height: 8px;">
                            <div style="background: var(--secondary); width: 32%; height: 100%; border-radius: 8px;"></div>
                        </div>
                        <p style="margin-top: 8px;">32% Progrès global</p>
                    </div>
                """, unsafe_allow_html=True)
            # ... (le reste du contenu reste similaire avec remplacement des couleurs par des variables CSS)

    # ... (les autres sections restent inchangées mais utilisent désormais les variables CSS)
