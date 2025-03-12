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
        st.markdown('<div class="profile-box">', unsafe_allow_html=True)
        st.image(profile_image, width=150)
        st.markdown(
            f"<h3 style='color: white; margin: 15px 0;'>{st.session_state.current_user}</h3>", 
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
        
        menu_options = ["Tableau de bord", "Missions", "Rapports"] if st.session_state.role == "directions" else ["Missions", "Rapports"]
        st.session_state.page_option = st.radio("Navigation", menu_options, label_visibility="collapsed")

    # Contenu principal
    st.markdown(
        f"<h1 style='color: var(--primary); border-bottom: 2px solid var(--secondary); padding-bottom: 0.5rem;'>{st.session_state.page_option}</h1>", 
        unsafe_allow_html=True
    )
    
    if st.session_state.page_option == "Tableau de bord":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>📊 Statistiques</h3>
                    <p>15 Nouvelles missions</p>
                    <div style="background: #f0f0f0; border-radius: 8px; height: 8px;">
                        <div style="background: var(--secondary); width: 32%; height: 100%; border-radius: 8px;"></div>
                    </div>
                    <p style="margin-top: 8px;">32% Progrès global</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>✅ Réalisations</h3>
                    <p>98% Taux de résolution</p>
                    <div style="display: flex; gap: 4px; margin-top: 12px;">
                        <span style="color: #ffd700;">★</span>
                        <span style="color: #ffd700;">★</span>
                        <span style="color: #ffd700;">★</span>
                        <span style="color: #ffd700;">★</span>
                        <span style="color: #ffd700;">☆</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>📅 Calendrier</h3>
                    <p>3 Échéances</p>
                    <p>2 Réunions</p>
                </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📈 Graphiques détaillés", expanded=True):
            st.line_chart({"Données": [1, 3, 2, 4, 5, 2, 4]}, height=300)
    
    elif st.session_state.page_option == "Missions":
        tab1, tab2 = st.tabs(["📋 Liste des missions", "➕ Création"])
        
        with tab1:
            st.dataframe(
                data={
                    "Mission": ["Inspection électrique", "Contrôle sécurité", "Audit réseau"],
                    "Statut": ["🟡 En cours", "🟢 Terminé", "🔴 En attente"],
                    "Progrès": [45, 100, 10],
                    "Échéance": ["2024-03-15", "2024-03-10", "2024-04-01"]
                },
                use_container_width=True,
                height=300
            )
        
        with tab2:
            with st.form("Nouvelle mission"):
                col1, col2 = st.columns(2)
                with col1:
                    titre = st.text_input("Titre de la mission")
                    date_echeance = st.date_input("Date d'échéance")
                with col2:
                    priorite = st.selectbox("Priorité", ["Haute", "Moyenne", "Basse"])
                    responsable = st.text_input("Responsable")
                description = st.text_area("Description", height=100)
                
                if st.form_submit_button("Créer mission 🚀"):
                    st.success("Mission créée avec succès!")
    
    elif st.session_state.page_option == "Rapports":
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                with st.form("rapport_form"):
                    date_debut = st.date_input("Date de début")
                    date_fin = st.date_input("Date de fin")
                    format_rapport = st.selectbox("Format", ["PDF", "Excel", "HTML"])
                    
                    if st.form_submit_button("Générer le rapport 🖨️"):
                        st.toast("Génération du rapport en cours...")
            
            with col2:
                st.markdown("""
                    <div style='background: white; 
                                padding: 2rem; 
                                border-radius: 12px; 
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h3 style='color: var(--primary);'>Aperçu du rapport</h3>
                        <p style='color: #666;'>Données du rapport généré...</p>
                    </div>
                """, unsafe_allow_html=True)

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
