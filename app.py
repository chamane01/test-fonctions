import streamlit as st

# Configuration CSS personnalisée
st.markdown("""
    <style>
        :root {
            --primary: #2C3E50;
            --secondary: #3498DB;
            --accent: #2980B9;
        }
        
        .profile-frame {
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            padding: 4px;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            margin: 0 auto;
        }
        
        .profile-frame:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, var(--primary) 60%, var(--secondary) 100%);
            color: white;
            padding: 1rem;
        }
        
        .stButton>button {
            background: var(--secondary) !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px 24px;
            transition: all 0.3s ease;
            border: none !important;
        }
        
        .stButton>button:hover {
            background: var(--accent) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .stTextInput>div>div>input {
            border-radius: 8px !important;
            padding: 12px !important;
            border: 1px solid #ddd !important;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .dataframe {
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        
        .login-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
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
            st.markdown("<h1 style='text-align: center; color: var(--primary); margin-bottom: 0.5rem;'>Ubuntu Détect</h1>", 
                       unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: var(--secondary); font-size: 1.2rem; margin-bottom: 2rem;'>L'Esprit d'Humanité dans la Détection des Défauts</div>", 
                       unsafe_allow_html=True)
            
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
        st.markdown(f"""
            <div style='text-align: center; padding: 20px 0;'>
                <div class="profile-frame">
                    <img src="{profile_image}" 
                         style="width:100px; height:100px; 
                                border-radius:50%; 
                                object-fit:cover;
                                border: 2px solid white;"/>
                </div>
                <h3 style='margin: 15px 0; color: white;'>{st.session_state.current_user}</h3>
                <div style='background: var(--accent); 
                            padding: 6px; 
                            border-radius: 8px;
                            margin-bottom: 30px;
                            color: white;'>
                    {st.session_state.role.capitalize()}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Déconnexion 🚪", use_container_width=True):
            logout()
        
        menu_options = ["Tableau de bord", "Missions", "Rapports"] if st.session_state.role == "directions" else ["Missions", "Rapports"]
        st.session_state.page_option = st.radio("Navigation", menu_options, label_visibility="collapsed")

    # Contenu principal
    st.markdown(f"<h1 style='color: var(--primary); border-bottom: 2px solid var(--secondary); padding-bottom: 0.5rem;'>{st.session_state.page_option}</h1>", 
               unsafe_allow_html=True)
    
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
