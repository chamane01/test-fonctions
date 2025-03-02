import streamlit as st
import pandas as pd
import json
import altair as alt
import plotly.express as px
import pydeck as pdk
from datetime import datetime, date, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from io import BytesIO
import calendar
import matplotlib.pyplot as plt

# Configuration générale de la page et CSS personnalisée
st.set_page_config(page_title="Suivi des Dégradations sur Routes Ivoiriennes", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
        color: #333;
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stMetric {
        background: linear-gradient(90deg, #8e44ad, #3498db);
        color: #fff;
        padding: 10px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True
)

# Menu latéral principal
menu_option = st.sidebar.radio("Menu", ["Tableau de bord", "Rapport"])

# Chargement des données (commun aux deux sections)
with open("jeu_donnees_missions (1).json", "r", encoding="utf-8") as f:
    data = json.load(f)
missions_df = pd.DataFrame(data)
missions_df['date'] = pd.to_datetime(missions_df['date'])

# Extraction des défauts
defects = []
for mission in data:
    for defect in mission.get("Données Défauts", []):
        defects.append(defect)
df_defects = pd.DataFrame(defects)
if 'date' in df_defects.columns:
    df_defects['date'] = pd.to_datetime(df_defects['date'])

#############################################
# Fonction de génération du PDF pour le rapport
#############################################
def generate_report_pdf(report_type, missions_df, df_defects, metadata, start_date, end_date):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    PAGE_WIDTH, PAGE_HEIGHT = A4
    margin = 40

    # En-tête avec le logo (même logo que pour le tableau de bord)
    logo_path = "images (5).png"
    try:
        img = ImageReader(logo_path)
        c.drawImage(img, margin, PAGE_HEIGHT - margin - 50, width=50, height=50, preserveAspectRatio=True, mask='auto')
    except Exception as e:
        st.error(f"Erreur de chargement du logo: {str(e)}")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin + 60, PAGE_HEIGHT - margin - 20, metadata.get("titre", "Rapport de Suivi"))
    c.setFont("Helvetica", 12)
    c.drawString(margin + 60, PAGE_HEIGHT - margin - 40, f"Type: {report_type} | Période: {start_date} au {end_date}")
    c.drawString(margin + 60, PAGE_HEIGHT - margin - 60, f"Éditeur: {metadata.get('editor', 'Inconnu')}")
    
    # Calcul des indicateurs pour la période sélectionnée
    filtered_missions = missions_df[(missions_df['date'].dt.date >= start_date) & (missions_df['date'].dt.date <= end_date)]
    num_missions = len(filtered_missions)
    total_distance = filtered_missions["distance(km)"].sum() if "distance(km)" in filtered_missions.columns else 0
    avg_distance = filtered_missions["distance(km)"].mean() if ("distance(km)" in filtered_missions.columns and num_missions > 0) else 0

    if not df_defects.empty and 'date' in df_defects.columns:
        filtered_defects = df_defects[(df_defects['date'].dt.date >= start_date) & (df_defects['date'].dt.date <= end_date)]
    else:
        filtered_defects = df_defects
    total_defects = len(filtered_defects)
    
    # Affichage des indicateurs principaux dans le PDF
    y = PAGE_HEIGHT - margin - 100
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Métriques Principales")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Nombre de Missions: {num_missions}")
    y -= 20
    c.drawString(margin, y, f"Distance Totale (km): {total_distance:.1f}")
    y -= 20
    c.drawString(margin, y, f"Distance Moyenne (km): {avg_distance:.1f}")
    y -= 20
    c.drawString(margin, y, f"Nombre de Défauts: {total_defects}")
    
    # Zone réservée aux graphiques et analyses complémentaires
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Graphiques et Analyses")
    y -= 20
    c.setFont("Helvetica", 10)
    
    # --- Création du premier graphique : Bar chart Missions vs Défauts ---
    fig1, ax1 = plt.subplots(figsize=(4,3))
    categories = ['Missions', 'Défauts']
    values = [num_missions, total_defects]
    bars = ax1.bar(categories, values, color=['#3498db', '#e74c3c'])
    ax1.set_title("Comparaison Missions / Défauts", fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    buf1 = BytesIO()
    plt.savefig(buf1, format='PNG', dpi=100)
    plt.close(fig1)
    buf1.seek(0)
    img_chart1 = ImageReader(buf1)
    
    # --- Création du second graphique : Répartition des Défauts par Catégorie ---
    fig2, ax2 = plt.subplots(figsize=(4,3))
    if not filtered_defects.empty and "classe" in filtered_defects.columns:
        defect_counts = filtered_defects['classe'].value_counts()
        ax2.pie(defect_counts, labels=defect_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
        ax2.set_title("Répartition Défauts par Catégorie", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Pas de données", horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    buf2 = BytesIO()
    plt.savefig(buf2, format='PNG', dpi=100)
    plt.close(fig2)
    buf2.seek(0)
    img_chart2 = ImageReader(buf2)
    
    # Positionnement des deux graphiques côte à côte
    chart_width = (PAGE_WIDTH - 3 * margin) / 2
    chart_height = 200  # Hauteur approximative des graphiques
    # On fixe la position verticale pour les graphiques
    y_chart = y - chart_height - 20
    c.drawImage(img_chart1, margin, y_chart, width=chart_width, height=chart_height, preserveAspectRatio=True, mask='auto')
    c.drawImage(img_chart2, margin + chart_width + margin, y_chart, width=chart_width, height=chart_height, preserveAspectRatio=True, mask='auto')
    
    # Pied de page avec la date de génération
    c.setFont("Helvetica", 8)
    c.drawRightString(PAGE_WIDTH - margin, margin, f"Rapport généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

#############################################
# Affichage selon le menu sélectionné
#############################################
if menu_option == "Tableau de bord":
    # -------------------
    # Partie Tableau de bord
    # -------------------
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("images (5).png", width=200)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.title("Tableau de bord de Suivi des Dégradations sur les Routes Ivoiriennes")
    st.markdown("Parce que nous croyons que la route précède le développement")
    
    if "distance(km)" not in missions_df.columns:
        st.error("Le fichier ne contient pas la colonne 'distance(km)'.")
    else:
        # Indicateurs principaux
        num_missions = len(missions_df)
        total_distance = missions_df["distance(km)"].sum()
        avg_distance = missions_df["distance(km)"].mean()
        total_defects = len(df_defects)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de Missions", num_missions)
        col2.metric("Nombre de Défauts", total_defects)
        col3.metric("Distance Totale (km)", f"{total_distance:.1f}")
        col4.metric("Distance Moyenne (km)", f"{avg_distance:.1f}")
        
        st.markdown("---")
        # Graphiques d'évolution dans le temps
        missions_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M")).size().reset_index(name="Missions")
        missions_over_time['date'] = missions_over_time['date'].dt.to_timestamp()
        chart_missions = alt.Chart(missions_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Missions:Q', title='Nombre de Missions'),
            tooltip=['date:T', 'Missions:Q']
        ).properties(width=350, height=300, title="Évolution des Missions")
        
        defects_over_time = df_defects.groupby(df_defects['date'].dt.to_period("M")).size().reset_index(name="Défauts")
        defects_over_time['date'] = defects_over_time['date'].dt.to_timestamp()
        chart_defects_time = alt.Chart(defects_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Défauts:Q', title='Nombre de Défauts'),
            tooltip=['date:T', 'Défauts:Q']
        ).properties(width=350, height=300, title="Évolution des Défauts")
        
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            st.altair_chart(chart_missions, use_container_width=True)
        with col_time2:
            st.altair_chart(chart_defects_time, use_container_width=True)
        
        st.markdown("---")
        # Graphiques Distance Totale et Score de Sévérité Moyen
        distance_over_time = missions_df.groupby(missions_df['date'].dt.to_period("M"))["distance(km)"].sum().reset_index(name="Distance Totale")
        distance_over_time['date'] = distance_over_time['date'].dt.to_timestamp()
        chart_distance_time = alt.Chart(distance_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Distance Totale:Q', title='Km Totaux', scale=alt.Scale(domain=[0, distance_over_time["Distance Totale"].max()*1.2])),
            tooltip=['date:T', 'Distance Totale:Q']
        ).properties(width=350, height=300, title="Évolution des Km")
        
        gravity_sizes = {1: 5, 2: 7, 3: 9}
        if 'gravite' in df_defects.columns:
            df_defects['severite'] = df_defects['gravite'].map(gravity_sizes)
        severity_over_time = df_defects.groupby(df_defects['date'].dt.to_period("M"))["severite"].mean().reset_index(name="Score Moyen")
        severity_over_time['date'] = severity_over_time['date'].dt.to_timestamp()
        chart_severity_over_time = alt.Chart(severity_over_time).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('Score Moyen:Q', title='Score de Sévérité Moyen'),
            tooltip=['date:T', 'Score Moyen:Q']
        ).properties(width=350, height=300, title="Score de Sévérité Moyen")
        
        col_time3, col_time4 = st.columns(2)
        with col_time3:
            st.altair_chart(chart_distance_time, use_container_width=True)
        with col_time4:
            st.altair_chart(chart_severity_over_time, use_container_width=True)
        
        st.markdown("---")
        # Diagramme circulaire : Répartition Globale des Défauts par Catégorie
        if 'classe' in df_defects.columns:
            defect_category_counts = df_defects['classe'].value_counts().reset_index()
            defect_category_counts.columns = ["Catégorie", "Nombre de Défauts"]
            fig_pie = px.pie(defect_category_counts, values='Nombre de Défauts', names='Catégorie',
                             title="Répartition Globale des Défauts par Catégorie",
                             color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        # Carte des Défauts (Pydeck)
        if 'lat' in df_defects.columns and 'long' in df_defects.columns:
            def get_red_color(gravite):
                if gravite == 1:
                    return [255, 200, 200]
                elif gravite == 2:
                    return [255, 100, 100]
                elif gravite == 3:
                    return [255, 0, 0]
                else:
                    return [255, 0, 0]
            df_defects['marker_color'] = df_defects['gravite'].apply(get_red_color)
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_defects,
                get_position='[long, lat]',
                get_color="marker_color",
                get_radius="severite * 50",
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=df_defects['lat'].mean(),
                longitude=df_defects['long'].mean(),
                zoom=8,
                pitch=0,
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Route: {routes}\nClasse: {classe}\nGravité: {gravite}"}
            )
            st.pydeck_chart(deck)
        
        st.markdown("---")
        # Section Complémentaire : Analyse par Route et par Type de Défaut
        show_all = st.checkbox("Afficher tous les éléments", value=False)
        limit = None if show_all else 7
        
        # Graphique 5 : Nombre de Défauts par Route
        route_defect_counts = df_defects['routes'].value_counts().reset_index()
        route_defect_counts.columns = ["Route", "Nombre de Défauts"]
        display_routes = route_defect_counts if show_all else route_defect_counts.head(limit)
        chart_routes = alt.Chart(display_routes).mark_bar().encode(
            x=alt.X("Route:N", sort='-y', title="Route",
                    axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            tooltip=["Route:N", "Nombre de Défauts:Q"],
            color=alt.Color("Route:N", scale=alt.Scale(scheme='tableau10'))
        ).properties(width=450, height=500, title="Nombre de Défauts par Route")
        
        # Graphique 6 : Score de Sévérité Total par Route
        route_severity = df_defects.groupby('routes')['severite'].sum().reset_index().sort_values(by='severite', ascending=False)
        display_severity = route_severity if show_all else route_severity.head(limit)
        chart_severity = alt.Chart(display_severity).mark_bar().encode(
            x=alt.X("routes:N", sort='-y', title="Route",
                    axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("severite:Q", title="Score de Sévérité Total"),
            tooltip=["routes:N", "severite:Q"],
            color=alt.Color("routes:N", scale=alt.Scale(scheme='tableau20'))
        ).properties(width=450, height=500, title="Score de Sévérité par Route")
        
        col_route, col_severity = st.columns(2)
        with col_route:
            st.altair_chart(chart_routes, use_container_width=True)
        with col_severity:
            st.altair_chart(chart_severity, use_container_width=True)
        
        # Graphique 7 : Analyse interactive par Type de Défaut
        st.markdown("### Analyse par Type de Défaut")
        defect_types = df_defects['classe'].unique()
        selected_defect = st.selectbox("Sélectionnez un type de défaut", defect_types)
        filtered_defects_type = df_defects[df_defects['classe'] == selected_defect]
        if not filtered_defects_type.empty:
            route_count_selected = filtered_defects_type['routes'].value_counts().reset_index()
            route_count_selected.columns = ["Route", "Nombre de Défauts"]
            display_selected = route_count_selected if show_all else route_count_selected.head(limit)
            chart_defect_type = alt.Chart(display_selected).mark_bar().encode(
                x=alt.X("Route:N", sort='-y', title="Route",
                        axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
                y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
                tooltip=["Route:N", "Nombre de Défauts:Q"],
                color=alt.Color("Route:N", scale=alt.Scale(scheme='category20b'))
            ).properties(width=900, height=500, title=f"Répartition des Défauts pour le Type : {selected_defect} (Top 7 par défaut)")
            st.altair_chart(chart_defect_type, use_container_width=True)
        else:
            st.write("Aucune donnée disponible pour ce type de défaut.")
        
        st.markdown("---")
        # Section Analyse par Route
        st.markdown("### Analyse par Route")
        selected_route = st.selectbox("Sélectionnez une route", sorted(df_defects['routes'].unique()))
        inventory = df_defects[df_defects['routes'] == selected_route]['classe'].value_counts().reset_index()
        inventory.columns = ["Dégradation", "Nombre de Défauts"]
        chart_route_inventory = alt.Chart(inventory).mark_bar().encode(
            x=alt.X("Dégradation:N", sort='-y', title="Dégradation",
                    axis=alt.Axis(labelAngle=45, labelOverlap=False, labelLimit=150)),
            y=alt.Y("Nombre de Défauts:Q", title="Nombre de Défauts"),
            tooltip=["Dégradation:N", "Nombre de Défauts:Q"],
            color=alt.Color("Dégradation:N", scale=alt.Scale(scheme='category20b'))
        ).properties(width=900, height=500, title=f"Inventaire des Dégradations pour la Route : {selected_route}")
        st.altair_chart(chart_route_inventory, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

elif menu_option == "Rapport":
    # -------------------
    # Partie Génération de Rapport PDF avec choix de la période
    # -------------------
    st.title("Génération de Rapports")
    report_type = st.selectbox("Sélectionnez le type de rapport", 
                               ["Journalier", "Semaine", "Mensuel", "Annuel", "Général"])
    
    # Sélecteur de période selon le type de rapport
    if report_type == "Journalier":
        selected_day = st.date_input("Sélectionnez le jour", value=date.today(), key="daily_date")
        start_date = selected_day
        end_date = selected_day
    elif report_type == "Semaine":
        selected_week = st.date_input("Sélectionnez une date de la semaine", value=date.today(), key="weekly_date")
        start_date = selected_week - timedelta(days=selected_week.weekday())
        end_date = start_date + timedelta(days=6)
    elif report_type == "Mensuel":
        selected_month = st.date_input("Sélectionnez un mois", value=date.today(), key="monthly_date")
        start_date = selected_month.replace(day=1)
        _, last_day = calendar.monthrange(selected_month.year, selected_month.month)
        end_date = selected_month.replace(day=last_day)
    elif report_type == "Annuel":
        selected_year = st.number_input("Sélectionnez l'année", value=date.today().year, step=1, key="yearly_year")
        start_date = date(int(selected_year), 1, 1)
        end_date = date(int(selected_year), 12, 31)
    else:
        start_date = missions_df['date'].min().date()
        end_date = missions_df['date'].max().date()
    
    # Métadonnées pour le rapport dans la sidebar
    st.sidebar.header("📝 Métadonnées pour le Rapport")
    titre = st.sidebar.text_input("Titre du rapport", "Rapport de Suivi")
    editor = st.sidebar.text_input("Éditeur", "Admin")
    metadata = {"titre": titre, "editor": editor}
    
    if st.button("Générer le Rapport PDF"):
        pdf_buffer = generate_report_pdf(report_type, missions_df, df_defects, metadata, start_date, end_date)
        st.success("✅ Rapport généré avec succès!")
        st.download_button("Télécharger le PDF", data=pdf_buffer, file_name=f"rapport_{report_type}.pdf", mime="application/pdf")
