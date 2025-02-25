import streamlit as st 
import laspy
import numpy as np
import matplotlib.pyplot as plt

def apply_smrf(points, cell_size, slope, window, elevation_threshold, iterations):
    """
    Pseudo-code d'un filtre SMRF :
    1. Créer une grille 2D à partir des points (X, Y).
    2. Calculer la hauteur minimale par cellule pour obtenir une surface initiale.
    3. Appliquer une opération morphologique (opening) avec la taille de fenêtre 'window'
       et itérer 'iterations' fois.
    4. Identifier les points hors-sol dont la hauteur est > (surface + elevation_threshold),
       en tenant compte du 'slope' pour les terrains en pente.
    5. Retourne un masque (True = sol, False = hors-sol).
    
    Note : Cette implémentation est simplifiée et retourne ici un masque fictif.
    """
    ground_mask = np.full(len(points), False, dtype=bool)
    return ground_mask

def main():
    st.title("Détection SMRF : Classification Sol vs Hors-sol")
    
    uploaded_file = st.file_uploader("Fichier LAS/LAZ", type=["las", "laz"])
    if uploaded_file is not None:
        las = laspy.read(uploaded_file)
        x, y, z = las.x, las.y, las.z
        points = np.vstack((x, y, z)).T

        st.write(f"Total points : {len(points)}")
        
        # Sélection de l'objet à traiter (les paramètres seront choisis en fonction)
        st.sidebar.header("Choix de l'objet à traiter")
        selected_object = st.sidebar.radio(
            "Objet",
            ["Bâtiments 🏢", "Basse végétation 🌱", "Arbustes 🌿", "Arbres 🌳", "Lignes électriques ⚡", "Cours d’eau 🌊"]
        )

        st.sidebar.header("Paramètres par objet")
        # Bâtiments
        with st.sidebar.expander("Bâtiments 🏢", expanded=False):
            b_cell_size = st.slider("Cell Size (m)", 0.1, 10.0, 2.0, step=0.1, key="b_cell_size")
            b_window = st.slider("Window Size (m)", 5, 50, 25, step=1, key="b_window")
            b_slope = st.slider("Slope Threshold (°)", 0, 45, 10, step=1, key="b_slope")
            b_elevation = st.slider("Elevation Threshold (m)", -5.0, 100.0, 3.5, step=0.1, key="b_elevation")
            b_iterations = st.slider("Nombre d'itérations", 1, 5, 1, step=1, key="b_iterations")
            st.info("Conseil : Un filtre de surface plane peut être appliqué en post-traitement.")

        # Basse végétation
        with st.sidebar.expander("Basse végétation 🌱", expanded=False):
            lv_cell_size = st.slider("Cell Size (m)", 0.1, 10.0, 0.75, step=0.1, key="lv_cell_size")
            lv_window = st.slider("Window Size (m)", 5, 50, 7, step=1, key="lv_window")
            lv_slope = st.slider("Slope Threshold (°)", 0, 45, 4, step=1, key="lv_slope")
            lv_elevation = st.slider("Elevation Threshold (m)", -5.0, 100.0, 0.6, step=0.1, key="lv_elevation")
            lv_iterations = st.slider("Nombre d'itérations", 1, 5, 1, step=1, key="lv_iterations")

        # Arbustes
        with st.sidebar.expander("Arbustes 🌿", expanded=False):
            arb_cell_size = st.slider("Cell Size (m)", 0.1, 10.0, 1.5, step=0.1, key="arb_cell_size")
            arb_window = st.slider("Window Size (m)", 5, 50, 11, step=1, key="arb_window")
            arb_slope = st.slider("Slope Threshold (°)", 0, 45, 7, step=1, key="arb_slope")
            arb_elevation = st.slider("Elevation Threshold (m)", -5.0, 100.0, 2, step=0.1, key="arb_elevation")
            arb_iterations = st.slider("Nombre d'itérations", 1, 5, 1, step=1, key="arb_iterations")

        # Arbres
        with st.sidebar.expander("Arbres 🌳", expanded=False):
            a_cell_size = st.slider("Cell Size (m)", 0.1, 10.0, 3.5, step=0.1, key="a_cell_size")
            a_window = st.slider("Window Size (m)", 5, 50, 30, step=1, key="a_window")
            a_slope = st.slider("Slope Threshold (°)", 0, 45, 15, step=1, key="a_slope")
            a_elevation = st.slider("Elevation Threshold (m)", -5.0, 100.0, 12.5, step=0.1, key="a_elevation")
            a_iterations = st.slider("Nombre d'itérations", 1, 5, 2, step=1, key="a_iterations")
            st.info("Conseil : Un post-traitement basé sur la canopée et la densité de points peut améliorer la détection.")

        # Lignes électriques
        with st.sidebar.expander("Lignes électriques ⚡", expanded=False):
            l_cell_size = st.slider("Cell Size (m)", 0.1, 10.0, 0.75, step=0.1, key="l_cell_size")
            l_window = st.slider("Window Size (m)", 5, 50, 12, step=1, key="l_window")
            l_slope = st.slider("Slope Threshold (°)", 0, 45, 22, step=1, key="l_slope")
            l_elevation = st.slider("Elevation Threshold (m)", -5.0, 100.0, 30, step=0.1, key="l_elevation")
            l_iterations = st.slider("Nombre d'itérations", 1, 5, 1, step=1, key="l_iterations")
            st.info("Conseil : Un post-traitement basé sur la détection de structures filaires peut être utile.")

        # Cours d’eau
        with st.sidebar.expander("Cours d’eau 🌊", expanded=False):
            c_cell_size = st.slider("Cell Size (m)", 0.1, 10.0, 2.0, step=0.1, key="c_cell_size")
            c_window = st.slider("Window Size (m)", 5, 50, 15, step=1, key="c_window")
            c_slope = st.slider("Slope Threshold (°)", 0, 45, 3, step=1, key="c_slope")
            c_elevation = st.slider("Elevation Threshold (m)", -5.0, 100.0, -0.5, step=0.1, key="c_elevation")
            c_iterations = st.slider("Nombre d'itérations", 1, 5, 1, step=1, key="c_iterations")
            st.info("Conseil : L’intégration d’un MNT peut améliorer la détection.")

        # En fonction de l'objet sélectionné, récupérer les paramètres correspondants
        if selected_object.startswith("Bâtiments"):
            cell_size = b_cell_size
            window = b_window
            slope = b_slope
            elevation_threshold = b_elevation
            iterations = b_iterations
        elif selected_object.startswith("Basse"):
            cell_size = lv_cell_size
            window = lv_window
            slope = lv_slope
            elevation_threshold = lv_elevation
            iterations = lv_iterations
        elif selected_object.startswith("Arbustes"):
            cell_size = arb_cell_size
            window = arb_window
            slope = arb_slope
            elevation_threshold = arb_elevation
            iterations = arb_iterations
        elif selected_object.startswith("Arbres"):
            cell_size = a_cell_size
            window = a_window
            slope = a_slope
            elevation_threshold = a_elevation
            iterations = a_iterations
        elif selected_object.startswith("Lignes"):
            cell_size = l_cell_size
            window = l_window
            slope = l_slope
            elevation_threshold = l_elevation
            iterations = l_iterations
        elif selected_object.startswith("Cours"):
            cell_size = c_cell_size
            window = c_window
            slope = c_slope
            elevation_threshold = c_elevation
            iterations = c_iterations

        if st.button("Lancer le traitement"):
            with st.spinner("Application du filtre SMRF..."):
                ground_mask = apply_smrf(points, cell_size, slope, window, elevation_threshold, iterations)
            
            # Séparation des points selon le masque SMRF
            ground_points = points[ground_mask]
            non_ground_points = points[~ground_mask]

            st.write(f"Points classifiés comme sol : {len(ground_points)}")
            st.write(f"Points classifiés comme hors-sol : {len(non_ground_points)}")

            # Visualisation des résultats
            fig, ax = plt.subplots(figsize=(8, 6))
            if len(ground_points) > 0:
                ax.scatter(ground_points[:, 0], ground_points[:, 1], s=1, color='green', label='Sol')
            if len(non_ground_points) > 0:
                ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], s=1, color='red', label='Hors-sol')
            ax.set_title("Résultats de la classification SMRF")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
