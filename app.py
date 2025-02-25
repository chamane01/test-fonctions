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
    # Exemple simplifié : ici, on considère tous les points comme hors-sol.
    ground_mask = np.full(len(points), False, dtype=bool)
    return ground_mask

def main():
    st.title("Détection SMRF : Classification Sol vs Hors-sol")
    
    uploaded_file = st.file_uploader("Fichier LAS/LAZ", type=["las", "laz"])
    if uploaded_file is not None:
        las = laspy.read(uploaded_file)
        x, y, z = las.x, las.y, las.z
        points = np.vstack((x, y, z)).T

        st.write(f"Total points: {len(points)}")

        # Sélection de l'objet à détecter
        object_type = st.selectbox(
            "Sélectionnez l'objet à détecter", 
            ["Bâtiments 🏢", "Basse végétation 🌱", "Arbustes 🌿", "Arbres 🌳", "Lignes électriques ⚡", "Cours d’eau 🌊"]
        )

        # Définition des paramètres par défaut selon l'objet
        if object_type.startswith("Bâtiments"):
            default_cell_size = 2.0   # entre 1 et 3 m
            default_window = 25       # entre 20 et 30 m
            default_slope = 10        # entre 5° et 15°
            default_elevation = 3.5   # entre 2 et 5 m
            default_iteration = 1     # 1 à 2
            st.info("Conseil : Un filtre de surface plane peut être appliqué en post-traitement.")
        elif object_type.startswith("Basse"):
            default_cell_size = 0.75  # entre 0.5 et 1 m
            default_window = 7        # entre 5 et 10 m
            default_slope = 4         # entre 2° et 7°
            default_elevation = 0.6   # entre 0.2 et 1 m
            default_iteration = 1
        elif object_type.startswith("Arbustes"):
            default_cell_size = 1.5   # entre 1 et 2 m
            default_window = 11     # entre 8 et 15 m
            default_slope = 7         # entre 5° et 10°
            default_elevation = 2     # entre 1 et 3 m
            default_iteration = 1     # 1 à 2
        elif object_type.startswith("Arbres"):
            default_cell_size = 3.5   # entre 2 et 5 m
            default_window = 30       # entre 20 et 40 m
            default_slope = 15        # entre 10° et 20°
            default_elevation = 12.5  # entre 5 et 20 m
            default_iteration = 2     # 2 à 3
            st.info("Conseil : Un post-traitement basé sur la canopée et la densité de points peut améliorer la détection.")
        elif object_type.startswith("Lignes"):
            default_cell_size = 0.75  # entre 0.5 et 1 m
            default_window = 12       # entre 10 et 15 m
            default_slope = 22        # entre 15° et 30°
            default_elevation = 30    # entre 10 et 50 m
            default_iteration = 1     # 1 à 2
            st.info("Conseil : Un post-traitement basé sur la détection de structures filaires peut être utile.")
        elif object_type.startswith("Cours"):
            default_cell_size = 2.0   # entre 1 et 3 m
            default_window = 15       # entre 10 et 20 m
            default_slope = 3         # entre 2° et 5°
            default_elevation = -0.5  # entre -2 et 1 m (zones en contrebas)
            default_iteration = 1
            st.info("Conseil : L’intégration d’un Modèle Numérique de Terrain (MNT) peut améliorer la détection.")

        st.write("### Réglage des paramètres SMRF pour l'objet sélectionné")
        cell_size = st.slider("Cell Size (m)", min_value=0.1, max_value=10.0, value=default_cell_size, step=0.1)
        window = st.slider("Window Size (m)", min_value=5, max_value=50, value=default_window, step=1)
        slope = st.slider("Slope Threshold (°)", min_value=0, max_value=45, value=default_slope, step=1)
        elevation_threshold = st.slider("Elevation Threshold (m)", min_value=-5.0, max_value=100.0, value=default_elevation, step=0.1)
        iterations = st.slider("Nombre d'itérations", min_value=1, max_value=5, value=default_iteration, step=1)

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
