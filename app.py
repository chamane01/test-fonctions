import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt

def apply_smrf(points, cell_size=1.0, slope=0.2, window=18, elevation_threshold=0.5):
    """
    Pseudo-code d'un filtre SMRF :
    1. Créer une grille 2D à partir des points (X, Y).
    2. Calculer la hauteur minimale par cellule pour obtenir une surface initiale.
    3. Appliquer une opération morphologique (opening) avec le paramètre 'window'.
    4. Identifier les points hors-sol dont la hauteur est > (surface + elevation_threshold),
       en tenant compte du 'slope' pour les terrains en pente.
    5. Retourne un masque (True = sol, False = hors-sol).
    
    Note : Cette implémentation est simplifiée et retourne ici un masque fictif.
    """
    # Exemple simplifié : ici, on considère tous les points comme hors-sol.
    # Dans une implémentation réelle, le filtre SMRF serait appliqué pour distinguer sol et hors-sol.
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

        # Paramètres SMRF
        cell_size = st.slider("Taille de cellule SMRF (m)", 0.5, 5.0, 1.0, 0.5)
        slope = st.slider("Slope (tolérance de pente)", 0.0, 1.0, 0.2, 0.05)
        window = st.slider("Window (morphological opening)", 5, 30, 18, 1)
        elevation_threshold = st.slider("Elevation threshold", 0.1, 2.0, 0.5, 0.1)

        if st.button("Lancer le traitement"):
            with st.spinner("Application du filtre SMRF..."):
                ground_mask = apply_smrf(points, cell_size, slope, window, elevation_threshold)

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
