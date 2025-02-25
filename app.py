import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt

def apply_smrf(points, cell_size=0.5, window_size=10, elevation_threshold=0.5, slope_threshold_deg=15):
    """
    Pseudo-code d'un filtre SMRF adapté à la détection de lignes fines :
    
    Paramètres :
      - cell_size : résolution spatiale pour la rasterisation (0.5–1 m)
      - window_size : taille de la fenêtre morphologique (10–20 m)
      - elevation_threshold : différence de hauteur maximale pour le filtrage (0.5–1 m)
      - slope_threshold_deg : pente maximale considérée comme sol (15–30°)
      - Initial Elevation : méthode d'initialisation fixée à "Minimum" (pour terrains variés)
      
    Processus (simplifié) :
      1. Créer une grille 2D à partir des coordonnées (X, Y).
      2. Calculer la hauteur minimale par cellule pour obtenir une surface de référence.
      3. Appliquer une opération morphologique (opening) avec la fenêtre spécifiée.
      4. Identifier les points dont l'altitude dépasse (surface + elevation_threshold),
         en tenant compte d'une tolérance de pente (convertie ici depuis les degrés en gradient).
      5. Retourner un masque booléen indiquant True pour les points considérés comme sol,
         et False pour les points hors-sol (potentiellement des lignes électriques).
         
    Remarque : Cette implémentation est une version simplifiée pour illustrer le flux de traitement.
    """
    # Pour l'exemple, nous simulons le filtrage SMRF :
    # On considère comme sol les points ayant une altitude inférieure ou égale à
    # la valeur minimale relevée plus un offset (elevation_threshold) ajusté par la pente.
    z = points[:, 2]
    # Calcul de l'altitude minimale
    z_min = np.min(z)
    # Conversion de la pente en gradient (approximatif : tan(angle en radians))
    slope_threshold = np.tan(np.deg2rad(slope_threshold_deg))
    # Calcul d'un seuil local simple (simulation) : altitude minimale + elevation_threshold ajusté
    threshold = z_min + elevation_threshold * (1 + slope_threshold)
    # On considère comme sol les points dont l'altitude est inférieure ou égale à ce seuil
    ground_mask = z <= threshold
    return ground_mask

def main():
    st.title("Détection des lignes électriques via SMRF")

    uploaded_file = st.file_uploader("Fichier LAS/LAZ", type=["las", "laz"])
    if uploaded_file is not None:
        las = laspy.read(uploaded_file)
        x, y, z = las.x, las.y, las.z
        points = np.vstack((x, y, z)).T

        st.write(f"Total points : {len(points)}")

        st.subheader("Paramètres SMRF recommandés")
        cell_size = st.slider("Cell Size (m)", 0.5, 1.0, 0.5, 0.1)
        window_size = st.slider("Window Size (m)", 10, 20, 10, 1)
        elevation_threshold = st.slider("Elevation Threshold (m)", 0.5, 1.0, 0.5, 0.1)
        slope_threshold_deg = st.slider("Slope Threshold (°)", 15, 30, 15, 1)
        st.info("Initial Elevation : Méthode 'Minimum' (fixé)")

        # Paramètres pour la détection des lignes électriques par filtrage en hauteur
        st.subheader("Filtrage post-traitement pour lignes électriques")
        z_min_line = st.slider("Hauteur minimum (m)", 0.0, 20.0, 5.0, 0.5)
        z_max_line = st.slider("Hauteur maximum (m)", 20.0, 50.0, 30.0, 0.5)

        if st.button("Lancer le traitement"):
            with st.spinner("Application du filtre SMRF..."):
                ground_mask = apply_smrf(points, cell_size, window_size, elevation_threshold, slope_threshold_deg)

            # Séparation des points en sol et hors-sol
            ground_points = points[ground_mask]
            non_ground_points = points[~ground_mask]

            st.write(f"Points classifiés comme sol : {len(ground_points)}")
            st.write(f"Points classifiés comme hors-sol : {len(non_ground_points)}")

            # Filtrage pour isoler les lignes électriques (points entre 5 m et 30 m)
            if len(non_ground_points) > 0:
                mask_lines = (non_ground_points[:, 2] >= z_min_line) & (non_ground_points[:, 2] <= z_max_line)
                candidate_line_points = non_ground_points[mask_lines]
                st.write(f"Points candidats pour lignes électriques : {len(candidate_line_points)}")
            else:
                candidate_line_points = np.empty((0, 3))
                st.write("Aucun point hors-sol détecté.")

            # Visualisation
            fig, ax = plt.subplots(figsize=(8, 6))
            if len(ground_points) > 0:
                ax.scatter(ground_points[:, 0], ground_points[:, 1], s=1, color='green', label='Sol')
            if len(candidate_line_points) > 0:
                ax.scatter(candidate_line_points[:, 0], candidate_line_points[:, 1], s=1, color='red', label='Lignes électriques')
            ax.set_title("Détection des lignes électriques via SMRF")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
