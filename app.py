import streamlit as st
import laspy
import numpy as np
import scipy.ndimage as ndimage
import tempfile
import os

st.title("Classification Lidar sans PDAL")

# Téléchargement du fichier LAS/LAZ
uploaded_file = st.file_uploader("Téléchargez votre fichier LAS/LAZ", type=["las", "laz"])
if uploaded_file is not None:
    # Sauvegarde temporaire du fichier téléchargé
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp:
        tmp.write(uploaded_file.read())
        input_filename = tmp.name

    st.write("Fichier téléchargé:", input_filename)
    
    # Lecture du nuage de points avec laspy
    las = laspy.read(input_filename)
    points = np.vstack((las.x, las.y, las.z)).T
    st.write("Nombre de points :", points.shape[0])
    
    # Paramètres de la grille et seuil de classification
    grid_resolution = st.number_input("Résolution de la grille (m)", value=1.0)
    threshold = st.number_input("Seuil de différence (m)", value=0.5)

    # Détermination des dimensions de la grille
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    nx = int(np.ceil((x_max - x_min) / grid_resolution))
    ny = int(np.ceil((y_max - y_min) / grid_resolution))
    
    # Initialisation de la grille pour stocker le minimum de z (estimation du terrain)
    ground_model = np.full((ny, nx), np.nan)
    ix = ((points[:, 0] - x_min) / grid_resolution).astype(int)
    iy = ((points[:, 1] - y_min) / grid_resolution).astype(int)
    
    # Calcul du minimum de z par cellule de la grille
    for i in range(points.shape[0]):
        row, col = iy[i], ix[i]
        z = points[i, 2]
        if np.isnan(ground_model[row, col]):
            ground_model[row, col] = z
        else:
            ground_model[row, col] = min(ground_model[row, col], z)
    
    # Remplacer les cellules vides par la valeur maximale trouvée
    ground_model = np.where(np.isnan(ground_model), np.nanmax(ground_model), ground_model)
    
    # Application d'un filtre minimum pour lisser le modèle de terrain
    ground_smoothed = ndimage.minimum_filter(ground_model, size=3)
    
    # Interpolation simple (voisin le plus proche) du modèle de terrain pour chaque point
    ground_levels = ground_smoothed[iy, ix]
    
    # Classification : on considère un point comme sol si sa hauteur est inférieure au MNT + seuil
    # On attribue la valeur 2 pour le sol et 1 pour non-sol (valeurs conventionnelles)
    classifications = np.where((points[:, 2] - ground_levels) < threshold, 2, 1)
    
    # Mise à jour de la classification dans le fichier LAS
    las.classification = classifications.astype(np.uint8)
    
    # Sauvegarde du fichier classifié
    output_file = "classified_output.las"
    las.write(output_file)
    
    st.success("Classification effectuée avec succès!")
    with open(output_file, "rb") as f:
        st.download_button("Télécharger le fichier classifié", f, file_name="classified_output.las")
