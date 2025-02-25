import numpy as np
import streamlit as st
from sklearn.cluster import DBSCAN

def main():
    st.title("Clustering des points candidats avec DBSCAN")

    # Chargement des données (remplace par tes propres données)
    # Exemple fictif : N points avec (x, y, z)
    candidate_points = np.random.rand(100, 3)  # Supposons 100 points aléatoires

    # Extraire uniquement les coordonnées X et Y
    candidate_points_xy = candidate_points[:, :2]

    # Vérifier si le tableau est vide
    if candidate_points_xy.shape[0] == 0:
        st.warning("Aucun point candidat n'a été trouvé après filtrage. Vérifiez vos seuils (z_min, z_max, etc.).")
        return  # Arrête l'exécution ici

    # Vérifier si le tableau a la bonne forme
    if candidate_points_xy.ndim != 2 or candidate_points_xy.shape[1] != 2:
        st.error("Les données doivent être un tableau 2D avec deux colonnes (x, y).")
        return

    # Vérifier s'il y a des NaN ou des valeurs infinies
    if np.isnan(candidate_points_xy).any() or np.isinf(candidate_points_xy).any():
        st.error("Les données contiennent des NaN ou des valeurs infinies. Nettoyez-les avant DBSCAN.")
        return

    # Paramètres DBSCAN (à ajuster selon tes besoins)
    eps = 0.1  # Distance maximale entre deux points voisins
    min_samples = 3  # Nombre minimal de points pour former un cluster

    # Appliquer DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(candidate_points_xy)

    # Afficher les résultats
    st.write(f"Nombre de clusters trouvés (hors bruit) : {len(set(labels)) - (1 if -1 in labels else 0)}")
    st.write("Labels des clusters :", labels)

if __name__ == "__main__":
    main()
