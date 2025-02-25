import numpy as np
import laspy
import scipy.ndimage as ndimage

# Supposons que 'las' contient le nuage chargé via laspy
points = np.vstack((las.x, las.y, las.z)).T

# Construction d'un MNT simple (comme dans l'exemple précédent)
# [Calcul du MNT par grille, puis interpolation pour obtenir ground_levels pour chaque point]
# ground_levels = ...

# Calcul de la hauteur normalisée
normalized_heights = points[:, 2] - ground_levels

# Calcul d'une mesure de variance locale (exemple simplifié par cellule de grille)
# Vous pouvez adapter en calculant l'écart-type sur un voisinage autour de chaque point.
local_variance = np.zeros_like(normalized_heights)
# (Ici, un calcul par cellule de la grille ou en utilisant un filtre glissant)

# Initialisation du tableau de classification
classifications = np.full(points.shape[0], 1, dtype=np.uint8)  # 1 = non classifié

for i in range(points.shape[0]):
    h = normalized_heights[i]
    var = local_variance[i]
    
    # Sol
    if h < 0.5:
        classifications[i] = 2
    # Végétation faible
    elif h < 2.0:
        classifications[i] = 3
    # Possibilité d'arbre ou de bâtiment
    elif h < 20:
        if var < 0.3:
            # Surface plane, potentiellement un bâtiment
            classifications[i] = 6
        elif var > 0.5:
            # Forte irrégularité => peut correspondre à une canopée d'arbre
            classifications[i] = 5
        else:
            # Zone intermédiaire
            classifications[i] = 4
    # Points très élevés
    else:
        # On pourrait aussi vérifier ici une possible détection de lignes électriques en cherchant l'alignement
        classifications[i] = 1  # Non classifié par défaut

# Pour les lignes électriques, vous pourriez isoler les points avec h entre 8 et 15 m,
# puis appliquer une analyse de clustering pour détecter des structures linéaires.
# Par exemple, après clustering, pour un cluster dont l'extension en largeur est très faible,
# réaffecter la classification à 18 (code pour ligne électrique).
