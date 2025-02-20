import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw

# --- Fonction pour dessiner une grille sur une image ---
def draw_grid(image, grid_cols=10, grid_rows=10, line_color=(255, 0, 0), line_width=1):
    """
    Dessine une grille sur l'image.
    La grille divise l'image en grid_cols colonnes et grid_rows lignes.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # Lignes verticales
    for i in range(1, grid_cols):
        x = i * width / grid_cols
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
    # Lignes horizontales
    for j in range(1, grid_rows):
        y = j * height / grid_rows
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
    return img

# --- Initialisation de l'état de session pour stocker les marqueurs ---
if 'markers' not in st.session_state:
    # Dictionnaire : clé = index de l'image, valeur = liste de marqueurs
    st.session_state.markers = {}

st.title("Application de Marquage d'Images")

# 1. Téléverser des images
uploaded_files = st.file_uploader("Téléverser des images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(image)
    
    # 2. Navigation entre images via un slider dans la barre latérale
    image_index = st.sidebar.slider("Image à afficher", 1, len(images), 1) - 1  # passage en index 0-based
    current_image = images[image_index]
    
    # Création de l'image avec grille (ici, une grille 10x10)
    grid_image = draw_grid(current_image, grid_cols=10, grid_rows=10)
    st.sidebar.write(f"Affichage de l'image {image_index+1} sur {len(images)}")
    
    # 3. Sélection de la classe et de la gravité
    selected_class = st.sidebar.selectbox("Sélectionner la classe", [f"Classe {i}" for i in range(1, 14)])
    selected_gravity = st.sidebar.selectbox("Sélectionner la gravité", [1, 2, 3])
    
    st.write("**Instructions :** Cliquez sur l'image pour placer un marqueur (point). Puis cliquez sur le bouton **Enregistrer marqueur** pour sauvegarder le marqueur avec la classe et la gravité choisies.")
    
    # Canvas interactif pour dessiner le marqueur sur l'image avec grille en fond.
    # Nous utilisons ici le mode "point" pour simuler un clic unique.
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)",  # couleur du marqueur
        stroke_width=10,                   # largeur (taille) du marqueur
        stroke_color="#0000FF",
        background_image=grid_image,
        update_streamlit=True,
        height=grid_image.height,
        width=grid_image.width,
        drawing_mode="point",              # mode de dessin : placement d'un point
        key="canvas",
    )
    
    # Bouton pour enregistrer le marqueur après avoir cliqué sur l'image
    if st.button("Enregistrer marqueur"):
        if canvas_result.json_data is not None:
            shapes = canvas_result.json_data["objects"]
            if shapes:
                # On récupère le dernier marqueur dessiné
                marker = shapes[-1]
                # Les coordonnées locales du marqueur (pixels sur l'image)
                x = marker["left"]
                y = marker["top"]
                # Calcul de la position dans la grille (par exemple, cellule 1-indexée)
                width, height = current_image.size
                cell_width = width / 10
                cell_height = height / 10
                col = int(x // cell_width) + 1
                row = int(y // cell_height) + 1
                
                # Constitution des informations du marqueur
                marker_data = {
                    "x": x,
                    "y": y,
                    "colonne": col,
                    "ligne": row,
                    "classe": selected_class,
                    "gravite": selected_gravity,
                }
                
                # Sauvegarde du marqueur dans l'état de session pour l'image courante
                if image_index not in st.session_state.markers:
                    st.session_state.markers[image_index] = []
                st.session_state.markers[image_index].append(marker_data)
                
                st.success(f"Marqueur enregistré : {marker_data}")
            else:
                st.warning("Aucun marqueur détecté sur le canvas.")
        else:
            st.warning("Veuillez dessiner un marqueur sur le canvas avant de l'enregistrer.")
    
    # Affichage des marqueurs enregistrés pour l'image actuelle
    st.subheader("Marqueurs enregistrés pour cette image")
    if image_index in st.session_state.markers and st.session_state.markers[image_index]:
        for idx, m in enumerate(st.session_state.markers[image_index], start=1):
            st.write(f"{idx}. Coordonnées : ({m['x']:.1f}, {m['y']:.1f}) — Grille : (Col {m['colonne']}, Ligne {m['ligne']}) — {m['classe']} / Gravité {m['gravite']}")
    else:
        st.write("Aucun marqueur enregistré pour cette image.")
    
    # Optionnel : Afficher l'ensemble des marqueurs pour toutes les images
    if st.checkbox("Afficher tous les marqueurs enregistrés"):
        st.write(st.session_state.markers)
else:
    st.info("Veuillez téléverser des images pour démarrer.")
