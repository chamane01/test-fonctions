import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Annotation d'images", layout="wide")

# Titre de l'application
st.title("Annotation d'images sur une carte dynamique")

# Instructions dans la sidebar
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Téléversez vos images.
2. Naviguez entre elles avec les boutons.
3. Cliquez sur le canvas pour placer un marqueur (petit cercle).
4. Attribuez une classe (13 classes) et une gravité (1, 2, 3) au marqueur.
Les coordonnées sont calculées localement à partir de l'image.
""")

# Téléversement des images
uploaded_files = st.file_uploader("Téléverser des images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Charger les images avec Pillow
    images = [Image.open(file).convert("RGB") for file in uploaded_files]

    # Initialiser le stockage des annotations dans la session
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {i: [] for i in range(len(images))}
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # Navigation entre images
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("<< Image précédente"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(images)
    with col3:
        if st.button("Image suivante >>"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(images)

    current_index = st.session_state.image_index
    st.subheader(f"Annotation de l'image {current_index + 1} sur {len(images)}")

    # Récupérer l'image courante
    current_image = images[current_index]
    # Définir la taille du canvas selon l'image
    canvas_width, canvas_height = current_image.size

    st.markdown("Cliquez sur l'image pour ajouter un marqueur (cercle rouge).")

    # Afficher le canvas avec l'image en fond
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # couleur de remplissage du marqueur
        stroke_width=3,
        stroke_color="red",
        background_image=current_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="circle",  # mode cercle pour simuler un marqueur
        key="canvas",
        initial_drawing=[],
    )

    # Vérifier si le canvas contient de nouveaux objets dessinés
    if canvas_result.json_data is not None:
        # Chaque objet dessiné se trouve dans canvas_result.json_data["objects"]
        for obj in canvas_result.json_data["objects"]:
            # On vérifie qu'il s'agit bien d'un cercle (marqueur)
            if obj.get("type") == "circle":
                # Calculer le centre du cercle (coordonnées locales)
                x = obj.get("left", 0) + obj.get("radius", 0)
                y = obj.get("top", 0) + obj.get("radius", 0)
                # Vérifier si ce marqueur est déjà enregistré (éviter les doublons)
                if not any(abs(x - ann["x"]) < 5 and abs(y - ann["y"]) < 5 for ann in st.session_state.annotations[current_index]):
                    st.write(f"Marqueur détecté aux coordonnées locales : ({int(x)}, {int(y)})")
                    
                    # Sélection de la classe et de la gravité
                    class_choice = st.selectbox("Sélectionnez la classe", [f"Classe {i}" for i in range(1, 14)])
                    severity_choice = st.selectbox("Sélectionnez la gravité", [1, 2, 3])
                    
                    # Sauvegarde de l'annotation
                    annotation = {
                        "x": int(x),
                        "y": int(y),
                        "classe": class_choice,
                        "gravité": severity_choice
                    }
                    st.session_state.annotations[current_index].append(annotation)
                    st.success(f"Annotation sauvegardée : {annotation}")

    # Afficher les annotations pour l'image courante
    st.markdown("### Annotations enregistrées pour cette image :")
    if st.session_state.annotations[current_index]:
        for idx, ann in enumerate(st.session_state.annotations[current_index], start=1):
            st.write(f"{idx}. Coordonnées : ({ann['x']}, {ann['y']}) - {ann['classe']} - Gravité {ann['gravité']}")
    else:
        st.info("Aucune annotation pour cette image pour le moment.")
else:
    st.info("Veuillez téléverser une ou plusieurs images pour commencer l'annotation.")
