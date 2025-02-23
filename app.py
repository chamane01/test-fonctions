import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Annotation d'images sur une carte")
st.write("Téléversez des images, naviguez entre elles et ajoutez des marqueurs avec attribution de classe et gravité.")

# 1. Téléversement des images
uploaded_files = st.file_uploader("Téléverser des images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Charger toutes les images téléversées
    images = [Image.open(file) for file in uploaded_files]
    
    # 2. Sélection de l'image à afficher (avec slider ou flèches)
    image_index = st.slider("Choisissez l'image", 1, len(images), 1) - 1
    current_image = images[image_index]
    st.write(f"Affichage de l'image {image_index+1} sur {len(images)}")
    
    # 3. Choix de la classe et de la gravité dans la barre latérale
    st.sidebar.header("Attribution de classe et gravité")
    classe = st.sidebar.selectbox("Classe", [f"Classe {i}" for i in range(1, 14)])
    gravite = st.sidebar.selectbox("Gravité", [1, 2, 3])
    
    # Initialiser les variables de session pour stocker les marqueurs et la clé du canvas
    if "markers" not in st.session_state:
        st.session_state.markers = {}  # Dictionnaire pour sauvegarder les marqueurs par image
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    st.write("Cliquez sur l'image pour ajouter un marqueur (un seul marqueur par action).")

    # Affichage du canvas avec l'image en fond
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # couleur de remplissage du marqueur
        stroke_width=5,
        stroke_color="#ff0000",
        background_image=current_image,
        update_streamlit=True,
        height=current_image.height,
        width=current_image.width,
        drawing_mode="point",  # mode « point » pour ajouter un marqueur
        key=f"canvas_{st.session_state.canvas_key}"
    )
    
    # Bouton pour enregistrer le marqueur ajouté
    if st.button("Enregistrer le marqueur"):
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                # On prend ici le dernier marqueur ajouté
                marker_obj = objects[-1]
                x = marker_obj.get("left")
                y = marker_obj.get("top")
                marker_data = {"x": x, "y": y, "classe": classe, "gravite": gravite}
                image_key = f"image_{image_index}"
                if image_key not in st.session_state.markers:
                    st.session_state.markers[image_key] = []
                st.session_state.markers[image_key].append(marker_data)
                st.success("Marqueur enregistré !")
                # Incrémente la clé pour réinitialiser le canvas (efface le marqueur actuel)
                st.session_state.canvas_key += 1
            else:
                st.warning("Aucun marqueur détecté. Veuillez cliquer sur l'image pour ajouter un marqueur.")
        else:
            st.warning("Aucun dessin détecté sur le canvas.")
    
    # Affichage des marqueurs enregistrés pour l'image courante
    image_key = f"image_{image_index}"
    if image_key in st.session_state.markers:
        st.write("Marqueurs enregistrés pour cette image :")
        for marker in st.session_state.markers[image_key]:
            st.write(marker)
else:
    st.info("Veuillez téléverser au moins une image.")
