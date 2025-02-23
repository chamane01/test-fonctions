import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Application de Marquage d'Images")
st.write("Téléversez vos images, naviguez entre elles et placez des marqueurs avec classe et gravité.")

# -------------------------------
# 1. Téléversement des images
# -------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Téléverser des images (png, jpg, jpeg)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Si aucune image n'est téléversée, on affiche un message d'instruction
if not uploaded_files:
    st.info("Veuillez téléverser au moins une image.")
    st.stop()

# Charger les images avec PIL
images = [Image.open(file) for file in uploaded_files]

# -------------------------------
# 2. Navigation entre les images
# -------------------------------
img_index = st.slider("Sélectionner l'image", 0, len(images) - 1, 0)
current_image = images[img_index]

# Affichage de quelques infos sur l'image (dimensions)
st.write(f"Dimensions de l'image : {current_image.width} x {current_image.height}")

# -------------------------------
# 3. Sélection par défaut de la classe et gravité
# -------------------------------
st.sidebar.markdown("### Paramètres par défaut pour les marqueurs")
default_class = st.sidebar.selectbox(
    "Classe par défaut",
    options=[f"Classe {i+1}" for i in range(13)]
)
default_severity = st.sidebar.selectbox(
    "Gravité par défaut",
    options=[1, 2, 3]
)

# -------------------------------
# 4. Affichage de l'image avec un canvas interactif
# -------------------------------
st.markdown("#### Placez vos marqueurs sur l'image")
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",  # couleur de remplissage pour les marqueurs
    stroke_width=5,
    stroke_color="red",
    background_image=current_image,
    update_streamlit=True,
    height=current_image.height,
    width=current_image.width,
    drawing_mode="point",  # mode point pour cliquer et placer un marqueur
    key="canvas"
)

# -------------------------------
# 5. Traitement des marqueurs et attribution des classes/gravités
# -------------------------------
markers = []

if canvas_result.json_data is not None:
    # Récupérer la liste des objets dessinés
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        st.markdown("#### Liste des marqueurs placés")
        for idx, obj in enumerate(objects):
            # Récupérer la position absolue du marqueur
            x_abs = obj.get("left", 0)
            y_abs = obj.get("top", 0)
            # Calcul des coordonnées relatives par rapport à l'image (entre 0 et 1)
            x_rel = x_abs / current_image.width
            y_rel = y_abs / current_image.height

            st.write(f"**Marqueur {idx+1}** : Position absolue ({x_abs:.1f}, {y_abs:.1f}) - Relative ({x_rel:.2f}, {y_rel:.2f})")

            # Option 1 : Attribution après avoir placé le marqueur
            col1, col2 = st.columns(2)
            with col1:
                marker_class = st.selectbox(
                    f"Classe pour le marqueur {idx+1}",
                    options=[f"Classe {i+1}" for i in range(13)],
                    index=int(default_class.split()[1]) - 1,
                    key=f"class_{idx}"
                )
            with col2:
                marker_severity = st.selectbox(
                    f"Gravité pour le marqueur {idx+1}",
                    options=[1, 2, 3],
                    index=[1, 2, 3].index(default_severity),
                    key=f"severity_{idx}"
                )

            # Enregistrer les informations du marqueur
            markers.append({
                "index": idx,
                "position_absolue": {"x": x_abs, "y": y_abs},
                "position_relative": {"x": x_rel, "y": y_rel},
                "classe": marker_class,
                "gravite": marker_severity
            })

        st.markdown("#### Données des marqueurs")
        st.json(markers)
    else:
        st.info("Aucun marqueur placé pour le moment.")

# Vous pouvez par la suite enregistrer les données (ex : dans un fichier ou une base de données)
