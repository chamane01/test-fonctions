import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import random
import math

# Titre de l'application
st.title("Carnet de dessin personnel")

st.markdown("""
Ce carnet vous permet de dessiner avec divers outils, de téléverser une image de fond et d’ajouter des entités aléatoires dans une zone définie.  
Vous pouvez également visualiser quelques mesures (longueur, aire, périmètre) des objets dessinés.
""")

# -----------------------------
# 1. Configuration dans la barre latérale
# -----------------------------
st.sidebar.header("Paramètres de dessin")

# Téléversement d'une image
uploaded_file = st.sidebar.file_uploader("Téléversez une photo pour dessiner", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Image téléversée", use_column_width=True)
else:
    image = None

# Choix du mode de dessin
drawing_mode = st.sidebar.selectbox("Mode de dessin", 
                                    ("freedraw", "line", "rect", "circle", "transform"),
                                    help="Sélectionnez l'outil de dessin")

# Réglages des couleurs et de l'épaisseur
stroke_width = st.sidebar.slider("Épaisseur du trait", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Couleur du trait", "#000000")
fill_color = st.sidebar.color_picker("Couleur de remplissage", "#ffffff")

st.sidebar.markdown("---")
st.sidebar.header("Génération d'entités aléatoires")

# Sélection du type d'entité
entity_type = st.sidebar.selectbox("Type d'entité", 
                                   ("polygon", "rectangle", "carré", "cercle"),
                                   help="Choisissez le type d'entité à générer")

# Nombre d'entités à générer
num_entities = st.sidebar.number_input("Nombre d'entités", min_value=1, max_value=100, value=1, step=1)

# Définition de la zone (emprise) dans laquelle les entités seront générées
st.sidebar.subheader("Définir la zone de génération")
# Vous pouvez définir la taille de votre canevas (par défaut ou via une image)
canvas_width = st.sidebar.number_input("Largeur du canevas", min_value=100, max_value=1000, value=500, step=10)
canvas_height = st.sidebar.number_input("Hauteur du canevas", min_value=100, max_value=1000, value=500, step=10)

# Zone de génération (parmi le canevas)
zone_x = st.sidebar.number_input("X de la zone", min_value=0, max_value=canvas_width, value=0, step=10)
zone_y = st.sidebar.number_input("Y de la zone", min_value=0, max_value=canvas_height, value=0, step=10)
zone_width = st.sidebar.number_input("Largeur de la zone", min_value=10, max_value=canvas_width, value=canvas_width, step=10)
zone_height = st.sidebar.number_input("Hauteur de la zone", min_value=10, max_value=canvas_height, value=canvas_height, step=10)

# Bouton pour générer les entités
if st.sidebar.button("Générer entités aléatoires"):
    shapes = []
    for i in range(num_entities):
        if entity_type in ["rectangle", "carré"]:
            # Position aléatoire dans la zone
            x0 = random.randint(zone_x, max(zone_x, zone_x + zone_width - 10))
            y0 = random.randint(zone_y, max(zone_y, zone_y + zone_height - 10))
            # Pour rectangle : largeur et hauteur aléatoires ; pour carré : même taille
            if entity_type == "rectangle":
                max_width = max(10, zone_x + zone_width - x0)
                max_height = max(10, zone_y + zone_height - y0)
                width = random.randint(10, max_width)
                height = random.randint(10, max_height)
            else:  # carré
                max_size = min(zone_x + zone_width - x0, zone_y + zone_height - y0)
                size = random.randint(10, max_size)
                width = size
                height = size
            # Création de l'objet rectangle (le type "rect" est utilisé)
            shape_obj = {
                "type": "rect",
                "version": "4.6.0",
                "originX": "left",
                "originY": "top",
                "left": x0,
                "top": y0,
                "width": width,
                "height": height,
                "fill": fill_color,
                "stroke": stroke_color,
                "strokeWidth": stroke_width,
                "opacity": 1,
                "angle": 0,
            }
            shapes.append(shape_obj)
        elif entity_type == "cercle":
            # Rayon aléatoire
            max_radius = min(zone_width, zone_height) // 2
            if max_radius < 10:
                max_radius = 10
            radius = random.randint(10, max_radius)
            # Centre aléatoire en veillant à ce que le cercle soit dans la zone
            cx = random.randint(zone_x + radius, zone_x + zone_width - radius)
            cy = random.randint(zone_y + radius, zone_y + zone_height - radius)
            shape_obj = {
                "type": "circle",
                "version": "4.6.0",
                "originX": "center",
                "originY": "center",
                "left": cx,
                "top": cy,
                "radius": radius,
                "fill": fill_color,
                "stroke": stroke_color,
                "strokeWidth": stroke_width,
                "opacity": 1,
                "angle": 0,
            }
            shapes.append(shape_obj)
        elif entity_type == "polygon":
            # Génération d'un polygone avec un nombre de points aléatoire entre 3 et 6
            num_points = random.randint(3, 6)
            points = []
            for j in range(num_points):
                px = random.randint(zone_x, zone_x + zone_width)
                py = random.randint(zone_y, zone_y + zone_height)
                points.append([px, py])
            shape_obj = {
                "type": "polygon",
                "version": "4.6.0",
                "originX": "left",
                "originY": "top",
                "left": 0,  # non utilisé directement quand on spécifie les points
                "top": 0,
                "points": points,
                "fill": fill_color,
                "stroke": stroke_color,
                "strokeWidth": stroke_width,
                "opacity": 1,
                "angle": 0,
            }
            shapes.append(shape_obj)
    # Enregistrer la liste des formes générées dans la session
    st.session_state["random_shapes"] = shapes

# Si des entités aléatoires ont été générées, on les ajoute comme dessin initial
if "random_shapes" in st.session_state:
    initial_drawing = {"objects": st.session_state["random_shapes"], "background": None}
else:
    initial_drawing = None

# -----------------------------
# 2. Création du canevas de dessin
# -----------------------------
st.markdown("## Zone de dessin")

# Si une image est téléversée, on l'utilise comme fond et on adapte la taille du canevas
if image:
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=image,
        background_color="#eee",
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode=drawing_mode,
        initial_drawing=initial_drawing,
        key="canvas",
    )
else:
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#eee",
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        initial_drawing=initial_drawing,
        key="canvas",
    )

# -----------------------------
# 3. Affichage et mesures des objets dessinés
# -----------------------------
if canvas_result.json_data is not None:
    objects = canvas_result.json_data["objects"]
    st.write("### Objets dessinés")
    st.json(objects)

    st.write("### Mesures des objets")
    for obj in objects:
        # Pour les lignes (si disponibles)
        if obj["type"] == "line":
            # Selon la configuration, une ligne peut posséder des coordonnées de début et fin
            try:
                x1 = obj["x1"]
                y1 = obj["y1"]
                x2 = obj["x2"]
                y2 = obj["y2"]
            except KeyError:
                continue
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            st.write(f"- **Ligne** : longueur = {length:.2f} pixels")
        # Pour les rectangles (et carrés)
        elif obj["type"] == "rect":
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            area = width * height
            perimeter = 2 * (width + height)
            st.write(f"- **Rectangle/Carré** : largeur = {width}, hauteur = {height}, aire = {area}, périmètre = {perimeter}")
        # Pour les cercles
        elif obj["type"] == "circle":
            radius = obj.get("radius", 0)
            area = math.pi * (radius ** 2)
            circumference = 2 * math.pi * radius
            st.write(f"- **Cercle** : rayon = {radius}, aire = {area:.2f}, circonférence = {circumference:.2f}")
        # Pour les polygones (calcul par la formule de Gauss)
        elif obj["type"] == "polygon":
            points = obj.get("points", [])
            if len(points) >= 3:
                area = 0
                n = len(points)
                for i in range(n):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % n]
                    area += x1 * y2 - x2 * y1
                area = abs(area) / 2
                st.write(f"- **Polygone** : {len(points)} points, aire ≈ {area:.2f} pixels²")
            else:
                st.write("- **Polygone** : moins de 3 points (non calculable)")
        else:
            st.write(f"- **Objet de type {obj['type']}** : mesures non définies")
