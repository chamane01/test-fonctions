import streamlit as st
from io import BytesIO
import base64
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import random, math

# --- Patch pour définir st.image_to_url si nécessaire ---
if not hasattr(st, "image_to_url"):
    def image_to_url(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    st.image_to_url = image_to_url

# ----------------------------
# Fonction de calcul des métriques
# ----------------------------
def calculate_metrics_for_object(obj):
    metrics = {}
    obj_type = obj.get("type")
    
    if obj_type == "line":
        points = obj.get("path", [])
        if points:
            distance = 0.0
            for i in range(1, len(points)):
                dx = points[i][1] - points[i-1][1]
                dy = points[i][2] - points[i-1][2]
                distance += math.sqrt(dx * dx + dy * dy)
            metrics["longueur"] = round(distance, 2)

    elif obj_type == "rect":
        width = obj.get("width", 0)
        height = obj.get("height", 0)
        area = width * height
        metrics["surface"] = round(area, 2)
        metrics["périmètre"] = round(2 * (width + height), 2)

    elif obj_type == "circle":
        diameter = obj.get("width", 0)
        radius = diameter / 2
        area = math.pi * radius * radius
        metrics["surface"] = round(area, 2)
        metrics["circonférence"] = round(2 * math.pi * radius, 2)

    elif obj_type == "polygon":
        points = obj.get("path", [])
        if points and len(points) >= 3:
            area = 0
            n = len(points)
            for i in range(n):
                x1, y1 = points[i][1], points[i][2]
                x2, y2 = points[(i+1) % n][1], points[(i+1) % n][2]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2
            metrics["surface"] = round(area, 2)
    return metrics

# ----------------------------
# Configuration et présentation de l'application
# ----------------------------
st.title("Carnet de Dessin Personnel")

# --- Sidebar pour les options générales ---
st.sidebar.header("Options de dessin")
drawing_mode = st.sidebar.selectbox("Outil de dessin", ["freedraw", "line", "rect", "circle", "transform"])
stroke_width = st.sidebar.slider("Épaisseur du trait", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Couleur du trait", "#000000")
bg_color = st.sidebar.color_picker("Couleur de fond", "#FFFFFF")
realtime_update = st.sidebar.checkbox("Mise à jour en temps réel", True)

# Téléversement d'une image de fond
uploaded_file = st.sidebar.file_uploader("Téléversez une image de fond", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    bg_image = Image.open(uploaded_file)
else:
    bg_image = None

# --- Création du canvas de dessin ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Couleur de remplissage pour les outils (ex: polygone)
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=bg_image,
    update_streamlit=realtime_update,
    height=500,
    width=700,
    drawing_mode=drawing_mode,
    key="canvas",
)

# ----------------------------
# Affichage des métriques des dessins réalisés
# ----------------------------
st.subheader("Métriques des objets dessinés")
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        for i, obj in enumerate(objects):
            st.markdown(f"**Objet {i+1}** — Type : `{obj.get('type')}`")
            metrics = calculate_metrics_for_object(obj)
            st.write(metrics)
    else:
        st.write("Aucun objet dessiné pour l'instant.")

# ----------------------------
# Génération aléatoire d'entités géométriques
# ----------------------------
st.subheader("Génération aléatoire d'entités")

st.markdown("Définissez la zone d'emprise (en pixels) dans laquelle les entités seront générées.")
col1, col2, col3, col4 = st.columns(4)
x_min = col1.number_input("X min", min_value=0, value=50)
y_min = col2.number_input("Y min", min_value=0, value=50)
x_max = col3.number_input("X max", min_value=0, value=300)
y_max = col4.number_input("Y max", min_value=0, value=300)

num_entities = st.number_input("Nombre d'entités à générer", min_value=1, value=5)
entity_type = st.selectbox("Type d'entité", ["rectangle", "carré", "cercle", "polygone"])

if st.button("Générer entités"):
    if bg_image is not None:
        image = bg_image.copy()
    else:
        image = Image.new("RGB", (700, 500), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    generated_entities = []

    for i in range(num_entities):
        if entity_type == "rectangle":
            x1 = random.randint(x_min, x_max)
            y1 = random.randint(y_min, y_max)
            x2 = random.randint(x1, x_max)
            y2 = random.randint(y1, y_max)
            draw.rectangle([x1, y1, x2, y2], outline=stroke_color, width=stroke_width)
            area = abs(x2 - x1) * abs(y2 - y1)
            generated_entities.append({
                "type": "rectangle",
                "coords": [x1, y1, x2, y2],
                "surface": area
            })

        elif entity_type == "carré":
            side_max = min(x_max - x_min, y_max - y_min)
            side = random.randint(10, side_max if side_max > 10 else 10)
            x1 = random.randint(x_min, x_max - side)
            y1 = random.randint(y_min, y_max - side)
            x2 = x1 + side
            y2 = y1 + side
            draw.rectangle([x1, y1, x2, y2], outline=stroke_color, width=stroke_width)
            area = side * side
            generated_entities.append({
                "type": "carré",
                "coords": [x1, y1, x2, y2],
                "surface": area
            })

        elif entity_type == "cercle":
            max_diameter = min(x_max - x_min, y_max - y_min)
            diameter = random.randint(10, max_diameter if max_diameter > 10 else 10)
            x1 = random.randint(x_min, x_max - diameter)
            y1 = random.randint(y_min, y_max - diameter)
            x2 = x1 + diameter
            y2 = y1 + diameter
            draw.ellipse([x1, y1, x2, y2], outline=stroke_color, width=stroke_width)
            area = math.pi * (diameter / 2) ** 2
            generated_entities.append({
                "type": "cercle",
                "coords": [x1, y1, x2, y2],
                "surface": round(area, 2)
            })

        elif entity_type == "polygone":
            num_points = 5
            points = []
            for j in range(num_points):
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                points.append((x, y))
            draw.polygon(points, outline=stroke_color)
            area = 0
            n = len(points)
            for j in range(n):
                x1p, y1p = points[j]
                x2p, y2p = points[(j + 1) % n]
                area += x1p * y2p - x2p * y1p
            area = abs(area) / 2
            generated_entities.append({
                "type": "polygone",
                "points": points,
                "surface": round(area, 2)
            })

    st.image(image, caption="Image avec entités générées", use_column_width=True)
    st.markdown("### Détails des entités générées")
    st.write(generated_entities)
