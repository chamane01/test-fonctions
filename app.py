import io, random, math
from PIL import Image
import ipywidgets as widgets
from ipycanvas import Canvas, hold_canvas
from IPython.display import display

# --- Paramètres du canvas ---
canvas_width = 800
canvas_height = 600

# Création du canvas avec une bordure
canvas = Canvas(width=canvas_width, height=canvas_height)
canvas.layout.border = '1px solid black'

# --- Variables globales ---
mode = None          # mode de dessin sélectionné
start_point = None   # point de départ pour les formes à deux clics
polygon_points = []  # liste des points pour le polygone
drawing = False      # indique si l'on est en train de dessiner

# --- Widgets pour l'interface utilisateur ---

# Sélecteur de mode de dessin
mode_selector = widgets.ToggleButtons(
    options=['Libre', 'Ligne', 'Rectangle', 'Cercle', 'Polygone', 'Mesure'],
    description='Mode:',
)

# Bouton pour finaliser un polygone (mode "Polygone")
finir_polygone_btn = widgets.Button(description="Finir Polygone")
finir_polygone_btn.disabled = True

# Label pour afficher la mesure (distance)
label_mesure = widgets.Label("Mesure: ")

# Widget pour téléverser une image (uniquement une image)
upload = widgets.FileUpload(accept='image/*', multiple=False)

# Widgets pour la génération aléatoire d’entités :
# Dropdown pour sélectionner la forme aléatoire
forme_dropdown = widgets.Dropdown(
    options=['Rectangle', 'Carré', 'Cercle', 'Polygone'],
    description='Forme:',
)

# Bouton pour générer une forme aléatoire
generer_forme_btn = widgets.Button(description="Générer forme aléatoire")

# Zone (emprise) dans laquelle générer la forme aléatoire
zone_x = widgets.IntText(value=0, description='X:')
zone_y = widgets.IntText(value=0, description='Y:')
zone_width = widgets.IntText(value=canvas_width, description='Largeur:')
zone_height = widgets.IntText(value=canvas_height, description='Hauteur:')

# --- Gestion du téléversement d'image ---
def on_upload_change(change):
    if upload.value:
        # On récupère le premier fichier téléversé
        for filename, file_info in upload.value.items():
            content = file_info['content']
            # Ouvrir l'image avec PIL
            image = Image.open(io.BytesIO(content))
            # Redimensionner l'image pour occuper tout le canvas
            image = image.resize((canvas_width, canvas_height))
            # Effacer le canvas et dessiner l'image en fond
            canvas.clear()
            canvas.draw_image(image, 0, 0)
            break

upload.observe(on_upload_change, names='value')

# --- Gestion du choix de mode de dessin ---
def set_mode(change):
    global mode, polygon_points, start_point
    mode = change['new']
    start_point = None
    polygon_points = []
    if mode == 'Polygone':
        finir_polygone_btn.disabled = False
    else:
        finir_polygone_btn.disabled = True

mode_selector.observe(set_mode, names='value')

# --- Fonction pour finir et dessiner un polygone ---
def finir_polygone(b):
    global polygon_points
    if len(polygon_points) > 2:
        with hold_canvas(canvas):
            canvas.stroke_style = 'blue'
            canvas.line_width = 2
            canvas.begin_path()
            canvas.move_to(*polygon_points[0])
            for pt in polygon_points[1:]:
                canvas.line_to(*pt)
            canvas.close_path()
            canvas.stroke()
    polygon_points = []

finir_polygone_btn.on_click(finir_polygone)

# --- Gestion des événements de la souris sur le canvas ---
def handle_mouse_down(x, y):
    global start_point, drawing, polygon_points
    if mode in ['Ligne', 'Rectangle', 'Cercle', 'Mesure']:
        start_point = (x, y)
        drawing = True
    elif mode == 'Libre':
        drawing = True
        start_point = (x, y)
    elif mode == 'Polygone':
        polygon_points.append((x, y))
        # Marquer le point cliqué (petit cercle rouge)
        canvas.fill_style = 'red'
        canvas.fill_circle(x, y, 3)

def handle_mouse_move(x, y):
    global start_point, drawing
    # En mode "Libre", dessiner en traçant des segments pendant le déplacement
    if drawing and mode == 'Libre':
        with hold_canvas(canvas):
            canvas.stroke_style = 'black'
            canvas.line_width = 2
            canvas.begin_path()
            canvas.move_to(*start_point)
            canvas.line_to(x, y)
            canvas.stroke()
        start_point = (x, y)
    # Pour d'autres modes, on pourrait ajouter un aperçu dynamique (non implémenté ici)

def handle_mouse_up(x, y):
    global start_point, drawing
    if drawing and mode == 'Ligne':
        with hold_canvas(canvas):
            canvas.stroke_style = 'black'
            canvas.line_width = 2
            canvas.begin_path()
            canvas.move_to(*start_point)
            canvas.line_to(x, y)
            canvas.stroke()
    elif drawing and mode == 'Rectangle':
        with hold_canvas(canvas):
            canvas.stroke_style = 'green'
            canvas.line_width = 2
            x0, y0 = start_point
            largeur = x - x0
            hauteur = y - y0
            canvas.stroke_rect(x0, y0, largeur, hauteur)
    elif drawing and mode == 'Cercle':
        with hold_canvas(canvas):
            canvas.stroke_style = 'purple'
            canvas.line_width = 2
            x0, y0 = start_point
            rayon = math.sqrt((x - x0)**2 + (y - y0)**2)
            canvas.begin_path()
            canvas.arc(x0, y0, rayon, 0, 2*math.pi)
            canvas.stroke()
    elif drawing and mode == 'Mesure':
        with hold_canvas(canvas):
            canvas.stroke_style = 'orange'
            canvas.line_width = 2
            canvas.begin_path()
            canvas.move_to(*start_point)
            canvas.line_to(x, y)
            canvas.stroke()
        distance = math.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
        label_mesure.value = f"Mesure: {distance:.2f} pixels"
    drawing = False
    start_point = None

canvas.on_mouse_down(handle_mouse_down)
canvas.on_mouse_move(handle_mouse_move)
canvas.on_mouse_up(handle_mouse_up)

# --- Génération aléatoire d'entités dans une zone donnée ---
def generer_forme(b):
    forme = forme_dropdown.value
    # Récupérer les coordonnées et dimensions de la zone
    x0 = zone_x.value
    y0 = zone_y.value
    w = zone_width.value
    h = zone_height.value
    # Générer un point aléatoire dans la zone
    rand_x = random.randint(x0, x0 + w)
    rand_y = random.randint(y0, y0 + h)
    with hold_canvas(canvas):
        if forme == 'Rectangle':
            rect_w = random.randint(20, max(20, w//2))
            rect_h = random.randint(20, max(20, h//2))
            canvas.stroke_style = 'brown'
            canvas.line_width = 2
            canvas.stroke_rect(rand_x, rand_y, rect_w, rect_h)
        elif forme == 'Carré':
            side = random.randint(20, max(20, min(w, h)//2))
            canvas.stroke_style = 'brown'
            canvas.line_width = 2
            canvas.stroke_rect(rand_x, rand_y, side, side)
        elif forme == 'Cercle':
            radius = random.randint(10, max(10, min(w, h)//4))
            canvas.stroke_style = 'brown'
            canvas.line_width = 2
            canvas.begin_path()
            canvas.arc(rand_x, rand_y, radius, 0, 2*math.pi)
            canvas.stroke()
        elif forme == 'Polygone':
            # Générer un polygone régulier avec 5 à 8 côtés
            n = random.randint(5, 8)
            angle = 2 * math.pi / n
            rayon = random.randint(20, max(20, min(w, h)//4))
            points = []
            for i in range(n):
                theta = i * angle
                px = rand_x + rayon * math.cos(theta)
                py = rand_y + rayon * math.sin(theta)
                points.append((px, py))
            canvas.stroke_style = 'brown'
            canvas.line_width = 2
            canvas.begin_path()
            canvas.move_to(*points[0])
            for pt in points[1:]:
                canvas.line_to(*pt)
            canvas.close_path()
            canvas.stroke()

generer_forme_btn.on_click(generer_forme)

# --- Assemblage de l'interface utilisateur ---
ui = widgets.VBox([
    widgets.HBox([mode_selector, finir_polygone_btn, label_mesure]),
    canvas,
    widgets.Label("Téléverser une image (pour dessiner par-dessus) :"),
    upload,
    widgets.HBox([forme_dropdown, generer_forme_btn]),
    widgets.HBox([zone_x, zone_y, zone_width, zone_height])
])

display(ui)
