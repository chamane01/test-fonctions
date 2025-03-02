import streamlit as st
from datetime import date, datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib import colors

# Configuration de la page
st.set_page_config(page_title="Générateur Structuré", layout="centered")

# Dimensions standard
PAGE_WIDTH, PAGE_HEIGHT = A4
SECTION_HEIGHT = PAGE_HEIGHT / 3
COLUMN_WIDTH = PAGE_WIDTH / 2

def create_element_controller():
    with st.expander("➕ Ajouter un élément", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            elem_type = st.selectbox("Type", ["Image", "Texte"], key="elem_type")
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="elem_size")
        with col2:
            vertical_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="v_pos")
            horizontal_pos = st.selectbox(
                "Position horizontale",
                ["Gauche", "Droite", "Centre"] if size == "Petit" else ["Gauche", "Droite"],
                key="h_pos"
            )
        
        if elem_type == "Image":
            content = st.file_uploader("Contenu (image)", type=["png", "jpg", "jpeg"], key="content_image")
            image_title = st.text_input("Titre de l'image", max_chars=50, key="image_title")
            description = st.text_input("Description brève (max 100 caractères)", max_chars=100, key="image_desc")
        else:
            content = st.text_area("Contenu", key="content_text")
        
        if st.button("Valider l'élément"):
            if elem_type == "Image" and content is None:
                st.error("Veuillez charger une image pour cet élément.")
                return None
            element_data = {
                "type": elem_type,
                "size": size,
                "v_pos": vertical_pos,
                "h_pos": horizontal_pos,
                "content": content,
            }
            if elem_type == "Image":
                element_data["image_title"] = image_title
                element_data["description"] = description
            return element_data
    return None

def calculate_dimensions(size):
    dimensions = {
        "Grand": (PAGE_WIDTH, SECTION_HEIGHT),
        "Moyen": (COLUMN_WIDTH, SECTION_HEIGHT),
        "Petit": (COLUMN_WIDTH / 1.5, SECTION_HEIGHT)
    }
    return dimensions.get(size, (PAGE_WIDTH, SECTION_HEIGHT))

def calculate_position(element):
    vertical_offset = {"Haut": 0, "Milieu": SECTION_HEIGHT, "Bas": SECTION_HEIGHT*2}[element['v_pos']]
    
    if element['size'] == "Grand":
        return (0, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)
    
    if element['h_pos'] == "Gauche":
        x = 0
    elif element['h_pos'] == "Droite":
        x = COLUMN_WIDTH
    else:  # Centre
        x = COLUMN_WIDTH / 2 - calculate_dimensions(element['size'])[0] / 2
    
    return (x, PAGE_HEIGHT - vertical_offset - SECTION_HEIGHT)

def draw_metadata(c, metadata):
    margin = 40
    x_left = margin
    y_top = PAGE_HEIGHT - margin
    line_height = 16

    logo_drawn = False
    if metadata['logo']:
        try:
            img = ImageReader(metadata['logo'])
            img_width, img_height = img.getSize()
            aspect = img_height / img_width
            desired_width = 40
            desired_height = desired_width * aspect
            
            c.drawImage(img, x_left, y_top - desired_height, width=desired_width, height=desired_height, preserveAspectRatio=True, mask='auto')
            logo_drawn = True
        except Exception as e:
            st.error(f"Erreur de chargement du logo: {str(e)}")
    
    if logo_drawn:
        x_title = x_left + 50
        y_title = y_top - 20
    else:
        x_title = x_left
        y_title = y_top - 20
    
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(colors.black)
    if metadata.get('titre'):
        c.drawString(x_title, y_title, metadata['titre'])
    
    c.setFont("Helvetica", 14)
    y_company = y_title - 25
    if metadata.get('company'):
        c.drawString(x_title, y_company, metadata['company'])
    
    y_line = y_company - 10
    c.setStrokeColor(colors.darkgray)
    c.setLineWidth(2)
    c.line(x_left, y_line, x_left + 150, y_line)
    c.setLineWidth(1)
    
    y_text = y_line - 20
    infos = [
        ("ID Rapport", metadata['report_id']),
        ("Date", metadata['date'].strftime('%d/%m/%Y') if hasattr(metadata['date'], "strftime") else metadata['date']),
        ("Heure", metadata['time'].strftime('%H:%M') if hasattr(metadata['time'], "strftime") else metadata['time']),
        ("Éditeur", metadata['editor']),
        ("Localisation", metadata['location'])
    ]
    
    value_x_offset = x_left + 70
    for label, value in infos:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.black)
        c.drawString(x_left, y_text, label + ":")
        c.setFont("Helvetica", 10)
        c.drawString(value_x_offset, y_text, str(value))
        y_text -= line_height

def generate_pdf(elements, metadata):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    c.setAuthor(metadata['editor'])
    c.setTitle(metadata['report_id'])
    
    for element in elements:
        width, height = calculate_dimensions(element['size'])
        x, y = calculate_position(element)
        
        if element['type'] == "Image":
            if element["content"] is not None:
                try:
                    img = ImageReader(element["content"])
                    # Réserver des marges pour le titre et la description
                    top_margin = 20
                    bottom_margin = 20
                    # Réduire l'image pour laisser de l'espace en haut et en bas
                    horizontal_scale = 0.9  # réduction horizontale à 90%
                    image_actual_width = width * horizontal_scale
                    image_actual_height = height - top_margin - bottom_margin
                    # Centrer l'image dans l'aire allouée
                    image_x = x + (width - image_actual_width) / 2
                    image_y = y + bottom_margin
                    c.drawImage(img, image_x, image_y, width=image_actual_width, height=image_actual_height, preserveAspectRatio=True, mask='auto')
                    
                    # Afficher le titre en haut, centré, dans la marge supérieure (en dehors de l'image) en majuscules
                    if element.get("image_title"):
                        c.setFont("Helvetica-Bold", 12)
                        image_title = element["image_title"].upper()
                        c.drawCentredString(x + width / 2, y + height - top_margin / 2, image_title)
                    
                    # Afficher la description en bas à droite, en gris, dans la marge inférieure (en dehors de l'image)
                    if element.get("description"):
                        c.setFont("Helvetica", 10)
                        c.setFillColor(colors.gray)
                        c.drawRightString(x + width - 10, y + bottom_margin / 2, element["description"][:100])
                        c.setFillColor(colors.black)
                except Exception as e:
                    st.error(f"Erreur d'image: {str(e)}")
            else:
                st.error("Une image validée est introuvable.")
        else:
            text = element['content']
            style = getSampleStyleSheet()["Normal"]
            style.fontSize = 14 if element['size'] == "Grand" else 12 if element['size'] == "Moyen" else 10
            p = Paragraph(text, style)
            p.wrapOn(c, width, height)
            p.drawOn(c, x, y)
    
    draw_metadata(c, metadata)
    
    c.save()
    buffer.seek(0)
    return buffer

def display_elements_preview(elements):
    st.markdown("## Aperçu des éléments validés")
    for idx, element in enumerate(elements, start=1):
        st.markdown(f"**Élément {idx}**")
        if element["type"] == "Image":
            # Affichage réduit de l'image (largeur fixe)
            st.image(element["content"], width=200)
            if element.get("image_title"):
                st.markdown(f"*Titre de l'image :* **{element['image_title'].upper()}**")
            if element.get("description"):
                st.markdown(
                    f"<span style='color:gray'>*Description :* {element['description']}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"**Texte :** {element['content']}")
        st.markdown("---")

def main():
    st.title("📐 Conception de Rapport Structuré")
    
    with st.sidebar:
        st.header("📝 Métadonnées")
        titre = st.text_input("Titre principal")
        report_id = st.text_input("ID du rapport")
        report_date = st.date_input("Date du rapport", date.today())
        report_time = st.time_input("Heure du rapport", datetime.now().time())
        editor = st.text_input("Éditeur")
        location = st.text_input("Localisation")
        company = st.text_input("Société")
        logo = st.file_uploader("Logo", type=["png", "jpg", "jpeg"])
    
    metadata = {
        'titre': titre,
        'report_id': report_id,
        'date': report_date,
        'time': report_time,
        'editor': editor,
        'location': location,
        'company': company,
        'logo': logo
    }
    
    if "elements" not in st.session_state:
        st.session_state["elements"] = []
    elements = st.session_state["elements"]
    
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
        st.session_state["elements"] = elements
        st.success("Élément validé avec succès !")
    
    if elements:
        display_elements_preview(elements)
    
    if elements:
        if st.button("Générer le PDF"):
            pdf = generate_pdf(elements, metadata)
            st.success("✅ Rapport généré avec succès!")
            st.download_button("Télécharger le PDF", pdf, "rapport_structuré.pdf", "application/pdf")

if __name__ == "__main__":
    main()
