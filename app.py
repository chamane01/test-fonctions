import streamlit as st
from PIL import Image, ExifTags

st.title("Calcul de l'Empreinte au Sol d'une Photo Aérienne")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléverser une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # Extraction des métadonnées EXIF
    exif_data = {}
    if hasattr(image, '_getexif'):
        exif_raw = image._getexif()
        if exif_raw is not None:
            for tag, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value

    # Récupérer la longueur focale depuis les métadonnées
    focal_length_exif = None
    if 'FocalLength' in exif_data:
        focal = exif_data['FocalLength']
        if isinstance(focal, tuple) and len(focal) == 2:
            focal_length_exif = focal[0] / focal[1]
        else:
            focal_length_exif = float(focal)

    # Récupérer l'altitude GPS depuis les métadonnées
    gps_altitude = None
    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']
        for key in gps_info:
            tag = ExifTags.GPSTAGS.get(key, key)
            if tag == 'GPSAltitude':
                alt_val = gps_info[key]
                if isinstance(alt_val, tuple) and len(alt_val) == 2:
                    gps_altitude = alt_val[0] / alt_val[1]
                else:
                    gps_altitude = float(alt_val)

    st.subheader("Métadonnées extraites")
    st.write("Longueur focale (EXIF) : ", focal_length_exif if focal_length_exif is not None else "Non disponible")
    st.write("Altitude GPS (EXIF) : ", gps_altitude if gps_altitude is not None else "Non disponible")
    st.write("Dimensions de l'image (pixels) : ", image.size)  # image.size retourne (largeur, hauteur)

    st.subheader("Entrer ou vérifier les paramètres nécessaires")
    # Permettre à l'utilisateur de saisir les valeurs (préremplies si disponibles)
    hauteur = st.number_input("Hauteur de vol (m)", value=(gps_altitude if gps_altitude is not None else 100.0))
    focale = st.number_input("Longueur focale (mm)", value=(focal_length_exif if focal_length_exif is not None else 50.0))
    largeur_capteur = st.number_input("Largeur du capteur (mm)", value=36.0)
    
    # Calculs : Empreinte au sol et GSD
    if st.button("Calculer"):
        # Empreinte au sol en m
        empreinte_sol = (hauteur * largeur_capteur) / focale  
        # Résolution horizontale de l'image (nombre de pixels)
        resolution_pixels = image.width  
        # Ground Sampling Distance (m/pixel)
        gsd = empreinte_sol / resolution_pixels  
        
        st.markdown("### Résultats")
        st.write(f"**Empreinte au sol :** {empreinte_sol:.2f} m")
        st.write(f"**Résolution au sol (GSD) :** {gsd*100:.2f} cm/pixel")
