import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image
import numpy as np

st.title("Affichage d'une image sur une carte dynamique avec Folium")

# Téléverser une image
uploaded_file = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    st.write(f"Taille de l'image : {width} x {height}")

    # Conversion de l'image en tableau numpy
    image_np = np.array(image)

    # Création d'une carte Folium avec CRS "Simple" (projection locale)
    # Les coordonnées de la carte correspondent aux pixels de l'image.
    m = folium.Map(
        location=[height / 2, width / 2],
        zoom_start=1,
        crs="Simple",
        max_bounds=True
    )

    # Définir les limites de l'image :
    # Ici, on considère le coin inférieur gauche à (0,0) et le coin supérieur droit à (height, width)
    bounds = [[0, 0], [height, width]]

    # Superposition de l'image sur la carte avec ImageOverlay
    folium.raster_layers.ImageOverlay(
        image=image_np,
        bounds=bounds,
        opacity=1,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Optionnel : ajouter un contrôle de couche
    folium.LayerControl().add_to(m)

    # Affichage de la carte dans Streamlit
    st_folium(m, width=700, height=500)
else:
    st.info("Veuillez téléverser une image")
