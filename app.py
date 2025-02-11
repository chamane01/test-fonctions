import streamlit as st
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

def read_hgt(file_path):
    """Lit un fichier .hgt et retourne un tableau numpy avec les altitudes."""
    size = 3601  # Taille des fichiers SRTM-3 (1° x 1°)
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, np.dtype('>i2'), size * size).reshape((size, size))
    return data

def convert_to_tiff(hgt_path, tiff_path):
    """Convertit un fichier .hgt en .tiff"""
    elevation_data = read_hgt(hgt_path)
    
    # Définition des métadonnées pour le GeoTIFF
    transform = from_origin(-180.0, 90.0, 1 / 3600, 1 / 3600)  # Ex: Coordonnées approximatives
    with rasterio.open(
        tiff_path,
        'w',
        driver='GTiff',
        height=elevation_data.shape[0],
        width=elevation_data.shape[1],
        count=1,
        dtype=elevation_data.dtype,
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(elevation_data, 1)

st.title("Convertisseur de fichiers .hgt en .tiff")

uploaded_file = st.file_uploader("Téléversez un fichier .hgt", type="hgt")

if uploaded_file:
    with open("temp.hgt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    output_tiff = "output.tiff"
    convert_to_tiff("temp.hgt", output_tiff)

    st.success("Conversion terminée ! Téléchargez le fichier ci-dessous.")
    
    with open(output_tiff, "rb") as f:
        st.download_button("Télécharger le fichier .tiff", f, file_name="converted.tiff", mime="image/tiff")

    os.remove("temp.hgt")
    os.remove(output_tiff)
