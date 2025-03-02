import streamlit as st 
import os
import rasterio
import rasterio.merge
import rasterio.mask
import folium
import math
import matplotlib.pyplot as plt
import numpy as np
from streamlit_folium import st_folium
from folium.plugins import Draw
from io import BytesIO
from datetime import date, datetime
import base64
import contextily as ctx  # pour le fond de carte

# ------------------------------
# Partie ReportLab pour le rapport
# ------------------------------
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib import colors

from shapely.geometry import shape

# Dimensions standard pour le PDF
PAGE_WIDTH, PAGE_HEIGHT = A4
SECTION_HEIGHT = PAGE_HEIGHT / 3
COLUMN_WIDTH = PAGE_WIDTH / 2

# ------------------------------
# Fonctions utilitaires pour la conversion d'images
# ------------------------------
def image_bytes_to_data_url(image_bytes):
    """Convertit des bytes d'image en data URL pour l'overlay Folium."""
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return 'data:image/png;base64,' + base64_str

def get_bounds_from_geometry(geometry):
    """Calcule les limites géographiques [ [sud, ouest], [nord, est] ] d'une géométrie de type Polygon."""
    coords = geometry.get("coordinates")[0]  # première bague
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

# ==============================
# Fonctions utilitaires - ANALYSE SPATIALE
# ==============================
def load_tiff_files(folder_path):
    """Charge les fichiers TIFF contenus dans un dossier."""
    try:
        tiff_files = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    except Exception as e:
        st.error(f"Erreur lors de la lecture du dossier {folder_path} : {e}")
        return []
    
    if not tiff_files:
        st.error("Aucun fichier TIFF trouvé dans le dossier.")
        return []
    
    return [f for f in tiff_files if os.path.exists(f)]

def build_mosaic(tiff_files, mosaic_path="mosaic.tif"):
    """Construit une mosaïque à partir de fichiers TIFF."""
    try:
        src_files = [rasterio.open(fp) for fp in tiff_files]
        mosaic, out_trans = rasterio.merge.merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()
        return mosaic_path
    except Exception as e:
        st.error(f"Erreur lors de la création de la mosaïque : {e}")
        return None

def create_map(mosaic_file):
    """
    Crée une carte Folium affichant l'emprise de la mosaïque et intégrant
    tous les outils de dessin pour augmenter les possibilités.
    (Cette carte servira uniquement à recueillir les dessins.)
    """
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Calque indiquant l'emprise de la mosaïque (transformé en EPSG:4326 pour Folium)
    mosaic_group = folium.FeatureGroup(name="Mosaïque")
    try:
        with rasterio.open(mosaic_file) as src:
            bounds = src.bounds
            if src.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform
                left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
                xs, ys = transform(src.crs, "EPSG:4326", [left, right], [bottom, top])
                bounds_latlon = [[min(ys), min(xs)], [max(ys), max(xs)]]
            else:
                bounds_latlon = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
            
            folium.Rectangle(
                bounds=bounds_latlon,
                color='blue',
                fill=False,
                tooltip="Emprise de la mosaïque"
            ).add_to(mosaic_group)
    except Exception as e:
        st.error(f"Erreur lors de l'ouverture de la mosaïque : {e}")
    
    mosaic_group.add_to(m)
    
    # Outils de dessin
    Draw(
        draw_options={
            # Pour les polylignes, on autorise les intersections et on spécifie un style
            'polyline': {'allowIntersection': True, 'shapeOptions': {'color': 'black', 'weight': 2}},
            'polygon': True,
            'rectangle': True,
            'circle': True,
            'marker': True,
            'circlemarker': True
        },
        edit_options={'edit': True, 'remove': True}
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

def generate_contours(mosaic_file, drawing_geometry, show_basemap=True):
    """
    Génère une figure matplotlib affichant les contours d'élévation
    pour la zone définie par drawing_geometry, en convertissant
    les données en coordonnées UTM. La figure est limitée à l'emprise
    dessinée (avec 5% de marge) et affiche, en arrière-plan, le fond de carte
    (avec une opacité de 50%) si le paramètre show_basemap est activé.
    Pour chaque autre dessin présent dans cette emprise, seule la partie intérieure
    est tracée en noir (trait continu).

    Les modifications apportées sont les suivantes :
      - Dans la légende, une seule entrée par classe de dessin est affichée (ex. "Profil")
        tandis que sur le dessin, chaque ligne (profil) conserve son numéro (ex. "Profil 1", "Profil 2", etc.)
      - Pour les lignes (profils), le texte est incrusté directement le long du trait.
      - Une classe est ajoutée pour chaque type de dessin (points, lignes, polygones, etc.).
    """
    try:
        with rasterio.open(mosaic_file) as src:
            # Conversion de la géométrie dessinée pour le masquage : de EPSG:4326 vers le CRS du raster
            geom = drawing_geometry
            if src.crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform_geom
                geom = transform_geom("EPSG:4326", src.crs, drawing_geometry)
            
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
            data = out_image[0]
            nodata = src.nodata

            nrows, ncols = data.shape
            # Création d'une grille en CRS du raster
            x_coords = np.arange(ncols) * out_transform.a + out_transform.c + out_transform.a/2
            y_coords = np.arange(nrows) * out_transform.e + out_transform.f + out_transform.e/2
            X, Y = np.meshgrid(x_coords, y_coords)

            # Détermination du CRS UTM à partir du centre de la zone masquée
            from rasterio.warp import transform
            center_x = out_transform.c + (ncols/2) * out_transform.a
            center_y = out_transform.f + (nrows/2) * out_transform.e
            if src.crs.to_string() != "EPSG:4326":
                lon, lat = transform(src.crs, "EPSG:4326", [center_x], [center_y])
                center_lon, center_lat = lon[0], lat[0]
            else:
                center_lon, center_lat = center_x, center_y
            utm_zone = int((center_lon + 180) / 6) + 1
            if center_lat >= 0:
                utm_crs = f"EPSG:{32600 + utm_zone}"
            else:
                utm_crs = f"EPSG:{32700 + utm_zone}"

            # Transformation de la grille en coordonnées UTM
            x_flat = X.flatten()
            y_flat = Y.flatten()
            X_utm_flat, Y_utm_flat = transform(src.crs, utm_crs, x_flat, y_flat)
            X_utm = np.array(X_utm_flat).reshape(X.shape)
            Y_utm = np.array(Y_utm_flat).reshape(Y.shape)

            # Conversion de la zone dessinée (en EPSG:4326) vers UTM
            from rasterio.warp import transform_geom
            geom_utm = transform_geom("EPSG:4326", utm_crs, drawing_geometry)
            # Création de l'enveloppe shapely de la zone dessinée
            envelope = shape(geom_utm)

            # Remplacement des pixels nodata par NaN
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)

            # Calcul des niveaux de contour
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            levels = np.linspace(vmin, vmax, 15)

            # Création de la figure en UTM
            fig, ax = plt.subplots(figsize=(8, 6))
            cs = ax.contour(X_utm, Y_utm, data, levels=levels, cmap='terrain', zorder=3)
            ax.clabel(cs, inline=True, fontsize=8)
            ax.set_title("Contours d'élévation (UTM)")
            ax.set_xlabel("UTM Easting")
            ax.set_ylabel("UTM Northing")

            # Limitation de l'affichage à l'enveloppe avec 5% de marge
            minx, miny, maxx, maxy = envelope.bounds
            dx = (maxx - minx) * 0.05
            dy = (maxy - miny) * 0.05
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)

            # Ajout conditionnel du fond de carte selon le paramètre show_basemap
            if show_basemap:
                ctx.add_basemap(ax, crs=utm_crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)

            # Initialisation des flags pour la légende
            added_profile = False
            added_polygon = False
            added_point = False

            # Compteur pour numéroter les profils
            profile_counter = 1

            # Parcours de tous les dessins enregistrés et tracé de leur intersection avec l'enveloppe
            if "raw_drawings" in st.session_state:
                for d in st.session_state["raw_drawings"]:
                    if isinstance(d, dict) and "geometry" in d:
                        # Exclure la géométrie utilisée pour générer les contours
                        if d.get("geometry") == drawing_geometry:
                            continue
                        try:
                            # Transformation de la géométrie du dessin depuis EPSG:4326 vers UTM
                            geom_other_utm = transform_geom("EPSG:4326", utm_crs, d["geometry"])
                            shapely_other = shape(geom_other_utm)
                            # Calcul de l'intersection avec l'enveloppe dessinée
                            if shapely_other.intersects(envelope):
                                clipped = shapely_other.intersection(envelope)
                                if clipped.is_empty:
                                    continue
                                # Pour les polygones : tracer le contour et ajouter une seule entrée légende "Polygone"
                                if clipped.geom_type in ["Polygon", "MultiPolygon"]:
                                    label = "Polygone" if not added_polygon else "_nolegend_"
                                    if not added_polygon:
                                        added_polygon = True
                                    if clipped.geom_type == "Polygon":
                                        x_other, y_other = clipped.exterior.xy
                                        ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=label, zorder=4)
                                    else:
                                        for part in clipped.geoms:
                                            x_other, y_other = part.exterior.xy
                                            ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=label, zorder=4)
                                # Pour les lignes : tracer le trait, incruster le numéro du profil sur le trait,
                                # et ajouter une seule entrée légende "Profil"
                                elif clipped.geom_type in ["LineString", "MultiLineString"]:
                                    current_profile_label = f"Profil {profile_counter}"
                                    profile_counter += 1
                                    legend_label = "Profil" if not added_profile else "_nolegend_"
                                    if not added_profile:
                                        added_profile = True
                                    if clipped.geom_type == "LineString":
                                        x_other, y_other = clipped.xy
                                        ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=legend_label, zorder=4)
                                        if len(x_other) >= 2:
                                            dx = x_other[1] - x_other[0]
                                            dy = y_other[1] - y_other[0]
                                            angle = np.degrees(np.arctan2(dy, dx))
                                        else:
                                            angle = 0
                                        centroid = clipped.centroid
                                        ax.text(centroid.x, centroid.y, current_profile_label, fontsize=8, color='black', ha='center', va='center', rotation=angle, zorder=6)
                                    else:
                                        for part in clipped.geoms:
                                            x_other, y_other = part.xy
                                            ax.plot(x_other, y_other, color='black', linestyle='-', linewidth=2, label=legend_label, zorder=4)
                                            if len(x_other) >= 2:
                                                dx = x_other[1] - x_other[0]
                                                dy = y_other[1] - y_other[0]
                                                angle = np.degrees(np.arctan2(dy, dx))
                                            else:
                                                angle = 0
                                            centroid = part.centroid
                                            ax.text(centroid.x, centroid.y, current_profile_label, fontsize=8, color='black', ha='center', va='center', rotation=angle, zorder=6)
                                # Pour les points : tracer le point et ajouter une seule entrée légende "Point"
                                elif clipped.geom_type in ["Point", "MultiPoint"]:
                                    label = "Point" if not added_point else "_nolegend_"
                                    if not added_point:
                                        added_point = True
                                    if clipped.geom_type == "Point":
                                        ax.plot(clipped.x, clipped.y, 'o', color='black', markersize=8, label=label, zorder=4)
                                    else:
                                        for part in clipped.geoms:
                                            ax.plot(part.x, part.y, 'o', color='black', markersize=8, label=label, zorder=4)
                        except Exception as e:
                            st.error(f"Erreur lors du tracé d'un dessin supplémentaire : {e}")

            # Re-affichage de l'enveloppe dessinée en rouge pour garantir sa visibilité (le contour doit rester rouge)
            x_env, y_env = envelope.exterior.xy
            ax.plot(x_env, y_env, color='red', linewidth=2, label="Zone dessinée", zorder=5)

            # Placement de la légende en bas à droite, avec fond blanc et taille réduite de 20%
            leg = ax.legend(loc='lower right', framealpha=1, facecolor='white', fontsize=8)
            for text in leg.get_texts():
                text.set_fontsize(8)
            return fig
    except Exception as e:
        st.error(f"Erreur lors de la génération des contours : {e}")
        return None

def haversine(lon1, lat1, lon2, lat2):
    """Calcule la distance (m) entre deux points GPS."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r

def interpolate_line(coords, step=50):
    """Interpole des points le long d'une ligne pour obtenir des échantillons réguliers."""
    if len(coords) < 2:
        return coords, [0]
    sampled_points = [coords[0]]
    cumulative_dist = [0]
    for i in range(len(coords)-1):
        start = coords[i]
        end = coords[i+1]
        seg_distance = haversine(start[0], start[1], end[0], end[1])
        num_steps = max(int(seg_distance // step), 1)
        for j in range(1, num_steps+1):
            fraction = j / num_steps
            lon = start[0] + fraction * (end[0]-start[0])
            lat = start[1] + fraction * (end[1]-start[1])
            dist = haversine(sampled_points[-1][0], sampled_points[-1][1], lon, lat)
            sampled_points.append([lon, lat])
            cumulative_dist.append(cumulative_dist[-1] + dist)
    return sampled_points, cumulative_dist

def generate_profile(mosaic_file, coords, profile_title):
    """
    Génère un profil d'élévation le long d'une ligne définie par 'coords'.
    Retourne la figure matplotlib.
    """
    try:
        points, distances = interpolate_line(coords)
        elevations = []
        with rasterio.open(mosaic_file) as src:
            for p in points:
                pt = p
                # Transformation du point de EPSG:4326 vers le CRS du raster si nécessaire
                if src.crs.to_string() != "EPSG:4326":
                    from rasterio.warp import transform
                    xs, ys = transform("EPSG:4326", src.crs, [p[0]], [p[1]])
                    pt = (xs[0], ys[0])
                elev = list(src.sample([pt]))[0][0]
                elevations.append(elev)
    except Exception as e:
        st.error(f"Erreur lors de la génération du profil : {e}")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(distances, elevations, 'b-', linewidth=1.5)
    ax.set_title(profile_title)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude (m)")
    return fig

def store_figure(fig, result_type, title):
    """
    Sauvegarde la figure matplotlib dans un buffer et stocke le résultat
    dans st.session_state["analysis_results"].
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = []
    st.session_state["analysis_results"].append({
        "type": result_type,
        "title": title,
        "image": buf.getvalue()
    })

# ==============================
# Interface - Analyse Spatiale
# ==============================
def run_analysis_spatiale():
    st.title("🔍 Analyse Spatiale")
    st.info("Ce module vous permet de générer des contours (à partir de rectangles sélectionnés) ou des profils d'élévation (à partir de lignes).")
    
    # Initialisation du mode pour cette partie
    if "analysis_mode" not in st.session_state:
        st.session_state["analysis_mode"] = "none"
    
    # Initialisation pour conserver les dessins
    st.session_state.setdefault("raw_drawings", [])
    
    # Saisie du nom de la carte
    map_name = st.text_input("Nom de votre carte", value="Ma Carte", key="analysis_map_name")
    
    # Chargement et construction de la mosaïque
    folder_path = "TIFF"
    if not os.path.exists(folder_path):
        st.error("Dossier TIFF introuvable")
        return
    tiff_files = load_tiff_files(folder_path)
    if not tiff_files:
        return
    mosaic_path = build_mosaic(tiff_files)
    if not mosaic_path:
        return
    
    # Création de la carte interactive (pour dessiner)
    m = create_map(mosaic_path)
    st.write("**Utilisez l'outil de dessin sur la carte ci-dessous.**")
    map_data = st_folium(m, width=700, height=500, key="analysis_map")
    
    # Sauvegarder les dessins dans la session pour persistance
    if map_data is not None and isinstance(map_data, dict) and "all_drawings" in map_data:
        st.session_state["raw_drawings"] = map_data["all_drawings"]
    
    # Menu de sélection du mode
    options_container = st.container()
    if st.session_state["analysis_mode"] == "none":
        col1, col2 = options_container.columns(2)
        if col1.button("Tracer des profils", key="btn_profiles"):
            st.session_state["analysis_mode"] = "profiles"
        if col2.button("Générer des contours", key="btn_contours"):
            st.session_state["analysis_mode"] = "contours"
    
    # Mode Générer des contours (à partir de rectangles dessinés)
    if st.session_state["analysis_mode"] == "contours":
        st.subheader("Générer des contours")
        # Ajout du bouton à cocher pour l'affichage du fond de carte
        show_basemap = st.checkbox("Afficher le fond de carte", value=True)
        drawing_geometries = []
        raw_drawings = st.session_state.get("raw_drawings") or []
        # Sélectionner uniquement les dessins de type Polygon (issus d'un rectangle ou d'un polygone)
        for drawing in raw_drawings:
            if isinstance(drawing, dict) and drawing.get("geometry", {}).get("type") == "Polygon":
                drawing_geometries.append(drawing.get("geometry"))
        if not drawing_geometries:
            st.warning("Veuillez dessiner au moins un rectangle sur la carte pour définir une zone.")
        else:
            # Permettre de sélectionner parmi les rectangles dessinés
            options_list = [f"Rectangle {i+1}" for i in range(len(drawing_geometries))]
            selected_indices = st.multiselect("Sélectionnez les rectangles pour générer des contours", options=options_list)
            if st.button("Générer les contours sélectionnés", key="generate_selected_contours"):
                for sel in selected_indices:
                    idx = int(sel.split()[1]) - 1  # extraire l'indice
                    geometry = drawing_geometries[idx]
                    fig = generate_contours(mosaic_path, geometry, show_basemap=show_basemap)
                    if fig is not None:
                        st.pyplot(fig)
                        store_figure(fig, "contour", f"Contours - Emprise {idx+1}")
        if st.button("Retour", key="retour_contours"):
            st.session_state["analysis_mode"] = "none"
    
    # Mode Tracer des profils (ligne)
    if st.session_state["analysis_mode"] == "profiles":
        st.subheader("Tracer des profils")
        raw_drawings = st.session_state.get("raw_drawings") or []
        current_drawings = []
        if isinstance(raw_drawings, list):
            current_drawings = [d for d in raw_drawings if isinstance(d, dict) and d.get("geometry", {}).get("type") == "LineString"]
        if not current_drawings:
            st.info("Aucune ligne tracée pour le moment. Veuillez dessiner une ligne sur la carte.")
        else:
            for i, drawing in enumerate(current_drawings):
                profile_title = f"{map_name} - Profil {i+1}"
                st.markdown(f"#### {profile_title}")
                try:
                    fig = generate_profile(mosaic_path, drawing["geometry"]["coordinates"], profile_title)
                    if fig is not None:
                        st.pyplot(fig)
                        store_figure(fig, "profile", profile_title)
                except Exception as e:
                    st.error(f"Erreur de traitement : {e}")
        if st.button("Retour", key="retour_profiles"):
            st.session_state["analysis_mode"] = "none"

# ==============================
# Menu pour ajouter une carte d'analyse spatiale
# (Similaire au formulaire pour les images téléversées)
# ==============================
def create_analysis_card_controller():
    with st.expander("➕ Ajouter une carte d'analyse spatiale", expanded=True):
        if "analysis_results" not in st.session_state or not st.session_state["analysis_results"]:
            st.info("Aucune carte d'analyse spatiale n'est disponible pour le moment.")
            return None
        
        # Sélection d'une carte issue des résultats d'analyse
        options = {f"{i+1} - {res['title']}": i for i, res in enumerate(st.session_state["analysis_results"])}
        chosen = st.selectbox("Choisissez une carte", list(options.keys()), key="analysis_card_select")
        idx = options[chosen]
        
        # Paramètres similaires à ceux du téléversement d'images
        col1, col2 = st.columns(2)
        with col1:
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="analysis_card_size")
        with col2:
            v_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="analysis_card_v_pos")
            h_pos = st.selectbox("Position horizontale", ["Gauche", "Droite", "Centre"], key="analysis_card_h_pos")
        
        title_input = st.text_input("Titre pour la carte", key="analysis_card_title", value=st.session_state["analysis_results"][idx]["title"])
        description_input = st.text_input("Description pour la carte", key="analysis_card_description", value="Carte générée depuis l'analyse spatiale")
        
        if st.button("Valider la carte d'analyse", key="validate_analysis_card"):
            return {
                "type": "Image",
                "size": size,
                "v_pos": v_pos,
                "h_pos": h_pos,
                "content": st.session_state["analysis_results"][idx]["image"],
                "image_title": title_input,
                "description": description_input,
                "analysis_ref": idx
            }
    return None

# ==============================
# Fonctions utilitaires - RAPPORT
# ==============================
def create_element_controller():
    with st.expander("➕ Ajouter un élément", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            elem_type = st.selectbox("Type", ["Image", "Texte"], key="rapport_elem_type")
            size = st.selectbox("Taille", ["Grand", "Moyen", "Petit"], key="rapport_elem_size")
        with col2:
            vertical_pos = st.selectbox("Position verticale", ["Haut", "Milieu", "Bas"], key="rapport_v_pos")
            horizontal_options = ["Gauche", "Droite", "Centre"] if size == "Petit" else ["Gauche", "Droite"]
            horizontal_pos = st.selectbox("Position horizontale", horizontal_options, key="rapport_h_pos")
        
        if elem_type == "Image":
            content = st.file_uploader("Contenu (image)", type=["png", "jpg", "jpeg"], key="rapport_content_image")
            image_title = st.text_input("Titre de l'image", max_chars=50, key="rapport_image_title")
            description = st.text_input("Description brève (max 100 caractères)", max_chars=100, key="rapport_image_desc")
        else:
            content = st.text_area("Contenu", key="rapport_content_text")
        
        if st.button("Valider l'élément", key="rapport_validate_element"):
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
            if isinstance(metadata['logo'], bytes):
                logo_stream = BytesIO(metadata['logo'])
            else:
                logo_stream = metadata['logo']
            img = ImageReader(logo_stream)
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
                    if isinstance(element["content"], bytes):
                        image_stream = BytesIO(element["content"])
                    else:
                        image_stream = element["content"]
                    img = ImageReader(image_stream)
                    
                    top_margin = 20
                    bottom_margin = 20
                    horizontal_scale = 0.9
                    image_actual_width = width * horizontal_scale
                    image_actual_height = height - top_margin - bottom_margin
                    image_x = x + (width - image_actual_width) / 2
                    image_y = y + bottom_margin
                    c.drawImage(img, image_x, image_y, width=image_actual_width, height=image_actual_height, preserveAspectRatio=True, mask='auto')
                    
                    if element.get("image_title"):
                        c.setFont("Helvetica-Bold", 12)
                        image_title = element["image_title"].upper()
                        c.drawCentredString(x + width / 2, y + height - top_margin / 2, image_title)
                    
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

# ==============================
# Interface - Rapport
# ==============================
def run_report():
    st.title("📄 Génération de Rapport")
    
    # Sidebar dédiée au rapport
    with st.sidebar:
        st.header("📝 Métadonnées du Rapport")
        titre = st.text_input("Titre principal", key="rapport_titre")
        report_id = st.text_input("ID du rapport", key="rapport_id")
        report_date = st.date_input("Date du rapport", date.today(), key="rapport_date")
        report_time = st.time_input("Heure du rapport", datetime.now().time(), key="rapport_time")
        editor = st.text_input("Éditeur", key="rapport_editor")
        location = st.text_input("Localisation", key="rapport_location")
        company = st.text_input("Société", key="rapport_company")
        logo = st.file_uploader("Logo", type=["png", "jpg", "jpeg"], key="rapport_logo")
    
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
    
    # Initialisation des éléments du rapport
    if "elements" not in st.session_state:
        st.session_state["elements"] = []
    elements = st.session_state["elements"]
    
    st.markdown("### 📌 Ajouter une carte d'analyse spatiale")
    analysis_card = create_analysis_card_controller()
    if analysis_card:
        # Éviter les doublons en se basant sur 'analysis_ref'
        if not any(el.get("analysis_ref") == analysis_card.get("analysis_ref") for el in elements if el["type"] == "Image"):
            elements.append(analysis_card)
            st.success("Carte d'analyse ajoutée avec succès !")
    
    st.markdown("### Ajouter d'autres éléments")
    new_element = create_element_controller()
    if new_element:
        elements.append(new_element)
        st.session_state["elements"] = elements
        st.success("Élément validé avec succès !")
    
    if elements:
        display_elements_preview(elements)
    
    if elements and st.button("Générer le PDF", key="generate_pdf"):
        pdf = generate_pdf(elements, metadata)
        st.success("✅ Rapport généré avec succès!")
        st.download_button("Télécharger le PDF", pdf, "rapport_structuré.pdf", "application/pdf")

# ==============================
# Application Principale
# ==============================
def main():
    st.set_page_config(page_title="Application SIG & Rapport", layout="wide")
    
    # Menu principal dans la sidebar
    menu = st.sidebar.radio("Menu Principal", ["Analyse Spatiale", "Rapport"], key="main_menu")
    
    if menu == "Analyse Spatiale":
        run_analysis_spatiale()
    elif menu == "Rapport":
        run_report()

if __name__ == "__main__":
    main()
