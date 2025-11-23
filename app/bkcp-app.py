import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Spotify AI Analyzer", page_icon="🎧", layout="centered")

# Estilos CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        font-size: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('modelo_xgboost_final.pkl')
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        # CAMBIA ESTO POR EL NOMBRE REAL DE TU DATASET
        df = pd.read_csv('./dataset/universal_top_spotify_songs.csv') 
        
        # Limpieza básica si es necesaria (eliminar nulos)
        df = df.dropna()
        
        # Asegurarnos de que 'name' sea string para buscar
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str)
            
        return df
    except FileNotFoundError:
        return None

model = load_model()
df_music = load_data()

# --- LISTA DE COLUMNAS EXACTAS DEL MODELO ---
columnas_modelo = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
    'valence', 'tempo', 'time_signature', 'intensity', 'dance_tempo', 'chill_factor'
]

# --- INTERFAZ ---
st.title("🎵 Spotify Kaggle Explorer")
st.markdown("### Elige una canción del Dataset y la IA la clasificará")

if model is None:
    st.error("🚨 ERROR: No se encontró 'modelo_xgboost_final.pkl'.")
    st.stop()

if df_music is None:
    st.error("🚨 ERROR: No se encontró 'dataset_music.csv'. Asegúrate de poner el archivo en la misma carpeta.")
    st.stop()

# --- SELECTOR DE CANCIÓN (BUSCADOR) ---
# Asumimos que tu CSV tiene columnas 'name' y 'artists'. Si no, ajusta los nombres.
if 'name' in df_music.columns and 'artists' in df_music.columns:
    # Creamos una columna combinada para el buscador
    df_music['display_name'] = df_music['name'] + " - " + df_music['artists']
    opciones = df_music['display_name'].unique()
    
    seleccion_nombre = st.selectbox("🔍 Busca una canción:", opciones, index=None, placeholder="Escribe el nombre aquí...")
else:
    st.error("El CSV debe tener columnas 'name' y 'artists'.")
    st.stop()


if seleccion_nombre:
    # Obtener la fila completa de la canción seleccionada
    cancion_data = df_music[df_music['display_name'] == seleccion_nombre].iloc[0]
    
    # Mostrar Info
    st.subheader(f"🎧 Analizando: {seleccion_nombre}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Popularity", cancion_data['popularity'])
    col2.metric("Tempo", f"{cancion_data['tempo']} BPM")
    col3.metric("Energy", f"{cancion_data['energy']:.2f}")

    # --- FEATURE ENGINEERING ---
    # Creamos un DataFrame solo con esta canción
    df_input = pd.DataFrame([cancion_data])
    
    # Generamos las variables sintéticas (Igual que en el training)
    df_input['intensity'] = df_input['energy'] * df_input['loudness']
    df_input['dance_tempo'] = df_input['danceability'] / (df_input['tempo'] + 1)
    df_input['chill_factor'] = df_input['valence'] - df_input['energy']
    
    # Filtramos y ordenamos columnas
    try:
        df_input = df_input[columnas_modelo]
    except KeyError as e:
        st.error(f"Error: El dataset no tiene todas las columnas necesarias. Falta: {e}")
        st.stop()

    # --- PREDICCIÓN ---
    if st.button("🧠 CLASIFICAR AHORA"):
        with st.spinner('Procesando...'):
            pred_num = model.predict(df_input)[0]
            probs = model.predict_proba(df_input)[0]
            
            clases = ['Acoustic', 'Classical', 'Dance', 'Hard-Rock']
            pred_label = clases[pred_num]
            
            st.markdown(f"<h2 style='text-align: center; color: #1DB954;'>Género Predicho: {pred_label.upper()}</h2>", unsafe_allow_html=True)
            
            # Si el dataset original tenía la columna 'music_genre' (la etiqueta real), comparamos
            if 'music_genre' in cancion_data:
                real_genre = cancion_data['music_genre']
                st.write(f"**Etiqueta Real en Dataset:** {real_genre}")
                
                if pred_label.lower() == str(real_genre).lower(): # Comparación segura
                    st.balloons()
                    st.success("¡La IA ha acertado!")
                else:
                    st.warning("La IA ha discrepado con la etiqueta original.")

            # Gráfica
            chart_data = pd.DataFrame(probs, index=clases, columns=["Probabilidad"])
            st.bar_chart(chart_data)

st.caption("Datos cargados desde Kaggle Dataset")
