import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE LA PÁGINA ---
# Cambiado a layout="wide" para ocupar toda la pantalla
st.set_page_config(page_title="Spotify AI Recommender", page_icon="🎧", layout="wide")

# Estilos CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    .rec-card {
        background-color: #282828;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #1DB954;
        height: 100%; /* Para que todas las tarjetas tengan la misma altura */
    }
    /* Ocultar el menú hamburguesa y el footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def load_resources():
    # 1. Cargar Modelo XGBoost
    try:
        model = joblib.load('modelo_xgboost_final.pkl')
    except FileNotFoundError:
        return None, None

    # 2. Cargar Dataset
    try:
        df = pd.read_csv('../dataset/universal_top_spotify_songs.csv')
        df = df.dropna()
        
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str)
        if 'artists' in df.columns:
            df['artists'] = df['artists'].astype(str)
        
        if 'name' in df.columns and 'artists' in df.columns:
            df = df.drop_duplicates(subset=['name', 'artists'], keep='first')
        
        if 'artists' in df.columns:
             df['display_name'] = df['name'] + " - " + df['artists']
        else:
             df['display_name'] = df['name']
             
        return model, df
    except FileNotFoundError:
        return model, None

model, df_music = load_resources()

# --- COLUMNAS DEL MODELO ---
columnas_modelo = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
    'valence', 'tempo', 'time_signature', 'intensity', 'dance_tempo', 'chill_factor'
]

# --- LÓGICA DEL RECOMENDADOR (KNN) ---
def get_recommendations(df, current_song_features, n_recommendations=4):
    df_features = df.copy()
    df_features['intensity'] = df_features['energy'] * df_features['loudness']
    df_features['dance_tempo'] = df_features['danceability'] / (df_features['tempo'] + 1)
    df_features['chill_factor'] = df_features['valence'] - df_features['energy']
    
    try:
        X = df_features[columnas_modelo]
    except KeyError:
        cols_basic = ['energy', 'danceability', 'acousticness', 'valence', 'tempo', 'loudness']
        X = df_features[cols_basic]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = NearestNeighbors(n_neighbors=n_recommendations+1, algorithm='auto', metric='euclidean')
    knn.fit(X_scaled)
    
    if isinstance(current_song_features, pd.DataFrame):
        try:
            current_feat = current_song_features[columnas_modelo]
        except:
            current_feat = current_song_features[cols_basic]
            
        current_scaled = scaler.transform(current_feat)
        distances, indices = knn.kneighbors(current_scaled)
        return df.iloc[indices[0][1:]]
    return None


# --- INTERFAZ PRINCIPAL ---
st.title("🎵 Spotify AI Analyzer")
st.markdown("Descubre el género musical y encuentra canciones similares.")

if model is None or df_music is None:
    st.error("🚨 Error: No se encontraron los archivos (modelo o dataset).")
    st.stop()

# --- BUSCADOR CENTRAL ---
# Usamos columnas para centrar el buscador aunque la pantalla sea ancha
c_search1, c_search2, c_search3 = st.columns([1, 2, 1])
with c_search2:
    st.markdown("### 🔍 Busca una canción")
    opciones = df_music['display_name'].unique()
    seleccion_nombre = st.selectbox(
        "Escribe el nombre de la canción:", 
        opciones, 
        index=None, 
        placeholder="Ej: Blinding Lights - The Weeknd",
        label_visibility="collapsed"
    )

if seleccion_nombre:
    cancion_data = df_music[df_music['display_name'] == seleccion_nombre].iloc[0]
    
    # Feature Engineering
    df_input = pd.DataFrame([cancion_data])
    df_input['intensity'] = df_input['energy'] * df_input['loudness']
    df_input['dance_tempo'] = df_input['danceability'] / (df_input['tempo'] + 1)
    df_input['chill_factor'] = df_input['valence'] - df_input['energy']
    
    df_input_model = df_input[columnas_modelo]

    # --- PREDICCIÓN ---
    pred_num = model.predict(df_input_model)[0]
    clases = ['Acoustic', 'Classical', 'Dance', 'Hard-Rock']
    pred_label = clases[pred_num]

    st.markdown("---")

    # --- RESULTADOS (GRID LAYOUT) ---
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.header(f"🎧 {seleccion_nombre}")
        if 'music_genre' in cancion_data:
            st.caption(f"Género original: {cancion_data['music_genre']}")
    with col_head2:
        color = "#1DB954"
        if pred_label == "Hard-Rock": color = "#E91E63"
        if pred_label == "Classical": color = "#2196F3"
        if pred_label == "Acoustic": color = "#FF9800"
        st.markdown(f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="margin:0">{pred_label}</h2>
            <small>Predicción IA</small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --- VISUALIZACIÓN Y MÉTRICAS ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### 📊 ADN Sónico")
        categories = ['Energy', 'Danceability', 'Acousticness', 'Valence', 'Instrumentalness']
        values = [cancion_data['energy'], cancion_data['danceability'], cancion_data['acousticness'], cancion_data['valence'], cancion_data['instrumentalness']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color=color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=350, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### 🧠 Análisis de Variables Clave")
        st.write("") 
        
        # Organizamos las 6 métricas en 3 filas conceptuales
        
        # FILA 1: INTENSIDAD (Energía y Volumen)
        m1, m2 = st.columns(2)
        m1.metric("Energía 🔥", f"{cancion_data['energy']:.2f}", help="Intensidad general (0-1)")
        m2.metric("Volumen 🔊", f"{cancion_data['loudness']:.1f} dB", help="Potencia sonora en decibelios")

        # FILA 2: RITMO (Bailabilidad y Tempo)
        m3, m4 = st.columns(2)
        m3.metric("Bailabilidad 💃", f"{cancion_data['danceability']:.2f}", help="Facilidad para bailar (0-1)")
        m4.metric("Tempo ⏱️", f"{cancion_data['tempo']:.0f} BPM", help="Velocidad de la canción")

        # FILA 3: ESTILO (Instrumental y Humor)
        m5, m6 = st.columns(2)
        instru_val = cancion_data['instrumentalness']
        m5.metric("Instrumental 🎻", f"{instru_val:.2f}", help="Probabilidad de que no tenga voz (>0.5 es instrumental)")
        m6.metric("Valencia 😊", f"{cancion_data['valence']:.2f}", help="Positividad musical (Triste-Feliz)")
        
        # Explicación Inteligente (Lógica de Negocio)
        st.write("")
        if pred_label == 'Classical':
            st.info(f"💡 **Dato Clave:** La IA ha detectado una **Instrumentalidad muy alta ({instru_val:.2f})**. Al no detectar voz humana, descarta géneros como Acoustic o Rock.")
        elif pred_label == 'Hard-Rock':
            st.info(f"💡 **Dato Clave:** El alto **Volumen ({cancion_data['loudness']:.1f} dB)** y Energía definen este género, diferenciándolo del Acoustic.")
        elif pred_label == 'Acoustic':
             st.info(f"💡 **Dato Clave:** A diferencia del Classical, aquí la Instrumentalidad es baja (hay voz), pero la Energía no es suficiente para ser Rock.")
        else:
            st.info(f"💡 **Dato Clave:** La combinación de **Bailabilidad ({cancion_data['danceability']:.2f})** y Energía posiciona esta canción en el clúster de {pred_label}.")

    st.markdown("---")

    # --- RECOMENDACIONES ---
    st.subheader("✨ Canciones Similares (KNN)")
    
    with st.spinner("Analizando base de datos..."):
        recomendaciones = get_recommendations(df_music, df_input_model, n_recommendations=4)
    
    if recomendaciones is not None:
        # Volvemos a 4 columnas porque ahora tenemos espacio de sobra
        cols = st.columns(4) 
        for i, (index, row) in enumerate(recomendaciones.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="rec-card">
                    <div style="font-weight: bold; font-size: 1.1em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['name']}</div>
                    <div style="color: #bbb; font-size: 0.9em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['artists']}</div>
                    <hr style="border-color: #444; margin: 8px 0;">
                    <div style="font-size: 0.85em; color: #1DB954; display: flex; justify-content: space-between;">
                        <span>⚡ {row['energy']:.2f}</span>
                        <span>⏱️ {row['tempo']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👆 Utiliza el buscador para comenzar.")
