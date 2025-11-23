import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Spotify AI Explorer", page_icon="🎧", layout="wide")

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
    .css-1boxi7i {
        background-color: #f0f2f6;
        padding: 20px;
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
        df = pd.read_csv('./dataset/universal_top_spotify_songs.csv')
        df = df.dropna()
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str)
        return df
    except FileNotFoundError:
        return None

model = load_model()
df_music = load_data()

# --- COLUMNAS DEL MODELO ---
columnas_modelo = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 
    'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
    'valence', 'tempo', 'time_signature', 'intensity', 'dance_tempo', 'chill_factor'
]

# --- INTERFAZ PRINCIPAL ---
st.title("🎵 Spotify AI Explorer")
st.markdown("### Descubre por qué una canción es Rock, Dance o Clásica")

if model is None or df_music is None:
    st.error("🚨 Faltan archivos (modelo o csv). Revisa la carpeta.")
    st.stop()

# --- PANEL LATERAL (BUSCADOR) ---
with st.sidebar:
    st.header("🔍 Buscador")
    if 'name' in df_music.columns and 'artists' in df_music.columns:
        df_music['display_name'] = df_music['name'] + " - " + df_music['artists']
        opciones = df_music['display_name'].unique()
        seleccion_nombre = st.selectbox("Elige canción:", opciones, index=None)
    else:
        st.error("CSV incorrecto.")
        st.stop()

if seleccion_nombre:
    cancion_data = df_music[df_music['display_name'] == seleccion_nombre].iloc[0]
    
    # --- PREPARACIÓN DE DATOS ---
    df_input = pd.DataFrame([cancion_data])
    df_input['intensity'] = df_input['energy'] * df_input['loudness']
    df_input['dance_tempo'] = df_input['danceability'] / (df_input['tempo'] + 1)
    df_input['chill_factor'] = df_input['valence'] - df_input['energy']
    
    try:
        df_input = df_input[columnas_modelo]
    except KeyError:
        st.error("Faltan columnas en el CSV.")
        st.stop()

    # --- PREDICCIÓN ---
    pred_num = model.predict(df_input)[0]
    probs = model.predict_proba(df_input)[0]
    clases = ['Acoustic', 'Classical', 'Dance', 'Hard-Rock']
    pred_label = clases[pred_num]

    # --- DISPLAY DE RESULTADOS ---
    
    # 1. TÍTULO Y PREDICCIÓN
    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader(f"🎧 {seleccion_nombre}")
    with col_res2:
        color = "#1DB954" # Verde Spotify
        if pred_label == "Hard-Rock": color = "#E91E63"
        if pred_label == "Classical": color = "#2196F3"
        if pred_label == "Acoustic": color = "#FF9800"
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin:0">IA: {pred_label.upper()}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 2. DASHBOARD DE MÉTRICAS (EXPANDIDO)
    st.markdown("#### 📊 ADN Matemático de la Canción")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Energy 🔥", f"{cancion_data['energy']:.2f}", help="Intensidad y actividad (0-1)")
    m2.metric("Danceability 💃", f"{cancion_data['danceability']:.2f}", help="¿Es fácil de bailar?")
    m3.metric("Valence 😊", f"{cancion_data['valence']:.2f}", help="Positividad musical (0=Triste, 1=Feliz)")
    m4.metric("Tempo ⏱️", f"{cancion_data['tempo']:.0f} BPM")
    m5.metric("Popularity ⭐", f"{cancion_data['popularity']}")

    # 3. GRÁFICO DE RADAR Y ANÁLISIS
    col_chart, col_text = st.columns([1, 1])

    with col_chart:
        # Datos para el radar (Normalizados 0-1)
        categories = ['Energy', 'Danceability', 'Acousticness', 'Valence', 'Instrumentalness']
        values = [
            cancion_data['energy'], 
            cancion_data['danceability'], 
            cancion_data['acousticness'], 
            cancion_data['valence'],
            cancion_data['instrumentalness']
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=seleccion_nombre,
            line_color=color
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        st.markdown("#### 🧠 ¿Por qué la IA decidió esto?")
        
        # Lógica explicativa simple
        if pred_label == "Dance":
            st.write(f"✅ **Alta Bailabilidad ({cancion_data['danceability']:.2f}):** Clave para el género Dance.")
            if cancion_data['energy'] > 0.7:
                st.write("✅ **Alta Energía:** Típico de música electrónica.")
            st.write("ℹ️ La diferencia con Rock es que el Dance suele tener menos 'Acousticness' y ritmos más constantes.")
            
        elif pred_label == "Hard-Rock":
            st.write("✅ **Alta Energía:** El Rock es intenso.")
            if cancion_data['danceability'] < 0.6:
                st.write(f"📉 **Menor Bailabilidad ({cancion_data['danceability']:.2f}):** A diferencia del Dance, el Rock es más caótico rítmicamente.")
            st.write("🎸 **Volumen:** La distorsión suele aumentar la sonoridad percibida.")

        elif pred_label == "Classical":
            st.write(f"✅ **Alta Acusticidad ({cancion_data['acousticness']:.2f}):** La falta de instrumentos eléctricos es determinante.")
            if cancion_data['instrumentalness'] > 0.5:
                st.write("🎻 **Instrumental:** La ausencia de voz es un indicador fuerte.")

        elif pred_label == "Acoustic":
            st.write("✅ **Acusticidad Media-Alta:** Sonidos orgánicos.")
            st.write("📉 **Energía Moderada:** Menos intensa que el Rock o Dance.")

        st.markdown("---")
        st.write("**Probabilidades completas:**")
        chart_data = pd.DataFrame(probs, index=clases, columns=["Probabilidad"])
        st.bar_chart(chart_data, height=150)

else:
    st.info("👈 Selecciona una canción en el menú de la izquierda para comenzar el análisis.")

st.caption("Powered by XGBoost & Streamlit")
