import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model  # Keras dentro de TensorFlow

st.set_page_config(page_title="Clasificador en vivo", page_icon="🎥", layout="wide")

st.title("🎥 Clasificación en vivo con Keras + Streamlit")
st.caption("Cámara dentro de la página y resultados en la misma interfaz. Incluye selector de cámara/calidad y registro a CSV.")

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model_cached(model_path: str):
    return load_model(model_path, compile=False)

@st.cache_data
def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# Cargar recursos
try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo/etiquetas: {e}")
    st.stop()

# --- Sidebar: opciones de cámara y logging ---
st.sidebar.header("Ajustes de cámara")
facing = st.sidebar.selectbox(
    "Tipo de cámara (facingMode)", 
    options=["auto (por defecto)", "user (frontal)", "environment (trasera)"],
    index=0
)
quality = st.sidebar.selectbox(
    "Calidad de video",
    options=["640x480", "1280x720", "1920x1080"],
    index=1
)
w, h = map(int, quality.split("x"))

# Media constraints para WebRTC
video_constraints: dict = {"width": w, "height": h}
if facing != "auto (por defecto)":
    # facingMode es respetado por navegadores móviles; en desktop puede no aplicar
    video_constraints["facingMode"] = facing.split(" ")[0]

media_constraints = {"video": video_constraints, "audio": False}

st.sidebar.header("Registro de predicciones")
enable_log = st.sidebar.checkbox("Habilitar registro (CSV)", value=True)
log_every_n_seconds = st.sidebar.slider("Intervalo de registro (s)", 0.2, 5.0, 1.0, 0.2)

if "pred_log" not in st.session_state:
    st.session_state.pred_log = pd.DataFrame(columns=["timestamp", "label", "confidence"])
if "last_log_ts" not in st.session_state:
    st.session_state.last_log_ts = 0.0

# Configuración STUN para WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # BGR
        # Preparación a 224x224 y normalización (-1,1)
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0

        # Predicción
        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])

        self.latest = {"class": label, "confidence": conf}

        # Overlay
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        # Caja en esquina superior izquierda
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

# Layout
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Cámara en vivo")
    webrtc_ctx = webrtc_streamer(
        key="keras-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )
    st.info(
        "Si no ves tu cámara, concede permisos del navegador o prueba con otro (Chrome recomendado). "
        "En móviles, 'user' = frontal y 'environment' = trasera (si el dispositivo lo soporta).",
        icon="ℹ️",
    )

with right:
    st.subheader("Resultados")
    result_placeholder = st.empty()
    progress_placeholder = st.empty()

    # Botón para limpiar log
    if enable_log and not st.session_state.pred_log.empty:
        if st.button("🧹 Limpiar registro"):
            st.session_state.pred_log = st.session_state.pred_log.iloc[0:0]
            st.session_state.last_log_ts = 0.0

    # Descarga CSV
    csv_bytes = st.session_state.pred_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV de predicciones",
        data=csv_bytes,
        file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        disabled=st.session_state.pred_log.empty,
    )

    # Actualización periódica de resultados y logging
    if webrtc_ctx and webrtc_ctx.state.playing:
        for _ in range(300000):
            if not webrtc_ctx.state.playing:
                break
            vt = webrtc_ctx.video_transformer
            if vt is not None and vt.latest["class"] is not None:
                cls = vt.latest["class"]
                conf = vt.latest["confidence"]
                result_placeholder.markdown(f"**Clase detectada:** `{cls}`\n\n**Confianza:** `{conf*100:.2f}%`")
                progress_placeholder.progress(min(max(conf, 0.0), 1.0))

                # Logging
                if enable_log:
                    now = time.time()
                    if now - st.session_state.last_log_ts >= log_every_n_seconds:
                        st.session_state.pred_log.loc[len(st.session_state.pred_log)] = [
                            datetime.utcnow().isoformat(),
                            cls,
                            round(conf, 6),
                        ]
                        st.session_state.last_log_ts = now

            time.sleep(0.2)
    else:
        st.write("Activa la cámara para ver aquí las predicciones.")

st.markdown("---")
with st.expander("⚠️ Modo alternativo (captura por foto, sin WebRTC)"):
    st.write("Si tu red bloquea WebRTC, usa una foto para predecir de forma puntual.")
    snap = st.camera_input("Captura una imagen")
    if snap is not None:
        file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        st.image(img, caption=f"{label} | {conf*100:.2f}%")
        st.success(f"Predicción: **{label}** ({conf*100:.2f}%)")

