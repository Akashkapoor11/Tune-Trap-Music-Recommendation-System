# modern_app_clean.py — Premium UI with proper placeholder control + visible Start/Stop buttons
import streamlit as st
import cv2
import time

from emotion_detector import EmotionDetector

try:
    from utils import get_youtube_recommendations, FALLBACK
except:
    FALLBACK = {
        "happy": [("Upbeat Pop Mix", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")],
        "sad": [("Chill Vibes", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")],
        "angry": [("Aggressive Beats", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")],
        "surprise": [("Surprise Mix", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")],
        "neutral": [("Chill Vibes", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")],
    }

st.set_page_config(page_title="TUNE TRAP — Premium", layout="wide")

# ---------------------------------------------------------------
# CSS – theme + button styles
# ---------------------------------------------------------------
st.markdown("""
<style>
html, body, .block-container {
    background-color: #0d1117 !important;
    color: #e6eef3 !important;
}
h1, h2, h3, h4, h5 {
    color: white !important;
    font-family: "Segoe UI", sans-serif;
}
.card {
    background-color: #161b22;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.stButton > button {
    background: linear-gradient(90deg, #00b4d8, #0077b6) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    border: none !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.9;
}
.emotion-badge {
    background: linear-gradient(90deg,#7C3AED,#06B6D4);
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 700;
    color: white;
}
.rec-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
}
.btn-play {
    background-color: #00b4d8;
    padding: 8px 14px;
    border-radius: 8px;
    color: white !important;
    font-weight: 700;
}
img[alt="Live Camera"] {
    border-radius: 12px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown("### 🎵 TUNE TRAP")
st.markdown("<h1>Emotion-aware Music Recommender</h1>", unsafe_allow_html=True)
st.markdown(
    '',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------
# START / STOP WEBCAM BUTTONS (centered)
# ---------------------------------------------------------------
col1, col2 = st.columns([1, 1])
with col1:
    start_btn = st.button("▶ Start Webcam")
with col2:
    stop_btn = st.button("⛔ Stop Webcam")

if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ---------------------------------------------------------------
# MAIN LAYOUT: LEFT = Camera, RIGHT = Info
# ---------------------------------------------------------------
left, right = st.columns([2, 1])

# ---------------------------------------------------------------
# LEFT SIDE — CAMERA WITH CONDITIONAL PLACEHOLDER
# ---------------------------------------------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Camera")

    # SHOW PLACEHOLDER ONLY BEFORE START
    if not st.session_state.running:
        st.markdown("""
            <div style="
                width: 100%;
                height: 350px;
                background-color: #0a0f17;
                border-radius: 12px;
                margin-bottom: 15px;
            "></div>
        """, unsafe_allow_html=True)

    # space where webcam frames will appear
    frame_display = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# RIGHT SIDE — Emotion, History, Recommendations
# ---------------------------------------------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Detected Emotion")
    emotion_box = st.empty()

    st.subheader("Recent History")
    history_box = st.empty()

    st.subheader("Recommendations")
    rec_box = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Emotion Detector
# ---------------------------------------------------------------
detector = EmotionDetector(mode="mediapipe")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------
# WEBCAM LOOP
# ---------------------------------------------------------------
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not available.")
    else:
        frame_count = 0
        skip = 2

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip != 0:
                continue

            label, conf, bbox = detector.detect(frame)

            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), (124, 58, 237), 2)

            text = f"{label.upper()} ({conf:.2f})"
            cv2.putText(display, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 4)
            cv2.putText(display, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

            frame_display.image(display, caption="Live Camera", channels="RGB")

            emoji_map = {"happy": "😊", "sad": "😢", "angry": "😡", "neutral": "😐", "surprise": "😯"}
            emoji = emoji_map.get(label.lower(), "🙂")

            emotion_box.markdown(
                f'<div class="emotion-badge">{emoji} {label.upper()} ({conf:.2f})</div>',
                unsafe_allow_html=True,
            )

            st.session_state.history.append(label)
            history_box.write(", ".join(st.session_state.history[-15:]))

            recs = FALLBACK.get(label, FALLBACK["neutral"])
            html = ""
            for title, link in recs:
                html += f"""
                <div class="rec-card">
                    <b>{title}</b><br>
                    <a class="btn-play" href="{link}" target="_blank">Play ▶</a>
                </div>
                """
            rec_box.markdown(html, unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()
