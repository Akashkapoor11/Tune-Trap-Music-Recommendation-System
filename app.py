import streamlit as st
import cv2
import time
from emotion_detector import EmotionDetector
from utils import get_youtube_recommendations, emotion_to_query

st.set_page_config(page_title="TUNE TRAP", layout="wide")

st.sidebar.title("TUNE TRAP")
st.sidebar.markdown("Real-time emotion detection → music recommender")
mode = st.sidebar.selectbox("Detection mode", ["mediapipe (fast, default)", "keras (accurate, optional)"])
model_path = st.sidebar.text_input("Keras model path (models/trained_model.h5)", value="models/trained_model.h5")
api_key = st.sidebar.text_input("YouTube API Key (optional)", value="")
start = st.sidebar.button("Start Webcam")
stop = st.sidebar.button("Stop Webcam")
skip_frames = st.sidebar.slider("Frame skip (1 = every frame, higher = faster)", 1, 6, 2)

st.title("🎵 TUNE TRAP — Emotion-aware music recommender")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Webcam")
    frame_placeholder = st.empty()

with col2:
    st.header("Detected Emotion")
    emotion_placeholder = st.empty()
    st.markdown("### Recent history")
    history_placeholder = st.empty()
    st.markdown("### Recommendations")
    rec_placeholder = st.empty()

if 'running' not in st.session_state:
    st.session_state['running'] = False

detector = EmotionDetector(mode="keras" if mode.startswith("keras") else "mediapipe", keras_model_path=model_path if mode.startswith("keras") else None)

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Make sure your webcam is connected and not used by another app.")
        return
    history = []
    frame_count = 0
    try:
        while st.session_state['running']:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to read from webcam.")
                break
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            label, conf, bbox = detector.detect(frame)
            # annotate frame for display
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if bbox:
                x,y,w,h = bbox
                cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display, f"{label} ({conf:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            frame_placeholder.image(display, channels="RGB")
            emotion_placeholder.markdown(f"#### **{label.upper()}** — confidence: {conf:.2f}")
            history.append(label)
            if len(history) > 20:
                history = history[-20:]
            history_placeholder.write(", ".join(history[::-1][:10]))
            # recommendations
            try:
                recs = get_youtube_recommendations(label, api_key, max_results=5)
            except Exception as e:
                recs = []
            if recs:
                md = ""
                for title, link in recs:
                    md += f"- [{title}]({link})  \n"
                rec_placeholder.markdown(md, unsafe_allow_html=True)
            else:
                rec_placeholder.info("No recommendations (provide YouTube API key or use offline mode).")
            time.sleep(0.03)
    finally:
        cap.release()

if start:
    st.session_state['running'] = True
    st.success("Starting webcam...")
    run_webcam()

if stop:
    st.session_state['running'] = False
    st.warning("Stopping webcam...")

st.markdown("---")
st.markdown("**Notes:**")
st.markdown("- Default mode uses Mediapipe landmarks and heuristics (works out-of-the-box).")
st.markdown("- Keras mode requires you to put a trained model at `models/trained_model.h5` (optional).")
st.markdown("- If you do not provide a YouTube API key, the app will use safe offline recommendations.")
st.markdown("- To run: `pip install -r requirements.txt` then `streamlit run app.py` in this folder.")
