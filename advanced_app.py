# advanced_app.py
# Optional improved app with threaded capture and nicer UI elements.
# This is an alternative to app.py. You can run either: `streamlit run app.py` or `streamlit run advanced_app.py`
import streamlit as st
import cv2
import time
import threading
import queue
from emotion_detector import EmotionDetector
from utils import get_youtube_recommendations

st.set_page_config(page_title="TUNE TRAP (Advanced)", layout="wide")
st.sidebar.title("TUNE TRAP — Advanced")
mode = st.sidebar.selectbox("Mode", ["mediapipe (fast)", "keras (accurate)"])
model_path = st.sidebar.text_input("Keras model path", "models/trained_model.h5")
api_key = st.sidebar.text_input("YouTube API Key (optional)")
start = st.sidebar.button("Start")
stop = st.sidebar.button("Stop")
skip = st.sidebar.slider("Frame skip", 1, 6, 2)

detector = EmotionDetector(mode="keras" if mode.startswith("keras") else "mediapipe", keras_model_path=model_path if mode.startswith("keras") else None)

col1, col2 = st.columns([2,1])
frame_placeholder = col1.image([], channels="RGB")
emotion_placeholder = col2.empty()
history_placeholder = col2.empty()
rec_placeholder = col2.empty()

q = queue.Queue(maxsize=2)
running = threading.Event()

def capture_loop(q, running, skip):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return
    frame_count = 0
    try:
        while running.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame_count += 1
            if frame_count % skip != 0:
                continue
            label, conf, bbox = detector.detect(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if bbox:
                x,y,w,h = bbox
                cv2.rectangle(frame_rgb, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame_rgb, f"{label} ({conf:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            try:
                if q.full():
                    q.get_nowait()
                q.put_nowait((frame_rgb, label, conf))
            except:
                pass
            time.sleep(0.01)
    finally:
        cap.release()

capture_thread = None
history = []

if start:
    running.set()
    capture_thread = threading.Thread(target=capture_loop, args=(q, running, skip), daemon=True)
    capture_thread.start()
    st.success("Webcam started")

if stop:
    running.clear()
    st.warning("Webcam stopped")

try:
    while True:
        if not q.empty():
            frame_rgb, label, conf = q.get_nowait()
            frame_placeholder.image(frame_rgb)
            emotion_placeholder.markdown(f"### {label.upper()} ({conf:.2f})")
            history.append(label)
            if len(history) > 30:
                history = history[-30:]
            history_placeholder.write("Recent: " + ", ".join(history[::-1][:8]))
            if api_key:
                recs = get_youtube_recommendations(label, api_key)
            else:
                recs = []
            if recs:
                md = ""
                for t,l in recs:
                    md += f"- [{t}]({l})  \n"
                rec_placeholder.markdown(md)
            else:
                rec_placeholder.info("Provide YouTube API key for live recommendations.")
        else:
            time.sleep(0.05)
        if not running.is_set():
            break
except Exception as e:
    st.error(f"Error: {e}")
finally:
    running.clear()
