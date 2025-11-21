import cv2
import numpy as np
import mediapipe as mp
import os

EMOTIONS = ["neutral", "happy", "sad", "surprise", "angry"]

class MediapipeHeuristic:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                               refine_landmarks=True, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def predict(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return "neutral", 0.0

        lm = res.multi_face_landmarks[0].landmark
        def p(i):
            return np.array([int(lm[i].x * w), int(lm[i].y * h)])

        # Simple heuristics using mediapipe face mesh indices
        top_lip = p(13)
        bottom_lip = p(14)
        mouth_open = np.linalg.norm(top_lip - bottom_lip) / h

        left_eye_top = p(159)
        left_eye_bottom = p(145)
        eye_open = np.linalg.norm(left_eye_top - left_eye_bottom) / h

        brow_inner = p(10)
        brow_outer = p(338)
        brow_dist = np.linalg.norm(brow_inner - brow_outer) / w

        # heuristics thresholds (tuned for typical webcams)
        if mouth_open > 0.035:
            return "surprise", float(mouth_open)
        if eye_open < 0.007 and brow_dist > 0.02:
            return "angry", 1.0 - float(eye_open)
        if mouth_open > 0.02 and eye_open > 0.008:
            return "happy", float(mouth_open)
        # fallback neutral
        return "neutral", 0.35

class KerasEmotionModel:
    def __init__(self, model_path=None, target_size=(224,224), labels=EMOTIONS):
        self.model_path = model_path
        self.target_size = target_size
        self.labels = labels
        self.model = None
        try:
            from tensorflow.keras.models import load_model
            if model_path and os.path.exists(model_path):
                self.model = load_model(model_path)
                print("Loaded Keras model:", model_path)
            else:
                print("Keras model path not found or not provided. Keras mode will not run.")
        except Exception as e:
            print("TensorFlow not available or failed to load model:", e)
            self.model = None

    def preprocess_face(self, face_bgr):
        import cv2
        from tensorflow.keras.preprocessing.image import img_to_array
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, self.target_size)
        face_array = img_to_array(face_resized)
        face_array = face_array.astype("float") / 255.0
        return np.expand_dims(face_array, axis=0)

    def predict(self, face_bgr):
        if self.model is None:
            return "neutral", 0.0
        x = self.preprocess_face(face_bgr)
        preds = self.model.predict(x)
        idx = int(np.argmax(preds))
        prob = float(np.max(preds))
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        return label, prob

class EmotionDetector:
    def __init__(self, mode="mediapipe", keras_model_path=None):
        self.mode = mode
        if mode == "mediapipe":
            self.detector = MediapipeHeuristic()
        elif mode == "keras":
            self.detector = KerasEmotionModel(model_path=keras_model_path)
        else:
            raise ValueError("mode must be 'mediapipe' or 'keras'")
        # Haar cascade for face crop (used in keras mode)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame):
        if self.mode == "mediapipe":
            label, conf = self.detector.predict(frame)
            return label, conf, None
        # keras mode: detect face and run model (if present)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
        if len(faces) == 0:
            return "neutral", 0.0, None
        faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w].copy()
        label, conf = self.detector.predict(face)
        return label, conf, (x, y, w, h)
