# TUNE TRAP

Real-time emotion detection + emotion-driven music recommendations.

## Overview
TUNE TRAP captures facial expressions from your webcam, detects the user's emotion (happy, sad, angry, surprise, neutral) and provides music suggestions via YouTube search. The app is built to run locally and is deployed using Streamlit for a responsive UI.

## What's included
- `app.py` — Streamlit application (main)
- `emotion_detector.py` — Mediapipe heuristic + optional Keras model support
- `utils.py` — YouTube helper (uses API key, fallback offline links if not provided)
- `requirements.txt` — Python dependencies
- `models/` — place your trained Keras model here as `trained_model.h5` (optional)
- `assets/` — optional images or logo

## Quick start (VS Code / Local)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On mac/linux
   source venv/bin/activate
   # On Windows (PowerShell)
   venv\\Scripts\\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Add YouTube API key:
   - Create `.streamlit/secrets.toml` with the line:
     ```toml
     YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"
     ```
   - Or paste your API key into the app sidebar when it runs.

4. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

5. Click **Start Webcam** in the sidebar. Allow webcam access if prompted by your OS.

## Notes & tips
- The app defaults to **mediapipe** heuristic mode which works out-of-the-box and requires no model file.
- **Keras mode** requires a trained model (`models/trained_model.h5`). The project includes a training script idea in comments in the repository; training is optional.
- If you want higher accuracy, collect webcam-face images for your domain and fine-tune a transfer-learning model (MobileNet/EfficientNet) — training scripts are not included in this zip to keep size small.
- If YouTube API calls fail or you don't supply a key, the app shows safe fallback playlist links.
- On some systems, OpenCV may use a different camera index (0,1...). Edit `cv2.VideoCapture(0)` in `app.py` if needed.

## To include in your GroundTruth submission
- Add a short demo video (screen recording) showing the app detecting different emotions and the changing recommendations.
- Link to the GitHub repo and include model/training notes and sample accuracy metric (on your validation set).

Good luck — paste this text into your GroundTruth form and attach the zip!
