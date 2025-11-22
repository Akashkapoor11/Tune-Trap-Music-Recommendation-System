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

## Deployment Link
https://youtu.be/aFha5go2teY


