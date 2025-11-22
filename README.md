## 🎧 TUNE TRAP – Emotion-Based Music Recommendation System

A real-time facial emotion detection system that recommends music based on your mood.

TUNE TRAP uses MediaPipe Face Mesh, OpenCV, and a heuristic emotion detection algorithm to classify emotions from webcam video frames. Based on the detected emotion, the system automatically generates YouTube music recommendations using the YouTube Search API (or offline fallback recommendations if no API key is provided).

🚀 Features
🎭 Emotion Detection

## Detects 5 emotions in real time:

🙂 Happy

😢 Sad

😡 Angry

😮 Surprise

😐 Neutral

## Uses:

MediaPipe FaceMesh

Custom geometric landmark-based emotion heuristic

🎵 Music Recommendation

For every detected emotion, TUNE TRAP searches YouTube and recommends top relevant songs:

Uses YouTube Data API (optional)

Works offline using fallback music lists

🖥 Real-time Webcam UI

Start / Stop webcam

Live emotion result

Song list generated instantly

## 📁 Project Structure
TUNE_TRAP_FULL/
│
├── app.py                    # Streamlit app
├── advanced_app.py           # (Optional UI version)
├── emotion_detector.py       # Core emotion detection logic
├── utils.py                  # YouTube search + fallbacks
├── accuracy_test.py          # Evaluate model accuracy
│
├── assets/
│   ├── logo.png
│   └── logo_dark.png
│
├── models/                   # (Optional) store Keras emotion model here
│   └── trained_model.h5
│
├── .streamlit/
│   ├── secrets.toml          # Contains YouTube API key
│   └── secrets.toml.example
│
├── requirements.txt          # Core dependencies
├── README.md
└── train_emotion.py          # Training script (optional)

## 🛠 Installation (Local System – VS Code)
1️⃣ Create Virtual Environment
python -m venv venv


Activate it:

Windows PowerShell

venv\Scripts\Activate.ps1


Mac/Linux

source venv/bin/activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Your YouTube API Key (Optional But Recommended)

Create file:

.streamlit/secrets.toml


Paste:

## YOUTUBE_API_KEY = "YOUR_API_KEY_HERE"


If no key is added → offline fallback song recommendations will be used.

## ▶️ Run the Application
streamlit run app.py


Then open the URL that appears (usually):

➡ http://localhost:8501/

In the app:

Click Start Webcam

See emotion detection in real time

Music suggestions appear instantly

## 📦 Deployment
Streamlit Cloud

Upload your GitHub repo → Select app.py as the main entry file.

Render / Hugging Face

If deploying without Streamlit, replace UI with:

FastAPI

Flask

Gradio

(Ask me if you want the FastAPI or Gradio version!)

## 📊 Model Accuracy Testing

Run:

python accuracy_test.py


This checks accuracy on your dataset using the heuristic or ML model.

❤️ Author

Akash Kapoor
🎓 B.Tech (CSE), PSIT Kanpur
💼 Emotion AI & ML Developer

## 📽 Demo Video

https://youtu.be/aFha5go2teY
