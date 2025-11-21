from googleapiclient.discovery import build
import streamlit as st

FALLBACK = {
    "happy": [("Upbeat Pop Mix", "https://www.youtube.com/results?search_query=upbeat+happy+songs")],
    "sad": [("Sad Songs Collection", "https://www.youtube.com/results?search_query=sad+songs")],
    "angry": [("Calming Music", "https://www.youtube.com/results?search_query=calming+music")],
    "surprise": [("Energetic Playlist", "https://www.youtube.com/results?search_query=energetic+songs")],
    "neutral": [("Chill Vibes", "https://www.youtube.com/results?search_query=chill+vibes+playlist")]
}

def emotion_to_query(emotion):
    mapping = {
        "happy": "upbeat happy songs playlist",
        "sad": "emotional sad songs playlist",
        "angry": "calming music for anger playlist",
        "surprise": "energetic music playlist",
        "neutral": "chill vibes playlist"
    }
    return mapping.get(emotion.lower(), f"{emotion} mood songs")

def get_youtube_recommendations(emotion, api_key, max_results=5):
    if not api_key:
        # return fallback links
        return FALLBACK.get(emotion.lower(), FALLBACK["neutral"])
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        q = emotion_to_query(emotion)
        req = youtube.search().list(part="snippet", q=q, type="video", maxResults=max_results)
        res = req.execute()
        videos = []
        for item in res.get("items", []):
            title = item["snippet"]["title"]
            vid = item["id"]["videoId"]
            videos.append((title, f"https://www.youtube.com/watch?v={vid}"))
        if not videos:
            return FALLBACK.get(emotion.lower(), FALLBACK["neutral"])
        return videos
    except Exception as e:
        st.warning(f"Could not fetch YouTube results: {e}")
        return FALLBACK.get(emotion.lower(), FALLBACK["neutral"])
