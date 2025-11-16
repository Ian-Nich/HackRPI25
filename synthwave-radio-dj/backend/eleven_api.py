import os
import requests

API_KEY = os.getenv("ELEVEN_API_KEY")
print("DEBUG: API KEY =", API_KEY)

VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # use any voice you want
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/" + VOICE_ID

def generate_voice(text):
    if not API_KEY:
        raise ValueError("ELEVEN_API_KEY is missing from environment variables")

    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(ELEVEN_URL, json=data, headers=headers)

    print("ElevenLabs status:", response.status_code)

    if response.status_code != 200:
        print("ElevenLabs returned:", response.content[:200])
        raise RuntimeError("ElevenLabs TTS failed")

    # Save output audio
    filename = "static/output.mp3"
    os.makedirs("static", exist_ok=True)

    with open(filename, "wb") as f:
        f.write(response.content)

    # Flask will serve this file
    return f"http://127.0.0.1:5000/{filename}"
