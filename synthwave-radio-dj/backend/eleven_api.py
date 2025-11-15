import requests

API_KEY = "YOUR_ELEVENLABS_API_KEY"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"   # Example voice

def generate_voice(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.3,
            "similarity_boost": 0.7
        }
    }

    response = requests.post(url, json=payload, headers=headers)

    filename = "static/output.mp3"
    with open(filename, "wb") as f:
        f.write(response.content)

    return "/static/output.mp3"
