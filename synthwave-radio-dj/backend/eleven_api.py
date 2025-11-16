import os
import requests
import base64

# Support both ELEVEN_API_KEY and ELEVENLABS_API_KEY for flexibility
API_KEY = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # change if you want
ELEVEN_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

def generate_voice_base64(text):
    if not API_KEY:
        raise ValueError("ELEVEN_API_KEY or ELEVENLABS_API_KEY environment variable missing")

    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    # make TTS request
    response = requests.post(ELEVEN_URL, json=payload, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs error: {response.text}")

    # convert binary audio -> base64
    audio_bytes = response.content
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return audio_b64
