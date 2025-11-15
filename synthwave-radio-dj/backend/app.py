from flask import Flask, request, jsonify
from flask_cors import CORS
from eleven_api import generate_voice
import random

app = Flask(__name__)

CORS(app)

def load_prompt(filename):
    with open(f"prompts/{filename}") as f:
        return f.read()


@app.route('/generate', methods=['POST'])
def generate():
    station = request.json['station']

    # Each station corresponds to a different vibe
    if station == "news":
        text_template = load_prompt("news.txt")
    elif station == "weather":
        text_template = load_prompt("weather.txt")
    elif station == "traffic":
        text_template = load_prompt("traffic.txt")
    else:
        return jsonify({"error": "Unknown station"}), 400

    # Fill in some random retro elements
    neon_cities = ["Neo-Tokyo", "Synth City", "Lazerport", "Chrome Ridge"]
    template_filled = text_template.replace("{{CITY}}", random.choice(neon_cities))

    # Generate audio with ElevenLabs
    audio_url = generate_voice(template_filled)

    return jsonify({"audio_url": audio_url})


if __name__ == "__main__":
    app.run(debug=True)
