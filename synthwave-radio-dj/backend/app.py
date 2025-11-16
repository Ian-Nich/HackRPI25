from flask import Flask, request, jsonify
from flask_cors import CORS
from eleven_api import generate_voice
import random
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

print("FLASK APP LOADED")

app = Flask(__name__)
CORS(app)

def load_prompt(filename):
    path = os.path.join("prompts", filename)
    if not os.path.exists(path):
        return "Missing prompt file."
    with open(path, "r") as f:
        return f.read()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True)

    if not data or 'station' not in data:
        return jsonify({"error": "Missing 'station' in request"}), 400

    station = data['station']

    # Map station → correct prompt file
    station_map = {
        "news": "news.txt",
        "weather": "weather.txt",
        "crime": "crime.txt",
        "music": "music.txt"
    }

    if station not in station_map:
        return jsonify({"error": "Unknown station"}), 400

    # Load the correct template
    text_template = load_prompt(station_map[station])

    # Fill in a retro city
    neon_cities = ["Neo-Tokyo", "Synth City", "Lazerport", "Chrome Ridge"]
    template_filled = text_template.replace("{{CITY}}", random.choice(neon_cities))

    # Generate audio via ElevenLabs
    try:
        audio_url = generate_voice(template_filled)
    except Exception as e:
        return jsonify({"error": f"Voice generation failed: {str(e)}"}), 500

    return jsonify({"audio_url": audio_url})

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
