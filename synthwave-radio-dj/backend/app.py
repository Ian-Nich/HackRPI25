from flask import Flask, request, jsonify
from flask_cors import CORS
from eleven_api import generate_voice_base64
from gpt_api import generate_story
from dotenv import load_dotenv
import random
import os
import traceback

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

def load_prompt(filename):
    """Load static prompt template as fallback."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "prompts", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r") as f:
        return f.read()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()

        if not data or "station" not in data:
            return jsonify({"error": "Missing station"}), 400

        station = data["station"]

        station_map = {
            "news": "news.txt",
            "weather": "weather.txt",
            "crime": "crime.txt",
            "music": "music.txt"
        }

        if station not in station_map:
            return jsonify({"error": "Invalid station"}), 400

        # Generate dynamic story using GPT, with fallback to static prompts
        city = random.choice(["Neo-Tokyo", "Synth City", "Lazerport", "Chrome Ridge"])
        script = None
        
        # Try GPT first
        try:
            script = generate_story(station, city)
            print(f"✓ GPT generated script for {station} in {city}: {script[:100]}...")
        except ValueError as e:
            # API key missing - use fallback
            print(f"⚠ GPT API key missing, using static prompt: {str(e)}")
        except RuntimeError as e:
            # OpenAI API error (quota exceeded, etc.) - use fallback
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                print(f"⚠ OpenAI quota exceeded, using static prompt fallback")
            else:
                print(f"⚠ OpenAI API error, using static prompt fallback: {error_msg}")
        except Exception as e:
            # Other GPT errors - use fallback
            print(f"⚠ GPT generation error, using static prompt fallback: {str(e)}")
        
        # Fallback to static prompts if GPT failed
        if script is None:
            try:
                template = load_prompt(station_map[station])
                script = template.replace("{{CITY}}", city)
                print(f"✓ Using static prompt for {station} in {city}")
            except Exception as e:
                print(f"ERROR loading static prompt: {str(e)}")
                return jsonify({"error": f"Failed to generate script: {str(e)}"}), 500

        # Convert script to audio using ElevenLabs
        try:
            audio_b64 = generate_voice_base64(script)
        except ValueError as e:
            # API key missing
            print(f"ERROR: {str(e)}")
            return jsonify({"error": "ElevenLabs API key not configured. Please set ELEVEN_API_KEY or ELEVENLABS_API_KEY in .env file."}), 500
        except RuntimeError as e:
            # ElevenLabs API error
            print(f"ERROR: {str(e)}")
            return jsonify({"error": f"ElevenLabs API error: {str(e)}"}), 500
        except Exception as e:
            # Other errors
            print(f"ERROR: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Audio generation error: {str(e)}"}), 500

        return jsonify({"audio": audio_b64})
    
    except Exception as e:
        print(f"ERROR in generate endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
