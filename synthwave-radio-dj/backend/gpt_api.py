import os
from openai import OpenAI

# Initialize OpenAI client
client = None

def get_client():
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable missing")
        client = OpenAI(api_key=api_key)
    return client

def generate_story(station_type, city):
    """
    Generate a dynamic story using GPT based on station type and city.
    Returns a script in 80s synthwave style.
    """
    client = get_client()

    # Define prompts for each station type
    station_prompts = {
        "news": f"""You are a retro 80s synthwave radio DJ reporting news from {city}.
Create a brief, atmospheric news report (2-3 sentences) in a neon-lit cyberpunk style.
Include references to synthwave elements like neon lights, digital landscapes, retro-futuristic technology, or cyberpunk themes.
Keep it engaging and mysterious, like a late-night radio broadcast from a dystopian future.""",

        "weather": f"""You are a retro 80s synthwave radio DJ reporting weather from {city}.
Create a brief, atmospheric weather forecast (2-3 sentences) in a neon-lit cyberpunk style.
Use creative, synthwave-themed weather descriptions like "neon rain," "electric storms," "pink cloud formations," or "digital fog."
Make it sound like a late-night radio broadcast from a retro-futuristic world.""",

        "crime": f"""You are a retro 80s synthwave radio DJ reporting crime news from {city}.
Create a brief, atmospheric crime report (3-4 sentences) in a neon-lit cyberpunk style.
Include references to synthwave elements like lightcycles, neon alleys, cybernetic signatures, or digital heists.
Make it sound mysterious and engaging, like a late-night radio broadcast from a dystopian future.""",

        "music": f"""You are a retro 80s synthwave radio DJ hosting a music show from {city}.
Create a brief, atmospheric DJ commentary (3-4 sentences) in a neon-lit cyberpunk style.
Talk about the music, the vibe, the synthwave atmosphere. Include references to outrun, retro synth, cosmic soundscapes, or midnight cruising.
Make it sound like a late-night radio broadcast from a retro-futuristic world."""
    }

    if station_type not in station_prompts:
        raise ValueError(f"Invalid station type: {station_type}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency, can change to gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are a retro 80s synthwave radio DJ. Your style is atmospheric, mysterious, and cyberpunk-inspired. Keep responses concise and engaging."},
                {"role": "user", "content": station_prompts[station_type]}
            ],
            max_tokens=200,
            temperature=0.8
        )

        story = response.choices[0].message.content.strip()
        return story

    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}")
