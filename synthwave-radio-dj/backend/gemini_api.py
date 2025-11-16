import os
import google.generativeai as genai

# Initialize Gemini client
client_initialized = False

def initialize_client():
    """Initialize Gemini client if not already done."""
    global client_initialized
    if not client_initialized:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable missing")
        genai.configure(api_key=api_key)
        client_initialized = True

def generate_story(station_type, city):
    """
    Generate a dynamic story using Gemini based on station type and city.
    Returns a script in 80s synthwave style.
    """
    initialize_client()
    
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
        # Prefer Gemini 2.0 Flash models only (per latest API guidance)
        model = None
        model_errors = []
        model_candidates = [
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-001",
        ]

        for name in model_candidates:
            try:
                model = genai.GenerativeModel(name)
                break
            except Exception as err:
                model_errors.append(f"{name}: {err}")

        if model is None:
            raise RuntimeError(
                "Unable to initialize Gemini model. Tried: " + "; ".join(model_errors)
            )
        
        # Create the full prompt with system context
        system_prompt = "You are a retro 80s synthwave radio DJ. Your style is atmospheric, mysterious, and cyberpunk-inspired. Keep responses concise and engaging (2-4 sentences)."
        user_prompt = station_prompts[station_type]
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate content
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.8
            )
        )
        
        story = response.text.strip()
        return story
    
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")

