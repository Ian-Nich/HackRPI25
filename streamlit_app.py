"""
🕹️ RetroBrain Streamlit UI
Dynamic UI morphing from 1980s → 2000s → 2020s
"""

import streamlit as st
import os
import requests
import json
import time

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="🕹️ RetroBrain",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load dynamic CSS with era interpolation"""
    css_path = os.path.join(os.path.dirname(__file__), "streamlit_styles.css")
    try:
        with open(css_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_slider_value():
    """Get slider value from session state"""
    if 'era_slider' not in st.session_state:
        st.session_state.era_slider = 50  # Default to 2000s
    return st.session_state.era_slider

def update_css_variables(slider_value):
    """
    Calculate CSS variable values based on slider position (0-100)
    0 = 1980s (fully retro)
    50 = 2000s (transitional)
    100 = 2020s (fully modern)
    """
    # Normalize slider to 0-1
    t = slider_value / 100.0
    
    # Font family interpolation
    if t < 0.33:
        # 0-33: Pure Press Start 2P
        font_family = "'Press Start 2P', monospace"
    elif t < 0.66:
        # 33-66: Transition from Press Start 2P to IBM Plex Mono
        if t < 0.5:
            # 33-50: Mostly Press Start 2P
            font_family = "'Press Start 2P', 'IBM Plex Mono', monospace"
        else:
            # 50-66: Mostly IBM Plex Mono
            font_family = "'IBM Plex Mono', 'Press Start 2P', monospace"
    else:
        # 66-100: Transition from IBM Plex Mono to Inter
        if t < 0.83:
            # 66-83: Mostly IBM Plex Mono
            font_family = "'IBM Plex Mono', 'Inter', sans-serif"
        else:
            # 83-100: Mostly Inter
            font_family = "'Inter', 'IBM Plex Mono', sans-serif"
    
    # Color interpolation (green → gray-blue → modern colors)
    def lerp_color(c1, c2, t):
        """Linear interpolation between two colors"""
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
        
        rgb1 = hex_to_rgb(c1)
        rgb2 = hex_to_rgb(c2)
        rgb = tuple(int(rgb1[i] + (rgb2[i] - rgb1[i]) * t) for i in range(3))
        return rgb_to_hex(rgb)
    
    # Primary color (green → cyan/gray → blue)
    if t < 0.5:
        primary_color = lerp_color("#00ff00", "#5b9bd5", t * 2)
    else:
        primary_color = lerp_color("#5b9bd5", "#4f46e5", (t - 0.5) * 2)
    
    # Background color (black → dark gray → light gray)
    if t < 0.5:
        bg_color = lerp_color("#000000", "#1e1e1e", t * 2)
    else:
        bg_color = lerp_color("#1e1e1e", "#f3f4f6", (t - 0.5) * 2)
    
    # Text color (green → gray → dark gray/black)
    if t < 0.5:
        text_color = lerp_color("#00ff00", "#d3d3d3", t * 2)
    else:
        text_color = lerp_color("#d3d3d3", "#111827", (t - 0.5) * 2)
    
    # Border radius (0px → 4px → 12px)
    if t < 0.5:
        border_radius = t * 8  # 0-4px
    else:
        border_radius = 4 + (t - 0.5) * 16  # 4-12px
    
    # Shadow intensity (glow → shadow → soft shadow)
    shadow_blur = t * 20  # 0-20px
    shadow_spread = max(0, (t - 0.5) * 4)  # 0-2px spread for modern
    
    # Scanline opacity (1.0 → 0.5 → 0.0)
    scanline_opacity = 1.0 - t
    
    # Glassmorphism (0 → 0 → 1)
    glass_opacity = max(0, (t - 0.5) * 2)  # 0 at 0-50, 1 at 100
    backdrop_blur = max(0, (t - 0.5) * 20)  # 0-10px
    
    # Pixelation (pixelated → auto → auto)
    image_rendering = "pixelated" if t < 0.3 else "auto"
    
    # Font size (small → medium → larger)
    base_font_size = 14 + t * 4  # 14-18px
    
    # Border width (thick → medium → thin)
    border_width = 3 - t * 2  # 3px → 1px
    
    # Return CSS variable string with direct application to all elements
    # (Since CSS variables might not update dynamically in Streamlit)
    return f"""
    :root {{
        --era-slider-value: {t};
        --primary-color: {primary_color};
        --bg-color: {bg_color};
        --text-color: {text_color};
        --border-radius: {border_radius}px;
        --shadow-blur: {shadow_blur}px;
        --shadow-spread: {shadow_spread}px;
        --scanline-opacity: {scanline_opacity};
        --glass-opacity: {glass_opacity};
        --backdrop-blur: {backdrop_blur}px;
        --image-rendering: {image_rendering};
        --base-font-size: {base_font_size}px;
        --border-width: {max(1, border_width)}px;
        --font-family: {font_family};
    }}
    
    /* Apply directly to ensure updates work */
    .main .block-container {{
        background: {bg_color} !important;
        color: {text_color} !important;
        font-family: {font_family} !important;
        font-size: {base_font_size}px !important;
        border: {max(1, border_width)}px solid {primary_color} !important;
        border-radius: {border_radius}px !important;
        box-shadow: 0 0 {shadow_blur}px {primary_color}, 
                    0 0 {shadow_blur * 2}px rgba(0, 255, 0, {scanline_opacity * 0.3}),
                    0 0 {shadow_blur * 3}px rgba(0, 255, 0, {scanline_opacity * 0.1}) !important;
        backdrop-filter: blur({backdrop_blur}px) !important;
        background: linear-gradient(135deg, 
            rgba(0, 0, 0, {1 - glass_opacity * 0.5}),
            rgba(30, 30, 30, {1 - glass_opacity * 0.5})) !important;
    }}
    
    .main::before {{
        opacity: {scanline_opacity} !important;
    }}
    
    h1, h2, h3, .retrobrain-header h1, .era-panel h3 {{
        font-family: {font_family} !important;
        color: {primary_color} !important;
    }}
    
    .retrobrain-header {{
        border: {max(1, border_width)}px solid {primary_color} !important;
        border-radius: {border_radius}px !important;
        background: rgba(0, 0, 0, {0.5 - glass_opacity * 0.3}) !important;
        box-shadow: 0 0 {shadow_blur}px {primary_color},
                    inset 0 0 {shadow_blur * 0.5}px rgba(0, 255, 0, {scanline_opacity * 0.2}) !important;
        backdrop-filter: blur({backdrop_blur}px) !important;
    }}
    
    .era-panel, .demo-card {{
        border: {max(1, border_width)}px solid {primary_color} !important;
        border-radius: {border_radius}px !important;
        backdrop-filter: blur({backdrop_blur}px) !important;
    }}
    
    img {{
        image-rendering: {image_rendering} !important;
    }}
    """

def make_prediction(text, era=None):
    """Make prediction via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text, "model_era": era},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_results():
    """Get evaluation results from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/results", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def main():
    # Load base CSS
    base_css = load_css()
    
    # Get slider value
    slider_value = get_slider_value()
    
    # Calculate and inject dynamic CSS variables
    dynamic_css = update_css_variables(slider_value)
    
    # Inject all CSS (must be first thing)
    st.markdown(f"""
    <style>
    {base_css}
    {dynamic_css}
    </style>
    """, unsafe_allow_html=True)
    
    # Force CSS update on slider change
    if 'slider_value_prev' not in st.session_state:
        st.session_state.slider_value_prev = slider_value
    
    # Re-inject CSS if slider changed (to force update)
    if st.session_state.slider_value_prev != slider_value:
        dynamic_css = update_css_variables(slider_value)
        st.markdown(f"""
        <style>
        {base_css}
        {dynamic_css}
        </style>
        """, unsafe_allow_html=True)
        st.session_state.slider_value_prev = slider_value
    
    # Header with era indicator
    era_name = "1980s RETRO" if slider_value < 33 else "2000s TRANSITION" if slider_value < 67 else "2020s MODERN"
    era_emoji = "🕹️" if slider_value < 33 else "💿" if slider_value < 67 else "🤖"
    
    st.markdown(f"""
    <div class="retrobrain-header">
        <h1>{era_emoji} RetroBrain: Reconstructing Modern Knowledge Using Outdated Models</h1>
        <p class="subtitle">RetroBrain compares how 1980s, 2000s, and modern AI models understand the same data, revealing the evolution of machine intelligence across time.</p>
        <div class="era-indicator">Current Era: <span class="era-value">{era_name}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Era Slider")
        st.markdown("**Move slider to morph UI through eras**")
        
        # Main slider
        slider_value = st.slider(
            "UI Era (0 = Retro, 100 = Modern)",
            min_value=0,
            max_value=100,
            value=slider_value,
            step=1,
            key='era_slider',
            help="Slide to transform the entire UI through design eras"
        )
        # Don't modify session_state after widget creation - use the return value directly
        
        # Era labels
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='era-label'>1980s</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='era-label'>2000s</div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='era-label'>2020s</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Controls
        st.markdown("### 🎮 Controls")
        if st.button("🔄 Train Models", use_container_width=True):
            with st.spinner("Training models..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/train", json={"dataset_type": "sentiment"})
                    if response.status_code == 200:
                        st.success("Training started!")
                    else:
                        st.error("Training failed")
                except:
                    st.error("Could not connect to API")
        
        if st.button("📊 Load Results", use_container_width=True):
            results = get_results()
            if results:
                st.success("Results loaded!")
                st.session_state.results = results
            else:
                st.error("Could not load results")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **RetroBrain** compares ML models across three eras:
        - **1980s**: Logistic Regression, Naive Bayes
        - **2000s**: Random Forest, SVM
        - **2020s**: Gemini API (LLM)
        """)
    
    # Main content area
    # Input section
    st.markdown("### 📝 Text Input")
    text_input = st.text_area(
        "Enter text for sentiment analysis:",
        placeholder="Type your text here...",
        height=100,
        key="text_input"
    )
    
    # Predict button and results
    col1, col2 = st.columns([3, 1])
    with col1:
        predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)
    with col2:
        predict_all = st.button("🌐 Predict All Eras", use_container_width=True)
    
    if predict_clicked or predict_all:
        if text_input:
            with st.spinner("Making prediction..."):
                if predict_all:
                    result = make_prediction(text_input)
                else:
                    # Determine era from slider
                    era = None
                    if slider_value < 33:
                        era = "1980s"
                    elif slider_value < 67:
                        era = "2000s"
                    else:
                        era = "2020s"
                    result = make_prediction(text_input, era)
                
                if result and "error" not in result.get("1980s", {}):
                    st.session_state.prediction_result = result
                    st.session_state.prediction_text = text_input
                else:
                    st.error("Prediction failed. Make sure models are trained.")
        else:
            st.warning("Please enter some text first.")
    
    # Display results in era panels
    if 'prediction_result' in st.session_state and 'prediction_text' in st.session_state:
        st.markdown("### 🎯 Predictions Across Eras")
        
        result = st.session_state.prediction_result
        
        # Three columns for three eras
        col1, col2, col3 = st.columns(3)
        
        eras = [
            ("1980s", "📼", result.get("1980s", {})),
            ("2000s", "💿", result.get("2000s", {})),
            ("2020s", "🤖", result.get("2020s", {}))
        ]
        
        for i, (era_name, emoji, era_result) in enumerate(eras):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="era-panel">
                    <h3>{emoji} {era_name} Era</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if era_result and "error" not in era_result:
                    prediction = era_result.get("prediction", 0)
                    probabilities = era_result.get("probabilities", [0.5, 0.5])
                    model = era_result.get("model", "unknown")
                    
                    pred_text = "POSITIVE" if prediction == 1 else "NEGATIVE"
                    confidence = max(probabilities) * 100
                    
                    st.metric("Prediction", pred_text)
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown(f"**Model:** {model}")
                    
                    # Confidence bar
                    st.progress(confidence / 100)
                else:
                    error_msg = era_result.get("error", "No prediction available")
                    st.error(f"Error: {error_msg}")
        
        # AI Explanation (2020s only)
        if "2020s" in result and "error" not in result["2020s"]:
            st.markdown("### 🤖 AI Explanation (2020s)")
            with st.expander("View Gemini Explanation", expanded=True):
                explanation = result["2020s"].get("explanation", "No explanation available")
                st.info(explanation)
    
    # Demo cards showing UI transformation
    st.markdown("### 🎨 UI Transformation Demo")
    st.markdown("These cards demonstrate how the UI morphs across eras:")
    
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    demo_cards = [
        ("1980s Style", "Pixel fonts, green CRT, scanlines, blocky borders"),
        ("2000s Style", "Monospaced fonts, gray-blue, rounded corners, gradients"),
        ("2020s Style", "Clean fonts, glassmorphism, soft shadows, smooth motion")
    ]
    
    for i, (title, desc) in enumerate(demo_cards):
        with [demo_col1, demo_col2, demo_col3][i]:
            st.markdown(f"""
            <div class="demo-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Leaderboard section
    if 'results' in st.session_state:
        st.markdown("### 📊 Model Leaderboard")
        results = st.session_state.results
        
        if "eras" in results:
            leaderboard_data = []
            for era_name, era_models in results["eras"].items():
                for model_name, model_result in era_models.items():
                    accuracy = model_result.get("accuracy", 0) * 100
                    leaderboard_data.append({
                        "Era": era_name.upper(),
                        "Model": model_name,
                        "Accuracy": f"{accuracy:.2f}%"
                    })
            
            if leaderboard_data:
                leaderboard_data.sort(key=lambda x: float(x["Accuracy"].rstrip("%")), reverse=True)
                st.dataframe(leaderboard_data, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="retrobrain-footer">
        <p>🕹️ Built with ❤️ for HackRPI 2025 | RetroBrain v1.0</p>
        <p>✨ Powered by AI • Revealing the Evolution of Machine Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

