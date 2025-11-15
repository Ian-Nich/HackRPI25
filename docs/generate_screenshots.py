"""
Generate demo screenshot placeholders
Creates fake screenshot PNGs for pitch deck
"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_dashboard_screenshot():
    """Create dashboard screenshot placeholder"""
    width, height = 1400, 900
    img = Image.new('RGB', (width, height), color='#0a0a0a')
    draw = ImageDraw.Draw(img)
    
    # Title
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Header
    draw.rectangle([0, 0, width, 100], fill='#000000', outline='#00ff00', width=3)
    draw.text((width//2 - 200, 30), "🕹️ RETROBRAIN DASHBOARD", 
              fill='#00ff00', font=font_large)
    
    # Time Slider
    draw.rectangle([50, 150, width - 50, 250], fill='#0a0a0a', outline='#00ffff', width=2)
    draw.text((width//2 - 150, 170), "TIME SLIDER: 1980s → 2000s → 2020s", 
              fill='#00ffff', font=font_medium)
    draw.rectangle([100, 210, width - 100, 230], fill='#000000', outline='#00ffff', width=2)
    draw.rectangle([width//2 - 30, 210, width//2 + 30, 230], fill='#00ffff')
    
    # Three Era Panels
    panel_width = (width - 100) // 3
    eras = [
        ('1980s', '#00ff00', 'Logistic Regression'),
        ('2000s', '#00ffff', 'Random Forest'),
        ('2020s', '#ff00ff', 'Gemini API')
    ]
    
    for i, (era, color, model) in enumerate(eras):
        x = 50 + i * (panel_width + 10)
        y = 300
        panel_height = 400
        
        draw.rectangle([x, y, x + panel_width, y + panel_height], 
                      fill='#0a0a0a', outline=color, width=3)
        draw.text((x + 20, y + 20), f"ERA: {era}", fill=color, font=font_medium)
        draw.text((x + 20, y + 60), f"Model: {model}", fill=color, font=font_small)
        
        # Prediction box
        draw.rectangle([x + 20, y + 120, x + panel_width - 20, y + 300], 
                      fill='#000000', outline=color, width=2)
        draw.text((x + 40, y + 140), "PREDICTION:", fill=color, font=font_small)
        draw.text((x + 40, y + 180), "POSITIVE", fill=color, font=font_large)
        draw.text((x + 40, y + 230), "Confidence: 85%", fill=color, font=font_small)
    
    # Footer
    draw.text((width//2 - 150, height - 50), "RETROBRAIN - HackRPI 2025", 
              fill='#00ff00', font=font_medium)
    
    output_path = 'retrobrain/demo_screenshots/dashboard.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✓ Created {output_path}")


def create_terminal_screenshot():
    """Create retro terminal screenshot"""
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), color='#0a0a0a')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    # Terminal window
    draw.rectangle([50, 50, width - 50, height - 50], 
                  fill='#000000', outline='#00ff00', width=3)
    
    # Header
    draw.rectangle([50, 50, width - 50, 120], fill='#000000', outline='#00ff00', width=2)
    draw.text((80, 70), "🕹️ RETROBRAIN COMPUTER TERMINAL v1.0", 
              fill='#00ff00', font=font_large)
    
    # Terminal text
    lines = [
        ">>> RETROBRAIN COMPUTER TERMINAL v1.0 <<<",
        ">>> INITIALIZING SYSTEMS...",
        ">>> LOADING 1980s MODELS...",
        ">>> LOADING 2000s MODELS...",
        ">>> LOADING 2020s MODELS...",
        ">>> READY...",
        "",
        "RETROBRAIN: RECONSTRUCTING MODERN KNOWLEDGE",
        "USING OUTDATED MODELS",
        "",
        ">>> QUICK START <<<",
        "[TRAIN MODELS] [GO TO DASHBOARD] [LOAD RESULTS]",
        "",
        ">>> ABOUT <<<",
        "RetroBrain compares ML models across three eras:",
        "  ERA 1 (1980s): Logistic Regression, Naive Bayes",
        "  ERA 2 (2000s): Random Forest, SVM",
        "  ERA 3 (2020s): Gemini API (LLM)"
    ]
    
    y_offset = 150
    for i, line in enumerate(lines):
        draw.text((80, y_offset + i * 30), line, fill='#00ff00', font=font_medium)
    
    output_path = 'retrobrain/demo_screenshots/retro_terminal.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✓ Created {output_path}")


def create_accuracy_chart():
    """Create accuracy comparison chart screenshot"""
    width, height = 1200, 600
    img = Image.new('RGB', (width, height), color='#0a0a0a')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    # Title
    draw.text((width//2 - 200, 30), "ACCURACY COMPARISON ACROSS ERAS", 
              fill='#00ffff', font=font_large)
    
    # Bar chart
    models = [
        ('1980s\nLogistic Regression', 0.72, '#00ff00'),
        ('1980s\nNaive Bayes', 0.68, '#00ff00'),
        ('2000s\nRandom Forest', 0.85, '#00ffff'),
        ('2000s\nSVM RBF', 0.82, '#00ffff'),
        ('2020s\nGemini', 0.92, '#ff00ff')
    ]
    
    bar_width = 150
    bar_spacing = 50
    start_x = 100
    max_height = 400
    base_y = height - 100
    
    for i, (name, accuracy, color) in enumerate(models):
        x = start_x + i * (bar_width + bar_spacing)
        bar_height = int(accuracy * max_height)
        y = base_y - bar_height
        
        draw.rectangle([x, y, x + bar_width, base_y], 
                      fill=color, outline=color, width=2)
        
        # Label
        draw.text((x, base_y + 10), name, fill=color, font=font_medium)
        # Value
        draw.text((x + bar_width//2 - 30, y - 30), f"{accuracy:.2f}", 
                 fill=color, font=font_medium)
    
    # Y-axis label
    draw.text((30, height//2 - 100), "Accuracy", fill='#00ff00', font=font_medium)
    
    output_path = 'retrobrain/demo_screenshots/accuracy_chart.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✓ Created {output_path}")


def create_time_slider_screenshot():
    """Create time slider screenshot"""
    width, height = 1400, 400
    img = Image.new('RGB', (width, height), color='#0a0a0a')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 32)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    # Title
    draw.text((width//2 - 200, 30), "TIME SLIDER: MORPH BETWEEN ERAS", 
              fill='#00ffff', font=font_large)
    
    # Slider track
    track_y = height // 2
    track_x1 = 200
    track_x2 = width - 200
    track_height = 40
    
    draw.rectangle([track_x1, track_y - track_height//2, 
                   track_x2, track_y + track_height//2], 
                  fill='#000000', outline='#00ffff', width=3)
    
    # Slider thumb (at 2000s position)
    thumb_x = width // 2
    thumb_size = 60
    draw.ellipse([thumb_x - thumb_size//2, track_y - thumb_size//2,
                  thumb_x + thumb_size//2, track_y + thumb_size//2],
                 fill='#00ffff', outline='#00ff00', width=3)
    
    # Labels
    draw.text((track_x1 - 80, track_y + 50), "1980s", fill='#00ff00', font=font_medium)
    draw.text((thumb_x - 40, track_y + 50), "2000s", fill='#00ffff', font=font_medium)
    draw.text((track_x2 - 50, track_y + 50), "2020s", fill='#ff00ff', font=font_medium)
    
    # Era info boxes
    eras_info = [
        ("STATISTICAL METHODS", "Logistic Regression, Naive Bayes", track_x1 - 150),
        ("ENSEMBLE & KERNELS", "Random Forest, SVM", thumb_x - 150),
        ("LARGE LANGUAGE MODELS", "Gemini API", track_x2 - 150)
    ]
    
    for title, subtitle, x_pos in eras_info:
        draw.rectangle([x_pos, 150, x_pos + 300, 230], 
                      fill='#000000', outline='#00ffff', width=2)
        draw.text((x_pos + 20, 160), title, fill='#00ffff', font=font_medium)
        draw.text((x_pos + 20, 200), subtitle, fill='#00ff00', font=ImageFont.load_default())
    
    output_path = 'retrobrain/demo_screenshots/time_slider.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✓ Created {output_path}")


if __name__ == "__main__":
    print("📸 Generating demo screenshots...")
    create_dashboard_screenshot()
    create_terminal_screenshot()
    create_accuracy_chart()
    create_time_slider_screenshot()
    print("\n✅ All screenshots generated!")

