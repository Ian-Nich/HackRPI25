# 🕹️ RetroBrain Streamlit UI

## Dynamic UI Morphing: 1980s → 2000s → 2020s

This Streamlit app features a **slider (0-100)** that smoothly transforms the entire UI through three design eras:

### Slider Positions:

- **0-33: 1980s RETRO**
  - Font: Press Start 2P (pixelated)
  - Colors: Green-on-black CRT
  - Effects: Scanlines, blocky borders, harsh edges
  - Image rendering: Pixelated

- **34-66: 2000s TRANSITION**
  - Font: IBM Plex Mono (monospaced)
  - Colors: Gray-blue web colors
  - Effects: Rounded corners, simple gradients
  - Image rendering: Auto

- **67-100: 2020s MODERN**
  - Font: Inter (clean sans-serif)
  - Colors: Modern palette (indigo/cyan)
  - Effects: Glassmorphism, soft shadows, smooth motion
  - Image rendering: Auto

### How It Works:

1. **Move the slider** in the sidebar (0-100)
2. **Watch the UI morph** - fonts, colors, borders, shadows all interpolate smoothly
3. **See real-time transformation** - everything updates with 0.5s transitions

### Running the App:

```bash
cd retrobrain
streamlit run streamlit_app.py --server.port 8502
```

Then open: http://localhost:8502

### Features:

- ✅ **Dynamic Font Interpolation**: Press Start 2P → IBM Plex Mono → Inter
- ✅ **Smooth Color Transitions**: Green → Gray-blue → Modern colors
- ✅ **Border Radius Morphing**: 0px → 4px → 12px
- ✅ **Shadow Evolution**: Glow → Shadow → Soft shadow
- ✅ **Scanline Fade**: Full opacity → Fade out
- ✅ **Glassmorphism**: Appears at high slider values
- ✅ **All elements transition**: 0.5s ease-in-out on everything

### API Integration:

The app connects to the RetroBrain FastAPI backend (default: http://localhost:8000)

- Train models
- Make predictions
- View results
- Compare eras

### CSS Interpolation:

All CSS properties interpolate based on slider value:
- Colors blend using linear interpolation
- Fonts fade between families
- Border radius increases smoothly
- Shadow intensity and blur evolve
- Scanline opacity fades out
- Glassmorphism appears gradually

---

**Built with ❤️ for HackRPI 2025**

