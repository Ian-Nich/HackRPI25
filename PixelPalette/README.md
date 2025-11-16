# Pixel Palette

A retro-style pixel art editor built with vanilla JavaScript, HTML5 Canvas, and CSS. Create beautiful pixel art sprites with a 32×32 grid and enhance them with intelligent upscaling and auto-shading.


## Features

### 🎨 Drawing Tools
- **Pencil Tool**: Draw individual pixels
- **Eraser Tool**: Remove pixels
- **Fill Tool**: Fill connected regions with color
- **Eyedropper Tool**: Pick colors from the canvas
  - Left-click: Set primary color
  - Right-click: Set secondary color

### 🎨 Color System
- **Primary & Secondary Colors**: Quick access to two colors
- **Retro Palettes**: Pre-configured color palettes
  - NES (Nintendo Entertainment System)
  - PICO-8
  - Game Boy
- **Custom Color Pickers**: Full color selection for both primary and secondary colors

### ⚡ Upscaling & Enhancement
- **Intelligent Upscaling**: Scale your 32×32 pixel art up to 16× (512×512 pixels)
- **Auto Shading**: Automatic color blending between adjacent pixels of different colors
  - Creates smooth anti-aliasing effects
  - Blends colors based on position within upscaled blocks
  - Produces professional-looking results

### 💾 Project Management
- **Save Projects**: Export your work as JSON files
- **Load Projects**: Import previously saved projects
- **Export PNG**: Download your pixel art as PNG images

### 🔄 Undo/Redo
- Full undo/redo system with keyboard shortcuts
  - `Ctrl+Z`: Undo
  - `Ctrl+Y`: Redo

## Getting Started

### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge)
- No installation or build process required!

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PixelPalette.git
cd PixelPalette
```

2. Open `index.html` in your web browser:
   - Simply double-click `index.html`, or
   - Use a local web server (optional):
     ```bash
     # Using Python 3
     python -m http.server 8000
     
     # Using Node.js (if you have http-server installed)
     npx http-server
     ```
   - Then navigate to `http://localhost:8000` in your browser

That's it! No build process, no dependencies, just open and create!

## Usage

### Basic Drawing
1. Select a tool from the toolbar (Pencil, Eraser, Fill, or Eyedropper)
2. Choose your colors using the color pickers or select from a retro palette
3. Click and drag on the canvas to draw
4. Use right-click to draw with the secondary color

### Upscaling Your Art
1. Create your pixel art on the 32×32 grid
2. Click "Upscale Image" button
3. Adjust settings:
   - **Scale Factor**: Choose 2×, 4×, 8×, or 16×
   - **Auto Shading**: Toggle color blending between adjacent pixels
4. View the upscaled result in the preview area

### Saving Your Work
- **Save Project**: Saves your current work as a JSON file
- **Load Project**: Import a previously saved project
- **Export PNG**: Download your pixel art as a PNG image file

### Keyboard Shortcuts
- `Ctrl+Z`: Undo last action
- `Ctrl+Y`: Redo last undone action

## Project Structure

```
PixelPalette/
├── index.html      # Main HTML structure
├── app.js          # Application logic and drawing engine
├── style.css       # Styling and retro theme
└── README.md       # This file
```

## Technical Details

### Technologies
- **HTML5 Canvas**: For pixel-perfect rendering
- **Vanilla JavaScript**: No frameworks, pure JavaScript
- **CSS3**: Modern styling with retro aesthetics

### Grid System
- **Canvas Size**: 32×32 pixels
- **Display Size**: 512×512 pixels (16× zoom)
- **Pixel Size**: 16×16 pixels per grid cell

### Upscaling Algorithm
The upscaling algorithm uses:
1. **Nearest Neighbor Upscaling**: Initial upscaling maintains pixel art aesthetic
2. **Auto Shading**: Intelligent color blending
   - Detects adjacent pixels with different colors
   - Calculates weighted averages based on sub-pixel position
   - Creates smooth transitions (anti-aliasing) between color boundaries

## Browser Compatibility

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Opera (latest)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by classic pixel art tools and retro game aesthetics
- Built for HackRPI 2025

## Future Enhancements

Potential features for future versions:
- Animation support
- Sprite sheet export
- More color palettes
- Custom palette creation
- Layer support
- Pattern brushes

---

**Made with ❤️ for pixel art enthusiasts**

