/**
 * 🕹️ RetroBrain JavaScript
 * CRT effects, animations, and API integration
 */

const API_BASE_URL = 'http://localhost:8000';

// Floating Pixel Particles
class PixelParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.init();
    }

    init() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        window.addEventListener('resize', () => {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
        });
        
        this.createParticles();
        this.animate();
    }

    createParticles() {
        const colors = ['#00ff00', '#00ffff', '#ff00ff', '#ffff00'];
        const particleCount = 50;
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 3 + 1,
                color: colors[Math.floor(Math.random() * colors.length)],
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5,
                opacity: Math.random() * 0.5 + 0.3
            });
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach(particle => {
            particle.x += particle.speedX;
            particle.y += particle.speedY;
            
            // Wrap around edges
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            if (particle.y > this.canvas.height) particle.y = 0;
            
            // Draw particle
            this.ctx.fillStyle = particle.color;
            this.ctx.globalAlpha = particle.opacity;
            this.ctx.fillRect(
                Math.floor(particle.x),
                Math.floor(particle.y),
                particle.size,
                particle.size
            );
            
            // Glow effect
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = particle.color;
            this.ctx.fillRect(
                Math.floor(particle.x),
                Math.floor(particle.y),
                particle.size,
                particle.size
            );
            this.ctx.shadowBlur = 0;
        });
        
        this.ctx.globalAlpha = 1;
        requestAnimationFrame(() => this.animate());
    }
}

// Typewriter Effect
function typeWriter(element, text, speed = 50, callback = null) {
    if (!element) return;
    
    element.textContent = '';
    let i = 0;
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        } else {
            if (callback) callback();
        }
    }
    
    type();
}

// Boot Sequence Animation
function bootSequence(element) {
    const bootText = [
        '>>> RETROBRAIN COMPUTER TERMINAL v1.0 <<<',
        '>>> INITIALIZING SYSTEMS...',
        '>>> LOADING 1980s MODELS...',
        '>>> LOADING 2000s MODELS...',
        '>>> LOADING 2020s MODELS...',
        '>>> READY...'
    ];
    
    let lineIndex = 0;
    
    function showNextLine() {
        if (lineIndex < bootText.length) {
            const line = document.createElement('div');
            line.className = 'typed-text';
            line.textContent = bootText[lineIndex];
            element.appendChild(line);
            lineIndex++;
            setTimeout(showNextLine, 800);
        }
    }
    
    showNextLine();
}

// API Functions
async function trainModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_type: 'sentiment'
            })
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Training error:', error);
        throw error;
    }
}

async function predict(text, era = null) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                model_era: era
            })
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

async function getResults() {
    try {
        const response = await fetch(`${API_BASE_URL}/results`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Results error:', error);
        throw error;
    }
}

async function explain(text, prediction, era = '2020s') {
    try {
        const response = await fetch(`${API_BASE_URL}/explain`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                prediction: prediction,
                era: era
            })
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Explanation error:', error);
        throw error;
    }
}

async function getComparison() {
    try {
        const response = await fetch(`${API_BASE_URL}/compare`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Comparison error:', error);
        throw error;
    }
}

// Display Functions
function displayPrediction(era, result, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    
    if (result.error) {
        container.innerHTML = `<div class="prediction-display">
            <div class="prediction-value">ERROR</div>
            <div>${result.error}</div>
        </div>`;
        return;
    }
    
    const prediction = result.prediction;
    const probabilities = result.probabilities || [0.5, 0.5];
    const confidence = Math.max(...probabilities);
    
    container.innerHTML = `
        <div class="prediction-display">
            <div class="prediction-value">${prediction === 1 ? 'POSITIVE' : 'NEGATIVE'}</div>
            <div>Confidence: ${(confidence * 100).toFixed(1)}%</div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${confidence * 100}%"></div>
            </div>
            <div>Negative: ${(probabilities[0] * 100).toFixed(1)}%</div>
            <div>Positive: ${(probabilities[1] * 100).toFixed(1)}%</div>
        </div>
    `;
}

function displayResults(results) {
    // Update leaderboard
    const leaderboard = document.getElementById('leaderboard');
    if (leaderboard && results.eras) {
        let entries = [];
        
        Object.entries(results.eras).forEach(([era, models]) => {
            Object.entries(models).forEach(([modelName, modelResult]) => {
                entries.push({
                    era: era,
                    model: modelName,
                    accuracy: modelResult.accuracy || 0
                });
            });
        });
        
        entries.sort((a, b) => b.accuracy - a.accuracy);
        
        leaderboard.innerHTML = entries.slice(0, 10).map((entry, index) => `
            <div class="leaderboard-entry">
                <span class="leaderboard-rank">#${index + 1}</span>
                <span class="leaderboard-name">${entry.era.toUpperCase()} - ${entry.model}</span>
                <span class="leaderboard-score">${(entry.accuracy * 100).toFixed(2)}%</span>
            </div>
        `).join('');
    }
}

// Time Slider Morphing - Changes entire website design with smooth interpolation
function morphPredictions(value) {
    // Normalize slider to 0-1
    const t = value / 100.0;
    
    // Apply smooth interpolation styling
    applyEraStylingSmooth(t);
    
    // Determine current era name
    const eras = ['1980s', '2000s', '2020s'];
    const eraIndex = Math.floor(t * eras.length);
    const currentEra = eras[Math.min(eraIndex, eras.length - 1)];
    
    // Update UI based on slider position
    document.querySelectorAll('.era-panel').forEach(panel => {
        const era = panel.classList[1]; // era-1980s, era-2000s, etc.
        if (era.includes(currentEra)) {
            panel.style.opacity = '1';
            panel.style.transform = 'scale(1.05)';
        } else {
            panel.style.opacity = '0.6';
            panel.style.transform = 'scale(1)';
        }
    });
    
    return currentEra;
}

// Smooth interpolation of CSS variables based on slider value (0-1)
function applyEraStylingSmooth(t) {
    const root = document.documentElement;
    const body = document.body;
    
    // Font family interpolation
    let fontFamily;
    if (t < 0.33) {
        fontFamily = "'Press Start 2P', monospace";
    } else if (t < 0.66) {
        if (t < 0.5) {
            fontFamily = "'Press Start 2P', 'IBM Plex Mono', monospace";
        } else {
            fontFamily = "'IBM Plex Mono', 'Press Start 2P', monospace";
        }
    } else {
        if (t < 0.83) {
            fontFamily = "'IBM Plex Mono', 'Inter', sans-serif";
        } else {
            fontFamily = "'Inter', 'IBM Plex Mono', sans-serif";
        }
    }
    
    // Color interpolation functions
    function lerpColor(c1, c2, t) {
        function hexToRgb(hex) {
            hex = hex.replace('#', '');
            return [
                parseInt(hex.substring(0, 2), 16),
                parseInt(hex.substring(2, 4), 16),
                parseInt(hex.substring(4, 6), 16)
            ];
        }
        function rgbToHex(rgb) {
            return '#' + rgb.map(c => Math.round(c).toString(16).padStart(2, '0')).join('');
        }
        const rgb1 = hexToRgb(c1);
        const rgb2 = hexToRgb(c2);
        const rgb = rgb1.map((c, i) => c + (rgb2[i] - c) * t);
        return rgbToHex(rgb);
    }
    
    // Primary color (green → gray-blue → indigo)
    let primaryColor;
    if (t < 0.5) {
        primaryColor = lerpColor('#00ff00', '#5b9bd5', t * 2);
    } else {
        primaryColor = lerpColor('#5b9bd5', '#4f46e5', (t - 0.5) * 2);
    }
    
    // Background color (black → dark gray → light gray)
    let bgColor;
    if (t < 0.5) {
        bgColor = lerpColor('#000000', '#1e1e1e', t * 2);
    } else {
        bgColor = lerpColor('#1e1e1e', '#f3f4f6', (t - 0.5) * 2);
    }
    
    // Text color (green → gray → dark)
    let textColor;
    if (t < 0.5) {
        textColor = lerpColor('#00ff00', '#d3d3d3', t * 2);
    } else {
        textColor = lerpColor('#d3d3d3', '#111827', (t - 0.5) * 2);
    }
    
    // Border radius (0px → 4px → 12px)
    let borderRadius;
    if (t < 0.5) {
        borderRadius = t * 8; // 0-4px
    } else {
        borderRadius = 4 + (t - 0.5) * 16; // 4-12px
    }
    
    // Shadow blur (0px → 10px → 20px)
    const shadowBlur = t * 20;
    
    // Scanline opacity (1.0 → 0.5 → 0.0)
    const scanlineOpacity = 1.0 - t;
    
    // Glassmorphism (backdrop blur)
    const backdropBlur = Math.max(0, (t - 0.5) * 20); // 0-10px
    
    // Font size (14px → 16px → 18px)
    const fontSize = 14 + t * 4;
    
    // Border width (3px → 2px → 1px)
    const borderWidth = Math.max(1, 3 - t * 2);
    
    // Image rendering
    const imageRendering = t < 0.3 ? 'pixelated' : 'auto';
    
    // Apply all CSS variables
    root.style.setProperty('--primary-color', primaryColor);
    root.style.setProperty('--bg-color', bgColor);
    root.style.setProperty('--text-color', textColor);
    root.style.setProperty('--border-radius', `${borderRadius}px`);
    root.style.setProperty('--shadow-blur', `${shadowBlur}px`);
    root.style.setProperty('--scanline-opacity', scanlineOpacity);
    root.style.setProperty('--backdrop-blur', `${backdropBlur}px`);
    root.style.setProperty('--font-size', `${fontSize}px`);
    root.style.setProperty('--border-width', `${borderWidth}px`);
    root.style.setProperty('--font-family', fontFamily);
    root.style.setProperty('--image-rendering', imageRendering);
    
    // Apply to body
    body.style.fontFamily = fontFamily;
    body.style.fontSize = `${fontSize}px`;
    body.style.imageRendering = imageRendering;
    
    // Update scanlines
    const scanlines = document.querySelector('.scanlines');
    if (scanlines) {
        scanlines.style.opacity = scanlineOpacity;
    }
    
    // Update era display
    let eraName;
    if (t < 0.33) {
        eraName = '1980s<br/>RETRO';
    } else if (t < 0.66) {
        eraName = '2000s<br/>TRANSITION';
    } else {
        eraName = '2020s<br/>MODERN';
    }
    
    const currentEraDisplay = document.getElementById('current-era');
    if (currentEraDisplay) {
        currentEraDisplay.innerHTML = eraName;
    }
    
    // Remove/add era classes for CSS targeting
    body.classList.remove('era-1980s-active', 'era-2000s-active', 'era-2020s-active');
    if (t < 0.33) {
        body.classList.add('era-1980s-active');
    } else if (t < 0.66) {
        body.classList.add('era-2000s-active');
    } else {
        body.classList.add('era-2020s-active');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize pixel particles
    const pixelCanvas = document.getElementById('pixel-canvas');
    if (pixelCanvas) {
        new PixelParticleSystem(pixelCanvas);
    }
    
    // Boot sequence if boot element exists
    const bootElement = document.getElementById('boot-sequence');
    if (bootElement) {
        bootSequence(bootElement);
    }
    
    // Typewriter effect for title
    const titleElement = document.getElementById('retrobrain-title');
    if (titleElement) {
        const originalText = titleElement.textContent;
        setTimeout(() => {
            typeWriter(titleElement, originalText, 80);
        }, 500);
    }
    
    // Time slider handler (on both index and dashboard)
    const timeSlider = document.getElementById('time-slider');
    if (timeSlider) {
        // Initialize with default era (2000s = 50)
        const initialValue = parseInt(timeSlider.value) || 50;
        morphPredictions(initialValue);
        
        // Handle slider input
        timeSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            morphPredictions(value);
        });
        
        // Handle slider change
        timeSlider.addEventListener('change', (e) => {
            const value = parseInt(e.target.value);
            morphPredictions(value);
        });
    }
    
    // Prediction form handler
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const textInput = document.getElementById('text-input');
            const text = textInput.value;
            
            if (!text) return;
            
            // Show loading
            const loading = document.getElementById('loading');
            if (loading) loading.style.display = 'block';
            
            try {
                // Get predictions from all eras
                const results = await predict(text);
                
                // Display results
                if (results['1980s']) {
                    displayPrediction('1980s', results['1980s'], 'prediction-1980s');
                }
                if (results['2000s']) {
                    displayPrediction('2000s', results['2000s'], 'prediction-2000s');
                }
                if (results['2020s']) {
                    displayPrediction('2020s', results['2020s'], 'prediction-2020s');
                }
                
                // Get explanation
                if (results['2020s']) {
                    const explanation = await explain(text, results['2020s'].prediction);
                    const explanationDiv = document.getElementById('explanation');
                    if (explanationDiv) {
                        explanationDiv.textContent = explanation.explanation || 'No explanation available';
                    }
                }
            } catch (error) {
                console.error('Prediction error:', error);
                alert('Error making prediction. Make sure models are trained and API is running.');
            } finally {
                if (loading) loading.style.display = 'none';
            }
        });
    }
    
    // Train button handler
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) {
        trainBtn.addEventListener('click', async () => {
            trainBtn.disabled = true;
            trainBtn.textContent = 'TRAINING...';
            
            try {
                await trainModels();
                alert('Training started! Check back in a few minutes.');
            } catch (error) {
                console.error('Training error:', error);
                alert('Error starting training. Make sure API is running.');
            } finally {
                trainBtn.disabled = false;
                trainBtn.textContent = 'TRAIN MODELS';
            }
        });
    }
    
    // Load results on page load
    const resultsBtn = document.getElementById('load-results-btn');
    if (resultsBtn) {
        resultsBtn.addEventListener('click', async () => {
            try {
                const results = await getResults();
                displayResults(results);
            } catch (error) {
                console.error('Results error:', error);
                alert('Error loading results. Train models first.');
            }
        });
    }
});

// Export functions for global use
window.RetroBrain = {
    trainModels,
    predict,
    getResults,
    explain,
    getComparison,
    displayPrediction,
    displayResults,
    morphPredictions,
    typeWriter,
    bootSequence
};

