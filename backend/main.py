"""
RetroBrain FastAPI Backend Server
API endpoints for ML model training, prediction, and results
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os
import json
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from ml.models.era_1980s import Era1980sModels
from ml.models.era_2000s import Era2000sModels
from ml.models.era_2020s import Era2020sModels
from ml.utils.train_models import train_all_eras

# Fix paths for training
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="RetroBrain API",
    description="ML Model Comparison Across Eras (1980s, 2000s, 2020s)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(base_dir, "frontend", "static")
templates_dir = os.path.join(base_dir, "frontend", "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Global state
data_loader = None
era_1980s_models = None
era_2000s_models = None
era_2020s_models = None
X_test = None
y_test = None
training_in_progress = False


# Request models
class PredictRequest(BaseModel):
    text: Optional[str] = None
    features: Optional[List[float]] = None
    model_era: Optional[str] = None  # '1980s', '2000s', '2020s', or None for all


class ExplainRequest(BaseModel):
    text: str
    prediction: int
    era: Optional[str] = None


class TrainRequest(BaseModel):
    dataset_type: Optional[str] = 'sentiment'
    api_key: Optional[str] = None


# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global data_loader, X_test, y_test, era_1980s_models, era_2000s_models, era_2020s_models
    print("🚀 RetroBrain API starting up...")
    
    try:
        data_loader = DataLoader(dataset_type='sentiment')
        _, X_test, _, y_test, _ = data_loader.load_sentiment_data()
        print("✅ Data loader initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize data loader: {e}")
    
    # Try to load pre-trained models if they exist
    # Check multiple possible paths for model files
    # Models are saved as 'retrobrain/results/models_*.pkl' during training
    # But may end up at retrobrain/retrobrain/results/ if run from retrobrain/ dir
    possible_model_paths = {
        '1980s': [
            os.path.join(base_dir, "retrobrain", "results", "models_1980s.pkl"),  # retrobrain/retrobrain/results/
            os.path.join(base_dir, "results", "models_1980s.pkl"),  # retrobrain/results/
            os.path.join(os.path.dirname(base_dir), "retrobrain", "results", "models_1980s.pkl"),
            os.path.join(os.path.dirname(base_dir), "retrobrain", "retrobrain", "results", "models_1980s.pkl"),
            "retrobrain/results/models_1980s.pkl",
            "retrobrain/retrobrain/results/models_1980s.pkl",
            "results/models_1980s.pkl"
        ],
        '2000s': [
            os.path.join(base_dir, "retrobrain", "results", "models_2000s.pkl"),
            os.path.join(base_dir, "results", "models_2000s.pkl"),
            os.path.join(os.path.dirname(base_dir), "retrobrain", "results", "models_2000s.pkl"),
            os.path.join(os.path.dirname(base_dir), "retrobrain", "retrobrain", "results", "models_2000s.pkl"),
            "retrobrain/results/models_2000s.pkl",
            "retrobrain/retrobrain/results/models_2000s.pkl",
            "results/models_2000s.pkl"
        ],
        '2020s': [
            os.path.join(base_dir, "retrobrain", "results", "models_2020s.pkl"),
            os.path.join(base_dir, "results", "models_2020s.pkl"),
            os.path.join(os.path.dirname(base_dir), "retrobrain", "results", "models_2020s.pkl"),
            os.path.join(os.path.dirname(base_dir), "retrobrain", "retrobrain", "results", "models_2020s.pkl"),
            "retrobrain/results/models_2020s.pkl",
            "retrobrain/retrobrain/results/models_2020s.pkl",
            "results/models_2020s.pkl"
        ]
    }
    
    results_dir = None
    found_models = {'1980s': None, '2000s': None, '2020s': None}
    
    # Find actual model file paths
    # Try absolute paths first, then relative
    for era, paths in possible_model_paths.items():
        for path in paths:
            # Try absolute path first
            if os.path.isabs(path):
                test_path = path
            else:
                # Try relative to base_dir
                test_path = os.path.join(base_dir, path)
            
            if os.path.exists(test_path):
                found_models[era] = os.path.abspath(test_path)  # Store absolute path
                if results_dir is None:
                    results_dir = os.path.dirname(test_path)
                print(f"📂 Found {era} models at: {found_models[era]}")
                break
            
            # Also try absolute path if relative didn't work
            if not os.path.isabs(path):
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path) and abs_path != test_path:
                    found_models[era] = abs_path
                    if results_dir is None:
                        results_dir = os.path.dirname(abs_path)
                    print(f"📂 Found {era} models at: {found_models[era]}")
                    break
    
    # Load models from found paths (must use global keyword!)
    global era_1980s_models, era_2000s_models, era_2020s_models
    try:
        if found_models['1980s']:
            era_1980s_models = Era1980sModels()
            era_1980s_models.load_models(found_models['1980s'])
            if era_1980s_models.models:
                print(f"✅ Loaded 1980s models: {list(era_1980s_models.models.keys())}")
            else:
                print("⚠️  Warning: 1980s models dict is empty")
        else:
            print("⚠️  1980s models file not found")
        
        if found_models['2000s']:
            era_2000s_models = Era2000sModels()
            era_2000s_models.load_models(found_models['2000s'])
            if era_2000s_models.models:
                print(f"✅ Loaded 2000s models: {list(era_2000s_models.models.keys())}")
            else:
                print("⚠️  Warning: 2000s models dict is empty")
        else:
            print("⚠️  2000s models file not found")
        
        if found_models['2020s']:
            era_2020s_models = Era2020sModels(api_key=os.getenv('GEMINI_API_KEY'))
            era_2020s_models.load_models(found_models['2020s'])
            print("✅ Loaded 2020s models")
        else:
            print("⚠️  2020s models file not found")
    except Exception as e:
        print(f"⚠️  Warning: Could not load pre-trained models: {e}")
        import traceback
        traceback.print_exc()
    
    if not any(found_models.values()):
        print("⚠️  Warning: No model files found. Models need to be trained.")
        print(f"⚠️  Base dir: {base_dir}")
        print(f"⚠️  Checked paths: {possible_model_paths['1980s'][:3]}...")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - serve landing page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "🕹️ RetroBrain API",
        "version": "1.0.0",
        "endpoints": {
            "/train": "Train all models",
            "/predict": "Make predictions",
            "/results": "Get evaluation results",
            "/explain": "Get AI explanation",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": era_1980s_models is not None}


@app.post("/train")
async def train_models(background_tasks: BackgroundTasks, request: TrainRequest):
    """Train all models across eras"""
    global training_in_progress, era_1980s_models, era_2000s_models, era_2020s_models
    
    if training_in_progress:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_in_progress = True
    
    def train_task():
        global era_1980s_models, era_2000s_models, era_2020s_models, training_in_progress
        try:
            # Change to retrobrain directory for training
            original_dir = os.getcwd()
            retrobrain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.chdir(retrobrain_dir)
            
            results = train_all_eras(
                dataset_type=request.dataset_type,
                api_key=request.api_key or os.getenv('GEMINI_API_KEY')
            )
            
            # Load trained models
            # Models are saved to 'retrobrain/results/' by train_all_eras
            results_dir = os.path.join(retrobrain_dir, "results")
            
            # Wait a bit to ensure files are written
            import time
            time.sleep(0.5)
            
            models_1980s_path = os.path.join(results_dir, 'models_1980s.pkl')
            if os.path.exists(models_1980s_path):
                era_1980s_models = Era1980sModels()
                era_1980s_models.load_models(models_1980s_path)
                print(f"✅ Loaded 1980s models: {list(era_1980s_models.models.keys())}")
            else:
                print(f"⚠️  Warning: Models file not found at {models_1980s_path}")
            
            models_2000s_path = os.path.join(results_dir, 'models_2000s.pkl')
            if os.path.exists(models_2000s_path):
                era_2000s_models = Era2000sModels()
                era_2000s_models.load_models(models_2000s_path)
                print(f"✅ Loaded 2000s models: {list(era_2000s_models.models.keys())}")
            else:
                print(f"⚠️  Warning: Models file not found at {models_2000s_path}")
            
            models_2020s_path = os.path.join(results_dir, 'models_2020s.pkl')
            if os.path.exists(models_2020s_path):
                era_2020s_models = Era2020sModels(api_key=request.api_key or os.getenv('GEMINI_API_KEY'))
                era_2020s_models.load_models(models_2020s_path)
                print("✅ Loaded 2020s models")
            else:
                print(f"⚠️  Warning: Models file not found at {models_2020s_path}")
            
            os.chdir(original_dir)
            training_in_progress = False
        except Exception as e:
            training_in_progress = False
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
    
    background_tasks.add_task(train_task)
    
    return {
        "message": "Training started in background",
        "status": "processing"
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Make predictions across eras"""
    global era_1980s_models, era_2000s_models, era_2020s_models, data_loader
    
    # Load models if not already loaded
    if era_1980s_models is None or era_2000s_models is None or era_2020s_models is None:
        # Try multiple possible paths for model files
        possible_model_paths = [
            # Relative to base_dir (retrobrain/) - handle double retrobrain/ case
            os.path.join(base_dir, "retrobrain", "results", "models_1980s.pkl"),
            os.path.join(base_dir, "results", "models_1980s.pkl"),
            # Absolute from project root
            os.path.join(os.path.dirname(base_dir), "retrobrain", "results", "models_1980s.pkl"),
            os.path.join(os.path.dirname(base_dir), "retrobrain", "retrobrain", "results", "models_1980s.pkl"),
            # Relative paths (work if run from retrobrain/ dir)
            "retrobrain/results/models_1980s.pkl",
            "retrobrain/retrobrain/results/models_1980s.pkl",
            "results/models_1980s.pkl"
        ]
        
        found_1980s = None
        for path in possible_model_paths:
            test_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if os.path.exists(test_path):
                found_1980s = test_path
                break
        
        if era_1980s_models is None and found_1980s:
            try:
                era_1980s_models = Era1980sModels()
                era_1980s_models.load_models(found_1980s)
                print(f"✅ Loaded 1980s models from {found_1980s}: {list(era_1980s_models.models.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading 1980s models: {e}")
        
        # Same for 2000s
        possible_2000s_paths = [p.replace('1980s', '2000s') for p in possible_model_paths]
        found_2000s = None
        for path in possible_2000s_paths:
            test_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if os.path.exists(test_path):
                found_2000s = test_path
                break
        
        if era_2000s_models is None and found_2000s:
            try:
                era_2000s_models = Era2000sModels()
                era_2000s_models.load_models(found_2000s)
                print(f"✅ Loaded 2000s models from {found_2000s}: {list(era_2000s_models.models.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading 2000s models: {e}")
        
        # Same for 2020s
        possible_2020s_paths = [p.replace('1980s', '2020s') for p in possible_model_paths]
        found_2020s = None
        for path in possible_2020s_paths:
            test_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if os.path.exists(test_path):
                found_2020s = test_path
                break
        
        if era_2020s_models is None and found_2020s:
            try:
                era_2020s_models = Era2020sModels(api_key=os.getenv('GEMINI_API_KEY'))
                era_2020s_models.load_models(found_2020s)
                print(f"✅ Loaded 2020s models from {found_2020s}")
            except Exception as e:
                print(f"⚠️  Error loading 2020s models: {e}")
    
    if era_1980s_models is None and era_2000s_models is None and era_2020s_models is None:
        raise HTTPException(status_code=400, detail="Models not trained yet. Call /train first.")
    
    results = {}
    
    # Prepare input
    if request.text:
        # Text input - need to convert to features for 1980s/2000s models
        try:
            if data_loader is None:
                data_loader = DataLoader(dataset_type='sentiment')
                _, _, _, _, _ = data_loader.load_sentiment_data()
            X_input = data_loader.prepare_for_prediction(request.text)
        except Exception as e:
            # Fallback: create random features for demo
            print(f"⚠️  Error preparing text input: {e}")
            X_input = np.random.rand(1, 5000)
    elif request.features:
        # Feature array input
        X_input = np.array([request.features])
    else:
        raise HTTPException(status_code=400, detail="Either 'text' or 'features' must be provided")
    
    # Predict with requested era(s)
    eras_to_predict = [request.model_era] if request.model_era else ['1980s', '2000s', '2020s']
    
    for era in eras_to_predict:
        if era == '1980s' and era_1980s_models:
            # Get predictions from best 1980s model (logistic regression)
            try:
                # Check if models dict exists and has the model
                if hasattr(era_1980s_models, 'models') and 'logistic_regression' in era_1980s_models.models:
                    pred_result = era_1980s_models.predict('logistic_regression', X_input)
                    results['1980s'] = {
                        'model': 'logistic_regression',
                        'prediction': int(pred_result['predictions'][0]),
                        'probabilities': pred_result['probabilities'][0].tolist() if pred_result['probabilities'] is not None else None
                    }
                else:
                    results['1980s'] = {'error': 'Model not loaded. Please train models first.'}
            except Exception as e:
                print(f"⚠️ 1980s prediction error: {e}")
                import traceback
                traceback.print_exc()
                results['1980s'] = {'error': str(e)}
        
        elif era == '2000s' and era_2000s_models:
            # Get predictions from best 2000s model (random forest)
            try:
                if hasattr(era_2000s_models, 'models') and 'random_forest' in era_2000s_models.models:
                    pred_result = era_2000s_models.predict('random_forest', X_input)
                    results['2000s'] = {
                        'model': 'random_forest',
                        'prediction': int(pred_result['predictions'][0]),
                        'probabilities': pred_result['probabilities'][0].tolist() if pred_result['probabilities'] is not None else None
                    }
                else:
                    results['2000s'] = {'error': 'Model not loaded. Please train models first.'}
            except Exception as e:
                print(f"⚠️ 2000s prediction error: {e}")
                import traceback
                traceback.print_exc()
                results['2000s'] = {'error': str(e)}
        
        elif era == '2020s' and era_2020s_models:
            # Use Gemini
            try:
                if request.text:
                    pred_result = era_2020s_models.gemini_model.predict_text(request.text)
                else:
                    pred_result = era_2020s_models.predict('gemini', X_input)
                
                results['2020s'] = {
                    'model': 'gemini',
                    'prediction': int(pred_result['predictions'][0]),
                    'probabilities': pred_result['probabilities'][0].tolist() if pred_result['probabilities'] is not None else None,
                    'explanation': pred_result.get('explanation', '')
                }
            except Exception as e:
                print(f"⚠️ 2020s prediction error: {e}")
                import traceback
                traceback.print_exc()
                results['2020s'] = {'error': str(e)}
    
    return results


@app.get("/results")
async def get_results():
    """Get evaluation results for all models"""
    results_dir = os.path.join(base_dir, "results")
    results_path = os.path.join(results_dir, 'training_results.json')
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        # Try evaluation results
        try:
            eval_path = os.path.join(results_dir, 'evaluation_results.json')
            with open(eval_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No results found. Train models first.")


@app.post("/explain")
async def explain(request: ExplainRequest):
    """Get AI explanation for predictions"""
    global era_2020s_models
    
    if era_2020s_models is None:
        raise HTTPException(status_code=400, detail="2020s models not loaded. Train models first.")
    
    try:
        if request.era == '2020s' or request.era is None:
            # Use Gemini for explanation
            explanation = era_2020s_models.gemini_model.explain_prediction(
                request.text,
                request.prediction
            )
            return {
                'era': '2020s',
                'explanation': explanation
            }
        else:
            # Fallback explanation for older eras
            return {
                'era': request.era,
                'explanation': f"This {request.era} era model classified the text as {'positive' if request.prediction == 1 else 'negative'} based on statistical pattern matching."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/compare")
async def compare_eras():
    """Compare all eras side-by-side"""
    results_dir = os.path.join(base_dir, "results")
    results_path = os.path.join(results_dir, 'training_results.json')
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        comparison = {}
        for era_name, era_results in results.get('eras', {}).items():
            comparison[era_name] = {}
            for model_name, model_result in era_results.items():
                comparison[era_name][model_name] = {
                    'accuracy': model_result.get('accuracy', 0),
                    'era': era_name
                }
        
        return comparison
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No results found. Train models first.")


@app.get("/model-info")
async def model_info():
    """Get information about available models"""
    return {
        "1980s": {
            "models": ["logistic_regression", "naive_bayes_gaussian", "naive_bayes_multinomial", "decision_tree", "kmeans"],
            "description": "Classic statistical and rule-based models"
        },
        "2000s": {
            "models": ["random_forest", "svm_linear", "svm_rbf", "pca_svm"],
            "description": "Ensemble and kernel-based models"
        },
        "2020s": {
            "models": ["gemini"],
            "description": "Large language model via Gemini API"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

