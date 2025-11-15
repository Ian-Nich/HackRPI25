"""
2020s Era ML Models
Gemini API integration for modern AI
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any
import time


class ModernModel:
    """2020s era model using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API model
        
        Args:
            api_key: Google Gemini API key (or use env variable)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = 'gemini'
        self.era = '2020s'
        
        if not self.api_key:
            print("⚠️ Warning: GEMINI_API_KEY not set. Using fallback mode.")
            self.use_fallback = True
        else:
            self.use_fallback = False
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
                self.model = genai.GenerativeModel('gemini-pro')
            except ImportError:
                print("⚠️ Warning: google-generativeai not installed. Using fallback mode.")
                self.use_fallback = True
    
    def _call_gemini_api(self, prompt: str, task_type: str = 'classification') -> Dict[str, Any]:
        """Call Gemini API with prompt"""
        if self.use_fallback:
            # Fallback: return mock response
            return self._fallback_response(prompt, task_type)
        
        try:
            import google.generativeai as genai
            response = self.model.generate_content(prompt)
            return {
                'text': response.text,
                'success': True
            }
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return self._fallback_response(prompt, task_type)
    
    def _fallback_response(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """Fallback response when API is unavailable"""
        if 'sentiment' in prompt.lower() or 'positive' in prompt.lower():
            return {
                'text': 'Positive sentiment (0.75 confidence)',
                'success': False,
                'fallback': True
            }
        return {
            'text': 'Class 1 (0.65 confidence)',
            'success': False,
            'fallback': True
        }
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict using text input via Gemini
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predictions and probabilities
        """
        prompt = f"""
        Analyze this text for sentiment classification (positive/negative):
        
        Text: "{text}"
        
        Provide:
        1. Predicted class (0 for negative, 1 for positive)
        2. Confidence score (0-1)
        3. Brief explanation
        
        Format as JSON: {{"class": 0 or 1, "confidence": 0.0-1.0, "explanation": "..."}}
        """
        
        response = self._call_gemini_api(prompt, 'classification')
        
        # Parse response
        try:
            # Try to extract JSON from response
            response_text = response['text']
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                return {
                    'predictions': np.array([result.get('class', 1)]),
                    'probabilities': np.array([[1 - result.get('confidence', 0.5), result.get('confidence', 0.5)]]),
                    'explanation': result.get('explanation', ''),
                    'confidence': result.get('confidence', 0.5)
                }
        except Exception as e:
            print(f"⚠️ Error parsing Gemini response: {e}")
        
        # Fallback response
        return {
            'predictions': np.array([1]),
            'probabilities': np.array([[0.3, 0.7]]),
            'explanation': 'Gemini API unavailable - using fallback',
            'confidence': 0.5
        }
    
    def predict_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict using feature array (convert to text description)
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary with predictions
        """
        # Summarize features as text for Gemini
        feature_summary = f"Features: {features.shape[0]} samples, {features.shape[1]} dimensions"
        feature_stats = f"Mean: {features.mean():.3f}, Std: {features.std():.3f}"
        
        prompt = f"""
        Based on these feature statistics, predict the class:
        {feature_summary}
        {feature_stats}
        
        Return JSON: {{"class": 0 or 1, "confidence": 0.0-1.0}}
        """
        
        response = self._call_gemini_api(prompt, 'classification')
        
        try:
            response_text = response['text']
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                class_pred = result.get('class', 1)
                conf = result.get('confidence', 0.5)
                
                # Return predictions for all samples
                n_samples = len(features)
                predictions = np.full(n_samples, class_pred)
                probabilities = np.tile([1 - conf, conf], (n_samples, 1))
                
                return {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'confidence': conf
                }
        except Exception as e:
            print(f"⚠️ Error parsing response: {e}")
        
        # Fallback
        n_samples = len(features)
        return {
            'predictions': np.ones(n_samples),
            'probabilities': np.tile([0.3, 0.7], (n_samples, 1)),
            'confidence': 0.5
        }
    
    def explain_prediction(self, text: str, prediction: int) -> str:
        """
        Get explanation for a prediction using Gemini
        
        Args:
            text: Input text
            prediction: Predicted class
            
        Returns:
            Explanation string
        """
        prompt = f"""
        Explain why this text was classified as {"positive" if prediction == 1 else "negative"}:
        
        Text: "{text}"
        Prediction: {"Positive" if prediction == 1 else "Negative"}
        
        Provide a brief, technical explanation of the key words/features that led to this classification.
        """
        
        response = self._call_gemini_api(prompt, 'explanation')
        return response.get('text', 'Explanation unavailable')
    
    def explain_model_comparison(self, era_1980s_result: Dict, era_2000s_result: Dict, 
                                era_2020s_result: Dict) -> str:
        """
        Explain differences between model eras using Gemini
        
        Args:
            era_1980s_result: Results from 1980s models
            era_2000s_result: Results from 2000s models
            era_2020s_result: Results from 2020s models
            
        Returns:
            Comparison explanation
        """
        prompt = f"""
        Explain the evolution of machine learning from 1980s to 2020s based on these results:
        
        1980s Models: {era_1980s_result.get('accuracy', 0):.3f} accuracy
        2000s Models: {era_2000s_result.get('accuracy', 0):.3f} accuracy
        2020s Models: {era_2020s_result.get('accuracy', 0):.3f} accuracy
        
        Discuss:
        1. Why accuracy improved over time
        2. Key algorithmic differences between eras
        3. Modern advantages of 2020s models
        4. When older models might still be useful
        
        Provide a clear, educational explanation.
        """
        
        response = self._call_gemini_api(prompt, 'explanation')
        return response.get('text', 'Comparison explanation unavailable')


class Era2020sModels:
    """2020s era models wrapper"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.models = {}
        self.gemini_model = ModernModel(api_key)
        self.models['gemini'] = self.gemini_model
    
    def train_all(self, X_train, y_train):
        """'Train' Gemini model (no training needed, just setup)"""
        print("\n🕹️ === INITIALIZING 2020s ERA MODELS ===")
        print("✅ Gemini API model ready (no training required)")
        print()
        return self.models
    
    def predict(self, model_name: str, X, y_actual=None):
        """
        Make predictions using 2020s model
        
        Args:
            model_name: Should be 'gemini'
            X: Input features or text
            y_actual: Actual labels (optional)
        """
        if model_name != 'gemini':
            raise ValueError(f"Unknown 2020s model: {model_name}")
        
        # Check if X is text or features
        if isinstance(X, str):
            return self.gemini_model.predict_text(X)
        elif isinstance(X, np.ndarray):
            return self.gemini_model.predict_features(X)
        else:
            # Try to handle list of strings
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
                # Process first item for demo
                return self.gemini_model.predict_text(X[0])
            else:
                # Convert to array and use features
                X_array = np.array(X)
                return self.gemini_model.predict_features(X_array)
    
    def evaluate(self, model_name: str, X_test, y_test):
        """Evaluate 2020s model"""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        result = self.predict(model_name, X_test)
        predictions = result['predictions']
        
        # Ensure predictions match y_test length
        if len(predictions) != len(y_test):
            # Use first prediction for all (fallback)
            predictions = np.full(len(y_test), predictions[0] if len(predictions) > 0 else 1)
        
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            'model_name': model_name,
            'era': '2020s',
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'explanation': result.get('explanation', '')
        }
    
    def save_models(self, path='results/models_2020s.pkl'):
        """Save model configuration"""
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_type': 'gemini',
                'has_api_key': self.gemini_model.api_key is not None
            }, f)
        print(f"✅ Saved 2020s model config to {path}")
    
    def load_models(self, path='results/models_2020s.pkl'):
        """Load model configuration"""
        import pickle
        with open(path, 'rb') as f:
            config = pickle.load(f)
        print(f"✅ Loaded 2020s model config from {path}")

