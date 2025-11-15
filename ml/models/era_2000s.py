"""
2000s Era ML Models
Random Forest, SVM (linear + RBF), PCA
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class Era2000sModels:
    """2000s era machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.pca_model = None
        self.model_names = [
            'random_forest',
            'svm_linear',
            'svm_rbf',
            'pca_svm'
        ]
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=10):
        """Train Random Forest (2000s ensemble)"""
        print(f"🔧 Training Random Forest (2000s, {n_estimators} trees)...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_svm_linear(self, X_train, y_train):
        """Train Linear SVM"""
        print("🔧 Training Linear SVM (2000s)...")
        # Use smaller subset for faster training
        if len(X_train) > 5000:
            indices = np.random.choice(len(X_train), 5000, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X_train_sample, y_train_sample)
        self.models['svm_linear'] = model
        return model
    
    def train_svm_rbf(self, X_train, y_train):
        """Train RBF Kernel SVM"""
        print("🔧 Training RBF Kernel SVM (2000s)...")
        # Use smaller subset for faster training
        if len(X_train) > 3000:
            indices = np.random.choice(len(X_train), 3000, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        model = SVC(kernel='rbf', probability=True, random_state=42, gamma='scale')
        model.fit(X_train_sample, y_train_sample)
        self.models['svm_rbf'] = model
        return model
    
    def train_pca_svm(self, X_train, y_train, n_components=100):
        """Train PCA + SVM pipeline"""
        print(f"🔧 Training PCA + SVM (2000s, {n_components} components)...")
        
        # Limit components to feature size
        n_components = min(n_components, X_train.shape[1], len(X_train))
        
        # Use smaller subset for faster training
        if len(X_train) > 3000:
            indices = np.random.choice(len(X_train), 3000, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca_model.fit_transform(X_train_sample)
        
        model = SVC(kernel='rbf', probability=True, random_state=42, gamma='scale')
        model.fit(X_train_pca, y_train_sample)
        
        self.models['pca_svm'] = {
            'pca': self.pca_model,
            'svm': model
        }
        return self.models['pca_svm']
    
    def train_all(self, X_train, y_train):
        """Train all 2000s models"""
        print("\n🕹️ === TRAINING 2000s ERA MODELS ===")
        
        self.train_random_forest(X_train, y_train)
        self.train_svm_linear(X_train, y_train)
        self.train_svm_rbf(X_train, y_train)
        self.train_pca_svm(X_train, y_train)
        
        print("✅ All 2000s models trained!\n")
        return self.models
    
    def predict(self, model_name, X):
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        # Special handling for PCA+SVM
        if model_name == 'pca_svm':
            X_transformed = model['pca'].transform(X)
            predictions = model['svm'].predict(X_transformed)
            probabilities = model['svm'].predict_proba(X_transformed)
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
        
        # Standard models
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def evaluate(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        result = self.predict(model_name, X_test)
        predictions = result['predictions']
        
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            'model_name': model_name,
            'era': '2000s',
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all 2000s models"""
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.evaluate(model_name, X_test, y_test)
        return results
    
    def save_models(self, path='results/models_2000s.pkl'):
        """Save all models"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"✅ Saved 2000s models to {path}")
    
    def load_models(self, path='results/models_2000s.pkl'):
        """Load saved models"""
        with open(path, 'rb') as f:
            self.models = pickle.load(f)
        print(f"✅ Loaded 2000s models from {path}")

