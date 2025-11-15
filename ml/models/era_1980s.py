"""
1980s Era ML Models
Logistic Regression, Naive Bayes, Decision Tree, K-means
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class Era1980sModels:
    """1980s era machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.model_names = [
            'logistic_regression',
            'naive_bayes_gaussian',
            'naive_bayes_multinomial',
            'decision_tree',
            'kmeans'
        ]
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression (1980s classic)"""
        print("🔧 Training Logistic Regression (1980s)...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_naive_bayes_gaussian(self, X_train, y_train):
        """Train Gaussian Naive Bayes"""
        print("🔧 Training Gaussian Naive Bayes (1980s)...")
        model = GaussianNB()
        model.fit(X_train, y_train)
        self.models['naive_bayes_gaussian'] = model
        return model
    
    def train_naive_bayes_multinomial(self, X_train, y_train):
        """Train Multinomial Naive Bayes"""
        print("🔧 Training Multinomial Naive Bayes (1980s)...")
        model = MultinomialNB()
        # Ensure non-negative values for multinomial
        X_train_scaled = X_train - X_train.min() + 1e-10
        model.fit(X_train_scaled, y_train)
        self.models['naive_bayes_multinomial'] = model
        return model
    
    def train_decision_tree(self, X_train, y_train, max_depth=5):
        """Train shallow Decision Tree (1980s style)"""
        print(f"🔧 Training Decision Tree (1980s, depth={max_depth})...")
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        self.models['decision_tree'] = model
        return model
    
    def train_kmeans(self, X_train, y_train, n_clusters=None):
        """Train K-means clustering"""
        if n_clusters is None:
            n_clusters = len(np.unique(y_train))
        
        print(f"🔧 Training K-means (1980s, clusters={n_clusters})...")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X_train)
        self.models['kmeans'] = model
        return model
    
    def train_all(self, X_train, y_train):
        """Train all 1980s models"""
        print("\n🕹️ === TRAINING 1980s ERA MODELS ===")
        
        self.train_logistic_regression(X_train, y_train)
        self.train_naive_bayes_gaussian(X_train, y_train)
        self.train_naive_bayes_multinomial(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.train_kmeans(X_train, y_train)
        
        print("✅ All 1980s models trained!\n")
        return self.models
    
    def predict(self, model_name, X):
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        # Special handling for MultinomialNB
        if model_name == 'naive_bayes_multinomial':
            X = X - X.min() + 1e-10
        
        # Special handling for K-means (returns cluster labels)
        if model_name == 'kmeans':
            return model.predict(X)
        
        # For classifiers, return predictions and probabilities
        if hasattr(model, 'predict_proba'):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            return {
                'predictions': predictions,
                'probabilities': probabilities
            }
        else:
            return {
                'predictions': model.predict(X),
                'probabilities': None
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
            'era': '1980s',
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def evaluate_all(self, X_test, y_test):
        """Evaluate all 1980s models"""
        results = {}
        for model_name in self.models.keys():
            if model_name != 'kmeans':  # Skip clustering for classification eval
                results[model_name] = self.evaluate(model_name, X_test, y_test)
        return results
    
    def save_models(self, path='results/models_1980s.pkl'):
        """Save all models"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"✅ Saved 1980s models to {path}")
    
    def load_models(self, path='results/models_1980s.pkl'):
        """Load saved models"""
        with open(path, 'rb') as f:
            self.models = pickle.load(f)
        print(f"✅ Loaded 1980s models from {path}")

