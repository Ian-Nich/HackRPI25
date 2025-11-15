"""
Evaluate all models and generate metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.data_loader import DataLoader
from ml.models.era_1980s import Era1980sModels
from ml.models.era_2000s import Era2000sModels
from ml.models.era_2020s import Era2020sModels
import json
import numpy as np


def evaluate_all_models():
    """Evaluate all trained models"""
    print("\n" + "="*60)
    print("🔍 RETROBRAIN: EVALUATING ALL MODELS")
    print("="*60 + "\n")
    
    # Load data
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test, _ = data_loader.load_sentiment_data()
    
    results = {}
    
    # Load and evaluate 1980s models
    print("📼 Evaluating 1980s models...")
    era_1980s = Era1980sModels()
    try:
        era_1980s.load_models('retrobrain/results/models_1980s.pkl')
        results['1980s'] = era_1980s.evaluate_all(X_test, y_test)
    except FileNotFoundError:
        print("⚠️  1980s models not found. Training first...")
        era_1980s.train_all(X_train, y_train)
        results['1980s'] = era_1980s.evaluate_all(X_test, y_test)
    
    # Load and evaluate 2000s models
    print("💿 Evaluating 2000s models...")
    era_2000s = Era2000sModels()
    try:
        era_2000s.load_models('retrobrain/results/models_2000s.pkl')
        results['2000s'] = era_2000s.evaluate_all(X_test, y_test)
    except FileNotFoundError:
        print("⚠️  2000s models not found. Training first...")
        era_2000s.train_all(X_train, y_train)
        results['2000s'] = era_2000s.evaluate_all(X_test, y_test)
    
    # Evaluate 2020s models
    print("🤖 Evaluating 2020s models...")
    era_2020s = Era2020sModels()
    try:
        era_2020s.load_models('retrobrain/results/models_2020s.pkl')
    except FileNotFoundError:
        era_2020s.train_all(X_train, y_train)
    
    # Sample for Gemini (faster)
    sample_size = min(100, len(X_test))
    sample_indices = range(sample_size)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    result_2020s = era_2020s.evaluate('gemini', X_sample, y_sample)
    results['2020s'] = {'gemini': result_2020s}
    
    # Save evaluation results
    os.makedirs('retrobrain/results', exist_ok=True)
    with open('retrobrain/results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print(f"📊 Results saved to: retrobrain/results/evaluation_results.json\n")
    
    return results


if __name__ == "__main__":
    evaluate_all_models()

