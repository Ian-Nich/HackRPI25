"""
Train all models across 3 eras
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.data_loader import DataLoader
from ml.models.era_1980s import Era1980sModels
from ml.models.era_2000s import Era2000sModels
from ml.models.era_2020s import Era2020sModels
import json
import time


def train_all_eras(dataset_type='sentiment', api_key=None):
    """
    Train models from all 3 eras
    
    Args:
        dataset_type: 'sentiment' or 'image'
        api_key: Gemini API key (optional)
    """
    print("\n" + "="*60)
    print("🕹️ RETROBRAIN: TRAINING ALL ERA MODELS")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    # Load data
    data_loader = DataLoader(dataset_type=dataset_type)
    X_train, X_test, y_train, y_test, _ = data_loader.load_sentiment_data()
    
    # Save preprocessor
    data_loader.save_preprocessor('retrobrain/data/preprocessor.pkl')
    
    results = {
        'dataset_type': dataset_type,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1],
        'classes': len(set(y_train)),
        'eras': {}
    }
    
    # Train 1980s models
    print("\n" + "-"*60)
    print("📼 ERA 1: 1980s MODELS")
    print("-"*60)
    era_1980s = Era1980sModels()
    era_1980s.train_all(X_train, y_train)
    results_1980s = era_1980s.evaluate_all(X_test, y_test)
    era_1980s.save_models('retrobrain/results/models_1980s.pkl')
    
    results['eras']['1980s'] = results_1980s
    
    # Train 2000s models
    print("\n" + "-"*60)
    print("💿 ERA 2: 2000s MODELS")
    print("-"*60)
    era_2000s = Era2000sModels()
    era_2000s.train_all(X_train, y_train)
    results_2000s = era_2000s.evaluate_all(X_test, y_test)
    era_2000s.save_models('retrobrain/results/models_2000s.pkl')
    
    results['eras']['2000s'] = results_2000s
    
    # Initialize 2020s models
    print("\n" + "-"*60)
    print("🤖 ERA 3: 2020s MODELS")
    print("-"*60)
    era_2020s = Era2020sModels(api_key=api_key)
    era_2020s.train_all(X_train, y_train)
    results_2020s = {}
    
    # Evaluate Gemini (on sample)
    sample_size = min(100, len(X_test))
    sample_indices = range(sample_size)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    result_2020s = era_2020s.evaluate('gemini', X_sample, y_sample)
    results_2020s['gemini'] = result_2020s
    era_2020s.save_models('retrobrain/results/models_2020s.pkl')
    
    results['eras']['2020s'] = results_2020s
    
    # Calculate timing
    elapsed_time = time.time() - start_time
    results['training_time_seconds'] = elapsed_time
    
    # Save results
    os.makedirs('retrobrain/results', exist_ok=True)
    with open('retrobrain/results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📊 Results saved to: retrobrain/results/training_results.json")
    print()
    
    # Print summary
    print("\n📈 ACCURACY SUMMARY:")
    print("-" * 60)
    for era_name, era_results in results['eras'].items():
        print(f"\n{era_name.upper()} ERA:")
        for model_name, model_result in era_results.items():
            accuracy = model_result.get('accuracy', 0)
            print(f"  {model_name:30s} {accuracy:.4f}")
    
    print("\n" + "="*60 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RetroBrain models')
    parser.add_argument('--dataset', type=str, default='sentiment',
                        choices=['sentiment', 'image'],
                        help='Dataset type')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Gemini API key')
    
    args = parser.parse_args()
    
    train_all_eras(dataset_type=args.dataset, api_key=args.api_key)

