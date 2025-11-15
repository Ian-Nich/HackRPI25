"""
RetroBrain Dataset Loader
Supports movie reviews sentiment analysis and CIFAR10
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import os
import pickle

class DataLoader:
    """Load and preprocess datasets for RetroBrain"""
    
    def __init__(self, dataset_type='sentiment'):
        """
        Initialize data loader
        
        Args:
            dataset_type: 'sentiment' for movie reviews or 'image' for CIFAR10
        """
        self.dataset_type = dataset_type
        self.vectorizer = None
        self.scaler = None
        
    def load_sentiment_data(self, test_size=0.2, max_features=5000):
        """
        Load movie reviews sentiment dataset
        
        Returns:
            X_train, X_test, y_train, y_test, vectorizer
        """
        print("📊 Loading movie reviews sentiment dataset...")
        
        # Use sklearn's movie reviews (or create synthetic data)
        # For demo, we'll create a synthetic sentiment dataset
        from sklearn.datasets import make_classification
        
        # Create synthetic text-like features for demo
        # In production, use actual movie reviews
        X, y = make_classification(
            n_samples=2000,
            n_features=max_features,
            n_informative=int(max_features * 0.3),
            n_redundant=int(max_features * 0.1),
            n_classes=2,
            random_state=42,
            n_clusters_per_class=1
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"✅ Loaded {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {len(np.unique(y_train))}")
        
        return X_train, X_test, y_train, y_test, self.scaler
    
    def load_text_data(self, categories=None, test_size=0.2, max_features=5000):
        """
        Load text classification data (20newsgroups as fallback)
        
        Args:
            categories: List of categories to load
            test_size: Test split ratio
            max_features: Maximum features for vectorization
            
        Returns:
            X_train, X_test, y_train, y_test, vectorizer
        """
        print("📊 Loading text classification dataset...")
        
        if categories is None:
            categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        
        try:
            # Load 20 newsgroups dataset
            newsgroups_train = fetch_20newsgroups(
                subset='train',
                categories=categories,
                shuffle=True,
                random_state=42
            )
            newsgroups_test = fetch_20newsgroups(
                subset='test',
                categories=categories,
                shuffle=True,
                random_state=42
            )
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            X_train = self.vectorizer.fit_transform(newsgroups_train.data).toarray()
            X_test = self.vectorizer.transform(newsgroups_test.data).toarray()
            
            y_train = newsgroups_train.target
            y_test = newsgroups_test.target
            
            # Scale features
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            print(f"✅ Loaded {len(X_train)} training samples, {len(X_test)} test samples")
            print(f"   Features: {X_train.shape[1]}")
            print(f"   Classes: {len(categories)}")
            
            return X_train, X_test, y_train, y_test, self.vectorizer
            
        except Exception as e:
            print(f"⚠️ Error loading 20newsgroups, using synthetic data: {e}")
            return self.load_sentiment_data(test_size, max_features)
    
    def prepare_for_prediction(self, raw_input):
        """
        Prepare raw input for model prediction
        
        Args:
            raw_input: Raw text or image array
            
        Returns:
            Processed input ready for models
        """
        if self.dataset_type == 'sentiment':
            if isinstance(raw_input, str):
                # If we have a vectorizer, use it
                if self.vectorizer:
                    vectorized = self.vectorizer.transform([raw_input]).toarray()
                else:
                    # Fallback: create random features (for demo)
                    vectorized = np.random.rand(1, 5000)
                
                if self.scaler:
                    return self.scaler.transform(vectorized)
                return vectorized
            else:
                # Assume it's already processed
                return raw_input
        else:
            return raw_input
    
    def save_preprocessor(self, path='data/preprocessor.pkl'):
        """Save preprocessor for later use"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'dataset_type': self.dataset_type
            }, f)
        print(f"✅ Saved preprocessor to {path}")
    
    def load_preprocessor(self, path='data/preprocessor.pkl'):
        """Load saved preprocessor"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.scaler = data['scaler']
            self.dataset_type = data['dataset_type']
        print(f"✅ Loaded preprocessor from {path}")

