import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import time
import os

def load_and_flatten_features(csv_path, max_samples=None, seq_len=210):
    """
    Load video features and flatten them for traditional ML models
    Ensures all videos have the same sequence length
    """
    df = pd.read_csv(csv_path)
    
    if max_samples:
        print(f"Using {max_samples} samples for faster training")
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    features_list = []
    labels_list = []
    
    print(f"Loading features from {len(df)} videos...")
    
    for idx, row in df.iterrows():
        try:
            # Load features
            features = np.load(row['features_path'])
            
            # Ensure consistent shape (seq_len, feature_dim)
            if features.shape[0] < seq_len:
                # Pad with zeros if too short
                pad_size = seq_len - features.shape[0]
                padding = np.zeros((pad_size, features.shape[1]))
                features = np.vstack([features, padding])
            elif features.shape[0] > seq_len:
                # Truncate if too long
                features = features[:seq_len]
            
            # Flatten to 1D vector for traditional ML
            features_flat = features.flatten()
            
            features_list.append(features_flat)
            labels_list.append(1 if row['label'] == 'real' else 0)
            
            if len(features_list) % 100 == 0:
                print(f"Loaded {len(features_list)} features...")
                
        except Exception as e:
            print(f"Error loading {row['features_path']}: {e}")
            continue
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Feature vector size per video: {X.shape[1]:,} features")
    print(f"Class distribution: Real={np.sum(y)}, Fake={len(y)-np.sum(y)}")
    
    return X, y

def apply_dimensionality_reduction(X_train, X_test, method='pca', n_components=100):
    """Apply dimensionality reduction for faster training"""
    print(f"Applying {method.upper()} dimensionality reduction to {n_components} components...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Only PCA supported for now")
    
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    
    print(f"Reduced from {X_train.shape[1]} to {X_train_reduced.shape[1]} features")
    print(f"Explained variance ratio: {np.sum(reducer.explained_variance_ratio_):.3f}")
    
    return X_train_reduced, X_test_reduced, reducer

def train_fast_models(X_train, X_test, y_train, y_test):
    """Train fast ML models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest (Fast)': RandomForestClassifier(
            n_estimators=50,  # Reduced from default 100
            max_depth=10,     # Limit depth
            random_state=42,
            n_jobs=-1
        ),
        'KNN (Fast)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'SVM (Linear)': SVC(kernel='linear', random_state=42, max_iter=1000),
        'XGBoost (Fast)': xgb.XGBClassifier(
            n_estimators=50,      # Much fewer trees
            max_depth=6,          # Shallow trees
            learning_rate=0.1,    # Standard rate
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'train_time': train_time,
            'predictions': y_pred
        }
    
    return results

def main():
    # Load data with sampling for faster experimentation
    print("Loading video features...")
    X, y = load_and_flatten_features('video_features_metadata.csv', max_samples=500)  # Use 500 samples for speed
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Apply PCA for dimensionality reduction
    X_train_reduced, X_test_reduced, pca = apply_dimensionality_reduction(
        X_train, X_test, method='pca', n_components=50  # Reduced components for speed
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    
    # Train models
    results = train_fast_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, result in sorted_results:
        print(f"{name:20} | Accuracy: {result['accuracy']:.4f} | Time: {result['train_time']:.2f}s")
    
    return results

if __name__ == "__main__":
    results = main()