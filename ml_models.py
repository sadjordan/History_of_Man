import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DeepfakeMLClassifier:
    def __init__(self, csv_path):
        """Initialize with video features metadata"""
        self.df = pd.read_csv(csv_path)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_process_features(self):
        """Load all video features and create feature matrix"""
        print("Loading and processing video features...")
        
        X_list = []
        y_list = []
        
        for idx, row in self.df.iterrows():
            try:
                # Load video features (shape: num_frames, 2048)
                features = np.load(row['features_path'])
                
                # Feature engineering: extract statistical summaries from sequence
                feature_vector = self.extract_statistical_features(features)
                
                X_list.append(feature_vector)
                y_list.append(row['label'])
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(self.df)} videos")
                    
            except Exception as e:
                print(f"Error processing {row['features_path']}: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Class distribution: {np.unique(y, return_counts=True)}")
        
        return X, y
    
    def extract_statistical_features(self, sequence_features):
        """Extract statistical features from the sequence of ResNet features"""
        # sequence_features shape: (num_frames, 2048)
        
        features = []
        
        # 1. Basic statistics across time dimension
        features.extend(np.mean(sequence_features, axis=0))      # Mean of each feature across time
        features.extend(np.std(sequence_features, axis=0))       # Std of each feature across time
        features.extend(np.max(sequence_features, axis=0))       # Max of each feature across time
        features.extend(np.min(sequence_features, axis=0))       # Min of each feature across time
        features.extend(np.median(sequence_features, axis=0))    # Median of each feature across time
        
        # 2. Temporal dynamics
        if len(sequence_features) > 1:
            # First and last frame differences
            first_last_diff = sequence_features[-1] - sequence_features[0]
            features.extend(first_last_diff)
            
            # Frame-to-frame differences statistics
            frame_diffs = np.diff(sequence_features, axis=0)
            features.extend(np.mean(frame_diffs, axis=0))
            features.extend(np.std(frame_diffs, axis=0))
        else:
            # Pad with zeros if only one frame
            features.extend(np.zeros(2048 * 3))
        
        # 3. Percentiles
        features.extend(np.percentile(sequence_features, 25, axis=0))  # 25th percentile
        features.extend(np.percentile(sequence_features, 75, axis=0))  # 75th percentile
        
        # 4. Global sequence statistics
        features.append(len(sequence_features))  # Number of frames
        features.append(np.mean(sequence_features))  # Global mean
        features.append(np.std(sequence_features))   # Global std
        features.append(np.var(sequence_features))   # Global variance
        
        return np.array(features)
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Load data and split into train/test sets"""
        X, y = self.load_and_process_features()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Class distribution in training: {np.unique(y_train, return_counts=True)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize all ML models with good default parameters"""
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            
            'Naive Bayes': GaussianNB()
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_and_evaluate_all(self):
        """Train and evaluate all models"""
        print("\n" + "="*50)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def hyperparameter_tuning(self, model_name='XGBoost'):
        """Perform hyperparameter tuning for best performing model"""
        print(f"\n--- Hyperparameter Tuning for {model_name} ---")
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
        
        else:
            print(f"No tuning parameters defined for {model_name}")
            return
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Evaluate best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Test F1-score: {f1_score(self.y_test, y_pred):.4f}")
        
        return best_model
    
    def print_detailed_results(self):
        """Print comprehensive results comparison"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS COMPARISON")
        print("="*80)
        
        # Sort by F1 score
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'CV-Acc':<10}")
        print("-" * 80)
        
        for name, metrics in sorted_results:
            print(f"{name:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['cv_mean']:<10.4f}")
        
        # Print confusion matrices for top 3 models
        print("\n" + "="*50)
        print("CONFUSION MATRICES (Top 3 Models)")
        print("="*50)
        
        for i, (name, metrics) in enumerate(sorted_results[:3]):
            print(f"\n{name}:")
            cm = confusion_matrix(self.y_test, metrics['predictions'])
            print(cm)
            print(f"Classification Report:")
            print(classification_report(self.y_test, metrics['predictions'], 
                                      target_names=self.label_encoder.classes_))

def main():
    """Main function to run all ML experiments"""
    # Initialize classifier
    classifier = DeepfakeMLClassifier('video_features_metadata.csv')
    
    # Prepare data
    classifier.prepare_data(test_size=0.2, random_state=42)
    
    # Initialize and train all models
    classifier.initialize_models()
    classifier.train_and_evaluate_all()
    
    # Print results
    classifier.print_detailed_results()
    
    # Hyperparameter tuning for best model
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Find best model
    best_model_name = max(classifier.results.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"Best performing model: {best_model_name}")
    
    # Tune hyperparameters
    best_tuned_model = classifier.hyperparameter_tuning(best_model_name)
    
    print(f"\nProcessing complete! Check results above.")
    print(f"Total videos processed: {len(classifier.df)}")
    print(f"Feature dimension: {classifier.X_train.shape[1]}")

if __name__ == "__main__":
    main()