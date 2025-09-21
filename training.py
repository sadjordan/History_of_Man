import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Data loader class
class VideoDataGenerator:
    def __init__(self, csv_path, seq_len=210, batch_size=32):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        
    def load_and_pad_features(self, path):
        features = np.load(path)
        if features.shape[0] < self.seq_len:
            # Pad with zeros
            pad = np.zeros((self.seq_len - features.shape[0], features.shape[1]))
            features = np.vstack([features, pad])
        else:
            # Truncate
            features = features[:self.seq_len]
        return features
    
    def generate_undersampled_batches(self, indices):
        """Generate batches with undersampling of majority class to achieve 60-40 split"""
        # Separate indices by class
        real_indices = []
        fake_indices = []
        
        for idx in indices:
            if self.df.iloc[idx]['label_encoded'] == 0:  # real
                real_indices.append(idx)
            else:  # fake
                fake_indices.append(idx)
        
        print(f"Original - Real: {len(real_indices)}, Fake: {len(fake_indices)}")
        
        # Determine majority and minority classes
        if len(real_indices) > len(fake_indices):
            majority_indices = real_indices
            minority_indices = fake_indices
            majority_class = "real"
        else:
            majority_indices = fake_indices
            minority_indices = real_indices
            majority_class = "fake"
        
        # Calculate target size for 60-40 split (minority = 40%, majority = 60%)
        target_minority_size = len(minority_indices)  # Keep all minority samples
        target_majority_size = int(target_minority_size * 1.5)  # 60% majority, 40% minority
        
        # Randomly sample from majority class
        np.random.seed(42)  # For reproducibility
        if target_majority_size < len(majority_indices):
            majority_indices_sampled = np.random.choice(
                majority_indices, 
                size=target_majority_size, 
                replace=False
            ).tolist()
        else:
            majority_indices_sampled = majority_indices
        
        # Combine all indices
        if majority_class == "real":
            all_indices = majority_indices_sampled + minority_indices
            print(f"After undersampling - Real: {len(majority_indices_sampled)}, Fake: {len(minority_indices)}")
        else:
            all_indices = minority_indices + majority_indices_sampled
            print(f"After undersampling - Real: {len(minority_indices)}, Fake: {len(majority_indices_sampled)}")
        
        print(f"Total training samples: {len(all_indices)}")
        print(f"Class distribution: {len(majority_indices_sampled)}/{len(all_indices)*100:.1f}% majority, {len(minority_indices)}/{len(all_indices)*100:.1f}% minority")
        
        while True:
            np.random.shuffle(all_indices)
            for i in range(0, len(all_indices), self.batch_size):
                batch_indices = all_indices[i:i+self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue
                    
                batch_x = []
                batch_y = []
                
                for idx in batch_indices:
                    row = self.df.iloc[idx]
                    features = self.load_and_pad_features(row['features_path'])
                    batch_x.append(features)
                    batch_y.append(row['label_encoded'])
                
                yield np.array(batch_x), np.array(batch_y)
    
    def generate_batches(self, indices):
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_x = []
                batch_y = []
                
                for idx in batch_indices:
                    row = self.df.iloc[idx]
                    features = self.load_and_pad_features(row['features_path'])
                    batch_x.append(features)
                    batch_y.append(row['label_encoded'])
                
                yield np.array(batch_x), np.array(batch_y)

# Create LSTM-CNN model
def create_lstm_cnn_model(input_shape, class_weights=None):
    model = Sequential([
        # 1D CNN layers for temporal feature extraction
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # LSTM layers for sequence modeling
        LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.3),
        LSTM(64, dropout=0.4, recurrent_dropout=0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        Dropout(0.6),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Use standard binary crossentropy since we have balanced data now
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',  # Standard loss for balanced data
        metrics=[
            'accuracy', 
            'precision', 
            'recall',
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.FalseNegatives(name='fn')
        ]
    )
    
    return model

# Load data and create generators
data_gen = VideoDataGenerator('video_features_metadata.csv', seq_len=210, batch_size=16)

# Check class distribution
print("Class distribution:")
print(data_gen.df['label'].value_counts())
print(f"Class balance: {data_gen.df['label'].value_counts(normalize=True)}")

# Check label encoding
print("\nLabel encoding:")
print("Real videos encoded as:", data_gen.label_encoder.transform(['real'])[0])
print("Fake videos encoded as:", data_gen.label_encoder.transform(['fake'])[0])

# Split data: 70% train, 15% validation, 15% test
train_indices, temp_indices = train_test_split(
    range(len(data_gen.df)), 
    test_size=0.3, 
    random_state=42,
    stratify=data_gen.df['label_encoded']
)

val_indices, test_indices = train_test_split(
    temp_indices, 
    test_size=0.5, 
    random_state=42,
    stratify=data_gen.df.iloc[temp_indices]['label_encoded']
)

# Create data generators with undersampling for training
train_gen = data_gen.generate_undersampled_batches(train_indices)
val_gen = data_gen.generate_batches(val_indices)
test_gen = data_gen.generate_batches(test_indices)

# Get input shape (sequence_length, feature_dim)
sample_features = np.load(data_gen.df.iloc[0]['features_path'])
input_shape = (210, sample_features.shape[1])  # (seq_len, feature_dim)

print(f"Input shape: {input_shape}")
print(f"Training samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Test samples: {len(test_indices)}")

# Create and train model
model = create_lstm_cnn_model(input_shape)
print(model.summary())

# Training callbacks with more aggressive early stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=3, 
        restore_best_weights=True,
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5, 
        patience=2,
        min_lr=0.00001
    )
]

# Train the model with undersampled data
# Calculate effective training size after undersampling
real_count = sum(1 for idx in train_indices if data_gen.df.iloc[idx]['label_encoded'] == 0)
fake_count = len(train_indices) - real_count

print(f"Before undersampling - Real: {real_count}, Fake: {fake_count}")

# Calculate effective size for 60-40 split
minority_count = min(real_count, fake_count)
majority_count = int(minority_count * 1.5)  # 60% majority, 40% minority
effective_train_size = minority_count + majority_count

print(f"Original training size: {len(train_indices)}")
print(f"Effective training size after undersampling: {effective_train_size}")

history = model.fit(
    train_gen,
    steps_per_epoch=effective_train_size // 16,
    validation_data=val_gen,
    validation_steps=len(val_indices) // 16,
    epochs=20,
    callbacks=callbacks
)

# Evaluate on test set
test_results = model.evaluate(
    test_gen, 
    steps=len(test_indices) // 16
)

# Extract the metrics we need (loss, accuracy, precision, recall are first 4)
test_loss = test_results[0]
test_acc = test_results[1] 
test_prec = test_results[2]
test_rec = test_results[3]
# Additional metrics: tp=4, tn=5, fp=6, fn=7

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall: {test_rec:.4f}")

# Also print the confusion matrix components
if len(test_results) >= 8:
    tp, tn, fp, fn = test_results[4], test_results[5], test_results[6], test_results[7]
    print(f"True Positives: {tp:.0f}")
    print(f"True Negatives: {tn:.0f}")
    print(f"False Positives: {fp:.0f}")
    print(f"False Negatives: {fn:.0f}")

# Calculate F1-score safely
if test_prec + test_rec > 0:
    f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec)
    print(f"Test F1-Score: {f1_score:.4f}")
else:
    print("Test F1-Score: 0.0000 (precision and recall are both 0)")

# Save the model
model.save('deepfake_detection_model.h5')
print("Model saved as 'deepfake_detection_model.h5'")