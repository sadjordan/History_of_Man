# NEW STANDALONE IMPLEMENTATION - run this in a fresh kernel/session

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from ultralytics import YOLO

# Force TensorFlow to use CPU to avoid GPU memory issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Force clean environment
tf.keras.backend.clear_session()

def process_single_video(video_path, output_dir=None):
    """Process a single video completely isolated from any other code"""
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = "processed_vectors"
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}.npy")
    
    print(f"Processing: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Step 1: Create a fresh ResNet model (switching from EfficientNet)
    print("Creating feature extractor...")
    inputs = keras.Input(shape=(224, 224, 3))
    resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_tensor=inputs)
    feature_extractor = Model(inputs=inputs, outputs=resnet.output)
    
    # Step 2: Load face detector
    print("Loading face detector...")
    face_detector = YOLO('yolov8n.pt')
    
    # Step 3: Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return
    
    # Extract faces from video
    print("Extracting faces from video...")
    face_frames = []
    frame_count = 0
    
    while frame_count < 210:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        results = face_detector.predict(source=frame, conf=0.5, classes=0, verbose=False)
        
        # Find largest face
        largest_face = None
        largest_area = 0
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area > largest_area:
                    largest_area = area
                    largest_face = (x1, y1, x2, y2)
        
        # Process face if found
        if largest_face:
            x1, y1, x2, y2 = largest_face
            face = frame[y1:y2, x1:x2]
            
            # Resize and convert to RGB (OpenCV uses BGR)
            face_resized = cv2.resize(face, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Add to list
            face_frames.append(face_rgb)
            if frame_count % 20 == 0:
                print(f"Processed frame {frame_count} - face detected")
        else:
            if frame_count % 20 == 0:
                print(f"Processed frame {frame_count} - no face detected")
        
        frame_count += 1
    
    cap.release()
    
    # Step 4: Extract features if faces found
    if len(face_frames) > 0:
        print(f"Found {len(face_frames)} faces. Extracting features...")
        
        # Convert to array and normalize to 0-1
        face_array = np.array(face_frames, dtype=np.float32) / 255.0
        
        # Process in smaller batches to avoid memory issues
        batch_size = 32
        all_features = []
        
        for i in range(0, len(face_array), batch_size):
            batch = face_array[i:i+batch_size]
            batch_features = feature_extractor.predict(batch, verbose=0)
            all_features.append(batch_features)
            print(f"Processed batch {i//batch_size + 1}/{(len(face_array)-1)//batch_size + 1}")
        
        # Combine features
        features = np.vstack(all_features)
        
        # Save features
        np.save(output_path, features)
        print(f"✅ SUCCESS: Saved {len(features)} feature vectors to {output_path}")
        return True
    else:
        print("❌ ERROR: No faces detected in video")
        return False

# Test with single video
print("\n==== STARTING FRESH VIDEO PROCESSING ====\n")

# Get video path from environment
from dotenv import load_dotenv
load_dotenv()

fake_path = os.getenv('DEEPFAKE_PATH')
if fake_path and os.path.exists(fake_path):
    fake_videos = [f for f in os.listdir(fake_path) if f.endswith('.mp4')]
    
    if fake_videos:
        # Use the first video for testing
        first_video = fake_videos[0]
        video_path = os.path.join(fake_path, first_video)
        
        # Create output directory
        output_dir = "processed_vectors"
        
        # Process the video
        process_single_video(video_path, output_dir)
    else:
        print("No videos found in the fake path")
else:
    print("DEEPFAKE_PATH not found in environment variables")
