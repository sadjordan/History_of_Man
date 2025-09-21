import pandas as pd
import numpy as np
import os
import csv
from tqdm import tqdm

def load_and_segment_features_to_csv(csv_path, output_csv_path, frames_per_segment=10, max_samples=None):
    """
    Load video features, split each video into segments of 10 frames each,
    and save to CSV file by writing rows directly (memory efficient).
    Only process all real videos and a random subset of fake videos equal to the number of real videos.
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Separate real and fake videos
    real_df = df[df['label'] == 'real']
    fake_df = df[df['label'] != 'real']
    
    # Randomly sample fake videos to match the number of real videos
    fake_df = fake_df.sample(n=len(real_df), random_state=42)
    
    # Combine balanced data
    balanced_df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    if max_samples:
        balanced_df = balanced_df.sample(n=max_samples, random_state=42)
        print(f"Using {len(balanced_df)} videos for processing")
    
    print(f"Processing {len(balanced_df)} videos into 10-frame segments (balanced real/fake)...")
    
    feature_dim = 2048
    total_features = frames_per_segment * feature_dim
    
    header = ['video_id', 'original_label', 'segment_idx', 'start_frame', 'end_frame', 'label_binary']
    header.extend([f'feature_{i}' for i in range(total_features)])
    
    total_segments = 0
    real_segments = 0
    fake_segments = 0
    processed_videos = 0
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df), desc="Processing videos"):
            try:
                features = np.load(row['features_path'])
                if features.shape[0] < frames_per_segment:
                    print(f"Skipping {row['video_id']} - only {features.shape[0]} frames")
                    continue
                if features.shape[0] < 210:
                    pad_frames = 210 - features.shape[0]
                    padding = np.zeros((pad_frames, features.shape[1]))
                    features = np.vstack([features, padding])
                else:
                    features = features[:210]
                num_segments = 210 // frames_per_segment
                for segment_idx in range(num_segments):
                    start_frame = segment_idx * frames_per_segment
                    end_frame = start_frame + frames_per_segment
                    segment = features[start_frame:end_frame]
                    segment_flat = segment.flatten()
                    label_binary = 1 if row['label'] == 'real' else 0
                    csv_row = [
                        row['video_id'],
                        row['label'],
                        segment_idx,
                        start_frame,
                        end_frame,
                        label_binary
                    ]
                    csv_row.extend(segment_flat.tolist())
                    writer.writerow(csv_row)
                    total_segments += 1
                    if label_binary == 1:
                        real_segments += 1
                    else:
                        fake_segments += 1
                processed_videos += 1
            except Exception as e:
                print(f"Error processing {row['video_id']}: {e}")
                continue
    
    print(f"\nData preparation complete:")
    print(f"Total segments: {total_segments}")
    print(f"Feature columns: {total_features}")
    print(f"Processed videos: {processed_videos}")
    print(f"Segments per video: {total_segments // processed_videos if processed_videos > 0 else 0}")
    print(f"Class distribution: Real={real_segments}, Fake={fake_segments}")
    print(f"✅ Successfully saved {total_segments} segments to CSV!")
    return None

def main():
    if not os.path.exists('video_features_metadata.csv'):
        print("Error: video_features_metadata.csv not found!")
        print("Please make sure you have run the feature extraction script first.")
        return
    print("Converting video features to segmented CSV format (balanced real/fake)...")
    load_and_segment_features_to_csv(
        csv_path='video_features_metadata.csv',
        output_csv_path='video_segments_10frames_balanced.csv',
        frames_per_segment=10,
        max_samples=None
    )
    print(f"\n🎉 Conversion complete!")
    print(f"Your balanced segmented data is now saved in 'video_segments_10frames_balanced.csv'")
    print(f"You can now use this CSV with any ML library or tool.")
    return None

if __name__ == "__main__":
    main()
