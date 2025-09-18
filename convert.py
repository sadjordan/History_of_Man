import cv2
import numpy as np
import argparse
import os

def npy_to_mp4(npy_path, output_path=None, fps=30):
    """
    Convert a .npy file containing video sequence to MP4
    
    Args:
        npy_path: Path to .npy file with shape (num_frames, 224, 224, 3)
        output_path: Output MP4 path (optional, defaults to same name as .npy)
        fps: Frames per second for output video
    """
    
    # Load the numpy array
    try:
        video_array = np.load(npy_path)
        print(f"Loaded video array with shape: {video_array.shape}")
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return
    
    # Validate array shape
    if len(video_array.shape) != 4 or video_array.shape[-1] != 3:
        print(f"Invalid array shape: {video_array.shape}. Expected (num_frames, height, width, 3)")
        return
    
    num_frames, height, width, channels = video_array.shape
    
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.splitext(npy_path)[0]
        output_path = f"{base_name}_converted.mp4"
    
    # Convert from [-1, 1] range back to [0, 255] uint8
    # The original scaling was: face_scaled = face_resized.astype(np.float32) / 127.5 - 1.0
    # So to reverse: (face_scaled + 1.0) * 127.5
    video_uint8 = ((video_array + 1.0) * 127.5).astype(np.uint8)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    print(f"Converting {num_frames} frames to MP4...")
    
    # Write each frame
    for i in range(num_frames):
        frame = video_uint8[i]
        # OpenCV expects BGR, but our data might be RGB, so convert
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{num_frames} frames")
    
    out.release()
    print(f"✅ Successfully converted to: {output_path}")
    print(f"Video info: {num_frames} frames, {width}x{height}, {fps} FPS")

def main():
    parser = argparse.ArgumentParser(description='Convert .npy video sequence to MP4')
    parser.add_argument('npy_path', help='Path to input .npy file')
    parser.add_argument('-o', '--output', help='Output MP4 path (optional)')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS (default: 30)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.npy_path):
        print(f"Error: File {args.npy_path} does not exist")
        return
    
    # Convert the file
    npy_to_mp4(args.npy_path, args.output, args.fps)

if __name__ == "__main__":
    main()