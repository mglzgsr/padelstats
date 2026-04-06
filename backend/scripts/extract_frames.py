import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval=30):
    """
    Extracts frames from a video every 'interval' frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path} to {output_dir}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            frame_name = f"frame_{count:06d}.jpg"
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} frames...")
            
    cap.release()
    print(f"Finished! Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames for training")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="dataset/images", help="Output directory")
    parser.add_argument("--interval", type=int, default=60, help="Frame interval")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval)
