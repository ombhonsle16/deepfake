import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
import pandas as pd
import random
from PIL import Image
import concurrent.futures

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess videos for deepfake detection')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed frames')
    parser.add_argument('--sample_rate', type=int, default=5, help='Sample every nth frame')
    parser.add_argument('--face_size', type=int, default=256, help='Size of face crops')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    return parser.parse_args()

def extract_frames(video_path, output_dir, sample_rate=5):
    """Extract frames from a video file."""
    video_name = os.path.basename(video_path).split('.')[0]
    frames_dir = os.path.join(output_dir, 'frames', video_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frame_path = os.path.join(frames_dir, f"{video_name}_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frames_dir, saved_count

def detect_and_align_face(image_path, detector, output_dir, face_size=256):
    """Detect and align faces in an image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(rgb_image)
        if not faces:
            return None
        
        # Use the face with highest confidence
        face = max(faces, key=lambda x: x['confidence'])
        
        # Get bounding box
        x, y, width, height = face['box']
        
        # Add margin (20%)
        margin = int(max(width, height) * 0.2)
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(image.shape[1], x + width + margin)
        y_max = min(image.shape[0], y + height + margin)
        
        # Crop face
        face_img = image[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        face_img = cv2.resize(face_img, (face_size, face_size))
        
        # Save face
        base_name = os.path.basename(image_path)
        face_path = os.path.join(output_dir, 'faces', base_name)
        os.makedirs(os.path.dirname(face_path), exist_ok=True)
        cv2.imwrite(face_path, face_img)
        
        return face_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_video(video_path, output_dir, detector, sample_rate=5, face_size=256):
    """Process a single video: extract frames and detect faces."""
    try:
        frames_dir, frame_count = extract_frames(video_path, output_dir, sample_rate)
        if frame_count == 0:
            print(f"No frames extracted from {video_path}")
            return 0
        
        # Process frames to extract faces
        face_count = 0
        for frame_file in os.listdir(frames_dir):
            frame_path = os.path.join(frames_dir, frame_file)
            face_path = detect_and_align_face(frame_path, detector, output_dir, face_size)
            if face_path:
                face_count += 1
        
        return face_count
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return 0

def apply_augmentation(face_dir, output_dir):
    """Apply data augmentation to face images."""
    augmented_dir = os.path.join(output_dir, 'augmented')
    os.makedirs(augmented_dir, exist_ok=True)
    
    face_files = [f for f in os.listdir(face_dir) if f.endswith(('.jpg', '.png'))]
    
    for face_file in tqdm(face_files, desc="Augmenting faces"):
        face_path = os.path.join(face_dir, face_file)
        img = Image.open(face_path)
        
        # Original
        img.save(os.path.join(augmented_dir, f"orig_{face_file}"))
        
        # Horizontal flip
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip.save(os.path.join(augmented_dir, f"flip_{face_file}"))
        
        # Rotation (slight)
        img_rot = img.rotate(random.uniform(-15, 15))
        img_rot.save(os.path.join(augmented_dir, f"rot_{face_file}"))
        
        # Brightness adjustment
        img_array = np.array(img)
        brightness = random.uniform(0.7, 1.3)
        img_bright = Image.fromarray(np.clip(img_array * brightness, 0, 255).astype(np.uint8))
        img_bright.save(os.path.join(augmented_dir, f"bright_{face_file}"))

def create_dataset_csv(output_dir, label_map=None):
    """Create a CSV file with paths and labels for the dataset."""
    if label_map is None:
        # Default: assume directory names are class labels
        label_map = {}
        
    faces_dir = os.path.join(output_dir, 'faces')
    dataset = []
    
    for root, _, files in os.walk(faces_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, output_dir)
                
                # Determine label from directory structure
                dir_name = os.path.basename(os.path.dirname(file_path))
                label = label_map.get(dir_name, dir_name)
                
                dataset.append({
                    'path': rel_path,
                    'label': label
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(dataset)
    csv_path = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset CSV created at {csv_path}")
    
    # Print class distribution
    print("Class distribution:")
    print(df['label'].value_counts())
    
    return csv_path

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'faces'), exist_ok=True)
    
    # Initialize face detector
    detector = MTCNN()
    
    # Get all video files
    video_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files")
    
    # Process videos in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for video_path in video_files:
            future = executor.submit(
                process_video, 
                video_path, 
                args.output_dir, 
                detector, 
                args.sample_rate, 
                args.face_size
            )
            futures.append(future)
        
        # Process results as they complete
        total_faces = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing videos"):
            face_count = future.result()
            total_faces += face_count
    
    print(f"Extracted {total_faces} faces from {len(video_files)} videos")
    
    # Apply data augmentation
    faces_dir = os.path.join(args.output_dir, 'faces')
    apply_augmentation(faces_dir, args.output_dir)
    
    # Create dataset CSV
    create_dataset_csv(args.output_dir)

if __name__ == "__main__":
    main() 