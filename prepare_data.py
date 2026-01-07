import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path

# Configuration
DATASET_ROOT = "UCLM_Thermal_Imaging_Dataset"
OUTPUT_DIR = "datasets/custom_weapon"
TRAIN_RATIO = 0.8

# Ensure output directories exist
for split in ['train', 'val']:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

def process_dataset():
    print(f"🚀 Starting data processing from {DATASET_ROOT}...")
    
    # Categories mapping (from label.json inspection)
    # 1: Handgun, 2: Person
    # We will map Handgun -> 0, Person -> 1
    CATEGORY_MAP = {1: 0}
    
    # Statistics
    total_frames = 0
    processed_count = 0
    
    # Walk through the dataset
    dataset_path = Path(DATASET_ROOT)
    # We look for folders containing label.json
    for label_file in dataset_path.rglob("label.json"):
        folder_path = label_file.parent
        print(f"Processing folder: {folder_path.name}")
        
        # Load labels
        with open(label_file, 'r') as f:
            data = json.load(f)
            
        # Video file path (assuming video.mp4 describes the file in the same folder)
        video_path = folder_path / "video.mp4"
        if not video_path.exists():
            print(f"⚠️ Video not found: {video_path}")
            continue
            
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            continue
            
        # Process annotations
        # Group annotations by image_id (frame)
        anns_by_frame = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in anns_by_frame:
                anns_by_frame[img_id] = []
            anns_by_frame[img_id].append(ann)
            
        # Map video ID to timestamp/frame info if needed, but 'annotations' usually link to 'video' or 'images' list via image_id
        # In this dataset format (COCO-like for video), 'video' list maps IDs to timestamps.
        # However, usually image_id corresponds to the frame index or a mapped file.
        # Let's inspect the 'video' list in label.json structure from previous view_file.
        # The 'video' list has objects with "id" (which matches image_id in annotations) and "time_stamp".
        # So image_id 1 is the first frame described, etc.
        # We need to find which frame index corresponds to valid data.
        # 'video' list helps us know the timestamp. 
        # BUT, simpler approach: iterate through 'video' list items. Each item is a "frame".
        
        video_metadata = {item['id']: item for item in data.get('video', [])}
        
        for img_id, anns in anns_by_frame.items():
            if img_id not in video_metadata:
                continue
                
            frame_meta = video_metadata[img_id]
            # timestamp = frame_meta['time_stamp'] 
             # Use timestamp to seek? Or just count frames?
             # Usually standard frame rate. 
             # Let's try getting frame by timestamp if possible, or just sequential if ids are sequential.
             # ids seem sequential 1..100+.
             # Let's check frame count.
             
            # Better approach: Read video sequentially and match with IDs if they match frame numbers.
            # "time_stamp": 0.16 -> implies not frame 0.
            # Let's assume ID is just a unique identifier and we rely on timestamp.
            
            timestamp = frame_meta['time_stamp']
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret:
                # Fallback: sometimes timestamp seek isn't perfect. 
                # If IDs are 1-based index, maybe frame index = id - 1?
                # Let's try frame index based on info if available.
                # Re-reading sequentially might be safer but slower.
                continue
                
            # Get dimensions
            h, w = frame.shape[:2]
            
            # Determine split (simple random or based on folder)
            split = 'train' if np.random.rand() < TRAIN_RATIO else 'val'
            
            # Filename
            base_name = f"{folder_path.name}_{img_id}"
            img_filename = f"{base_name}.jpg"
            txt_filename = f"{base_name}.txt"
            
            # Save Image
            save_path = f"{OUTPUT_DIR}/images/{split}/{img_filename}"
            cv2.imwrite(save_path, frame)
            
            # Save Labels
            label_path = f"{OUTPUT_DIR}/labels/{split}/{txt_filename}"
            with open(label_path, 'w') as lf:
                for ann in anns:
                    cat_id = ann['category_id']
                    if cat_id not in CATEGORY_MAP:
                        continue
                    
                    cls = CATEGORY_MAP[cat_id]
                    
                    # BBox is [x, y, w, h] (top-left)
                    bbox = ann['position'] # checking json sample: "position":[52,524,72,108]
                    # Is it [x, y, w, h]? "type":"Rectangle". Usually [x, y, w, h] in COCO.
                    bx, by, bw, bh = bbox
                    
                    # Normalize to [x_center, y_center, width, height]
                    x_center = (bx + bw / 2) / w
                    y_center = (by + bh / 2) / h
                    width = bw / w
                    height = bh / h
                    
                    lf.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed_count += 1
            
        cap.release()
        
    print(f"✅ Data processing complete! Processed {processed_count} frames.")
    print(f"📁 Output saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()
