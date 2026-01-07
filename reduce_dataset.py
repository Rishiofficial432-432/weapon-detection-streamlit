import os
import random
import shutil
from pathlib import Path

def reduce_files(img_dir, lbl_dir, keep_count):
    images = list(Path(img_dir).glob("*.jpg"))
    if len(images) <= keep_count:
        print(f"Directory {img_dir} has {len(images)} images, keeping all.")
        return

    print(f"Reducing {img_dir} from {len(images)} to {keep_count}...")
    
    # Shuffle and pick files to DELETE
    random.shuffle(images)
    to_delete = images[keep_count:]
    
    for img_path in to_delete:
        # Delete image
        img_path.unlink()
        
        # Delete corresponding label
        lbl_path = Path(lbl_dir) / f"{img_path.stem}.txt"
        if lbl_path.exists():
            lbl_path.unlink()

def main():
    root = "datasets/custom_weapon"
    reduce_files(f"{root}/images/train", f"{root}/labels/train", 400)
    reduce_files(f"{root}/images/val", f"{root}/labels/val", 100)
    print("Dataset reduction complete.")

if __name__ == "__main__":
    main()
