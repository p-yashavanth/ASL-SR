import os
import shutil
import cv2
import numpy as np

SRC_DIR = r"C:\Users\P Yashvanth\Downloads\archive (1)\own_dataset"
DST_DIR = r"d:\ITZ_ME\projects\asl_app\static\asl_ref"

def main():
    print("Preparing ASL reference guide images...")
    
    # Ensure destination directory exists
    os.makedirs(DST_DIR, exist_ok=True)
    
    if not os.path.exists(SRC_DIR):
        print(f"Error: Source directory {SRC_DIR} does not exist!")
        return

    # 1. Copy one image per letter/space from dataset
    folders = sorted(os.listdir(SRC_DIR))
    for folder in folders:
        folder_path = os.path.join(SRC_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Get all jpg files
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"Warning: No images found in {folder}")
            continue
            
        # Copy the first image
        src_img = os.path.join(folder_path, files[0])
        dst_name = f"{folder.lower()}.jpg"
        dst_img = os.path.join(DST_DIR, dst_name)
        
        try:
            shutil.copy(src_img, dst_img)
            print(f"  Copied {folder} -> {dst_name}")
        except Exception as e:
            print(f"  Error copying {folder}: {e}")

    # 2. Generate a custom image for 'del' (backspace) to match the visual style
    del_img_path = os.path.join(DST_DIR, "del.jpg")
    try:
        # Create a 300x300 background (dark gray to contrast nicely or match light theme)
        # We will make it look like the other images: 300x300, 12px pink border, and a center drawing
        img = np.ones((300, 300, 3), dtype=np.uint8) * 240 # Light gray background
        
        # Pink/magenta border (12px)
        # Color: B=180, G=50, R=255 (pinkish magenta)
        cv2.rectangle(img, (0, 0), (300, 300), (180, 50, 255), 12)
        
        # Draw a backspace symbol in the middle
        # A large "<- [X]" symbol or backspace arrow
        cv2.putText(img, "DELETE", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 3)
        cv2.putText(img, "<- [X]", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 3)
        
        cv2.imwrite(del_img_path, img)
        print("  Generated custom reference image for 'del' -> del.jpg")
    except Exception as e:
        print(f"  Error generating 'del' image: {e}")

    print("Reference image setup complete.")

if __name__ == '__main__':
    main()
