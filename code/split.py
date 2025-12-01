import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_DIR = r"../data_ok"
OUTPUT_ROOT = r"../dataset_split"

TEST_SIZE = 0.2  # 20% for test
SEED = 42        # To fix split results

def split_dataset_into_folders():
    # 1. Check input
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Directory not found: {INPUT_DIR}")
        return

    # 2. Create train/test directory structure
    for split in ['train', 'test']:
        split_path = os.path.join(OUTPUT_ROOT, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            print(f"📁 Created directory: {split_path}")

    # Get list of classes (labels)
    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    print(f"🚀 Starting to split {len(classes)} data classes...")

    for class_name in classes:
        class_in_path = os.path.join(INPUT_DIR, class_name)
        
        # Get list of image files in this class
        images = [f for f in os.listdir(class_in_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split file list
        train_files, test_files = train_test_split(images, test_size=TEST_SIZE, random_state=SEED)

        # Helper function to copy files
        def copy_files(files, split_type):
            dest_dir = os.path.join(OUTPUT_ROOT, split_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for file_name in files:
                src = os.path.join(class_in_path, file_name)
                dst = os.path.join(dest_dir, file_name)
                shutil.copy2(src, dst) # Copy file
                
        # Execute copy
        copy_files(train_files, 'train')
        copy_files(test_files, 'test')

        print(f"   ✅ Class '{class_name}': Train={len(train_files)}, Test={len(test_files)}")

    print("-" * 50)
    print(f"🏁 Complete! Data has been split into: {OUTPUT_ROOT}")

if __name__ == "__main__":
    split_dataset_into_folders()