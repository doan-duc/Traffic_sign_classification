import os
from PIL import Image

# ==============================================================================
# PATH CONFIGURATION (Relative)
# ==============================================================================
INPUT_DIR = r"../dataset"
OUTPUT_DIR = r"../dataset_ok"

# Configuration
TARGET_SIZE = (64, 64)
MIN_SIZE_THRESHOLD = 30 

# ==============================================================================
# PROCESSING FUNCTION
# ==============================================================================
def process_dataset():
    # Check input
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Source directory not found at: {INPUT_DIR}")
        return

    # Create output directory (Create new only, don't delete old)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✅ Created new output directory: {OUTPUT_DIR}")
    else:
        print(f"ℹ️ Output directory already exists: {OUTPUT_DIR} (Will append images here)")

    print("-" * 60)

    # Get list of classes
    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    total_success = 0
    total_skipped = 0
    
    # Iterate through each class
    for class_name in classes:
        print(f"\n📂 Processing class: '{class_name}'...")
        
        class_in_path = os.path.join(INPUT_DIR, class_name)
        class_out_path = os.path.join(OUTPUT_DIR, class_name)
        
        # Create subfolder at destination
        os.makedirs(class_out_path, exist_ok=True)

        files = os.listdir(class_in_path)
        class_count = 0

        for file_name in files:
            file_path = os.path.join(class_in_path, file_name)
            save_path = os.path.join(class_out_path, file_name)

            # Skip if it's a directory
            if os.path.isdir(file_path):
                continue

            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    
                    # 1. Check size (Print if skipped)
                    if w < MIN_SIZE_THRESHOLD or h < MIN_SIZE_THRESHOLD:
                        print(f"   ⚠️ Skipped: {file_name:<20} | Reason: Too small ({w}x{h})")
                        total_skipped += 1
                        continue 

                    # 2. Process image
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # 3. Save image and print success message
                    img_resized.save(save_path)
                    class_count += 1
                    total_success += 1
                    
                    # Print details of each completed file
                    print(f"   ✅ Done: {file_name:<20} | Original: {w}x{h} -> New: 64x64")

            except Exception as e:
                print(f"   ❌ File error: {file_name} - {e}")

        print(f"   ---> Class '{class_name}' complete: {class_count} new images created.")

    # Final summary
    print("\n" + "="*60)
    print("🏁 COMPLETION REPORT")
    print(f"📍 Source directory (Kept intact): {INPUT_DIR}")
    print(f"📍 New directory (Created)       : {OUTPUT_DIR}")
    print("-" * 60)
    print(f"✅ Total images resized to standard: {total_success}")
    print(f"🗑️ Total images skipped (too small): {total_skipped}")
    print("="*60)

if __name__ == "__main__":
    process_dataset()