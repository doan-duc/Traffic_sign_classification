import os
import shutil
from PIL import Image
from pathlib import Path

# Standard size you want
TARGET_SIZE = (64, 64)
# Path to your source data directory
INPUT_PATH = r"../dataset_ok"

# Path where you want to save classified data
OUTPUT_PATH = r"../data_ok"

def filter_and_organize_dataset(input_path, output_path, target_size=(64, 64)):
    # Check input path
    if not os.path.exists(input_path):
        print(f"❌ Error: Input path not found: {input_path}")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"📁 Created output directory: {output_path}")

    # Get list of classes (subfolders)
    class_names = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    
    total_ok = 0
    total_error = 0

    print(f"🚀 Starting to process {len(class_names)} data classes...")
    print("-" * 50)

    for class_name in class_names:
        class_input_dir = os.path.join(input_path, class_name)
        
        # Create paths for _ok and _error folders
        class_ok_dir = os.path.join(output_path, f"{class_name}_ok")
        class_error_dir = os.path.join(output_path, f"{class_name}_error")

        # Create these directories
        os.makedirs(class_ok_dir, exist_ok=True)
        os.makedirs(class_error_dir, exist_ok=True)

        count_ok = 0
        count_err = 0

        # Iterate through files in the class directory
        for file_name in os.listdir(class_input_dir):
            file_path = os.path.join(class_input_dir, file_name)
            
            # Skip if it's a subdirectory
            if os.path.isdir(file_path):
                continue

            is_valid = False
            
            try:
                # Open image to check
                with Image.open(file_path) as img:
                    # Check size and color mode
                    if img.size == target_size and img.mode in ["RGB", "RGBA"]:
                        is_valid = True
                    else:
                        # Uncomment the line below for detailed error info
                        # print(f"  [Invalid] {file_name}: {img.size}, Mode: {img.mode}")
                        is_valid = False
            except Exception as e:
                print(f"  ⚠️ Unreadable file error: {file_name} - {e}")
                is_valid = False

            # Copy file to the corresponding directory
            try:
                if is_valid:
                    shutil.copy2(file_path, os.path.join(class_ok_dir, file_name))
                    count_ok += 1
                else:
                    shutil.copy2(file_path, os.path.join(class_error_dir, file_name))
                    count_err += 1
            except Exception as e:
                print(f"  ❌ Error copying file {file_name}: {e}")

        print(f"✅ Processed class '{class_name}': OK={count_ok}, Error={count_err}")
        
        total_ok += count_ok
        total_error += count_err

    print("-" * 50)
    print("🏁 FILTERING PROCESS COMPLETED")
    print(f"📊 Total valid images (OK): {total_ok}")
    print(f"📊 Total invalid/wrong size images (Error): {total_error}")
    print(f"📂 New data saved at: {output_path}")


# Run function
if __name__ == "__main__":
    # If running on real machine or Colab with data, just call the line below.
    if not os.path.exists(INPUT_PATH):
        print(f"Note: Path {INPUT_PATH} does not exist. Please check mount drive or path.")
    else:
        filter_and_organize_dataset(INPUT_PATH, OUTPUT_PATH, TARGET_SIZE)