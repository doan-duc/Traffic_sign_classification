import os
import numpy as np
import cv2  # Or use PIL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_DIR = r"../data_ok"
IMG_SIZE = (64, 64)

def prepare_data_for_cnn(dataset_path):
    print("🔄 Loading images and labels from directory...")
    
    data = []
    labels = []
    
    # Get list of classes (subfolders)
    # Sort alphabetically to ensure consistent label order: accident -> 0, bus -> 1...
    classes = sorted(os.listdir(dataset_path))
    print(f"🏷️ Labels found: {classes}")

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # Read image
                # Note: OpenCV reads images in BGR, need to convert to RGB
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Even though images in dataset_ok are already resized, resize again to be sure
                    img = cv2.resize(img, IMG_SIZE)
                    
                    data.append(img)
                    labels.append(class_name)
            except Exception as e:
                print(f"⚠️ Error reading file {img_name}: {e}")

    # 1. Convert to Numpy array
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    print(f"📊 Total images loaded: {data.shape[0]}")
    
    # 2. Normalization: Scale pixels from 0-255 to 0-1
    # This helps CNN converge much faster
    data = data / 255.0
    print("✅ Normalized data to range [0, 1]")

    # 3. Label Encoding + One-hot
    le = LabelEncoder()
    labels_int = le.fit_transform(labels) # Convert text to numbers: 'bus' -> 1
    
    # Convert numbers to one-hot vectors: 1 -> [0, 1, 0, 0]
    # This step is necessary if using loss='categorical_crossentropy'
    labels_one_hot = to_categorical(labels_int, num_classes=len(classes))
    print("✅ Encoded labels to one-hot format")

    # 4. Split Train/Test (80% Train, 20% Test)
    # stratify=labels helps ensure the same class ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_one_hot, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\n🏁 Preprocessing complete!")
    print(f"   - Train shape: {X_train.shape}")
    print(f"   - Test shape : {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, classes

# ==============================================================================
# RUN FUNCTION
# ==============================================================================
if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        X_train, X_test, y_train, y_test, class_names = prepare_data_for_cnn(DATASET_DIR)
        
        # Save for use in other training files if needed (Optional)
        # np.save('X_train.npy', X_train)
        # np.save('y_train.npy', y_train)
        # np.save('X_test.npy', X_test)
        # np.save('y_test.npy', y_test)
    else:
        print(f"❌ Directory not found: {DATASET_DIR}")