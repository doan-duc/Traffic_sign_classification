import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# Relative paths
TRAIN_DIR = r"../dataset_split/train"
TEST_DIR  = r"../dataset_split/test"

IMG_SIZE = (64, 64)
EPOCHS = 10

# ==============================================================================
# 2. DATA LOADING FUNCTION
# ==============================================================================
def load_data_from_dir(data_dir, target_size):
    print(f"📂 Loading data from: {data_dir}")
    
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory not found: {data_dir}")
        return np.array([]), np.array([])

    class_names = sorted(os.listdir(data_dir))
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        files = os.listdir(class_path)
        
        for img_name in files:
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(class_name)
            except Exception:
                pass

    X = np.array(images, dtype="float32") / 255.0 
    y = np.array(labels)
    
    print(f"✅ Loaded: {X.shape[0]} images.")
    return X, y

# ==============================================================================
# 3. CNN MODEL
# ==============================================================================
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Use Input layer first to avoid warnings
    model.add(Input(shape=input_shape))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==============================================================================
# 4. CUSTOM TRAINING FUNCTION
# ==============================================================================
def train_cnn_custom(model, X_train, y_train, X_test, y_test, epochs=10):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print(f"\n🚀 Starting training ({epochs} epochs)...")

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        
        train_losses.append(history.history['loss'][0])
        train_accuracies.append(history.history['accuracy'][0])
        test_losses.append(history.history['val_loss'][0])
        test_accuracies.append(history.history['val_accuracy'][0])
        
        print(f"Epoch {epoch + 1:02d} | "
              f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]:.4f} | "
              f"Test Loss: {test_losses[-1]:.4f}, Acc: {test_accuracies[-1]:.4f}")

    # Plot graphs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(range(1, epochs+1), train_losses, label='Train Loss')
    ax1.plot(range(1, epochs+1), test_losses, label='Test Loss', linestyle='--')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(range(1, epochs+1), train_accuracies, label='Train Acc')
    ax2.plot(range(1, epochs+1), test_accuracies, label='Test Acc', linestyle='--')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    return model

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("--- 1. LOAD TRAIN DATA ---")
    X_train, y_train_labels = load_data_from_dir(TRAIN_DIR, IMG_SIZE)
    
    print("\n--- 2. LOAD TEST DATA ---")
    X_test, y_test_labels = load_data_from_dir(TEST_DIR, IMG_SIZE)

    if len(X_train) > 0 and len(X_test) > 0:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_labels)
        y_test = le.transform(y_test_labels)
        
        print(f"\nLabel mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        num_classes = len(le.classes_)
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
        
        cnn_model = create_cnn_model(input_shape, num_classes)
        trained_model = train_cnn_custom(cnn_model, X_train, y_train, X_test, y_test, epochs=EPOCHS)
        
        trained_model.save('../cnn_traffic_sign.keras')
        print("\n💾 Model saved: cnn_traffic_sign.keras")
    else:
        print("\n⚠️ Training or Test data is empty.")