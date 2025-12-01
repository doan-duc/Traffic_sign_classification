import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

# ==============================================================================
# 1. LABEL AND MODEL CONFIGURATION
# ==============================================================================
# Trained model file path (using relative path)
MODEL_PATH = '../cnn_traffic_sign.keras'

# Image size (Must match exactly with training: 64x64)
IMG_SIZE = (64, 64)

# LABEL LIST (Sorted A -> Z based on folder names)
# This order is extremely important, wrong order = wrong prediction names
# Note: Keys are folder names, values are full Vietnamese names
CLASS_NAMES = [
    'Đoạn đường hay xảy ra tai nạn (accident_ok)',  # 0
    'Điểm dừng xe buýt (bus_ok)',                    # 1
    'Hạn chế chiều cao (high_ok)',                   # 2
    'Đi chậm (slow_ok)'                              # 3
]

print("⏳ Loading model...")
# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: File '{MODEL_PATH}' not found.")
    print("👉 Please copy the model file to the same directory as this code.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==============================================================================
# 2. IMAGE PROCESSING & PREDICTION FUNCTION
# ==============================================================================
def predict_traffic_sign(image):
    if image is None:
        return None
    
    # --- Step 1: Preprocessing ---
    # Gradio returns RGB image, resize to match model input
    img_processed = cv2.resize(image, IMG_SIZE)
    
    # Normalize to [0, 1] exactly like during training
    img_processed = img_processed.astype("float32") / 255.0
    
    # Add batch dimension: (64, 64, 3) -> (1, 64, 64, 3)
    img_processed = np.expand_dims(img_processed, axis=0)
    
    # --- Step 2: Prediction (Inference) ---
    predictions = model.predict(img_processed)
    
    # predictions is a 2D array [[0.1, 0.8, 0.05, 0.05]]
    scores = predictions[0]
    
    # --- Step 3: Return results for Gradio ---
    # Gradio needs dictionary format: {'Label': probability, ...}
    results = {}
    for i, class_name in enumerate(CLASS_NAMES):
        results[class_name] = float(scores[i])
        
    return results

# ==============================================================================
# 3. RUN INTERFACE
# ==============================================================================
interface = gr.Interface(
    fn=predict_traffic_sign, 
    inputs=gr.Image(label="Upload traffic sign image here"),
    outputs=gr.Label(num_top_classes=4, label="Prediction results"),
    title="TRAFFIC SIGN RECOGNITION DEMO",
    description="Classification system for 4 types of traffic signs: Accident-Prone Area, Bus Stop, Height Restriction, Slow Down.",
    examples=[] # You can add sample image paths here if desired
)

if __name__ == "__main__":
    # share=True to create a public link to share with friends
    interface.launch(share=True)