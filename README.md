# Traffic Sign Classification System

A deep learning-based traffic sign recognition system using CNN (Convolutional Neural Network). This project classifies 4 types of traffic signs: Accident-Prone Area, Bus Stop, Height Restriction, and Slow Down.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Dataset](#dataset)
- [License](#license)

## ✨ Features

- **CNN Model Training**: Train a custom CNN model for traffic sign classification
- **Data Preprocessing**: Automated image preprocessing and validation
- **Dataset Splitting**: Automatic train/test split with stratification
- **Web Interface**: Interactive Gradio-based web interface for real-time prediction
- **High Accuracy**: Achieves 94%+ accuracy on test set
- **Portable**: Uses relative paths for easy deployment

## 📁 Project Structure

```
Traffic_Sign_Classification_System/
├── code/                          # Source code directory
│   ├── train.py                   # Model training script
│   ├── gra.py                     # Gradio web interface
│   ├── check.py                   # Dataset validation
│   ├── split.py                   # Train/test splitting
│   ├── normalize.py               # Data normalization
│   └── pre_process.py             # Image preprocessing
├── dataset_split/                 # Split dataset (train/test)
│   ├── train/                     # Training images
│   └── test/                      # Testing images
├── data_ok/                       # Validated dataset
├── dataset_ok/                    # Processed images (64x64)
├── cnn_traffic_sign.keras         # Trained model file
└── README.md                      # This file
```

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- scikit-learn
- Gradio
- Matplotlib

### Install Dependencies

```bash
pip install tensorflow opencv-python pillow numpy scikit-learn gradio matplotlib
```

## 🚀 Installation

1. **Clone or download this project**
   ```bash
   cd Traffic_Sign_Classification_System
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify directory structure**
   Ensure you have the `code/` directory with all Python scripts

## 💻 Usage

### 1. Data Preprocessing

Process raw images to standard 64x64 size:

```bash
cd code
python pre_process.py
```

### 2. Dataset Validation

Validate image formats and sizes:

```bash
python check.py
```

### 3. Split Dataset

Split data into train (80%) and test (20%):

```bash
python split.py
```

### 4. Train Model

Train the CNN model:

```bash
python train.py
```

**Training Output:**
- Model will train for 10 epochs
- Progress displayed for each epoch
- Model saved as `../cnn_traffic_sign.keras`

### 5. Run Web Interface

Launch the Gradio web interface for predictions:

```bash
python gra.py
```

**Access the interface:**
- Local: `http://127.0.0.1:7860`
- Public link will be displayed if `share=True`

## 🧠 Model Information

### Architecture

- **Type**: Sequential CNN
- **Input Shape**: (64, 64, 3)
- **Layers**:
  - Conv2D (32 filters) + MaxPooling
  - Conv2D (64 filters) + MaxPooling
  - Conv2D (128 filters) + MaxPooling
  - Flatten
  - Dense (128 units) + Dropout (0.5)
  - Dense (4 units, softmax)

### Performance

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~94%
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

### Classes

The model classifies 4 traffic sign types:

| Class ID | Label    | Description         |
|----------|----------|---------------------|
| 0        | accident | Accident-Prone Area|
| 1        | bus      | Bus Stop           |
| 2        | high     | Height Restriction |
| 3        | slow     | Slow Down          |

## 📊 Dataset

### Image Requirements

- **Size**: 64x64 pixels
- **Format**: RGB or RGBA
- **File Types**: PNG, JPG, JPEG

### Dataset Statistics

- **Total Images**: ~385 images
- **Training Set**: 80% (~308 images)
- **Test Set**: 20% (~77 images)
- **Classes**: 4 (balanced distribution)

## 🛠️ Scripts Description

| Script | Purpose |
|--------|---------|
| `train.py` | Trains the CNN model and saves it |
| `gra.py` | Launches Gradio web interface for predictions |
| `check.py` | Validates dataset images |
| `split.py` | Splits dataset into train/test sets |
| `normalize.py` | Normalizes and preprocesses data |
| `pre_process.py` | Resizes and filters images |

## 📝 Notes

- All paths use relative references for portability
- Model file format: `.keras` (recommended by TensorFlow 2.x)
- Run all scripts from the `code/` directory
- Ensure dataset directories exist before running

## 🐛 Troubleshooting

### Model Not Found Error

If you get "Model not found" error:
- Ensure `cnn_traffic_sign.keras` exists in project root
- Run `train.py` to generate the model

### Path Errors

If you encounter path errors:
- Ensure you're running scripts from the `code/` directory
- Check that data directories exist at parent level

### Import Errors

If packages are missing:
```bash
pip install --upgrade tensorflow opencv-python pillow numpy scikit-learn gradio matplotlib
```

## 📄 License

This project is for educational purposes.

## 👤 Author

Doan Sinh Duc

---

**Vietnamese Version**: See [README_VI.md](README_VI.md) for Vietnamese documentation.



