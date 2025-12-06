# Real-Time Facial Expression Recognition for Automatic Emoji Matching

A deep learning application that detects facial expressions in real-time using your webcam and displays the corresponding emoji.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green)

## Features

- 🎥 Real-time face detection using Haar Cascades
- 🧠 CNN-based emotion classification (8 emotions)
- 😃 Automatic emoji overlay on detected faces
- 📋 Copy emoji to clipboard with 's' key

## Supported Emotions

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😃 |
| Sad | 😢 |
| Surprise | 😲 |
| Neutral | 😐 |
| Contempt | 😏 |

## Dataset

This project uses the **Balanced AffectNet** dataset:
- Source: [Kaggle - Balanced AffectNet](https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet)
- 41,008 images total
- 8 emotion classes (~5,126 images per class)
- RGB images at 75×75 pixels
- Pre-balanced for better training

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CyrilHup/real_time_facial_expression_recognition_for_automatic_emoji_matching.git
cd real_time_facial_expression_recognition_for_automatic_emoji_matching
```

### 2. Create a conda environment

```bash
conda create -n fer_project python=3.11 -y
conda activate fer_project
```

### 3. Install dependencies

```bash
pip install torch torchvision pandas opencv-python pillow pyperclip albumentations mediapipe kaggle
```

### 4. Download the Balanced AffectNet Dataset

Download the dataset from [Kaggle Balanced AffectNet](https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet) and extract the folders directly into `data/`:

```
data/
├── train/
│   ├── Anger/
│   ├── Contempt/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

## Usage

### Train the model

```bash
python train_affectnet.py
```

This will train the CNN model on the Balanced AffectNet dataset and save it as `emotion_model.pth`.

### Run the application

You can run different versions of the application depending on your needs:

**Basic Version (Fast & Simple):**
```bash
python app.py
```

**Advanced Version (Recommended):**
Includes hand gesture recognition, facial feature analysis, and improved accuracy.
```bash
python app_v3.py
```

**Controls:**
- Press `q` to quit
- Press `s` to copy the current emoji to clipboard
- Press `e` to toggle emotion probability bars (v3)
- Press `f` to toggle facial feature analysis (v3)
- Press `h` to toggle hand tracking (v3)

## Model Architecture

The CNN architecture consists of:
- 4 Convolutional blocks with BatchNorm, MaxPooling, and progressive Dropout
- Global Average Pooling for flexibility
- 3 Fully connected layers
- Output: 8 emotion classes

```
Input (3, 75, 75) - RGB Image
    ↓
Conv2D(64) × 2 → BatchNorm → ReLU → MaxPool → Dropout(0.1)
    ↓
Conv2D(128) × 2 → BatchNorm → ReLU → MaxPool → Dropout(0.1)
    ↓
Conv2D(256) × 2 → BatchNorm → ReLU → MaxPool → Dropout(0.15)
    ↓
Conv2D(512) × 2 → BatchNorm → ReLU → MaxPool → Dropout(0.2)
    ↓
Global Average Pooling
    ↓
FC(512→256) → BatchNorm → ReLU → Dropout(0.4)
    ↓
FC(256→128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
FC(128→8) → Output
```

## Project Structure

```
├── app.py                  # Basic real-time webcam application
├── app_v3.py               # Advanced app with hand gestures & feature analysis
├── train_affectnet.py      # Training script for AffectNet
├── model.py                # CNN architecture definition
├── dataset_affectnet.py    # Balanced AffectNet dataset loader
├── data/                   # Dataset (not included, download from Kaggle)
│   ├── train/
│   ├── val/
│   └── test/
├── emotion_model.pth       # Trained model (generated after training)
├── report/
│   └── report.tex          # Technical report (LaTeX)
└── README.md
```

## Training Features

- **Mixup Augmentation**: Blend samples for better generalization
- **Label Smoothing**: Prevent overconfidence
- **Advanced Augmentation**: Using Albumentations library
- **OneCycleLR Scheduler**: Optimal learning rate scheduling
- **Early Stopping**: Prevent overfitting

## Requirements

- Python 3.11+
- PyTorch 2.0+
- OpenCV 4.0+
- Pillow
- albumentations
- mediapipe
- pyperclip

## License

MIT License

## Acknowledgments

- [Balanced AffectNet Dataset](https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet)
- PyTorch team for the deep learning framework
- Albumentations team for the augmentation library
