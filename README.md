# Real-Time Facial Expression Recognition for Automatic Emoji Matching

A deep learning application that detects facial expressions in real-time using your webcam and displays the corresponding emoji.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green)

## Features

- 🎥 Real-time face detection using Haar Cascades
- 🧠 CNN-based emotion classification (7 emotions)
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
pip install torch torchvision pandas opencv-python pillow pyperclip
```

### 4. Download the FER-2013 Dataset

Download the dataset from [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` in the `data/` folder:

```
data/
└── fer2013.csv
```

## Usage

### Train the model

```bash
python train.py
```

This will train the CNN model and save it as `emotion_model.pth`.

### Run the application

```bash
python app.py
```

**Controls:**
- Press `q` to quit
- Press `s` to copy the current emoji to clipboard

## Model Architecture

The CNN architecture consists of:
- 3 Convolutional blocks with BatchNorm and MaxPooling
- Dropout (0.5) for regularization
- 2 Fully connected layers
- Output: 7 emotion classes

```
Input (1, 48, 48)
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool
    ↓
Flatten → Dropout(0.5)
    ↓
FC(512) → ReLU
    ↓
FC(7) → Output
```

## Project Structure

```
├── app.py          # Real-time webcam application
├── train.py        # Model training script
├── model.py        # CNN architecture definition
├── dataset.py      # FER-2013 dataset loader
├── data/
│   └── fer2013.csv # Dataset (not included, download from Kaggle)
└── emotion_model.pth # Trained model (generated after training)
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- OpenCV 4.0+
- Pillow
- pandas
- pyperclip

## License

MIT License

## Acknowledgments

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- PyTorch team for the deep learning framework
