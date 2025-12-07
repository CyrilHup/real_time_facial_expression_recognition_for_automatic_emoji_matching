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

Training now uses a **multi-source, unified 8-class dataset**:
- **Balanced AffectNet (RGB, 75×75)** — main source (~41k images, 8 classes)
- **FER+ (48×48 → upscaled to 75×75)** — FER2013 images with Microsoft-voted labels (adds **Contempt**)
- (Optional) **FER2013** — legacy labels (7 classes); avoid mixing with FER+ at the same time because they share images.

The notebook auto-downloads datasets with `kagglehub` and builds a combined loader that maps every source to the same 8 emotions: Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt.

Manual placement (if you download yourself):
```
data/
├── affectnet/
│   ├── train|val|test/Anger|Disgust|Fear|Happy|Sad|Surprise|Neutral|Contempt/
├── ferplus_generated/           # produced from FER2013 CSV + fer2013new.csv
│   ├── FER2013Train|FER2013Valid|FER2013Test/
│   └── fer2013new.csv
└── fer2013/ (optional legacy, 7 classes)
```

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
pip install torch torchvision pandas opencv-python pillow pyperclip albumentations mediapipe
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

**Recommended (multi-dataset, optimized):** run the notebook `train_affectnet_notebook (2).ipynb` which:
- Downloads AffectNet + FER+ via `kagglehub`
- Merges them with the unified `CombinedEmotionDataset`
- Uses AMP + `torch.compile` for fast large-batch training
- Saves the best weights to `emotion_model_best.pth` and a deployable `emotion_model.pth`

**Legacy script (single-dataset):**
```bash
python train_affectnet.py
```
Trains on AffectNet only and produces `emotion_model.pth`.

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

- **Multi-dataset fusion**: AffectNet + FER+ (unified 8-class mapping); optional FER2013 fallback
- **SE-Block CNN**: Attention-enhanced conv blocks with global avg pooling
- **Advanced augmentation**: Albumentations (flip, affine, noise/blur, color jitter, coarse dropout) + balanced intensity
- **Mixup (on)**, CutMix (off by default), **Label Smoothing**; optional Focal Loss
- **Class balancing**: adaptive class weights; oversized batches with gradient clipping
- **Optimizers**: AdamW + OneCycleLR; **AMP** + **torch.compile (max-autotune)** for speed
- **Regularization**: dropout, weight decay, early stopping, optional SWA
- **Evaluation**: per-class metrics and optional TTA (flip + brightness variants)

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
