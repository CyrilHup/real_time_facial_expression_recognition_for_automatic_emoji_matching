"""
Emoji Recognition V4 - Multi-Dataset Support & Enhanced Features
================================================================
Key improvements over V3:
- Auto-detection of dataset type (FER2013, FER+, AffectNet) from model weights
- Multi-model support with automatic configuration
- Enhanced preprocessing pipeline per dataset
- Improved emotion mapping with confidence calibration
- Better UI with dataset info display
- Performance optimizations
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pyperclip
from collections import deque, Counter
from model import FaceEmotionCNN, FaceEmotionCNN_SE, create_model, load_model_smart
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import os
import time

# Try to import ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
import time
from dataclasses import dataclass
from enum import Enum


# ============================================
# DATASET CONFIGURATIONS
# ============================================
class DatasetType(Enum):
    FER2013 = "fer2013"
    FERPLUS = "ferplus"
    AFFECTNET = "affectnet"
    UNKNOWN = "unknown"


@dataclass
class DatasetConfig:
    """Configuration for each dataset type"""
    name: str
    num_classes: int
    in_channels: int
    img_size: int
    emotions: Dict[int, str]
    use_normalization: bool
    mean: List[float]
    std: List[float]


# Dataset configurations
DATASET_CONFIGS = {
    DatasetType.FER2013: DatasetConfig(
        name="FER2013",
        num_classes=7,
        in_channels=1,
        img_size=48,
        emotions={
            0: "angry", 1: "disgust", 2: "fear", 3: "happy",
            4: "sad", 5: "surprise", 6: "neutral"
        },
        use_normalization=False,
        mean=[0.5],
        std=[0.5]
    ),
    DatasetType.FERPLUS: DatasetConfig(
        name="FER+ (Enhanced Labels)",
        num_classes=8,
        in_channels=1,
        img_size=48,
        emotions={
            0: "angry", 1: "disgust", 2: "fear", 3: "happy",
            4: "sad", 5: "surprise", 6: "neutral", 7: "contempt"
        },
        use_normalization=False,
        mean=[0.5],
        std=[0.5]
    ),
    DatasetType.AFFECTNET: DatasetConfig(
        name="AffectNet (RGB)",
        num_classes=8,
        in_channels=3,
        img_size=75,
        emotions={
            0: "angry", 1: "disgust", 2: "fear", 3: "happy",
            4: "sad", 5: "surprise", 6: "neutral", 7: "contempt"
        },
        use_normalization=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
}


# Global device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Emojis for emotions
BASE_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😃",
    "sad": "😢", "surprise": "😲", "neutral": "😐", "contempt": "😏"
}

EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (0, 165, 255),
    "neutral": (128, 128, 128), "contempt": (139, 69, 19)
}

# Hand gestures
HAND_GESTURES = {
    "thumbs_up": ("👍", "Thumbs Up"), "thumbs_down": ("👎", "Thumbs Down"),
    "peace": ("✌️", "Peace"), "ok": ("👌", "OK"), "rock": ("🤘", "Rock"),
    "call_me": ("🤙", "Call Me"), "fist": ("✊", "Fist"),
    "open_hand": ("🖐️", "High Five"), "point_up": ("☝️", "Point Up"),
    "pray": ("🙏", "Pray"), "love_you": ("🤟", "Love You"),
}


# ============================================
# MODEL DETECTOR
# ============================================
class ModelDetector:
    """Detects dataset type from model weights"""
    
    @staticmethod
    def detect_from_checkpoint(checkpoint: dict) -> Tuple[DatasetType, int, int]:
        """
        Analyze checkpoint to determine dataset type.
        Returns: (dataset_type, num_classes, in_channels)
        """
        # Get state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Check first conv layer shape
        conv1_key = None
        for key in state_dict.keys():
            if 'conv1' in key and 'weight' in key:
                conv1_key = key
                break
        
        if conv1_key is None:
            print("Warning: Could not find conv1 layer, defaulting to FER+")
            return DatasetType.FERPLUS, 8, 1
        
        conv1_shape = state_dict[conv1_key].shape
        in_channels = conv1_shape[1]
        
        # Check output layer for num_classes
        fc_key = None
        for key in state_dict.keys():
            if ('fc3' in key or 'fc2' in key or 'classifier' in key) and 'weight' in key:
                fc_key = key
        
        num_classes = 8  # Default
        if fc_key:
            num_classes = state_dict[fc_key].shape[0]
        
        # Determine dataset type
        if in_channels == 3:
            dataset_type = DatasetType.AFFECTNET
        elif num_classes == 7:
            dataset_type = DatasetType.FER2013
        else:
            dataset_type = DatasetType.FERPLUS
        
        return dataset_type, num_classes, in_channels
    
    @staticmethod
    def get_config(dataset_type: DatasetType) -> DatasetConfig:
        """Get configuration for dataset type"""
        return DATASET_CONFIGS.get(dataset_type, DATASET_CONFIGS[DatasetType.FERPLUS])


# ============================================
# EMOTION CLASSIFIER WITH AUTO-DETECTION
# ============================================
class EmotionClassifier:
    """CNN-based emotion classifier with automatic dataset detection"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model_path = model_path
        self.use_onnx = False
        self.onnx_session = None
        
        # Check if ONNX model exists and ONNX Runtime is available
        onnx_path = model_path.replace('.pth', '.onnx').replace('.pt', '.onnx')
        if ONNX_AVAILABLE and os.path.exists(onnx_path) and onnx_path != model_path:
            print(f"Loading ONNX model from: {onnx_path}")
            try:
                # Create ONNX Runtime session
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                self.use_onnx = True
                
                # Get model info from PyTorch checkpoint to configure dataset
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Detect num_classes and in_channels from ONNX metadata
                input_meta = self.onnx_session.get_inputs()[0]
                output_meta = self.onnx_session.get_outputs()[0]
                
                self.in_channels = input_meta.shape[1]  # [batch, channels, h, w]
                self.num_classes = output_meta.shape[1]  # [batch, num_classes]
                
                print(f"  ✓ ONNX Runtime loaded (optimized inference)")
                print(f"  Provider: {self.onnx_session.get_providers()[0]}")
                self.model = None  # Don't load PyTorch model
            except Exception as e:
                print(f"  ⚠ ONNX loading failed, falling back to PyTorch: {e}")
                self.use_onnx = False
                self.onnx_session = None
        
        # Load PyTorch model if ONNX not available
        if not self.use_onnx:
            print(f"Loading model from: {model_path}")
            self.model, model_info = load_model_smart(model_path, device)
            
            # Extract configuration from model info
            self.num_classes = model_info['num_classes']
            self.in_channels = model_info['in_channels']
        
        # Detect dataset type based on model info
        if self.in_channels == 1:
            if self.num_classes == 7:
                self.dataset_type = DatasetType.FER2013
            else:
                self.dataset_type = DatasetType.FERPLUS
        else:
            self.dataset_type = DatasetType.AFFECTNET
        
        self.config = ModelDetector.get_config(self.dataset_type)
        
        if not self.use_onnx:
            print(f"  Architecture: {model_info['architecture']}")
        print(f"  Detected dataset: {self.config.name}")
        print(f"  Classes: {self.num_classes}, Channels: {self.in_channels}")
        print(f"  Image size: {self.config.img_size}x{self.config.img_size}")
        
        # Build transforms based on dataset
        self._build_transforms()
        
        # Emotion-specific adjustments
        self.emotion_adjustments = self._get_emotion_adjustments()
        
        # History for temporal smoothing
        self.history = deque(maxlen=7)
        
        # Performance tracking
        self.inference_times = deque(maxlen=30)
    
    def _build_transforms(self):
        """Build preprocessing transforms based on dataset"""
        img_size = self.config.img_size
        
        if self.config.use_normalization:
            # RGB with ImageNet normalization (AffectNet)
            self.base_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std),
            ])
            self.tta_transforms = [
                self.base_transform,
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.config.mean, std=self.config.std),
                ]),
            ]
        else:
            # Grayscale without normalization (FER/FER+)
            self.base_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            self.tta_transforms = [
                self.base_transform,
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                ]),
            ]
    
    def _get_emotion_adjustments(self) -> Dict[int, float]:
        """Get emotion-specific probability adjustments"""
        if self.dataset_type == DatasetType.FER2013:
            return {
                0: 1.2, 1: 1.5, 2: 1.3, 3: 0.95,
                4: 1.0, 5: 1.0, 6: 0.85
            }
        else:  # FER+ and AffectNet
            return {
                0: 1.2,   # angry
                1: 1.6,   # disgust - underrepresented
                2: 1.3,   # fear
                3: 0.95,  # happy - well represented
                4: 1.0,   # sad
                5: 1.0,   # surprise
                6: 0.8,   # neutral - often over-predicted
                7: 1.4,   # contempt - hard to detect
            }
    
    def preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face ROI based on dataset type"""
        if self.in_channels == 1:
            # Convert to grayscale
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            face_roi = clahe.apply(face_roi)
        else:
            # Ensure RGB
            if len(face_roi.shape) == 2:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            elif face_roi.shape[2] == 4:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGBA2RGB)
            else:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            # Apply CLAHE per channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            channels = cv2.split(face_roi)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            face_roi = cv2.merge(clahe_channels)
        
        return face_roi
    
    def predict_onnx(self, face_roi: np.ndarray, use_tta: bool = True) -> Tuple[str, float, np.ndarray]:
        """
        ONNX-optimized prediction (2-3x faster)
        Returns: (emotion_name, confidence, all_probabilities)
        """
        face_roi = self.preprocess(face_roi)
        
        # Convert to PIL and resize
        img_size = self.config.img_size
        if self.in_channels == 1:
            face_pil = Image.fromarray(face_roi).convert('L')
        else:
            face_pil = Image.fromarray(face_roi)
        
        face_pil = face_pil.resize((img_size, img_size))
        
        # Convert to tensor and normalize
        face_tensor = transforms.ToTensor()(face_pil)
        if self.config.use_normalization:
            face_tensor = transforms.Normalize(mean=self.config.mean, std=self.config.std)(face_tensor)
        
        face_input = face_tensor.unsqueeze(0).numpy()  # ONNX needs numpy
        
        # Run inference
        if use_tta:
            # Test-Time Augmentation with ONNX
            predictions = []
            
            # Original
            ort_inputs = {self.onnx_session.get_inputs()[0].name: face_input}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            predictions.append(ort_outs[0][0])
            
            # Horizontal flip
            face_flip = np.flip(face_input, axis=3).copy()
            ort_inputs = {self.onnx_session.get_inputs()[0].name: face_flip}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            predictions.append(ort_outs[0][0])
            
            # Average predictions
            logits = np.mean(predictions, axis=0)
        else:
            # Single inference
            ort_inputs = {self.onnx_session.get_inputs()[0].name: face_input}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            logits = ort_outs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Apply emotion adjustments
        for idx, adjustment in self.emotion_adjustments.items():
            probs[idx] *= adjustment
        probs = probs / probs.sum()
        
        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        emotion = self.config.emotions.get(pred_idx, "unknown")
        
        return emotion, confidence, probs
    
    def predict(self, face_roi: np.ndarray, use_tta: bool = True) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion with optional TTA
        Returns: (emotion_name, confidence, all_probabilities)
        """
        start_time = time.perf_counter()
        
        # Use ONNX if available (faster)
        if self.use_onnx:
            emotion, confidence, probs = self.predict_onnx(face_roi, use_tta)
            
            # Track performance
            inference_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            return emotion, confidence, probs
        
        # PyTorch inference (fallback)
        face_roi = self.preprocess(face_roi)
        
        with torch.no_grad():
            if use_tta:
                all_probs = []
                for transform in self.tta_transforms:
                    tensor = transform(face_roi).unsqueeze(0).to(self.device)
                    output = self.model(tensor)
                    probs = F.softmax(output, dim=1)[0]
                    all_probs.append(probs)
                probs = torch.stack(all_probs).mean(dim=0)
            else:
                tensor = self.base_transform(face_roi).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)[0]
            
            # Apply adjustments
            adjusted_probs = probs.clone()
            for idx, factor in self.emotion_adjustments.items():
                if idx < len(adjusted_probs):
                    adjusted_probs[idx] *= factor
            
            # Renormalize
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            # Temporal smoothing
            self.history.append(adjusted_probs.cpu().numpy())
            
            if len(self.history) >= 3:
                weights = np.exp(np.linspace(-1, 0, len(self.history)))
                weights /= weights.sum()
                smoothed = np.zeros(self.num_classes)
                for i, h in enumerate(self.history):
                    smoothed += weights[i] * h
                final_probs = smoothed
            else:
                final_probs = adjusted_probs.cpu().numpy()
            
            # Get prediction
            pred_idx = np.argmax(final_probs)
            confidence = final_probs[pred_idx]
            emotion = self.config.emotions.get(pred_idx, "unknown")
            
            # Track performance
            inference_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            return emotion, confidence, final_probs
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in ms"""
        if self.inference_times:
            return sum(self.inference_times) / len(self.inference_times)
        return 0.0


# ============================================
# FACIAL FEATURE ANALYZER
# ============================================
class FacialAnalyzer:
    """Analyzes facial features using MediaPipe Face Mesh"""
    
    LANDMARKS = {
        "left_eye": [33, 133, 159, 145],
        "right_eye": [263, 362, 386, 374],
        "left_brow": [70, 107, 105],
        "right_brow": [300, 336, 334],
        "mouth": [61, 291, 13, 14, 0, 17],
        "nose": [4, 48, 278],
        "face": [10, 152],
    }
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=5)
        
    def analyze(self, frame_rgb: np.ndarray) -> Dict[str, float]:
        """Extract facial features from frame"""
        results = self.face_mesh.process(frame_rgb)
        
        features = {
            "mouth_open": 0.0,
            "smile": 0.0,
            "brow_raised": 0.0,
            "brow_furrowed": 0.0,
            "eyes_wide": 0.0,
            "eyes_closed": 0.0,
            "asymmetric_mouth": 0.0,
            "nose_wrinkle": 0.0,
            "brow_squeeze": 0.0,
            "wink_left": False,
            "wink_right": False,
        }
        
        if not results.multi_face_landmarks:
            return features
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_rgb.shape[:2]
        
        def point(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
        def dist(i1, i2):
            return np.linalg.norm(point(i1) - point(i2))
        
        face_h = dist(10, 152)
        if face_h < 10:
            return features
        
        # Mouth analysis
        mouth_h = dist(13, 14)
        mouth_w = dist(61, 291)
        features["mouth_open"] = np.clip(mouth_h / (face_h * 0.12), 0, 1)
        
        if mouth_h > 1:
            smile_ratio = mouth_w / mouth_h
            features["smile"] = np.clip((smile_ratio - 2.5) / 4.0, 0, 1)
        
        left_corner = point(61)
        right_corner = point(291)
        asymmetry = abs(left_corner[1] - right_corner[1]) / (face_h * 0.05)
        features["asymmetric_mouth"] = np.clip(asymmetry, 0, 1)
        
        # Eye analysis
        left_eye_h = dist(159, 145)
        right_eye_h = dist(386, 374)
        left_eye_w = dist(33, 133)
        right_eye_w = dist(263, 362)
        
        avg_eye_h = (left_eye_h + right_eye_h) / 2
        avg_eye_w = (left_eye_w + right_eye_w) / 2
        
        if avg_eye_w > 1:
            ear = avg_eye_h / avg_eye_w
            features["eyes_wide"] = np.clip((ear - 0.25) / 0.15, 0, 1)
            features["eyes_closed"] = np.clip((0.18 - ear) / 0.1, 0, 1)
            
            left_ear = left_eye_h / left_eye_w if left_eye_w > 1 else 0.3
            right_ear = right_eye_h / right_eye_w if right_eye_w > 1 else 0.3
            features["wink_left"] = left_ear < 0.15 and right_ear > 0.2
            features["wink_right"] = right_ear < 0.15 and left_ear > 0.2
        
        # Eyebrow analysis
        left_brow_h = dist(105, 159)
        right_brow_h = dist(334, 386)
        avg_brow_h = (left_brow_h + right_brow_h) / 2
        
        brow_norm = avg_brow_h / (face_h * 0.08)
        features["brow_raised"] = np.clip((brow_norm - 1.0) / 0.5, 0, 1)
        features["brow_furrowed"] = np.clip((1.0 - brow_norm) / 0.4, 0, 1)
        
        left_brow_inner = point(107)
        right_brow_inner = point(336)
        brow_dist = np.linalg.norm(left_brow_inner - right_brow_inner)
        features["brow_squeeze"] = np.clip(1.0 - brow_dist / (face_h * 0.25), 0, 1)
        
        # Nose analysis
        nose_w = dist(48, 278)
        nose_norm = nose_w / (face_h * 0.12)
        features["nose_wrinkle"] = np.clip((nose_norm - 1.0) / 0.3, 0, 1)
        
        # Smooth features
        self.history.append(features.copy())
        if len(self.history) >= 2:
            smoothed = {}
            for key in features:
                if isinstance(features[key], bool):
                    smoothed[key] = sum(h.get(key, False) for h in self.history) > len(self.history) / 2
                else:
                    values = [h.get(key, 0) for h in self.history]
                    weights = np.exp(np.linspace(-1, 0, len(values)))
                    smoothed[key] = np.average(values, weights=weights)
            return smoothed
        
        return features
    
    def close(self):
        self.face_mesh.close()


# ============================================
# HAND GESTURE RECOGNIZER
# ============================================
class HandRecognizer:
    """Recognizes hand gestures using MediaPipe Hands"""
    
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5, model_complexity=1):
        """
        Initialize hand recognizer with configurable parameters.
        
        Args:
            detection_confidence: Min confidence for detection (0.0-1.0). Lower = more detections but more false positives.
            tracking_confidence: Min confidence for tracking (0.0-1.0). Lower = smoother but less accurate.
            model_complexity: 0 (lite/fast) or 1 (full/accurate).
        """
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,  # 0=lite (faster), 1=full (more accurate)
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.history = deque(maxlen=8)  # Increased for better smoothing
        self.gesture_history = deque(maxlen=5)  # Separate gesture smoothing
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.last_detection_time = 0
        self.detection_cooldown = 0.05  # 50ms cooldown between detections
        
    def _get_finger_states(self, landmarks, is_right: bool) -> List[bool]:
        """Determine which fingers are extended"""
        lm = landmarks.landmark
        TIPS = [4, 8, 12, 16, 20]
        PIPS = [3, 6, 10, 14, 18]
        
        fingers = []
        
        if is_right:
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)
        
        for tip, pip in zip(TIPS[1:], PIPS[1:]):
            fingers.append(lm[tip].y < lm[pip].y)
        
        return fingers
    
    def _classify(self, fingers: List[bool], landmarks) -> Optional[str]:
        """Classify hand gesture based on finger states"""
        lm = landmarks.landmark
        thumb, index, middle, ring, pinky = fingers
        count = sum(fingers)
        
        if thumb and not any([index, middle, ring, pinky]):
            return "thumbs_up" if lm[4].y < lm[2].y else "thumbs_down"
        
        if not thumb and index and middle and not ring and not pinky:
            return "peace"
        
        if thumb and index:
            dist = np.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
            if dist < 0.06 and middle and ring and pinky:
                return "ok"
        
        if not thumb and index and not middle and not ring and pinky:
            return "rock"
        
        if thumb and not index and not middle and not ring and pinky:
            return "call_me"
        
        if thumb and index and not middle and not ring and pinky:
            return "love_you"
        
        if not thumb and index and not middle and not ring and not pinky:
            return "point_up"
        
        if count >= 4:
            return "open_hand"
        
        if count == 0:
            return "fist"
        
        return None
    
    def _get_hand_angle(self, landmarks) -> float:
        """Calculate hand rotation angle for better gesture detection"""
        lm = landmarks.landmark
        wrist = np.array([lm[0].x, lm[0].y])
        middle_mcp = np.array([lm[9].x, lm[9].y])
        direction = middle_mcp - wrist
        angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        return angle
    
    def detect(self, frame_rgb: np.ndarray) -> Tuple[Optional[str], List, Dict]:
        """
        Detect hand gestures with improved accuracy.
        Returns: (gesture_name, hand_data, debug_info)
        """
        results = self.hands.process(frame_rgb)
        
        gestures = []
        hand_data = []
        debug_info = {"hands_detected": 0, "confidence": 0.0}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            debug_info["hands_detected"] = len(results.multi_hand_landmarks)
            
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                is_right = handedness.classification[0].label == "Right"
                hand_confidence = handedness.classification[0].score
                debug_info["confidence"] = max(debug_info["confidence"], hand_confidence)
                
                fingers = self._get_finger_states(hand_lm, is_right)
                gesture = self._classify(fingers, hand_lm)
                hand_angle = self._get_hand_angle(hand_lm)
                
                if gesture:
                    gestures.append(gesture)
                hand_data.append({
                    "landmarks": hand_lm, 
                    "gesture": gesture,
                    "is_right": is_right,
                    "confidence": hand_confidence,
                    "angle": hand_angle,
                    "fingers": fingers
                })
        
        # Two-hand gesture detection
        if len(hand_data) == 2:
            palms = [hd["landmarks"].landmark[9] for hd in hand_data]
            dist = np.sqrt((palms[0].x - palms[1].x)**2 + (palms[0].y - palms[1].y)**2)
            if dist < 0.12:
                gestures = ["pray"]
        
        # Improved temporal smoothing with weighted voting
        self.gesture_history.append(gestures[0] if gestures else None)
        
        final_gesture = None
        if self.gesture_history:
            valid = [g for g in self.gesture_history if g is not None]
            if valid:
                # Weight recent gestures more heavily
                gesture_counts = Counter()
                for i, g in enumerate(valid):
                    weight = 1 + i * 0.5  # More recent = higher weight
                    gesture_counts[g] += weight
                
                most_common = gesture_counts.most_common(1)
                if most_common and most_common[0][1] >= 1.5:  # Threshold for confirmation
                    final_gesture = most_common[0][0]
        
        return final_gesture, hand_data, debug_info
    
    def draw(self, frame, hand_data, show_points: bool = False):
        """Draw hand landmarks on frame with optional detailed points"""
        h, w = frame.shape[:2]
        
        for hd in hand_data:
            # Draw connections
            self.mp_drawing.draw_landmarks(
                frame, hd["landmarks"], self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))
            
            # Draw detailed point labels if enabled
            if show_points:
                landmarks = hd["landmarks"].landmark
                point_names = [
                    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
                    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
                    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
                    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
                ]
                
                for i, lm in enumerate(landmarks):
                    px, py = int(lm.x * w), int(lm.y * h)
                    # Draw point circle
                    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                    # Draw point number
                    cv2.putText(frame, str(i), (px + 5, py - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return frame
    
    def close(self):
        self.hands.close()


# ============================================
# EMOJI MAPPER
# ============================================
def get_emoji(emotion: str, features: Dict, confidence: float, hand_gesture: Optional[str]) -> Tuple[str, str]:
    """Map emotion + features + gesture to emoji"""
    
    # Check for wink first
    if features.get("wink_left") or features.get("wink_right"):
        if emotion in ["happy", "neutral"]:
            return "😉", "Winking"
    
    # Emotion-specific mapping
    if emotion == "happy":
        if features["mouth_open"] > 0.6 and features["smile"] > 0.4:
            emoji, desc = "😂", "Laughing"
        elif features["eyes_closed"] > 0.5:
            emoji, desc = "😊", "Smiling"
        elif features["smile"] > 0.6:
            emoji, desc = "😁", "Grinning"
        elif features["asymmetric_mouth"] > 0.4:
            emoji, desc = "😏", "Smirk"
        else:
            emoji, desc = "😃", "Happy"
    
    elif emotion == "sad":
        if features["mouth_open"] > 0.4:
            emoji, desc = "😭", "Crying"
        elif features["brow_raised"] > 0.4:
            emoji, desc = "🥺", "Pleading"
        else:
            emoji, desc = "😢", "Sad"
    
    elif emotion == "angry":
        if features["mouth_open"] > 0.5:
            emoji, desc = "🤬", "Rage"
        elif features["brow_furrowed"] > 0.5 or features.get("brow_squeeze", 0) > 0.5:
            emoji, desc = "😡", "Furious"
        else:
            emoji, desc = "😠", "Angry"
    
    elif emotion == "surprise":
        if features["mouth_open"] > 0.7:
            emoji, desc = "😱", "Shocked"
        elif features["eyes_wide"] > 0.6:
            emoji, desc = "🤯", "Mind Blown"
        else:
            emoji, desc = "😲", "Surprised"
    
    elif emotion == "fear":
        if features["mouth_open"] > 0.5:
            emoji, desc = "😱", "Terrified"
        elif features["eyes_wide"] > 0.5:
            emoji, desc = "😨", "Fearful"
        else:
            emoji, desc = "😰", "Anxious"
    
    elif emotion == "disgust":
        if features.get("nose_wrinkle", 0) > 0.3:
            emoji, desc = "🤢", "Disgusted"
        elif features["asymmetric_mouth"] > 0.3:
            emoji, desc = "🤨", "Skeptical"
        else:
            emoji, desc = "😖", "Disgusted"
    
    elif emotion == "contempt":
        if features["asymmetric_mouth"] > 0.4:
            emoji, desc = "😏", "Smug"
        else:
            emoji, desc = "🤨", "Contempt"
    
    else:  # neutral
        if features["brow_raised"] > 0.4:
            emoji, desc = "🤔", "Thinking"
        elif features["eyes_closed"] > 0.6:
            emoji, desc = "😴", "Sleepy"
        elif features["asymmetric_mouth"] > 0.3:
            emoji, desc = "😒", "Unamused"
        else:
            emoji, desc = "😐", "Neutral"
    
    # Add hand gesture emoji
    if hand_gesture and hand_gesture in HAND_GESTURES:
        hand_emoji, hand_desc = HAND_GESTURES[hand_gesture]
        if hand_gesture in ["thumbs_up", "peace", "ok", "rock", "love_you"]:
            emoji = emoji + hand_emoji
            desc = f"{desc} + {hand_desc}"
    
    return emoji, desc


# ============================================
# DRAWING UTILITIES
# ============================================
def draw_emoji(frame, emoji: str, position: Tuple[int, int], size: int = 80):
    """Draw emoji on frame using PIL"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("seguiemj.ttf", size)
    except:
        font = ImageFont.load_default()
    draw.text(position, emoji, font=font, fill=(255, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_bars(frame, probs: np.ndarray, emotions: Dict[int, str], x: int, y: int, w: int):
    """Draw emotion probability bars"""
    bar_h, bar_w = 14, 140
    colors = [(0,0,255), (0,128,0), (128,0,128), (0,255,255), 
              (255,0,0), (0,165,255), (128,128,128), (139,69,19)]
    
    for i in range(len(probs)):
        emo = emotions.get(i, f"class_{i}")
        prob = probs[i]
        color = colors[i % len(colors)]
        
        by = y + i * (bar_h + 4)
        cv2.rectangle(frame, (x+w+10, by), (x+w+10+bar_w, by+bar_h), (40,40,40), -1)
        fill_w = int(bar_w * prob)
        cv2.rectangle(frame, (x+w+10, by), (x+w+10+fill_w, by+bar_h), color, -1)
        cv2.putText(frame, f"{emo.capitalize()}: {prob*100:.0f}%", (x+w+15, by+11),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)


def draw_features(frame, features: Dict, x: int, y: int):
    """Draw facial feature bars"""
    bar_h, bar_w = 10, 80
    feat_names = ["smile", "mouth_open", "brow_raised", "brow_furrowed", 
                  "eyes_wide", "asymmetric_mouth", "nose_wrinkle"]
    
    for i, feat in enumerate(feat_names):
        val = features.get(feat, 0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        
        by = y + i * (bar_h + 2)
        cv2.rectangle(frame, (x, by), (x + bar_w, by + bar_h), (40,40,40), -1)
        fill_w = int(bar_w * min(1.0, val))
        color = (0,255,0) if val > 0.5 else (200,200,0) if val > 0.25 else (80,80,80)
        cv2.rectangle(frame, (x, by), (x + fill_w, by + bar_h), color, -1)
        cv2.putText(frame, f"{feat[:8]}: {val:.1f}", (x + bar_w + 3, by + 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200,200,200), 1)


def draw_info_panel(frame, classifier: EmotionClassifier, fps: float):
    """Draw information panel at top of frame"""
    h, w = frame.shape[:2]
    
    # Background panel
    cv2.rectangle(frame, (0, 0), (w, 35), (30, 30, 30), -1)
    
    # Dataset info with ONNX indicator
    model_type = " [ONNX]" if classifier.use_onnx else ""
    dataset_text = f"Model: {classifier.config.name}{model_type}"
    cv2.putText(frame, dataset_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
    
    # Performance info
    avg_time = classifier.get_avg_inference_time()
    perf_text = f"Inference: {avg_time:.1f}ms | FPS: {fps:.1f}"
    cv2.putText(frame, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Device info
    device_text = f"Device: {'GPU' if classifier.device.type == 'cuda' else 'CPU'}"
    cv2.putText(frame, device_text, (w - 100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
    
    # Classes info
    classes_text = f"Classes: {classifier.num_classes}"
    cv2.putText(frame, classes_text, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_button_bar(frame, buttons: Dict[str, Tuple[str, bool]]):
    """
    Draw a button bar at the bottom of the frame on a black background.
    buttons: dict with key as shortcut, value as (label, is_active)
    """
    h, w = frame.shape[:2]
    bar_height = 35
    
    # Draw black background bar
    cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    
    # Calculate button positions
    num_buttons = len(buttons)
    button_width = w // num_buttons
    
    for i, (key, (label, is_active)) in enumerate(buttons.items()):
        x_start = i * button_width
        x_end = x_start + button_width - 2
        y_start = h - bar_height + 3
        y_end = h - 3
        
        # Button background
        if is_active:
            bg_color = (0, 120, 0)  # Green for active
        else:
            bg_color = (50, 50, 50)  # Dark gray for inactive
        
        cv2.rectangle(frame, (x_start + 2, y_start), (x_end, y_end), bg_color, -1)
        cv2.rectangle(frame, (x_start + 2, y_start), (x_end, y_end), (100, 100, 100), 1)
        
        # Button text with key highlight
        text = f"[{key}] {label}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x_start + (button_width - text_size[0]) // 2
        text_y = h - bar_height // 2 + 5
        
        # Draw text - key in yellow, label in white
        key_text = f"[{key}]"
        key_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        
        cv2.putText(frame, key_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, label, (text_x + key_size[0] + 3, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


# ============================================
# MODEL SELECTION
# ============================================
def find_models() -> List[str]:
    """Find available model files (.pth, .pt, .onnx)"""
    models = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in os.listdir(current_dir):
        if file.endswith(('.pth', '.pt', '.onnx')):
            # Prefer .pth/.pt files (ONNX loaded automatically)
            base_name = file.replace('.onnx', '').replace('.pth', '').replace('.pt', '')
            pth_exists = os.path.exists(os.path.join(current_dir, base_name + '.pth')) or \
                         os.path.exists(os.path.join(current_dir, base_name + '.pt'))
            
            # If ONNX available and Runtime installed, prefer PyTorch files (ONNX loaded automatically)
            # This way user selects .pth and gets .onnx speedup automatically
            if file.endswith('.onnx') and pth_exists:
                continue  # Skip .onnx in list, will be loaded automatically from .pth
            
            models.append(file)
    
    return sorted(models)


def select_model() -> str:
    """Let user select a model"""
    models = find_models()
    
    if not models:
        print("No model files found (.pth, .pt, .onnx)")
        return None
    
    print("\nAvailable models:")
    for i, model in enumerate(models):
        # Check if ONNX version exists
        base_name = model.replace('.pth', '').replace('.pt', '').replace('.onnx', '')
        onnx_exists = os.path.exists(base_name + '.onnx')
        onnx_indicator = " [ONNX ✓]" if onnx_exists and not model.endswith('.onnx') else ""
        print(f"  [{i+1}] {model}{onnx_indicator}")
    
    if len(models) == 1:
        print(f"\nUsing: {models[0]}")
        return models[0]
    
    while True:
        try:
            choice = input(f"\nSelect model [1-{len(models)}] (default: 2): ").strip()
            if not choice:
                if len(models) > 1:
                    return models[1]
                else:
                    return models[0]
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def select_models_for_comparison() -> List[str]:
    """Let user select multiple models for comparison"""
    models = find_models()
    
    if not models:
        print("No model files found (.pth, .pt, .onnx)")
        return []
    
    print("\nAvailable models:")
    for i, model in enumerate(models):
        # Check if ONNX version exists
        base_name = model.replace('.pth', '').replace('.pt', '').replace('.onnx', '')
        onnx_exists = os.path.exists(base_name + '.onnx')
        onnx_indicator = " [ONNX ✓]" if onnx_exists and not model.endswith('.onnx') else ""
        print(f"  [{i+1}] {model}{onnx_indicator}")
    
    print("\nSelect models to compare (e.g., '1,2,3' or '1 2 3'):")
    while True:
        try:
            choice = input("Models: ").strip()
            if not choice:
                return []
            
            # Parse input
            indices = []
            for part in choice.replace(',', ' ').split():
                idx = int(part) - 1
                if 0 <= idx < len(models):
                    indices.append(idx)
            
            if len(indices) < 2:
                print("Please select at least 2 models")
                continue
            
            if len(indices) > 4:
                print("Maximum 4 models allowed")
                continue
            
            selected = [models[i] for i in indices]
            print(f"\nSelected {len(selected)} models for comparison")
            return selected
        except ValueError:
            print("Invalid input, try again.")


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    print(f"\n{'='*65}")
    print("    Emoji Recognition V4 - Multi-Dataset Support")
    print(f"{'='*65}")
    print(f"Device: {DEVICE}")
    
    # Select mode
    print("\nSelect mode:")
    print("  [1] Normal mode (single model)")
    print("  [2] Comparison mode (multiple models - stacked display)")
    print("  [3] Ensemble mode (multiple models - fusion prediction)")
    
    mode = input("\nMode [1-3] (default: 1): ").strip()
    
    if mode == '2':
        # Comparison mode
        model_paths = select_models_for_comparison()
        if not model_paths:
            return
        main_comparison(model_paths)
    elif mode == '3':
        # Ensemble mode
        model_paths = select_models_for_comparison()
        if not model_paths:
            return
        main_ensemble(model_paths)
    else:
        # Normal mode
        model_path = select_model()
        if not model_path:
            return
        main_single(model_path)


def main_single(model_path: str):
    """Normal mode with single model"""
    # Initialize classifier with auto-detection
    print("\n" + "-"*50)
    classifier = EmotionClassifier(model_path, DEVICE)
    print("-"*50)
    
    print("\nInitializing face analyzer...")
    face_analyzer = FacialAnalyzer()
    print("  Face analyzer ready")
    
    print("Initializing hand recognizer...")
    hand_recognizer = HandRecognizer()
    print("  Hand recognizer ready")
    
    # Setup camera
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  Camera {idx} opened")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("Error: No camera found")
        return
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # State
    current_emoji = ""
    show_emotions = False
    show_features = False
    show_hands = True
    show_hand_points = False  # Show detailed hand landmarks
    fullscreen = False  # Fullscreen mode
    window_name = 'Emoji Recognition V4'
    
    # FPS counter
    fps_counter = deque(maxlen=30)
    last_time = time.perf_counter()
    
    print(f"\n{'='*65}")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Copy emoji to clipboard")
    print("  'e' - Toggle emotion bars")
    print("  'f' - Toggle feature bars (use F11 for fullscreen)")
    print("  'h' - Toggle hand visualization")
    print("  'p' - Toggle hand points (landmarks)")
    print("  'F11' or 'z' - Toggle fullscreen")
    print("  'm' - Change model")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"{'='*65}")
    print("\nTips for better detection:")
    print("  - Good lighting on your face")
    print("  - Face the camera directly")
    print("  - Exaggerate expressions slightly")
    print(f"{'='*65}\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            current_time = time.perf_counter()
            fps_counter.append(1.0 / (current_time - last_time + 1e-6))
            last_time = current_time
            fps = sum(fps_counter) / len(fps_counter)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw info panel
            draw_info_panel(frame, classifier, fps)
            
            # Detect hands (now returns debug_info too)
            hand_gesture, hand_data, hand_debug = hand_recognizer.detect(frame_rgb)
            
            if show_hands and hand_data:
                frame = hand_recognizer.draw(frame, hand_data, show_points=show_hand_points)
                
                # Show hand detection info
                if show_hand_points and hand_debug["hands_detected"] > 0:
                    info_text = f"Hands: {hand_debug['hands_detected']} | Conf: {hand_debug['confidence']:.0%}"
                    cv2.putText(frame, info_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if hand_gesture and hand_gesture in HAND_GESTURES:
                h_emoji, h_desc = HAND_GESTURES[hand_gesture]
                frame = draw_emoji(frame, h_emoji, (frame.shape[1] - 70, 40), 50)
            
            # Analyze facial features
            features = face_analyzer.analyze(frame_rgb)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                # Extract face with margin
                margin = int(0.1 * min(w, h))
                y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
                x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
                
                # Use appropriate format for model
                if classifier.in_channels == 3:
                    face_roi = frame_rgb[y1:y2, x1:x2]
                else:
                    face_roi = frame[y1:y2, x1:x2]
                
                # Predict emotion
                emotion, confidence, probs = classifier.predict(face_roi)
                
                # Get emoji
                emoji, description = get_emoji(emotion, features, confidence, hand_gesture)
                current_emoji = emoji
                color = EMOTION_COLORS.get(emotion, (128, 128, 128))
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y+35), (x+w, y+h+35), color, 2)
                
                # Draw label
                label = f"{description}: {confidence*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x, y+35-22), (x+tw+8, y+35), color, -1)
                cv2.putText(frame, label, (x+4, y+35-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                
                # Draw emoji
                frame = draw_emoji(frame, emoji, (x - 90, y+35), 80)
                
                # Optional displays
                if show_emotions:
                    draw_bars(frame, probs, classifier.config.emotions, x, y+35, w)
                
                if show_features:
                    draw_features(frame, features, 10, frame.shape[0] - 100)
            
            # Draw button bar at bottom
            frame = draw_button_bar(frame, {
                'E': ('Emotions', show_emotions),
                'F': ('Features', show_features),
                'H': ('Hands', show_hands),
                'P': ('Points', show_hand_points),
                'Z': ('Fullscreen', fullscreen),
                'S': ('Copy', False),
                'Q': ('Quit', False)
            })
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and current_emoji:
                pyperclip.copy(current_emoji)
                print(f"Copied: {current_emoji}")
            elif key == ord('e'):
                show_emotions = not show_emotions
            elif key == ord('f'):
                show_features = not show_features
            elif key == ord('h'):
                show_hands = not show_hands
            elif key == ord('p'):
                show_hand_points = not show_hand_points
                print(f"Hand points: {'ON' if show_hand_points else 'OFF'}")
            elif key == ord('z') or key == 122 or key == 0x7A:  # 'z' key for fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
            elif key == ord('m'):
                # Change model
                cv2.destroyAllWindows()
                new_model = select_model()
                if new_model:
                    classifier = EmotionClassifier(new_model, DEVICE)
                    print(f"\nSwitched to: {new_model}")
    
    finally:
        face_analyzer.close()
        hand_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


def main_comparison(model_paths: List[str]):
    """Comparison mode with multiple models on single screen with all features"""
    num_models = len(model_paths)
    
    # Initialize all classifiers
    classifiers = []
    model_names = []
    
    print("\n" + "="*65)
    print("Loading models for comparison...")
    print("="*65)
    
    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{num_models}] Loading {model_path}...")
        classifier = EmotionClassifier(model_path, DEVICE)
        classifiers.append(classifier)
        model_names.append(os.path.basename(model_path).replace('.pth', '').replace('.pt', ''))
        print(f"  ✓ Loaded: {classifier.config.name}")
    
    print("\nInitializing face analyzer...")
    face_analyzer = FacialAnalyzer()
    print("  Face analyzer ready")
    
    print("Initializing hand recognizer...")
    hand_recognizer = HandRecognizer()
    print("  Hand recognizer ready")
    
    # Setup camera
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  Camera {idx} opened")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("Error: No camera found")
        return
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # State - same as normal mode
    current_emoji = ""
    show_emotions = False
    show_features = False
    show_hands = True
    show_hand_points = False
    fullscreen = False
    window_name = 'Model Comparison - V4'
    
    # FPS counter
    fps_counter = deque(maxlen=30)
    last_time = time.perf_counter()
    
    print(f"\n{'='*65}")
    print("Controls (same as normal mode):")
    print("  'q' - Quit")
    print("  's' - Copy emoji to clipboard")
    print("  'e' - Toggle emotion bars")
    print("  'f' - Toggle feature bars")
    print("  'h' - Toggle hand visualization")
    print("  'p' - Toggle hand points (landmarks)")
    print("  'z' - Toggle fullscreen")
    print(f"{'='*65}\n")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            current_time = time.perf_counter()
            fps_counter.append(1.0 / (current_time - last_time + 1e-6))
            last_time = current_time
            fps = sum(fps_counter) / len(fps_counter)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw info panel with model comparison info
            panel_text = f"Comparing {num_models} models | FPS: {fps:.1f}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (40, 40, 40), -1)
            cv2.putText(frame, panel_text, (10, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detect hands
            hand_gesture, hand_data, hand_debug = hand_recognizer.detect(frame_rgb)
            
            if show_hands and hand_data:
                frame = hand_recognizer.draw(frame, hand_data, show_points=show_hand_points)
                
                if show_hand_points and hand_debug["hands_detected"] > 0:
                    info_text = f"Hands: {hand_debug['hands_detected']} | Conf: {hand_debug['confidence']:.0%}"
                    cv2.putText(frame, info_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if hand_gesture and hand_gesture in HAND_GESTURES:
                h_emoji, h_desc = HAND_GESTURES[hand_gesture]
                frame = draw_emoji(frame, h_emoji, (frame.shape[1] - 70, 40), 50)
            
            # Analyze facial features
            features = face_analyzer.analyze(frame_rgb)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            # Process each face with ALL models
            for (x, y, w, h) in faces:
                # Extract face with margin
                margin = int(0.1 * min(w, h))
                y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
                x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
                
                # Predictions from all models
                predictions = []
                for classifier, model_name in zip(classifiers, model_names):
                    # Use appropriate format for model
                    if classifier.in_channels == 3:
                        face_roi = frame_rgb[y1:y2, x1:x2]
                    else:
                        face_roi = frame[y1:y2, x1:x2]
                    
                    # Predict emotion
                    start = time.perf_counter()
                    emotion, confidence, probs = classifier.predict(face_roi)
                    inference_time = (time.perf_counter() - start) * 1000
                    
                    # Get emoji
                    emoji, description = get_emoji(emotion, features, confidence, hand_gesture)
                    color = EMOTION_COLORS.get(emotion, (128, 128, 128))
                    
                    predictions.append({
                        'model_name': model_name,
                        'emotion': emotion,
                        'description': description,
                        'confidence': confidence,
                        'emoji': emoji,
                        'color': color,
                        'probs': probs,
                        'inference_time': inference_time,
                        'classifier': classifier
                    })
                
                # Use first model's emoji as main emoji
                current_emoji = predictions[0]['emoji']
                main_color = predictions[0]['color']
                
                # Draw face rectangle with main color
                cv2.rectangle(frame, (x, y+35), (x+w, y+h+35), main_color, 2)
                
                # Draw labels for each model (stacked vertically)
                label_y_offset = 0
                for i, pred in enumerate(predictions):
                    label = f"{pred['model_name'][:8]}: {pred['description']} {pred['confidence']*100:.0f}% ({pred['inference_time']:.1f}ms)"
                    
                    # Smaller font for multiple models
                    font_scale = 0.4 if num_models > 2 else 0.5
                    thickness = 1
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Background for label
                    label_y = y + 35 - 22 - label_y_offset
                    cv2.rectangle(frame, (x, label_y), (x+tw+8, label_y+th+4), pred['color'], -1)
                    cv2.putText(frame, label, (x+4, label_y+th), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
                    
                    label_y_offset += th + 6
                
                # Draw main emoji (from first model)
                frame = draw_emoji(frame, current_emoji, (x - 90, y+35), 80)
                
                # Optional displays (using first model's data)
                if show_emotions:
                    # Show comparison of all models' predictions
                    bar_x = x
                    bar_y = y + 35
                    bar_spacing = 120 if num_models <= 2 else 80
                    
                    for i, pred in enumerate(predictions):
                        draw_bars(frame, pred['probs'], pred['classifier'].config.emotions, 
                                 bar_x + i * bar_spacing, bar_y, min(bar_spacing - 10, w // num_models))
                
                if show_features:
                    draw_features(frame, features, 10, frame.shape[0] - 100)
            
            # Draw button bar at bottom
            frame = draw_button_bar(frame, {
                'E': ('Emotions', show_emotions),
                'F': ('Features', show_features),
                'H': ('Hands', show_hands),
                'P': ('Points', show_hand_points),
                'Z': ('Fullscreen', fullscreen),
                'S': ('Copy', False),
                'Q': ('Quit', False)
            })
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and current_emoji:
                pyperclip.copy(current_emoji)
                print(f"Copied: {current_emoji}")
            elif key == ord('e'):
                show_emotions = not show_emotions
            elif key == ord('f'):
                show_features = not show_features
            elif key == ord('h'):
                show_hands = not show_hands
            elif key == ord('p'):
                show_hand_points = not show_hand_points
                print(f"Hand points: {'ON' if show_hand_points else 'OFF'}")
            elif key == ord('z'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
    
    finally:
        face_analyzer.close()
        hand_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


def main_ensemble(model_paths: List[str]):
    """Ensemble mode - Fuse multiple models into single prediction"""
    num_models = len(model_paths)
    
    # Initialize all classifiers
    classifiers = []
    model_names = []
    
    print("\n" + "="*65)
    print("Loading models for ensemble...")
    print("="*65)
    
    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{num_models}] Loading {model_path}...")
        classifier = EmotionClassifier(model_path, DEVICE)
        classifiers.append(classifier)
        model_names.append(os.path.basename(model_path).replace('.pth', '').replace('.pt', ''))
        print(f"  ✓ Loaded: {classifier.config.name}")
    
    print("\nInitializing face analyzer...")
    face_analyzer = FacialAnalyzer()
    print("  Face analyzer ready")
    
    print("Initializing hand recognizer...")
    hand_recognizer = HandRecognizer()
    print("  Hand recognizer ready")
    
    # Setup camera
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  Camera {idx} opened")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("Error: No camera found")
        return
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # State
    current_emoji = ""
    show_emotions = False
    show_features = False
    show_hands = True
    show_hand_points = False
    show_consensus = True  # Show model agreement
    fullscreen = False
    fusion_method = 'weighted'  # 'weighted', 'voting', 'average'
    window_name = 'Ensemble Mode - V4'
    
    # FPS counter
    fps_counter = deque(maxlen=30)
    last_time = time.perf_counter()
    
    print(f"\n{'='*65}")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Copy emoji to clipboard")
    print("  'e' - Toggle emotion bars")
    print("  'f' - Toggle feature bars")
    print("  'h' - Toggle hand visualization")
    print("  'p' - Toggle hand points")
    print("  'c' - Toggle consensus display")
    print("  'm' - Change fusion method (weighted/voting/average)")
    print("  'z' - Toggle fullscreen")
    print(f"{'='*65}\n")
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            current_time = time.perf_counter()
            fps_counter.append(1.0 / (current_time - last_time + 1e-6))
            last_time = current_time
            fps = sum(fps_counter) / len(fps_counter)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw info panel with ensemble info
            panel_text = f"Ensemble ({num_models} models) | Method: {fusion_method} | FPS: {fps:.1f}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (40, 40, 40), -1)
            cv2.putText(frame, panel_text, (10, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detect hands
            hand_gesture, hand_data, hand_debug = hand_recognizer.detect(frame_rgb)
            
            if show_hands and hand_data:
                frame = hand_recognizer.draw(frame, hand_data, show_points=show_hand_points)
                
                if show_hand_points and hand_debug["hands_detected"] > 0:
                    info_text = f"Hands: {hand_debug['hands_detected']} | Conf: {hand_debug['confidence']:.0%}"
                    cv2.putText(frame, info_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if hand_gesture and hand_gesture in HAND_GESTURES:
                h_emoji, h_desc = HAND_GESTURES[hand_gesture]
                frame = draw_emoji(frame, h_emoji, (frame.shape[1] - 70, 40), 50)
            
            # Analyze facial features
            features = face_analyzer.analyze(frame_rgb)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            # Process each face with ensemble
            for (x, y, w, h) in faces:
                # Extract face with margin
                margin = int(0.1 * min(w, h))
                y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
                x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
                
                # Collect predictions from all models
                all_predictions = []
                all_probs = []
                inference_times = []
                
                for classifier in classifiers:
                    # Use appropriate format for model
                    if classifier.in_channels == 3:
                        face_roi = frame_rgb[y1:y2, x1:x2]
                    else:
                        face_roi = frame[y1:y2, x1:x2]
                    
                    # Predict emotion
                    start = time.perf_counter()
                    emotion, confidence, probs = classifier.predict(face_roi)
                    inference_times.append((time.perf_counter() - start) * 1000)
                    
                    all_predictions.append(emotion)
                    all_probs.append(probs)
                
                # FUSION: Combine predictions
                if fusion_method == 'weighted':
                    # Weighted average by confidence
                    weights = np.array([np.max(p) for p in all_probs])
                    weights = weights / weights.sum()
                    
                    # Average probabilities weighted by confidence
                    ensemble_probs = np.zeros_like(all_probs[0])
                    for prob, weight in zip(all_probs, weights):
                        ensemble_probs += prob * weight
                    
                    final_emotion_idx = np.argmax(ensemble_probs)
                    final_confidence = ensemble_probs[final_emotion_idx]
                    
                elif fusion_method == 'voting':
                    # Majority voting
                    from collections import Counter
                    vote_counts = Counter(all_predictions)
                    final_emotion = vote_counts.most_common(1)[0][0]
                    
                    # Get average prob for voted emotion
                    final_emotion_idx = list(classifiers[0].config.emotions.values()).index(final_emotion)
                    ensemble_probs = np.mean(all_probs, axis=0)
                    final_confidence = ensemble_probs[final_emotion_idx]
                    
                else:  # 'average'
                    # Simple average of probabilities
                    ensemble_probs = np.mean(all_probs, axis=0)
                    final_emotion_idx = np.argmax(ensemble_probs)
                    final_confidence = ensemble_probs[final_emotion_idx]
                
                # Get final emotion name
                final_emotion = classifiers[0].config.emotions[final_emotion_idx]
                
                # Calculate consensus (agreement between models)
                consensus_count = sum(1 for pred in all_predictions if pred == final_emotion)
                consensus_pct = consensus_count / num_models
                
                # Get emoji
                emoji, description = get_emoji(final_emotion, features, final_confidence, hand_gesture)
                current_emoji = emoji
                color = EMOTION_COLORS.get(final_emotion, (128, 128, 128))
                
                # Adjust color based on consensus
                if consensus_pct < 0.5:
                    # Low consensus - yellowish
                    color = (color[0]//2 + 128, color[1]//2 + 128, color[2]//2)
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y+35), (x+w, y+h+35), color, 2)
                
                # Draw main label with ensemble info
                avg_time = np.mean(inference_times)
                label = f"Ensemble: {description} {final_confidence*100:.0f}% ({avg_time:.1f}ms)"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x, y+35-22), (x+tw+8, y+35), color, -1)
                cv2.putText(frame, label, (x+4, y+35-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                
                # Show consensus info
                if show_consensus:
                    consensus_label = f"Agreement: {consensus_count}/{num_models} ({consensus_pct*100:.0f}%)"
                    consensus_color = (0, 255, 0) if consensus_pct >= 0.7 else (0, 255, 255) if consensus_pct >= 0.5 else (0, 165, 255)
                    cv2.putText(frame, consensus_label, (x, y+h+55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, consensus_color, 1)
                    
                    # Show individual model predictions (compact)
                    models_text = " | ".join([f"{name[:4]}:{pred[:3]}" for name, pred in zip(model_names, all_predictions)])
                    cv2.putText(frame, models_text, (x, y+h+70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                
                # Draw emoji
                frame = draw_emoji(frame, emoji, (x - 90, y+35), 80)
                
                # Optional displays
                if show_emotions:
                    draw_bars(frame, ensemble_probs, classifiers[0].config.emotions, x, y+35, w)
                
                if show_features:
                    draw_features(frame, features, 10, frame.shape[0] - 100)
            
            # Draw button bar at bottom
            frame = draw_button_bar(frame, {
                'E': ('Emotions', show_emotions),
                'F': ('Features', show_features),
                'H': ('Hands', show_hands),
                'P': ('Points', show_hand_points),
                'C': ('Consensus', show_consensus),
                'M': (f'Method:{fusion_method[:3]}', False),
                'Z': ('Fullscreen', fullscreen),
                'S': ('Copy', False),
                'Q': ('Quit', False)
            })
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and current_emoji:
                pyperclip.copy(current_emoji)
                print(f"Copied: {current_emoji}")
            elif key == ord('e'):
                show_emotions = not show_emotions
            elif key == ord('f'):
                show_features = not show_features
            elif key == ord('h'):
                show_hands = not show_hands
            elif key == ord('p'):
                show_hand_points = not show_hand_points
                print(f"Hand points: {'ON' if show_hand_points else 'OFF'}")
            elif key == ord('c'):
                show_consensus = not show_consensus
                print(f"Consensus display: {'ON' if show_consensus else 'OFF'}")
            elif key == ord('m'):
                # Cycle through fusion methods
                methods = ['weighted', 'voting', 'average']
                current_idx = methods.index(fusion_method)
                fusion_method = methods[(current_idx + 1) % len(methods)]
                print(f"Fusion method: {fusion_method}")
            elif key == ord('z'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
    
    finally:
        face_analyzer.close()
        hand_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
