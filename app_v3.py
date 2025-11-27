"""
Emoji Recognition V3 - Optimized for Accuracy
- Better CNN preprocessing with TTA (Test-Time Augmentation)
- Improved facial feature detection with calibration
- Smoother hand gesture recognition
- Better emotion thresholds tuned for real-world use
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pyperclip
from collections import deque, Counter
from model import FaceEmotionCNN, create_model
import mediapipe as mp
from typing import Dict, List, Tuple, Optional

# ============================================
# CONFIGURATION
# ============================================
NUM_CLASSES = 8
IMG_SIZE = 75  # AffectNet uses 75x75 RGB images
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_EMOTIONS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy", 
    4: "sad", 5: "surprise", 6: "neutral", 7: "contempt"
}

# Émojis pour chaque émotion de base
BASE_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😃",
    "sad": "😢", "surprise": "😲", "neutral": "😐", "contempt": "😏"
}

EMOTION_COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happy": (0, 255, 255), "sad": (255, 0, 0), "surprise": (0, 165, 255),
    "neutral": (128, 128, 128), "contempt": (139, 69, 19)
}

# Gestes de main
HAND_GESTURES = {
    "thumbs_up": ("👍", "Thumbs Up"), "thumbs_down": ("👎", "Thumbs Down"),
    "peace": ("✌️", "Peace"), "ok": ("👌", "OK"), "rock": ("🤘", "Rock"),
    "call_me": ("🤙", "Call Me"), "fist": ("✊", "Fist"),
    "open_hand": ("🖐️", "High Five"), "point_up": ("☝️", "Point Up"),
    "pray": ("🙏", "Pray"), "love_you": ("🤟", "Love You"),
}

# ============================================
# EMOTION CLASSIFIER WITH TTA
# ============================================
class EmotionClassifier:
    """CNN-based emotion classifier with Test-Time Augmentation for better accuracy"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        # Use create_model for RGB 75x75 input (AffectNet)
        self.model = create_model(num_classes=NUM_CLASSES, dataset='affectnet').to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Preprocessing for RGB 75x75 with ImageNet normalization
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # TTA transforms (light augmentations)
        self.tta_transforms = [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),  # Flip
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]),
        ]
        
        # Emotion-specific adjustments (learned from real usage)
        # Ces facteurs compensent les biais du dataset FER2013
        self.emotion_adjustments = {
            0: 1.3,   # angry - souvent confondu avec sad
            1: 1.8,   # disgust - très sous-représenté
            2: 1.4,   # fear - souvent confondu avec surprise
            3: 0.95,  # happy - bien représenté
            4: 1.0,   # sad
            5: 1.0,   # surprise
            6: 0.8,   # neutral - trop souvent prédit par défaut
            7: 1.5,   # contempt - difficile à détecter
        }
        
        # History for temporal smoothing
        self.history = deque(maxlen=7)
        
    def preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for RGB images"""
        # Ensure RGB format (should already be RGB)
        if len(face_roi.shape) == 2:
            # If grayscale, convert to RGB
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        elif face_roi.shape[2] == 4:
            # If RGBA, convert to RGB
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGBA2RGB)
        
        # Apply CLAHE to each channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        channels = cv2.split(face_roi)
        clahe_channels = [clahe.apply(ch) for ch in channels]
        face_roi = cv2.merge(clahe_channels)
        
        return face_roi
    
    def predict(self, face_roi: np.ndarray, use_tta: bool = True) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion with optional TTA
        Returns: (emotion_name, confidence, all_probabilities)
        """
        face_roi = self.preprocess(face_roi)
        
        with torch.no_grad():
            if use_tta:
                # Aggregate predictions from multiple augmentations
                all_probs = []
                for transform in self.tta_transforms:
                    tensor = transform(face_roi).unsqueeze(0).to(self.device)
                    output = self.model(tensor)
                    probs = F.softmax(output, dim=1)[0]
                    all_probs.append(probs)
                
                # Average probabilities
                probs = torch.stack(all_probs).mean(dim=0)
            else:
                tensor = self.base_transform(face_roi).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)[0]
            
            # Apply emotion-specific adjustments
            adjusted_probs = probs.clone()
            for idx, factor in self.emotion_adjustments.items():
                adjusted_probs[idx] *= factor
            
            # Renormalize
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            # Temporal smoothing
            self.history.append(adjusted_probs.cpu().numpy())
            
            if len(self.history) >= 3:
                # Weighted average with more weight on recent frames
                weights = np.exp(np.linspace(-1, 0, len(self.history)))
                weights /= weights.sum()
                smoothed = np.zeros(NUM_CLASSES)
                for i, h in enumerate(self.history):
                    smoothed += weights[i] * h
                final_probs = smoothed
            else:
                final_probs = adjusted_probs.cpu().numpy()
            
            # Get prediction
            pred_idx = np.argmax(final_probs)
            confidence = final_probs[pred_idx]
            emotion = BASE_EMOTIONS[pred_idx]
            
            return emotion, confidence, final_probs


# ============================================
# FACIAL FEATURE ANALYZER
# ============================================
class FacialAnalyzer:
    """Analyzes facial features using MediaPipe Face Mesh"""
    
    # Key landmark indices
    LANDMARKS = {
        # Eyes
        "left_eye": [33, 133, 159, 145],  # outer, inner, top, bottom
        "right_eye": [263, 362, 386, 374],
        # Eyebrows
        "left_brow": [70, 107, 105],  # outer, inner, top
        "right_brow": [300, 336, 334],
        # Mouth
        "mouth": [61, 291, 13, 14, 0, 17],  # left, right, top, bottom, upper_lip, lower_lip
        # Nose
        "nose": [4, 48, 278],  # tip, left, right
        # Face bounds
        "face": [10, 152],  # forehead, chin
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
        
        # Face height for normalization
        face_h = dist(10, 152)
        if face_h < 10:
            return features
        
        # === Mouth ===
        mouth_h = dist(13, 14)  # vertical opening
        mouth_w = dist(61, 291)  # width
        
        features["mouth_open"] = np.clip(mouth_h / (face_h * 0.12), 0, 1)
        
        # Smile: width/height ratio
        if mouth_h > 1:
            smile_ratio = mouth_w / mouth_h
            features["smile"] = np.clip((smile_ratio - 2.5) / 4.0, 0, 1)
        
        # Asymmetric mouth (contempt)
        left_corner = point(61)
        right_corner = point(291)
        mouth_center_y = (left_corner[1] + right_corner[1]) / 2
        asymmetry = abs(left_corner[1] - right_corner[1]) / (face_h * 0.05)
        features["asymmetric_mouth"] = np.clip(asymmetry, 0, 1)
        
        # === Eyes ===
        left_eye_h = dist(159, 145)
        right_eye_h = dist(386, 374)
        left_eye_w = dist(33, 133)
        right_eye_w = dist(263, 362)
        
        avg_eye_h = (left_eye_h + right_eye_h) / 2
        avg_eye_w = (left_eye_w + right_eye_w) / 2
        
        if avg_eye_w > 1:
            ear = avg_eye_h / avg_eye_w  # Eye aspect ratio
            features["eyes_wide"] = np.clip((ear - 0.25) / 0.15, 0, 1)
            features["eyes_closed"] = np.clip((0.18 - ear) / 0.1, 0, 1)
            
            # Wink detection
            left_ear = left_eye_h / left_eye_w if left_eye_w > 1 else 0.3
            right_ear = right_eye_h / right_eye_w if right_eye_w > 1 else 0.3
            features["wink_left"] = left_ear < 0.15 and right_ear > 0.2
            features["wink_right"] = right_ear < 0.15 and left_ear > 0.2
        
        # === Eyebrows ===
        left_brow_h = dist(105, 159)  # brow top to eye top
        right_brow_h = dist(334, 386)
        avg_brow_h = (left_brow_h + right_brow_h) / 2
        
        brow_norm = avg_brow_h / (face_h * 0.08)
        features["brow_raised"] = np.clip((brow_norm - 1.0) / 0.5, 0, 1)
        features["brow_furrowed"] = np.clip((1.0 - brow_norm) / 0.4, 0, 1)
        
        # Brow squeeze (for angry/fear)
        left_brow_inner = point(107)
        right_brow_inner = point(336)
        brow_dist = np.linalg.norm(left_brow_inner - right_brow_inner)
        features["brow_squeeze"] = np.clip(1.0 - brow_dist / (face_h * 0.25), 0, 1)
        
        # === Nose wrinkle (disgust) ===
        nose_w = dist(48, 278)
        nose_norm = nose_w / (face_h * 0.12)
        features["nose_wrinkle"] = np.clip((nose_norm - 1.0) / 0.3, 0, 1)
        
        # Smooth features
        self.history.append(features.copy())
        if len(self.history) >= 2:
            smoothed = {}
            for key in features:
                if isinstance(features[key], bool):
                    # Majority vote for booleans
                    smoothed[key] = sum(h.get(key, False) for h in self.history) > len(self.history) / 2
                else:
                    # Exponential moving average for floats
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
    
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.history = deque(maxlen=5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
    def _get_finger_states(self, landmarks, is_right: bool) -> List[bool]:
        """Determine which fingers are extended"""
        lm = landmarks.landmark
        
        # Finger tip and pip indices
        TIPS = [4, 8, 12, 16, 20]
        PIPS = [3, 6, 10, 14, 18]
        
        fingers = []
        
        # Thumb (different logic based on hand)
        if is_right:
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)
        
        # Other fingers
        for tip, pip in zip(TIPS[1:], PIPS[1:]):
            fingers.append(lm[tip].y < lm[pip].y)
        
        return fingers
    
    def _classify(self, fingers: List[bool], landmarks) -> Optional[str]:
        """Classify hand gesture based on finger states"""
        lm = landmarks.landmark
        thumb, index, middle, ring, pinky = fingers
        count = sum(fingers)
        
        # Thumbs up/down
        if thumb and not any([index, middle, ring, pinky]):
            return "thumbs_up" if lm[4].y < lm[2].y else "thumbs_down"
        
        # Peace (V sign)
        if not thumb and index and middle and not ring and not pinky:
            return "peace"
        
        # OK sign
        if thumb and index:
            dist = np.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
            if dist < 0.06 and middle and ring and pinky:
                return "ok"
        
        # Rock
        if not thumb and index and not middle and not ring and pinky:
            return "rock"
        
        # Call me (shaka)
        if thumb and not index and not middle and not ring and pinky:
            return "call_me"
        
        # Love you
        if thumb and index and not middle and not ring and pinky:
            return "love_you"
        
        # Point up
        if not thumb and index and not middle and not ring and not pinky:
            return "point_up"
        
        # Open hand
        if count >= 4:
            return "open_hand"
        
        # Fist
        if count == 0:
            return "fist"
        
        return None
    
    def detect(self, frame_rgb: np.ndarray) -> Tuple[Optional[str], List]:
        """Detect hand gestures"""
        results = self.hands.process(frame_rgb)
        
        gestures = []
        hand_data = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                is_right = handedness.classification[0].label == "Right"
                fingers = self._get_finger_states(hand_lm, is_right)
                gesture = self._classify(fingers, hand_lm)
                
                if gesture:
                    gestures.append(gesture)
                hand_data.append({"landmarks": hand_lm, "gesture": gesture})
        
        # Check for two-hand gestures
        if len(hand_data) == 2:
            # Pray gesture
            palms = [hd["landmarks"].landmark[9] for hd in hand_data]
            dist = np.sqrt((palms[0].x - palms[1].x)**2 + (palms[0].y - palms[1].y)**2)
            if dist < 0.12:
                gestures = ["pray"]
        
        # Temporal smoothing
        self.history.append(gestures[0] if gestures else None)
        
        if self.history:
            # Get most common non-None gesture
            valid = [g for g in self.history if g is not None]
            if valid:
                most_common = Counter(valid).most_common(1)
                if most_common and most_common[0][1] >= 2:
                    return most_common[0][0], hand_data
        
        return gestures[0] if gestures else None, hand_data
    
    def draw(self, frame, hand_data):
        """Draw hand landmarks on frame"""
        for hd in hand_data:
            self.mp_drawing.draw_landmarks(
                frame, hd["landmarks"], self.mp_hands.HAND_CONNECTIONS)
        return frame
    
    def close(self):
        self.hands.close()


# ============================================
# EMOJI MAPPER
# ============================================
def get_emoji(emotion: str, features: Dict, confidence: float, hand_gesture: Optional[str]) -> Tuple[str, str]:
    """
    Map emotion + features + gesture to emoji
    Returns: (emoji, description)
    """
    
    # Check for wink first
    if features.get("wink_left") or features.get("wink_right"):
        if emotion in ["happy", "neutral"]:
            return "😉", "Winking"
    
    # Emotion-specific mapping with feature refinement
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
    
    # Add hand gesture emoji if present
    if hand_gesture and hand_gesture in HAND_GESTURES:
        hand_emoji, hand_desc = HAND_GESTURES[hand_gesture]
        # Combine face + hand
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


def draw_bars(frame, probs: np.ndarray, x: int, y: int, w: int):
    """Draw emotion probability bars"""
    bar_h, bar_w = 14, 140
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]
    colors = [(0,0,255), (0,128,0), (128,0,128), (0,255,255), 
              (255,0,0), (0,165,255), (128,128,128), (139,69,19)]
    
    for i, (emo, prob) in enumerate(zip(emotions, probs)):
        by = y + i * (bar_h + 4)
        # Background
        cv2.rectangle(frame, (x+w+10, by), (x+w+10+bar_w, by+bar_h), (40,40,40), -1)
        # Fill
        fill_w = int(bar_w * prob)
        cv2.rectangle(frame, (x+w+10, by), (x+w+10+fill_w, by+bar_h), colors[i], -1)
        # Text
        cv2.putText(frame, f"{emo}: {prob*100:.0f}%", (x+w+15, by+11),
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


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    print(f"\n{'='*60}")
    print("    Emoji Recognition V3 - Optimized for Accuracy")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    
    # Initialize components
    print("Loading model...")
    classifier = EmotionClassifier('emotion_model.pth', DEVICE)
    print("✓ Emotion classifier loaded (RGB 75x75)")
    
    print("Initializing face analyzer...")
    face_analyzer = FacialAnalyzer()
    print("✓ Face analyzer ready")
    
    print("Initializing hand recognizer...")
    hand_recognizer = HandRecognizer()
    print("✓ Hand recognizer ready")
    
    # Setup camera
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"✓ Camera {idx} opened")
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
    
    print(f"\n{'='*60}")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Copy emoji to clipboard")
    print("  'e' - Toggle emotion bars")
    print("  'f' - Toggle feature bars")
    print("  'h' - Toggle hand visualization")
    print(f"{'='*60}")
    print("\nTips for better detection:")
    print("  • Good lighting on your face")
    print("  • Face the camera directly")
    print("  • Exaggerate expressions slightly")
    print("  • For DISGUST: wrinkle nose, raise upper lip")
    print("  • For CONTEMPT: one-sided smirk")
    print("  • For ANGRY: frown and squeeze eyebrows")
    print(f"{'='*60}\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect hands
            hand_gesture, hand_data = hand_recognizer.detect(frame_rgb)
            
            # Draw hands
            if show_hands and hand_data:
                frame = hand_recognizer.draw(frame, hand_data)
            
            # Show hand gesture
            if hand_gesture and hand_gesture in HAND_GESTURES:
                h_emoji, h_desc = HAND_GESTURES[hand_gesture]
                frame = draw_emoji(frame, h_emoji, (frame.shape[1] - 70, 5), 50)
            
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
                
                # Use RGB for the model (AffectNet format)
                face_roi = frame_rgb[y1:y2, x1:x2]
                
                # Predict emotion
                emotion, confidence, probs = classifier.predict(face_roi)
                
                # Get emoji
                emoji, description = get_emoji(emotion, features, confidence, hand_gesture)
                current_emoji = emoji
                color = EMOTION_COLORS[emotion]
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{description}: {confidence*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x, y-22), (x+tw+8, y), color, -1)
                cv2.putText(frame, label, (x+4, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                
                # Draw emoji
                frame = draw_emoji(frame, emoji, (x - 90, y), 80)
                
                # Optional displays
                if show_emotions:
                    draw_bars(frame, probs, x, y, w)
                
                if show_features:
                    draw_features(frame, features, 10, frame.shape[0] - 100)
            
            # Help text
            cv2.putText(frame, "'e'=emotions 'f'=features 'h'=hands 's'=copy 'q'=quit", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            
            cv2.imshow('Emoji Recognition V3', frame)
            
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
    
    finally:
        face_analyzer.close()
        hand_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
