"""
Extended Emotion Recognition with Enhanced Emoji Detection
Combines CNN emotion detection with facial landmark analysis for richer emoji suggestions
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pyperclip
from collections import deque
from model import FaceEmotionCNN
import mediapipe as mp

# ============================================
# CONFIGURATION
# ============================================
NUM_CLASSES = 8  # 8 émotions AffectNet
IN_CHANNELS = 3  # RGB
INPUT_SIZE = 75  # 75x75 pixels

# Émotions de base (CNN)
BASE_EMOTIONS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy", 
    4: "sad", 5: "surprise", 6: "neutral", 7: "contempt"
}

# Panel étendu d'émojis basé sur les combinaisons émotion + features faciales
EXTENDED_EMOJI_MAP = {
    # Happy variations
    ("happy", "big_smile"): ("😁", "Big Grin"),
    ("happy", "teeth_showing"): ("😄", "Grinning"),
    ("happy", "eyes_closed"): ("😊", "Smiling Eyes"),
    ("happy", "wink"): ("😉", "Wink"),
    ("happy", "laughing"): ("😂", "Laughing"),
    ("happy", "love"): ("😍", "Heart Eyes"),
    ("happy", "default"): ("😃", "Happy"),
    ("happy", "smirk"): ("😏", "Smirk"),
    
    # Sad variations
    ("sad", "crying"): ("😢", "Crying"),
    ("sad", "very_sad"): ("😭", "Sobbing"),
    ("sad", "disappointed"): ("😞", "Disappointed"),
    ("sad", "pleading"): ("🥺", "Pleading"),
    ("sad", "default"): ("😔", "Pensive"),
    
    # Angry variations
    ("angry", "very_angry"): ("😡", "Pouting"),
    ("angry", "annoyed"): ("😤", "Huffing"),
    ("angry", "rage"): ("🤬", "Cursing"),
    ("angry", "default"): ("😠", "Angry"),
    ("angry", "frustrated"): ("😣", "Persevering"),
    
    # Surprise variations
    ("surprise", "shocked"): ("😱", "Screaming"),
    ("surprise", "amazed"): ("🤩", "Star-Struck"),
    ("surprise", "wow"): ("😮", "Wow"),
    ("surprise", "default"): ("😲", "Astonished"),
    ("surprise", "mind_blown"): ("🤯", "Mind Blown"),
    
    # Fear variations
    ("fear", "scared"): ("😨", "Fearful"),
    ("fear", "anxious"): ("😰", "Anxious"),
    ("fear", "cold_sweat"): ("😥", "Cold Sweat"),
    ("fear", "default"): ("😧", "Anguished"),
    
    # Disgust variations
    ("disgust", "nauseated"): ("🤢", "Nauseated"),
    ("disgust", "vomiting"): ("🤮", "Vomiting"),
    ("disgust", "sick"): ("😷", "Sick"),
    ("disgust", "default"): ("🤢", "Disgusted"),
    ("disgust", "skeptical"): ("🤨", "Raised Eyebrow"),
    
    # Neutral variations
    ("neutral", "thinking"): ("🤔", "Thinking"),
    ("neutral", "sleepy"): ("😴", "Sleeping"),
    ("neutral", "bored"): ("😑", "Expressionless"),
    ("neutral", "confused"): ("😕", "Confused"),
    ("neutral", "default"): ("😐", "Neutral"),
    ("neutral", "unamused"): ("😒", "Unamused"),
    ("neutral", "rolling_eyes"): ("🙄", "Rolling Eyes"),
    ("neutral", "side_eye"): ("👀", "Eyes"),
    
    # Contempt variations
    ("contempt", "smug"): ("😏", "Smug"),
    ("contempt", "judging"): ("🤨", "Judging"),
    ("contempt", "sarcastic"): ("😒", "Sarcastic"),
    ("contempt", "default"): ("😏", "Contempt"),
    
    # Special expressions
    ("any", "tongue_out"): ("😛", "Tongue Out"),
    ("any", "crazy"): ("🤪", "Zany"),
    ("any", "cool"): ("😎", "Cool"),
    ("any", "kiss"): ("😘", "Blowing Kiss"),
    ("any", "shush"): ("🤫", "Shushing"),
    ("any", "yawning"): ("🥱", "Yawning"),
    ("any", "relieved"): ("😌", "Relieved"),
}

# Couleurs pour chaque émotion de base (BGR)
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
    "neutral": (128, 128, 128),
    "contempt": (139, 69, 19)
}

# ============================================
# MEDIAPIPE FACE MESH SETUP
# ============================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Indices importants des landmarks
LANDMARKS = {
    # Yeux
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    
    # Sourcils
    "left_eyebrow_inner": 107,
    "left_eyebrow_outer": 70,
    "left_eyebrow_top": 105,
    "right_eyebrow_inner": 336,
    "right_eyebrow_outer": 300,
    "right_eyebrow_top": 334,
    
    # Bouche
    "mouth_top": 13,
    "mouth_bottom": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip_top": 0,
    "lower_lip_bottom": 17,
    
    # Nez
    "nose_tip": 4,
    
    # Visage
    "chin": 152,
    "forehead": 10,
}


class FacialFeatureAnalyzer:
    """Analyse les features faciales à partir des landmarks MediaPipe"""
    
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=5)
        
    def analyze(self, frame_rgb, face_rect=None):
        """
        Analyse les features faciales et retourne un dictionnaire de caractéristiques
        """
        results = self.face_mesh.process(frame_rgb)
        
        features = {
            "mouth_open": 0.0,
            "smile_intensity": 0.0,
            "eyebrows_raised": 0.0,
            "eyebrows_furrowed": 0.0,
            "eyes_wide": 0.0,
            "eyes_closed": 0.0,
            "left_eye_closed": False,
            "right_eye_closed": False,
            "asymmetric_smile": 0.0,
            "head_tilt": 0.0,
            "looking_away": False,
            "tongue_out": False,
        }
        
        if not results.multi_face_landmarks:
            return features
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_rgb.shape[:2]
        
        def get_point(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
        def distance(p1, p2):
            return np.linalg.norm(get_point(p1) - get_point(p2))
        
        # Face height for normalization
        face_height = distance(LANDMARKS["forehead"], LANDMARKS["chin"])
        if face_height < 1:
            return features
            
        # === Mouth Analysis ===
        mouth_height = distance(LANDMARKS["mouth_top"], LANDMARKS["mouth_bottom"])
        mouth_width = distance(LANDMARKS["mouth_left"], LANDMARKS["mouth_right"])
        
        # Mouth openness (normalized)
        features["mouth_open"] = min(1.0, mouth_height / (face_height * 0.15))
        
        # Lip distance (for bigger open mouth)
        lip_distance = distance(LANDMARKS["upper_lip_top"], LANDMARKS["lower_lip_bottom"])
        
        # Smile detection (mouth width vs height ratio)
        if mouth_height > 0:
            smile_ratio = mouth_width / mouth_height
            features["smile_intensity"] = min(1.0, max(0, (smile_ratio - 2.0) / 3.0))
        
        # Asymmetric smile (smirk detection)
        mouth_left = get_point(LANDMARKS["mouth_left"])
        mouth_right = get_point(LANDMARKS["mouth_right"])
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
        left_height = abs(mouth_left[1] - mouth_center_y)
        right_height = abs(mouth_right[1] - mouth_center_y)
        if max(left_height, right_height) > 0:
            features["asymmetric_smile"] = abs(left_height - right_height) / max(left_height, right_height, 1)
        
        # === Eye Analysis ===
        left_eye_height = distance(LANDMARKS["left_eye_top"], LANDMARKS["left_eye_bottom"])
        right_eye_height = distance(LANDMARKS["right_eye_top"], LANDMARKS["right_eye_bottom"])
        left_eye_width = distance(LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"])
        right_eye_width = distance(LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"])
        
        # Eye openness
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        if avg_eye_width > 0:
            eye_aspect_ratio = avg_eye_height / avg_eye_width
            features["eyes_wide"] = min(1.0, max(0, (eye_aspect_ratio - 0.25) / 0.15))
            features["eyes_closed"] = max(0, (0.2 - eye_aspect_ratio) / 0.15)
        
        # Individual eye closure (for wink detection)
        if left_eye_width > 0:
            left_ear = left_eye_height / left_eye_width
            features["left_eye_closed"] = left_ear < 0.15
        if right_eye_width > 0:
            right_ear = right_eye_height / right_eye_width
            features["right_eye_closed"] = right_ear < 0.15
        
        # === Eyebrow Analysis ===
        left_brow_height = distance(LANDMARKS["left_eyebrow_top"], LANDMARKS["left_eye_top"])
        right_brow_height = distance(LANDMARKS["right_eyebrow_top"], LANDMARKS["right_eye_top"])
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        
        # Normalized eyebrow position
        brow_normalized = avg_brow_height / (face_height * 0.1)
        features["eyebrows_raised"] = min(1.0, max(0, (brow_normalized - 1.0) / 0.5))
        features["eyebrows_furrowed"] = min(1.0, max(0, (1.0 - brow_normalized) / 0.3))
        
        # === Head Pose ===
        left_face = get_point(LANDMARKS["left_eye_outer"])
        right_face = get_point(LANDMARKS["right_eye_outer"])
        head_tilt = (left_face[1] - right_face[1]) / face_height
        features["head_tilt"] = head_tilt
        
        # Looking away detection (eyes not centered)
        nose = get_point(LANDMARKS["nose_tip"])
        face_center_x = (left_face[0] + right_face[0]) / 2
        face_width = abs(left_face[0] - right_face[0])
        if face_width > 0:
            nose_offset = abs(nose[0] - face_center_x) / face_width
            features["looking_away"] = nose_offset > 0.15
        
        # Smooth features over time
        self.history.append(features)
        if len(self.history) >= 2:
            smoothed = {}
            for key in features:
                if isinstance(features[key], bool):
                    # For boolean, take majority vote
                    smoothed[key] = sum(1 for h in self.history if h.get(key, False)) > len(self.history) / 2
                else:
                    # For floats, average
                    smoothed[key] = np.mean([h.get(key, 0) for h in self.history])
            return smoothed
            
        return features
    
    def close(self):
        self.face_mesh.close()


def get_extended_emoji(base_emotion: str, features: dict, confidence: float) -> tuple:
    """
    Détermine l'émoji étendu basé sur l'émotion de base et les features faciales
    Returns: (emoji, description)
    """
    
    # Détections spéciales qui override l'émotion de base
    
    # Wink detection
    if features["left_eye_closed"] != features["right_eye_closed"]:
        if base_emotion in ["happy", "neutral"]:
            return ("😉", "Winking")
    
    # Tongue out (would need additional detection - placeholder)
    
    # Based on emotion + features
    if base_emotion == "happy":
        if features["mouth_open"] > 0.6 and features["smile_intensity"] > 0.5:
            return ("😂", "Laughing")
        elif features["eyes_closed"] > 0.5 and features["smile_intensity"] > 0.3:
            return ("😊", "Smiling Eyes")
        elif features["smile_intensity"] > 0.7:
            return ("😁", "Big Grin")
        elif features["asymmetric_smile"] > 0.4:
            return ("😏", "Smirk")
        else:
            return ("😃", "Happy")
    
    elif base_emotion == "sad":
        if features["mouth_open"] > 0.4 and features["eyebrows_raised"] > 0.3:
            return ("😭", "Sobbing")
        elif features["eyebrows_raised"] > 0.5:
            return ("🥺", "Pleading")
        elif features["eyes_closed"] > 0.3:
            return ("😢", "Crying")
        else:
            return ("😔", "Pensive")
    
    elif base_emotion == "angry":
        if features["mouth_open"] > 0.5 and features["eyebrows_furrowed"] > 0.4:
            return ("🤬", "Cursing")
        elif features["eyebrows_furrowed"] > 0.5:
            return ("😡", "Pouting")
        elif confidence > 80:
            return ("😤", "Huffing")
        else:
            return ("😠", "Angry")
    
    elif base_emotion == "surprise":
        if features["mouth_open"] > 0.7 and features["eyes_wide"] > 0.6:
            return ("😱", "Screaming")
        elif features["eyes_wide"] > 0.5 and features["eyebrows_raised"] > 0.5:
            return ("🤯", "Mind Blown")
        elif features["mouth_open"] > 0.5:
            return ("😮", "Wow")
        else:
            return ("😲", "Astonished")
    
    elif base_emotion == "fear":
        if features["mouth_open"] > 0.5:
            return ("😰", "Anxious")
        elif features["eyes_wide"] > 0.5:
            return ("😨", "Fearful")
        else:
            return ("😥", "Cold Sweat")
    
    elif base_emotion == "disgust":
        if features["asymmetric_smile"] > 0.3:
            return ("🤨", "Skeptical")
        elif features["mouth_open"] > 0.3:
            return ("🤮", "Vomiting")
        else:
            return ("🤢", "Nauseated")
    
    elif base_emotion == "neutral":
        if features["eyebrows_raised"] > 0.4 and features["looking_away"]:
            return ("🙄", "Rolling Eyes")
        elif features["eyebrows_raised"] > 0.3:
            return ("🤔", "Thinking")
        elif features["eyes_closed"] > 0.6:
            return ("😴", "Sleeping")
        elif features["asymmetric_smile"] > 0.3:
            return ("😒", "Unamused")
        elif abs(features["head_tilt"]) > 0.1:
            return ("😕", "Confused")
        else:
            return ("😐", "Neutral")
    
    elif base_emotion == "contempt":
        if features["asymmetric_smile"] > 0.4:
            return ("😏", "Smug")
        elif features["eyebrows_raised"] > 0.3:
            return ("🤨", "Judging")
        else:
            return ("😒", "Sarcastic")
    
    # Default fallback
    default_emojis = {
        "angry": ("😠", "Angry"),
        "disgust": ("🤢", "Disgusted"),
        "fear": ("😨", "Fearful"),
        "happy": ("😃", "Happy"),
        "sad": ("😢", "Sad"),
        "surprise": ("😲", "Surprised"),
        "neutral": ("😐", "Neutral"),
        "contempt": ("😏", "Contempt")
    }
    return default_emojis.get(base_emotion, ("❓", "Unknown"))


# ============================================
# MAIN APPLICATION
# ============================================

# Load the CNN Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FaceEmotionCNN(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, input_size=INPUT_SIZE).to(device)

try:
    checkpoint = torch.load('emotion_model_best.pth', map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ CNN Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
    
model.eval()

# Initialize facial feature analyzer
print("Initializing MediaPipe Face Mesh...")
feature_analyzer = FacialFeatureAnalyzer()
print("✓ Face Mesh initialized!")

# Image preprocessing for AffectNet (RGB 75x75)
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Temporal smoothing
SMOOTHING_WINDOW = 5
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

def get_smoothed_prediction(current_probs):
    prediction_history.append(current_probs.cpu().numpy())
    if len(prediction_history) < 2:
        return current_probs
    weights = np.linspace(0.5, 1.0, len(prediction_history))
    weights = weights / weights.sum()
    smoothed = np.zeros(NUM_CLASSES)
    for i, probs in enumerate(prediction_history):
        smoothed += weights[i] * probs
    return torch.tensor(smoothed).to(device)

# Setup Webcam
cap = None
for camera_index in [0, 1, 2]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✓ Camera found at index {camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        break
    cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit(1)

# Face detection
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

current_emoji = ""
show_features = False
show_all_emotions = False

def draw_emoji_on_frame(frame, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("seguiemj.ttf", 80)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(255, 255, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_feature_bars(frame, features, x, y):
    """Affiche les features faciales détectées"""
    bar_height = 12
    bar_width = 100
    start_y = y
    
    feature_names = ["mouth_open", "smile_intensity", "eyebrows_raised", 
                     "eyes_wide", "eyes_closed", "asymmetric_smile"]
    
    for i, feat in enumerate(feature_names):
        bar_y = start_y + i * (bar_height + 3)
        value = features.get(feat, 0)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        
        # Background
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Fill
        fill_width = int(bar_width * min(1.0, value))
        color = (0, 255, 0) if value > 0.5 else (255, 255, 0) if value > 0.25 else (100, 100, 100)
        cv2.rectangle(frame, (x, bar_y), (x + fill_width, bar_y + bar_height), color, -1)
        
        # Label
        cv2.putText(frame, f"{feat}: {value:.2f}", (x + bar_width + 5, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

def draw_emotion_bars(frame, probabilities, x, y, w):
    """Affiche les probabilités de chaque émotion"""
    bar_height = 15
    bar_width = 150
    start_y = y
    
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]
    colors = [(0,0,255), (0,128,0), (128,0,128), (0,255,255), (255,0,0), (0,165,255), (128,128,128), (139,69,19)]
    
    for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
        bar_y = start_y + i * (bar_height + 5)
        cv2.rectangle(frame, (x + w + 10, bar_y), (x + w + 10 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        fill_width = int(bar_width * prob)
        cv2.rectangle(frame, (x + w + 10, bar_y), (x + w + 10 + fill_width, bar_y + bar_height), colors[i], -1)
        cv2.putText(frame, f"{emotion}: {prob*100:.1f}%", (x + w + 15, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

print("\n" + "="*60)
print("    Extended Emoji Recognition - Enhanced Panel")
print("="*60)
print("Controls:")
print("  'q' - Quit")
print("  's' - Save emoji to clipboard")
print("  'e' - Toggle emotion bars")
print("  'f' - Toggle facial features display")
print("="*60)
print("\nEmoji variations based on:")
print("  • Base emotion (CNN)")
print("  • Smile intensity & asymmetry (smirk)")
print("  • Eye openness (wide, closed, wink)")
print("  • Eyebrow position (raised, furrowed)")
print("  • Mouth openness")
print("="*60 + "\n")

# Liste des émojis disponibles
print("Available emojis:")
all_emojis = sorted(set([v[0] for v in EXTENDED_EMOJI_MAP.values()]))
print("  " + " ".join(all_emojis))
print()

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Analyze facial features for the whole frame
        features = feature_analyzer.analyze(frame_rgb)

        for (x, y, w, h) in faces:
            margin = int(0.1 * min(w, h))
            y1 = max(0, y - margin)
            y2 = min(gray_frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(gray_frame.shape[1], x + w + margin)
            
            # Extract RGB ROI for the model (AffectNet uses RGB)
            roi_rgb = frame_rgb[y1:y2, x1:x2]
            
            try:
                # Preprocess for model (RGB 75x75)
                roi_tensor = data_transform(roi_rgb)
                roi_tensor = roi_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(roi_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    smoothed_probs = get_smoothed_prediction(probabilities)
                    max_prob, predicted_idx = torch.max(smoothed_probs, 0)
                    
                    idx = int(predicted_idx.item())
                    confidence = max_prob.item() * 100
                    base_emotion = BASE_EMOTIONS[idx]
                    
                    # Get extended emoji based on features
                    emoji, description = get_extended_emoji(base_emotion, features, confidence)
                    current_emoji = emoji
                    color = EMOTION_COLORS[base_emotion]

                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Label with extended description
                    label = f"{description}: {confidence:.1f}%"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y - 25), (x + text_w + 10, y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw emoji
                    frame = draw_emoji_on_frame(frame, emoji, (x - 90, y))
                    
                    # Optional displays
                    if show_all_emotions:
                        draw_emotion_bars(frame, smoothed_probs.cpu().numpy(), x, y, w)
                    
                    if show_features:
                        draw_feature_bars(frame, features, 10, frame.shape[0] - 120)
                    
            except Exception as e:
                print(f"Error: {e}")

        # Help text
        cv2.putText(frame, "Press 'e' for emotions, 'f' for features", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Extended Emoji Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_emoji:
                pyperclip.copy(current_emoji)
                print(f"Copied {current_emoji} to clipboard!")
        elif key == ord('e'):
            show_all_emotions = not show_all_emotions
            print(f"Emotion bars: {'ON' if show_all_emotions else 'OFF'}")
        elif key == ord('f'):
            show_features = not show_features
            print(f"Feature bars: {'ON' if show_features else 'OFF'}")

finally:
    feature_analyzer.close()
    cap.release()
    cv2.destroyAllWindows()
