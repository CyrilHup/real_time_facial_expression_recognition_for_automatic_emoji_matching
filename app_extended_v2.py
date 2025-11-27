"""
Extended Emotion Recognition V2
- CNN emotion detection with adjusted thresholds for difficult emotions
- Facial landmark analysis for nuanced expressions
- Hand gesture detection for additional emojis
- Combination of face + hands for composite expressions
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pyperclip
from collections import deque
from model import FaceEmotionCNN, create_model
import mediapipe as mp

# ============================================
# CONFIGURATION
# ============================================
NUM_CLASSES = 8
IMG_SIZE = 75  # AffectNet uses 75x75 RGB images

BASE_EMOTIONS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy", 
    4: "sad", 5: "surprise", 6: "neutral", 7: "contempt"
}

# Boost pour les émotions difficiles à détecter
# Ces émotions ont souvent des scores plus bas car moins représentées dans le dataset
EMOTION_BOOST = {
    "angry": 1.4,      # Boost significatif
    "disgust": 1.5,    # Le plus difficile - gros boost
    "fear": 1.3,       # Boost modéré
    "happy": 1.0,      # Pas de boost
    "sad": 1.0,        # Pas de boost
    "surprise": 1.0,   # Pas de boost
    "neutral": 0.85,   # Légère réduction (trop souvent prédit)
    "contempt": 1.4,   # Boost significatif
}

# Seuils de confiance minimum par émotion (plus bas = plus facile à déclencher)
EMOTION_THRESHOLDS = {
    "angry": 0.15,
    "disgust": 0.12,
    "fear": 0.15,
    "happy": 0.25,
    "sad": 0.20,
    "surprise": 0.20,
    "neutral": 0.30,
    "contempt": 0.12,
}

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
# GESTES DE MAIN -> EMOJIS
# ============================================
HAND_GESTURES = {
    "thumbs_up": ("👍", "Thumbs Up"),
    "thumbs_down": ("👎", "Thumbs Down"),
    "peace": ("✌️", "Peace"),
    "ok": ("👌", "OK"),
    "rock": ("🤘", "Rock"),
    "call_me": ("🤙", "Call Me"),
    "wave": ("👋", "Wave"),
    "fist": ("✊", "Fist"),
    "open_hand": ("🖐️", "High Five"),
    "point_up": ("☝️", "Point Up"),
    "pray": ("🙏", "Pray/Thanks"),
    "clap": ("👏", "Clap"),
    "heart_hands": ("🫶", "Heart Hands"),
    "middle_finger": ("🖕", "Middle Finger"),
    "love_you": ("🤟", "Love You"),
}

# Combinaisons visage + mains
FACE_HAND_COMBOS = {
    ("happy", "thumbs_up"): ("😊👍", "Happy Thumbs Up"),
    ("happy", "peace"): ("😄✌️", "Happy Peace"),
    ("happy", "ok"): ("😉👌", "Perfect"),
    ("sad", "thumbs_down"): ("😢👎", "Sad Thumbs Down"),
    ("angry", "fist"): ("😠✊", "Angry Fist"),
    ("surprise", "open_hand"): ("😲🖐️", "Shocked"),
    ("neutral", "wave"): ("👋", "Wave"),
    ("happy", "rock"): ("🤘😎", "Rock On"),
    ("neutral", "pray"): ("🙏", "Thanks"),
    ("happy", "heart_hands"): ("🥰🫶", "Love"),
    ("sad", "pray"): ("🥺🙏", "Please"),
    ("angry", "middle_finger"): ("🤬🖕", "Rage"),
    ("happy", "love_you"): ("😍🤟", "Love You"),
}

# ============================================
# MEDIAPIPE SETUP
# ============================================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmarks du visage
FACE_LANDMARKS = {
    "left_eye_top": 159, "left_eye_bottom": 145,
    "left_eye_inner": 133, "left_eye_outer": 33,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "left_eyebrow_inner": 107, "left_eyebrow_outer": 70, "left_eyebrow_top": 105,
    "right_eyebrow_inner": 336, "right_eyebrow_outer": 300, "right_eyebrow_top": 334,
    "mouth_top": 13, "mouth_bottom": 14,
    "mouth_left": 61, "mouth_right": 291,
    "upper_lip_top": 0, "lower_lip_bottom": 17,
    "nose_tip": 4, "chin": 152, "forehead": 10,
    # Pour détecter le plissement du nez (disgust)
    "nose_left": 48, "nose_right": 278,
    # Coins de la bouche pour contempt
    "mouth_corner_left": 61, "mouth_corner_right": 291,
}


class HandGestureRecognizer:
    """Reconnaissance des gestes de la main avec MediaPipe"""
    
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.gesture_history = deque(maxlen=5)
    
    def _get_finger_states(self, hand_landmarks, handedness):
        """Détermine quels doigts sont levés"""
        landmarks = hand_landmarks.landmark
        
        # Indices des bouts de doigts et articulations
        FINGER_TIPS = [4, 8, 12, 16, 20]  # Pouce, Index, Majeur, Annulaire, Auriculaire
        FINGER_PIPS = [3, 6, 10, 14, 18]  # Articulations intermédiaires
        
        fingers_up = []
        
        # Pouce (logique différente selon la main)
        is_right = handedness == "Right"
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        if is_right:
            fingers_up.append(thumb_tip.x < thumb_ip.x)
        else:
            fingers_up.append(thumb_tip.x > thumb_ip.x)
        
        # Autres doigts (index, majeur, annulaire, auriculaire)
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            fingers_up.append(landmarks[tip].y < landmarks[pip].y)
        
        return fingers_up
    
    def _classify_gesture(self, fingers_up, hand_landmarks, handedness):
        """Classifie le geste basé sur les doigts levés"""
        landmarks = hand_landmarks.landmark
        
        thumb, index, middle, ring, pinky = fingers_up
        
        # Compter les doigts levés
        count = sum(fingers_up)
        
        # === GESTES SPÉCIFIQUES ===
        
        # Pouce en l'air (seulement le pouce)
        if thumb and not index and not middle and not ring and not pinky:
            # Vérifier si le pouce pointe vers le haut ou le bas
            if landmarks[4].y < landmarks[2].y:
                return "thumbs_up"
            else:
                return "thumbs_down"
        
        # Peace / Victory (index + majeur)
        if not thumb and index and middle and not ring and not pinky:
            return "peace"
        
        # OK sign (pouce + index formant un cercle, autres levés)
        if thumb and index:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            if distance < 0.05 and middle and ring and pinky:
                return "ok"
        
        # Rock / Metal (index + auriculaire)
        if not thumb and index and not middle and not ring and pinky:
            return "rock"
        
        # Call me / Shaka (pouce + auriculaire)
        if thumb and not index and not middle and not ring and pinky:
            return "call_me"
        
        # Love you sign (pouce + index + auriculaire)
        if thumb and index and not middle and not ring and pinky:
            return "love_you"
        
        # Point up (seulement index)
        if not thumb and index and not middle and not ring and not pinky:
            return "point_up"
        
        # Main ouverte (tous les doigts)
        if count >= 4:
            return "open_hand"
        
        # Poing fermé
        if count == 0:
            return "fist"
        
        # Majeur seul
        if not thumb and not index and middle and not ring and not pinky:
            return "middle_finger"
        
        return None
    
    def detect(self, frame_rgb):
        """Détecte les gestes des mains"""
        results = self.hands.process(frame_rgb)
        
        gestures = []
        hand_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                handedness = handedness_info.classification[0].label
                fingers_up = self._get_finger_states(hand_landmarks, handedness)
                gesture = self._classify_gesture(fingers_up, hand_landmarks, handedness)
                
                if gesture:
                    gestures.append(gesture)
                
                hand_data.append({
                    "landmarks": hand_landmarks,
                    "handedness": handedness,
                    "gesture": gesture,
                    "fingers_up": fingers_up
                })
        
        # Détecter les gestes à deux mains
        if len(hand_data) == 2:
            # Pray / Namaste (deux mains ensemble)
            left_palm = None
            right_palm = None
            for hd in hand_data:
                palm_center = hd["landmarks"].landmark[9]  # Centre de la paume
                if hd["handedness"] == "Left":
                    left_palm = palm_center
                else:
                    right_palm = palm_center
            
            if left_palm and right_palm:
                distance = np.sqrt(
                    (left_palm.x - right_palm.x)**2 + 
                    (left_palm.y - right_palm.y)**2
                )
                if distance < 0.15:
                    # Mains proches = pray ou clap
                    gestures = ["pray"]
        
        # Lissage temporel
        self.gesture_history.append(gestures)
        
        # Retourner le geste le plus fréquent
        if self.gesture_history:
            all_gestures = [g for sublist in self.gesture_history for g in sublist]
            if all_gestures:
                from collections import Counter
                most_common = Counter(all_gestures).most_common(1)
                if most_common and most_common[0][1] >= 2:
                    return most_common[0][0], hand_data
        
        return gestures[0] if gestures else None, hand_data
    
    def close(self):
        self.hands.close()


class FacialFeatureAnalyzer:
    """Analyse les features faciales avec focus sur les émotions difficiles"""
    
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=5)
        
    def analyze(self, frame_rgb):
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
            # Nouvelles features pour émotions difficiles
            "nose_wrinkle": 0.0,      # Pour disgust
            "lip_corner_down": 0.0,    # Pour sad/angry
            "lip_corner_asymmetric": 0.0,  # Pour contempt
            "brow_squeeze": 0.0,       # Pour angry/fear
            "upper_lip_raise": 0.0,    # Pour disgust/fear
        }
        
        if not results.multi_face_landmarks:
            return features
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_rgb.shape[:2]
        
        def get_point(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
        def distance(p1, p2):
            return np.linalg.norm(get_point(p1) - get_point(p2))
        
        face_height = distance(FACE_LANDMARKS["forehead"], FACE_LANDMARKS["chin"])
        if face_height < 1:
            return features
        
        # === Standard features ===
        mouth_height = distance(FACE_LANDMARKS["mouth_top"], FACE_LANDMARKS["mouth_bottom"])
        mouth_width = distance(FACE_LANDMARKS["mouth_left"], FACE_LANDMARKS["mouth_right"])
        
        features["mouth_open"] = min(1.0, mouth_height / (face_height * 0.15))
        
        if mouth_height > 0:
            smile_ratio = mouth_width / mouth_height
            features["smile_intensity"] = min(1.0, max(0, (smile_ratio - 2.0) / 3.0))
        
        # Asymétrie du sourire (contempt)
        mouth_left = get_point(FACE_LANDMARKS["mouth_left"])
        mouth_right = get_point(FACE_LANDMARKS["mouth_right"])
        mouth_center = (mouth_left + mouth_right) / 2
        left_corner_height = mouth_left[1] - mouth_center[1]
        right_corner_height = mouth_right[1] - mouth_center[1]
        
        features["lip_corner_asymmetric"] = abs(left_corner_height - right_corner_height) / (face_height * 0.1)
        features["asymmetric_smile"] = features["lip_corner_asymmetric"]
        
        # Coins de la bouche vers le bas (sad/angry)
        neutral_mouth_y = get_point(FACE_LANDMARKS["nose_tip"])[1] + face_height * 0.25
        avg_corner_y = (mouth_left[1] + mouth_right[1]) / 2
        features["lip_corner_down"] = max(0, (avg_corner_y - neutral_mouth_y) / (face_height * 0.1))
        
        # === Yeux ===
        left_eye_height = distance(FACE_LANDMARKS["left_eye_top"], FACE_LANDMARKS["left_eye_bottom"])
        right_eye_height = distance(FACE_LANDMARKS["right_eye_top"], FACE_LANDMARKS["right_eye_bottom"])
        left_eye_width = distance(FACE_LANDMARKS["left_eye_inner"], FACE_LANDMARKS["left_eye_outer"])
        right_eye_width = distance(FACE_LANDMARKS["right_eye_inner"], FACE_LANDMARKS["right_eye_outer"])
        
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        if avg_eye_width > 0:
            eye_aspect_ratio = avg_eye_height / avg_eye_width
            features["eyes_wide"] = min(1.0, max(0, (eye_aspect_ratio - 0.25) / 0.15))
            features["eyes_closed"] = max(0, (0.2 - eye_aspect_ratio) / 0.15)
        
        if left_eye_width > 0:
            features["left_eye_closed"] = (left_eye_height / left_eye_width) < 0.15
        if right_eye_width > 0:
            features["right_eye_closed"] = (right_eye_height / right_eye_width) < 0.15
        
        # === Sourcils ===
        left_brow_height = distance(FACE_LANDMARKS["left_eyebrow_top"], FACE_LANDMARKS["left_eye_top"])
        right_brow_height = distance(FACE_LANDMARKS["right_eyebrow_top"], FACE_LANDMARKS["right_eye_top"])
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        
        brow_normalized = avg_brow_height / (face_height * 0.08)
        features["eyebrows_raised"] = min(1.0, max(0, (brow_normalized - 1.0) / 0.5))
        features["eyebrows_furrowed"] = min(1.0, max(0, (1.0 - brow_normalized) / 0.3))
        
        # Rapprochement des sourcils (angry/fear)
        left_brow_inner = get_point(FACE_LANDMARKS["left_eyebrow_inner"])
        right_brow_inner = get_point(FACE_LANDMARKS["right_eyebrow_inner"])
        brow_distance = np.linalg.norm(left_brow_inner - right_brow_inner)
        features["brow_squeeze"] = max(0, 1.0 - brow_distance / (face_height * 0.25))
        
        # === Nez (disgust) ===
        nose_width = distance(FACE_LANDMARKS["nose_left"], FACE_LANDMARKS["nose_right"])
        nose_base_width = face_height * 0.15  # Largeur normale approximative
        features["nose_wrinkle"] = max(0, (nose_width - nose_base_width) / nose_base_width)
        
        # Lèvre supérieure relevée (disgust/fear)
        upper_lip = get_point(FACE_LANDMARKS["upper_lip_top"])
        nose_bottom = get_point(FACE_LANDMARKS["nose_tip"])
        lip_nose_dist = nose_bottom[1] - upper_lip[1]
        features["upper_lip_raise"] = max(0, min(1.0, lip_nose_dist / (face_height * 0.08)))
        
        # === Tête ===
        left_face = get_point(FACE_LANDMARKS["left_eye_outer"])
        right_face = get_point(FACE_LANDMARKS["right_eye_outer"])
        features["head_tilt"] = (left_face[1] - right_face[1]) / face_height
        
        nose = get_point(FACE_LANDMARKS["nose_tip"])
        face_center_x = (left_face[0] + right_face[0]) / 2
        face_width = abs(left_face[0] - right_face[0])
        if face_width > 0:
            features["looking_away"] = abs(nose[0] - face_center_x) / face_width > 0.15
        
        # Lissage
        self.history.append(features)
        if len(self.history) >= 2:
            smoothed = {}
            for key in features:
                if isinstance(features[key], bool):
                    smoothed[key] = sum(1 for h in self.history if h.get(key, False)) > len(self.history) / 2
                else:
                    smoothed[key] = np.mean([h.get(key, 0) for h in self.history])
            return smoothed
            
        return features
    
    def close(self):
        self.face_mesh.close()


def adjust_emotion_scores(probabilities):
    """Ajuste les scores d'émotion pour les émotions difficiles"""
    adjusted = probabilities.clone()
    
    for idx, emotion in BASE_EMOTIONS.items():
        boost = EMOTION_BOOST.get(emotion, 1.0)
        adjusted[idx] = adjusted[idx] * boost
    
    # Re-normaliser
    adjusted = adjusted / adjusted.sum()
    
    return adjusted


def get_emotion_with_features(probabilities, features):
    """Utilise les features faciales pour aider à décider l'émotion"""
    
    adjusted = probabilities.clone()
    
    # Boost basé sur les features faciales
    
    # Disgust: nez plissé, lèvre supérieure levée, sourcils froncés
    if features["nose_wrinkle"] > 0.2 or features["upper_lip_raise"] > 0.5:
        adjusted[1] *= 1.5  # Boost disgust
    
    # Angry: sourcils froncés et rapprochés, coins bouche vers le bas
    if features["brow_squeeze"] > 0.4 and features["eyebrows_furrowed"] > 0.3:
        adjusted[0] *= 1.4  # Boost angry
    
    # Fear: yeux grand ouverts, sourcils levés mais rapprochés
    if features["eyes_wide"] > 0.5 and features["eyebrows_raised"] > 0.3:
        adjusted[2] *= 1.3  # Boost fear
    
    # Contempt: sourire asymétrique
    if features["lip_corner_asymmetric"] > 0.3:
        adjusted[7] *= 1.5  # Boost contempt
    
    # Réduire neutral si d'autres signaux sont présents
    if features["brow_squeeze"] > 0.3 or features["lip_corner_down"] > 0.3:
        adjusted[6] *= 0.7  # Reduce neutral
    
    # Re-normaliser
    adjusted = adjusted / adjusted.sum()
    
    return adjusted


def get_extended_emoji(base_emotion, features, confidence):
    """Détermine l'émoji étendu basé sur l'émotion et les features"""
    
    # Wink
    if features["left_eye_closed"] != features["right_eye_closed"]:
        if base_emotion in ["happy", "neutral"]:
            return ("😉", "Winking")
    
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
        elif features["lip_corner_down"] > 0.4:
            return ("😞", "Disappointed")
        else:
            return ("😢", "Sad")
    
    elif base_emotion == "angry":
        if features["mouth_open"] > 0.5:
            return ("🤬", "Rage")
        elif features["brow_squeeze"] > 0.5:
            return ("😡", "Furious")
        elif features["eyebrows_furrowed"] > 0.4:
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
        if features["mouth_open"] > 0.5 and features["eyes_wide"] > 0.5:
            return ("😱", "Terrified")
        elif features["brow_squeeze"] > 0.3:
            return ("😰", "Anxious")
        elif features["eyes_wide"] > 0.4:
            return ("😨", "Fearful")
        else:
            return ("😥", "Worried")
    
    elif base_emotion == "disgust":
        if features["upper_lip_raise"] > 0.5:
            return ("🤮", "Vomiting")
        elif features["nose_wrinkle"] > 0.3:
            return ("🤢", "Nauseated")
        elif features["asymmetric_smile"] > 0.3:
            return ("🤨", "Skeptical")
        else:
            return ("😖", "Disgusted")
    
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
        if features["lip_corner_asymmetric"] > 0.5:
            return ("😏", "Smug")
        elif features["eyebrows_raised"] > 0.3:
            return ("🤨", "Judging")
        else:
            return ("😒", "Contempt")
    
    return ("❓", "Unknown")


# ============================================
# MAIN APPLICATION
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Use create_model for RGB 75x75 input (AffectNet)
model = create_model(num_classes=NUM_CLASSES, dataset='affectnet').to(device)

try:
    checkpoint = torch.load('emotion_model.pth', map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✓ CNN Model loaded (RGB 75x75)!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
    
model.eval()

# Initialize analyzers
print("Initializing MediaPipe...")
feature_analyzer = FacialFeatureAnalyzer()
hand_recognizer = HandGestureRecognizer()
print("✓ Face Mesh + Hands initialized!")

# Image preprocessing for RGB 75x75 (AffectNet format)
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    ),
])

# Temporal smoothing
SMOOTHING_WINDOW = 3  # Réduit pour plus de réactivité
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

haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

current_emoji = ""
show_features = False
show_all_emotions = False
show_hands = True

def draw_emoji_on_frame(frame, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("seguiemj.ttf", 80)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(255, 255, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_small_emoji(frame, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("seguiemj.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(255, 255, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_feature_bars(frame, features, x, y):
    bar_height = 12
    bar_width = 100
    start_y = y
    
    feature_names = [
        "mouth_open", "smile_intensity", "eyebrows_raised", 
        "brow_squeeze", "eyes_wide", "asymmetric_smile",
        "nose_wrinkle", "lip_corner_down", "upper_lip_raise"
    ]
    
    for i, feat in enumerate(feature_names):
        bar_y = start_y + i * (bar_height + 3)
        value = features.get(feat, 0)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        fill_width = int(bar_width * min(1.0, value))
        color = (0, 255, 0) if value > 0.5 else (255, 255, 0) if value > 0.25 else (100, 100, 100)
        cv2.rectangle(frame, (x, bar_y), (x + fill_width, bar_y + bar_height), color, -1)
        cv2.putText(frame, f"{feat}: {value:.2f}", (x + bar_width + 5, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def draw_emotion_bars(frame, probabilities, x, y, w):
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
print("   Extended Emoji Recognition V2 - Face + Hands")
print("="*60)
print("Controls:")
print("  'q' - Quit")
print("  's' - Save emoji to clipboard")
print("  'e' - Toggle emotion bars")
print("  'f' - Toggle facial features display")
print("  'h' - Toggle hand tracking visualization")
print("="*60)
print("\n🖐️ Hand Gestures Supported:")
print("  👍 Thumbs Up    👎 Thumbs Down   ✌️ Peace")
print("  👌 OK           🤘 Rock          🤙 Call Me")
print("  🤟 Love You     ☝️ Point Up      🖐️ High Five")
print("  ✊ Fist         🙏 Pray          🖕 Middle Finger")
print("="*60)
print("\n😤 Tips for difficult emotions:")
print("  ANGRY: Frown hard, squeeze eyebrows together")
print("  DISGUST: Wrinkle nose, raise upper lip")
print("  FEAR: Open eyes wide, raise eyebrows")
print("  CONTEMPT: One-sided smirk")
print("="*60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect hands
        hand_gesture, hand_data = hand_recognizer.detect(frame_rgb)
        
        # Draw hand landmarks
        if show_hands and hand_data:
            for hd in hand_data:
                mp_drawing.draw_landmarks(
                    frame,
                    hd["landmarks"],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Show hand gesture emoji
        hand_emoji = ""
        if hand_gesture and hand_gesture in HAND_GESTURES:
            hand_emoji, hand_desc = HAND_GESTURES[hand_gesture]
            # Draw hand emoji in corner
            frame = draw_small_emoji(frame, hand_emoji, (frame.shape[1] - 60, 10))
            cv2.putText(frame, hand_desc, (frame.shape[1] - 120, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Analyze facial features
        features = feature_analyzer.analyze(frame_rgb)
        
        face_emoji = ""
        base_emotion = "neutral"

        for (x, y, w, h) in faces:
            margin = int(0.1 * min(w, h))
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)
            
            # Use RGB image for the model (AffectNet format)
            roi_rgb = frame_rgb[y1:y2, x1:x2]
            
            try:
                roi_tensor = data_transform(roi_rgb)
                roi_tensor = roi_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(roi_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Ajuster les scores
                    adjusted_probs = adjust_emotion_scores(probabilities)
                    
                    # Utiliser les features pour affiner
                    adjusted_probs = get_emotion_with_features(adjusted_probs, features)
                    
                    # Lisser
                    smoothed_probs = get_smoothed_prediction(adjusted_probs)
                    max_prob, predicted_idx = torch.max(smoothed_probs, 0)
                    
                    idx = int(predicted_idx.item())
                    confidence = max_prob.item() * 100
                    base_emotion = BASE_EMOTIONS[idx]
                    
                    # Get extended emoji
                    face_emoji, description = get_extended_emoji(base_emotion, features, confidence)
                    color = EMOTION_COLORS[base_emotion]

                    # Check for face + hand combo
                    combo_emoji = None
                    if hand_gesture and (base_emotion, hand_gesture) in FACE_HAND_COMBOS:
                        combo_emoji, combo_desc = FACE_HAND_COMBOS[(base_emotion, hand_gesture)]
                        current_emoji = combo_emoji
                        description = combo_desc
                    else:
                        current_emoji = face_emoji + (hand_emoji if hand_gesture else "")

                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Label
                    label = f"{description}: {confidence:.1f}%"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y - 25), (x + text_w + 10, y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw emoji
                    if combo_emoji:
                        frame = draw_emoji_on_frame(frame, combo_emoji, (x - 90, y))
                    else:
                        frame = draw_emoji_on_frame(frame, face_emoji, (x - 90, y))
                    
                    # Optional displays
                    if show_all_emotions:
                        draw_emotion_bars(frame, smoothed_probs.cpu().numpy(), x, y, w)
                    
                    if show_features:
                        draw_feature_bars(frame, features, 10, frame.shape[0] - 150)
                    
            except Exception as e:
                print(f"Error: {e}")

        # Help text
        cv2.putText(frame, "'e'=emotions 'f'=features 'h'=hands 's'=copy", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow('Extended Emoji Recognition V2', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_emoji:
                pyperclip.copy(current_emoji)
                print(f"Copied {current_emoji} to clipboard!")
        elif key == ord('e'):
            show_all_emotions = not show_all_emotions
        elif key == ord('f'):
            show_features = not show_features
        elif key == ord('h'):
            show_hands = not show_hands

finally:
    feature_analyzer.close()
    hand_recognizer.close()
    cap.release()
    cv2.destroyAllWindows()
