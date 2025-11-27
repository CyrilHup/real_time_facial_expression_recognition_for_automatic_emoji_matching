import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pyperclip  # For clipboard support
from collections import deque
from model import FaceEmotionCNN

# Configuration: 8 émotions avec FER+ (inclut Contempt)
NUM_CLASSES = 8  # 7 émotions FER2013 + Contempt (FER+)

# 1. Load the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FaceEmotionCNN(num_classes=NUM_CLASSES).to(device)

# Charger les poids (supporte les deux formats)
try:
    checkpoint = torch.load('emotion_model.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying legacy model...")
    from model import FaceEmotionCNNLegacy
    model = FaceEmotionCNNLegacy().to(device)
    model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
    
model.eval()

# 2. Define Definitions (8 émotions avec FER+)
# Note: Contempt est la 8ème émotion ajoutée par FER+

emotion_dict = {
    0: "Angry 😠", 
    1: "Disgust 🤢", 
    2: "Fear 😨", 
    3: "Happy 😃", 
    4: "Sad 😢", 
    5: "Surprise 😲", 
    6: "Neutral 😐",
    7: "Contempt 😏"  # Mépris - ajouté par FER+
}

emoji_only = {
    0: "😠", 1: "🤢", 2: "😨", 3: "😃", 4: "😢", 5: "😲", 6: "😐", 7: "😏"
}

# Couleurs pour chaque émotion (BGR)
emotion_colors = {
    0: (0, 0, 255),     # Rouge - Angry
    1: (0, 128, 0),     # Vert foncé - Disgust
    2: (128, 0, 128),   # Violet - Fear
    3: (0, 255, 255),   # Jaune - Happy
    4: (255, 0, 0),     # Bleu - Sad
    5: (0, 165, 255),   # Orange - Surprise
    6: (128, 128, 128), # Gris - Neutral
    7: (139, 69, 19)    # Marron - Contempt
}

# Image Preprocessing amélioré (avec normalisation)
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# 3. Configuration du lissage temporel pour des prédictions plus stables
SMOOTHING_WINDOW = 5  # Nombre de frames pour lisser les prédictions
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
confidence_threshold = 0.3  # Seuil de confiance minimum

def get_smoothed_prediction(current_probs):
    """
    Lisse les prédictions sur plusieurs frames pour éviter les oscillations.
    Utilise une moyenne pondérée avec plus de poids sur les frames récentes.
    """
    prediction_history.append(current_probs.cpu().numpy())
    
    if len(prediction_history) < 2:
        return current_probs
    
    # Moyenne pondérée (frames récentes ont plus de poids)
    weights = np.linspace(0.5, 1.0, len(prediction_history))
    weights = weights / weights.sum()
    
    smoothed = np.zeros(NUM_CLASSES)  # 8 classes maintenant
    for i, probs in enumerate(prediction_history):
        smoothed += weights[i] * probs
    
    return torch.tensor(smoothed).to(device)

# 4. Setup Webcam and Face Detection
cap = None
for camera_index in [0, 1, 2]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera found at index {camera_index}")
        # Améliorer la qualité de capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        break
    cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open any camera. Please check your webcam connection.")
    exit(1)

# Détection de visage avec paramètres optimisés
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_emoji = ""
show_all_emotions = False  # Toggle pour afficher toutes les probabilités

def draw_emoji_on_frame(frame, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("seguiemj.ttf", 80) 
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(255, 255, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_emotion_bars(frame, probabilities, x, y, w):
    """Affiche des barres de progression pour chaque émotion."""
    bar_height = 15
    bar_width = 150
    start_y = y + 10
    
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]
    
    for i, (emotion, prob) in enumerate(zip(emotions, probabilities)):
        bar_y = start_y + i * (bar_height + 5)
        
        # Fond de la barre
        cv2.rectangle(frame, (x + w + 10, bar_y), 
                     (x + w + 10 + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Barre de progression
        fill_width = int(bar_width * prob)
        cv2.rectangle(frame, (x + w + 10, bar_y), 
                     (x + w + 10 + fill_width, bar_y + bar_height), 
                     emotion_colors[i], -1)
        
        # Texte
        cv2.putText(frame, f"{emotion}: {prob*100:.1f}%", 
                   (x + w + 15, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

print("\n" + "="*50)
print("Real-Time Facial Expression Recognition")
print("="*50)
print("Controls:")
print("  'q' - Quit")
print("  's' - Save emoji to clipboard")
print("  'e' - Toggle emotion bars display")
print("="*50 + "\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection de visage avec paramètres optimisés
    faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1,  # Plus précis (était 1.3)
        minNeighbors=5,
        minSize=(48, 48),  # Taille minimum du visage
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Crop face avec marge
        margin = int(0.1 * min(w, h))
        y1 = max(0, y - margin)
        y2 = min(gray_frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(gray_frame.shape[1], x + w + margin)
        
        roi_gray = gray_frame[y1:y2, x1:x2]
        
        try:
            # Prétraitement amélioré
            # Égalisation d'histogramme pour normaliser l'éclairage
            roi_gray = cv2.equalizeHist(roi_gray)
            
            # Prepare image for AI
            roi_tensor = data_transform(roi_gray).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(roi_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Appliquer le lissage temporel
                smoothed_probs = get_smoothed_prediction(probabilities)
                
                max_prob, predicted_idx = torch.max(smoothed_probs, 0)
                
                idx = predicted_idx.item()
                confidence = max_prob.item() * 100
                emotion_text = emotion_dict[idx]
                current_emoji = emoji_only[idx]
                color = emotion_colors[idx]

                # Dessiner le rectangle avec la couleur de l'émotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Overlay Text avec fond
                label = f"{emotion_text.split()[0]}: {confidence:.1f}%"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - 25), (x + text_w + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Overlay Emoji
                frame = draw_emoji_on_frame(frame, current_emoji, (x - 90, y))
                
                # Afficher les barres d'émotion si activé
                if show_all_emotions:
                    draw_emotion_bars(frame, smoothed_probs.cpu().numpy(), x, y, w)
                
        except Exception as e:
            print(f"Error: {e}")

    # Afficher FPS
    cv2.putText(frame, f"Press 'e' for emotion bars", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Real-Time Facial Expression Recognition', frame)

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

cap.release()
cv2.destroyAllWindows()