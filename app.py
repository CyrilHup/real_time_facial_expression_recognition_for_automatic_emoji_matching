import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pyperclip  # For clipboard support
from model import FaceEmotionCNN

# 1. Load the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceEmotionCNN().to(device)
model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
model.eval()

# 2. Define Definitions
emotion_dict = {
    0: "Angry 😠", 
    1: "Disgust 🤢", 
    2: "Fear 😨", 
    3: "Happy 😃", 
    4: "Sad 😢", 
    5: "Surprise 😲", 
    6: "Neutral 😐"
}

emoji_only = {
    0: "😠", 1: "🤢", 2: "😨", 3: "😃", 4: "😢", 5: "😲", 6: "😐"
}

# Image Preprocessing (Must match training)
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# 3. Setup Webcam and Face Detection
# Try different camera indices (0, 1, 2) to find the correct one
cap = None
for camera_index in [0, 1, 2]:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
    if cap.isOpened():
        print(f"Camera found at index {camera_index}")
        break
    cap.release()

if not cap or not cap.isOpened():
    print("Error: Could not open any camera. Please check your webcam connection.")
    exit(1)

# Load standard Haar Cascade for face detection (included in cv2)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_emoji = ""

def draw_emoji_on_frame(frame, text, position):
    # Convert CV2 frame (BGR) to PIL Image to draw emoji
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # You might need to provide a path to a font that supports Emojis, e.g., "seguiemj.ttf" on Windows
    # If this fails, it will just show squares.
    try:
        font = ImageFont.truetype("seguiemj.ttf", 80) 
    except:
        font = ImageFont.load_default() # Fallback

    draw.text(position, text, font=font, fill=(255, 255, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

print("Press 'q' to quit. Press 's' to save emoji to clipboard.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop face
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        try:
            # Prepare image for AI
            roi_tensor = data_transform(roi_gray).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(roi_tensor)
                # Get probabilities using Softmax
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, predicted_idx = torch.max(probabilities, 1)
                
                # Get Data
                idx = predicted_idx.item()
                confidence = max_prob.item() * 100
                emotion_text = emotion_dict[idx]
                current_emoji = emoji_only[idx]

                # Overlay Text (Match: 82%)
                label = f"Match: {confidence:.2f}%"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Overlay Emoji (Requires PIL trick)
                frame = draw_emoji_on_frame(frame, current_emoji, (x - 80, y))
                
        except Exception as e:
            print(f"Error: {e}")

    cv2.imshow('Real-Time Facial Expression Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if current_emoji:
            pyperclip.copy(current_emoji)
            print(f"Copied {current_emoji} to clipboard!")

cap.release()
cv2.destroyAllWindows()