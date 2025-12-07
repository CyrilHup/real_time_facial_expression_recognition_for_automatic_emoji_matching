"""
Model Comparison Script - Compare FER2013, FER+, and AffectNet models
======================================================================
Loads all available emotion models and compares their performance on a test set.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import time
from torchvision import transforms
from PIL import Image

from model import FaceEmotionCNN, create_model


# ============================================
# MODEL CONFIGURATIONS
# ============================================
MODEL_CONFIGS = {
    'FER2013': {
        'num_classes': 7,
        'in_channels': 1,
        'img_size': 48,
        'dataset': 'fer2013',
        'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    },
    'FER+': {
        'num_classes': 8,
        'in_channels': 1,
        'img_size': 48,
        'dataset': 'ferplus',
        'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    },
    'AffectNet': {
        'num_classes': 8,
        'in_channels': 3,
        'img_size': 75,
        'dataset': 'affectnet',
        'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
    }
}


# ============================================
# MODEL LOADER
# ============================================
class ModelComparator:
    """Compare multiple emotion recognition models"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.transforms = {}
        self.configs = {}
        
    def load_model(self, model_path: str, config_name: str):
        """Load a model with its configuration"""
        if config_name not in MODEL_CONFIGS:
            print(f"Unknown config: {config_name}")
            return False
        
        config = MODEL_CONFIGS[config_name]
        
        try:
            # Create model
            model = create_model(
                dataset=config['dataset'],
                num_classes=config['num_classes']
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            # Create transforms
            if config['in_channels'] == 3:
                # RGB with normalization
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((config['img_size'], config['img_size'])),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                # Grayscale
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((config['img_size'], config['img_size'])),
                    transforms.ToTensor(),
                ])
            
            self.models[config_name] = model
            self.transforms[config_name] = transform
            self.configs[config_name] = config
            
            print(f"✓ Loaded {config_name} model from {model_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {config_name}: {e}")
            return False
    
    def auto_load_models(self):
        """Automatically detect and load all available models"""
        print("\n" + "="*70)
        print("  AUTO-DETECTING MODELS")
        print("="*70)
        
        # Search for model files
        model_patterns = {
            'FER2013': ['fer2013_model.pth', 'emotion_model_fer2013.pth'],
            'FER+': ['emotion_model.pth', 'emotion_model_best.pth', 'ferplus_model.pth'],
            'AffectNet': ['affectnet_model.pth', 'emotion_model_affectnet.pth']
        }
        
        current_dir = Path('.')
        
        for config_name, patterns in model_patterns.items():
            for pattern in patterns:
                model_path = current_dir / pattern
                if model_path.exists():
                    print(f"\nFound: {pattern}")
                    # Try to detect dataset type from checkpoint
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # Check channels and classes
                        in_channels = state_dict['conv1a.weight'].shape[1]
                        num_classes = state_dict['fc3.weight'].shape[0]
                        
                        # Determine correct config
                        if in_channels == 3:
                            detected_config = 'AffectNet'
                        elif num_classes == 7:
                            detected_config = 'FER2013'
                        else:
                            detected_config = 'FER+'
                        
                        print(f"  Detected: {detected_config} (channels={in_channels}, classes={num_classes})")
                        self.load_model(str(model_path), detected_config)
                        break  # Only load one model per config
                        
                    except Exception as e:
                        print(f"  Error detecting: {e}")
        
        if not self.models:
            print("\n✗ No models found!")
            print("\nExpected model files:")
            for config_name, patterns in model_patterns.items():
                print(f"  {config_name}: {', '.join(patterns)}")
        else:
            print(f"\n✓ Loaded {len(self.models)} model(s)")
    
    def preprocess_image(self, image: np.ndarray, config_name: str) -> torch.Tensor:
        """Preprocess image for specific model"""
        config = self.configs[config_name]
        
        # Convert to grayscale or RGB
        if config['in_channels'] == 1:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            image = clahe.apply(image)
        else:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Apply CLAHE per channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            channels = cv2.split(image)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            image = cv2.merge(clahe_channels)
        
        # Apply transforms
        tensor = self.transforms[config_name](image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_single(self, image: np.ndarray, config_name: str):
        """Predict with a single model"""
        tensor = self.preprocess_image(image, config_name)
        
        with torch.no_grad():
            start = time.perf_counter()
            output = self.models[config_name](tensor)
            inference_time = (time.perf_counter() - start) * 1000
            
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]
            emotion = self.configs[config_name]['emotions'][pred_idx]
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probs,
            'inference_time': inference_time
        }
    
    def compare_on_image(self, image: np.ndarray):
        """Compare all models on single image"""
        results = {}
        
        for config_name, model in self.models.items():
            results[config_name] = self.predict_single(image, config_name)
        
        return results
    
    def compare_webcam(self):
        """Real-time comparison on webcam feed"""
        print("\n" + "="*70)
        print("  WEBCAM COMPARISON MODE")
        print("="*70)
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Setup camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("✗ Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    margin = int(0.1 * min(w, h))
                    y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
                    x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # Compare models
                    results = self.compare_on_image(face_roi)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display results
                    y_offset = 30
                    for i, (model_name, result) in enumerate(results.items()):
                        emotion = result['emotion']
                        conf = result['confidence']
                        time_ms = result['inference_time']
                        
                        text = f"{model_name}: {emotion} ({conf*100:.0f}%) {time_ms:.1f}ms"
                        color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][i % 3]
                        
                        cv2.putText(frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y_offset += 25
                
                # Instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                           (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow('Model Comparison', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"comparison_{screenshot_count}.png"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Saved: {filename}")
                    screenshot_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def benchmark(self, num_iterations=100):
        """Benchmark inference speed of all models"""
        print("\n" + "="*70)
        print("  BENCHMARK MODE")
        print("="*70)
        
        # Create dummy inputs
        dummy_inputs = {}
        for config_name, config in self.configs.items():
            if config['in_channels'] == 1:
                dummy = np.random.randint(0, 255, (config['img_size'], config['img_size']), dtype=np.uint8)
            else:
                dummy = np.random.randint(0, 255, (config['img_size'], config['img_size'], 3), dtype=np.uint8)
            dummy_inputs[config_name] = dummy
        
        print(f"\nRunning {num_iterations} iterations per model...\n")
        
        for config_name, model in self.models.items():
            times = []
            dummy = dummy_inputs[config_name]
            
            # Warmup
            for _ in range(10):
                _ = self.predict_single(dummy, config_name)
            
            # Benchmark
            for _ in range(num_iterations):
                result = self.predict_single(dummy, config_name)
                times.append(result['inference_time'])
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"{config_name}:")
            print(f"  Avg: {avg_time:.2f}ms (±{std_time:.2f}ms)")
            print(f"  Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")
            print(f"  FPS: {1000/avg_time:.1f}")
            print()
    
    def print_summary(self):
        """Print summary of loaded models"""
        print("\n" + "="*70)
        print("  MODEL SUMMARY")
        print("="*70)
        
        for config_name, config in self.configs.items():
            model = self.models[config_name]
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"\n{config_name}:")
            print(f"  Dataset: {config['dataset']}")
            print(f"  Classes: {config['num_classes']} - {', '.join(config['emotions'])}")
            print(f"  Input: {config['in_channels']} channels × {config['img_size']}×{config['img_size']}")
            print(f"  Parameters: {total_params:,}")


# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "="*70)
    print("  EMOTION MODEL COMPARATOR")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize comparator
    comparator = ModelComparator(device=device)
    
    # Auto-detect and load models
    comparator.auto_load_models()
    
    if not comparator.models:
        print("\nNo models to compare!")
        return
    
    # Print summary
    comparator.print_summary()
    
    # Menu
    while True:
        print("\n" + "="*70)
        print("  MENU")
        print("="*70)
        print("1. Webcam comparison (real-time)")
        print("2. Benchmark speed")
        print("3. Exit")
        
        choice = input("\nSelect option [1-3]: ").strip()
        
        if choice == '1':
            comparator.compare_webcam()
        elif choice == '2':
            comparator.benchmark()
        elif choice == '3':
            print("\nExiting...")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
