import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
import os
import time

# Importer les datasets (amélioré si disponible, sinon original)
try:
    from dataset_improved import FERPlusDataset, FER2013Dataset
    IMPROVED_DATASET = True
    print("✓ Using improved dataset module")
except ImportError:
    from dataset import FER2013Dataset
    IMPROVED_DATASET = False
    print("Using original dataset module")

from model import FaceEmotionCNN

print("Imports successful!")

# Hyperparameters optimisés
BATCH_SIZE = 64
LEARNING_RATE = 0.0005  # Réduit de 0.001 à 0.0005 pour meilleure convergence
EPOCHS = 60  # Plus d'epochs avec early stopping
PATIENCE = 10  # Plus de patience pour laisser le modèle converger
VALIDATION_SPLIT = 0.15  # 15% pour validation
USE_FERPLUS = True  # Mettre False pour utiliser FER2013 standard avec 7 classes
NUM_CLASSES = 8 if USE_FERPLUS else 7  # 7 émotions FER2013 + Contempt si FER+

# Data Augmentation modérée - trop d'augmentation peut nuire
print("Setting up transforms with moderate data augmentation...")

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal aléatoire
    transforms.RandomRotation(5),  # Réduit de 10° à 5° - expressions faciales sensibles à la rotation
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.05, 0.05),  # Réduit de 0.1 à 0.05
        scale=(0.95, 1.05)  # Réduit de (0.9, 1.1)
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Réduit de 0.2
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

print("Loading dataset...")

# Vérifier si FER+ est disponible
ferplus_file = './data/fer2013new.csv'
fer_file = './data/fer2013.csv'

# Fonction pour calculer les poids de classe (intégrée pour éviter les bugs)
def compute_class_weights(labels, num_classes):
    """Calcule les poids inversement proportionnels à la fréquence des classes."""
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)  # Éviter division par zéro
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt'][:num_classes]
    print("\nClass distribution:")
    for i, (emotion, count, weight) in enumerate(zip(emotions, class_counts, weights)):
        print(f"  {emotion}: {count} samples, weight: {weight:.3f}")
    
    return torch.FloatTensor(weights)

if USE_FERPLUS and IMPROVED_DATASET and os.path.exists(ferplus_file):
    print("✓ Using FER+ dataset with corrected annotations!")
    # Charger FER+ pour train et validation séparément
    train_dataset_raw = FERPlusDataset(
        fer_file, './data/', 
        transform=None, 
        usage='Training',
        use_soft_labels=False,
        ignore_uncertain=True
    )
    val_dataset_raw = FERPlusDataset(
        fer_file, './data/',
        transform=None,
        usage='PublicTest',
        use_soft_labels=False,
        ignore_uncertain=True
    )
    
    # Extraire les labels pour calculer les poids
    print("\nExtracting labels for class balancing...")
    train_labels = []
    for i in range(len(train_dataset_raw)):
        _, label = train_dataset_raw[i]
        train_labels.append(label)
    train_labels = np.array(train_labels)
    
    class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    USE_BALANCED_SAMPLER = False  # Désactivé - les class weights suffisent et sont plus stables
    
else:
    print("Using FER2013 dataset (consider downloading FER+ for better results)")
    # Charger le dataset complet
    full_dataset = FER2013Dataset(fer_file, transform=None)
    print(f"Dataset loaded! Size: {len(full_dataset)} samples")
    
    # Séparer en train et validation
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset_raw, val_dataset_raw = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    criterion = nn.CrossEntropyLoss()
    USE_BALANCED_SAMPLER = False

print(f"Training samples: {len(train_dataset_raw)}, Validation samples: {len(val_dataset_raw)}")

# Wrapper pour appliquer les transforms appropriés
class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        pixels, label = self.subset[idx]
        if self.transform:
            pixels = self.transform(pixels)
        return pixels, label

train_dataset = TransformDataset(train_dataset_raw, train_transform)
val_dataset = TransformDataset(val_dataset_raw, val_transform)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

print("Creating DataLoaders...")

# Utiliser un sampler équilibré si nécessaire (désactivé par défaut)
if USE_BALANCED_SAMPLER:
    print("✓ Using balanced sampler to handle class imbalance")
    # Extraire les labels pour le sampler
    sampler_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        sampler_labels.append(label)
    sampler_labels = np.array(sampler_labels)
    
    class_counts = np.bincount(sampler_labels, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    sample_weights = list(weights[sampler_labels])  # Convert to list for type safety
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print("DataLoaders created!")

# Setup Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FaceEmotionCNN(num_classes=NUM_CLASSES).to(device)

# Criterion est défini plus haut (avec ou sans class weights)
if 'criterion' not in dir():
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning Rate Scheduler - OneCycleLR pour meilleure convergence
# Alternative: ReduceLROnPlateau si OneCycleLR ne fonctionne pas bien
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE * 10,  # LR max = 10x le LR initial
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)

# Compteur de paramètres
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Fonction de validation
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), 100 * correct / total

print("\n" + "="*60)
print("Starting Training with Early Stopping...")
print("="*60 + "\n")

best_val_loss = float('inf')
best_val_acc = 0.0
patience_counter = 0
best_epoch = 0

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # OneCycleLR step après chaque batch

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Pas de scheduler.step() ici car OneCycleLR step après chaque batch
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}")
    
    # Sauvegarder le meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'emotion_model_best.pth')
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

elapsed_time = time.time() - start_time

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Total time: {elapsed_time/60:.2f} minutes")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Best validation loss: {best_val_loss:.4f}")

# Charger le meilleur modèle et sauvegarder les poids finaux
checkpoint = torch.load('emotion_model_best.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
torch.save(model.state_dict(), 'emotion_model.pth')
print("\nBest model weights saved to 'emotion_model.pth'")