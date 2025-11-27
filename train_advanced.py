"""
Advanced Training Script for Facial Emotion Recognition
========================================================
Key improvements:
- Focal Loss for class imbalance (disgust, contempt, fear)
- Mixup & CutMix augmentation for better generalization
- Label smoothing to prevent overconfidence
- Advanced data augmentation with albumentations
- Cosine annealing with warm restarts
- Gradient accumulation for larger effective batch size
- Knowledge distillation ready
- Better validation with per-class metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import os
import time
from collections import defaultdict
import random

# Try to import albumentations for advanced augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
    print("✓ Albumentations available for advanced augmentation")
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠ Install albumentations for better augmentation: pip install albumentations")

from model import FaceEmotionCNN

# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data
    FER_FILE = './data/fer2013/fer2013.csv'
    FERPLUS_FILE = './data/fer2013/fer2013new.csv'
    
    # Model
    NUM_CLASSES = 8  # 7 FER + Contempt
    
    # Training - same as working train.py
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 1
    LEARNING_RATE = 0.0005  # Same as train.py
    WEIGHT_DECAY = 1e-4
    EPOCHS = 80  # More epochs since we have better regularization
    PATIENCE = 15  # More patience for rare class learning
    
    # Advanced techniques
    USE_MIXUP = True  # ENABLED - helps with class imbalance and generalization
    MIXUP_ALPHA = 0.4  # Increased for stronger mixing
    USE_CUTMIX = False  # Keep disabled - mixup is enough
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.5
    
    USE_LABEL_SMOOTHING = True  # ENABLED - helps prevent overconfidence
    LABEL_SMOOTHING = 0.1
    
    USE_FOCAL_LOSS = False  # Keep disabled - class weights are sufficient
    FOCAL_GAMMA = 2.0
    
    # Augmentation - keep this, it helps!
    USE_ADVANCED_AUG = True
    
    # Class balancing
    USE_OVERSAMPLING = True  # NEW: Oversample rare classes
    MAX_CLASS_WEIGHT = 5.0  # Increased from 3.0 for rare classes
    
    # Validation
    VAL_TTA = True  # ENABLED - Test-Time Augmentation
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()
print(f"Device: {config.DEVICE}")

# ============================================
# LOSSES
# ============================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified examples, focuses on hard ones.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            targets_smooth = torch.zeros_like(inputs)
            targets_smooth.fill_(self.label_smoothing / (n_classes - 1))
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weight
        probs = torch.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        targets_smooth = torch.zeros_like(log_probs)
        targets_smooth.fill_(self.smoothing / (n_classes - 1))
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        return loss.mean()


# ============================================
# MIXUP & CUTMIX
# ============================================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: blend two samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation: cut and paste patches between samples."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get bounding box
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for mixup/cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================
# ADVANCED AUGMENTATION
# ============================================
def get_train_transforms():
    """Get training transforms with advanced augmentation."""
    if HAS_ALBUMENTATIONS and config.USE_ADVANCED_AUG:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(4, 8),
                hole_width_range=(4, 8),
                fill=0,
                p=0.3
            ),
            ToTensorV2(),
        ])
    else:
        # Fallback to torchvision transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])


def get_val_transforms():
    """Get validation transforms."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])


def get_tta_transforms():
    """Get test-time augmentation transforms."""
    if HAS_ALBUMENTATIONS:
        return [
            # Original
            A.Compose([ToTensorV2()]),
            # Flip
            A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()]),
            # Slight rotation
            A.Compose([A.Rotate(limit=(5, 5), p=1.0), ToTensorV2()]),
            A.Compose([A.Rotate(limit=(-5, -5), p=1.0), ToTensorV2()]),
        ]
    else:
        return None


# ============================================
# DATASET
# ============================================
class FERPlusDatasetAdvanced(Dataset):
    """Advanced FER+ dataset with proper augmentation support."""
    
    EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 
                'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    
    FERPLUS_TO_IDX = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3,
        'sadness': 4, 'surprise': 5, 'neutral': 6, 'contempt': 7,
    }
    
    def __init__(self, fer_csv, ferplus_csv, transform=None, usage='Training', 
                 ignore_uncertain=True, use_albumentations=False):
        import pandas as pd
        
        self.transform = transform
        self.use_albumentations = use_albumentations
        
        # Load FER2013
        fer_data = pd.read_csv(fer_csv)
        fer_data = fer_data[fer_data['Usage'] == usage].reset_index(drop=True)
        
        # Load FER+ labels
        if os.path.exists(ferplus_csv):
            ferplus = pd.read_csv(ferplus_csv)
            ferplus = ferplus[ferplus['Usage'] == usage].reset_index(drop=True)
            
            self.pixels = []
            self.labels = []
            
            for i in range(len(fer_data)):
                votes = ferplus.iloc[i][self.EMOTIONS].values.astype(int)
                
                # Skip uncertain images
                if ignore_uncertain:
                    if votes[8] > 0 or votes[9] > 0:  # unknown or NF
                        continue
                    if votes[:8].max() < 3:  # No clear majority
                        continue
                
                # Get majority vote
                valid_votes = votes[:8]
                max_idx = valid_votes.argmax()
                emotion_name = self.EMOTIONS[max_idx]
                
                if emotion_name in self.FERPLUS_TO_IDX:
                    self.pixels.append(fer_data.iloc[i]['pixels'])
                    self.labels.append(self.FERPLUS_TO_IDX[emotion_name])
            
            print(f"Loaded {len(self.pixels)} samples for {usage}")
        else:
            # Fallback to FER2013
            self.pixels = fer_data['pixels'].values.tolist()
            self.labels = fer_data['emotion'].values.tolist()
            print(f"FER+ not found, using FER2013: {len(self.pixels)} samples")
    
    def __len__(self):
        return len(self.pixels)
    
    def __getitem__(self, idx):
        # Parse pixels
        pixels = np.array(self.pixels[idx].split(), dtype=np.uint8).reshape(48, 48)
        label = self.labels[idx]
        
        # Apply transform
        if self.transform:
            if self.use_albumentations:
                augmented = self.transform(image=pixels)
                pixels = augmented['image']
                # ToTensorV2 outputs uint8 tensor, need to convert to float [0, 1]
                pixels = pixels.float() / 255.0
                if pixels.dim() == 2:
                    pixels = pixels.unsqueeze(0)
            else:
                pixels = pixels[:, :, np.newaxis]
                pixels = self.transform(pixels)
        else:
            pixels = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return pixels, label
    
    def get_class_distribution(self):
        """Get class distribution for weighted sampling."""
        counts = np.bincount(self.labels, minlength=config.NUM_CLASSES)
        return counts


# ============================================
# TRAINING UTILITIES
# ============================================
class AverageMeter:
    """Track average values."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset."""
    counts = dataset.get_class_distribution()
    counts = np.maximum(counts, 1)
    
    # Inverse frequency weighting - but capped to avoid extreme values
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    
    # Cap extreme weights - increased cap for very rare classes
    weights = np.clip(weights, 0.3, config.MAX_CLASS_WEIGHT)
    weights = weights / weights.sum() * len(weights)  # Re-normalize
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    print("\nClass distribution and weights (capped):")
    for i, (emo, count, weight) in enumerate(zip(emotions, counts, weights)):
        print(f"  {emo:10s}: {count:5d} samples, weight: {weight:.3f}")
    
    return torch.FloatTensor(weights)


def get_oversampling_weights(dataset):
    """Get per-sample weights for oversampling rare classes."""
    counts = dataset.get_class_distribution()
    counts = np.maximum(counts, 1)
    
    # More aggressive oversampling for very rare classes (Disgust, Contempt)
    # Use inverse frequency for rare classes, sqrt for medium classes
    weights = np.zeros_like(counts, dtype=np.float64)
    median_count = np.median(counts)
    
    for i, count in enumerate(counts):
        if count < 100:  # Very rare (Disgust, Contempt)
            weights[i] = 1.0 / count * 2  # Extra boost
        elif count < median_count:  # Rare
            weights[i] = 1.0 / np.sqrt(count)
        else:  # Common
            weights[i] = 1.0 / np.sqrt(count)
    
    weights = weights / weights.sum()
    
    # Get weight for each sample
    sample_weights = [weights[label] for label in dataset.labels]
    return sample_weights


def validate(model, val_loader, criterion, device, per_class=False):
    """Validate with optional per-class metrics."""
    model.eval()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    if per_class:
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss_meter.update(loss.item(), inputs.size(0))
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if per_class:
                for pred, label in zip(predicted, labels):
                    class_total[label.item()] += 1
                    if pred == label:
                        class_correct[label.item()] += 1
    
    accuracy = 100.0 * correct / total
    
    if per_class:
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        print("\n  Per-class accuracy:")
        for i, emo in enumerate(emotions):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                print(f"    {emo:10s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]})")
    
    return loss_meter.avg, accuracy


def validate_with_tta(model, dataset, device, num_augments=4):
    """Validate with test-time augmentation for better accuracy."""
    model.eval()
    
    # TTA transforms
    tta_transforms_list = [
        # Original
        lambda img: img,
        # Horizontal flip
        lambda img: np.fliplr(img).copy(),
        # Small shifts
        lambda img: np.roll(img, 2, axis=0),
        lambda img: np.roll(img, 2, axis=1),
    ]
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            pixels = np.array(dataset.pixels[idx].split(), dtype=np.uint8).reshape(48, 48)
            label = dataset.labels[idx]
            
            # Collect predictions from all augmentations
            probs_list = []
            for transform in tta_transforms_list[:num_augments]:
                aug_pixels = transform(pixels)
                # Convert to tensor
                tensor = torch.tensor(aug_pixels, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                tensor = tensor.to(device)
                
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                probs_list.append(probs)
            
            # Average predictions
            avg_probs = torch.stack(probs_list).mean(dim=0)
            pred = avg_probs.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)
    
    # Compute accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / len(all_labels)
    
    # Per-class accuracy
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    print("\n  TTA Per-class accuracy:")
    for i, emo in enumerate(emotions):
        class_total = sum(1 for l in all_labels if l == i)
        class_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == i and p == l)
        if class_total > 0:
            acc = 100.0 * class_correct / class_total
            print(f"    {emo:10s}: {acc:5.1f}% ({class_correct}/{class_total})")
    
    return accuracy


# ============================================
# MAIN TRAINING LOOP
# ============================================
def train():
    print("="*70)
    print("  ADVANCED TRAINING FOR FACIAL EMOTION RECOGNITION")
    print("="*70)
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = FERPlusDatasetAdvanced(
        config.FER_FILE, config.FERPLUS_FILE,
        transform=train_transform,
        usage='Training',
        ignore_uncertain=True,
        use_albumentations=HAS_ALBUMENTATIONS
    )
    
    val_dataset = FERPlusDatasetAdvanced(
        config.FER_FILE, config.FERPLUS_FILE,
        transform=val_transform,
        usage='PublicTest',
        ignore_uncertain=True,
        use_albumentations=HAS_ALBUMENTATIONS
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset).to(config.DEVICE)
    
    # Create dataloaders with optional oversampling
    if config.USE_OVERSAMPLING:
        print("\n✓ Using oversampling for rare classes")
        sample_weights = get_oversampling_weights(train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nTrain: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create model
    model = FaceEmotionCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss function
    if config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=config.FOCAL_GAMMA,
            alpha=class_weights,
            label_smoothing=config.LABEL_SMOOTHING if config.USE_LABEL_SMOOTHING else 0.0
        )
        print(f"✓ Using Focal Loss (gamma={config.FOCAL_GAMMA})")
    elif config.USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.LABEL_SMOOTHING)
        print(f"✓ Using Label Smoothing (smoothing={config.LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # For validation (no focal loss)
    val_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler - OneCycleLR like the working train.py
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 10,  # 10x base LR
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos'
    )
    
    # Training settings display
    print(f"\n{'='*70}")
    print("Training Configuration:")
    print(f"  Batch size: {config.BATCH_SIZE} x {config.ACCUMULATION_STEPS} = {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.EPOCHS}, Patience: {config.PATIENCE}")
    print(f"  Mixup: {config.USE_MIXUP} (alpha={config.MIXUP_ALPHA})")
    print(f"  CutMix: {config.USE_CUTMIX} (alpha={config.CUTMIX_ALPHA})")
    print(f"  Focal Loss: {config.USE_FOCAL_LOSS}")
    print(f"  Label Smoothing: {config.USE_LABEL_SMOOTHING} ({config.LABEL_SMOOTHING})")
    print(f"  Oversampling: {config.USE_OVERSAMPLING}")
    print(f"  Advanced Aug: {HAS_ALBUMENTATIONS and config.USE_ADVANCED_AUG}")
    print(f"  TTA: {config.VAL_TTA}")
    print(f"{'='*70}\n")
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        model.train()
        
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Apply mixup or cutmix
            use_mixup = config.USE_MIXUP and random.random() > 0.5
            use_cutmix = config.USE_CUTMIX and random.random() < config.CUTMIX_PROB and not use_mixup
            
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, config.MIXUP_ALPHA)
            elif use_cutmix:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, config.CUTMIX_ALPHA)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / config.ACCUMULATION_STEPS
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # OneCycleLR steps after each batch
                optimizer.zero_grad()
            
            # Track metrics
            loss_meter.update(loss.item() * config.ACCUMULATION_STEPS, inputs.size(0))
            
            if not (use_mixup or use_cutmix):
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / max(total, 1)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, val_criterion, config.DEVICE, 
                                     per_class=(epoch % 10 == 0))
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train Loss: {loss_meter.avg:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
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
                'config': {
                    'num_classes': config.NUM_CLASSES,
                    'focal_loss': config.USE_FOCAL_LOSS,
                    'mixup': config.USE_MIXUP,
                    'cutmix': config.USE_CUTMIX,
                }
            }, 'emotion_model_best.pth')
            print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs!")
                break
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final per-class evaluation
    print("\nFinal per-class evaluation on best model:")
    checkpoint = torch.load('emotion_model_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    validate(model, val_loader, val_criterion, config.DEVICE, per_class=True)
    
    # TTA evaluation
    if config.VAL_TTA:
        print("\n" + "="*70)
        print("Test-Time Augmentation (TTA) Evaluation:")
        print("="*70)
        tta_acc = validate_with_tta(model, val_dataset, config.DEVICE, num_augments=4)
        print(f"\n  TTA Accuracy: {tta_acc:.2f}% (vs {best_val_acc:.2f}% without TTA)")
    
    # Save final weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': config.NUM_CLASSES,
    }, 'emotion_model.pth')
    print("\nModel saved to 'emotion_model.pth'")


if __name__ == "__main__":
    train()
