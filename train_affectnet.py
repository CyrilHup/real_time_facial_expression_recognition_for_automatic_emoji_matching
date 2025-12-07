"""
Training Script for Balanced AffectNet Dataset
===============================================
Facial Emotion Recognition using CNN.

Dataset: Balanced AffectNet
- 8 classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 41,008 images (75x75 RGB)
- Pre-balanced: ~5,126 images per class

Key features:
- RGB images (3 channels) instead of grayscale
- 75x75 pixel resolution
- Mixup & CutMix augmentation for better generalization
- Label smoothing to prevent overconfidence
- Advanced data augmentation with albumentations
- OneCycleLR scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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

from model import FaceEmotionCNN, create_model
from dataset_affectnet import BalancedAffectNetDataset, get_class_weights, get_balanced_sampler

# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data
    DATASET_ROOT = './data'
    
    # Model
    NUM_CLASSES = 8  # 8 émotions AffectNet
    IN_CHANNELS = 3  # RGB
    INPUT_SIZE = 75  # 75x75 pixels
    
    # Training
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 1
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-4
    EPOCHS = 80
    PATIENCE = 15
    
    # Advanced techniques
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    USE_CUTMIX = False
    CUTMIX_ALPHA = 1.0
    CUTMIX_PROB = 0.5
    
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1
    
    USE_FOCAL_LOSS = False
    FOCAL_GAMMA = 2.0
    
    # Augmentation
    USE_ADVANCED_AUG = True
    
    # Class balancing (moins nécessaire car dataset déjà équilibré)
    USE_OVERSAMPLING = False
    MAX_CLASS_WEIGHT = 3.0
    
    # Validation
    VAL_TTA = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()
print(f"Device: {config.DEVICE}")

# ============================================
# LOSSES
# ============================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            targets_smooth = torch.zeros_like(inputs)
            targets_smooth.fill_(self.label_smoothing / (n_classes - 1))
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        probs = torch.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
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
    
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for mixup/cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================
# ADVANCED AUGMENTATION (RGB)
# ============================================
def get_train_transforms():
    """Get training transforms for RGB 75x75 images."""
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
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
            ], p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(6, 12),
                hole_width_range=(6, 12),
                fill=0,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_val_transforms():
    """Get validation transforms for RGB 75x75 images."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ============================================
# DATASET WRAPPER
# ============================================
class AffectNetDatasetWithTransform(BalancedAffectNetDataset):
    """Extended dataset with proper transform handling for albumentations."""
    
    def __init__(self, root_dir, split, transform, use_albumentations=False):
        super().__init__(root_dir=root_dir, split=split, transform=None, grayscale=False)
        self.custom_transform = transform
        self.use_albumentations = use_albumentations
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image as RGB
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transform
        if self.custom_transform:
            if self.use_albumentations:
                augmented = self.custom_transform(image=image)
                image = augmented['image']
            else:
                image = self.custom_transform(image)
        else:
            # Default: just convert to tensor
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return image, label


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


# ============================================
# MAIN TRAINING LOOP
# ============================================
def train():
    print("="*70)
    print("  TRAINING FOR BALANCED AFFECTNET - FACIAL EMOTION RECOGNITION")
    print("="*70)
    
    # Create datasets
    print("\nLoading Balanced AffectNet dataset...")
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = AffectNetDatasetWithTransform(
        root_dir=config.DATASET_ROOT,
        split='train',
        transform=train_transform,
        use_albumentations=HAS_ALBUMENTATIONS
    )
    
    val_dataset = AffectNetDatasetWithTransform(
        root_dir=config.DATASET_ROOT,
        split='val',
        transform=val_transform,
        use_albumentations=HAS_ALBUMENTATIONS
    )
    
    # Compute class weights (même si dataset équilibré, peut aider)
    class_weights = get_class_weights(train_dataset, max_weight=config.MAX_CLASS_WEIGHT).to(config.DEVICE)
    
    # Create dataloaders
    if config.USE_OVERSAMPLING:
        print("\n✓ Using oversampling for rare classes")
        sampler = get_balanced_sampler(train_dataset)
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
    
    # Create model for AffectNet (RGB 75x75)
    model = create_model(dataset='affectnet', num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
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
    
    val_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler - OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 10,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training settings display
    print(f"\n{'='*70}")
    print("Training Configuration:")
    print(f"  Dataset: Balanced AffectNet (75x75 RGB, 8 classes)")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.EPOCHS}, Patience: {config.PATIENCE}")
    print(f"  Mixup: {config.USE_MIXUP} (alpha={config.MIXUP_ALPHA})")
    print(f"  Label Smoothing: {config.USE_LABEL_SMOOTHING} ({config.LABEL_SMOOTHING})")
    print(f"  Advanced Aug: {HAS_ALBUMENTATIONS and config.USE_ADVANCED_AUG}")
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
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
        elapsed_epoch = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train Loss: {loss_meter.avg:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed_epoch/60:.1f}min")
        
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
                    'in_channels': config.IN_CHANNELS,
                    'input_size': config.INPUT_SIZE,
                    'dataset': 'affectnet',
                }
            }, 'emotion_affectnet_model_best.pth')
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
    checkpoint = torch.load('emotion_affectnet_model_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    validate(model, val_loader, val_criterion, config.DEVICE, per_class=True)
    
    # Save final weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': config.NUM_CLASSES,
        'in_channels': config.IN_CHANNELS,
        'input_size': config.INPUT_SIZE,
        'dataset': 'affectnet',
    }, 'emotion_affectnet_model.pth')
    print("\nModel saved to 'emotion_affectnet_model.pth'")


if __name__ == "__main__":
    train()
