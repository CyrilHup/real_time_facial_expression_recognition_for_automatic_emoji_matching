"""
Training Script for Hand Gesture Recognition
============================================
Train CNN model on LeapGestRecog dataset from Kaggle.

Dataset structure expected:
./data/leapGestRecog/
    00_palm/
        01_palm_*.png
    01_l/
        02_l_*.png
    02_fist/
        ...
    (etc. for all 10 gesture classes)

Download dataset from: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from model_hand_gesture import create_hand_gesture_model


# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data
    DATA_DIR = './data/leapGestRecog'
    
    # Gesture classes (LeapGestRecog has 10 classes)
    GESTURE_CLASSES = {
        0: 'palm',
        1: 'l',
        2: 'fist',
        3: 'fist_moved',
        4: 'thumb',
        5: 'index',
        6: 'ok',
        7: 'palm_moved',
        8: 'c',
        9: 'down'
    }
    
    # Model
    NUM_CLASSES = 10
    INPUT_SIZE = 128  # Resize images to 128x128
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    SAVE_DIR = './checkpoints'
    MODEL_NAME = 'hand_gesture_model.pth'


config = Config()


# ============================================
# DATASET
# ============================================
class LeapGestureDataset(Dataset):
    """Dataset loader for LeapGestRecog"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir: Path to leapGestRecog folder
            transform: Transforms to apply
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load all image paths and labels
        self.samples = []
        self.labels = []
        
        # Map folder names to class indices
        self.class_to_idx = {}
        
        print(f"\nLoading {split} dataset from {data_dir}...")
        
        # Find all class folders
        class_folders = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        
        if len(class_folders) == 0:
            raise ValueError(f"No class folders found in {data_dir}")
        
        for idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name.split('_', 1)[1] if '_' in class_folder.name else class_folder.name
            self.class_to_idx[class_name] = idx
            
            # Get all images in this class
            images = list(class_folder.glob('*.png'))
            
            # Split data
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(images)
            
            n_train = int(len(images) * config.TRAIN_SPLIT)
            n_val = int(len(images) * config.VAL_SPLIT)
            
            if split == 'train':
                selected = images[:n_train]
            elif split == 'val':
                selected = images[n_train:n_train + n_val]
            else:  # test
                selected = images[n_train + n_val:]
            
            for img_path in selected:
                self.samples.append(str(img_path))
                self.labels.append(idx)
        
        print(f"  Loaded {len(self.samples)} images for {split}")
        print(f"  Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get class distribution for weighted sampling"""
        counts = np.bincount(self.labels, minlength=config.NUM_CLASSES)
        return counts


# ============================================
# DATA AUGMENTATION
# ============================================
def get_train_transforms():
    """Augmentation for training"""
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])


def get_val_transforms():
    """Transforms for validation/test"""
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
    ])


# ============================================
# TRAINING UTILITIES
# ============================================
class AverageMeter:
    """Computes and stores the average and current value"""
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
    """Compute class weights for imbalanced dataset"""
    counts = dataset.get_class_distribution()
    total = counts.sum()
    weights = total / (len(counts) * counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        loss_meter.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Acc: {100.*correct/total:.2f}%")
    
    accuracy = 100. * correct / total
    return loss_meter.avg, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class stats
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
    
    accuracy = 100. * correct / total
    
    # Print per-class accuracy
    print("\n  Per-class accuracy:")
    for cls_idx in sorted(class_total.keys()):
        cls_name = config.GESTURE_CLASSES[cls_idx]
        cls_acc = 100. * class_correct[cls_idx] / class_total[cls_idx]
        print(f"    {cls_name:15s}: {cls_acc:5.1f}% ({class_correct[cls_idx]}/{class_total[cls_idx]})")
    
    return loss_meter.avg, accuracy


# ============================================
# MAIN TRAINING LOOP
# ============================================
def train():
    print("="*70)
    print("  HAND GESTURE RECOGNITION TRAINING")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    
    # Create datasets
    train_dataset = LeapGestureDataset(
        config.DATA_DIR,
        transform=get_train_transforms(),
        split='train'
    )
    
    val_dataset = LeapGestureDataset(
        config.DATA_DIR,
        transform=get_val_transforms(),
        split='val'
    )
    
    test_dataset = LeapGestureDataset(
        config.DATA_DIR,
        transform=get_val_transforms(),
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = create_hand_gesture_model(
        num_classes=config.NUM_CLASSES,
        input_size=config.INPUT_SIZE
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss function with class weights
    class_weights = compute_class_weights(train_dataset).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        print("\nValidating...")
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'config': {
                    'num_classes': config.NUM_CLASSES,
                    'input_size': config.INPUT_SIZE,
                }
            }
            
            save_path = os.path.join(config.SAVE_DIR, config.MODEL_NAME)
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Test final model
    print("\n" + "="*70)
    print("Testing best model...")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(config.SAVE_DIR, config.MODEL_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, config.DEVICE)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Val Acc:  {best_val_acc:.2f}%")
    print(f"Test Acc:      {test_acc:.2f}%")
    print(f"Model saved to: {os.path.join(config.SAVE_DIR, config.MODEL_NAME)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train()
