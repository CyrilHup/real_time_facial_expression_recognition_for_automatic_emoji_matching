"""
Dataset pour Balanced AffectNet.

Dataset source: https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet

Caractéristiques:
- 8 classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Images RGB 75x75 pixels
- Dataset équilibré (~5,126 images par classe)
- Total: 41,008 images
- Split: train (29,526), val (7,382), test (4,100)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os


class BalancedAffectNetDataset(Dataset):
    """
    Dataset pour Balanced AffectNet.
    
    Structure attendue:
    data/
        train/
            Anger/
            Contempt/
            Disgust/
            Fear/
            Happy/
            Neutral/
            Sad/
            Surprise/
        val/
            ...
        test/
            ...
    """
    
    # 8 classes d'émotions
    NUM_CLASSES = 8
    
    # Mapping des noms de dossiers vers les indices
    EMOTION_CLASSES = {
        'Anger': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6,
        'Contempt': 7,
    }
    
    # Mapping inverse pour affichage
    IDX_TO_EMOTION = {v: k for k, v in EMOTION_CLASSES.items()}
    
    def __init__(self, root_dir='./data', split='train', transform=None, 
                 grayscale=False, target_size=None):
        """
        Args:
            root_dir: Dossier racine du dataset (contient train/, val/, test/)
            split: 'train', 'val', ou 'test'
            transform: Transformations à appliquer
            grayscale: Si True, convertir en niveaux de gris (1 channel)
            target_size: Tuple (H, W) pour redimensionner. None = garder 75x75
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.grayscale = grayscale
        self.target_size = target_size
        
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Dataset not found at {split_dir}\n"
                f"Please download Balanced AffectNet from:\n"
                f"https://www.kaggle.com/datasets/dollyprajapati182/balanced-affectnet\n"
                f"And extract the train/, val/, test/ folders directly into {root_dir}"
            )
        
        # Charger toutes les images
        for emotion_name, emotion_idx in self.EMOTION_CLASSES.items():
            emotion_dir = os.path.join(split_dir, emotion_name)
            
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} not found, skipping...")
                continue
            
            for img_name in os.listdir(emotion_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(emotion_dir, img_name))
                    self.labels.append(emotion_idx)
        
        print(f"Loaded {len(self.images)} images from AffectNet {split}")
        
        # Afficher la distribution des classes
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Affiche la distribution des classes."""
        counts = np.bincount(self.labels, minlength=self.NUM_CLASSES)
        print(f"  Class distribution ({self.split}):")
        for idx, count in enumerate(counts):
            emotion = self.IDX_TO_EMOTION.get(idx, f"Class_{idx}")
            print(f"    {emotion:10s}: {count:5d} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Charger l'image
        image = Image.open(img_path)
        
        # Convertir en grayscale si demandé
        if self.grayscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        
        # Redimensionner si demandé
        if self.target_size is not None:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convertir en numpy pour les transforms
        image = np.array(image)
        
        # Ajouter dimension channel si grayscale
        if self.grayscale and image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        # Appliquer les transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Retourne le nombre d'images par classe."""
        return np.bincount(self.labels, minlength=self.NUM_CLASSES)
    
    def get_labels(self):
        """Retourne tous les labels (utile pour le sampler équilibré)."""
        return np.array(self.labels)


def get_class_weights(dataset, max_weight=5.0):
    """
    Calcule les poids pour équilibrer les classes.
    
    Args:
        dataset: Instance de BalancedAffectNetDataset
        max_weight: Poids maximum (cap pour éviter les valeurs extrêmes)
    
    Returns:
        torch.FloatTensor des poids par classe
    """
    counts = dataset.get_class_distribution()
    counts = np.maximum(counts, 1)  # Éviter division par zéro
    
    # Poids inversement proportionnels à la fréquence
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    
    # Limiter les poids extrêmes
    weights = np.clip(weights, 0.3, max_weight)
    weights = weights / weights.sum() * len(weights)  # Re-normaliser
    
    print("\nClass weights:")
    for i, (count, weight) in enumerate(zip(counts, weights)):
        emotion = BalancedAffectNetDataset.IDX_TO_EMOTION.get(i, f"Class_{i}")
        print(f"  {emotion:10s}: {count:5d} samples, weight: {weight:.3f}")
    
    return torch.FloatTensor(weights)


def get_balanced_sampler(dataset):
    """
    Crée un sampler pour équilibrer les classes pendant l'entraînement.
    
    Args:
        dataset: Instance de BalancedAffectNetDataset
    
    Returns:
        WeightedRandomSampler
    """
    labels = dataset.get_labels()
    counts = np.bincount(labels, minlength=BalancedAffectNetDataset.NUM_CLASSES)
    counts = np.maximum(counts, 1)
    
    # Poids par sample
    weights = 1.0 / counts
    sample_weights = weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def download_instructions():
    """Affiche les instructions pour télécharger le dataset."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║              TÉLÉCHARGER BALANCED AFFECTNET DATASET              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Le dataset Balanced AffectNet est équilibré avec 8 classes      ║
    ║  d'émotions et ~5,126 images par classe.                         ║
    ║                                                                  ║
    ║  Caractéristiques:                                               ║
    ║  • 41,008 images totales                                         ║
    ║  • Images RGB 75x75 pixels                                       ║
    ║  • 8 émotions: Anger, Contempt, Disgust, Fear,                   ║
    ║                Happy, Neutral, Sad, Surprise                     ║
    ║                                                                  ║
    ║  Étapes:                                                         ║
    ║  1. Aller sur:                                                   ║
    ║     https://www.kaggle.com/datasets/dollyprajapati182/           ║
    ║     balanced-affectnet                                           ║
    ║                                                                  ║
    ║  2. Télécharger et extraire le dataset                           ║
    ║                                                                  ║
    ║  3. Placer les dossiers directement dans: ./data/                ║
    ║                                                                  ║
    ║  Structure attendue:                                             ║
    ║  data/                                                           ║
    ║      train/                                                      ║
    ║          Anger/                                                  ║
    ║          Contempt/                                               ║
    ║          ...                                                     ║
    ║      val/                                                        ║
    ║          ...                                                     ║
    ║      test/                                                       ║
    ║          ...                                                     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    # Test du dataset
    print("Testing Balanced AffectNet Dataset...")
    
    try:
        # Test avec les paramètres par défaut
        train_dataset = BalancedAffectNetDataset(
            root_dir='./data',
            split='train'
        )
        
        val_dataset = BalancedAffectNetDataset(
            root_dir='./data',
            split='val'
        )
        
        test_dataset = BalancedAffectNetDataset(
            root_dir='./data',
            split='test'
        )
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
        # Test d'un sample
        img, label = train_dataset[0]
        print(f"\nSample shape: {img.shape}")
        print(f"Sample label: {label} ({BalancedAffectNetDataset.IDX_TO_EMOTION[label]})")
        
        # Test des poids de classe
        weights = get_class_weights(train_dataset)
        print(f"\nClass weights tensor: {weights}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        download_instructions()
