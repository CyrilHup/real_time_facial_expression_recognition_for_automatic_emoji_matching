"""
Dataset amélioré avec support pour plusieurs datasets d'émotions.
- FER2013 (original)
- FER+ (annotations corrigées par Microsoft)
- AffectNet (si disponible)
- RAF-DB (si disponible)
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image
import os


class FER2013Dataset(Dataset):
    """Dataset FER2013 original."""
    
    def __init__(self, csv_file, transform=None, usage=None):
        """
        Args:
            csv_file: Chemin vers fer2013.csv
            transform: Transformations à appliquer
            usage: 'Training', 'PublicTest', 'PrivateTest' ou None pour tout
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        if usage is not None:
            self.data = self.data[self.data['Usage'] == usage]
        
        # FER-2013 Emotion mappings: 
        # 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx]['pixels']
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        pixels = pixels[:, :, np.newaxis]  # Ajouter dimension channel
        
        label = int(self.data.iloc[idx]['emotion'])

        if self.transform:
            pixels = self.transform(pixels)
            
        return pixels, label


class FERPlusDataset(Dataset):
    """
    Dataset FER+ avec annotations corrigées par Microsoft.
    Utilise un vote majoritaire ou des soft labels.
    Supporte 8 classes (7 FER2013 + Contempt)
    
    Téléchargement: https://github.com/microsoft/FERPlus
    """
    
    # Nombre de classes (8 avec Contempt)
    NUM_CLASSES = 8
    
    # Mapping FER+ labels (10 annotateurs par image)
    EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 
                'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    
    # Mapping vers les 8 classes (incluant Contempt)
    FERPLUS_TO_FER = {
        'anger': 0,
        'disgust': 1, 
        'fear': 2,
        'happiness': 3,
        'sadness': 4,
        'surprise': 5,
        'neutral': 6,
        'contempt': 7,  # Contempt comme 8ème classe
    }
    
    def __init__(self, fer_csv, ferplus_folder, transform=None, usage='Training', 
                 use_soft_labels=False, ignore_uncertain=True):
        """
        Args:
            fer_csv: Chemin vers fer2013.csv
            ferplus_folder: Dossier contenant les fichiers FER+ labels
            transform: Transformations
            usage: 'Training', 'PublicTest', 'PrivateTest'
            use_soft_labels: Si True, retourne distribution de probabilités
            ignore_uncertain: Si True, ignore les images avec label 'unknown' ou 'NF'
        """
        self.transform = transform
        self.use_soft_labels = use_soft_labels
        
        # Charger FER2013
        fer_data = pd.read_csv(fer_csv)
        fer_data = fer_data[fer_data['Usage'] == usage].reset_index(drop=True)
        
        # Charger les labels FER+
        label_file = os.path.join(ferplus_folder, f'fer2013new.csv')
        
        if os.path.exists(label_file):
            ferplus_labels = pd.read_csv(label_file)
            ferplus_labels = ferplus_labels[ferplus_labels['Usage'] == usage].reset_index(drop=True)
            
            self.pixels = fer_data['pixels'].values
            self.votes = ferplus_labels[self.EMOTIONS].values
            
            # Filtrer les images incertaines
            if ignore_uncertain:
                valid_mask = []
                for votes in self.votes:
                    max_idx = np.argmax(votes)
                    # Ignorer si 'unknown' ou 'NF' ont le plus de votes
                    if self.EMOTIONS[max_idx] not in ['unknown', 'NF']:
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                
                valid_mask = np.array(valid_mask)
                self.pixels = self.pixels[valid_mask]
                self.votes = self.votes[valid_mask]
        else:
            print(f"FER+ labels not found at {label_file}, falling back to FER2013")
            self.pixels = fer_data['pixels'].values
            self.votes = None
            self.labels = fer_data['emotion'].values
    
    def __len__(self):
        return len(self.pixels)
    
    def __getitem__(self, idx):
        # Charger l'image
        pixels = self.pixels[idx]
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        pixels = pixels[:, :, np.newaxis]
        
        if self.votes is not None:
            votes = self.votes[idx]
            
            if self.use_soft_labels:
                # Soft labels: distribution de probabilités sur 8 classes
                soft_label = np.zeros(self.NUM_CLASSES, dtype=np.float32)
                for i, emotion in enumerate(self.EMOTIONS[:8]):  # Exclure unknown et NF
                    if emotion in self.FERPLUS_TO_FER:
                        soft_label[self.FERPLUS_TO_FER[emotion]] += votes[i]
                
                # Normaliser
                if soft_label.sum() > 0:
                    soft_label = soft_label / soft_label.sum()
                
                label = torch.tensor(soft_label)
            else:
                # Hard label: vote majoritaire
                # Ignorer 'unknown' et 'NF' seulement
                valid_votes = votes[:8].copy()  # Inclut contempt maintenant
                max_idx = np.argmax(valid_votes)
                emotion = self.EMOTIONS[max_idx]
                label = self.FERPLUS_TO_FER.get(emotion, 6)
        else:
            label = int(self.labels[idx])
        
        if self.transform:
            pixels = self.transform(pixels)
            
        return pixels, label


class AffectNetDataset(Dataset):
    """
    Dataset AffectNet - le plus grand dataset d'émotions.
    Nécessite de télécharger depuis: http://mohammadmahoor.com/affectnet/
    
    Structure attendue:
    affectnet/
        train/
            0/ (Neutral)
            1/ (Happy)
            2/ (Sad)
            ...
        val/
            0/
            1/
            ...
    """
    
    # Mapping AffectNet vers FER
    AFFECTNET_TO_FER = {
        0: 6,  # Neutral -> Neutral
        1: 3,  # Happy -> Happy
        2: 4,  # Sad -> Sad
        3: 5,  # Surprise -> Surprise
        4: 2,  # Fear -> Fear
        5: 1,  # Disgust -> Disgust
        6: 0,  # Anger -> Angry
        7: 6,  # Contempt -> Neutral
    }
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"AffectNet not found at {split_dir}")
        
        for emotion_idx in range(8):
            emotion_dir = os.path.join(split_dir, str(emotion_idx))
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(emotion_dir, img_name))
                        self.labels.append(self.AFFECTNET_TO_FER[emotion_idx])
        
        print(f"Loaded {len(self.images)} images from AffectNet {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        image = image.resize((48, 48))
        image = np.array(image)[:, :, np.newaxis]
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class RAFDBDataset(Dataset):
    """
    Dataset RAF-DB (Real-world Affective Faces Database).
    Téléchargement: http://www.whdeng.cn/raf/model1.html
    
    Structure attendue:
    RAF-DB/
        basic/
            Image/
                train/
                test/
            EmoLabel/
                list_patition_label.txt
    """
    
    # Mapping RAF-DB vers FER (1-indexed dans RAF-DB)
    RAFDB_TO_FER = {
        1: 5,  # Surprise
        2: 2,  # Fear
        3: 1,  # Disgust
        4: 3,  # Happiness
        5: 4,  # Sadness
        6: 0,  # Anger
        7: 6,  # Neutral
    }
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        label_file = os.path.join(root_dir, 'basic', 'EmoLabel', 'list_patition_label.txt')
        image_dir = os.path.join(root_dir, 'basic', 'Image', 'aligned')
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"RAF-DB labels not found at {label_file}")
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, label = parts
                    
                    # Vérifier si c'est train ou test
                    if split == 'train' and img_name.startswith('train'):
                        img_path = os.path.join(image_dir, img_name.replace('.jpg', '_aligned.jpg'))
                        if os.path.exists(img_path):
                            self.images.append(img_path)
                            self.labels.append(self.RAFDB_TO_FER[int(label)])
                    elif split == 'test' and img_name.startswith('test'):
                        img_path = os.path.join(image_dir, img_name.replace('.jpg', '_aligned.jpg'))
                        if os.path.exists(img_path):
                            self.images.append(img_path)
                            self.labels.append(self.RAFDB_TO_FER[int(label)])
        
        print(f"Loaded {len(self.images)} images from RAF-DB {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        image = image.resize((48, 48))
        image = np.array(image)[:, :, np.newaxis]
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_class_weights(dataset, num_classes=8):
    """
    Calcule les poids pour équilibrer les classes.
    Utile car FER2013 a très peu d'exemples de 'Disgust'.
    Supporte 8 classes (incluant Contempt).
    """
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.argmax().item()
        labels.append(label)
    
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Éviter division par zéro
    class_counts = np.maximum(class_counts, 1)
    
    # Poids inversement proportionnels à la fréquence
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    
    print("Class distribution:")
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    for i, (emotion, count, weight) in enumerate(zip(emotions, class_counts, weights)):
        print(f"  {emotion}: {count} samples, weight: {weight:.3f}")
    
    return torch.FloatTensor(weights)


def get_balanced_sampler(dataset, num_classes=8):
    """
    Crée un sampler pour équilibrer les classes pendant l'entraînement.
    Supporte 8 classes (incluant Contempt).
    """
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.argmax().item()
        labels.append(label)
    
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    
    # Poids par sample
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def download_ferplus_labels():
    """
    Instructions pour télécharger les labels FER+.
    """
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TÉLÉCHARGER FER+ LABELS                       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  FER+ améliore FER2013 avec des annotations corrigées par        ║
    ║  10 annotateurs humains (Microsoft Research).                    ║
    ║                                                                  ║
    ║  Étapes:                                                         ║
    ║  1. Aller sur: https://github.com/microsoft/FERPlus              ║
    ║  2. Télécharger 'fer2013new.csv' depuis le dossier 'data'        ║
    ║  3. Placer le fichier dans: ./data/fer2013new.csv                ║
    ║                                                                  ║
    ║  Ou exécuter:                                                    ║
    ║  git clone https://github.com/microsoft/FERPlus.git              ║
    ║  copy FERPlus\\data\\fer2013new.csv .\\data\\                       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    # Test des datasets
    print("Testing datasets...")
    
    # Test FER2013
    try:
        fer = FER2013Dataset('./data/fer2013/fer2013.csv')
        print(f"FER2013: {len(fer)} samples")
        weights = get_class_weights(fer)
    except Exception as e:
        print(f"FER2013 not found: {e}")
    
    # Test FER+
    try:
        ferplus = FERPlusDataset('./data/fer2013/fer2013.csv', './data/fer2013/')
        print(f"FER+: {len(ferplus)} samples")
    except Exception as e:
        print(f"FER+ not available: {e}")
        download_ferplus_labels()
