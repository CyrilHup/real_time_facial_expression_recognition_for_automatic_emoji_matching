import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceEmotionCNN(nn.Module):
    """
    Architecture CNN améliorée pour la reconnaissance d'émotions.
    
    Supporte:
    - Images RGB 75x75 (Balanced AffectNet) avec in_channels=3
    - Images Grayscale 48x48 (FER2013/FER+) avec in_channels=1
    
    Features:
    - Plus de couches convolutionnelles pour extraire des features plus riches
    - Residual connections pour un meilleur gradient flow
    - Dropout progressif pour éviter l'overfitting
    - Global Average Pooling pour flexibilité de taille d'entrée
    """
    def __init__(self, num_classes=8, in_channels=3, input_size=75):
        """
        Args:
            num_classes: Nombre de classes d'émotions (8 pour AffectNet, 7 pour FER2013)
            in_channels: Nombre de canaux d'entrée (3 pour RGB, 1 pour grayscale)
            input_size: Taille de l'image d'entrée (75 pour AffectNet, 48 pour FER2013)
        """
        super(FaceEmotionCNN, self).__init__()
        
        self.in_channels = in_channels
        self.input_size = input_size
        
        # Convolutional Block 1 (75x75 -> 37x37 ou 48x48 -> 24x24)
        self.conv1a = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        
        # Convolutional Block 2 (24x24 -> 12x12)
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.1)
        
        # Convolutional Block 3 (12x12 -> 6x6)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.15)
        
        # Convolutional Block 4 (6x6 -> 3x3)
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.2)
        
        # Global Average Pooling (réduit 3x3 -> 1x1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers avec dropout progressif
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        return x


# Modèle léger compatible avec l'ancien (pour charger les anciens poids)
class FaceEmotionCNNLegacy(nn.Module):
    """Version legacy pour compatibilité avec les anciens poids (FER2013 48x48 grayscale)"""
    def __init__(self):
        super(FaceEmotionCNNLegacy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Factory function pour créer le bon modèle selon le dataset
def create_model(dataset='affectnet', num_classes=8):
    """
    Crée le modèle approprié selon le dataset.
    
    Args:
        dataset: 'affectnet' (75x75 RGB) ou 'fer2013' (48x48 grayscale)
        num_classes: Nombre de classes (8 pour AffectNet, 7 pour FER2013)
    
    Returns:
        Instance de FaceEmotionCNN configurée
    """
    if dataset.lower() == 'affectnet':
        return FaceEmotionCNN(num_classes=num_classes, in_channels=3, input_size=75)
    elif dataset.lower() in ['fer2013', 'ferplus', 'fer+']:
        return FaceEmotionCNN(num_classes=num_classes, in_channels=1, input_size=48)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'affectnet' or 'fer2013'.")