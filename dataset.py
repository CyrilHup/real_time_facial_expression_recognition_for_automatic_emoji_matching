import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # FER-2013 Emotion mappings: 
        # 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Pixels are stored as a string of numbers separated by space
        pixels = self.data.iloc[idx]['pixels']
        pixels = np.array(pixels.split(), dtype='uint8').reshape(48, 48)
        
        # Add channel dimension (1, 48, 48) for Grayscale
        pixels = pixels[:, :, np.newaxis]
        
        label = int(self.data.iloc[idx]['emotion'])

        if self.transform:
            pixels = self.transform(pixels)
            
        return pixels, label