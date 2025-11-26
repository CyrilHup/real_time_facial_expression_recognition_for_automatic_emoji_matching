import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FER2013Dataset
from model import FaceEmotionCNN

print("Imports successful!")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 25 # Increase this if accuracy is low

# Setup Data
print("Setting up transforms...")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), # Converts to 0-1 range
])

print("Loading dataset...")
dataset = FER2013Dataset('./data/fer2013.csv', transform=transform)
print(f"Dataset loaded! Size: {len(dataset)} samples")

print("Creating DataLoader...")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print("DataLoader created!")

# Setup Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceEmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting Training...")

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'emotion_model.pth')
print("Model saved to emotion_model.pth")