import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Define the custom dataset
class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0] + ".jpeg"  # Add extension
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[index, -1])  # 'target' column

        if self.transform:
            image = self.transform(image)

        return image, label


# Define the Convolutional Autoencoder (CAE)
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Reduce size
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Further reduce
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Restore original size
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the CNN
class CNN(nn.Module):
    def __init__(self, pretrained_encoder):
        super(CNN, self).__init__()
        # Initialize with CAE encoder weights
        self.features = pretrained_encoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),  # Adjust dimensions based on input size
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification (malignant/benign)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Paths and transforms
csv_file = "F:/Mayur/vit/TY/EDI/new_data/train.csv"
root_dir = "F:/Mayur/vit/TY/EDI/new_data/allimages"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
dataset = SkinCancerDataset(csv_file, root_dir, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the CAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cae = CAE().to(device)
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.Adam(cae.parameters(), lr=0.001)

print("Training CAE...")
for epoch in range(10):
    cae.train()
    train_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = cae(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}")

# Transfer weights to CNN
print("Transferring weights from CAE to CNN...")
pretrained_encoder = cae.encoder
cnn = CNN(pretrained_encoder).to(device)

# Train the CNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

print("Training CNN...")
for epoch in range(10):
    cnn.train()
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")

# Evaluate on test data
cnn.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
print(f"Test Accuracy: {100. * correct / total:.2f}%")
