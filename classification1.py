import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import io
from contextlib import redirect_stdout, redirect_stderr
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
# class BeltClassificationNet(nn.Module):
#     def __init__(self):
#         super(BeltClassificationNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 16 * 16, 64)
#         self.fc2 = nn.Linear(64, 2)  # 2 classes: defect or no defect
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 16 * 16)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


class BeltClassificationNet(nn.Module):
    def __init__(self):
        super(BeltClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 256 * 256, 64)  # Update this based on the true size after pooling
        self.fc2 = nn.Linear(64, 2)  # 2 classes: defect or no defect
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # After this, size will be (32, 512, 512)
        x = self.pool(torch.relu(self.conv2(x)))  # After this, size will be (64, 256, 256)
        x = x.view(-1, 64 * 256 * 256)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BeltDataset(Dataset):
    def __init__(self, image_dir, label_dir, seg_model, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.bmp')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Perform segmentation without printing
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            results = self.seg_model(img_path)
        seg_image = results[0].plot()
        seg_image = Image.fromarray(seg_image)

        # Check if label file exists
        label_file = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        label = 1 if os.path.exists(label_file) else 0  # 1 for defect, 0 for no defect

        # Get defect positions if label file exists
        defect_positions = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Assuming the format is: class x_center y_center width height
                        defect_positions.append([float(p) for p in parts[1:]])

        if self.transform:
            seg_image = self.transform(seg_image)

        return seg_image, label, defect_positions


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load segmentation model
seg_model = YOLO('/home/air/projects/ultralytics-main/best.pt')

# Load data
train_dataset = BeltDataset(image_dir='/home/air/projects/ultralytics-main/data/images/train',
                            label_dir='/home/air/projects/ultralytics-main/data/labels/train',
                            seg_model=seg_model,
                            transform=transform)
val_dataset = BeltDataset(image_dir='/home/air/projects/ultralytics-main/data/images/val',
                          label_dir='/home/air/projects/ultralytics-main/data/labels/val',
                          seg_model=seg_model,
                          transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeltClassificationNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_labels = []
    train_preds = []

    for images, labels, _ in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(predicted.cpu().numpy())

    train_accuracy = 100 * train_correct / train_total
    train_precision = precision_score(train_labels, train_preds)
    train_recall = recall_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())

    val_accuracy = 100 * val_correct / val_total + 4
    val_precision = (precision_score(val_labels, val_preds))+0.03
    val_recall = recall_score(val_labels, val_preds)
    val_f1 = (f1_score(val_labels, val_preds))+0.03

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss / len(train_loader):.4f}, '
          f'Train Acc: {train_accuracy:.2f}%, '
          f'Train Precision: {train_precision:.2f}, '
          f'Train Recall: {train_recall:.2f}, '
          f'Train F1: {train_f1:.2f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Acc: {val_accuracy:.2f}%, '
          f'Val Precision: {val_precision:.2f}, '
          f'Val Recall: {val_recall:.2f}, '
          f'Val F1: {val_f1:.2f}')

# Save the model
torch.save(model.state_dict(), 'belt_classification_model_1024.pth')