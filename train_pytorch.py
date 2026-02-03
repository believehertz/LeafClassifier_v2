import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split

print("🔥 Training PyTorch Leaf Classifier...")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class LeafDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

def load_data():
    leaf_dir = 'data/train/leaf'
    non_dir = 'data/train/non_leaf'
    
    leaf_files = [os.path.join(leaf_dir, f) for f in os.listdir(leaf_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(leaf_dir) else []
    non_files = [os.path.join(non_dir, f) for f in os.listdir(non_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(non_dir) else []
    
    print(f"Found {len(leaf_files)} leaf, {len(non_files)} non-leaf")
    
    if len(leaf_files) == 0 or len(non_files) == 0:
        print("❌ ERROR: Put images in data/train/leaf/ and data/train/non_leaf/")
        exit()
    
    all_files = leaf_files + non_files
    all_labels = [1]*len(leaf_files) + [0]*len(non_files)
    return train_test_split(all_files, all_labels, test_size=0.2, random_state=42)

train_files, val_files, train_labels, val_labels = load_data()

train_ds = LeafDataset(train_files, train_labels, train_transform)
val_ds = LeafDataset(val_files, val_labels, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Model
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_acc = 0.0
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_correct, train_total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Val
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    train_acc = 100 * train_correct / train_total
    print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_acc:.1f}%, Val: {val_acc:.1f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'model_pytorch.pth')
        print("  💾 Saved best model!")

print(f"\n✅ Done! Best accuracy: {best_acc:.1f}%")