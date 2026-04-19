#train.py

from torchvision import datasets, transforms
# Step 1: Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize for ViT
    transforms.ToTensor(),          # convert image to tensor
])

# Step 2: Load dataset
train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Step 3: Print dataset size
print("Number of training images:", len(train_data))

# Load pretrained Vision Transformer
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

print("Model loaded successfully")

#STEP 4: Training Loop (Model Learns Now)

import torch
from torch.utils.data import DataLoader

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(1):   # 1 epoch for now
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch completed, Loss: {total_loss:.4f}")
#saves to Google drive with name vit_model.pth
torch.save(model.state_dict(), "/content/drive/MyDrive/vit_model.pth")
print("Model saved to Google Drive")
