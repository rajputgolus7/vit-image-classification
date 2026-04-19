from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from google.colab import files
import torch

# Upload image
uploaded = files.upload()

# Get file
img_path = list(uploaded.keys())[0]
image = Image.open(img_path).convert("RGB")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_tensor = transform(image).unsqueeze(0).to(device)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(pixel_values=img_tensor)
    pred = outputs.logits.argmax(dim=1).item()

# Class labels
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Show result
plt.imshow(image)
plt.title(f"Predicted: {classes[pred]}")
plt.axis('off')
plt.show()
