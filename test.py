import matplotlib.pyplot as plt

# Get one sample image
image, label = train_data[0]

# Move to device
model.eval()
with torch.no_grad():
    outputs = model(pixel_values=image.unsqueeze(0).to(device))
    pred = outputs.logits.argmax(dim=1).item()

# Class names
classes = train_data.classes

# Show image
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Predicted: {classes[pred]}")
plt.axis('off')
plt.show()

print("Actual:", classes[label])