# 🧠 Making Of — Vision Transformer Image Classification

This file documents the exact steps and learning process followed while building this project.

---

## ⚙️ Environment Setup

* Faced compatibility issues with newer Python versions
* Installed Python **3.10.2** for stable support

### Install required libraries:

```
& "C:\Program Files\Python310\python.exe" -m pip install torch torchvision transformers matplotlib
```

---

## 📦 Dataset Loading (CIFAR-10)

### Key Concepts:

* **transforms.Resize((224, 224))**
  → Vision Transformer requires fixed input size

* **transforms.ToTensor()**
  → Converts image into numerical format

* **datasets.CIFAR10(...)**
  → Loads dataset with:

  * 10 classes
  * 50,000 training images

* **download=True**
  → Automatically downloads dataset

### Run training file:

```
& "C:\Program Files\Python310\python.exe" train.py
```

---

## 🧠 Vision Transformer (ViT)

### Key Concepts:

* **ViTForImageClassification**
  → Pretrained Vision Transformer model

* **from_pretrained(...)**
  → Loads pretrained weights (no training from scratch)

* **num_labels = 10**
  → Adjusts output layer for CIFAR-10 classes

---

## 🔁 Training Loop

### Core Components:

* **DataLoader**
  → Feeds data in batches

* **device (CPU/GPU)**
  → Uses GPU if available

* **optimizer (Adam)**
  → Updates model weights

---

### Training Process:

Forward pass → Prediction
Loss calculation → Error
Backward pass → Gradients
Optimizer step → Weight update

👉 This is how the model learns

---

## 💾 Model Saving (Important Step)

After training, model is saved:

```
torch.save(model.state_dict(), "vit_model.pth")
```

👉 This allows reuse without retraining

---

## 🔄 Model Loading (After Restart / New Session)

Instead of training again:

```
model.load_state_dict(torch.load("vit_model.pth"))
```

👉 Enables direct prediction without retraining

---

## 🔍 Prediction (Testing Model)

### Dataset-based testing:

train_data[index] → selects image
unsqueeze(0) → converts to batch format
model.eval() → switches to evaluation mode
argmax → selects highest probability class

---

## 🖼️ External Image Prediction

files.upload() → upload image
Image.open() → load image
Resize → match model input
ToTensor() → convert to tensor
unsqueeze(0) → batch format
model(pixel_values) → prediction
argmax → predicted class

---

## ⚠️ Limitations

* Model is trained on **CIFAR-10 only**
* Can predict only these 10 classes:

airplane → automobile → bird → cat → deer → dog → frog → horse → ship → truck

---

## 🚀 Key Learnings

* How pretrained models are fine-tuned
* Importance of correct environment setup
* Difference between training vs inference
* How to save and reuse ML models
* End-to-end ML workflow

---

## 🔄 Workflow Summary

Training → Save → Load → Predict

* Training = one-time
* Loading = every session
* Prediction = final usage

---

## 📌 Status

🚧 Work in Progress
Next steps:

* Improve accuracy
* Add UI
* Optimize training

---
