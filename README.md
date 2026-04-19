# 🧠 Vision Transformer Image Classification (ViT)

## 🚀 Overview

This project implements an image classification system using a **Vision Transformer (ViT)** model fine-tuned on the **CIFAR-10 dataset**.

The goal was to understand and build a complete ML pipeline:
**data → model → training → saving → loading → prediction**

---

## 🧠 Model Details

* Model: `google/vit-base-patch16-224`
* Framework: PyTorch + Hugging Face Transformers
* Dataset: CIFAR-10 (10 classes)
* Fine-tuning: Output layer adjusted to 10 classes

---

## 📂 Project Structure

```id="0kz0gt"
vit-image-classification/
├── train.py          # Training + saving model
├── test.py           # Prediction on images
├── app.py            # (Optional UI - in progress)
├── requirements.txt
├── README.md
├── MakingOf.md       # Development notes (step-by-step process)
```

---

## ⚙️ Installation

### 1. Install Python (Recommended: 3.10)

### 2. Install dependencies

```id="q7l4xk"
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Run:

```id="8c66hc"
python train.py
```

What happens:

* Loads CIFAR-10 dataset
* Loads pretrained ViT
* Fine-tunes model
* Saves model as `vit_model.pth`

---

## 💾 Model Saving

After training:

```id="m4q5pp"
vit_model.pth
```

👉 This file stores trained weights (no need to retrain every time)

---

## 🔄 Using the Model (Prediction)

Run:

```id="qrgk60"
python test.py
```

What happens:

* Loads saved model
* Takes input image
* Predicts class

---

## 🖼️ Supported Classes (CIFAR-10)

```id="2h3r8z"
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

⚠️ Model will always predict one of these 10 classes

---

## 🧪 Example Output

```id="bg0b8g"
Predicted: cat
```

---

## ⚠️ Limitations

* Trained only for 1 epoch (basic accuracy)
* Limited dataset (CIFAR-10 only)
* Predictions may not be highly accurate yet

---

## 🧠 Learnings

* Fine-tuning pretrained transformers for vision tasks
* Handling model mismatch (1000 → 10 classes)
* Training vs inference workflow
* Saving and loading models
* GPU usage via Google Colab

---

## 🚧 Project Status

Work in Progress

Planned improvements:

* Improve accuracy (more training epochs)
* Add proper evaluation (accuracy metric)
* Build UI using Streamlit
* Support custom image inputs

---

## 🙌 Author

Golu Kumar Singh

---

## ⭐ Notes

This project is part of my **learning journey in AI/ML**, documenting real challenges and solutions while building with Vision Transformers.
