# 🧠 MNIST Handwritten Digit Classifier (PyTorch)

This project implements a high-accuracy digit classification model on the MNIST dataset using modern deep learning techniques in **PyTorch**.

> 🎯 Final test accuracy: **99.48%** (with data augmentation + CNN + regularization)

---

## 🚀 Features

- ✅ Convolutional Neural Network (CNN) for image classification  
- ✅ Batch Normalization and Dropout for regularization  
- ✅ Data Augmentation for improved generalization  
- ✅ Training reproducibility with fixed random seeds  
- ✅ Best-model saving during training (checkpointing)  
- ✅ MPS backend support for Apple Silicon (e.g., M3 Max)

---

## 📁 Project Structure

```
.
├── data/               # MNIST dataset (auto-downloaded)
├── main.py             # Main training & evaluation logic
├── model.py            # CNN architecture with batchnorm & dropout
├── utils.py            # Helpers (e.g., seed setting, model save/load)
├── best_model.pth      # Saved weights of the best model (optional)
└── README.md
```

---

## 🧪 Quickstart

### 1. Clone and create virtual environment

```bash
git clone https://github.com/SabaMG/mnist-cnn-pytorch.git
cd mnist-cnn-pytorch
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

The best model is saved automatically as `best_model.pth`.

---

## 🧾 Results

| Epochs | Accuracy | Loss     |
|--------|----------|----------|
| 30     | 99.48%   | ~45      |

Trained with:
- CNN (2 conv layers + 2 FC layers)
- Dropout 0.3
- BatchNorm
- AdamW optimizer
- Seeded runs
- Data augmentation (rotation, affine shift)

---

## 🛠️ Dependencies

- Python 3.8+
- torch
- torchvision
- numpy
- matplotlib (optional for plotting)

---

## 📌 Notes

- Tested on macOS with Apple Silicon (`mps` backend)
- You can tweak the model or try other datasets like `FashionMNIST` or `CIFAR10` to go further

---

## 🪪 License

MIT License