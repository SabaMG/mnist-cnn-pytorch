# 🧠 MNIST Handwritten Digit Classifier (PyTorch)

This project implements a high-accuracy digit classification model on the MNIST dataset using modern deep learning techniques in **PyTorch**.

> 🎯 Final test accuracy: **99.59%** (with 4-layer CNN + data augmentation + regularization)

---

## 🚀 Features

- ✅ Convolutional Neural Network (CNN) for image classification  
- ✅ Batch Normalization and Dropout for regularization  
- ✅ Data Augmentation for improved generalization  
- ✅ Training reproducibility with fixed random seeds  
- ✅ Best-model saving during training (checkpointing)  
- ✅ TensorBoard logging for metrics and visualizations  
- ✅ MPS backend support for Apple Silicon (e.g., M3 Max)

---

## 📁 Project Structure

```
.
├── data/               # MNIST dataset (auto-downloaded)
├── model.py            # CNN architecture
├── train.py            # Training loop
├── evaluate.py         # Evaluation logic
├── dataset.py          # Data loading and transforms
├── utils.py            # Helpers (seed, weight init, etc.)
├── config.py           # Configs (device, paths, etc.)
├── main.py             # Main launcher
├── requirements.txt    # Dependencies
├── best_model.pth      # (optional) Best saved model
├── runs/               # TensorBoard logs
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

The best model will be saved automatically as `best_model.pth`.

### 4. Launch TensorBoard (optional)

```bash
tensorboard --logdir=runs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 📊 Model Comparison (Architectures)

| Model Name           | Conv Layers | Pooling Strategy     | Final Accuracy | Max Accuracy | Training Time (approx.) | Notes                                      |
|----------------------|-------------|-----------------------|----------------|--------------|--------------------------|--------------------------------------------|
| `mnist_cnn_baseline` | 2           | MaxPool               | 99.41%         | 99.48%       | ★★★☆☆                   | Baseline simple, robuste et efficace       |
| `cnn_v1_deep_3conv`  | 3           | MaxPool               | **99.48%**         | **99.58%**       | ★★★★☆                   | Meilleure performance globale              |
| `cnn_v2_deep_3conv`  | 3           | MaxPool               | 99.45%         | 99.55%       | ★★★★☆                   | Très proche du meilleur, stable            |
| `CNN_3Conv_Stride`   | 3           | Strided Convolutions  | 99.38%         | 99.48%       | ★★☆☆☆                   | Le plus rapide à entraîner                 |
| `CNN_4Conv`          | 4           | MaxPool               | 99.59%         | 99.59%       | ★★★★★                   | Lourd, long à entraîner mais performant    |

> 🧪 *Training time scale:*  
> ★☆☆☆☆ = very fast — ★★★★★ = long training

---

## 📈 Final Results

| Epochs | Accuracy | Loss     |
|--------|----------|----------|
| 30     | **99.58%**   | ~42      |

Trained with:
- CNN (up to 3 convolutional layers)
- Dropout 0.3
- Batch Normalization
- AdamW optimizer
- Data augmentation: rotation + affine shift
- Early stopping + best model checkpoint
- Reproducibility: fixed random seed
- TensorBoard visual logging

---

## 🛠️ Dependencies

- Python 3.11
- torch
- torchvision
- numpy
- matplotlib *(optional)*
- tensorboard *(optional)*

---

## 📌 Notes

- Optimized for macOS with Apple Silicon (`mps` backend), but works on CPU and CUDA as well.
- You can switch model versions by editing the architecture inside `model.py`.

---

## 🪪 License

MIT License