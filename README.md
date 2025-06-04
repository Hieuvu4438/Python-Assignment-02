# 🧠 CIFAR-10 Image Classification with CNN (GoogLeNet) and MLP

Welcome to this project where we explore two different deep learning approaches for classifying images from the CIFAR-10 dataset:

- 🔍 **GoogLeNet CNN**: A convolutional neural network inspired by the Inception architecture.
- 🧱 **MLP (Multi-Layer Perceptron)**: A fully connected feed-forward neural network for image classification.

---

## 📖 Table of Contents

- 🎯 [Project Overview](#-project-overview)
- 📂 [Dataset](#-dataset)
- 🛠️ [Requirements](#️-requirements)
- 🏗️ [Model Architectures](#-model-architectures)
  - MLP Architecture
  - CNN (GoogLeNet) Architecture
- 🚀 [Implementation Details](#-implementation-details)
  - Data Preprocessing
  - Training Process
  - Evaluation Metrics
- 📊 [Results](#-results)
- 📈 [Visualizations](#-visualizations)
- ⚙️ [Usage](#-usage)
- 📝 [Notes](#-notes)
- 🤝 [Contributing](#-contributing)
- 📜 [License](#-license)

---

## 🎯 Project Overview

This repository demonstrates two models for image classification using the CIFAR-10 dataset:
- A custom GoogLeNet-style Convolutional Neural Network (CNN)
- A fully connected Multi-Layer Perceptron (MLP)

The goal is to compare their effectiveness, training procedures, and evaluation outcomes.

---

## 📂 Dataset

📦 **CIFAR-10**: A dataset of 60,000 32x32 RGB images divided into 10 classes, with 50,000 training and 10,000 testing samples.

```
['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

---

## 🛠️ Requirements

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

You also need Python 3.7+ and Jupyter Notebook.

---

## 🏗️ Model Architectures

### 🧱 MLP Architecture

- Input layer: 32 × 32 × 3 flattened
- Hidden layers: 5 fully connected layers with `LeakyReLU` and `Dropout`
- Output: 10-class classification
- Optimizer: `Adam`, Scheduler: `ReduceLROnPlateau`
- Loss: `CrossEntropyLoss`

### 🔍 CNN (GoogLeNet) Architecture

- Inception blocks combining 1×1, 3×3, 5×5 convolutions + pooling
- Pre-layer: Conv2D + BatchNorm + ReLU
- Deep stacking of inception modules
- Final classifier: Average Pooling + Fully connected layer
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`

---

## 🚀 Implementation Details

### 🧹 Data Preprocessing

- **Train**: Random crop + Horizontal flip + Normalize
- **Test**: Normalize only

### 🏋️ Training Process

- Mini-batch training with `batch_size=128` (MLP) and `256` (CNN)
- Validation accuracy tracked for early stopping
- Learning rate adjusted via scheduler
- Checkpoints saved for best models

### 📏 Evaluation Metrics

- Accuracy
- Loss
- Confusion Matrix

---

## 📊 Results

| Model     | Final Accuracy | Notes                          |
|-----------|----------------|--------------------------------|
| GoogLeNet | ~90-92%        | Better performance, more depth |
| MLP       | ~55-60%        | Simpler architecture           |

---

## 📈 Visualizations

- 📉 Training & validation loss and accuracy curves
- 📊 Confusion matrix for classification performance

---

## ⚙️ Usage

1. Clone the repo:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Run in Jupyter Notebook:
```bash
jupyter notebook
```

3. Open and run:
- `CNN.ipynb` for GoogLeNet model
- `MLP.ipynb` for MLP model

---

## 📝 Notes

- GoogLeNet is more powerful but slower to train.
- MLP is faster but less accurate due to lack of spatial feature learning.
- Code is cleanly modular and can be extended to other datasets (e.g., Fashion-MNIST).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to submit a pull request or create an issue to improve this project.

---

## 📜 License

This project is licensed under the MIT License.

---

⭐ **Star this repository** if you find it useful and educational!
