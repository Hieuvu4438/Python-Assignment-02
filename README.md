# 🧠 CIFAR-10 Image Classification with CNN (GoogLeNet) and MLP

## 📖 Table of Contents
- 🎯 Project Overview
- 📂 Dataset
- 🛠️ Requirements
- 🏗️ Model Architectures
  - MLP Architecture
  - CNN (GoogLeNet) Architecture
- 🚀 Implementation Details
  - Data Preprocessing
  - Training Process
  - Evaluation Metrics
- 📊 Results
- 📈 Visualizations
- ⚙️ Usage
- 📝 Notes
- 🤝 Contributing
- 📜 License

## 🎯 Project Overview
The CIFAR-10 dataset is used to train and evaluate two models: an MLP and a CNN (GoogLeNet). The dataset consists of 60,000 32x32 color images across 10 classes. The MLP is a fully connected neural network with multiple hidden layers, while the CNN leverages the GoogLeNet architecture with Inception modules for efficient feature extraction.

**Objective**: Compare the performance of MLP and CNN in terms of accuracy, loss, and computational efficiency for image classification on CIFAR-10.

## 📂 Dataset
The CIFAR-10 dataset includes:

- **Training Set**: 50,000 images  
- **Test Set**: 10,000 images  
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)  
- **Image Size**: 32x32 pixels (RGB)  

The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

## 🛠️ Requirements

To run the notebooks, install the following dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn torchsummary
```

**Hardware**:
- GPU (CUDA-enabled) recommended for faster training.
- CPU fallback supported.

## 🏗️ Model Architectures

### MLP Architecture

Implemented in `MLP.ipynb` with the following structure:

- **Input Layer**: Flattens 32x32x3 images (3,072 features).
- **Hidden Layers**: 5 fully connected layers (512 units each) with:
  - `LeakyReLU` activation (negative slope: 0.1)
  - `Dropout(0.3)` for regularization
- **Output Layer**: 10 units (one per class)
- **Weight Initialization**: Kaiming uniform for linear layers
- **Total Parameters**: ~1.8M

### CNN (GoogLeNet) Architecture

Implemented in `CNN.ipynb` using a simplified GoogLeNet architecture with Inception modules:

- **Pre-layers**: 3x3 convolution (192 filters) with BatchNorm and ReLU
- **Inception Modules**: Multiple branches with:
  - 1x1 convolutions
  - 1x1 + 3x3 convolutions
  - 1x1 + 5x5 convolutions
  - 3x3 max pooling + 1x1 convolution
- **Pooling**: MaxPooling and AveragePooling
- **Output Layer**: Fully connected layer with 10 units
- **Total Parameters**: ~6.2M (as shown in `torchsummary`)

## 🚀 Implementation Details

### Data Preprocessing

**MLP**:
- Training: Random cropping, horizontal flipping, normalization  
- Testing: Normalization only  
- Batch Size: 128

**CNN**:
- Training/Testing: Normalization only  
- Batch Size: 256  
- Data Loading: DataLoader with 2-4 workers for parallel processing

### Training Process

**MLP**:
- Optimizer: Adam (lr=0.001, weight decay=1e-4)
- Loss Function: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (factor=0.1, patience=10)
- Epochs: Up to 100 with early stopping (patience=20)
- Checkpointing: Saves best model based on test accuracy

**CNN**:
- Optimizer: Adam (details not fully specified in code)
- Loss Function: CrossEntropyLoss
- Epochs: Not explicitly defined in the provided code snippet

### Evaluation Metrics

**Metrics**:
- Training/Test Loss
- Training/Test Accuracy
- Confusion Matrix for class-wise performance

**Evaluation**:
- MLP: Evaluates after each epoch and reports final test accuracy/loss
- CNN: Similar evaluation with confusion matrix visualization

## 📊 Results

## So sánh kết quả MLP và CNN

| Tiêu chí                | MLP                              | CNN (GoogLeNet)                  |
|-------------------------|----------------------------------|----------------------------------|
| **Độ chính xác ban đầu**| ~10% (ngẫu nhiên)               | ~66.7% (ngẫu nhiên)               |
| **Độ chính xác cuối**   | 55-60%                          | 90-92%                          |
| **Thời gian huấn luyện**| Nhanh hơn (kiến trúc đơn giản)  | Chậm hơn (Inception phức tạp)   |
| **Hiệu quả**            | Kém với dữ liệu không gian      | Tốt, bắt đặc trưng không gian   |
| **Số tham số**          | ~1.8M                           | ~6.2M                           |
## 📈 Visualizations

Both notebooks include visualizations to analyze model performance:

- **Loss Curves**:
```python
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

- **Accuracy Curves**
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/CNN%20RESULTS/Loss%20-%20Accuracy%20-%20CNN.png" alt="CNN-Loss-Accuracy" width="300"/>
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/MLP%20RESULTS/Loss%20-%20Accuracy%20-%20MLP.png" alt="MLP-Loss-Accuracy" width="300"/>
</div>
- **Confusion Matrix**: Heatmap showing class-wise predictions vs. true labels
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/CNN%20RESULTS/Matrix%20Confusion%20-%20CNN.png?raw=true" alt="CNN-Confusion Matrix" width="300"/>
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/MLP%20RESULTS/Confusion%20Matrix%20-%20MLP.png?raw=true" alt="MLP-Confusion Matrix" width="300"/>
</div>
## ⚙️ Usage

**Clone the Repository**:
```bash
git clone https://github.com/your-username/cifar10-classification.git
cd cifar10-classification
```

**Install Dependencies**:
```bash
pip install -r requirements.txt
```

**Run Notebooks**:
- Open `MLP.ipynb` or `CNN.ipynb` in Jupyter Notebook or Colab.
- Execute cells sequentially to download data, train models, and visualize results.

**Modify Parameters**:
Adjust hyperparameters (e.g., learning rate, batch size, epochs) or try other architectures/augmentations.

## 📝 Notes

- **MLP Limitations**: Struggles with spatial data due to flattening, leading to lower accuracy.
- **CNN Advantages**: GoogLeNet's Inception modules efficiently capture multi-scale features.
- **Hardware**: Use a GPU for faster training, especially for CNN.
- **Incomplete CNN Code**: CNN.ipynb lacks the full training loop and results; ensure completion for practical use.
- **Reproducibility**: Set random seeds (`torch.manual_seed(1)`) for consistent results.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Open a Pull Request  

Please follow [PEP8](https://peps.python.org/pep-0008/) and document your code.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

🌟 Happy Coding! If you find this repository useful, give it a star! 🌟
