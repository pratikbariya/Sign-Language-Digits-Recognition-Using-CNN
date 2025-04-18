# Sign Language Digits Recognition Using CNN

This project focuses on building a deep learning model using Convolutional Neural Networks (CNNs) to classify American Sign Language (ASL) digits (0–9). The model is trained on the [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset.git) and achieves high accuracy through careful preprocessing and optimization techniques.

---

## 📂 Dataset

- **Source**: [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset.git)
- **Classes**: Digits 0 through 9
- **Image Size**: 64x64 grayscale images
- **Shape**: `(64, 64, 1)`

---

## 🧠 Model Architecture

The model is a deep CNN built using TensorFlow and Keras. Key layers include:

- `Conv2D` layers with ReLU activation and L2 regularization
- `BatchNormalization` and `MaxPooling` after each convolutional block
- `Dense` layers followed by Dropout for regularization
- `Softmax` output layer with 10 units (for digits 0–9)

---

## 🏋️ Training

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 16
- **Epochs**: 70 (with EarlyStopping and ReduceLROnPlateau)
- **Callbacks**:
  - EarlyStopping (patience = 10)
  - ReduceLROnPlateau (patience = 3, factor = 0.5)

---

## 📈 Performance

- **Final Accuracy**: ~95% on validation data
- **Evaluation Metrics**:
  - Precision, Recall, F1-score (per class)
  - Confusion Matrix

---

## 📊 Visualizations

- Training & Validation Accuracy and Loss Plots
- Confusion Matrix Heatmap using Seaborn

---

## 📚 Prerequisites

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

  
