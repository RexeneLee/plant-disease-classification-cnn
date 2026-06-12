# Plant Disease Classification using Transfer Learning

## Project Overview

This project investigates the performance of transfer learning based Convolutional Neural Network (CNN) architectures for plant leaf disease classification.

The study compares three popular deep learning models:

- ResNet50
- EfficientNetB0
- MobileNetV2

The models are evaluated using the PlantVillage dataset and compared based on classification performance and computational efficiency.

---

## Dataset

PlantVillage Dataset

The dataset contains healthy and diseased plant leaf images belonging to multiple crop species.

Dataset Source:

https://www.kaggle.com/datasets/emmarex/plantdisease

---

## Methodology

### Data Preprocessing

- Image resizing (224x224)
- Data augmentation
- Train/Validation/Test split

### Deep Learning Models

1. ResNet50
2. EfficientNetB0
3. MobileNetV2

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

## Results

The comparative analysis demonstrates that EfficientNetB0 achieves the best balance between classification accuracy and computational cost.

---

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-Learn

---

## Author

Muhammed Emin Taş

Department of Artificial Intelligence and Data Engineering

Ankara University

---

## Project Type

Deep Learning Course Project

## Future Work

- Vision Transformers (ViT)
- Explainable AI (Grad-CAM)
- Ensemble Learning
- Real-world field image testing
