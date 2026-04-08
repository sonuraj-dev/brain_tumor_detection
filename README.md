# 🧠 Brain Tumor Classification using Deep Learning

## 📌 Project Overview
This project focuses on classifying brain tumors from MRI images using deep learning models. The goal is to assist in early detection and diagnosis of brain tumors.

---

## 🎯 Objective
To develop and compare multiple deep learning models (ResNet, DenseNet, Xception) for accurate classification of brain tumors.

---

## 🧪 Dataset
- Source: Kaggle Brain Tumor MRI Dataset  
- Classes:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor  

---

## ⚙️ Models Used
- ResNet50  
- DenseNet121  
- Xception  

---

## 🏗️ Methodology
1. Data preprocessing (resize, augmentation)
2. Train/Validation/Test split
3. Transfer learning using pretrained models
4. Model training (20 epochs)
5. Evaluation using metrics

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-score |
|------|---------|----------|--------|----------|
| ResNet | ~98% | 0.97 | 0.97 | 0.97 |
| DenseNet | ~98% | ~0.98 | ~0.98 | ~0.98 |
| Xception | ~98% | ~0.98 | ~0.98 | ~0.98 |

---

## 📈 Confusion Matrix
- Included for all models in the repository

---

## 🧠 Key Insights
- All models achieved high accuracy (>97%)
- DenseNet and Xception performed slightly better
- Meningioma class showed slight confusion

---

## 🛠️ Tech Stack
- Python
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- seaborn

---

## 🚀 How to Run

### 1. Clone repo
```bash
git clone https://github.com/sonuraj-dev/brain_tumor_detection.git
cd brain_tumor_project