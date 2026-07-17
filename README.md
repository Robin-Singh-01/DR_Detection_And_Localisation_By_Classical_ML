# 🩺 Diabetic Retinopathy Detection & Lesion Localisation using Classical Machine Learning

An end-to-end **Computer Vision** and **Machine Learning** project that detects **Diabetic Retinopathy (DR)** from retinal fundus images and localizes disease-affected regions using classical image processing techniques.

Unlike many deep learning-based solutions, this project demonstrates how carefully engineered image features combined with traditional machine learning algorithms can achieve meaningful medical image analysis while remaining computationally efficient and interpretable.

---

## 📌 Project Overview

Diabetic Retinopathy is one of the leading causes of preventable blindness worldwide. Early detection is essential for timely treatment and preventing vision loss.

This project implements a complete classical machine learning pipeline that:

- Processes retinal fundus images
- Extracts handcrafted image features
- Performs feature scaling
- Reduces feature dimensionality using PCA
- Trains a Support Vector Machine (SVM) classifier
- Localizes suspicious retinal lesions

The project demonstrates how classical Computer Vision techniques can still be effective for medical image analysis while remaining lightweight, interpretable, and computationally efficient.

---

## 🎯 Objectives

- Detect Diabetic Retinopathy from retinal fundus images
- Localize abnormal retinal regions
- Demonstrate an end-to-end Classical Machine Learning pipeline
- Apply Computer Vision techniques for medical image analysis
- Build a lightweight and interpretable prediction system

---

## 🧠 Machine Learning Pipeline

```text
Retinal Fundus Image
        │
        ▼
Image Preprocessing
        │
        ▼
Feature Extraction
        │
        ▼
Feature Scaling
        │
        ▼
Principal Component Analysis (PCA)
        │
        ▼
Support Vector Machine (SVM)
        │
        ▼
Prediction & Lesion Localisation
```

---

## 🚀 Features

- Classical Computer Vision pipeline
- Medical image preprocessing
- Handcrafted feature extraction
- Principal Component Analysis (PCA)
- Support Vector Machine (SVM)
- Lesion localization
- Lightweight and interpretable model
- Modular implementation

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|------|
| Language | Python |
| Computer Vision | OpenCV |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Model Serialization | Joblib |

---

## 🔬 Methodology

### 1. Image Preprocessing

Retinal fundus images are preprocessed to improve image quality before feature extraction.

Preprocessing includes:

- Image normalization
- Noise reduction
- Contrast enhancement
- Image preparation for feature extraction

---

### 2. Feature Extraction

Instead of using deep learning embeddings, handcrafted image features are extracted to represent texture and structural characteristics of retinal images.

This provides:

- Better interpretability
- Faster inference
- Lower computational requirements

---

### 3. Feature Scaling

The extracted features are standardized before model training to ensure consistent feature distributions.

---

### 4. Principal Component Analysis (PCA)

PCA reduces the dimensionality of the extracted features while preserving the most informative components.

Benefits include:

- Reduced computational cost
- Reduced feature redundancy
- Faster model training
- Better generalization

---

### 5. Classification

A **Support Vector Machine (SVM)** with an RBF kernel is trained on the transformed feature space to classify retinal images.

SVM was selected because of its strong performance on high-dimensional datasets and relatively small training datasets.

---

### 6. Lesion Localisation

After classification, the system highlights suspicious retinal regions using classical Computer Vision techniques, improving model interpretability compared to only providing a prediction label.

---

## 📂 Project Structure

```text
DR_Detection_And_Localisation_By_Classical_ML/

├── Models/
│   ├── feat_scaler.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── svm.pkl
│
├── Dataset/
├── app.py
├── V0IVPproject.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/DR_Detection_And_Localisation_By_Classical_ML.git
```

Navigate to the project directory

```bash
cd DR_Detection_And_Localisation_By_Classical_ML
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## 📊 Dataset

This project uses the **Indian Diabetic Retinopathy Image Dataset (IDRiD)**.

The dataset contains high-resolution retinal fundus images and corresponding annotations for diabetic retinopathy research.

**Dataset:**  
https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

---

## 🌍 Applications

- Automated diabetic retinopathy screening
- Clinical decision support
- Medical image analysis research
- Computer Vision education
- Healthcare AI applications

---

## 🔮 Future Improvements

- Integrate CNN-based models
- Vision Transformer implementation
- Multi-class DR severity classification
- Explainable AI (Grad-CAM / SHAP)
- REST API deployment
- Docker support
- Cloud deployment
- Improved lesion segmentation

---

## 📚 Skills Demonstrated

- Computer Vision
- Image Processing
- Feature Engineering
- Machine Learning
- Principal Component Analysis (PCA)
- Support Vector Machines (SVM)
- Data Preprocessing
- Medical Image Analysis
- Model Deployment

---

## 👨‍💻 Author

**Robin Singh**

If you found this project helpful, consider giving it a ⭐ on GitHub.
