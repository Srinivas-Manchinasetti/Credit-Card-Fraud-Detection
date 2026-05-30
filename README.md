# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using supervised and unsupervised techniques — built with Python, scikit-learn, and Google Colab.

Project Link: 

[![Open in Colab]
(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OHOLF3gIC9Ab5M06AC_BIED7YdTDZJI8?usp=sharing)
[![Open in nbviewer]
(https://img.shields.io/badge/Open%20in-nbviewer-orange)](https://nbviewer.org/github/Srinivas-Manchinasetti/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection.ipynb)

---

## 📌 Project Overview

Financial fraud causes billions in losses every year. This project builds an end-to-end ML pipeline that automatically classifies a credit card transaction as **legitimate or fraudulent**, using a real-world dataset of 284,807 transactions.

The core challenge: only **0.17%** of transactions are fraud — making this a highly imbalanced classification problem that requires careful handling beyond simple accuracy metrics.

---

## 🎯 Objective

- Detect fraudulent transactions with high **recall** (minimize missed fraud)
- Handle severe class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique)
- Compare multiple ML models and select the best using **F1 Score** and **ROC-AUC**

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 features
- **Fraud cases:** 492 (0.17% of total)
- **Features:**
  - `V1–V28`: PCA-transformed anonymized features (provided by the bank)
  - `Amount`: Transaction amount
  - `Time`: Seconds elapsed since first transaction
  - `Class`: Target label — `0 = Legitimate`, `1 = Fraud`

> ⚠️ The dataset is not included in this repository due to size. Download it from Kaggle and place `creditcard.csv` in the root directory.

---

## 🧠 ML Pipeline

```
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Preprocessing — Scale Amount & Time
   ↓
Train/Test Split (80/20, stratified)
   ↓
SMOTE — Applied only on training data
   ↓
Model Training — Logistic Regression + Decision Tree
   ↓
Evaluation — F1, Recall, ROC-AUC, Confusion Matrix
   ↓
Model Comparison & Conclusion
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| scikit-learn | ML models & evaluation |
| imbalanced-learn | SMOTE for class imbalance |
| pandas & numpy | Data manipulation |
| matplotlib & seaborn | Visualizations |
| Google Colab | Development environment (CPU) |

---

## 📊 Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Non-linear, interpretable tree model |

Both models are evaluated on:
- **Precision** — Of predicted frauds, how many are real?
- **Recall** — Of real frauds, how many did we catch?
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Overall discrimination ability

> Accuracy is intentionally **not** used as the primary metric — a model predicting everything as legitimate would achieve 99.83% accuracy while catching zero fraud.

---

## 🔑 Key Concepts Demonstrated

- Handling **class imbalance** with SMOTE
- Correct **train/test split before SMOTE** (prevents data leakage)
- Choosing **appropriate evaluation metrics** for imbalanced data
- **PCA visualization** of fraud vs. legitimate clusters
- End-to-end ML pipeline from raw data to model comparison

---

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the root folder

3. Open `fraud_detection.ipynb` in Google Colab or Jupyter Notebook

4. Run all cells in order (top to bottom)

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.ipynb   # Main Colab notebook
├── README.md               # Project documentation
└── creditcard.csv          # Dataset (not included — download from Kaggle)
```

---

## 📈 Results

| Model | F1 Score | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.xx | ~0.xx | ~0.xx |
| Decision Tree | ~0.xx | ~0.xx | ~0.xx |

> Fill in your actual scores after running the notebook.

---

## 💡 Lessons Learned

- Raw accuracy is misleading for imbalanced datasets
- SMOTE must only be applied **after** train/test split to avoid data leakage
- Recall is often more important than precision in fraud detection (missing fraud is costly)
- Real-world datasets are messy — preprocessing and feature scaling matter

---

## 🙋 Author

**Srinivas**
Student — School of Computer Science and Engineering, VIT-AP University

---

