# 🩺 Breast Cancer Detection System

A Machine Learning based web application that predicts whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using different classification algorithms.

The system provides prediction results with confidence score and visual analytics through an interactive Streamlit dashboard.

---

# 🚀 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection plays an important role in improving treatment success.

This project uses Machine Learning techniques to analyze medical features and classify breast cancer cases.

The application allows users to:

- Enter patient medical measurements
- Select different ML models
- Predict cancer type
- Compare model performance
- Analyze dataset visually

---

# ✨ Features

## 🩺 Prediction System

- Real-time breast cancer prediction
- Multiple Machine Learning model support
- Confidence score display
- Benign/Malignant classification


## 📊 Dashboard

- Dataset overview
- Statistical analysis
- Feature distribution
- Correlation heatmap
- Visualization charts


## 📈 Model Comparison

Compare different algorithms:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix


## 📥 Data Export

- Download dataset
- Download statistics report
- Download model comparison results

---

# 🤖 Machine Learning Models

The project uses the following classification algorithms:

| Model | Type |
|---|---|
| Logistic Regression | Linear Classification |
| Decision Tree | Tree Based Learning |
| Random Forest | Ensemble Learning |
| K-Nearest Neighbors | Distance Based Learning |
| Support Vector Machine | Kernel Based Learning |
| Naive Bayes | Probabilistic Classification |

---

# 📂 Project Structure

```
BreastCancerDetection/

│
├── app.py

│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Prediction.py
│   ├── 3_Dashboard.py
│   ├── 4_Model_Comparison.py
│   ├── 5_About.py
│   └── 6_Contact.py

│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── naive_bayes.pkl
│   └── scaler.pkl

│
├── breast-cancer.csv

│
├── requirements.txt

│
└── README.md
```

---

# 📊 Dataset Information

Dataset:

**Breast Cancer Wisconsin Diagnostic Dataset**

Dataset contains:

- 569 patient samples
- 30 medical features
- Binary classification target

Classes:

```
0 → Benign
1 → Malignant
```

Features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

---

# 🛠️ Technology Stack

## Programming Language

- Python


## Machine Learning

- Scikit-Learn


## Data Processing

- Pandas
- NumPy


## Visualization

- Matplotlib
- Seaborn
- Plotly


## Web Application

- Streamlit


## Model Saving

- Joblib

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/BreastCancerDetection.git
```

Go to project folder:

```bash
cd BreastCancerDetection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

Start Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📌 Workflow

```
Dataset
   |
   ↓
Data Cleaning
   |
   ↓
Feature Processing
   |
   ↓
Model Training
   |
   ↓
Model Evaluation
   |
   ↓
Prediction
   |
   ↓
Result Visualization
```

---

# 📈 Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

---

# ⚠️ Disclaimer

This application is developed for educational and research purposes only.

It is not a replacement for professional medical diagnosis.

Always consult qualified healthcare professionals for medical decisions.

---

# 👨‍💻 Developer

**Farhan**

Diploma in Computer Science & Technology

Project Category:

Machine Learning + Healthcare AI

---

# ⭐ Future Improvements

Future upgrades may include:

- Deep Learning Model
- CNN Based Cancer Detection
- Cloud Deployment
- User Authentication
- Patient History Management
- AI Medical Assistant Integration

---

# 📜 License

This project is created for educational purposes.