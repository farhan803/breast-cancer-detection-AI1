# ==============================
# Breast Cancer Detection
# Support Vector Machine (SVM)
# ==============================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Create models folder
# ==========================

os.makedirs("models", exist_ok=True)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("breast-cancer.csv")

# ==========================
# Data Cleaning
# ==========================

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# ==========================
# Encode Target
# ==========================

if df["diagnosis"].dtype == object:
    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0,
        "Malignant": 1,
        "Benign": 0
    })

# ==========================
# Split Dataset
# ==========================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Train SVM
# ==========================

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("SVM Accuracy :", round(accuracy * 100, 2), "%")
print("=" * 50)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Save Model
# ==========================

joblib.dump(model, "models/svm.pkl")
joblib.dump(scaler, "models/scaler.pkl")

metrics = {
    "accuracy": accuracy,
    "classification_report": classification_report(
        y_test,
        y_pred,
        output_dict=True
    ),
    "confusion_matrix": confusion_matrix(
        y_test,
        y_pred
    )
}

joblib.dump(
    metrics,
    "models/svm_metrics.pkl"
)

print("\n✅ SVM Model Saved Successfully!")