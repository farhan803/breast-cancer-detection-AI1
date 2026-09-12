# ==============================
# Breast Cancer Detection
# Random Forest Model
# ==============================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
# Split Data
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
# Train Model
# ==========================

model = RandomForestClassifier(
    n_estimators=200,
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
print("Accuracy :", round(accuracy * 100, 2), "%")
print("=" * 50)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Save Model
# ==========================

joblib.dump(model, "models/random_forest.pkl")
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
    "models/random_forest_metrics.pkl"
)

print("\n✅ Random Forest Model Saved Successfully!")