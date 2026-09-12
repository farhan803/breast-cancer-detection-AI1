# ==============================
# Breast Cancer Detection
# Logistic Regression Model
# ==============================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("breast-cancer.csv")   # নিজের dataset path দিন

# ==========================
# Data Cleaning
# ==========================

# ID থাকলে remove করুন
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Unnamed column থাকলে remove করুন
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ==========================
# Target Encode
# ==========================

# যদি diagnosis কলামে M/B থাকে
if df['diagnosis'].dtype == object:
    df['diagnosis'] = df['diagnosis'].map({
        'M':1,
        'B':0,
        'Malignant':1,
        'Benign':0
    })

# ==========================
# Split Data
# ==========================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
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
# Train Logistic Regression
# ==========================

model = LogisticRegression(
    max_iter=1000,
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

print("="*50)
print("Accuracy :", round(accuracy*100,2), "%")
print("="*50)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Save Model
# ==========================

joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")

# ==========================
# Save Model
# ==========================

joblib.dump(model, "models/logistic_regression.pkl")
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
    "models/logistic_regression_metrics.pkl"
)

print("Model Saved Successfully!")