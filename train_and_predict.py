"""
train_and_predict.py
---------------------
This is the main script for the PLM + AI tolerance optimization project.

What it does, step by step:
  1. Loads the training dataset (data/tolerance_training_data.csv)
  2. Trains two ML models:
       - A Classifier  -> predicts Status (PASS / FAIL)
       - A Regressor   -> predicts Cost_Index (estimated manufacturing cost)
  3. Prints accuracy / error metrics so you know how good the model is
  4. Loads your new parts (data/new_parts.csv)
  5. Predicts Status and Cost for each new part
  6. Saves everything to outputs/predictions.csv
  7. Saves a short text report to outputs/report.txt
  8. Saves two charts (PNG images) to outputs/ for your presentation:
       - tolerance_vs_status.png
       - predicted_cost_by_part.png

Run it with:
    python train_and_predict.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it works even without a display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os

DATA_DIR = "data"
OUT_DIR = "outputs"
MODEL_DIR = "model"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL_COLS = ["Part_Name", "Manufacturing_Process", "Material", "Fit_Type"]
NUMERIC_COLS = ["Nominal_Dimension_mm", "Tolerance_mm", "Allowable_Tolerance_mm"]

# ---------------------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading training dataset")
print("=" * 60)
train_path = os.path.join(DATA_DIR, "tolerance_training_data.csv")
df = pd.read_csv(train_path)
print(f"Loaded {len(df)} training rows from {train_path}")

# ---------------------------------------------------------------------
# 2. Encode categorical columns
# ---------------------------------------------------------------------
encoders = {}
df_encoded = df.copy()
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

status_encoder = LabelEncoder()
df_encoded["Status_enc"] = status_encoder.fit_transform(df_encoded["Status"])

feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
X = df_encoded[feature_cols]
y_status = df_encoded["Status_enc"]
y_cost = df_encoded["Cost_Index"]

# ---------------------------------------------------------------------
# 3. Train / test split + train models
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Training models")
print("=" * 60)

X_train, X_test, ys_train, ys_test, yc_train, yc_test = train_test_split(
    X, y_status, y_cost, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, ys_train)
status_pred_test = clf.predict(X_test)
acc = accuracy_score(ys_test, status_pred_test)
print(f"Status (PASS/FAIL) classifier accuracy: {acc*100:.1f}%")

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train, yc_train)
cost_pred_test = reg.predict(X_test)
mae = mean_absolute_error(yc_test, cost_pred_test)
print(f"Cost regressor mean absolute error: {mae:.2f} cost units")

# Save trained models + encoders so the GUI (app.py) can reuse them
joblib.dump(clf, os.path.join(MODEL_DIR, "status_classifier.pkl"))
joblib.dump(reg, os.path.join(MODEL_DIR, "cost_regressor.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(status_encoder, os.path.join(MODEL_DIR, "status_encoder.pkl"))
print(f"Models saved to {MODEL_DIR}/")

# ---------------------------------------------------------------------
# 4. Load new parts and predict
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Predicting on new_parts.csv")
print("=" * 60)

new_path = os.path.join(DATA_DIR, "new_parts.csv")
new_df = pd.read_csv(new_path)
print(f"Loaded {len(new_df)} new parts from {new_path}")

new_encoded = new_df.copy()
for col in CATEGORICAL_COLS:
    le = encoders[col]
    # handle any category value not seen during training
    new_encoded[col] = new_encoded[col].apply(
        lambda v: v if v in le.classes_ else le.classes_[0]
    )
    new_encoded[col] = le.transform(new_encoded[col])

X_new = new_encoded[feature_cols]
status_pred = clf.predict(X_new)
status_pred_labels = status_encoder.inverse_transform(status_pred)
cost_pred = reg.predict(X_new)

results = new_df.copy()
results["Predicted_Status"] = status_pred_labels
results["Predicted_Cost_Index"] = np.round(cost_pred, 2)

results_path = os.path.join(OUT_DIR, "predictions.csv")
results.to_csv(results_path, index=False)
print(f"Predictions saved to {results_path}")
print(results[["Part_ID", "Part_Name", "Tolerance_mm", "Predicted_Status", "Predicted_Cost_Index"]])

# ---------------------------------------------------------------------
# 5. Text report
# ---------------------------------------------------------------------
report_lines = [
    "AI-BASED TOLERANCE OPTIMIZATION - PREDICTION REPORT",
    "=" * 55,
    f"Training rows used:      {len(df)}",
    f"Classifier accuracy:     {acc*100:.1f}%",
    f"Cost regressor MAE:      {mae:.2f}",
    "",
    f"New parts evaluated:     {len(results)}",
    f"  PASS: {(results['Predicted_Status'] == 'PASS').sum()}",
    f"  FAIL: {(results['Predicted_Status'] == 'FAIL').sum()}",
    "",
    "Per-part results:",
]
for _, row in results.iterrows():
    report_lines.append(
        f"  {row['Part_ID']:<8} {row['Part_Name']:<15} "
        f"Tol={row['Tolerance_mm']:<8} -> {row['Predicted_Status']:<5} "
        f"(est. cost {row['Predicted_Cost_Index']})"
    )

report_path = os.path.join(OUT_DIR, "report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nText report saved to {report_path}")

# ---------------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Generating charts")
print("=" * 60)

# Chart 1: Tolerance vs Status scatter (training data, colored by pass/fail)
plt.figure(figsize=(7, 5))
colors = df["Status"].map({"PASS": "#2e7d32", "FAIL": "#c62828"})
plt.scatter(df["Nominal_Dimension_mm"], df["Tolerance_mm"], c=colors, alpha=0.6, s=25)
plt.xlabel("Nominal Dimension (mm)")
plt.ylabel("Tolerance (mm)")
plt.title("Training Data: Tolerance vs Dimension (Green=PASS, Red=FAIL)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tolerance_vs_status.png"), dpi=150)
plt.close()

# Chart 2: Predicted cost per new part, colored by predicted status
plt.figure(figsize=(8, 5))
bar_colors = ["#2e7d32" if s == "PASS" else "#c62828" for s in results["Predicted_Status"]]
plt.bar(results["Part_ID"], results["Predicted_Cost_Index"], color=bar_colors)
plt.xlabel("Part ID")
plt.ylabel("Predicted Cost Index")
plt.title("Predicted Manufacturing Cost per New Part")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "predicted_cost_by_part.png"), dpi=150)
plt.close()

print(f"Charts saved to {OUT_DIR}/tolerance_vs_status.png and {OUT_DIR}/predicted_cost_by_part.png")

print("\n" + "=" * 60)
print("DONE. Open the 'outputs' folder to see predictions.csv, report.txt, and charts.")
print("=" * 60)
