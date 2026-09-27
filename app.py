"""
app.py
-------
A simple, no-terminal-needed GUI for the tolerance optimization project,
built with Streamlit.

This gives you buttons for:
  - Upload CSV   (upload a new_parts.csv style file)
  - Predict      (run the trained AI model on it)
  - View Results (see a results table + charts on screen)
  - Export Report (download predictions.csv and report.txt)

HOW TO RUN THIS (in VS Code terminal):
    pip install streamlit pandas numpy scikit-learn matplotlib joblib
    streamlit run app.py

It will open automatically in your web browser.

NOTE: Run train_and_predict.py at least once first (or click "Train
Model" below) so that the model/ folder contains trained models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import subprocess
import sys

DATA_DIR = "data"
OUT_DIR = "outputs"
MODEL_DIR = "model"
CATEGORICAL_COLS = ["Part_Name", "Manufacturing_Process", "Material", "Fit_Type"]
NUMERIC_COLS = ["Nominal_Dimension_mm", "Tolerance_mm", "Allowable_Tolerance_mm"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

st.set_page_config(page_title="AI Tolerance Optimization", layout="wide")

st.title("🔧 AI-Based Tolerance Optimization")
st.caption("PLM + AI system for predicting PASS/FAIL and manufacturing cost of mechanical components")

# ---------------------------------------------------------------------
# Sidebar: train model
# ---------------------------------------------------------------------
st.sidebar.header("1. Model")
models_exist = os.path.exists(os.path.join(MODEL_DIR, "status_classifier.pkl"))

if models_exist:
    st.sidebar.success("Trained model found ✅")
else:
    st.sidebar.warning("No trained model yet.")

if st.sidebar.button("🔁 (Re)train model on training dataset"):
    with st.spinner("Training model..."):
        result = subprocess.run(
            [sys.executable, "train_and_predict.py"],
            capture_output=True, text=True
        )
    if result.returncode == 0:
        st.sidebar.success("Model trained successfully.")
        st.sidebar.text(result.stdout[-600:])
    else:
        st.sidebar.error("Training failed. See details below.")
        st.sidebar.text(result.stderr[-800:])

st.sidebar.divider()
st.sidebar.header("2. Upload new parts")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV in the same format as data/new_parts.csv", type=["csv"]
)
use_sample = st.sidebar.checkbox("Use existing data/new_parts.csv instead", value=uploaded_file is None)

# ---------------------------------------------------------------------
# Load input data
# ---------------------------------------------------------------------
input_df = None
if uploaded_file is not None and not use_sample:
    input_df = pd.read_csv(uploaded_file)
elif use_sample:
    sample_path = os.path.join(DATA_DIR, "new_parts.csv")
    if os.path.exists(sample_path):
        input_df = pd.read_csv(sample_path)

if input_df is not None:
    st.subheader("📄 Parts to evaluate")
    st.dataframe(input_df, use_container_width=True)

# ---------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------
predict_clicked = st.button("🚀 Predict", type="primary", disabled=(input_df is None or not models_exist))

if not models_exist:
    st.info("Train the model first using the button in the sidebar.")

if predict_clicked and input_df is not None:
    clf = joblib.load(os.path.join(MODEL_DIR, "status_classifier.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "cost_regressor.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    status_encoder = joblib.load(os.path.join(MODEL_DIR, "status_encoder.pkl"))

    enc_df = input_df.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        enc_df[col] = enc_df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        enc_df[col] = le.transform(enc_df[col])

    X_new = enc_df[FEATURE_COLS]
    status_pred = status_encoder.inverse_transform(clf.predict(X_new))
    cost_pred = np.round(reg.predict(X_new), 2)

    results = input_df.copy()
    results["Predicted_Status"] = status_pred
    results["Predicted_Cost_Index"] = cost_pred

    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, "predictions.csv")
    results.to_csv(results_path, index=False)

    report_lines = [
        "AI-BASED TOLERANCE OPTIMIZATION - PREDICTION REPORT", "=" * 55,
        f"Parts evaluated: {len(results)}",
        f"  PASS: {(results['Predicted_Status'] == 'PASS').sum()}",
        f"  FAIL: {(results['Predicted_Status'] == 'FAIL').sum()}", "",
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

    st.session_state["results"] = results

# ---------------------------------------------------------------------
# View Results
# ---------------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]
    st.subheader("✅ Results")

    def highlight_status(row):
        color = "#c6f6c6" if row["Predicted_Status"] == "PASS" else "#f6c6c6"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(results.style.apply(highlight_status, axis=1), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        n_pass = (results["Predicted_Status"] == "PASS").sum()
        n_fail = (results["Predicted_Status"] == "FAIL").sum()
        fig1, ax1 = plt.subplots()
        ax1.pie([n_pass, n_fail], labels=["PASS", "FAIL"],
                colors=["#2e7d32", "#c62828"], autopct="%1.0f%%")
        ax1.set_title("Predicted Pass / Fail Ratio")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        bar_colors = ["#2e7d32" if s == "PASS" else "#c62828" for s in results["Predicted_Status"]]
        ax2.bar(results["Part_ID"], results["Predicted_Cost_Index"], color=bar_colors)
        ax2.set_xlabel("Part ID")
        ax2.set_ylabel("Predicted Cost Index")
        ax2.set_title("Predicted Cost per Part")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

    st.subheader("📤 Export")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions.csv", csv_bytes, "predictions.csv", "text/csv")

    report_path = os.path.join(OUT_DIR, "report.txt")
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button("Download report.txt", f, "report.txt", "text/plain")
