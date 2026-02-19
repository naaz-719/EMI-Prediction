import os
import streamlit as st
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import zipfile

# Extract mlruns if the folder doesn't exist but the zip does
if not os.path.exists("mlruns") and os.path.exists("mlruns.zip"):
    with zipfile.ZipFile("mlruns.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# -------------------------------
# ⚙️ CONFIG
# -------------------------------
st.set_page_config(page_title="💰 EMI Eligibility & Prediction Dashboard", layout="wide")
st.title("💰 EMI Eligibility & Prediction Dashboard")

ARTIFACTS_DIR = "artifacts"
os.makedirs(f"{ARTIFACTS_DIR}/classification", exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/regression", exist_ok=True)

# --- Step 1: Retrieve best runs from MLflow ---
# --- Step 1: Retrieve best runs from MLflow ---
try:
    clf_runs = mlflow.search_runs(experiment_names=["EMI_Classification_Experiment"])
    reg_runs = mlflow.search_runs(experiment_names=["EMI_Regression_Experiment"])

    # Check if classification runs exist
    if clf_runs.empty:
        st.error("❌ No runs found for 'EMI_Classification_Experiment'.")
        st.info("Check if your 'mlruns' folder is uploaded to GitHub or if the experiment name is correct.")
        st.stop()
    
    # Check if regression runs exist
    if reg_runs.empty:
        st.error("❌ No runs found for 'EMI_Regression_Experiment'.")
        st.stop()

    # If both exist, safely sort and select
    best_clf_run = clf_runs.sort_values(by="metrics.accuracy", ascending=False).iloc[0]
    best_reg_run = reg_runs.sort_values(by="metrics.RMSE", ascending=True).iloc[0]

    # Display success in console/logs
    print(f"✅ Best Classification Run: {best_clf_run['run_id']}")
    print(f"✅ Best Regression Run: {best_reg_run['run_id']}")

except Exception as e:
    st.error(f"⚠️ MLflow Error: {e}")
    st.stop()

# --- Step 2: Load models from MLflow artifacts ---
# Adjust artifact folder names (check your mlruns folder if needed)
clf_model = mlflow.sklearn.load_model(f"runs:/{best_clf_run['run_id']}/xgb_model")
reg_model = mlflow.sklearn.load_model(f"runs:/{best_reg_run['run_id']}/models")

# --- Step 3: Save for Streamlit ---
joblib.dump(clf_model, "/content/artifacts/classification/XGBClassifier_pipeline.joblib")
joblib.dump(reg_model, "/content/artifacts/regression/RandomForestRegressor_pipeline.joblib")

print("\n✅ Models exported successfully and ready for Streamlit!")

# -------------------------------
# 🧠 LOAD MODELS FROM MLFLOW OR LOCAL FILES
# -------------------------------
def load_or_download_model(experiment_name, best_run, save_path, artifact_subdir):
    """Load from MLflow if not already saved locally"""
    try:
        if not os.path.exists(save_path):
            st.info(f"🔄 Exporting {experiment_name} model from MLflow...")

            model_uri = f"runs:/{best_run['run_id']}/{artifact_subdir}"
            model = mlflow.sklearn.load_model(model_uri)

            joblib.dump(model, save_path)
            st.success(f"✅ {experiment_name} model saved to {save_path}")
        else:
            st.success(f"✅ Loaded existing {experiment_name} model from {save_path}")
            model = joblib.load(save_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load {experiment_name} model: {e}")
        return None


# -------------------------------
# 🧾 INPUT SECTION
# -------------------------------
st.sidebar.header("⚙️ Input Parameters")

classification_input = {
    "Income": st.sidebar.number_input("Monthly Income", 1000, 200000, 45000),
    "Loan Amount": st.sidebar.number_input("Loan Amount", 1000, 1000000, 200000),
    "Credit Score": st.sidebar.slider("Credit Score", 300, 900, 700),
    "Loan Term (months)": st.sidebar.slider("Loan Term (months)", 6, 120, 24)
}

regression_input = {
    "Principal": st.sidebar.number_input("Principal Amount", 1000, 1000000, 300000),
    "Rate of Interest (%)": st.sidebar.slider("Interest Rate", 1.0, 20.0, 7.5),
    "Tenure (months)": st.sidebar.slider("Tenure (months)", 6, 120, 24),
    "Credit Score": st.sidebar.slider("Credit Score", 300, 900, 720)
}

# -------------------------------
# ⚡ LOAD BEST MODELS (AUTO)
# -------------------------------
st.subheader("🧠 Loading Models Automatically from MLflow")

try:
    # 🔍 Replace these with your actual best run IDs
    best_clf_run = {"run_id": "a2311f10071e412cbbd09b1ccd0360d4"}
    best_reg_run = {"run_id": "2d58f75184764a07835ebb1ac3a304d4"}

    clf_model = load_or_download_model(
        "Classification (XGBClassifier)",
        best_clf_run,
        f"{ARTIFACTS_DIR}/classification/XGBClassifier_pipeline.joblib",
        "xgb_model"  # adjust this folder name if different
    )

    reg_model = load_or_download_model(
        "Regression (RandomForestRegressor)",
        best_reg_run,
        f"{ARTIFACTS_DIR}/regression/RandomForestRegressor_pipeline.joblib",
        "models"  # adjust this folder name if different
    )
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

# -------------------------------
# 🎯 CLASSIFICATION
# -------------------------------
st.header("🎯 EMI Eligibility Prediction (Classification)")

if clf_model:
    if st.button("🔍 Predict EMI Eligibility"):
        input_df = pd.DataFrame([classification_input])
        pred = clf_model.predict(input_df)[0]
        st.success(f"🏦 Predicted EMI Eligibility: **{pred}**")

# -------------------------------
# 📈 REGRESSION
# -------------------------------
st.header("📈 EMI Amount Prediction (Regression)")

if reg_model:
    if st.button("💰 Predict EMI Amount"):
        input_df = pd.DataFrame([regression_input])
        pred = reg_model.predict(input_df)[0]
        st.success(f"💵 Predicted EMI Amount: **₹{pred:,.2f}**")

# -------------------------------
# 📊 MLFLOW DASHBOARD LINK
# -------------------------------
st.markdown("---")
st.subheader("📊 MLflow Experiment Tracking Dashboard")
st.info("To view experiment metrics, open the MLflow UI below:")
st.code("!mlflow ui --port 5000")
st.markdown("Then open [http://localhost:5000](http://localhost:5000) to explore models and metrics.")


