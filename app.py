
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🔄 Customer Churn Prediction", layout="centered")
st.title("📉 Predict Customer Churn")

model_choice = st.selectbox("Choose Model", ["RandomForest", "XGBoost", "LogisticRegression"])
model_path = f"{model_choice.lower()}_churn_model.pkl"

try:
    model = joblib.load(model_path)
    features = joblib.load("churn_features.pkl")
except FileNotFoundError:
    st.error("Please run model_training.py to train models.")
    st.stop()

st.subheader("Upload Customer Data (CSV)")
uploaded_file = st.file_uploader("Upload File", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    input_df = df[features]
    prediction = model.predict(input_df)
    df["ChurnPrediction"] = prediction
    df["ChurnPrediction"] = df["ChurnPrediction"].map({1: "Yes", 0: "No"})
    st.write(df.head())
    st.download_button("Download Predictions", df.to_csv(index=False), "churn_predictions.csv")
