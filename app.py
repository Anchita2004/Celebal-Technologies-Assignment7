import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.set_page_config(page_title="Wine Quality Predictor")
st.title("🍷 Wine Quality Prediction App")
st.markdown("Enter the chemical properties of the wine to predict its quality (0–10).")

# Feature input
def user_input():
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
    volatile_acidity = st.slider("Volatile Acidity", 0.10, 1.50, 0.5)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.25)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 300, 50)
    density = st.slider("Density", 0.990, 1.005, 0.996)
    pH = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates", 0.2, 2.0, 0.6)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)
    
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Prediction
if st.button("Predict Quality"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)

    st.success(f"✅ Predicted Wine Quality: **{prediction}**")
    st.write("📊 Prediction Probabilities:")
    st.bar_chart(proba)

# Show feature importances
st.subheader("🔍 Model Feature Importances")
importances = model.feature_importances_
cols = input_df.columns
imp_df = pd.DataFrame({'Feature': cols, 'Importance': importances}).sort_values(by="Importance", ascending=False)
st.bar_chart(imp_df.set_index("Feature"))
