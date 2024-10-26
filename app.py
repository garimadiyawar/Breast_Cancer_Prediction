import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset to retrieve normal (mean) values
data = pd.read_csv("breast_cancer_data.csv")
normal_values = data.mean()

# Page title and description
st.title("Breast Cancer Prediction App")
st.write("Input your test data below to check for benign or malignant diagnosis. Each feature's average value is displayed for reference.")

# Define function for prediction
def predict_breast_cancer(features):
    # Scale the input features using the same scaler used in training
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Malignant" if prediction[0] == 1 else "Benign"

# Collecting user inputs and displaying mean values for comparison
user_inputs = []
for feature in data.columns[:-1]:  # exclude 'diagnosis' column
    avg_value = normal_values[feature]
    user_value = st.number_input(f"{feature} (Normal: {avg_value:.2f})", min_value=-1.0, max_value=100.0, step=0.1)
    user_inputs.append(user_value)

# Convert the list of user inputs to a numpy array for prediction
user_inputs = np.array(user_inputs)

# Display prediction when button is clicked
if st.button("Predict"):
    result = predict_breast_cancer(user_inputs)
    st.write("Prediction:", result)
