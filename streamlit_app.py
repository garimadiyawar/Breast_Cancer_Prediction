import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Cache model and data loading for performance
@st.cache(allow_output_mutation=True)
def load_assets():
    # Load the trained model
    model = joblib.load('breast_cancer_model.pkl')
    # Load raw dataset for scaler fitting and examples
    data = pd.read_csv('breast_cancer_data.csv')
    # Prepare scaler on full dataset
    X = data.drop('diagnosis', axis=1)
    scaler = StandardScaler().fit(X)
    return model, scaler, X.columns, data

model, scaler, feature_names, data = load_assets()

# App header
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🎗️ Breast Cancer Tumor Prediction App")
st.markdown("This app predicts whether a tumor is **benign** or **malignant** based on your input measurements.")

# Sidebar for user inputs
st.sidebar.header("Input Tumor Measurements")
def user_input_features():
    inputs = {}
    for feature in feature_names:
        mean_val = float(data[feature].mean())
        inputs[feature] = st.sidebar.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=float(data[feature].min()),
            max_value=float(data[feature].max()),
            value=mean_val
        )
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

# Display results
st.subheader("🔍 Prediction Result")
if prediction == 1:
    st.error("The model predicts **Malignant**.")
else:
    st.success("The model predicts **Benign**.")

st.subheader("📊 Prediction Probability")
prob_df = pd.DataFrame({
    'Outcome': ['Benign', 'Malignant'],
    'Probability': [pred_proba[0], pred_proba[1]]
})
st.bar_chart(prob_df.set_index('Outcome'))

# Example cases
st.markdown("---")
st.header("Example Cases")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Benign Example")
    example_benign = data[data['diagnosis'] == 0].iloc[0]
    st.write(example_benign.drop('diagnosis'))
    sb_scaled = scaler.transform([example_benign.drop('diagnosis')])
    sb_pred = model.predict(sb_scaled)[0]
    if sb_pred == 0:
        st.success("Correctly Predicted: Benign")
    else:
        st.error("Incorrectly Predicted: Malignant")

with col2:
    st.subheader("Malignant Example")
    example_malig = data[data['diagnosis'] == 1].iloc[0]
    st.write(example_malig.drop('diagnosis'))
    sm_scaled = scaler.transform([example_malig.drop('diagnosis')])
    sm_pred = model.predict(sm_scaled)[0]
    if sm_pred == 1:
        st.error("Correctly Predicted: Malignant")
    else:
        st.success("Incorrectly Predicted: Benign")

st.markdown("---")
st.write("Built with ❤️ for clarity and ease of use.")
