import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and data with new Streamlit caching
@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_model.pkl')

@st.cache_data
def load_data():
    data = pd.read_csv('breast_cancer_data.csv')
    # Ensure diagnosis is numeric: map 'B'->0, 'M'->1 if needed
    if data['diagnosis'].dtype == object:
        data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    return data

# Load assets
model = load_model()
data = load_data()
# Prepare scaler
X = data.drop('diagnosis', axis=1)
scaler = StandardScaler().fit(X)
feature_names = X.columns

# App configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🎗️ Breast Cancer Tumor Prediction App")
st.markdown("This app predicts whether a tumor is **benign** or **malignant** based on your input measurements.")

# Sidebar for user inputs
st.sidebar.header("Input Tumor Measurements")
def user_input_features():
    inputs = {}
    for feature in feature_names:
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        mean_val = float(data[feature].mean())
        inputs[feature] = st.sidebar.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

# Display prediction
st.subheader("🔍 Prediction Result")
if prediction == 1:
    st.error("The model predicts **Malignant**.")
else:
    st.success("The model predicts **Benign**.")

# Display probabilities
st.subheader("📊 Prediction Probability")
prob_df = pd.DataFrame({
    'Outcome': ['Benign', 'Malignant'],
    'Probability': [pred_proba[0], pred_proba[1]]
})
st.bar_chart(prob_df.set_index('Outcome'))

# Example cases with error handling
st.markdown("---")
st.header("Example Cases")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Benign Example")
    benign_examples = data[data['diagnosis'] == 0]
    if not benign_examples.empty:
        example_benign = benign_examples.iloc[0]
        st.write(example_benign.drop('diagnosis'))
        sb_scaled = scaler.transform([example_benign.drop('diagnosis')])
        sb_pred = model.predict(sb_scaled)[0]
        if sb_pred == 0:
            st.success("Correctly Predicted: Benign")
        else:
            st.error("Incorrectly Predicted: Malignant")
    else:
        st.write("No benign examples available.")

with col2:
    st.subheader("Malignant Example")
    malignant_examples = data[data['diagnosis'] == 1]
    if not malignant_examples.empty:
        example_malig = malignant_examples.iloc[0]
        st.write(example_malig.drop('diagnosis'))
        sm_scaled = scaler.transform([example_malig.drop('diagnosis')])
        sm_pred = model.predict(sm_scaled)[0]
        if sm_pred == 1:
            st.error("Correctly Predicted: Malignant")
        else:
            st.success("Incorrectly Predicted: Benign")
    else:
        st.write("No malignant examples available.")

st.markdown("---")
st.write("Built with ❤️ for clarity and ease of use.")
