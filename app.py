import pandas as pd         # For data handling
import numpy as np          # For numerical operations
import seaborn as sns       # For visualizations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import streamlit as st
from ucimlrepo import fetch_ucirepo

# Function to load the dataset and preprocess it
@st.cache
def load_data():
    from ucimlrepo import fetch_ucirepo
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # Convert feature and target data to a single DataFrame
    data = X.copy()
    data['diagnosis'] = y

    # Handle missing values
    data = data.dropna()

    # Encode target variable
    label_encoder = LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

    # Separate features and target
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, data

# Function to train the model and return the best model
@st.cache
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model_rf = grid_search.best_estimator_
    
    return best_model_rf, X_test, y_test

# Function to display the confusion matrix
def show_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

# Load data
X, y, data = load_data()

# Calculate average values and range for each feature, excluding the 'diagnosis' column
feature_means = data.drop(columns=['diagnosis']).mean()
feature_min = data.drop(columns=['diagnosis']).min()
feature_max = data.drop(columns=['diagnosis']).max()

# Train model
best_model_rf, X_test, y_test = train_model(X, y)

# Display metrics
y_pred_best_rf = best_model_rf.predict(X_test)
st.write("Best Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_best_rf))

# User inputs
st.title("Breast Cancer Prediction App")
st.write("Enter the features for prediction:")

user_inputs = {}
for feature in feature_means.index:  # Loop through only the feature columns
    average_value = feature_means[feature]
    min_value = feature_min[feature]
    max_value = feature_max[feature]
    
    user_inputs[feature] = st.number_input(
        f"{feature} (Average: {average_value:.2f})",
        min_value=min_value,
        max_value=max_value,
        value=average_value
    )

# Convert user input to DataFrame for prediction
user_input_df = pd.DataFrame([user_inputs])

if st.button("Predict"):
    # Scale the user input
    user_input_scaled = StandardScaler().fit(X).transform(user_input_df)

    # Make prediction
    prediction = best_model_rf.predict(user_input_scaled)
    prediction_label = "Malignant" if prediction[0] == 1 else "Benign"
    
    st.write(f"The predicted diagnosis is: **{prediction_label}**")

    # Show confusion matrix
    show_confusion_matrix(y_test, y_pred_best_rf)
