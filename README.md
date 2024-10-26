# Breast_Cancer_Prediction App
This is a Breast Cancer Prediction Web App built using Streamlit and Machine Learning. The app allows users to input their test data for various features of breast cancer, compare them to normal values, and predict whether the diagnosis would be "Benign" or "Malignant".

Table of Contents
Overview
Features
Setup Instructions
Usage
Project Structure
Dependencies
Overview
The app uses a trained Random Forest Classifier model to classify breast cancer diagnoses based on multiple cell nucleus characteristics. It has been trained on the Breast Cancer Wisconsin Diagnostic Dataset from the UCI repository and is designed to provide real-time predictions from user-provided input values.

Features
Input fields for all relevant cell nucleus features
Normal value comparison displayed beside each feature
Real-time prediction results (Benign or Malignant)
Intuitive user interface built with Streamlit
Setup Instructions
Clone the Repository

bash
Copy code
git clone https://github.com/your-username/breast-cancer-prediction-app.git
cd breast-cancer-prediction-app
Install the Dependencies

bash
Copy code
pip install -r requirements.txt
Download and Save the Dataset

Download the Breast Cancer Wisconsin Diagnostic Dataset from UCI's Machine Learning Repository or use the ucimlrepo library to directly fetch it.
Convert it to a CSV file (breast_cancer_data.csv).
Run the Training Script (if not already trained)

Open the Jupyter notebook (or .py file) for training the model and save the model and scaler as breast_cancer_model.pkl and scaler.pkl.
Ensure both files are in the root folder.
Run the App

bash
Copy code
streamlit run app.py
Usage
Launch the app following the instructions above.
Input values for each feature in the sidebar. The normal (mean) value of each feature is displayed beside each field for easy comparison.
Press the Predict button.
The app will display the diagnosis as either "Benign" or "Malignant."
Project Structure
graphql
Copy code
breast-cancer-prediction-app/
│
├── app.py                      # Streamlit app code for user interface and prediction
├── breast_cancer_model.pkl     # Saved trained model (Random Forest)
├── scaler.pkl                  # Saved StandardScaler for feature scaling
├── breast_cancer_data.csv      # CSV file containing dataset with feature values
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
└── training_script.py          # Model training and preprocessing code
Dependencies
Python 3.7+
Streamlit
Pandas
NumPy
Scikit-Learn
Seaborn (for visualization in training scripts)
Matplotlib (for visualization in training scripts)
Joblib (for model saving/loading)
Install all dependencies with:

bash
Copy code
pip install -r requirements.txt
License
This project is open-source and available under the MIT License.

