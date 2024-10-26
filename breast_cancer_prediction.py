import pandas as pd         # For data handling
import numpy as np          # For numerical operations
import seaborn as sns       # For visualizations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from ucimlrepo import fetch_ucirepo 

def fetch_data():
    """Fetch the breast cancer dataset from UCI ML Repository."""
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    return X, y

def process_data(X, y):
    """Process the data: combine features and target, handle missing values, encode and scale."""
    # Convert feature and target data to a single DataFrame
    data = X.copy()  # Copy feature data
    data['diagnosis'] = y  # Add target data as a new column

    # Handle missing values
    data = data.dropna()

    # Encode target variable
    label_encoder = LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

    # Separate features and target
    X = data.drop(['diagnosis'], axis=1)  # Features
    y = data['diagnosis']                 # Target variable

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # Returns train/test splits

def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest models and return the best model."""
    # Initialize and train Logistic Regression model
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)

    # Initialize and train Random Forest model
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and evaluate
    best_model_rf = grid_search.best_estimator_

    return best_model_rf

def plot_feature_importance(model, feature_names):
    """Plot the feature importance for the provided model."""
    importances = model.feature_importances_
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

if __name__ == "__main__":
    # Fetch data
    X, y = fetch_data()

    # Process data
    X_train, X_test, y_train, y_test = process_data(X, y)

    # Train models
    best_model_rf = train_models(X_train, y_train)

    # Evaluate accuracy
    y_pred_rf = best_model_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf}")

    # Confusion matrix for Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Classification report for Random Forest
    print("Classification Report for Random Forest:")
    print(classification_report(y_test, y_pred_rf))

    # Plot feature importance
    plot_feature_importance(best_model_rf, X.columns)

    # Save the trained model
    joblib.dump(best_model_rf, "breast_cancer_model.pkl")
