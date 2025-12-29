from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models and scaler
# We use a try-except block to handle cases where the models haven't been trained yet
lr_model_path = 'model/logistic_regression_model.pkl'
rf_model_path = 'model/random_forest_model.pkl'
scaler_path = 'model/scaler.pkl'

lr_model = None
rf_model = None
scaler = None

if os.path.exists(lr_model_path) and os.path.exists(rf_model_path) and os.path.exists(scaler_path):
    lr_model = joblib.load(lr_model_path)
    rf_model = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    print("Both models and scaler loaded successfully.")
    print("  - Logistic Regression model loaded")
    print("  - Random Forest model loaded")
else:
    print("Models or scaler not found. Please run 'train_model.py' first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if lr_model is None or rf_model is None or scaler is None:
        return render_template('index.html', prediction_text="Error: Models not loaded. Run train_model.py first.")

    try:
        # Get values from form
        # Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        
        feature_values = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        
        # Convert to numpy array and reshape
        features = np.array(feature_values).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Predict with BOTH models
        # Logistic Regression
        lr_prediction = lr_model.predict(features_scaled)[0]
        lr_proba = lr_model.predict_proba(features_scaled)[0]
        lr_prob_no_disease = round(lr_proba[0] * 100, 2)
        lr_prob_disease = round(lr_proba[1] * 100, 2)
        lr_confidence = round(max(lr_proba) * 100, 2)
        
        # Random Forest
        rf_prediction = rf_model.predict(features_scaled)[0]
        rf_proba = rf_model.predict_proba(features_scaled)[0]
        rf_prob_no_disease = round(rf_proba[0] * 100, 2)
        rf_prob_disease = round(rf_proba[1] * 100, 2)
        rf_confidence = round(max(rf_proba) * 100, 2)
        
        # Format results
        lr_result = "Heart Disease Detected" if lr_prediction == 1 else "No Heart Disease"
        rf_result = "Heart Disease Detected" if rf_prediction == 1 else "No Heart Disease"
        
        lr_class = "danger" if lr_prediction == 1 else "success"
        rf_class = "danger" if rf_prediction == 1 else "success"
        
        return render_template('index.html',
                            # Logistic Regression results
                            lr_result=lr_result,
                            lr_class=lr_class,
                            lr_prob_no_disease=lr_prob_no_disease,
                            lr_prob_disease=lr_prob_disease,
                            lr_confidence=lr_confidence,
                            # Random Forest results
                            rf_result=rf_result,
                            rf_class=rf_class,
                            rf_prob_no_disease=rf_prob_no_disease,
                            rf_prob_disease=rf_prob_disease,
                            rf_confidence=rf_confidence,
                            # Flag to show results
                            show_results=True)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text=f'Error: {str(e)}', prediction_class="error")

if __name__ == "__main__":
    app.run(debug=True, port=5000)

