import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
# Target is 'target' (renamed from 'num')

# Load dataset
data_path = r'c:\Users\semi\Desktop\AI HC\heart.csv'

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# Data Mapping - Convert string descriptions to numeric values expected by the app
print("Mapping string values to numeric...")

# Mappings based on index.html and standard UCI encoding
sex_map = {'Male': 1, 'Female': 0}
cp_map = {'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4}
fbs_map = {True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}
restecg_map = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
exang_map = {True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}
slope_map = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
thal_map = {'normal': 3, 'fixed defect': 6, 'reversable defect': 7}

# Apply mappings
df['sex'] = df['sex'].map(sex_map)
df['cp'] = df['cp'].map(cp_map)
df['fbs'] = df['fbs'].map(fbs_map)
df['restecg'] = df['restecg'].map(restecg_map)
df['exang'] = df['exang'].map(exang_map)
df['slope'] = df['slope'].map(slope_map)
df['thal'] = df['thal'].map(thal_map)

# Rename 'thalch' to 'thalach' and 'num' to 'target'
df = df.rename(columns={'thalch': 'thalach', 'num': 'target'})

# Data Cleaning
print("Cleaning data...")
# Fill missing values with median
# ca and thal often have missing values in some datasets
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())

# For other columns, check for NaNs and fill with median
for col in df.columns:
    if df[col].isnull().any():
        print(f"Imputing missing values for {col}")
        df[col] = df[col].fillna(df[col].median())

# Convert target to binary: 0 = No Disease, 1-4 = Disease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print(f"Data shape: {df.shape}")
print(df.head())

# Feature Selection - ensure exact order as in columns list
X = df[columns]
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print Dataset Split Statistics
total_samples = len(df)
train_samples = len(X_train)
test_samples = len(X_test)
train_pct = (train_samples / total_samples) * 100
test_pct = (test_samples / total_samples) * 100
print("\n--- Dataset Split Statistics ---")
print(f"Total samples: {total_samples}")
print(f"Training set:  {train_samples} samples ({train_pct:.1f}%)")
print(f"Testing set:   {test_samples} samples ({test_pct:.1f}%)")
# Scaling (Important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(classification_report(y_test, lr_pred))

# Model 2: Random Forest
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Tree-based models don't strictly require scaling, but consistent data is good. let's use unscaled to be safe with user inputs later if we forget scaling. 
# actually, let's just use the scaled version for consistency in pipeline or keep track. 
# To make it easier for the web app, let's use the SCALED data for training, so we just need to scale input in the app.
rf_model.fit(X_train_scaled, y_train) 
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

# Comparison
print("\n--- Model Comparison ---")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
if rf_acc > lr_acc:
    print("Random Forest performed better.")
else:
    print("Logistic Regression performed better.")

# Save BOTH Models and Scaler
os.makedirs('model', exist_ok=True)
joblib.dump(lr_model, 'model/logistic_regression_model.pkl')
joblib.dump(rf_model, 'model/random_forest_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("\nBoth models saved:")
print("  - Logistic Regression: model/logistic_regression_model.pkl")
print("  - Random Forest: model/random_forest_model.pkl")
print("  - Scaler: model/scaler.pkl")
