# Heart Disease Prediction App

This is a Flask-based web application that predicts the likelihood of heart disease using Machine Learning (Logistic Regression and Random Forest).

## Dataset

The application uses the `heart.csv` dataset. The training script is configured to look for the dataset at:
`c:\Users\semi\Desktop\AI HC\heart.csv`

## Prerequisites

- Python 3.x
- pip

## Installation

1.  Navigate to the project directory:
    ```bash
    cd "c:\Users\semi\Desktop\AI HC\heart_disease_app"
    ```

2.  Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn flask
    ```

## Usage

### 1. Training the Model

Before running the application, you need to train the models using the dataset. This step processes the `heart.csv` file and saves the trained models.

```bash
c
```

Output files will be saved in the `model/` directory.

### 2. Running the Application

Start the web server:

```bash
python app.py
```

Open your web browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Features

-   **Interactive Form**: Enter patient health metrics.
-   **Dual Model Prediction**: Uses both Logistic Regression and Random Forest classifiers.
-   **Visual Results**: clear "Heart Disease Detected" or "No Heart Disease" indicators with confidence scores.
