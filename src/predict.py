"""
Inference logic — load model and predict on new data.
"""

import joblib
import pandas as pd
from loguru import logger
from src.config import PIPELINE_FILE


def load_model(path=PIPELINE_FILE):
    """Load saved model pipeline."""
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def predict(data: dict | pd.DataFrame, model=None):
    """
    Predict churn for a single record (dict) or batch (DataFrame).
    Returns dict with 'churn_prediction' (0/1) and 'churn_probability'.
    """
    if model is None:
        model = load_model()

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    return {
        "churn_prediction": predictions.tolist(),
        "churn_probability": probabilities.tolist(),
    }


if __name__ == "__main__":
    # Example single prediction
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2,
    }

    result = predict(sample)
    print(f"Churn: {result['churn_prediction'][0]} | Probability: {result['churn_probability'][0]:.2%}")
