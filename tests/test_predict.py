"""
Unit tests for the prediction API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_no_model():
    """Should return 503 if model is not loaded."""
    with patch("api.main.model", None):
        payload = {
            "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35, "TotalCharges": 844.2,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
