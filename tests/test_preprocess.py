"""
Unit tests for preprocessing module.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocess import clean_data, get_feature_groups, split_features_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "tenure": [12, 5, 60],
        "MonthlyCharges": [70.5, 45.0, 90.0],
        "TotalCharges": ["840.0", " ", "5400.0"],
        "Churn": ["Yes", "No", "No"],
    })


def test_clean_data_drops_customer_id(sample_df):
    cleaned = clean_data(sample_df)
    assert "customerID" not in cleaned.columns


def test_clean_data_encodes_target(sample_df):
    cleaned = clean_data(sample_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})


def test_clean_data_fixes_total_charges(sample_df):
    cleaned = clean_data(sample_df)
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_split_features_target(sample_df):
    cleaned = clean_data(sample_df)
    X, y = split_features_target(cleaned)
    assert "Churn" not in X.columns
    assert y.name == "Churn"
    assert len(X) == len(y)
