"""
Data preprocessing and feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from loguru import logger

from src.config import RAW_DATA_FILE, PROCESSED_TRAIN_FILE, PROCESSED_TEST_FILE, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def load_raw_data(filepath=RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle data quality issues."""
    logger.info("Cleaning data...")
    df = df.copy()

    # Drop customerID — not a feature
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has spaces instead of NaN — fix it
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target: Yes → 1, No → 0
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)

    logger.info(f"Missing values after cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


def get_feature_groups(df: pd.DataFrame):
    """Identify numeric and categorical columns (excluding target)."""
    exclude = [TARGET_COLUMN]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor


def split_features_target(df: pd.DataFrame):
    """Split DataFrame into features X and target y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    df = clean_data(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save processed splits
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    PROCESSED_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_FILE, index=False)
    test_df.to_csv(PROCESSED_TEST_FILE, index=False)
    logger.info(f"Saved train ({train_df.shape}) and test ({test_df.shape}) splits.")
