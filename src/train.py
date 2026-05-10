"""
Model training, evaluation, and MLflow experiment tracking.
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, average_precision_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger

from src.config import (
    RAW_DATA_FILE, MODEL_FILE, PIPELINE_FILE,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
)
from src.preprocess import load_raw_data, clean_data, get_feature_groups, build_preprocessor, split_features_target


def get_models():
    """Return dict of candidate models."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_STATE, verbose=-1
        ),
    }


def evaluate(model, X_test, y_test):
    """Compute evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "avg_precision": average_precision_score(y_test, y_prob),
    }


def train():
    """Main training loop with MLflow tracking."""
    # ── Load & prep ────────────────────────────────────────────────────────────
    df = load_raw_data()
    df = clean_data(df)
    X, y = split_features_target(df)
    numeric_cols, categorical_cols = get_feature_groups(df.drop(columns=[TARGET_COLUMN]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}, Churn rate: {y.mean():.2%}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_model = None
    best_roc_auc = 0.0

    for name, clf in get_models().items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training: {name}")

            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", clf),
            ])

            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)

            # Log to MLflow
            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(f"{name} → ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model = pipeline
                best_name = name

    # ── Save best model ────────────────────────────────────────────────────────
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, PIPELINE_FILE)
    logger.info(f"Best model: {best_name} (ROC-AUC={best_roc_auc:.4f}) saved to {PIPELINE_FILE}")

    return best_model


if __name__ == "__main__":
    train()
