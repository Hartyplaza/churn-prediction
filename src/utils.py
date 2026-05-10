"""
Shared utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
import shap


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="steelblue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot SHAP-based feature importance."""
    clf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # Get transformed feature names
    try:
        cat_features = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
            preprocessor.transformers_[1][2]
        ).tolist()
        num_features = preprocessor.transformers_[0][2]
        all_features = list(num_features) + cat_features
    except Exception:
        all_features = feature_names

    explainer = shap.TreeExplainer(clf)
    X_transformed = preprocessor.transform(feature_names)
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_transformed, feature_names=all_features, max_display=top_n)


def class_distribution(y: pd.Series, label="Target"):
    counts = y.value_counts()
    print(f"\n{label} Distribution:")
    print(counts)
    print(f"Churn rate: {y.mean():.2%}\n")
