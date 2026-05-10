"""
Project-wide configuration and paths.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlflow"

# ── Data ───────────────────────────────────────────────────────────────────────
RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_FILE = MODELS_DIR / "best_model.pkl"
PIPELINE_FILE = MODELS_DIR / "pipeline.pkl"

TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = str(MLFLOW_DIR)
EXPERIMENT_NAME = "churn-prediction"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
