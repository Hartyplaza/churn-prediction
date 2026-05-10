# 🏦 Customer Churn Prediction System

An end-to-end machine learning project to predict customer churn using structured/tabular data.

## 📌 Project Overview

Customer churn prediction helps businesses identify customers likely to leave, enabling proactive retention strategies. This project covers the full ML lifecycle — from raw data to a deployed REST API with a UI.

## 🗂 Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                  # Original, unmodified data
│   └── processed/            # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py              # Project-wide configs & paths
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train.py               # Model training & evaluation
│   ├── predict.py             # Inference logic
│   └── utils.py               # Helper functions
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # Request/response models
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
├── mlflow/                    # MLflow experiment tracking
├── models/                    # Saved model artifacts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/raw/`.

### 3. Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train Model
```bash
python src/train.py
```

### 5. Start API
```bash
uvicorn api.main:app --reload
```

### 6. Launch UI
```bash
streamlit run app/streamlit_app.py
```

### 7. Run with Docker
```bash
docker-compose up --build
```

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | pandas, numpy, scikit-learn |
| Modeling | XGBoost, LightGBM, scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker |
| Testing | pytest |

## 📊 ML Pipeline

1. **EDA** — understand distributions, correlations, class imbalance
2. **Preprocessing** — handle missing values, encode categoricals, scale features
3. **Feature Engineering** — interaction features, binning
4. **Modeling** — baseline → XGBoost/LightGBM with hyperparameter tuning
5. **Evaluation** — ROC-AUC, Precision-Recall, SHAP explainability
6. **Deployment** — FastAPI + Docker + Streamlit

## 📈 Results

| Model | ROC-AUC | F1 Score |
|---|---|---|
| Logistic Regression | TBD | TBD |
| XGBoost | TBD | TBD |
| LightGBM | TBD | TBD |

## 👤 Author

Your Name — [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
