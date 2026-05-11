# ChurnGuard AI — Customer Churn Prediction System

An end-to-end machine learning system for predicting and preventing customer churn in the telecom industry. Built with a full production stack including a REST API, interactive dashboard, and SHAP explainability.

---

## Project Overview

Customer churn is one of the most costly problems in the telecom industry. Acquiring a new customer costs **5–7x more** than retaining an existing one. This project builds a production-ready ML pipeline that:

- Identifies at-risk customers before they leave
- Explains *why* a customer is at risk using SHAP values
- Serves real-time predictions via a FastAPI REST endpoint
- Provides an interactive multi-page Streamlit dashboard

---

## Live Demo

| Component | URL |
|---|---|
| **Streamlit App (Live)** | https://churn-prediction-6ogsccahkfmnttdtwrhcqb.streamlit.app/ |
| FastAPI Swagger Docs | `http://localhost:8000/docs` (run locally) |
| API Health Check | `http://localhost:8000/health` (run locally) |

---

## Results

### Model Leaderboard

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **XGBoost (Tuned)** | **0.8429** | **0.6129** | **0.5640** | **0.6711** |
| Logistic Regression | 0.8426 | 0.6227 | 0.5168 | 0.7834 |
| XGBoost | 0.8381 | 0.5923 | 0.5884 | 0.5963 |
| LightGBM | 0.8370 | 0.5884 | 0.5807 | 0.5963 |

**Best model: XGBoost (Tuned)** — selected based on ROC-AUC and balanced F1/Recall trade-off.

### Key Findings from EDA

| Segment | Churn Rate |
|---|---|
| Month-to-month contract | 42.7% |
| Fiber optic + no security | 51.2% |
| Tenure < 12 months | 47.7% |
| Electronic check payment | 45.3% |
| Two-year contract | 2.8% |
| Tenure > 48 months | 6.6% |

---

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 raw features
- **Target:** Churn (Yes/No) — 26.5% positive rate (class imbalance handled with SMOTE)

---

## Engineered Features

Six new features were created from domain knowledge and EDA insights:

| Feature | Description | Importance |
|---|---|---|
| `charges_per_tenure` | TotalCharges / (tenure + 1) — monthly spend efficiency | High |
| `tenure_group` | Bins tenure into new / developing / established / loyal | High |
| `is_monthly` | Flag for month-to-month contracts | Very High |
| `no_support` | Flag for no security AND no tech support | High |
| `risky_payment` | Flag for paperless billing + electronic check | High |
| `num_addons` | Count of add-on services subscribed | Medium |

---

## Project Structure

```
churn-prediction/
├── .streamlit/
│   └── config.toml            # Dark theme config
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Train/test splits
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── config.py              # Project paths & settings
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train.py               # Model training & MLflow tracking
│   ├── predict.py             # Inference logic
│   └── utils.py               # Helper functions
├── api/
│   ├── main.py                # FastAPI application
│   └── schemas.py             # Pydantic request/response models
├── app/
│   └── streamlit_app.py       # Multi-page Streamlit UI
├── models/                    # Saved model artifacts
├── mlflow/                    # MLflow experiment tracking
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ML Pipeline

```
Raw Data → EDA → Feature Engineering → Preprocessing Pipeline
       → SMOTE (balance classes) → Model Training → Evaluation
       → SHAP Explainability → MLflow Tracking → FastAPI → Streamlit
```

### Preprocessing
- Numeric features: Median imputation → StandardScaler
- Categorical features: Mode imputation → OneHotEncoder
- Class imbalance: SMOTE applied on training set only

### Models Trained
- Logistic Regression (baseline)
- XGBoost (default + RandomizedSearchCV tuning)
- LightGBM

### Hyperparameter Tuning
- Method: RandomizedSearchCV
- Iterations: 20
- Cross-validation: 5-fold Stratified KFold
- Scoring metric: ROC-AUC

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | pandas, NumPy, scikit-learn |
| Modeling | XGBoost, LightGBM, scikit-learn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| UI | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Testing | pytest |
| Version Control | Git, GitHub |

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Hartyplaza/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run Notebooks in Order

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

### 4. Start FastAPI (Terminal 1)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start Streamlit (Terminal 2)

```bash
streamlit run app/streamlit_app.py
```

### 6. Run with Docker

```bash
docker-compose up --build
```

---

## API Usage

### Predict Churn

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
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
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0
  }'
```

### Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8742,
  "risk_level": "High"
}
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| Project Overview | Problem statement, pipeline, tech stack |
| Dashboard & Plots | EDA charts — churn by contract, tenure, payment, services |
| Prediction | Real-time churn risk assessment with gauge chart |
| Engineered Features | Feature explanations with formulas and impact charts |
| Model Metrics | ROC-AUC comparison, SHAP importance, confusion matrix |

---

## Author

**Ofigwe Hart**:

Data Scientist / ML Engineer

- LinkedIn: [linkedin.com/in/hart-ofigwe](https://www.linkedin.com/in/hart-ofigwe)
- GitHub: [github.com/Hartyplaza](https://github.com/Hartyplaza)

---

## License

This project is open source and available under the [MIT License](LICENSE).
