from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import pandas as pd
import joblib
import os

from api.schemas import CustomerFeatures, PredictionResponse

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None

@app.on_event("startup")
def load_ml_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_pipeline.pkl')
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning(f"Model not found: {e}")

def engineer_features(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                                 labels=['new','developing','established','loyal'])
    service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    df['num_addons'] = df[service_cols].apply(lambda row: (row=='Yes').sum(), axis=1)
    df['risky_payment'] = ((df['PaperlessBilling']=='Yes') & (df['PaymentMethod']=='Electronic check')).astype(int)
    df['no_support'] = ((df['OnlineSecurity']=='No') & (df['TechSupport']=='No')).astype(int)
    df['is_monthly'] = (df['Contract']=='Month-to-month').astype(int)
    df.drop(columns=['TotalCharges'], inplace=True)
    return df

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        input_df = engineer_features(input_df)
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob >= 0.5)
        risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
        return PredictionResponse(churn_prediction=prediction,
                                  churn_probability=round(float(prob), 4),
                                  risk_level=risk)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))