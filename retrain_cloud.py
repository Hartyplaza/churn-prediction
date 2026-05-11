"""
Retrain and save model - run this once on Streamlit Cloud startup.
This ensures the model is always compatible with the installed sklearn version.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def engineer_features(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                                labels=['new','developing','established','loyal'])
    service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    df['num_addons'] = df[service_cols].apply(lambda row: (row=='Yes').sum(), axis=1)
    df['risky_payment'] = ((df['PaperlessBilling']=='Yes') &
                           (df['PaymentMethod']=='Electronic check')).astype(int)
    df['no_support'] = ((df['OnlineSecurity']=='No') &
                        (df['TechSupport']=='No')).astype(int)
    df['is_monthly'] = (df['Contract']=='Month-to-month').astype(int)
    df.drop(columns=['TotalCharges'], inplace=True)
    return df

def train_and_save():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw',
                             'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    if not os.path.exists(data_path):
        print("Dataset not found — skipping retraining")
        return False
    
    df = pd.read_csv(data_path)
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df = engineer_features(df)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('scl', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         categorical_cols),
    ])
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model_pipeline.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model retrained and saved to {model_path}")
    return True

if __name__ == '__main__':
    train_and_save()
