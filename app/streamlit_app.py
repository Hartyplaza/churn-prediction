"""
Streamlit UI for Customer Churn Prediction.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.markdown("Enter customer details to predict churn probability.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    submitted = st.form_submit_button("🔍 Predict Churn")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        prob = result["churn_probability"]
        risk = result["risk_level"]
        pred = result["churn_prediction"]

        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk]

        st.markdown("---")
        st.subheader("Prediction Result")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Churn Prediction", "Yes ⚠️" if pred == 1 else "No ✅")
        col_b.metric("Churn Probability", f"{prob:.1%}")
        col_c.metric("Risk Level", f"{color} {risk}")
        st.progress(prob)

        if pred == 1:
            st.error("⚠️ This customer is likely to churn. Consider a retention offer.")
        else:
            st.success("✅ This customer is likely to stay.")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
