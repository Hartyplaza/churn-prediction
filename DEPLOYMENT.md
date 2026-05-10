# 🚀 Deployment Guide — Churn Prediction App

You have two components to run:
- **FastAPI** — the backend REST API (serves predictions)
- **Streamlit** — the frontend UI (form to enter customer data)

---

## ✅ Prerequisites

Make sure these are installed:
```bash
pip install fastapi uvicorn streamlit loguru --user
```

---

## 📁 Folder Check

Your project should look like this:
```
churn-prediction/
├── models/
│   └── best_model_pipeline.pkl   ← must exist (from notebook 03)
├── api/
│   ├── main.py
│   └── schemas.py
├── app/
│   └── streamlit_app.py
```

---

## PART 1 — Run the FastAPI Backend

### Step 1 — Open a terminal and go to your project root
```bash
cd path\to\churn-prediction
```

### Step 2 — Start the API server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3 — Verify it's running
Open your browser and go to:
```
http://localhost:8000
```
You should see:
```json
{"message": "Churn Prediction API is running 🚀"}
```

### Step 4 — Explore the auto-generated docs
```
http://localhost:8000/docs
```
This gives you an interactive Swagger UI to test the API directly in the browser — great for your portfolio!

---

## PART 2 — Run the Streamlit Frontend

### Step 1 — Open a SECOND terminal (keep FastAPI running in the first)
```bash
cd path\to\churn-prediction
```

### Step 2 — Launch Streamlit
```bash
streamlit run app/streamlit_app.py
```

### Step 3 — Open the UI
Streamlit will automatically open your browser at:
```
http://localhost:8501
```

---

## 🧪 Test a Prediction

In the Streamlit UI, try this high-risk customer profile:
| Field | Value |
|---|---|
| Contract | Month-to-month |
| Internet Service | Fiber optic |
| Tenure | 2 months |
| Monthly Charges | $85 |
| Online Security | No |
| Tech Support | No |
| Payment Method | Electronic check |

Expected result: **High churn risk** 🔴

---

## 🔍 Test the API Directly (Optional)

You can also test via the Swagger UI at `http://localhost:8000/docs`:
1. Click **POST /predict**
2. Click **Try it out**
3. Paste this sample JSON and click Execute:

```json
{
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
}
```

---

## ⚠️ Common Issues

**"Model not loaded" error**
→ Make sure `models/best_model_pipeline.pkl` exists. Re-run notebook 03 Section 9.

**"Connection refused" in Streamlit**
→ Make sure FastAPI is running in another terminal first.

**Port already in use**
→ Change the port: `uvicorn api.main:app --port 8001`

---

## 🎯 What This Shows on Your Portfolio

- ✅ End-to-end ML pipeline
- ✅ Production-style REST API with FastAPI
- ✅ Interactive UI with Streamlit
- ✅ Auto-generated API documentation (Swagger)
- ✅ Risk classification (Low / Medium / High)
