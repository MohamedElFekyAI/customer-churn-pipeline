from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-end Machine Learning API for predicting customer churn risk.",
    version="1.0.0"
)

try:
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception:
    model, scaler = None, None

class CustomerData(BaseModel):
    tenure: int = Field(..., example=12, description="Number of months customer stayed")
    monthly_charges: float = Field(..., example=75.5, description="Monthly billing amount")
    total_charges: float = Field(..., example=906.0, description="Total amount paid")
    contract_type: int = Field(..., example=0, description="0: Month-to-month, 1: One year, 2: Two year")

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model/Scaler not loaded. Run training first.")
    
    input_df = pd.DataFrame([{
        'tenure': customer.tenure,
        'monthly_charges': customer.monthly_charges,
        'total_charges': customer.total_charges,
        'contract_type': customer.contract_type
    }])
    
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    churn_probability = float(model.predict_proba(scaled_input)[0][1])
    
    return {
        "churn_prediction": prediction,
        "is_churn_risk": bool(prediction == 1),
        "churn_probability": round(churn_probability, 4),
        "risk_level": "High" if churn_probability > 0.6 else "Medium" if churn_probability > 0.3 else "Low"
    }
