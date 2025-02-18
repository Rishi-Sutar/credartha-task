from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load("saved_models/best_logistic_regression_model.pkl")
scaler = joblib.load("../artifacts/standard_scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

class PredictionRequest(BaseModel):
    Age: float
    Credit_Score: float
    Existing_Loans: float
    Utilization: float
    Missed_Payments_12M: float
    Total_Outstanding_Debt: float
    Debt_to_Income_Ratio: float
    Monthly_Debt: float
    Dept_per_loan: float
    Loan_per_age: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the risk classification service!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Rename columns to match the feature names used during training
        data.columns = [
            "Age", "Credit Score", "Existing Loans", "Utilization", 
            "Missed Payments (12M)", "Total Outstanding Debt", 
            "Debt-to-Income Ratio", "Monthly Debt", "Dept per loan", "Loan per age"
        ]
        
        # Apply the same scaling as during training
        scaled_data = scaler.transform(data)
        
        # Perform prediction
        prediction = model.predict(scaled_data)
        risk_mapping = {0: "Low Risk", 1: "High Risk"}
        ans = risk_mapping[prediction[0]]
        
        return {"prediction": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#uvicorn app:app --reload