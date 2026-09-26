"""Synthetic data: tests never depend on a private Downloads path."""
import pandas as pd


def make_frame(n=100):
    rows = []
    for i in range(n):
        rows.append({
            "customerID": f"{i:05d}", "gender": "Female" if i % 2 else "Male",
            "SeniorCitizen": str(i % 2), "Partner": "No", "Dependents": "No",
            "tenure": str(i % 40), "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "Month-to-month" if i % 3 else "One year",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": str(20 + i % 30), "TotalCharges": str(100 + i * 10),
            "Churn": "Yes" if i % 2 else "No",
        })
    return pd.DataFrame(rows)
