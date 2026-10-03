import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_data(file_path, labeled=True):
    data = pd.read_csv(file_path, dtype={"customerID": str}, keep_default_na=False)

    expected_columns = ["customerID"] + FEATURE_COLUMNS
    if labeled:
        expected_columns = expected_columns + ["Churn"]

    missing_columns = set(expected_columns) - set(data.columns)
    extra_columns = set(data.columns) - set(expected_columns)
    if missing_columns or extra_columns:
        raise ValueError(f"Missing columns: {missing_columns}; extra columns: {extra_columns}")

    if data.empty:
        raise ValueError("The CSV must contain at least one customer.")

    data["customerID"] = data["customerID"].str.strip()
    if data["customerID"].eq("").any() or data["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be nonempty and unique.")

    for column in NUMERIC_COLUMNS:
        values = data[column].replace(r"^\s*$", np.nan, regex=True)
        data[column] = pd.to_numeric(values, errors="raise").astype(float)
        if (data[column] < 0).any() or np.isinf(data[column]).any():
            raise ValueError(f"{column} must contain nonnegative, finite numbers or blanks.")

    if (data["tenure"].dropna() % 1 != 0).any():
        raise ValueError("tenure must contain whole months.")

    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype(str).str.strip()
        if data[column].eq("").any():
            raise ValueError(f"{column} must not contain blank categories.")

    if labeled:
        data["Churn"] = data["Churn"].astype(str).str.strip()
        if not data["Churn"].isin(["Yes", "No"]).all():
            raise ValueError("Churn must contain only Yes or No.")
        data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    return data
