import argparse
import pickle
from pathlib import Path

import pandas as pd

from data import FEATURE_COLUMNS, load_data

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "churn_pipeline.pkl"
EXAMPLE_PATH = PROJECT_DIR / "data" / "example_batch.csv"
OUTPUT_PATH = PROJECT_DIR / "reports" / "predictions.csv"


def predict_batch(data_path, model_path, output_path):
    data_path = Path(data_path).resolve()
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path in [data_path, model_path]:
        raise ValueError("Prediction output must not overwrite the input CSV or model.")

    data = load_data(data_path, labeled=False)
    with model_path.open("rb") as file:
        model = pickle.load(file)

    features = data[FEATURE_COLUMNS]
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    results = pd.DataFrame({
        "customerID": data["customerID"].to_numpy(),
        "churn_probability": probabilities,
        "predicted_churn": predictions,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict churn for a customer CSV.")
    parser.add_argument("--data", type=Path, default=EXAMPLE_PATH, help="Feature-only CSV.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Trusted fitted pipeline.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Prediction CSV destination.")
    args = parser.parse_args()
    predict_batch(args.data, args.model, args.output)
