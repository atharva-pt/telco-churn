import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data import FEATURE_COLUMNS, load_data
from preprocessing import build_pipeline
from plots import save_plots

RANDOM_SEED = 42
TEST_SIZE = 0.2
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "telco.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_pipeline.pkl"
REPORT_DIR = PROJECT_DIR / "reports"


def train_model(data_path, model_path, report_dir):
    source_path = Path(data_path).resolve()
    model_path = Path(model_path).resolve()
    report_dir = Path(report_dir).resolve()
    if model_path == source_path:
        raise ValueError("The model output must not overwrite the input CSV.")
    report_names = [
        "metrics.json", "test_predictions.csv", "training_distributions.png",
        "churn_by_contract.png", "confusion_matrix.png",
    ]
    for name in report_names:
        if (report_dir / name).resolve() in [source_path, model_path]:
            raise ValueError("Reports must not overwrite the input CSV or model.")

    data = load_data(data_path)
    data = data.sort_values("customerID")
    if data["Churn"].nunique() != 2:
        raise ValueError("Training requires both churn and no-churn customers.")

    training_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=data["Churn"],
    )
    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data["Churn"]
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["Churn"]
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both training and test data need both churn outcomes; add more examples.")

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    metrics = {
        "training_rows": len(training_data),
        "test_rows": len(test_data),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    test_results = pd.DataFrame({
        "customerID": test_data["customerID"].to_numpy(),
        "actual_churn": y_test.to_numpy(),
        "churn_probability": probabilities,
        "predicted_churn": predictions,
    })
    test_results.to_csv(report_dir / "test_predictions.csv", index=False)
    save_plots(training_data, y_test, predictions, report_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved reports: {report_dir}")
    return metrics


if __name__ == "__main__":
    train_model(DATA_PATH, MODEL_PATH, REPORT_DIR)
