import importlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]


def example_data():
    base = {
        "gender": "Female", "SeniorCitizen": "0", "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    }
    rows = []
    for number in range(40):
        row = base.copy()
        row["customerID"] = f"{number:05d}"
        row["tenure"] = number + 1
        row["MonthlyCharges"] = 20 + number
        row["TotalCharges"] = str(100 + 10 * number)
        row["Churn"] = "Yes" if number % 4 == 0 else "No"
        rows.append(row)
    return pd.DataFrame(rows)


class ProjectTests(unittest.TestCase):

    def module(self, name):
        self.assertTrue((PROJECT_DIR / f"{name}.py").is_file(), f"Implement {name}.py")
        return importlib.import_module(name)

    def test_cleaning_preserves_ids_and_leaves_imputation_for_training(self):
        data = self.module("data")
        with tempfile.TemporaryDirectory() as folder:
            raw = example_data()
            raw.loc[0, "TotalCharges"] = "   "
            raw.loc[0, "Contract"] = " Month-to-month "
            path = Path(folder) / "customers.csv"
            raw.to_csv(path, index=False)
            cleaned = data.load_data(path)
            self.assertEqual(cleaned.loc[0, "customerID"], "00000")
            self.assertEqual(cleaned.loc[0, "Contract"], "Month-to-month")
            self.assertEqual(cleaned.loc[0, "Churn"], 1)
            self.assertTrue(pd.isna(cleaned.loc[0, "TotalCharges"]))

    def test_invalid_input_fails_with_a_clear_error(self):
        data = self.module("data")
        bad_inputs = {
            "missing feature": example_data().drop(columns="tenure"),
            "extra feature": example_data().assign(unexpected=1),
            "missing ID": example_data().assign(customerID=" "),
            "duplicate IDs": example_data().assign(customerID="same-id"),
            "invalid charge": example_data().assign(TotalCharges="oops"),
            "infinite charge": example_data().assign(MonthlyCharges=np.inf),
            "negative charge": example_data().assign(MonthlyCharges=-1),
            "invalid label": example_data().assign(Churn="Maybe"),
            "blank category": example_data().assign(Contract=" "),
            "empty input": example_data().iloc[:0],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for name, frame in bad_inputs.items():
                with self.subTest(name=name):
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        data.load_data(path)

    def test_preprocessing_learns_only_from_the_rows_given_to_fit(self):
        data = self.module("data")
        preprocessing = self.module("preprocessing")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "customers.csv"
            example_data().to_csv(path, index=False)
            frame = data.load_data(path)
            model = preprocessing.build_pipeline()
            model.fit(frame.iloc[:20][data.FEATURE_COLUMNS], frame.iloc[:20]["Churn"])
            numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
            imputer = numeric.named_steps["imputer"]
            np.testing.assert_allclose(imputer.statistics_, [10.5, 29.5, 195.0])
            heldout = frame.iloc[20:][data.FEATURE_COLUMNS].copy()
            heldout["TotalCharges"] = 1e10
            heldout["Contract"] = "New contract"
            probabilities = model.predict_proba(heldout)
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(imputer.statistics_, [10.5, 29.5, 195.0])

    def test_a_fully_missing_numeric_column_keeps_its_place(self):
        data = self.module("data")
        preprocessing = self.module("preprocessing")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "customers.csv"
            example_data().assign(TotalCharges=" ").to_csv(path, index=False)
            frame = data.load_data(path)
            model = preprocessing.build_pipeline()
            model.fit(frame[data.FEATURE_COLUMNS], frame["Churn"])
            numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
            transformed = numeric.transform(frame[data.NUMERIC_COLUMNS])
            self.assertEqual(transformed.shape, (40, 3))
            np.testing.assert_array_equal(transformed[:, 2], np.zeros(40))

    def test_training_saves_a_complete_pipeline_and_readable_results(self):
        data = self.module("data")
        train = self.module("train")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "customers.csv"
            model_path = root / "models" / "churn_pipeline.pkl"
            reports = root / "reports"
            example_data().to_csv(path, index=False)
            metrics = train.train_model(path, model_path, reports)
            with model_path.open("rb") as file:
                model = pickle.load(file)
            self.assertIsInstance(model, Pipeline)
            frame = data.load_data(path).sort_values("customerID")
            training, testing = train_test_split(
                frame, test_size=0.2, random_state=42, stratify=frame["Churn"]
            )
            numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
            np.testing.assert_allclose(
                numeric.named_steps["imputer"].statistics_,
                training[data.NUMERIC_COLUMNS].median().to_numpy(),
            )
            self.assertEqual(metrics["training_rows"], len(training))
            self.assertEqual(metrics["test_rows"], len(testing))
            self.assertEqual(json.loads((reports / "metrics.json").read_text()), metrics)
            for name in ["training_distributions.png", "churn_by_contract.png", "confusion_matrix.png"]:
                self.assertGreater((reports / name).stat().st_size, 0)
            scored = pd.read_csv(reports / "test_predictions.csv", dtype={"customerID": str})
            self.assertEqual(scored["customerID"].tolist(), testing["customerID"].tolist())

    def test_too_few_churners_for_test_evaluation_fails_before_saving(self):
        train = self.module("train")
        with tempfile.TemporaryDirectory() as folder:
            raw = example_data().assign(Churn="No")
            raw.loc[:1, "Churn"] = "Yes"
            path = Path(folder) / "rare-churn.csv"
            raw.to_csv(path, index=False)
            model_path = Path(folder) / "model.pkl"
            with self.assertRaisesRegex(ValueError, "both"):
                train.train_model(path, model_path, Path(folder) / "reports")
            self.assertFalse(model_path.exists())

    def test_prediction_reloads_without_fitting_and_preserves_customer_order(self):
        data = self.module("data")
        preprocessing = self.module("preprocessing")
        predict = self.module("predict")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training_path = root / "training.csv"
            example_data().to_csv(training_path, index=False)
            frame = data.load_data(training_path)
            model = preprocessing.build_pipeline()
            model.fit(frame[data.FEATURE_COLUMNS], frame["Churn"])
            model_path = root / "model.pkl"
            with model_path.open("wb") as file:
                pickle.dump(model, file)
            batch = example_data().drop(columns="Churn").iloc[:3].iloc[::-1, ::-1]
            batch_path = root / "batch.csv"
            batch.to_csv(batch_path, index=False)
            with patch.object(Pipeline, "fit", side_effect=AssertionError("Prediction must not fit")):
                result = predict.predict_batch(batch_path, model_path, root / "predictions.csv")
            self.assertEqual(result["customerID"].tolist(), batch["customerID"].tolist())
            cleaned = data.load_data(batch_path, labeled=False)
            np.testing.assert_allclose(
                result["churn_probability"], model.predict_proba(cleaned[data.FEATURE_COLUMNS])[:, 1]
            )
            np.testing.assert_array_equal(result["predicted_churn"], model.predict(cleaned[data.FEATURE_COLUMNS]))
            batch.iloc[:1].to_csv(batch_path, index=False)
            self.assertEqual(len(predict.predict_batch(batch_path, model_path, root / "single.csv")), 1)
            with self.assertRaises(ValueError):
                predict.predict_batch(training_path, model_path, root / "invalid.csv")

    def test_output_paths_cannot_overwrite_inputs_or_each_other(self):
        data = self.module("data")
        preprocessing = self.module("preprocessing")
        train = self.module("train")
        predict = self.module("predict")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training_path = root / "training.csv"
            batch_path = root / "batch.csv"
            model_path = root / "model.pkl"
            example_data().to_csv(training_path, index=False)
            example_data().drop(columns="Churn").to_csv(batch_path, index=False)
            frame = data.load_data(training_path)
            model = preprocessing.build_pipeline()
            model.fit(frame[data.FEATURE_COLUMNS], frame["Churn"])
            with model_path.open("wb") as file:
                pickle.dump(model, file)
            original_data = training_path.read_bytes()
            original_batch = batch_path.read_bytes()
            original_model = model_path.read_bytes()
            for destination in [batch_path, model_path]:
                with self.subTest(destination=destination.name):
                    batch_path.write_bytes(original_batch)
                    model_path.write_bytes(original_model)
                    with self.assertRaisesRegex(ValueError, "overwrite"):
                        predict.predict_batch(batch_path, model_path, destination)
            self.assertEqual(batch_path.read_bytes(), original_batch)
            self.assertEqual(model_path.read_bytes(), original_model)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                train.train_model(training_path, training_path, root / "reports")
            self.assertEqual(training_path.read_bytes(), original_data)
            reports = root / "reports"
            reports.mkdir()
            report_input = reports / "test_predictions.csv"
            report_input.write_bytes(original_data)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                train.train_model(report_input, model_path, reports)
            self.assertEqual(report_input.read_bytes(), original_data)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                train.train_model(training_path, reports / "metrics.json", reports)
            self.assertFalse((reports / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
