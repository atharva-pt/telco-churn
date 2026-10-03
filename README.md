# Telco Customer Churn Prediction

An end-to-end classification workflow that estimates customer churn from account, service and billing attributes. Built with Python, pandas, NumPy, scikit-learn and Matplotlib, it supports data validation, model training, evaluation and batch prediction.

## Features

- Shared CSV validation and cleaning for training and inference, with explicit feature columns and customer ID preservation.
- Numeric median imputation and standardization, plus one-hot encoding with support for unseen categories.
- A scikit-learn `Pipeline` that fits preprocessing only on training data and saves the fitted transformations with the classifier.
- Logistic regression evaluated against a majority-class baseline using accuracy, precision, recall, F1 and ROC-AUC.
- Batch predictions, customer-level evaluation results, distribution plots and a confusion matrix.
- Automated tests covering invalid inputs, training-only preprocessing statistics, saved-model inference, row order and output-path protection.

## Results

Dataset: **7,043 customers and 19 input features**. A stratified 80/20 split with seed 42 uses **5,634 training rows** and **1,409 test rows**. Customer IDs and the target label are excluded from model features. The classifier uses its default decision rule; no hyperparameter or threshold search is performed in this version.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8430 |
| Accuracy | 79.35% |
| Precision | 63.52% |
| Recall | 52.14% |
| F1 | 0.5727 |
| Majority-class baseline accuracy | 73.46% |

Results are recorded in `reports/metrics.json` after training. The test partition was inspected in earlier development iterations, so these are development results rather than an untouched independent benchmark. The dataset has no observation dates for assessing performance on future customer cohorts.

## Pipeline

```text
Numeric features     → Median imputation → Standardization ─┐
                                                           ├→ Logistic regression
Categorical features → One-hot encoding ────────────────────┘
```

Training separates rows before fitting the pipeline. Inference applies the same cleaning rules and reloads the complete fitted pipeline without refitting preprocessing or the classifier.

## Repository structure

```text
telco-churn/
├── data.py                 Input validation and cleaning
├── preprocessing.py        Preprocessing and classifier pipeline
├── train.py                Training, evaluation and model persistence
├── predict.py              Batch prediction CLI
├── plots.py                Training distributions and evaluation charts
├── tests/test_project.py   Automated behavior tests
├── data/example_batch.csv  Feature-only scoring sample
└── requirements.txt        Pinned dependencies
```

Raw data, the virtual environment, serialized models and generated reports are excluded from Git. Training creates `models/` and `reports/` locally.

## Setup

Use **Python 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/raw
```

Download the [Telco Customer Churn CSV from IBM's sample repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) and save it as `data/raw/telco.csv`. Training input contains `customerID`, the 19 feature columns and the `Churn` target with `Yes`/`No` labels.

## Train and predict

```bash
python train.py
python predict.py
```

The training command saves `models/churn_pipeline.pkl` and the evaluation outputs. The prediction command scores the included sample and writes `reports/predictions.csv`.

To score another CSV:

```bash
python predict.py --data path/to/customers.csv --output reports/customer_predictions.csv
```

Prediction input must contain the ID and feature columns, without `Churn`. Output contains `customerID`, `churn_probability` and `predicted_churn`, preserving input row order. `1` represents churn and `0` represents no churn. Run `python predict.py --help` for the model and file-path options.

Normal reruns replace generated outputs. Path checks prevent outputs from overwriting input data or replacing a model with a report. Load only trusted pickle files, using the same dependency versions as training.

The included sample comes from the previously reserved test customers and demonstrates the prediction interface; it is not a separate evaluation dataset.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Scope

Implemented as a local batch ML workflow. Deployment, monitoring, temporal validation and a business policy for acting on predictions are outside the current scope. The reported metrics do not establish the effect of a retention intervention.
