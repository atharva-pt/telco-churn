# Telco customer churn

A Python package for training, evaluating, saving, and using a churn classifier. Built with NumPy, pandas, Matplotlib, scikit-learn, and Python's standard library. The interface is local batch prediction.

## Start here

Read `docs/learning-guide.md` alongside the source. Start with `data.py`, then `splits.py`, `preprocessing.py`, `train.py`, and `predict.py`. `__main__.py` shows how the pieces connect. Each module has one responsibility; no notebook state is required.

The source copy is `data/raw/telco.csv`. Your original Downloads file is preserved. Generated source copies, figures, model binaries, and the virtual environment are ignored by Git.

## Setup

Use Python 3.12. Run these commands from this project's directory. A working `.venv` is already included in the local workspace; the following commands reproduce it on another machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m unittest discover -s tests -v
```

The four direct data/ML dependencies also install their required dependencies. The lock file records exact installed versions, including installation tooling. No additional testing or ML framework is used. If sharing the repository, supply the source CSV separately at `data/raw/telco.csv`; it is deliberately not committed.

## Run the workflow

Activate the environment above first. Pick new output names for each experiment; commands refuse to overwrite earlier runs. The delivered example uses `reports/development` and `artifacts/telco-v1`. These commands use fresh names so they remain runnable:

```bash
python -m telco_churn validate --data data/raw/telco.csv

python -m telco_churn explore \
  --data data/raw/telco.csv \
  --config configs/training.json \
  --output reports/my-experiment

python -m telco_churn train \
  --data data/raw/telco.csv \
  --config configs/training.json \
  --split-manifest reports/my-experiment/split.json \
  --run-dir artifacts/my-experiment

python -m telco_churn evaluate \
  --data data/raw/telco.csv \
  --run-dir artifacts/my-experiment

python -m telco_churn predict \
  --data data/example_batch.csv \
  --run-dir artifacts/my-experiment \
  --output reports/my-experiment-predictions.csv
```

`data/example_batch.csv` is supplied locally as a scoring demonstration drawn from held-out customers. It is not independent evidence of model quality. The prediction input has the customer ID and 19 features, with no `Churn` column. Its columns may appear in any order. The output contains:

```text
customerID,churn_probability,predicted_churn,threshold,run_id
```

`predicted_churn` equals 1 when probability is greater than or equal to the saved threshold. Probabilities are model estimates and have not been calibrated.

## How the data moves

```text
CSV → deterministic validation → reserved train / validation / test IDs
                                  │
                    training rows → cross-validation → selected fitted pipeline
                                  │                             │
                 validation rows ───────────────────────→ decision threshold
                                                                │
                                                        complete saved bundle
                                                         /              \
                                               test evaluation       new batch scoring
```

About 60% trains the model, 20% selects the threshold, and 20% evaluates the frozen result. Five-fold cross-validation compares a dummy baseline, logistic regression, and a random forest using **average precision**. Every CV fold fits its own preprocessing. The chosen model remains fitted on training rows after threshold selection.

The default threshold maximizes validation F1. It is an explicit learning policy, not a financially optimized retention decision. Use `fixed_0_5` in a new experiment's configuration to study the difference. Once you inspect the test result, further tuning makes that test set part of your development process, even if you choose a new run directory. The software prevents accidental repeat evaluation within one run; scientific discipline still matters across runs.

## Files you will use

| Path | Purpose |
|---|---|
| `src/telco_churn/data.py` | Schema and deterministic cleaning |
| `src/telco_churn/config.py` | Validated settings |
| `src/telco_churn/splits.py` | Source-bound train/validation/test membership |
| `src/telco_churn/preprocessing.py` | Complete fitted pipeline |
| `src/telco_churn/train.py` | Cross-validation and model selection |
| `src/telco_churn/metrics.py` | Evaluation definitions and threshold policy |
| `src/telco_churn/artifacts.py` | Saving, compatibility, and checksums |
| `src/telco_churn/evaluate.py` | Frozen test evaluation |
| `src/telco_churn/predict.py` | Batch inference |
| `src/telco_churn/plots.py` | Reproducible charts |
| `tests/` | Contract, leakage, persistence, and CLI tests |
| `configs/training.json` | Small, explicit model search |
| `docs/model-card.md` | Measured results and limits |

A training bundle contains `pipeline.pkl`, `metadata.json`, `split.json`, `cv_results.json`, `validation.json`, and a `COMPLETE` marker. Evaluation adds its own `evaluation/metrics.json`, three figures, and a completion marker. Inspect JSON files directly before opening a binary artifact.

## Failure modes are part of the interface

- Invalid columns, undocumented categories, missing labels, duplicate IDs, malformed numeric tokens, negative charges, and fractional tenure fail validation with field-level errors.
- Blank numeric values are permitted and imputed inside training folds. Categorical blanks are rejected. Numeric columns that are entirely empty in a fit fold use the documented zero fallback while retaining their feature position.
- A category permitted by the schema but absent from a training fold can still be scored. Its one-hot block is all zeros; the model has not learned a category-specific effect for it.
- A different source CSV or configuration cannot silently reuse a split manifest. Make a new manifest for an intentional new experiment.
- Existing output paths fail rather than being overwritten. An interrupted model/evaluation directory remains incomplete and cannot be mistaken for a successful run. Inspect it and choose a fresh directory for a retry.
- Expected input or filesystem failures exit with status 2. Logs contain aggregate counts and field names, not customer records.

## Model trust and reproducibility

Only load models produced locally by this project that you trust. Python pickle can execute code; a matching checksum is not a security boundary. Loading checks completion, schema, source-code hash, model/report hashes, Python major/minor, and exact installed package versions. Using the dedicated locked environment makes compatibility checks predictable. Editing package code intentionally invalidates previously saved bundles: restore the matching revision or train a new run before evaluating a still-unseen holdout.

This project is a reproducible offline learning system. The CSV contains no observation timestamps, so it cannot prove a future prediction horizon, temporal performance, or that a retention action causes fewer customers to leave. A live rollout would also need validated feature availability, dated evaluation, monitoring ownership, and a business decision policy.
