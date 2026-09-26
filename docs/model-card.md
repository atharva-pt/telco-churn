# Model card: telco-v2

## Intended use

An offline learning project that classifies the supplied Telco churn label and demonstrates a reproducible batch prediction interface. It has not been deployed or validated for a real future retention decision.

## Data and protocol

- Source file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`, provided locally by the learner.
- Dataset: 7,043 customers, 19 predictors, one customer identifier, and one Yes/No target.
- Source SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
- Split seed: 42; training 4,225, validation 1,409, test 1,409 customers.
- Customer IDs are excluded from model features. Numerical imputation/scaling and one-hot encoding are fitted inside each training CV fold.
- Five-fold stratified CV selects by average precision. Validation selects the maximum-F1 threshold; exact ties prefer the higher threshold. The selected model remains fitted only on training rows.
- Final test evaluation: `2026-09-26T05:36:31.417475+00:00`. No model or threshold changes were made using this result.

## Selected model

`RandomForestClassifier` with 200 trees, maximum depth 6, and minimum leaf size 5.

Training CV average precision: **0.6707**, fold-score standard deviation **0.0085**. This is fold variability, not a confidence interval.

Decision threshold: **0.398925**. Predict churn when estimated probability is greater than or equal to this threshold.

| Candidate | Parameters | Mean CV average precision | Fold standard deviation |
|---|---|---:|---:|
| DummyClassifier | `{}` | 0.2653 | 0.0005 |
| LogisticRegression | `{"model__C": 0.1}` | 0.6669 | 0.0217 |
| LogisticRegression | `{"model__C": 1.0}` | 0.6676 | 0.0205 |
| LogisticRegression | `{"model__C": 10.0}` | 0.6668 | 0.0192 |
| RandomForestClassifier | `{"model__max_depth": 6, "model__min_samples_leaf": 1, "model__n_estimators": 200}` | 0.6679 | 0.0095 |
| RandomForestClassifier | `{"model__max_depth": 6, "model__min_samples_leaf": 5, "model__n_estimators": 200}` | 0.6707 | 0.0085 |
| RandomForestClassifier | `{"model__max_depth": null, "model__min_samples_leaf": 1, "model__n_estimators": 200}` | 0.6264 | 0.0188 |
| RandomForestClassifier | `{"model__max_depth": null, "model__min_samples_leaf": 5, "model__n_estimators": 200}` | 0.6703 | 0.0114 |

## Validation decision policy

At threshold 0.5, validation precision was 0.6241, recall 0.4706, and F1 0.5366. At the selected threshold, precision was 0.5908, recall 0.6524, and F1 0.6201. The lower threshold trades more false alerts for more captured churners. F1 is an educational choice; no actual contact costs or retention benefits were supplied.

## Reserved test results

Test customers: 1,409; 374 churn and 1,035 non-churn. Always predicting non-churn would achieve 73.46% accuracy on this partition.

| Metric | Value |
|---|---:|
| average precision | 0.6638 |
| roc auc | 0.8432 |
| precision | 0.5925 |
| recall | 0.6337 |
| f1 | 0.6124 |
| accuracy | 0.7871 |

The confusion matrix uses actual labels as rows and predicted labels as columns:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 872 | 163 |
| Actual churn | 137 | 237 |

The model flagged 400 customers: 237 had a churn label and 163 did not. It missed 137 churners. Average precision is not trapezoidal PR-AUC. Full-precision metrics and figures are stored in `artifacts/telco-v2/evaluation/`.

## One exploration finding

- Month-to-month: 43.2% observed churn among 2318 training customers.
- One year: 10.4% observed churn among 868 training customers.
- Two year: 2.8% observed churn among 1039 training customers.

These are training-data associations, not evidence that changing a contract causes a change in churn.

## Limits and assumptions

- The CSV has no observation/event timestamps. There is no verified future prediction horizon or evaluation on a later cohort.
- Predictor availability at scoring time is an assumption, not demonstrated by this snapshot.
- Probabilities have not been calibrated. Threshold selection does not establish monetary benefit or the effect of a retention intervention.
- Results describe a single deterministic split, not deployment reliability or performance on a changing population.
- Training contains historical customer attributes; population coverage and subgroup performance require further validation for any real application.
- The ten-row example batch is drawn from held-out records and demonstrates scoring mechanics; it is not independent performance evidence.
- A live rollout would need dated data, feature availability checks, a decision policy, monitoring ownership, and a process for receiving outcome labels.

## Reproduction and verification

Use the pinned Python 3.12 environment and README commands. Loading requires matching code and packages and trusted local model files. The source code fingerprint is `4e976ecfef95d70e5f8a7a60bed07ce1f7e932bb15aa2754015c897e58f719c3`.

The final suite passed **29 tests** in a second clean virtual environment. The clean installation reproduced all ten example probabilities within `rtol=1e-10, atol=1e-12`, with matching IDs, row order, and decisions. Both environments passed dependency checks. The original Downloads CSV still matches the recorded fingerprint.
