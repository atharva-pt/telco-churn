# Learn the engineering through the code

You already know pandas. This guide focuses on the boundaries that turn a model experiment into a repeatable software workflow. Run a command, read its module, inspect its output, then try the exercise in a separate experiment.

## 1. Define what input means

Read `src/telco_churn/data.py` and run `validate`. A CSV can be readable yet unsuitable for a model: a duplicate customer ID, misspelled category, or shifted column can change its meaning.

`normalize_features` makes a copy and applies deterministic rules. A blank numeric field means missing; the string `oops` means invalid. Neither tells us which value should replace missing data. The median must be learned later, from training data. The same schema function is used during ingestion and inside the saved pipeline.

IDs remain strings so `00001` is not changed to `1`. IDs are used for joining and split membership, never as model features. Targets map Yes to 1 and No to 0 only at labeled ingestion; inference rejects the target column.

**Exercise:** Make a separate copy of three feature-only rows, reorder the columns, and add whitespace to a Contract value. Validate it. Then deliberately remove `tenure` and read the error. Explain why replacing every invalid value with a missing value would hide an upstream bug.

## 2. Decide which data may influence which decision

Read `config.py` and `splits.py`. Settings live in JSON and are validated before expensive work. A fixed seed alone is not a complete experiment record; the input bytes, settings, and actual customer IDs matter too.

The splitter sorts IDs, reserves test data, then divides development data into train and validation. Stratification preserves class representation. The manifest records the exact source hash and membership. Exploration uses only training rows. A changed source file must not reuse an old manifest silently.

**Exercise:** Inspect the three ID counts in `reports/development/split.json`. Explain why imputation, feature selection, or threshold choices made after reading test labels could make the final result optimistic.

## 3. Compose preprocessing with the estimator

Read `preprocessing.py`. `Pipeline` executes schema normalization, then `ColumnTransformer`, then the model. Numeric columns receive median imputation and scaling. Categorical columns receive one-hot encoding; `SeniorCitizen` is categorical even though its source values look numeric.

`fit` learns state; `transform` applies it. For a numeric feature, fitted state includes the median, mean, and scale. Cross-validation clones the entire pipeline so validation folds cannot leak into these statistics. Saving the whole pipeline prevents the serving code from independently recreating preprocessing with slightly different rules.

The custom schema transformer has a small purpose and follows scikit-learn's interface. Model types use existing estimators; no trainer inheritance framework is needed.

**Exercise:** Read `test_fit_statistics_exclude_heldout`. Its hand-computed medians are 9.5, 29.5, and 195. Explain which implementation mistake would make that test fail. Then read the all-missing-column test and explain the zero fallback.

## 4. Treat a saved model as a reproducible run

Read `artifacts.py` and inspect `metadata.json`. A model filename does not tell you which data, code, dependencies, or threshold produced it. A bundle records those together.

An exclusive write fails if a destination exists. The completion marker is written last: a half-written directory cannot load as a successful run. Code, data, model, and report fingerprints detect accidental mismatches. They do not establish trust in a pickle file supplied by another person.

**Exercise:** Inspect the run metadata without loading the pickle. Identify the chosen model, threshold, source hash, and Python version. Explain why changing the library version can matter even when your own Python source has not changed.

## 5. Separate ranking from an action decision

Read `train.py` and `metrics.py`. The dummy classifier gives a reference for whether learning helps. Logistic regression and random forest candidates are compared using average precision, a ranking metric suited to examining the positive class. It is not the same calculation as trapezoidal area under a precision-recall curve.

Cross-validation estimates how candidate settings generalize within the training portion. The winning pipeline is refitted once on that training portion. Validation probabilities then choose a threshold. A threshold changes decisions, not the underlying ranking: lowering it usually increases both captured churners and unnecessary alerts.

Default maximum F1 is a transparent exercise objective. It assumes a particular balance between precision and recall and contains no actual contact cost or retention benefit. The saved model is not refitted after selecting its threshold, so the probability mapping stays fixed.

**Exercise:** Before looking at test results, compare `validation.json` at 0.5 and at the selected threshold. Which policy catches more positives? Which sends more false alerts? Describe a business constraint that would change the preferred rule.

## 6. Keep evaluation and prediction free of fitting

Read `evaluate.py` and `predict.py`. Both load the same fitted pipeline. Evaluation selects the frozen test IDs and compares predictions with labels. Prediction receives feature-only rows and writes one result per input row, retaining order and IDs.

The confusion matrix is ordered [0,1]: rows are actual labels and columns are predicted labels. Its entries are true negatives, false positives, false negatives, and true positives. Precision asks how many alerts were correct; recall asks how many churners were found. Accuracy also counts the large non-churn group, so it needs context.

Model probability is an estimate, not a guarantee or an established causal effect. The example batch demonstrates the interface with held-out records; it is not a new evaluation dataset.

**Exercise:** Trace one row through `predict_batch`. Find where the positive probability column is selected and where `p >= threshold` becomes a decision. Explain why calling `fit_transform` on that batch would violate the model's contract.

## 7. Prove the workflow outside your interactive session

Read `tests/test_workflow.py`, then run the full test command from the README. The integration test uses synthetic data in a temporary directory and invokes new Python processes. This catches accidental dependence on the current directory, an imported notebook variable, or an already-fitted object in memory.

Tests target behavior: invalid input fails, training rows alone enter fitting, saved predictions survive reload, IDs stay in order, and repeat writes fail. Tests use tiny searches so they stay fast; the delivered dataset run uses the full documented search. Source changes have local Git commits so the implementation can be inspected increment by increment.

**Exercise:** Add a test for a failure you think could happen at a real data boundary. Make it fail first, then implement the smallest correction and rerun the suite. Keep the original holdout untouched while learning to change the code.

## What a real operating system would add

This CSV does not contain the observation and outcome dates needed to establish a future window. Before a real rollout, agree when each predictor exists, define the outcome window, evaluate on later dated cohorts, and determine who responds to alerts. Monitor input quality and changes in predictions; measure performance when delayed labels arrive. Retraining and retention actions need explicit ownership. Those requirements should drive additional infrastructure, rather than adding tools merely to make the folder structure look complex.

Further reading: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [pipelines](https://scikit-learn.org/stable/modules/compose.html), and [persistence](https://scikit-learn.org/stable/model_persistence.html).
