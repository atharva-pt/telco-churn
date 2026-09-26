# Build and verification record

This project was implemented locally on the `build-churn` branch. No remote repository was created and nothing was published.

## Execution decisions

- Used a dedicated new repository in this chat's outputs. There was no existing repository or branch to isolate in an additional worktree; existing work was not affected.
- Continued implementation directly after the learner asked where the code was. Earlier design/plan documents had delayed the requested package. The guide provides a path through the implemented code at the learner's own pace.
- Kept the result as a local project; there is no base branch or remote requiring integration.

## Independent review and corrections

A separate reviewer inspected the code and ran the suite. No ML leakage was found. Three findings were addressed in one correction pass:

1. Valid-looking edits to metadata could change the decision threshold or reported data provenance. Regression tests reproduced the failure. The completion record now contains the metadata hash, and loading also requires valid provenance fields before deserialization.
2. An interrupted evaluation could fail before reserving its output directory. This was treated as a required audit fix because the interface promises to record evaluation attempts. A regression test reproduced it; the attempt directory is now reserved before test-set scoring.
3. A test could pass despite dropping an entirely missing numeric feature. The assertion now checks all three numeric positions and the documented zero fallback. Temporarily mutating the imputer to drop empty features caused the strengthened test to fail, then the unchanged correct configuration passed.

The reviewer left real-data results to the final run and treated untrusted pickle security as outside the documented local-artifact contract. Final evaluation was performed after the fixes. The trust limitation remains explicit in the README; checksums do not make untrusted pickle safe. No reported minor findings remain deferred.

## Evidence

- Final clean-install suite: 29 tests passed, including a fresh-process workflow outside the project directory.
- Dependency checks: no broken requirements.
- Real training: eight candidates, five folds, 4,225 training rows; validation threshold chosen on 1,409 separate rows.
- Final model: random forest, 200 trees, depth 6, minimum leaf size 5, threshold 0.3989245445.
- Final test evaluation: performed once on 1,409 reserved customers. Results are in `model-card.md` and the saved metrics JSON.
- Prediction: ten feature-only records scored using the saved model; a second clean installation reproduced IDs, order, probabilities, and decisions.
- Source CSV: original Downloads bytes match the recorded SHA-256.
- Charts: training distributions, contract comparison, and test confusion matrix were visually inspected.

An initial model was trained before the review corrections but was never evaluated on the test set. It was preserved in the chat's intermediate work directory. The delivered compatible model is `artifacts/telco-v2`.
