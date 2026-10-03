from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from data import NUMERIC_COLUMNS


def save_plots(training_data, actual_labels, predicted_labels, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, column in zip(axes, NUMERIC_COLUMNS):
        axis.hist(training_data[column].dropna(), bins=25, color="#28798c")
        axis.set_xlabel(column)
        axis.set_ylabel("Customers")
    figure.suptitle("Training data: numeric features")
    figure.savefig(output_dir / "training_distributions.png", dpi=150)
    plt.close(figure)

    churn_rates = training_data.groupby("Contract")["Churn"].mean()
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.bar(churn_rates.index, churn_rates.values, color="#28798c")
    axis.set_ylabel("Fraction with churn label")
    axis.set_ylim(0, 1)
    axis.set_title("Training data: churn by contract")
    figure.savefig(output_dir / "churn_by_contract.png", dpi=150)
    plt.close(figure)

    display = ConfusionMatrixDisplay.from_predictions(
        actual_labels, predicted_labels, labels=[0, 1],
        display_labels=["No churn", "Churn"], cmap="Blues", colorbar=False,
    )
    display.ax_.set_title("Test data: actual vs predicted")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)
