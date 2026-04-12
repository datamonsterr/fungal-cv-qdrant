import json
import os

import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def plot_training_history(
    history_path: str, output_path: str = str(RESULTS_DIR / "training_history.png")
) -> None:
    """
    Plot training history from a JSON file.
    """
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Training Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    if "accuracy" in history:
        plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def compare_models(
    results_dir: str = str(RESULTS_DIR),
    output_path: str = str(RESULTS_DIR / "model_comparison.png"),
) -> None:
    """
    Compare different models based on their evaluation results.
    Assumes results are stored in JSON files in the results directory.
    """
    # This is a placeholder for logic that would read multiple result files
    # and plot a comparison bar chart.
    pass


if __name__ == "__main__":
    pass
