from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from src.config import RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_fold_accuracy_plot(results_root: Path) -> Path:
    metrics_path = results_root / "metrics.csv"
    df = pd.read_csv(metrics_path)
    summary = df.groupby("fold_id", as_index=False)["metric_accuracy"].mean()
    figure_path = results_root / "fold_accuracy.png"
    plt.figure(figsize=(6, 4))
    plt.plot(summary["fold_id"], summary["metric_accuracy"], marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Fold")
    plt.ylabel("Mean accuracy")
    plt.title("YOLO cross-validation fold accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def build_visualization_index(results_root: Path | None = None) -> dict[str, object]:
    root = results_root or (RESULTS_DIR / "cross_validation_yolo")
    generated = []
    metrics_path = root / "metrics.csv"
    if metrics_path.exists():
        generated.append(str(generate_fold_accuracy_plot(root)))
    figures = sorted({str(path) for path in root.glob("**/*.png")} | set(generated))
    return {
        "results_root": str(root),
        "figure_count": len(figures),
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize YOLO cross-validation visualizations"
    )
    parser.add_argument(
        "--results-root", type=Path, default=RESULTS_DIR / "cross_validation_yolo"
    )
    args = parser.parse_args()
    print(json.dumps(build_visualization_index(args.results_root), indent=2))


if __name__ == "__main__":
    main()
