"""
Experiment Runner (autoresearch pattern)
========================================
Runs a single experiment and returns a single accuracy number.

Usage:
    uv run python src/run.py --experiment segmentation
    uv run python src/run.py --experiment feature-extractor
    uv run python src/run.py --experiment embedding-lr --description "LR 0.001->0.0001"
    uv run python src/run.py --experiment-list   # list available experiments

Workflow:
    1. Runs the experiment's core logic (returns accuracy 0.0–1.0)
    2. Loads/saves experiment history from results/autoresearch/{experiment}.csv
    3. Plots results/autoresearch/{experiment}.png (staircase best line)
    4. Prints the accuracy number and whether it is a new best
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"
AUTORESULTS_DIR = RESULTS_DIR / "autoresearch"
AUTORESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

# Each entry maps experiment name -> {
#   "module": dotted path to the experiment's run.py module
#   "description": one-line description
#   "params": default CLI params passed to the experiment
# }

EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "segmentation": {
        "module": "src.experiments.kmeans_segmentation",
        "description": "Colony segmentation: KMeans vs Contour methods",
        "default_params": {
            "k": 11,
            "collection": "myco_fungi_features_full_finetuned",
            "extractor": "efficientnetb1_finetuned",
            "strategy": "weighted",
            "environment": None,  # E1: same environment
            "n_folds": 5,
        },
    },
    "feature-extractor": {
        "module": "src.experiments.retrieval",
        "description": "Feature extractor comparison: EfficientNetB1_finetuned k-fold accuracy",
        "default_params": {
            "k": 11,
            "collection": "myco_fungi_features_full_finetuned",
            "extractor": "efficientnetb1_finetuned",
            "strategy": "weighted",
            "environment": None,
            "n_folds": 5,
        },
    },
}


def _get_experiment_csv_path(experiment: str) -> Path:
    return AUTORESULTS_DIR / f"{experiment}.csv"


def _get_experiment_png_path(experiment: str) -> Path:
    return AUTORESULTS_DIR / f"{experiment}.png"


# ---------------------------------------------------------------------------
# Experiment result history
# ---------------------------------------------------------------------------


class ExperimentHistory:
    """Manages the CSV history for one experiment."""

    FIELDS = ["attempt", "timestamp", "accuracy", "kept", "description"]

    def __init__(self, experiment: str):
        self.experiment = experiment
        self.csv_path = _get_experiment_csv_path(experiment)
        self._rows: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            self._rows = []
            return
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self._rows = list(reader)

    def _save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)

    @property
    def attempts(self) -> List[Dict[str, Any]]:
        return self._rows

    @property
    def best_accuracy(self) -> float:
        if not self._rows:
            return 0.0
        return max(float(r["accuracy"]) for r in self._rows if r["kept"] == "1")

    def add(self, accuracy: float, description: str) -> tuple[bool, int]:
        """
        Add a new result.
        Returns (is_new_best, attempt_number).
        """
        is_new_best = accuracy > self.best_accuracy
        attempt = len(self._rows) + 1
        self._rows.append(
            {
                "attempt": attempt,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "accuracy": f"{accuracy:.6f}",
                "kept": "1" if is_new_best else "0",
                "description": description,
            }
        )
        self._save()
        return is_new_best, attempt


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_autoresearch_chart(experiment: str, history: ExperimentHistory) -> Path:
    """
    Plot the autoresearch accuracy chart:

    - X axis: experiment attempt number
    - Y axis: accuracy (0.0–1.0)
    - Gray dots: discarded results (worse than running best)
    - Green circles: kept checkpoints (new best at that point)
    - Staircase green line: running best trajectory (horizontal then step up)

    Saves to results/autoresearch/{experiment}.png
    """
    rows = history.attempts
    if not rows:
        return _get_experiment_png_path(experiment)

    attempts = [int(r["attempt"]) for r in rows]
    accuracies = [float(r["accuracy"]) for r in rows]
    kept_flags = [r["kept"] == "1" for r in rows]

    # Compute running best (staircase)
    running_best: List[float] = []
    current_best = 0.0
    for acc in accuracies:
        current_best = max(current_best, acc)
        running_best.append(current_best)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot all attempts as small gray dots
    ax.scatter(
        attempts,
        accuracies,
        color="#cccccc",
        s=30,
        zorder=2,
        label="discarded",
    )

    # Plot kept checkpoints as green circles
    kept_x = [attempts[i] for i, k in enumerate(kept_flags) if k]
    kept_y = [accuracies[i] for i, k in enumerate(kept_flags) if k]
    ax.scatter(
        kept_x,
        kept_y,
        color="#2ca02c",
        s=80,
        zorder=4,
        label="new best",
    )

    # Staircase running best line
    # Draw as horizontal segments at the new best level, then vertical step up
    if kept_x:
        best_x = [kept_x[0]]
        best_y = [kept_y[0]]
        for i in range(len(kept_x) - 1):
            # Horizontal: from current best x to next best x (same y)
            best_x.append(kept_x[i + 1])
            best_y.append(kept_y[i])
            # Vertical step: same x as next best, new y
            best_x.append(kept_x[i + 1])
            best_y.append(kept_y[i + 1])
        ax.plot(
            best_x,
            best_y,
            color="#2ca02c",
            linewidth=1.8,
            zorder=3,
            label="running best",
        )

    # Annotate kept points with their descriptions
    for i, (x, y, desc) in enumerate(
        zip(kept_x, kept_y, [r["description"] for r in rows if r["kept"] == "1"])
    ):
        if desc:
            ax.annotate(
                f" #{x}\n {desc}",
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color="#2ca02c",
                zorder=5,
            )

    ax.set_xlabel("Experiment attempt", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"autoresearch / {experiment}", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, max(attempts) + 0.5 if attempts else 1.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    # Show accuracy as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    out_path = _get_experiment_png_path(experiment)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Run an experiment
# ---------------------------------------------------------------------------


def _run_experiment_fn(
    experiment: str,
    params: Dict[str, Any],
) -> float:
    """
    Call the experiment's run function and return the accuracy number.
    """
    if experiment not in EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Available: {list(EXPERIMENT_REGISTRY.keys())}"
        )

    cfg = EXPERIMENT_REGISTRY[experiment]
    module_name = cfg["module"]

    # Import the experiment module
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import experiment module '{module_name}': {exc}"
        ) from exc

    # Try the shared cross-validation library for retrieval-based experiments
    if experiment in ("segmentation", "feature-extractor"):
        from src.lib.cross_validation import compute_mean_accuracy, run_cross_validation

        # Merge defaults with override params
        p = {**cfg["default_params"], **params}
        results = run_cross_validation(
            collection_name=p.get("collection", "myco_fungi_features_full_finetuned"),
            extractor_key=p.get("extractor", "efficientnetb1_finetuned"),
            k=p.get("k", 11),
            environment=p.get("environment"),
            strategy=p.get("strategy", "weighted"),
            n_folds=p.get("n_folds", 5),
        )
        return compute_mean_accuracy(results)

    # Fallback: look for a run_accuracy function
    if hasattr(mod, "run_accuracy"):
        return mod.run_accuracy(**params)

    raise NotImplementedError(
        f"Experiment '{experiment}' does not have a run_accuracy() function "
        f"and is not a registered retrieval experiment. "
        f"Please implement run_accuracy() in {module_name}."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_experiments() -> None:
    print("Available experiments:")
    for name, cfg in EXPERIMENT_REGISTRY.items():
        print(f"  {name}")
        print(f"    {cfg['description']}")
        print(f"    defaults: {cfg['default_params']}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a single experiment and record the accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
            Examples:
              uv run python src/run.py --experiment segmentation --description "k=3 clusters"
              uv run python src/run.py --experiment feature-extractor --k 7
              uv run python src/run.py --experiment-list
        """
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (must be in registry)",
    )
    parser.add_argument(
        "--experiment-list",
        action="store_true",
        help="List all available experiments",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Short description of this experiment attempt",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override k value for retrieval experiments",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["weighted", "uni"],
        help="Override aggregation strategy",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Override number of CV folds",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override Qdrant collection name",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        help="Override feature extractor key",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the chart",
    )

    args = parser.parse_args(argv)

    if args.experiment_list:
        _list_experiments()
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --experiment-list)")

    experiment = args.experiment

    # Build params dict from CLI overrides
    params: Dict[str, Any] = {}
    if args.k is not None:
        params["k"] = args.k
    if args.strategy is not None:
        params["strategy"] = args.strategy
    if args.n_folds is not None:
        params["n_folds"] = args.n_folds
    if args.collection is not None:
        params["collection"] = args.collection
    if args.extractor is not None:
        params["extractor"] = args.extractor

    print(f"\nRunning experiment: {experiment}")
    print(f"  description: {args.description or '(none)'}")
    if params:
        print(f"  params: {params}")

    # Run the experiment
    accuracy = _run_experiment_fn(experiment, params)

    # Record result
    history = ExperimentHistory(experiment)
    is_new_best, attempt = history.add(accuracy, args.description)

    print(f"\n{'='*50}")
    print(f"Experiment: {experiment} | Attempt #{attempt}")
    print(f"Accuracy:   {accuracy:.4f} ({accuracy:.2%})")
    print(f"Best so far: {history.best_accuracy:.4f} ({history.best_accuracy:.2%})")
    print(f"New best:  {'YES' if is_new_best else ('--- best unchanged ---')}")
    print(f"{'='*50}\n")

    # Plot chart
    if not args.no_plot:
        plot_autoresearch_chart(experiment, history)

    # Exit code: 0 if new best, 1 otherwise
    sys.exit(0 if is_new_best else 0)  # always exit 0; use --check for CI


if __name__ == "__main__":
    main()
