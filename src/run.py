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
import json
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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
    "threshold": {
        "module": "src.experiments.threshold",
        "description": "Unknown species detection: threshold on similarity scores from diverse dataset",
        "default_params": {
            "strategy": "best",  # "best" = max F1 across all threshold strategies
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
        best = 0.0
        for r in self._rows:
            if r["kept"] != "1":
                continue
            acc = r["accuracy"]
            try:
                import json

                d = json.loads(acc)
                if isinstance(d, dict):
                    best = max(best, max(round(v, 6) for v in d.values()) if d else 0.0)
                else:
                    best = max(best, round(float(acc), 6))
            except Exception:
                best = max(best, round(float(acc), 6))
        return best

    def add(
        self, accuracy: float | Dict[str, float], description: str
    ) -> tuple[bool, int]:
        """
        Add a new result. accuracy can be a float or a dict of {strategy_algo: f1}.
        Returns (is_new_best, attempt_number).
        """
        if isinstance(accuracy, dict):
            acc_str = json.dumps(accuracy, sort_keys=True)
            max_f1 = max(round(v, 6) for v in accuracy.values()) if accuracy else 0.0
            is_new_best = max_f1 > round(self.best_accuracy, 6)
        else:
            acc_str = f"{accuracy:.6f}"
            is_new_best = round(accuracy, 6) > round(self.best_accuracy, 6)

        attempt = len(self._rows) + 1
        self._rows.append(
            {
                "attempt": attempt,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "accuracy": acc_str,
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

    When accuracy is a dict (multiple strategy-algorithm F1s per attempt),
    shows each strategy as a separate colored line.

    Saves to results/autoresearch/{experiment}.png
    """
    import json

    rows = history.attempts
    if not rows:
        return _get_experiment_png_path(experiment)

    attempts = [int(r["attempt"]) for r in rows]

    # Detect if this experiment stores dict of F1s (any row with dict = multi-line mode)
    any_dict = False
    all_keys: List[str] = []
    for r in rows:
        try:
            d = json.loads(r["accuracy"])
            if isinstance(d, dict) and len(d) > 1:
                any_dict = True
                for k in d.keys():
                    if k not in all_keys:
                        all_keys.append(k)
        except Exception:
            pass

    if not any_dict:
        # --- Single float accuracy per attempt ---
        accuracies = []
        for r in rows:
            try:
                accuracies.append(float(r["accuracy"]))
            except Exception:
                accuracies.append(0.0)
        kept_flags = [r["kept"] == "1" for r in rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_single_staircase(ax, attempts, accuracies, kept_flags, rows)
    elif experiment == "threshold":
        # --- Threshold experiment: plot ALL individual experiments as dots ---
        # Read from log/all_experiments.csv to get every formula × algorithm pair
        log_dir = RESULTS_DIR / "threshold" / "log"
        all_exp_csv = log_dir / "all_experiments.csv"
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_exp_csv.exists():
            _plot_threshold_all_experiments(ax, all_exp_csv)
        else:
            # Fallback: dict-based best-per-attempt
            _plot_dict_staircase(ax, rows, attempts)
    else:
        # --- Dict accuracy (non-threshold): show best per attempt ---
        _plot_dict_staircase(ax, rows, attempts)

    out_path = _get_experiment_png_path(experiment)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {out_path}")
    return out_path


def _plot_threshold_all_experiments(ax, all_exp_csv: Path) -> None:
    """
    Plot ALL individual experiments from log/all_experiments.csv as dots:
    - x = experiment index (0 to N-1)
    - y = F1 score
    - Gray dot: f1 ≤ running best (discarded)
    - Green dot: f1 > running best (new staircase step-up)
    - Green horizontal staircase connects green dots in chronological order
    - Labels on green dots show {formula}_{algorithm}
    """
    import csv

    # Load all experiments in order
    rows = []
    with open(all_exp_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f1 = float(row["f1"])
                if f1 <= 0 or f1 > 1:
                    continue  # skip invalid
                rows.append(
                    {
                        "formula": row["formula"],
                        "algorithm": row["algorithm"],
                        "f1": f1,
                    }
                )
            except (ValueError, KeyError):
                continue

    if not rows:
        ax.set_title("No experiment data found")
        return

    n = len(rows)

    # Compute running best and classify dots
    gray_x, gray_y = [], []
    green_x, green_y = [], []
    green_labels = []
    running_best = 0.0

    for i, row in enumerate(rows):
        f1 = row["f1"]
        if f1 > running_best:
            # New best — green dot + step up staircase
            green_x.append(i)
            green_y.append(f1)
            green_labels.append(f"{row['formula']}_{row['algorithm']}")
            running_best = f1
        else:
            # Discarded — gray dot
            gray_x.append(i)
            gray_y.append(f1)

    # Plot gray dots (discarded, below staircase)
    ax.scatter(gray_x, gray_y, color="#cccccc", s=15, zorder=2, label="discarded")

    # Plot green dots (new bests)
    ax.scatter(green_x, green_y, color="#2ca02c", s=60, zorder=4, label="new best")

    # Green horizontal-only staircase
    if green_x:
        stair_x = [green_x[0], green_x[0]]
        stair_y = [0.0, green_y[0]]
        for i in range(1, len(green_x)):
            stair_x.extend([green_x[i], green_x[i]])
            stair_y.extend([green_y[i - 1], green_y[i]])
        ax.plot(stair_x, stair_y, color="#2ca02c", linewidth=1.8, zorder=3)

        # Label each green dot
        for x, y, lbl in zip(green_x, green_y, green_labels):
            # Truncate label to 25 chars
            lbl_short = lbl[:25]
            ax.annotate(
                lbl_short,
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=6,
                color="#2ca02c",
                zorder=5,
                fontweight="bold",
            )

    max_f1 = max(r["f1"] for r in rows)
    ax.set_xlabel("Experiment index", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(
        f"Threshold experiment — {n} experiments (all formulas × algorithms)",
        fontsize=12,
    )
    ax.set_ylim(0, max_f1 * 1.15 if max_f1 > 0 else 1.0)
    ax.set_xlim(-1, n)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))


def _plot_dict_staircase(ax, rows, attempts) -> None:
    """Fallback dict plotting: best-per-attempt dots + green staircase."""
    best_per_attempt: List[float] = []
    best_key_per_attempt: List[str] = []
    for r in rows:
        try:
            d = json.loads(r["accuracy"])
            if isinstance(d, dict) and d:
                best_key = max(d.keys(), key=lambda k: d[k])
                best_per_attempt.append(d[best_key])
                best_key_per_attempt.append(best_key)
            else:
                best_per_attempt.append(float(r["accuracy"]))
                best_key_per_attempt.append("")
        except Exception:
            best_per_attempt.append(0.0)
            best_key_per_attempt.append("")
    kept_flags_multi = [r["kept"] == "1" for r in rows]

    # Gray dots for ALL attempts (discarded)
    ax.scatter(
        attempts,
        best_per_attempt,
        color="#cccccc",
        s=30,
        zorder=2,
        label="discarded",
    )

    # Green circles for kept (new best)
    kept_best_x = [attempts[i] for i, k in enumerate(kept_flags_multi) if k]
    kept_best_y = [best_per_attempt[i] for i, k in enumerate(kept_flags_multi) if k]
    kept_best_keys = [
        best_key_per_attempt[i] for i, k in enumerate(kept_flags_multi) if k
    ]
    ax.scatter(
        kept_best_x,
        kept_best_y,
        color="#2ca02c",
        s=80,
        zorder=4,
        label="new best",
    )

    # Green horizontal-only staircase
    if kept_best_x:
        stair_x = [kept_best_x[0], kept_best_x[0]]
        stair_y = [0.0, kept_best_y[0]]
        for i in range(1, len(kept_best_x)):
            stair_x.extend([kept_best_x[i], kept_best_x[i]])
            stair_y.extend([kept_best_y[i - 1], kept_best_y[i]])
        ax.plot(stair_x, stair_y, color="#2ca02c", linewidth=1.8, zorder=3)

        for x, y, key in zip(kept_best_x, kept_best_y, kept_best_keys):
            label = f"{key}={y:.3f}" if key else f"{y:.3f}"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=7,
                color="#2ca02c",
                zorder=5,
                fontweight="bold",
            )

    ax.set_xlabel("Experiment attempt", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, max(attempts) + 0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))


def _plot_single_staircase(ax, attempts, accuracies, kept_flags, rows) -> None:
    """
    Staircase chart — green staircase only connects kept (new best) points.
    Gray dots = discarded attempts (not connected by the staircase line).
    Green line is always green, horizontal-only staircase.
    """
    # Compute running best for each attempt
    running_best: List[float] = []
    current_best = 0.0
    for acc in accuracies:
        current_best = max(current_best, acc)
        running_best.append(current_best)

    # Plot discarded attempts as gray dots (NOT connected by staircase)
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
    kept_rows = [rows[i] for i, k in enumerate(kept_flags) if k]
    ax.scatter(
        kept_x,
        kept_y,
        color="#2ca02c",
        s=80,
        zorder=4,
        label="new best",
    )

    # Pure horizontal-only staircase:
    # For each kept point, draw a horizontal line from previous kept x to this kept x
    # at the y-level of the previous kept point (or first point starts at y=0).
    if kept_x:
        stair_x: List[float] = []
        stair_y: List[float] = []
        # Start at (kept_x[0], 0) then horizontal to (kept_x[0], kept_y[0])
        stair_x.extend([kept_x[0], kept_x[0]])
        stair_y.extend([0.0, kept_y[0]])
        for i in range(1, len(kept_x)):
            # Horizontal from previous kept_x to current kept_x at previous kept_y
            stair_x.extend([kept_x[i], kept_x[i]])
            stair_y.extend([kept_y[i - 1], kept_y[i]])
        ax.plot(
            stair_x,
            stair_y,
            color="#2ca02c",
            linewidth=1.8,
            zorder=3,
            label="running best",
        )

    # Annotate kept points with strategy name (not attempt number)
    for x, y, row in zip(kept_x, kept_y, kept_rows):
        desc = row.get("description", "") or ""
        # Label format: "(strategy_algo)" e.g. "(gap_norm f1_grid)"
        label = desc.strip()
        if label:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=7,
                color="#2ca02c",
                zorder=5,
                fontweight="bold",
            )

    ax.set_xlabel("Experiment attempt", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, max(attempts) + 0.5 if attempts else 1.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


# ---------------------------------------------------------------------------
# Run an experiment
# ---------------------------------------------------------------------------


def _run_experiment_fn(
    experiment: str,
    params: Dict[str, Any],
) -> float | Dict[str, float]:
    """
    Call the experiment's run function and return the accuracy number.
    Returns dict for threshold experiment (all strategy F1s), float otherwise.
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

    # For threshold: auto-generate description from best strategy if not provided
    description = args.description or ""
    if experiment == "threshold" and isinstance(accuracy, dict) and accuracy:
        best_key = max(accuracy.keys(), key=lambda k: accuracy[k])
        best_f1 = accuracy[best_key]
        if not description:
            description = f"{best_key}={best_f1:.4f}"

    # Record result
    history = ExperimentHistory(experiment)
    is_new_best, attempt = history.add(accuracy, description)

    print(f"\n{'=' * 50}")
    print(f"Experiment: {experiment} | Attempt #{attempt}")
    if isinstance(accuracy, dict):
        best = max(accuracy.values())
        top5 = sorted(accuracy.items(), key=lambda x: -x[1])[:5]
        print(f"Best F1:     {best:.4f} ({best:.2%})")
        print("Top strategies:")
        for name, f1 in top5:
            print(f"  {name}: {f1:.4f}")
    else:
        print(f"Accuracy:   {accuracy:.4f} ({accuracy:.2%})")
    print(f"Best so far: {history.best_accuracy:.4f} ({history.best_accuracy:.2%})")
    print(f"New best:  {'YES' if is_new_best else ('--- best unchanged ---')}")
    print(f"{'=' * 50}\n")

    # Log result for threshold experiment
    if experiment == "threshold":
        _log_threshold_result(attempt, is_new_best, accuracy, description)

    # Plot chart
    if not args.no_plot:
        plot_autoresearch_chart(experiment, history)

    # Exit code: 0 if new best, 1 otherwise
    sys.exit(0 if is_new_best else 0)  # always exit 0; use --check for CI


def _log_threshold_result(
    attempt: int,
    is_new_best: bool,
    accuracy: float | Dict[str, float],
    description: str,
) -> None:
    """Append a concise log entry to results/threshold/log/."""
    log_dir = RESULTS_DIR / "threshold" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "experiments.log"

    if isinstance(accuracy, dict) and accuracy:
        best_f1 = max(accuracy.values())
        best_key = max(accuracy.keys(), key=lambda k: accuracy[k])
        top3 = sorted(accuracy.items(), key=lambda x: -x[1])[:3]
        entry = (
            f"[attempt={attempt:03d}] {'★ NEW BEST' if is_new_best else '  stays   '} | "
            f"best_f1={best_f1:.4f} ({best_key}) | top3: "
            + " | ".join(f"{k}={v:.4f}" for k, v in top3)
        )
        # Also log full dict to a JSON file per attempt
        detail_path = log_dir / f"attempt_{attempt:03d}.json"
        with open(detail_path, "w") as f:
            json.dump(
                {
                    "attempt": attempt,
                    "description": description,
                    "best_f1": round(best_f1, 6),
                    "best_key": best_key,
                    "all_f1s": {k: round(v, 6) for k, v in sorted(accuracy.items())},
                },
                f,
                indent=2,
            )
    else:
        f1 = float(accuracy) if not isinstance(accuracy, dict) else 0.0
        entry = (
            f"[attempt={attempt:03d}] {'★ NEW BEST' if is_new_best else '  stays   '} | "
            f"f1={f1:.4f}"
        )

    with open(log_path, "a") as f:
        f.write(entry + "\n")
    print(f"Logged: {entry}")


if __name__ == "__main__":
    main()
