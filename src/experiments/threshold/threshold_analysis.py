"""
Step 2: Threshold analysis for unknown species detection.

Reads results/threshold/diverse_retrieval_results.csv and tests multiple
threshold strategies to separate known from unknown species.

Strategies (score formulas using s0..s4 top-k neighbours):
  Raw scores:       s0, s1, s2, s3, s4
  Arithmetic:       abs_gap = s0-s1, gap_norm = (s0-s1)/(s0+s1), weighted_sum_exp = s0 - 0.5*s1 - 0.25*s2 - 0.125*s3 - 0.0625*s4
                    weighted_sum_lin = s0 - s1/2 - s2/3 - s3/4 - s4/5
                    gap_s0s2 = s0-s2, gap_s0s3 = s0-s3, gap_s0s4 = s0-s4
  Ratios:           ratio = s0/s1, ratio_s0s2 = s0/s2, ratio_s0s3 = s0/s3
  Products:         product = s0*s1, s0_sq = s0*s0
  Entropy:          neg_entropy = 1 - H_norm(s0..s4), max_minus_min = s0 - min(s0..s4)
  Averages:         top2_avg = (s0+s1)/2, top3_avg = (s0+s1+s2)/3, top4_avg = (s0+s1+s2+s3)/4
  Spread:           min_top2 = min(s0,s1), max_top2 = max(s0,s1), std_top3
  Geometric:       geom_mean = (s0*s1*s2)^(1/3), harm_mean = 3/(1/s0+1/s1+1/s2)

Threshold-finding algorithms:
  f1_grid    : sweep t, pick argmax F1
  roc_opt    : maximise Youden's J (sensitivity + specificity - 1)
  otsu       : minimise intra-class variance
  fpr_5pct   : threshold giving ~5% FPR on unknown samples
  fpr_10pct  : threshold giving ~10% FPR on unknown samples

Outputs:
  results/threshold/threshold_analysis.csv
  results/threshold/threshold_curves.png
  results/threshold/roc_curves.png
  results/threshold/confusion_matrices.png
  results/threshold/all_strategy_results.csv  (all strategy × algorithm F1s)

Usage:
    uv run python -m src.experiments.threshold.threshold_analysis
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402

INPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"
OUTPUT_DIR = RESULTS_DIR / "threshold"
ANALYSIS_CSV = OUTPUT_DIR / "threshold_analysis.csv"
ALL_RESULTS_CSV = OUTPUT_DIR / "all_strategy_results.csv"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_results(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        labels     : 1=known, 0=unknown  shape (N,)
        scores_mat : shape (N, 5) — s0..s4 scores (0 if missing)
    """
    labels = []
    scores_mat = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row.get("is_known", 0)))
            row_scores = []
            for i in range(5):
                val = row.get(f"s{i}_score", "")
                row_scores.append(float(val) if val else 0.0)
            scores_mat.append(row_scores)

    return np.array(labels, dtype=float), np.array(scores_mat, dtype=float)


# ---------------------------------------------------------------------------
# All strategies
# ---------------------------------------------------------------------------


def compute_all_strategy_scores(scores_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute 22 strategy scores per sample.
    """
    s = scores_arr  # shape (N, 5)
    eps = 1e-12

    strategies: Dict[str, np.ndarray] = {}

    # --- Raw scores ---
    for i in range(5):
        strategies[f"s{i}"] = s[:, i]

    # --- Arithmetic gap ---
    strategies["abs_gap"] = s[:, 0] - s[:, 1]
    strategies["gap_norm"] = (s[:, 0] - s[:, 1]) / (s[:, 0] + s[:, 1] + eps)

    # --- Weighted sums (decay) ---
    # exponential decay: s0 - 0.5*s1 - 0.25*s2 - 0.125*s3 - 0.0625*s4
    w_exp = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    strategies["weighted_sum_exp"] = np.sum(s * w_exp, axis=1)

    # linear decay: s0 - s1/2 - s2/3 - s3/4 - s4/5
    w_lin = np.array([1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 5])
    strategies["weighted_sum_lin"] = np.sum(s * w_lin, axis=1)

    # --- Gaps between non-adjacent ranks ---
    strategies["gap_s0s2"] = s[:, 0] - s[:, 2]
    strategies["gap_s0s3"] = s[:, 0] - s[:, 3]
    strategies["gap_s0s4"] = s[:, 0] - s[:, 4]

    # --- Ratios ---
    strategies["ratio_s0s1"] = s[:, 0] / (s[:, 1] + eps)
    strategies["ratio_s0s2"] = s[:, 0] / (s[:, 2] + eps)
    strategies["ratio_s0s3"] = s[:, 0] / (s[:, 3] + eps)

    # --- Products ---
    strategies["product_s0s1"] = s[:, 0] * s[:, 1]
    strategies["s0_sq"] = s[:, 0] * s[:, 0]

    # --- Entropy ---
    top5 = s[:, :5]
    row_sum = top5.sum(axis=1, keepdims=True) + eps
    p = top5 / row_sum
    p_safe = np.where(p > 0, p, eps)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    max_entropy = np.log(5)
    strategies["neg_entropy"] = 1.0 - entropy / max_entropy

    # --- Min of top-k ---
    strategies["min_top2"] = np.minimum(s[:, 0], s[:, 1])
    strategies["max_top2"] = np.maximum(s[:, 0], s[:, 1])

    # --- Averages ---
    strategies["top2_avg"] = (s[:, 0] + s[:, 1]) / 2.0
    strategies["top3_avg"] = (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0
    strategies["top4_avg"] = (s[:, 0] + s[:, 1] + s[:, 2] + s[:, 3]) / 4.0

    # --- Spread ---
    strategies["std_top3"] = s[:, :3].std(axis=1)

    # --- Geometric mean of top-3 ---
    geom = np.power(s[:, 0] * s[:, 1] * s[:, 2] + eps, 1.0 / 3.0)
    strategies["geom_mean_top3"] = geom

    # --- Top formulas from expanded analysis ---
    # Normalized gap (s0 vs s2) - best in expanded analysis
    strategies["gnorm_0_2"] = (s[:, 0] - s[:, 2]) / (s[:, 0] + s[:, 2] + eps)
    # Normalized entropy top-5
    top5 = s[:, :5]
    row_sum5 = top5.sum(axis=1, keepdims=True) + eps
    p5 = top5 / row_sum5
    p5_safe = np.where(p5 > 0, p5, eps)
    ent5 = -np.sum(p5_safe * np.log(p5_safe), axis=1)
    max_ent5 = np.log(5)
    strategies["ne_top5"] = 1.0 - ent5 / (max_ent5 + eps)
    # Ratio s0 to s3 (rank 3)
    strategies["ratio_0_3"] = s[:, 0] / (s[:, 3] + eps)
    # s0 relative to min of top-5
    strategies["s0_rel_min"] = s[:, 0] / (s[:, :5].min(axis=1) + eps)
    # Coefficient of variation top-5
    strategies["cv_top5"] = s[:, :5].std(axis=1) / (s[:, :5].mean(axis=1) + eps)

    return strategies


# ---------------------------------------------------------------------------
# Threshold algorithms
# ---------------------------------------------------------------------------


def f1_grid_threshold(
    scores: np.ndarray, labels: np.ndarray, n_steps: int = 500
) -> Tuple[float, float]:
    """Grid-search threshold maximising F1. Returns (best_t, best_f1)."""
    from sklearn.metrics import f1_score

    min_s, max_s = np.percentile(scores, 1), np.percentile(scores, 99)
    if min_s == max_s:
        return float(min_s), 0.0

    best_t, best_f1 = float(min_s), 0.0
    for t in np.linspace(min_s, max_s, n_steps):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def roc_optimal_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Find threshold maximising Youden's J = sensitivity + specificity - 1.
    Returns (best_t, auroc, sensitivity, specificity).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    idx = np.argmax(youden)
    best_t = float(thresholds[idx])
    sens = float(tpr[idx])
    spec = float(1.0 - fpr[idx])
    auroc = float(roc_auc_score(labels, scores))
    return best_t, auroc, sens, spec


def otsu_threshold(scores: np.ndarray, labels: np.ndarray, n_bins: int = 512) -> float:
    """Otsu's method: minimise intra-class variance."""
    min_s, max_s = np.percentile(scores, 1), np.percentile(scores, 99)
    if min_s == max_s:
        return float(min_s)

    thresholds = np.linspace(min_s, max_s, n_bins)
    best_t, best_var = thresholds[0], float("inf")

    for t in thresholds:
        above = scores >= t
        n_above, n_below = above.sum(), (~above).sum()
        if n_above == 0 or n_below == 0:
            continue
        w1, w2 = n_above / len(scores), n_below / len(scores)
        var1 = labels[above].var() if n_above > 1 else 0.0
        var2 = labels[~above].var() if n_below > 1 else 0.0
        intra_var = w1 * var1 + w2 * var2
        if intra_var < best_var:
            best_var = intra_var
            best_t = t

    return float(best_t)


def fpr_target_threshold(
    scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Find threshold that gives approximately target_fpr on unknown (label=0) samples.
    Returns (threshold, achieved_fpr).
    """
    from sklearn.metrics import roc_curve

    fpr_arr, _, thresholds = roc_curve(labels, scores)
    # Find threshold closest to target_fpr
    idx = np.argmin(np.abs(fpr_arr - target_fpr))
    return float(thresholds[idx]), float(fpr_arr[idx])


# ---------------------------------------------------------------------------
# Evaluate at threshold
# ---------------------------------------------------------------------------


def evaluate_at_threshold(
    scores: np.ndarray, labels: np.ndarray, t: float
) -> Dict[str, float]:
    """Compute TP, FP, TN, FN, precision, recall, F1 at threshold t."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = (scores >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    return {
        "threshold": t,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision_score(labels, preds, zero_division=0.0)),
        "recall": float(recall_score(labels, preds, zero_division=0.0)),
        "f1": float(f1_score(labels, preds, zero_division=0.0)),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Run full analysis — returns dict of all strategy-algorithm best F1s
# ---------------------------------------------------------------------------


ALGORITHMS = ["f1_grid", "roc_opt", "otsu"]
ALGORITHM_COLORS = {
    "f1_grid": "#2196F3",
    "roc_opt": "#4CAF50",
    "otsu": "#FF9800",
}


def run_analysis() -> Dict[str, Dict[str, float]]:
    """
    Run full threshold analysis.
    Returns: {f"{strategy}_{algo}": best_f1} for all combinations.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found: {INPUT_CSV}")
        print("Run: uv run python -m src.experiments.threshold.retrieve_diverse")
        sys.exit(1)

    print(f"Loading results from: {INPUT_CSV}")
    labels, scores_arr = load_results(INPUT_CSV)
    n_known = int(labels.sum())
    n_unknown = int((labels == 0).sum())
    print(f"  Total samples: {len(labels)} | Known: {n_known} | Unknown: {n_unknown}")

    if n_known == 0 or n_unknown == 0:
        print("ERROR: Need both known and unknown samples")
        sys.exit(1)

    strategies = compute_all_strategy_scores(scores_arr)
    print(f"\nStrategies ({len(strategies)}): {sorted(strategies.keys())}")
    print(f"Algorithms: {ALGORITHMS}\n")

    analysis_rows: List[Dict] = []
    all_f1s: Dict[str, Dict[str, float]] = {}  # strategy -> {algo: f1}

    for name, scores in sorted(strategies.items()):
        all_f1s[name] = {}

        # --- F1 grid ---
        best_t_f1, best_f1 = f1_grid_threshold(scores, labels)
        m = evaluate_at_threshold(scores, labels, best_t_f1)
        all_f1s[name]["f1_grid"] = best_f1
        analysis_rows.append(_make_row(name, "f1_grid", best_t_f1, m, None))

        # --- ROC optimal (Youden-J) ---
        try:
            best_t_roc, auroc, sens, spec = roc_optimal_threshold(scores, labels)
            m_roc = evaluate_at_threshold(scores, labels, best_t_roc)
            all_f1s[name]["roc_opt"] = m_roc["f1"]
            analysis_rows.append(_make_row(name, "roc_opt", best_t_roc, m_roc, auroc))
        except Exception:
            all_f1s[name]["roc_opt"] = 0.0
            analysis_rows.append(
                _make_row(
                    name,
                    "roc_opt",
                    0.0,
                    evaluate_at_threshold(scores, labels, 0.0),
                    0.0,
                )
            )

        # --- Otsu ---
        otsu_t = otsu_threshold(scores, labels)
        m_otsu = evaluate_at_threshold(scores, labels, otsu_t)
        all_f1s[name]["otsu"] = m_otsu["f1"]
        analysis_rows.append(_make_row(name, "otsu", otsu_t, m_otsu, None))

        # --- FPR @ 5% ---
        fpr5_t, achieved_fpr5 = fpr_target_threshold(scores, labels, 0.05)
        m_fpr5 = evaluate_at_threshold(scores, labels, fpr5_t)
        all_f1s[name]["fpr_5pct"] = m_fpr5["f1"]
        analysis_rows.append(_make_row(name, "fpr_5pct", fpr5_t, m_fpr5, achieved_fpr5))

        # --- FPR @ 10% ---
        fpr10_t, achieved_fpr10 = fpr_target_threshold(scores, labels, 0.10)
        m_fpr10 = evaluate_at_threshold(scores, labels, fpr10_t)
        all_f1s[name]["fpr_10pct"] = m_fpr10["f1"]
        analysis_rows.append(
            _make_row(name, "fpr_10pct", fpr10_t, m_fpr10, achieved_fpr10)
        )

    # --- Save analysis CSV (best F1 per strategy-algorithm) ---
    fields = [
        "strategy",
        "algorithm",
        "threshold",
        "f1",
        "precision",
        "recall",
        "specificity",
        "tp",
        "fp",
        "tn",
        "fn",
        "auroc",
    ]
    with open(ANALYSIS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis_rows)
    print(f"Analysis CSV: {ANALYSIS_CSV}")

    # --- Save ALL results (every row, not just best) ---
    with open(ALL_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis_rows)
    print(f"All results CSV: {ALL_RESULTS_CSV}")

    # --- Plots ---
    plot_threshold_curves(strategies, labels, OUTPUT_DIR / "threshold_curves.png")
    plot_roc_curves(strategies, labels, OUTPUT_DIR / "roc_curves.png")
    plot_confusion_matrices(
        strategies, labels, all_f1s, OUTPUT_DIR / "confusion_matrices.png"
    )

    # --- Summary: best per algorithm ---
    print("\n=== Best F1 per algorithm ===")
    for algo in ALGORITHMS:
        best_strat = max(all_f1s.keys(), key=lambda s: all_f1s[s].get(algo, 0.0))
        best_f1_val = all_f1s[best_strat].get(algo, 0.0)
        print(f"  {algo:12s}: F1={best_f1_val:.4f} ({best_strat})")

    print("\n=== All strategies ranked by f1_grid ===")
    ranked = sorted(
        all_f1s.keys(), key=lambda s: all_f1s[s].get("f1_grid", 0.0), reverse=True
    )
    for i, name in enumerate(ranked[:15], 1):
        f1g = all_f1s[name].get("f1_grid", 0.0)
        f1r = all_f1s[name].get("roc_opt", 0.0)
        print(f"  {i:2d}. {name:22s}  f1_grid={f1g:.4f}  roc_opt={f1r:.4f}")

    return all_f1s


def _make_row(
    strategy: str,
    algo: str,
    t: float,
    m: Dict[str, float],
    auroc: float | None,
) -> Dict:
    return {
        "strategy": strategy,
        "algorithm": algo,
        "threshold": f"{t:.6f}",
        "f1": f"{m['f1']:.6f}",
        "precision": f"{m['precision']:.6f}",
        "recall": f"{m['recall']:.6f}",
        "specificity": f"{m['specificity']:.6f}",
        "tp": m["tp"],
        "fp": m["fp"],
        "tn": m["tn"],
        "fn": m["fn"],
        "auroc": f"{auroc:.6f}" if auroc is not None else "",
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_threshold_curves(
    strategies: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot F1 vs threshold for each strategy (up to 18 in 3×6 grid)."""
    from sklearn.metrics import f1_score
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(strategies)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten()

    for ax, (name, scores) in zip(axes_flat, sorted(strategies.items())):
        mn, mx = np.percentile(scores[scores > 0], [2, 98])
        if mn == mx:
            ax.set_visible(False)
            continue
        ts = np.linspace(mn, mx, 300)
        f1s = [
            f1_score(labels, (scores >= t).astype(int), zero_division=0.0) for t in ts
        ]
        best_idx = int(np.argmax(f1s))

        color = ALGORITHM_COLORS.get(name, "#607D8B")
        ax.plot(ts, f1s, linewidth=1.5, color=color)
        ax.axvline(
            ts[best_idx],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"best t={ts[best_idx]:.3f}\nF1={f1s[best_idx]:.3f}",
        )
        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel("Threshold", fontsize=7)
        ax.set_ylabel("F1", fontsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("F1 vs Threshold — Unknown Species Detection", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_roc_curves(
    strategies: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC curves for each strategy, faceted by algorithm."""
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_algos = len(ALGORITHMS)
    cols = min(3, n_algos)
    rows = (n_algos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = np.array(axes).flatten()

    for ax, algo in zip(axes_flat, ALGORITHMS):
        for name, scores in sorted(strategies.items()):
            try:
                fpr_arr, tpr, _ = roc_curve(labels, scores)
                auc = roc_auc_score(labels, scores)
                lw = 2 if "f1_grid" in algo else 1.0
                ax.plot(
                    fpr_arr,
                    tpr,
                    linewidth=lw,
                    label=f"{name} (AUC={auc:.2f})",
                    alpha=0.8,
                )
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {algo}", fontsize=10)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes_flat[n_algos:]:
        ax.set_visible(False)

    fig.suptitle("ROC Curves by Algorithm", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_matrices(
    strategies: Dict[str, np.ndarray],
    labels: np.ndarray,
    all_f1s: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot top-12 strategy confusion matrices at f1_grid optimal threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Top 12 by f1_grid
    ranked = sorted(
        strategies.keys(), key=lambda s: all_f1s[s].get("f1_grid", 0.0), reverse=True
    )
    top12 = ranked[:12]

    cols = 4
    rows = (len(top12) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten()

    from sklearn.metrics import f1_score

    for ax, name in zip(axes_flat, top12):
        scores = strategies[name]
        mn, mx = np.percentile(scores[scores > 0], [2, 98])
        ts = np.linspace(mn, mx, 500)
        f1s = [
            f1_score(labels, (scores >= t).astype(int), zero_division=0.0) for t in ts
        ]
        best_t = ts[np.argmax(f1s)]

        preds = (scores >= best_t).astype(int)
        cm = confusion_matrix(labels.astype(int), preds)
        # Reformat for display: [[TN, FP], [FN, TP]]
        cm_display = np.array(
            [
                [
                    cm[0, 0] if cm.shape == (2, 2) else 0,
                    cm[0, 1] if cm.shape == (2, 2) else 0,
                ],
                [
                    cm[1, 0] if cm.shape == (2, 2) else 0,
                    cm[1, 1] if cm.shape == (2, 2) else 0,
                ],
            ]
        )

        best_f1 = max(f1s)
        ax.imshow(cm_display, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                color = "white" if cm_display[i, j] > cm_display.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(cm_display[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_title(f"{name}\nF1={best_f1:.3f}", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Unknown", "Known"])
        ax.set_yticklabels(["Pred Unknown", "Pred Known"])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    for ax in axes_flat[len(top12) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Confusion Matrices (Top-12 strategies, F1-grid threshold)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_analysis()
