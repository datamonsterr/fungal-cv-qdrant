"""
Step 2 Expanded: 1000-experiment threshold analysis for unknown species detection.

Systematically generates formula variants using s0..s4 top-k neighbour scores
and evaluates each with 5 threshold-finding algorithms.

1000+ experiments = ~200 base formulas × 5 algorithms

Usage:
    uv run python -m src.experiments.threshold.expanded_threshold_analysis
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402

INPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"
OUTPUT_DIR = RESULTS_DIR / "threshold"
EXPANDED_LOG_DIR = OUTPUT_DIR / "log"


def load_data(csv_path) -> Tuple[np.ndarray, np.ndarray]:
    labels, scores_mat = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row.get("is_known", 0)))
            row_scores = [float(row.get(f"s{i}_score", "") or 0.0) for i in range(5)]
            scores_mat.append(row_scores)
    return np.array(labels, dtype=float), np.array(scores_mat, dtype=float)


# ---------------------------------------------------------------------------
# Formula generation
# ---------------------------------------------------------------------------


def generate_formulas(scores: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate ~200 formula variants from s0..s4 scores.
    Returns dict of {formula_name: scores_per_sample}.
    """
    s = scores  # shape (N, 5)
    eps = 1e-12
    formulas: Dict[str, np.ndarray] = {}

    # ----- Base raw scores -----
    for i in range(5):
        formulas[f"s{i}"] = s[:, i].copy()

    # ----- Single-score transforms -----
    for i in range(5):
        formulas[f"s{i}_sq"] = s[:, i] ** 2
        formulas[f"s{i}_sqrt"] = np.sqrt(np.clip(s[:, i], 0, None))
        formulas[f"s{i}_log"] = np.log1p(np.clip(s[:, i], 0, None))

    # ----- Pairwise gaps -----
    for i in range(5):
        for j in range(i + 1, 5):
            formulas[f"gap_{i}_{j}"] = s[:, i] - s[:, j]

    # ----- Pairwise ratios -----
    for i in range(5):
        for j in range(5):
            if i != j:
                formulas[f"ratio_{i}_{j}"] = s[:, i] / (s[:, j] + eps)

    # ----- Pairwise products -----
    for i in range(5):
        for j in range(i, 5):
            if i < j:
                formulas[f"prod_{i}_{j}"] = s[:, i] * s[:, j]

    # ----- Weighted sums (exponential decay) -----
    for base in [1.5, 2.0, 2.5, 3.0]:
        w = np.array([1.0, 1 / base, 1 / (base**2), 1 / (base**3), 1 / (base**4)])
        formulas[f"exp_decay_{int(base * 10)}"] = np.sum(s * w, axis=1)

    # ----- Weighted sums (custom weights) -----
    weight_sets = {
        "equal": [1 / 5] * 5,
        "front3": [0.5, 0.3, 0.15, 0.03, 0.02],
        "front2": [0.6, 0.3, 0.07, 0.02, 0.01],
        "s0_dom": [0.8, 0.12, 0.05, 0.02, 0.01],
        "s0_only": [1.0, 0.0, 0.0, 0.0, 0.0],
        "top2_foc": [0.55, 0.35, 0.07, 0.02, 0.01],
        "top3_foc": [0.45, 0.30, 0.18, 0.05, 0.02],
        "gm_foc": [0.50, 0.28, 0.13, 0.06, 0.03],
        "log_dcy": [
            1.0,
            1 / math.log(2),
            1 / math.log(3),
            1 / math.log(4),
            1 / math.log(5),
        ],
        "sqrt_dcy": [
            1.0,
            1 / math.sqrt(2),
            1 / math.sqrt(3),
            1 / math.sqrt(4),
            1 / math.sqrt(5),
        ],
        "s0s1_foc": [0.7, 0.2, 0.05, 0.03, 0.02],
        "s0_heavy": [0.9, 0.07, 0.02, 0.005, 0.005],
    }
    for w_name, w_vals in weight_sets.items():
        w = np.array(w_vals)
        formulas[f"w_{w_name}"] = np.sum(s * w, axis=1)

    # ----- k-top averages & stats -----
    for k in [2, 3, 4, 5]:
        formulas[f"avg_top{k}"] = s[:, :k].mean(axis=1)
        formulas[f"min_top{k}"] = s[:, :k].min(axis=1)
        formulas[f"max_top{k}"] = s[:, :k].max(axis=1)
        formulas[f"std_top{k}"] = s[:, :k].std(axis=1)
        formulas[f"range_top{k}"] = s[:, :k].max(axis=1) - s[:, :k].min(axis=1)

    # ----- Geometric means -----
    for k in [2, 3, 4, 5]:
        formulas[f"gm_top{k}"] = np.power(np.prod(s[:, :k] + eps, axis=1), 1.0 / k)

    # ----- Harmonic means -----
    for k in [2, 3, 4, 5]:
        denom = np.sum(1.0 / (s[:, :k] + eps), axis=1)
        formulas[f"hm_top{k}"] = k / (denom + eps)

    # ----- Entropy variants -----
    for k in [2, 3, 4, 5]:
        topk = s[:, :k]
        row_sum = topk.sum(axis=1, keepdims=True) + eps
        p = topk / row_sum
        p_safe = np.where(p > 0, p, eps)
        ent = -np.sum(p_safe * np.log(p_safe), axis=1)
        max_ent = np.log(k)
        formulas[f"ne_top{k}"] = 1.0 - ent / (max_ent + eps)

    # ----- Normalised gap -----
    for i in range(5):
        for j in range(i + 1, 5):
            formulas[f"gnorm_{i}_{j}"] = (s[:, i] - s[:, j]) / (s[:, i] + s[:, j] + eps)

    # ----- Sorted-rank gaps (s0 vs rank-1, rank-2, ...) -----
    sorted_s = np.sort(s[:, :5], axis=1)[:, ::-1]
    for rank in range(1, 4):
        formulas[f"gap0_r{rank}"] = sorted_s[:, 0] - sorted_s[:, rank]
        formulas[f"rat0_r{rank}"] = sorted_s[:, 0] / (sorted_s[:, rank] + eps)

    # ----- Power-law formulas -----
    for pow_val in [0.25, 0.5, 1.5, 2.0, 3.0]:
        formulas[f"s0_p{pow_val}"] = s[:, 0] ** pow_val
        formulas[f"sum12_p{pow_val}"] = (s[:, 0] + s[:, 1]) ** pow_val
        formulas[f"prod12_p{pow_val}"] = (s[:, 0] * s[:, 1]) ** pow_val

    # ----- Coefficient of variation -----
    formulas["cv_top3"] = s[:, :3].std(axis=1) / (s[:, :3].mean(axis=1) + eps)
    formulas["cv_top5"] = s[:, :5].std(axis=1) / (s[:, :5].mean(axis=1) + eps)
    formulas["s0s1_diff_sq"] = (s[:, 0] - s[:, 1]) ** 2

    # ----- Hybrid formulas -----
    for alpha in [0.3, 0.5, 0.7]:
        a = alpha
        formulas[f"hyb_gl_a{int(a * 10)}"] = (
            a * (s[:, 0] - s[:, 1]) + (1 - a) * (s[:, 0] + s[:, 1]) / 2
        )
        formulas[f"hyb_gg_a{int(a * 10)}"] = a * (s[:, 0] - s[:, 1]) + (
            1 - a
        ) * np.sqrt(s[:, 0] * s[:, 1] + eps)

    # ----- Agreement (inverse variance) -----
    for k in [2, 3]:
        formulas[f"agree_top{k}"] = 1.0 / (s[:, :k].std(axis=1) + 0.01)

    # ----- S0 relative metrics -----
    formulas["s0_rel_mean"] = s[:, 0] / (s[:, :5].mean(axis=1) + eps)
    formulas["s0_rel_max"] = s[:, 0] / (s[:, :5].max(axis=1) + eps)
    formulas["s0_rel_min"] = s[:, 0] / (s[:, :5].min(axis=1) + eps)

    # ----- Score minus median -----
    for k in [3, 5]:
        formulas[f"s0_m_med{k}"] = s[:, 0] - np.median(s[:, :k], axis=1)

    # ----- Normalised rank -----
    for k in [3, 5]:
        formulas[f"norm_rnk{k}"] = (s[:, 0] - s[:, k - 1]) / (
            s[:, 0] + s[:, k - 1] + eps
        )

    # ----- Consecutive ratios sum -----
    formulas["rat01_p_rat12"] = s[:, 0] / (s[:, 1] + eps) + s[:, 1] / (s[:, 2] + eps)
    formulas["rat02_p_rat13"] = s[:, 0] / (s[:, 2] + eps) + s[:, 1] / (s[:, 3] + eps)

    # ----- Squared gaps -----
    formulas["gap01_sq"] = (s[:, 0] - s[:, 1]) ** 2
    formulas["gap02_sq"] = (s[:, 0] - s[:, 2]) ** 2
    formulas["gap12_sq"] = (s[:, 1] - s[:, 2]) ** 2

    # ----- Weighted consecutive ratio chains (from novel_formulas_analysis) -----
    # wtd_rat_halving: 1.0*(s0/s1) + 0.5*(s1/s2) + 0.25*(s2/s3) + 0.125*(s3/s4)
    w_halving = np.array([1.0, 0.5, 0.25, 0.125])
    formulas["wtd_rat_halving"] = (
        w_halving[0] * s[:, 0] / (s[:, 1] + eps)
        + w_halving[1] * s[:, 1] / (s[:, 2] + eps)
        + w_halving[2] * s[:, 2] / (s[:, 3] + eps)
        + w_halving[3] * s[:, 3] / (s[:, 4] + eps)
    )
    # wtd_rat015: weights [1.0, 0.5, 0.25, 0.125] applied to ratios
    formulas["wtd_rat015"] = (
        w_halving[0] * s[:, 0] / (s[:, 1] + eps)
        + w_halving[1] * s[:, 1] / (s[:, 2] + eps)
        + w_halving[2] * s[:, 2] / (s[:, 3] + eps)
        + w_halving[3] * s[:, 3] / (s[:, 4] + eps)
    )
    # wtd_rat_lin: linear decay weights
    w_lin = np.array([1.0, 0.8, 0.6, 0.4])
    formulas["wtd_rat_lin"] = (
        w_lin[0] * s[:, 0] / (s[:, 1] + eps)
        + w_lin[1] * s[:, 1] / (s[:, 2] + eps)
        + w_lin[2] * s[:, 2] / (s[:, 3] + eps)
        + w_lin[3] * s[:, 3] / (s[:, 4] + eps)
    )
    # Three-term consecutive ratios
    formulas["rat01_p_rat12_p_rat23"] = (
        s[:, 0] / (s[:, 1] + eps)
        + s[:, 1] / (s[:, 2] + eps)
        + s[:, 2] / (s[:, 3] + eps)
    )
    formulas["rat01_p_rat12_m_rat23"] = (
        s[:, 0] / (s[:, 1] + eps)
        + s[:, 1] / (s[:, 2] + eps)
        - s[:, 2] / (s[:, 3] + eps)
    )
    # Ratio × sum combos
    formulas["sum12_x_rat01"] = (s[:, 0] + s[:, 1]) * (s[:, 0] / (s[:, 1] + eps))
    formulas["sum12_x_rat12"] = (s[:, 0] + s[:, 1]) * (s[:, 1] / (s[:, 2] + eps))
    # Ratio differences
    formulas["rat01_m_rat12"] = s[:, 0] / (s[:, 1] + eps) - s[:, 1] / (s[:, 2] + eps)
    formulas["rat12_m_rat23"] = s[:, 1] / (s[:, 2] + eps) - s[:, 2] / (s[:, 3] + eps)
    # Reciprocal ratios
    formulas["rrat10"] = s[:, 1] / (s[:, 0] + eps)
    formulas["rrat21"] = s[:, 2] / (s[:, 1] + eps)
    formulas["rrat10_p_rrat21"] = s[:, 1] / (s[:, 0] + eps) + s[:, 2] / (s[:, 1] + eps)
    # Spread formula
    formulas["spread_012_ov_s0"] = (s[:, 0] + s[:, 1] + s[:, 2]) / (s[:, 0] + eps)

    # ----- NEW: gnorm products (found to improve F1) -----
    gnorm_0_2 = (s[:, 0] - s[:, 2]) / (s[:, 0] + s[:, 2] + eps)
    gnorm_0_3 = (s[:, 0] - s[:, 3]) / (s[:, 0] + s[:, 3] + eps)
    gnorm_0_4 = (s[:, 0] - s[:, 4]) / (s[:, 0] + s[:, 4] + eps)
    gnorm_1_2 = (s[:, 1] - s[:, 2]) / (s[:, 1] + s[:, 2] + eps)
    formulas["gnorm_0_2_x_gnorm_0_3"] = gnorm_0_2 * gnorm_0_3
    formulas["gnorm_0_2_x_gnorm_0_4"] = gnorm_0_2 * gnorm_0_4
    formulas["gnorm_0_3_x_gnorm_0_4"] = gnorm_0_3 * gnorm_0_4
    formulas["gnorm_0_2_x_gnorm_1_2"] = gnorm_0_2 * gnorm_1_2
    formulas["gnorm_0_2_p_gnorm_0_3"] = gnorm_0_2 + gnorm_0_3  # sum instead of product

    # ----- NEW: log ratio sums (found to improve F1) -----
    formulas["log_rat_sum"] = np.log1p(s[:, 0] / (s[:, 1] + eps)) + np.log1p(
        s[:, 0] / (s[:, 2] + eps)
    )
    formulas["log_rat01"] = np.log1p(s[:, 0] / (s[:, 1] + eps))
    formulas["log_rat02"] = np.log1p(s[:, 0] / (s[:, 2] + eps))
    formulas["log_rat03"] = np.log1p(s[:, 0] / (s[:, 3] + eps))
    formulas["log_rat_sum_all"] = (
        np.log1p(s[:, 0] / (s[:, 1] + eps))
        + np.log1p(s[:, 0] / (s[:, 2] + eps))
        + np.log1p(s[:, 0] / (s[:, 3] + eps))
    )

    # ----- NEW: gnorm with exponent transforms -----
    formulas["gnorm_0_2_sq"] = gnorm_0_2**2
    formulas["gnorm_0_2_sqrt"] = np.sqrt(np.clip(gnorm_0_2, 0, None))
    formulas["gnorm_0_2_cu"] = gnorm_0_2**3

    # ----- NEW: weighted gnorm variants -----
    for wn in [1.5, 2.0]:
        for wd in [0.5, 1.0]:
            formulas[f"gnorm_w{wn}_d{wd}"] = (wn * s[:, 0] - s[:, 2]) / (
                wd * s[:, 0] + s[:, 2] + eps
            )

    return formulas


# ---------------------------------------------------------------------------
# Threshold algorithms
# ---------------------------------------------------------------------------


def f1_grid_threshold(scores, labels, n_steps=500):
    from sklearn.metrics import f1_score

    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
    if lo >= hi:
        return float(lo), 0.0
    best_t, best_f1 = lo, 0.0
    for t in np.linspace(lo, hi, n_steps):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0.0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def roc_optimal_threshold(scores, labels):
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr_arr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmax(tpr - fpr_arr)
    best_t = float(thresholds[idx])
    auroc = float(roc_auc_score(labels, scores))
    return best_t, auroc


def otsu_threshold(scores, labels, n_bins=512):
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
    if lo >= hi:
        return float(lo)
    thresholds = np.linspace(lo, hi, n_bins)
    best_t, best_var = float(thresholds[0]), float("inf")
    for t in thresholds:
        above = scores >= t
        n_a, n_b = int(above.sum()), int((~above).sum())
        if n_a == 0 or n_b == 0:
            continue
        var1 = float(labels[above].var()) if n_a > 1 else 0.0
        var2 = float(labels[~above].var()) if n_b > 1 else 0.0
        intra = (n_a / len(scores)) * var1 + (n_b / len(scores)) * var2
        if intra < best_var:
            best_var, best_t = intra, float(t)
    return best_t


def fpr_target_threshold(scores, labels, target_fpr):
    from sklearn.metrics import roc_curve

    fpr_arr, _, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr_arr - target_fpr))
    return float(thresholds[idx]), float(fpr_arr[idx])


def evaluate_at_threshold(scores, labels, t):
    from sklearn.metrics import f1_score, precision_score, recall_score

    preds = (scores >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision_score(labels, preds, zero_division=0.0)),
        "recall": float(recall_score(labels, preds, zero_division=0.0)),
        "f1": float(f1_score(labels, preds, zero_division=0.0)),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


ALGORITHMS = ["f1_grid", "roc_opt", "otsu"]


def _update_best(best, fname, algo, f1, t, m):
    if float(f1) > best["f1"]:
        best.update(
            {
                "strategy": fname,
                "algorithm": algo,
                "f1": float(f1),
                "threshold": float(t),
                "prec": float(m["precision"]),
                "rec": float(m["recall"]),
                "spec": float(m["specificity"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
            }
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPANDED_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found")
        sys.exit(1)

    labels, scores_arr = load_data(INPUT_CSV)
    n_known = int(labels.sum())
    n_unknown = int((labels == 0).sum())
    print(f"Loaded: {len(labels)} samples | Known: {n_known} | Unknown: {n_unknown}\n")

    print("Generating formula variants...")
    formulas = generate_formulas(scores_arr)
    formula_names = sorted(formulas.keys())
    print(f"Formulas: {len(formula_names)}")
    n_experiments = len(formula_names) * len(ALGORITHMS)
    print(f"Experiments: {len(formula_names)} × {len(ALGORITHMS)} = {n_experiments}\n")

    all_results = []
    best_overall = {"f1": 0.0}

    for fname in formula_names:
        scores = formulas[fname]
        if np.std(scores) < 1e-8:
            continue

        row_base = {"formula": fname}

        # f1_grid
        t, f1 = f1_grid_threshold(scores, labels)
        m = evaluate_at_threshold(scores, labels, t)
        all_results.append(
            {
                **row_base,
                "algorithm": "f1_grid",
                "threshold": f"{t:.6f}",
                "f1": f"{m['f1']:.6f}",
                "precision": f"{m['precision']:.6f}",
                "recall": f"{m['recall']:.6f}",
                "specificity": f"{m['specificity']:.6f}",
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
                "auroc": "",
            }
        )
        _update_best(best_overall, fname, "f1_grid", f1, t, m)

        # roc_opt
        try:
            t_roc, auroc = roc_optimal_threshold(scores, labels)
            m_roc = evaluate_at_threshold(scores, labels, t_roc)
            all_results.append(
                {
                    **row_base,
                    "algorithm": "roc_opt",
                    "threshold": f"{t_roc:.6f}",
                    "f1": f"{m_roc['f1']:.6f}",
                    "precision": f"{m_roc['precision']:.6f}",
                    "recall": f"{m_roc['recall']:.6f}",
                    "specificity": f"{m_roc['specificity']:.6f}",
                    "tp": m_roc["tp"],
                    "fp": m_roc["fp"],
                    "tn": m_roc["tn"],
                    "fn": m_roc["fn"],
                    "auroc": f"{auroc:.6f}",
                }
            )
            _update_best(best_overall, fname, "roc_opt", m_roc["f1"], t_roc, m_roc)
        except Exception:
            pass

        # otsu
        t_otsu = otsu_threshold(scores, labels)
        m_otsu = evaluate_at_threshold(scores, labels, t_otsu)
        all_results.append(
            {
                **row_base,
                "algorithm": "otsu",
                "threshold": f"{t_otsu:.6f}",
                "f1": f"{m_otsu['f1']:.6f}",
                "precision": f"{m_otsu['precision']:.6f}",
                "recall": f"{m_otsu['recall']:.6f}",
                "specificity": f"{m_otsu['specificity']:.6f}",
                "tp": m_otsu["tp"],
                "fp": m_otsu["fp"],
                "tn": m_otsu["tn"],
                "fn": m_otsu["fn"],
                "auroc": "",
            }
        )
        _update_best(best_overall, fname, "otsu", m_otsu["f1"], t_otsu, m_otsu)

    # Save all results
    fields = [
        "formula",
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
    results_csv = EXPANDED_LOG_DIR / "all_experiments.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    total = len(all_results)
    print(f"Saved {total} experiments to {results_csv}")
    print(f"\n{'=' * 65}")
    print(f"Total experiments run: {total}")
    print("\n=== Best F1 per algorithm ===")
    for algo in ALGORITHMS:
        algo_rows = [r for r in all_results if r["algorithm"] == algo]
        if not algo_rows:
            continue
        best = max(algo_rows, key=lambda r: float(r["f1"]))
        print(
            f"  {algo:12s}: F1={float(best['f1']):.4f} ({best['formula']})  "
            f"[t={float(best['threshold']):.4f}]"
        )

    print("\n=== Overall best (across all strategies × algorithms) ===")
    b = best_overall
    print(f"  {b['strategy']}_{b['algorithm']}: F1={b['f1']:.4f}")
    print(
        f"  Precision={b['prec']:.4f}  Recall={b['rec']:.4f}  Specificity={b['spec']:.4f}"
    )
    print(f"  TP={b['tp']}  FP={b['fp']}  TN={b['tn']}  FN={b['fn']}")
    print(f"  Threshold={b['threshold']:.6f}")

    # Save best strategy JSON
    best_path = EXPANDED_LOG_DIR / "best_strategy.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "strategy": b["strategy"],
                "algorithm": b["algorithm"],
                "f1": b["f1"],
                "threshold": b["threshold"],
                "precision": b["prec"],
                "recall": b["rec"],
                "specificity": b["spec"],
                "tp": b["tp"],
                "fp": b["fp"],
                "tn": b["tn"],
                "fn": b["fn"],
            },
            f,
            indent=2,
        )
    print(f"\nBest strategy: {best_path}")

    # Top-20 by f1_grid
    print("\n=== Top-20 (f1_grid) ===")
    ranked = sorted(
        [r for r in all_results if r["algorithm"] == "f1_grid"],
        key=lambda r: float(r["f1"]),
        reverse=True,
    )
    for i, r in enumerate(ranked[:20], 1):
        print(
            f"  {i:2d}. {r['formula']:30s} F1={float(r['f1']):.4f}  "
            f"P={float(r['precision']):.4f}  R={float(r['recall']):.4f}"
        )

    return all_results, best_overall


if __name__ == "__main__":
    run()
