"""
Step 2 Attempt 6: 100 novel formula variants for unknown species detection.

Novel structural families not covered in previous attempts:
1. Products of consecutive ratios (vs sums in rat01_p_rat12)
2. Ratios of gaps (gap_{i}_{j} / gap_{k}_{l})
3. Min/max/sum blends
4. Polynomial combos of adjacent pairs
5. Log-ratio variants
6. Harmonic-fraction variants
7. Cross-ratio chains
8. Score spread metrics

Usage:
    uv run python -m src.experiments.threshold.novel_formulas_analysis
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402

INPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"
OUTPUT_DIR = RESULTS_DIR / "threshold"
LOG_DIR = OUTPUT_DIR / "log"


def load_data(csv_path) -> Tuple[np.ndarray, np.ndarray]:
    labels, scores_mat = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row.get("is_known", 0)))
            row_scores = [float(row.get(f"s{i}_score", "") or 0.0) for i in range(5)]
            scores_mat.append(row_scores)
    return np.array(labels, dtype=float), np.array(scores_mat, dtype=float)


# ---------------------------------------------------------------------------
# 100 Novel formulas
# ---------------------------------------------------------------------------


def generate_formulas(scores: np.ndarray) -> Dict[str, np.ndarray]:
    s = scores  # shape (N, 5)
    eps = 1e-12
    formulas: Dict[str, np.ndarray] = {}

    # ── 1. Products of consecutive ratios (vs sum in rat01_p_rat12) ──────────
    formulas["prod_rat01_12"] = (s[:, 0] / (s[:, 1] + eps)) * (
        s[:, 1] / (s[:, 2] + eps)
    )
    formulas["prod_rat12_23"] = (s[:, 1] / (s[:, 2] + eps)) * (
        s[:, 2] / (s[:, 3] + eps)
    )
    formulas["prod_rat02_13"] = (s[:, 0] / (s[:, 2] + eps)) * (
        s[:, 1] / (s[:, 3] + eps)
    )
    formulas["prod_rat01_23"] = (s[:, 0] / (s[:, 1] + eps)) * (
        s[:, 1] / (s[:, 2] + eps)
    )
    formulas["prod_rat01_34"] = (s[:, 0] / (s[:, 1] + eps)) * (
        s[:, 2] / (s[:, 3] + eps)
    )
    formulas["prod_rat02_23"] = (s[:, 0] / (s[:, 2] + eps)) * (
        s[:, 1] / (s[:, 3] + eps)
    )

    # ── 2. Three-term ratio sums ─────────────────────────────────────────────
    formulas["rat012_sum"] = (
        s[:, 0] / (s[:, 1] + eps)
        + s[:, 1] / (s[:, 2] + eps)
        + s[:, 2] / (s[:, 3] + eps)
    )
    formulas["rat123_sum"] = (
        s[:, 1] / (s[:, 2] + eps)
        + s[:, 2] / (s[:, 3] + eps)
        + s[:, 3] / (s[:, 4] + eps)
    )
    formulas["rat013_sum"] = (
        s[:, 0] / (s[:, 1] + eps)
        + s[:, 1] / (s[:, 3] + eps)
        + s[:, 2] / (s[:, 4] + eps)
    )
    formulas["rat012_13_sum"] = (
        s[:, 0] / (s[:, 1] + eps)
        + s[:, 1] / (s[:, 2] + eps)
        + s[:, 0] / (s[:, 3] + eps)
    )

    # ── 3. Ratio × gap products ─────────────────────────────────────────────
    formulas["rat01_x_gap01"] = (s[:, 0] / (s[:, 1] + eps)) * (s[:, 0] - s[:, 1])
    formulas["rat12_x_gap12"] = (s[:, 1] / (s[:, 2] + eps)) * (s[:, 1] - s[:, 2])
    formulas["rat02_x_gap02"] = (s[:, 0] / (s[:, 2] + eps)) * (s[:, 0] - s[:, 2])
    formulas["rat01_x_gap12"] = (s[:, 0] / (s[:, 1] + eps)) * (s[:, 1] - s[:, 2])

    # ── 4. Ratios of gaps ───────────────────────────────────────────────────
    formulas["gap01_ov_gap12"] = (s[:, 0] - s[:, 1]) / (s[:, 1] - s[:, 2] + eps)
    formulas["gap01_ov_gap23"] = (s[:, 0] - s[:, 1]) / (s[:, 2] - s[:, 3] + eps)
    formulas["gap02_ov_gap13"] = (s[:, 0] - s[:, 2]) / (s[:, 1] - s[:, 3] + eps)
    formulas["gap12_ov_gap34"] = (s[:, 1] - s[:, 2]) / (s[:, 3] - s[:, 4] + eps)
    formulas["gap01_ov_gap02"] = (s[:, 0] - s[:, 1]) / (s[:, 0] - s[:, 2] + eps)

    # ── 5. Squared / cube-root / power transforms of ratios ─────────────────
    formulas["rat01_sq"] = (s[:, 0] / (s[:, 1] + eps)) ** 2
    formulas["rat12_sq"] = (s[:, 1] / (s[:, 2] + eps)) ** 2
    formulas["rat01_sqrt"] = np.sqrt(np.clip(s[:, 0] / (s[:, 1] + eps), 0, None))
    formulas["rat12_cbrt"] = np.power(
        np.clip(s[:, 1] / (s[:, 2] + eps), 0, None), 1 / 3
    )
    formulas["rat01_pow15"] = np.power(s[:, 0] / (s[:, 1] + eps), 1.5)
    formulas["rat01_pow025"] = np.power(s[:, 0] / (s[:, 1] + eps), 0.25)

    # ── 6. Gap ratios squared ────────────────────────────────────────────────
    formulas["gnorm01_sq"] = ((s[:, 0] - s[:, 1]) / (s[:, 0] + s[:, 1] + eps)) ** 2
    formulas["gnorm12_sq"] = ((s[:, 1] - s[:, 2]) / (s[:, 1] + s[:, 2] + eps)) ** 2
    formulas["gnorm02_sq"] = ((s[:, 0] - s[:, 2]) / (s[:, 0] + s[:, 2] + eps)) ** 2
    formulas["gnorm01_gnorm12_sum"] = (s[:, 0] - s[:, 1]) / (
        s[:, 0] + s[:, 1] + eps
    ) + (s[:, 1] - s[:, 2]) / (s[:, 1] + s[:, 2] + eps)

    # ── 7. Polynomial combos of adjacent pairs ───────────────────────────────
    formulas["poly2_01"] = s[:, 0] ** 2 - s[:, 1] ** 2
    formulas["poly2_12"] = s[:, 1] ** 2 - s[:, 2] ** 2
    formulas["poly2_02"] = s[:, 0] ** 2 - s[:, 2] ** 2
    formulas["poly2_01_12_sum"] = (s[:, 0] ** 2 - s[:, 1] ** 2) + (
        s[:, 1] ** 2 - s[:, 2] ** 2
    )
    formulas["poly2_01_12_prod"] = (s[:, 0] ** 2 - s[:, 1] ** 2) * (
        s[:, 1] ** 2 - s[:, 2] ** 2
    )

    # ── 8. Min / max blend formulas ─────────────────────────────────────────
    formulas["s0_mx_s1"] = np.maximum(s[:, 0], s[:, 1])
    formulas["s0_mn_s1"] = np.minimum(s[:, 0], s[:, 1])
    formulas["s0_mx_s1_x_gap01"] = np.maximum(s[:, 0], s[:, 1]) * (s[:, 0] - s[:, 1])
    formulas["s0_div_mn12"] = s[:, 0] / (np.minimum(s[:, 1], s[:, 2]) + eps)
    formulas["max_ov_sum12"] = np.maximum(s[:, 0], s[:, 1]) / (s[:, 0] + s[:, 1] + eps)
    formulas["min_ov_max12"] = np.minimum(s[:, 0], s[:, 1]) / (
        np.maximum(s[:, 0], s[:, 1]) + eps
    )

    # ── 9. Log-ratio variants ──────────────────────────────────────────────
    formulas["log_rat01"] = np.log1p(np.clip(s[:, 0] / (s[:, 1] + eps), 0, None))
    formulas["log_rat12"] = np.log1p(np.clip(s[:, 1] / (s[:, 2] + eps), 0, None))
    formulas["log_rat02"] = np.log1p(np.clip(s[:, 0] / (s[:, 2] + eps), 0, None))
    formulas["log_rat01_p_log_rat12"] = np.log1p(
        np.clip(s[:, 0] / (s[:, 1] + eps), 0, None)
    ) + np.log1p(np.clip(s[:, 1] / (s[:, 2] + eps), 0, None))
    formulas["log_gap01"] = np.log1p(np.clip(s[:, 0] - s[:, 1], 0, None))

    # ── 10. Cross-ratio chains (non-consecutive ratios) ──────────────────────
    formulas["rat03_sum_rat14"] = s[:, 0] / (s[:, 3] + eps) + s[:, 1] / (s[:, 4] + eps)
    formulas["rat04_sum_rat13"] = s[:, 0] / (s[:, 4] + eps) + s[:, 1] / (s[:, 3] + eps)
    formulas["rat03_prod_rat14"] = (s[:, 0] / (s[:, 3] + eps)) * (
        s[:, 1] / (s[:, 4] + eps)
    )
    formulas["rat01_prod_rat34"] = (s[:, 0] / (s[:, 1] + eps)) * (
        s[:, 2] / (s[:, 3] + eps)
    )

    # ── 11. Score spread / diversity metrics ─────────────────────────────────
    formulas["spread_012"] = s[:, :3].std(axis=1) / (s[:, :3].mean(axis=1) + eps)
    formulas["spread_012_ov_s0"] = (
        s[:, :3].std(axis=1) / (s[:, :3].mean(axis=1) + eps)
    ) / (s[:, 0] + eps)
    formulas["range_ov_s0"] = (s[:, :3].max(axis=1) - s[:, :3].min(axis=1)) / (
        s[:, 0] + eps
    )
    formulas["iqr_12"] = (
        np.percentile(s[:, :3], 75, axis=1) - np.percentile(s[:, :3], 25, axis=1)
    ) / (s[:, 0] + eps)

    # ── 12. Harmonic-fraction variants ──────────────────────────────────────
    formulas["hm_rat01_12"] = 2 / (
        1 / (s[:, 0] / (s[:, 1] + eps) + eps)
        + 1 / (s[:, 1] / (s[:, 2] + eps) + eps)
        + eps
    )
    formulas["hm_gap01_gap12"] = 2 / (
        1 / (s[:, 0] - s[:, 1] + eps) + 1 / (s[:, 1] - s[:, 2] + eps) + eps
    )

    # ── 13. Gapped ratio products ────────────────────────────────────────────
    formulas["rat0_gap12"] = s[:, 0] / (s[:, 1] - s[:, 2] + eps)
    formulas["rat1_gap23"] = s[:, 1] / (s[:, 2] - s[:, 3] + eps)
    formulas["rat2_gap34"] = s[:, 2] / (s[:, 3] - s[:, 4] + eps)
    formulas["rat0_gap23"] = s[:, 0] / (s[:, 2] - s[:, 3] + eps)

    # ── 14. Weighted ratio chains (non-uniform decay) ────────────────────────
    # base=1.5 decay weights applied to ratios
    w = np.array([1.0, 1 / 1.5, 1 / (1.5**2), 1 / (1.5**3)])
    formulas["wtd_rat015"] = (
        w[0] * s[:, 0] / (s[:, 1] + eps)
        + w[1] * s[:, 1] / (s[:, 2] + eps)
        + w[2] * s[:, 2] / (s[:, 3] + eps)
        + w[3] * s[:, 3] / (s[:, 4] + eps)
    )
    w2 = np.array([1.0, 0.5, 0.25, 0.125])
    formulas["wtd_rat_halving"] = (
        w2[0] * s[:, 0] / (s[:, 1] + eps)
        + w2[1] * s[:, 1] / (s[:, 2] + eps)
        + w2[2] * s[:, 2] / (s[:, 3] + eps)
        + w2[3] * s[:, 3] / (s[:, 4] + eps)
    )
    w3 = np.array([1.0, 0.8, 0.6, 0.4])
    formulas["wtd_rat_lin"] = (
        w3[0] * s[:, 0] / (s[:, 1] + eps)
        + w3[1] * s[:, 1] / (s[:, 2] + eps)
        + w3[2] * s[:, 2] / (s[:, 3] + eps)
        + w3[3] * s[:, 3] / (s[:, 4] + eps)
    )

    # ── 15. Ratio difference (gap of ratios) ─────────────────────────────────
    formulas["rat01_m_rat12"] = s[:, 0] / (s[:, 1] + eps) - s[:, 1] / (s[:, 2] + eps)
    formulas["rat12_m_rat23"] = s[:, 1] / (s[:, 2] + eps) - s[:, 2] / (s[:, 3] + eps)
    formulas["rat02_m_rat13"] = s[:, 0] / (s[:, 2] + eps) - s[:, 1] / (s[:, 3] + eps)
    formulas["rat01_m_rat13"] = s[:, 0] / (s[:, 1] + eps) - s[:, 1] / (s[:, 3] + eps)
    formulas["rat01_m_rat34"] = s[:, 0] / (s[:, 1] + eps) - s[:, 2] / (s[:, 3] + eps)

    # ── 16. Triple ratio products (3 ratios multiplied) ─────────────────────
    formulas["rat010_112_prod"] = (
        (s[:, 0] / (s[:, 1] + eps))
        * (s[:, 0] / (s[:, 2] + eps))
        * (s[:, 1] / (s[:, 2] + eps))
    )
    formulas["rat01_12_23_prod"] = (
        (s[:, 0] / (s[:, 1] + eps))
        * (s[:, 1] / (s[:, 2] + eps))
        * (s[:, 2] / (s[:, 3] + eps))
    )
    formulas["rat010_213_prod"] = (
        (s[:, 0] / (s[:, 1] + eps))
        * (s[:, 0] / (s[:, 3] + eps))
        * (s[:, 1] / (s[:, 3] + eps))
    )

    # ── 17. Sum × ratio combos ───────────────────────────────────────────────
    formulas["sum12_x_rat01"] = (s[:, 0] + s[:, 1]) * (s[:, 0] / (s[:, 1] + eps))
    formulas["sum12_x_rat12"] = (s[:, 0] + s[:, 1]) * (s[:, 1] / (s[:, 2] + eps))
    formulas["sum01_x_rat12"] = (s[:, 0] + s[:, 1]) * (s[:, 1] / (s[:, 2] + eps))
    formulas["sum12_x_rat02"] = (s[:, 0] + s[:, 1]) * (s[:, 0] / (s[:, 2] + eps))

    # ── 18. Normalised consecutive ratio difference ───────────────────────────
    formulas["gnorm_rat01_m_rat12"] = (
        s[:, 0] / (s[:, 1] + eps) - s[:, 1] / (s[:, 2] + eps)
    ) / (s[:, 0] / (s[:, 1] + eps) + s[:, 1] / (s[:, 2] + eps) + eps)
    formulas["gnorm_rat01_p_rat12"] = (
        s[:, 0] / (s[:, 1] + eps) + s[:, 1] / (s[:, 2] + eps)
    ) / (s[:, 0] / (s[:, 1] + eps) + s[:, 1] / (s[:, 2] + eps) + eps)

    # ── 19. Reciprocal ratios (s_{i+1}/s_i) ─────────────────────────────────
    formulas["rrat10"] = s[:, 1] / (s[:, 0] + eps)
    formulas["rrat21"] = s[:, 2] / (s[:, 1] + eps)
    formulas["rrat10_p_rrat21"] = s[:, 1] / (s[:, 0] + eps) + s[:, 2] / (s[:, 1] + eps)
    formulas["rrat10_m_rrat21"] = s[:, 1] / (s[:, 0] + eps) - s[:, 2] / (s[:, 1] + eps)

    # ── 20. Complex multi-term ratio expressions ─────────────────────────────
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
    formulas["rat01_x_rat12_p_rat23"] = (s[:, 0] / (s[:, 1] + eps)) * (
        s[:, 1] / (s[:, 2] + eps)
    ) + s[:, 2] / (s[:, 3] + eps)
    formulas["rat01_p_rat12_x_rat23"] = s[:, 0] / (s[:, 1] + eps) + (
        s[:, 1] / (s[:, 2] + eps)
    ) * (s[:, 2] / (s[:, 3] + eps))
    formulas["rat01_x_rat12_x_rat23"] = (
        (s[:, 0] / (s[:, 1] + eps))
        * (s[:, 1] / (s[:, 2] + eps))
        * (s[:, 2] / (s[:, 3] + eps))
    )

    # ── 21. Inverse gap ratios (gap normalised by sum) ────────────────────────
    formulas["inv_gnorm01"] = (s[:, 0] + s[:, 1] + eps) / (s[:, 0] - s[:, 1] + eps)
    formulas["inv_gnorm12"] = (s[:, 1] + s[:, 2] + eps) / (s[:, 1] - s[:, 2] + eps)
    formulas["inv_gnorm01_p_inv_gnorm12"] = (s[:, 0] + s[:, 1] + eps) / (
        s[:, 0] - s[:, 1] + eps
    ) + (s[:, 1] + s[:, 2] + eps) / (s[:, 1] - s[:, 2] + eps)

    # ── 22. s0 normalised by various aggregates ───────────────────────────────
    formulas["s0_ov_avg12"] = s[:, 0] / ((s[:, 0] + s[:, 1]) / 2 + eps)
    formulas["s0_ov_avg123"] = s[:, 0] / ((s[:, 0] + s[:, 1] + s[:, 2]) / 3 + eps)
    formulas["s0_ov_gm12"] = s[:, 0] / (np.sqrt(s[:, 0] * s[:, 1] + eps) + eps)
    formulas["s0_ov_hm12"] = s[:, 0] / (
        2 / (1 / (s[:, 0] + eps) + 1 / (s[:, 1] + eps) + eps) + eps
    )
    formulas["s0_m_avg12"] = s[:, 0] - (s[:, 0] + s[:, 1]) / 2
    formulas["s0_m_gm12"] = s[:, 0] - np.sqrt(s[:, 0] * s[:, 1] + eps)

    # ── 23. Entropy of top-k as feature (negentropy already tried; try shannon) ─
    for k in [2, 3]:
        topk = s[:, :k]
        row_sum = topk.sum(axis=1, keepdims=True) + eps
        p = topk / row_sum
        p_safe = np.where(p > 0, p, eps)
        shan = -np.sum(p_safe * np.log(p_safe), axis=1)
        formulas[f"shan_top{k}"] = shan
        formulas[f"shan_top{k}_ov_s0"] = shan / (s[:, 0] + eps)

    # ── 24. Ratio of successive gaps ─────────────────────────────────────────
    formulas["gap01_ov_gap01_p_gap12"] = (s[:, 0] - s[:, 1]) / (
        (s[:, 0] - s[:, 1]) + (s[:, 1] - s[:, 2]) + eps
    )
    formulas["gap01_p_gap12_ov_gap02"] = ((s[:, 0] - s[:, 1]) + (s[:, 1] - s[:, 2])) / (
        s[:, 0] - s[:, 2] + eps
    )

    # ── 25. Geometric mean of consecutive gnorms ─────────────────────────────
    formulas["gm_gnorm01_12"] = np.sqrt(
        ((s[:, 0] - s[:, 1]) / (s[:, 0] + s[:, 1] + eps))
        * ((s[:, 1] - s[:, 2]) / (s[:, 1] + s[:, 2] + eps))
    )
    formulas["gm_gnorm01_12_23"] = np.power(
        ((s[:, 0] - s[:, 1]) / (s[:, 0] + s[:, 1] + eps))
        * ((s[:, 1] - s[:, 2]) / (s[:, 1] + s[:, 2] + eps))
        * ((s[:, 2] - s[:, 3]) / (s[:, 2] + s[:, 3] + eps)),
        1 / 3,
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
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "specificity": float(m["specificity"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
            }
        )


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found")
        sys.exit(1)

    labels, scores_arr = load_data(INPUT_CSV)
    n_known = int(labels.sum())
    n_unknown = int((labels == 0).sum())
    print(f"Loaded: {len(labels)} samples | Known: {n_known} | Unknown: {n_unknown}\n")

    print("Generating 100 novel formula variants...")
    formulas = generate_formulas(scores_arr)
    formula_names = sorted(formulas.keys())
    print(f"Formulas: {len(formula_names)}")
    n_experiments = len(formula_names) * len(ALGORITHMS)
    print(f"Experiments: {len(formula_names)} × {len(ALGORITHMS)} = {n_experiments}\n")

    all_results = []
    best_overall = {"f1": 0.0}

    for fname in formula_names:
        sc = formulas[fname]
        if np.std(sc) < 1e-8:
            continue

        row_base = {"formula": fname}

        # f1_grid
        t, f1 = f1_grid_threshold(sc, labels)
        m = evaluate_at_threshold(sc, labels, t)
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
            t_roc, auroc = roc_optimal_threshold(sc, labels)
            m_roc = evaluate_at_threshold(sc, labels, t_roc)
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
        t_otsu = otsu_threshold(sc, labels)
        m_otsu = evaluate_at_threshold(sc, labels, t_otsu)
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
    results_csv = LOG_DIR / "all_experiments.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    total = len(all_results)
    print(f"Saved {total} experiments to {results_csv}")
    print(f"\n{'=' * 65}")
    print(f"Total experiments: {total}")

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

    b = best_overall
    print("\n=== Overall best ===")
    print(f"  {b['strategy']}_{b['algorithm']}: F1={b['f1']:.4f}")
    print(
        f"  Precision={b['precision']:.4f}  Recall={b['recall']:.4f}  Specificity={b['specificity']:.4f}"
    )
    print(f"  TP={b['tp']}  FP={b['fp']}  TN={b['tn']}  FN={b['fn']}")
    print(f"  Threshold={b['threshold']:.6f}")

    # Top-20 by f1_grid
    print("\n=== Top-20 (f1_grid) ===")
    ranked = sorted(
        [r for r in all_results if r["algorithm"] == "f1_grid"],
        key=lambda r: float(r["f1"]),
        reverse=True,
    )
    for i, r in enumerate(ranked[:20], 1):
        print(
            f"  {i:2d}. {r['formula']:35s} F1={float(r['f1']):.4f}  "
            f"P={float(r['precision']):.4f}  R={float(r['recall']):.4f}"
        )

    # Save best strategy
    best_path = LOG_DIR / "best_strategy.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "strategy": b["strategy"],
                "algorithm": b["algorithm"],
                "f1": b["f1"],
                "threshold": b["threshold"],
                "precision": b["precision"],
                "recall": b["recall"],
                "specificity": b["specificity"],
                "tp": b["tp"],
                "fp": b["fp"],
                "tn": b["tn"],
                "fn": b["fn"],
            },
            f,
            indent=2,
        )
    print(f"\nBest strategy saved to {best_path}")

    return all_results, best_overall


if __name__ == "__main__":
    run()
