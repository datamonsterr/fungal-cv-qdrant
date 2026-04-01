"""
Cross-Validation Result Visualizations
=======================================
Reads ``report/week_1_2/cv_results.csv`` (produced by cross_validation.py)
and generates the following figures in ``report/week_1_2/images/``:

1. accuracy_vs_k.png          — line plot: mean accuracy vs K for all 4 setting combos
2. fold_variance.png           — box plot: accuracy distribution per fold, grouped by setting
3. heatmap_env_strategy_k.png — heatmap: (env × strategy) vs K, colour = mean accuracy
4. env_comparison.png          — bar chart: E1 vs E2 mean accuracy grouped by strategy

Also writes ``report/week_1_2/cv_summary_table.csv`` (always regenerated).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORT_DIR = Path(__file__).parent.parent.parent / "report" / "week_1_2"
CV_RESULTS_CSV = REPORT_DIR / "cv_results.csv"
CV_SUMMARY_CSV = REPORT_DIR / "cv_summary_table.csv"
IMAGES_DIR = REPORT_DIR / "images"

SETTING_ORDER = ["E1 / uni", "E1 / weighted", "E2 / uni", "E2 / weighted"]
SETTING_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
K_VALUES = [3, 5, 7, 9, 11]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setting_label(env: str, agg: str) -> str:
    return f"{env} / {agg}"


def _load_fold_accuracies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold accuracy per (fold, env_strategy, agg_strategy, k)."""
    return (
        df.groupby(["fold", "env_strategy", "agg_strategy", "k"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "fold_accuracy"})
    )


def _summary(fold_acc: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_acc.groupby(["env_strategy", "agg_strategy", "k"])["fold_accuracy"]
        .agg(
            mean_accuracy="mean",
            std_accuracy="std",
            min_accuracy="min",
            max_accuracy="max",
        )
        .reset_index()
    )
    summary["setting"] = summary.apply(
        lambda r: _setting_label(r["env_strategy"], r["agg_strategy"]), axis=1
    )
    return summary


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy vs K
# ---------------------------------------------------------------------------


def plot_accuracy_vs_k(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, setting in enumerate(SETTING_ORDER):
        sub = summary[summary["setting"] == setting].sort_values("k")
        if sub.empty:
            continue
        color = SETTING_COLORS[i % len(SETTING_COLORS)]
        ax.plot(
            sub["k"],
            sub["mean_accuracy"],
            marker="o",
            label=setting,
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            sub["k"],
            sub["mean_accuracy"] - sub["std_accuracy"].fillna(0),
            sub["mean_accuracy"] + sub["std_accuracy"].fillna(0),
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("K (number of neighbours)", fontsize=12)
    ax.set_ylabel("Mean accuracy (across folds)", fontsize=12)
    ax.set_title("Accuracy vs K — EfficientNetB1_finetuned", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.legend(title="env / strategy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Fold Variance Box Plot
# ---------------------------------------------------------------------------


def plot_fold_variance(fold_acc: pd.DataFrame, out: Path) -> None:
    """Box plot of fold accuracy for each K, one group per setting combination."""
    n_settings = len(SETTING_ORDER)
    n_k = len(K_VALUES)
    group_width = 0.8
    box_width = group_width / n_settings

    fig, ax = plt.subplots(figsize=(11, 5))
    x_positions = np.arange(n_k)

    for i, setting in enumerate(SETTING_ORDER):
        env, agg = setting.split(" / ")
        sub = fold_acc[
            (fold_acc["env_strategy"] == env) & (fold_acc["agg_strategy"] == agg)
        ]
        positions = []
        data_groups = []
        for ki, k in enumerate(K_VALUES):
            k_sub = sub[sub["k"] == k]["fold_accuracy"].values
            positions.append(x_positions[ki] + (i - n_settings / 2 + 0.5) * box_width)
            data_groups.append(k_sub)

        bp = ax.boxplot(
            data_groups,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        color = SETTING_COLORS[i % len(SETTING_COLORS)]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    # Dummy handles for legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=SETTING_COLORS[i], alpha=0.7, label=s)
        for i, s in enumerate(SETTING_ORDER)
    ]
    ax.legend(handles=legend_patches, title="env / strategy", fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"K={k}" for k in K_VALUES])
    ax.set_ylabel("Accuracy (per fold)", fontsize=12)
    ax.set_title(
        "Fold Accuracy Distribution by K — EfficientNetB1_finetuned", fontsize=13
    )
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(summary: pd.DataFrame, out: Path) -> None:
    pivot = summary.pivot_table(index="setting", columns="k", values="mean_accuracy")
    # Ensure row order
    ordered_rows = [s for s in SETTING_ORDER if s in pivot.index]
    pivot = pivot.loc[ordered_rows]
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Mean accuracy")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"K={k}" for k in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for row_i in range(len(pivot.index)):
        for col_i in range(len(pivot.columns)):
            val = pivot.values[row_i, col_i]
            if not np.isnan(val):
                text_color = "black" if 0.35 < val < 0.85 else "white"
                ax.text(
                    col_i,
                    row_i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                )

    ax.set_title("Mean Accuracy: (env × strategy) vs K", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — E1 vs E2 Bar Chart
# ---------------------------------------------------------------------------


def plot_env_comparison(summary: pd.DataFrame, out: Path) -> None:
    # Average over K for each (env, agg)
    env_agg = (
        summary.groupby(["env_strategy", "agg_strategy"])["mean_accuracy"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    strategies = env_agg["agg_strategy"].unique()
    x = np.arange(len(strategies))
    n_envs = 2
    bar_w = 0.35

    env_colors = {"E1": "#1f77b4", "E2": "#2ca02c"}
    for ei, env in enumerate(["E1", "E2"]):
        sub = env_agg[env_agg["env_strategy"] == env].set_index("agg_strategy")
        vals = [
            sub.loc[s, "mean_accuracy"] if s in sub.index else 0 for s in strategies
        ]
        offset = (ei - n_envs / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w, label=env, color=env_colors[env], alpha=0.8
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel("Mean accuracy (avg over K & folds)", fontsize=11)
    ax.set_title("E1 vs E2 — Mean Accuracy by Aggregation Strategy", fontsize=12)
    ax.legend(title="Env strategy")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_visualizations(
    results_csv: Path = CV_RESULTS_CSV,
    output_dir: Path = IMAGES_DIR,
) -> None:
    if not results_csv.exists():
        print(f"Results CSV not found: {results_csv}")
        print("Run 'uv run python -m src.experiments.cross_validation.run' first.")
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        print("Results CSV is empty — no visualizations to generate.")
        return

    df["correct"] = df["correct"].astype(float)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_acc = _load_fold_accuracies(df)
    summary = _summary(fold_acc)

    # Write summary CSV
    summary_out = results_csv.parent / "cv_summary_table.csv"
    summary_sorted = summary.sort_values("mean_accuracy", ascending=False)
    summary_sorted.to_csv(summary_out, index=False)
    print(f"  Summary table: {summary_out}")

    print("\nGenerating visualizations...")
    plot_accuracy_vs_k(summary, output_dir / "accuracy_vs_k.png")
    plot_fold_variance(fold_acc, output_dir / "fold_variance.png")
    plot_heatmap(summary, output_dir / "heatmap_env_strategy_k.png")
    plot_env_comparison(summary, output_dir / "env_comparison.png")
    print("\nAll visualizations saved to:", output_dir)


def main() -> None:
    run_visualizations()


if __name__ == "__main__":
    main()
