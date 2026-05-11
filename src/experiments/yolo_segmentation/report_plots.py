"""Generate training plots and charts for YOLOv26 finetune report."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from src.config import WORKSPACE_ROOT

RESULTS_DIR = WORKSPACE_ROOT / "results"
REPORT_DIR = Path(__file__).resolve().parent.parent / "report" / "yolo_segmentation" / "001"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = REPORT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["metrics/mAP50(B)"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="blue", alpha=0.7)
    ax1.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss", color="red", alpha=0.7)
    ax1.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="blue", linestyle="--", alpha=0.7)
    ax1.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss", color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("YOLOv26n Training & Validation Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # mAP curves
    ax2.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50", color="green", linewidth=2)
    ax2.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95", color="orange", linewidth=2)
    ax2.fill_between(df["epoch"], 0.4, 1.0, alpha=0.05, color="green")
    ax2.axhline(y=0.70, color="gray", linestyle=":", label="Target (70%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.set_title("YOLOv26n Detection Accuracy (mAP)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.02)

    plt.tight_layout()
    out = IMAGES_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return df


def plot_comparison_stats(summary_path: Path):
    summary = json.loads(summary_path.read_text())
    samples = summary["samples"]

    species_counts = {}
    for s in samples:
        sp = s["species"]
        if sp not in species_counts:
            species_counts[sp] = {"yolo26_sum": 0, "kmeans_sum": 0, "count": 0}
        species_counts[sp]["yolo26_sum"] += s["yolo26_count"]
        species_counts[sp]["kmeans_sum"] += s["kmeans_count"]
        species_counts[sp]["count"] += 1

    species = list(species_counts.keys())
    yolo_avg = [species_counts[s]["yolo26_sum"] / species_counts[s]["count"] for s in species]
    kmeans_avg = [species_counts[s]["kmeans_sum"] / species_counts[s]["count"] for s in species]

    x = np.arange(len(species))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w/2, yolo_avg, w, label="YOLOv26", color="#2ecc71", edgecolor="white")
    bars2 = ax.bar(x + w/2, kmeans_avg, w, label="KMeans", color="#3498db", edgecolor="white")
    ax.set_xlabel("Species")
    ax.set_ylabel("Avg Detections per Image")
    ax.set_title("YOLOv26 vs KMeans: Average Detections per Image by Species")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("penicillium-", "P. ").title() for s in species], rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 3.5)
    ax.axhline(y=3.0, color="gray", linestyle=":", alpha=0.5, label="Max possible (3)")
    ax.grid(True, alpha=0.2, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{bar.get_height():.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = IMAGES_DIR / "detection_comparison_by_species.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_best_map_progression(df: pd.DataFrame):
    """Plot the running best mAP@50 across epochs."""
    df = df.dropna(subset=["metrics/mAP50(B)"])
    best = np.maximum.accumulate(df["metrics/mAP50(B)"].to_numpy())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], alpha=0.4, color="gray", linewidth=0.8, label="Per-epoch mAP@50")
    ax.plot(df["epoch"], best, color="green", linewidth=2.5, label="Running Best mAP@50")
    ax.fill_between(df["epoch"], best, alpha=0.1, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@50")
    ax.set_title("YOLOv26n: mAP@50 Convergence (Best: {:.3f})".format(float(best[-1])))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.70, color="red", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = IMAGES_DIR / "best_map_progression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    csv_path = RESULTS_DIR / "yolo26_finetune" / "results.csv"
    summary_path = RESULTS_DIR / "yolo26_comparison" / "comparison_summary.json"

    print("Generating training curves...")
    df = plot_training_curves(csv_path)

    print("Generating best mAP progression...")
    plot_best_map_progression(df)

    if summary_path.exists():
        print("Generating comparison stats...")
        plot_comparison_stats(summary_path)

    # Print key metrics for report
    best_idx = df["metrics/mAP50(B)"].idxmax()
    print(f"\n=== Key Metrics ===")
    print(f"Best mAP@50: {df.loc[best_idx, 'metrics/mAP50(B)']:.4f} (epoch {int(df.loc[best_idx, 'epoch'])})")
    print(f"Best mAP@50-95: {df.loc[best_idx, 'metrics/mAP50-95(B)']:.4f}")
    print(f"Final mAP@50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
    print(f"Epochs completed: {len(df)}")
    print(f"Train images: 348 | Val images: 87")
    print(f"Model: YOLOv26n-seg (detection mode)")
    print(f"GPU: NVIDIA RTX 2060 12GB")
    print(f"Estimated training time: ~35 min (79 epochs, crashed at thread limit)")


if __name__ == "__main__":
    main()
