"""Generate EDA report charts for dataset collections."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = REPORT_DIR / "images"
EDA_OUTPUT = REPORT_DIR / "eda_data.json"


def _run_eda() -> dict:
    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "src.analysis.dataset_eda",
            "--format", "json",
        ],
        cwd=REPORT_DIR.parent.parent.parent,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"EDA failed: {result.stderr}")
    return json.loads(result.stdout)


def _bar_chart(
    data: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    top_n: int = 15,
    figsize: tuple = (10, 6),
) -> Path:
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, values = zip(*sorted_items) if sorted_items else ([], [])

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)

    path = IMAGES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _grouped_bar(
    curated: dict[str, int],
    incoming: dict[str, int],
    title: str,
    filename: str,
    top_n: int = 8,
    figsize: tuple = (8, 5),
) -> Path:
    shared = sorted(set(curated) & set(incoming))
    if len(shared) < 2:
        sorted_keys = sorted(
            set(curated) | set(incoming),
            key=lambda k: -(curated.get(k, 0) + incoming.get(k, 0))
        )[:top_n]
    else:
        sorted_keys = shared[:top_n]

    curated_vals = [curated.get(k, 0) for k in sorted_keys]
    incoming_vals = [incoming.get(k, 0) for k in sorted_keys]

    x = np.arange(len(sorted_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width / 2, curated_vals, width, label="curated_primary", color="#2ca02c")
    bars2 = ax.bar(x + width / 2, incoming_vals, width, label="incoming_low_quality", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_keys, rotation=45, ha="right")
    ax.set_ylabel("Images")
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    path = IMAGES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _overview_donut(
    curated_count: int,
    incoming_count: int,
    filename: str,
) -> Path:
    sizes = [curated_count, incoming_count]
    labels = [f"curated_primary\n({curated_count})", f"incoming_low_quality\n({incoming_count})"]
    colors = ["#2ca02c", "#d62728"]
    explode = (0.02, 0.02)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, explode=explode,
        startangle=140, pctdistance=0.6,
    )
    for t in autotexts:
        t.set_fontsize(12)
    ax.set_title("Source Collection Distribution")

    path = IMAGES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data = _run_eda()

    curated = next(c for c in data["collections"] if c["collection_key"] == "curated")
    incoming = next(c for c in data["collections"] if c["collection_key"] == "incoming")

    # Donut overview
    _overview_donut(curated["image_count"], incoming["image_count"], "collection_distribution.png")

    # Curated species
    _bar_chart(
        curated["species_image_counts"],
        "Species Distribution — curated_primary",
        "Images", "Species",
        "curated_species.png",
        top_n=8,
    )

    # Incoming species (top 20)
    _bar_chart(
        incoming["species_image_counts"],
        "Species Distribution — incoming_low_quality (top 20)",
        "Images", "Species",
        "incoming_species.png",
        top_n=20,
        figsize=(10, 8),
    )

    # Curated environments
    _bar_chart(
        curated["environment_image_counts"],
        "Environment Distribution — curated_primary",
        "Images", "Environment",
        "curated_environments.png",
    )

    # Strain counts
    _bar_chart(
        curated["strain_image_counts"],
        "Strain Distribution — curated_primary",
        "Images", "Strain",
        "curated_strains.png",
        top_n=31,
        figsize=(10, 10),
    )

    # Parse status
    _grouped_bar(
        curated["parse_status_counts"],
        incoming["parse_status_counts"],
        "Metadata Parse Status by Collection",
        "parse_status.png",
        top_n=3,
        figsize=(6, 5),
    )

    print(f"Charts saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
