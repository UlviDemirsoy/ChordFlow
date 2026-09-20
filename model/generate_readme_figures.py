"""Generate research-style figures for the README from local CSVs / run outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"

LEFT_NAMES = [
    "Unknown",
    "Deg 1",
    "Deg 2",
    "Deg 3",
    "Deg 4",
    "Deg 5",
    "Deg 6",
    "Deg 7",
]
RIGHT_NAMES = ["Unknown", "Major", "Sus4", "Dom7", "Minor", "Dim"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def plot_class_distribution(left: pd.DataFrame, right: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    panels = [
        (axes[0], left, LEFT_NAMES, "Left hand — scale degree", "#1f4e79"),
        (axes[1], right, RIGHT_NAMES, "Right hand — chord quality", "#2e7d4f"),
    ]
    for ax, frame, names, title, color in panels:
        counts = frame["label"].value_counts().sort_index()
        xs = list(range(len(names)))
        ys = [int(counts.get(i, 0)) for i in range(len(names))]
        bars = ax.bar(
            xs,
            ys,
            color=color,
            width=0.72,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Samples")
        ax.set_title(f"{title}\nN = {len(frame):,}")
        for bar, value in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(ys) * 0.015,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, max(ys) * 1.18)

    fig.suptitle("Dataset class distribution", fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "dataset_class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_split(split_csv: Path, names: list[str], out_name: str, title: str, colors: list[str]) -> None:
    split = pd.read_csv(split_csv)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    bottoms = [0] * len(names)
    for split_name, color in zip(("train", "validation", "test"), colors):
        part = split[split["split"] == split_name].set_index("label")["sample_count"]
        vals = [int(part.get(i, 0)) for i in range(len(names))]
        ax.bar(
            range(len(names)),
            vals,
            bottom=bottoms,
            label=split_name.capitalize(),
            color=color,
            width=0.72,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, vals)]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG_DIR / out_name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    left = pd.read_csv(ROOT / "model" / "data" / "left" / "landmarks.csv")
    right = pd.read_csv(ROOT / "model" / "data" / "right" / "landmarks.csv")
    plot_class_distribution(left, right)

    left_run = ROOT / "model" / "training_outputs" / "left_hand" / "run_20260729_193928"
    right_run = ROOT / "model" / "training_outputs" / "right_hand" / "run_20260729_205333"

    plot_split(
        left_run / "split_summary.csv",
        LEFT_NAMES,
        "left_split_distribution.png",
        "Left hand temporal block split (train / val / test)",
        ["#1f4e79", "#5b8db8", "#a8c5de"],
    )
    plot_split(
        right_run / "split_summary.csv",
        RIGHT_NAMES,
        "right_split_distribution.png",
        "Right hand temporal block split (train / val / test)",
        ["#2e7d4f", "#6aa882", "#b5d6c2"],
    )

    copies = [
        (left_run / "confusion_matrix.png", "left_confusion_matrix.png"),
        (left_run / "training_history.png", "left_training_history.png"),
        (left_run / "confidence_analysis.png", "left_confidence_analysis.png"),
        (right_run / "confusion_matrix.png", "right_confusion_matrix.png"),
        (right_run / "training_history.png", "right_training_history.png"),
        (right_run / "confidence_analysis.png", "right_confidence_analysis.png"),
    ]
    for source, destination in copies:
        shutil.copy2(source, FIG_DIR / destination)

    left_metrics = json.loads((left_run / "metrics.json").read_text(encoding="utf-8"))
    right_metrics = json.loads((right_run / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "left": {
            "n": int(len(left)),
            "mean_handedness": float(left["handedness_score"].mean()),
            "test_accuracy": left_metrics["test_accuracy"],
            "test_macro_f1": left_metrics["test_macro_f1"],
            "best_epoch": left_metrics["best_epoch"],
            "run": left_run.name,
        },
        "right": {
            "n": int(len(right)),
            "mean_handedness": float(right["handedness_score"].mean()),
            "test_accuracy": right_metrics["test_accuracy"],
            "test_macro_f1": right_metrics["test_macro_f1"],
            "best_epoch": right_metrics["best_epoch"],
            "run": right_run.name,
        },
    }
    (FIG_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Wrote figures to", FIG_DIR)
    for path in sorted(FIG_DIR.iterdir()):
        print(" ", path.name)


if __name__ == "__main__":
    main()
