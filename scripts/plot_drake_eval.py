"""Generate evaluation plots from a drake_eval_protocol summary.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

def _latest_eval_dir() -> Path:
    folders = sorted(
        (REPO_ROOT / "results").glob("drake_eval_protocol_*"),
        key=lambda p: p.name,
    )
    if not folders:
        raise FileNotFoundError("No drake_eval_protocol_* folders found in results/")
    return folders[-1]

EVAL_DIR = _latest_eval_dir()

C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_RED    = "#C44E52"
C_PURPLE = "#8172B2"


def _load(eval_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {model_label: {physics: condition_dict}}."""
    with open(eval_dir / "summary.json") as f:
        raw = json.load(f)["summary"]
    data: dict[str, dict[str, dict]] = {}
    for cond in raw:
        label = cond["model_label"]
        physics = cond["physics"]
        data.setdefault(label, {})[physics] = cond
    return data


def _ordered_labels(data: dict) -> list[str]:
    """Preserve insertion order from JSON (model order)."""
    seen: list[str] = []
    for label in data:
        if label not in seen:
            seen.append(label)
    return seen


def plot_hits_comparison(data: dict, labels: list[str], out_dir: Path) -> None:
    nom_hits  = [data[l]["nominal"]["mean_hits"]    for l in labels]
    rand_hits = [data[l]["randomized"]["mean_hits"] for l in labels]
    nom_std   = [data[l]["nominal"]["std_hits"]     for l in labels]
    rand_std  = [data[l]["randomized"]["std_hits"]  for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, nom_hits,  w, yerr=nom_std,  label="Nominal Physics",
                color=C_BLUE,   capsize=4, alpha=0.85)
    b2 = ax.bar(x + w/2, rand_hits, w, yerr=rand_std, label="Randomized Physics (OOD)",
                color=C_ORANGE, capsize=4, alpha=0.85)

    ax.set_ylabel("Mean Hits per Episode", fontsize=13)
    ax.set_title("Performance Comparison: Nominal vs Randomized Physics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    path = out_dir / "hits_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_degradation(data: dict, labels: list[str], out_dir: Path) -> None:
    pct = []
    for l in labels:
        n = data[l]["nominal"]["mean_hits"]
        r = data[l]["randomized"]["mean_hits"]
        pct.append((r - n) / n * 100 if n > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [C_RED if p < 0 else C_GREEN for p in pct]
    bars = ax.bar(labels, pct, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("% Change in Hits (Nominal → Randomized)", fontsize=12)
    ax.set_title("Robustness: Performance Degradation Under Physics Perturbation", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, pct):
        y_off = 2 if val >= 0 else -4
        ax.annotate(f"{val:+.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, y_off), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "degradation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_variance(data: dict, labels: list[str], out_dir: Path) -> None:
    rand_std  = [data[l]["randomized"]["std_hits"]  for l in labels]
    rand_hits = [data[l]["randomized"]["mean_hits"] for l in labels]
    cv = [s / m * 100 if m > 0 else 0 for s, m in zip(rand_std, rand_hits)]

    colors = [C_PURPLE, C_GREEN][:len(labels)]
    if len(labels) > 2:
        colors = [C_PURPLE] * len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(labels, rand_std, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Std Dev of Hits", fontsize=11)
    ax1.set_title("Absolute Variance (Randomized Physics)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(rand_std):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(labels, cv, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Coefficient of Variation (%)", fontsize=11)
    ax2.set_title("Relative Variance (lower = more consistent)", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(cv):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Consistency Under Domain Randomization", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "variance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_reward_comparison(data: dict, labels: list[str], out_dir: Path) -> None:
    nom_rew  = [data[l]["nominal"]["mean_reward"]    for l in labels]
    rand_rew = [data[l]["randomized"]["mean_reward"] for l in labels]
    rand_std = [data[l]["randomized"]["std_reward"]  for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, nom_rew,  w, label="Nominal Physics",           color=C_BLUE,   alpha=0.85)
    ax.bar(x + w/2, rand_rew, w, yerr=rand_std, label="Randomized Physics (OOD)",
           color=C_ORANGE, capsize=4, alpha=0.85)

    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Reward Comparison: Nominal vs Randomized Physics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "reward_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_survival_time(data: dict, labels: list[str], out_dir: Path) -> None:
    nom_t  = [data[l]["nominal"]["mean_sim_time"]    for l in labels]
    rand_t = [data[l]["randomized"]["mean_sim_time"] for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, nom_t,  w, label="Nominal Physics",          color=C_BLUE,   alpha=0.85)
    ax.bar(x + w/2, rand_t, w, label="Randomized Physics (OOD)", color=C_ORANGE, alpha=0.85)
    ax.axhline(y=20, color="gray", linestyle="--", linewidth=1, label="Max episode (20s)")

    ax.set_ylabel("Mean Simulation Time (s)", fontsize=13)
    ax.set_title("Episode Survival Time", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(labels)):
        for val, xpos in [(nom_t[i], x[i] - w/2), (rand_t[i], x[i] + w/2)]:
            ax.text(xpos, val + 0.3, f"{val:.1f}s", ha="center", fontsize=9)

    fig.tight_layout()
    path = out_dir / "survival_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_survival_rate(data: dict, labels: list[str], out_dir: Path) -> None:
    nom_sr  = [data[l]["nominal"]["survival_rate"]    * 100 for l in labels]
    rand_sr = [data[l]["randomized"]["survival_rate"] * 100 for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, nom_sr,  w, label="Nominal Physics",          color=C_BLUE,   alpha=0.85)
    ax.bar(x + w/2, rand_sr, w, label="Randomized Physics (OOD)", color=C_ORANGE, alpha=0.85)

    ax.set_ylabel("Survival Rate (%)", fontsize=13)
    ax.set_ylim(0, 115)
    ax.set_title("Survival Rate: Episodes Reaching Full Horizon (20s)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(labels)):
        for val, xpos in [(nom_sr[i], x[i] - w/2), (rand_sr[i], x[i] + w/2)]:
            ax.text(xpos, val + 1.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "survival_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    print(f"Loading results from: {EVAL_DIR}")
    data = _load(EVAL_DIR)
    labels = _ordered_labels(data)
    print(f"Models: {labels}")

    plot_hits_comparison(data, labels, EVAL_DIR)
    plot_degradation(data, labels, EVAL_DIR)
    plot_variance(data, labels, EVAL_DIR)
    plot_reward_comparison(data, labels, EVAL_DIR)
    plot_survival_time(data, labels, EVAL_DIR)
    plot_survival_rate(data, labels, EVAL_DIR)

    print("\nAll plots saved to:", EVAL_DIR)


if __name__ == "__main__":
    main()
