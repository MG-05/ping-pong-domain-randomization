"""Generate evaluation comparison plots for the RL training report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent

LABELS = ["FSM\nBaseline", "Nominal SAC\n(final)", "Nominal SAC\n(best)", "Robust SAC\n(final)", "Robust SAC\n(best)"]
NOMINAL_HITS = [159.0, 35.0, 54.0, 20.0, 21.0]
RANDOM_HITS = [135.5, 33.9, 37.0, 22.6, 22.1]
RANDOM_STD = [36.4, 7.5, 7.9, 1.7, 2.1]
NOMINAL_STD = [0.0, 0.0, 0.0, 0.0, 0.0]

COLORS_NOM = "#4C72B0"
COLORS_RAND = "#DD8452"


def plot_hits_comparison():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(LABELS))
    width = 0.35

    bars1 = ax.bar(x - width / 2, NOMINAL_HITS, width, yerr=NOMINAL_STD,
                   label="Nominal Physics", color=COLORS_NOM, capsize=4, alpha=0.85)
    bars2 = ax.bar(x + width / 2, RANDOM_HITS, width, yerr=RANDOM_STD,
                   label="Randomized Physics (OOD)", color=COLORS_RAND, capsize=4, alpha=0.85)

    ax.set_ylabel("Mean Hits per Episode", fontsize=13)
    ax.set_title("Performance Comparison: Nominal vs Randomized Physics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(NOMINAL_HITS) * 1.15)

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hits_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'hits_comparison.png'}")


def plot_degradation():
    pct_change = [(r - n) / n * 100 if n > 0 else 0 for n, r in zip(NOMINAL_HITS, RANDOM_HITS)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#C44E52" if p < 0 else "#55A868" for p in pct_change]
    bars = ax.bar(LABELS, pct_change, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("% Change in Hits (Nominal → Randomized)", fontsize=12)
    ax.set_title("Robustness: Performance Change Under Physics Perturbation", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, pct_change):
        y_offset = 1.5 if val >= 0 else -3.5
        ax.annotate(f"{val:+.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, y_offset), textcoords="offset points", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "degradation_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'degradation_comparison.png'}")


def plot_variance():
    cv_values = [s / m * 100 if m > 0 else 0 for s, m in zip(RANDOM_STD, RANDOM_HITS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(LABELS, RANDOM_STD, color=["#8172B2", "#8172B2", "#8172B2", "#55A868", "#55A868"],
            alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Std Dev of Hits (under randomized physics)", fontsize=11)
    ax1.set_title("Absolute Variance", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(RANDOM_STD):
        ax1.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax2.bar(LABELS, cv_values, color=["#8172B2", "#8172B2", "#8172B2", "#55A868", "#55A868"],
            alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Coefficient of Variation (%)", fontsize=11)
    ax2.set_title("Relative Variance (lower = more consistent)", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(cv_values):
        ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Consistency Under Domain Randomization", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "variance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'variance_comparison.png'}")


def plot_per_episode_distribution():
    fsm_rand = [113, 85, 177, 174, 120, 152, 141, 65, 92, 168, 117, 84, 134, 120, 143, 115, 154, 198, 173, 185]
    nom_rand = [36, 28, 29, 28, 60, 38, 34, 27, 39, 28, 28, 38, 28, 34, 35, 26, 35, 41, 33, 32]
    rob_rand = [22, 21, 23, 23, 21, 23, 22, 22, 23, 22, 23, 22, 25, 22, 21, 22, 25, 28, 21, 22]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([fsm_rand, nom_rand, rob_rand],
                    tick_labels=["FSM Baseline", "Nominal SAC\n(final)", "Robust SAC\n(final)"],
                    patch_artist=True, widths=0.5)

    box_colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Hits per Episode", fontsize=13)
    ax.set_title("Hit Distribution Under Randomized Physics (20 episodes)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, data in enumerate([fsm_rand, nom_rand, rob_rand], 1):
        ax.scatter([i] * len(data), data, alpha=0.5, color="black", s=20, zorder=3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hit_distribution_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'hit_distribution_boxplot.png'}")


if __name__ == "__main__":
    plot_hits_comparison()
    plot_degradation()
    plot_variance()
    plot_per_episode_distribution()
    print("\nAll plots generated successfully.")
