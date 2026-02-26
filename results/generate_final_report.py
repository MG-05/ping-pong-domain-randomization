"""Generate comprehensive evaluation plots from all_eval_results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent

with open(RESULTS_DIR / "all_eval_results.json") as f:
    RAW = json.load(f)


def _lookup(label: str, physics: str) -> dict:
    for r in RAW:
        if r["label"] == label and r["physics"] == physics:
            return r
    raise KeyError(f"{label}/{physics} not found")


ORDERED_LABELS = [
    "FSM Baseline",
    "Nominal 500k", "Nominal 750k", "Nominal 1M",
    "Robust 500k", "Robust 750k", "Robust 1M",
]

NOM_HITS = [_lookup(l, "nominal")["mean_hits"] for l in ORDERED_LABELS]
RAND_HITS = [_lookup(l, "randomized")["mean_hits"] for l in ORDERED_LABELS]
RAND_STD = [_lookup(l, "randomized")["std_hits"] for l in ORDERED_LABELS]
NOM_STD = [_lookup(l, "nominal")["std_hits"] for l in ORDERED_LABELS]

SHORT_LABELS = ["FSM\nBaseline", "Nom\n500k", "Nom\n750k", "Nom\n1M",
                "Rob\n500k", "Rob\n750k", "Rob\n1M"]

C_BLUE = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN = "#55A868"
C_RED = "#C44E52"
C_PURPLE = "#8172B2"


def plot_hits_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(SHORT_LABELS))
    w = 0.35

    b1 = ax.bar(x - w/2, NOM_HITS, w, yerr=NOM_STD, label="Nominal Physics",
                color=C_BLUE, capsize=4, alpha=0.85)
    b2 = ax.bar(x + w/2, RAND_HITS, w, yerr=RAND_STD, label="Randomized Physics (OOD)",
                color=C_ORANGE, capsize=4, alpha=0.85)

    ax.set_ylabel("Mean Hits per Episode", fontsize=13)
    ax.set_title("Performance Comparison: All Models × Both Physics Conditions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 10:
                ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_hits_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: final_hits_comparison.png")


def plot_learning_curves():
    step_labels = ["500k", "750k", "1M"]
    nom_nom = [_lookup(f"Nominal {s}", "nominal")["mean_hits"] for s in step_labels]
    nom_rand = [_lookup(f"Nominal {s}", "randomized")["mean_hits"] for s in step_labels]
    rob_nom = [_lookup(f"Robust {s}", "nominal")["mean_hits"] for s in step_labels]
    rob_rand = [_lookup(f"Robust {s}", "randomized")["mean_hits"] for s in step_labels]

    nom_rand_std = [_lookup(f"Nominal {s}", "randomized")["std_hits"] for s in step_labels]
    rob_rand_std = [_lookup(f"Robust {s}", "randomized")["std_hits"] for s in step_labels]

    steps_k = [500, 750, 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Nominal physics
    ax1.plot(steps_k, nom_nom, "o-", color=C_BLUE, linewidth=2, markersize=8, label="Nominal SAC")
    ax1.plot(steps_k, rob_nom, "s-", color=C_GREEN, linewidth=2, markersize=8, label="Robust SAC")
    ax1.axhline(y=159, color="gray", linestyle="--", linewidth=1.5, label="FSM Baseline (159)")
    ax1.set_xlabel("Training Steps (×1000)", fontsize=12)
    ax1.set_ylabel("Mean Hits", fontsize=12)
    ax1.set_title("Evaluated on Nominal Physics", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(steps_k)
    for i, (nn, rn) in enumerate(zip(nom_nom, rob_nom)):
        ax1.annotate(f"{nn:.0f}", (steps_k[i], nn), textcoords="offset points",
                     xytext=(8, 5), fontsize=9, color=C_BLUE, fontweight="bold")
        ax1.annotate(f"{rn:.0f}", (steps_k[i], rn), textcoords="offset points",
                     xytext=(8, -12), fontsize=9, color=C_GREEN, fontweight="bold")

    # Randomized physics
    ax2.errorbar(steps_k, nom_rand, yerr=nom_rand_std, fmt="o-", color=C_BLUE,
                 linewidth=2, markersize=8, capsize=5, label="Nominal SAC")
    ax2.errorbar(steps_k, rob_rand, yerr=rob_rand_std, fmt="s-", color=C_GREEN,
                 linewidth=2, markersize=8, capsize=5, label="Robust SAC")
    ax2.axhline(y=145.3, color="gray", linestyle="--", linewidth=1.5, label="FSM Baseline (145)")
    ax2.set_xlabel("Training Steps (×1000)", fontsize=12)
    ax2.set_ylabel("Mean Hits", fontsize=12)
    ax2.set_title("Evaluated on Randomized Physics (OOD)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(steps_k)
    for i, (nr, rr) in enumerate(zip(nom_rand, rob_rand)):
        ax2.annotate(f"{nr:.0f}", (steps_k[i], nr), textcoords="offset points",
                     xytext=(8, 5), fontsize=9, color=C_BLUE, fontweight="bold")
        ax2.annotate(f"{rr:.0f}", (steps_k[i], rr), textcoords="offset points",
                     xytext=(8, -12), fontsize=9, color=C_GREEN, fontweight="bold")

    fig.suptitle("Learning Curves: Hits vs Training Budget", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: final_learning_curves.png")


def plot_degradation():
    pct = []
    for l in ORDERED_LABELS:
        n = _lookup(l, "nominal")["mean_hits"]
        r = _lookup(l, "randomized")["mean_hits"]
        pct.append((r - n) / n * 100 if n > 0 else 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [C_RED if p < 0 else C_GREEN for p in pct]
    bars = ax.bar(SHORT_LABELS, pct, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("% Change in Hits (Nominal → Randomized)", fontsize=12)
    ax.set_title("Robustness: Performance Degradation Under Physics Perturbation", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, pct):
        y_off = 2 if val >= 0 else -4
        ax.annotate(f"{val:+.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, y_off), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_degradation.png", dpi=150)
    plt.close(fig)
    print(f"Saved: final_degradation.png")


def plot_variance():
    cv = [s / m * 100 if m > 0 else 0 for s, m in zip(RAND_STD, RAND_HITS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    nom_colors = [C_PURPLE] * 4 + [C_GREEN] * 3
    ax1.bar(SHORT_LABELS, RAND_STD, color=nom_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Std Dev of Hits", fontsize=11)
    ax1.set_title("Absolute Variance (Randomized Physics)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(RAND_STD):
        ax1.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(SHORT_LABELS, cv, color=nom_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Coefficient of Variation (%)", fontsize=11)
    ax2.set_title("Relative Variance (lower = more consistent)", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(cv):
        ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Consistency Under Domain Randomization", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: final_variance.png")


def plot_reward_comparison():
    nom_rew = [_lookup(l, "nominal")["mean_reward"] for l in ORDERED_LABELS]
    rand_rew = [_lookup(l, "randomized")["mean_reward"] for l in ORDERED_LABELS]
    rand_rew_std = [_lookup(l, "randomized")["std_reward"] for l in ORDERED_LABELS]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(SHORT_LABELS))
    w = 0.35

    ax.bar(x - w/2, nom_rew, w, label="Nominal Physics", color=C_BLUE, alpha=0.85)
    ax.bar(x + w/2, rand_rew, w, yerr=rand_rew_std, label="Randomized Physics (OOD)",
           color=C_ORANGE, capsize=4, alpha=0.85)

    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("Reward Comparison: All Models × Both Physics Conditions", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_reward_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: final_reward_comparison.png")


def plot_survival_time():
    nom_t = [_lookup(l, "nominal")["mean_sim_time"] for l in ORDERED_LABELS]
    rand_t = [_lookup(l, "randomized")["mean_sim_time"] for l in ORDERED_LABELS]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(SHORT_LABELS))
    w = 0.35

    ax.bar(x - w/2, nom_t, w, label="Nominal Physics", color=C_BLUE, alpha=0.85)
    ax.bar(x + w/2, rand_t, w, label="Randomized Physics (OOD)", color=C_ORANGE, alpha=0.85)

    ax.axhline(y=20, color="gray", linestyle="--", linewidth=1, label="Max episode (20s)")
    ax.set_ylabel("Mean Simulation Time (s)", fontsize=13)
    ax.set_title("Episode Survival Time", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(SHORT_LABELS)):
        for val, xpos in [(nom_t[i], x[i] - w/2), (rand_t[i], x[i] + w/2)]:
            ax.text(xpos, val + 0.3, f"{val:.1f}s", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "final_survival_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved: final_survival_time.png")


if __name__ == "__main__":
    plot_hits_comparison()
    plot_learning_curves()
    plot_degradation()
    plot_variance()
    plot_reward_comparison()
    plot_survival_time()
    print("\nAll plots generated.")
