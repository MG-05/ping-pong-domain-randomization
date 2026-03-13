"""
Convergence analysis: subsample first N episodes from the 500-episode run
at N in [10, 50, 100, 200, 500] and show how mean_hits and degradation
stabilize as episode count grows.

Usage:
  python -m mujoco_transfer.plot_convergence                  # auto-latest 500-ep run
  python -m mujoco_transfer.plot_convergence --dir results/mujoco_eval_protocol_XYZ
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITION_ORDER = [
    "FSM Baseline (Nominal)",
    "FSM Baseline (Randomized)",
    "Nominal (Nominal Physics)",
    "Nominal (Randomized Physics)",
    "Robust (Nominal Physics)",
    "Robust (Randomized Physics)",
]

SHORT_LABELS = {
    "FSM Baseline (Nominal)":       "FSM (Nominal)",
    "FSM Baseline (Randomized)":    "FSM (Randomized)",
    "Nominal (Nominal Physics)":    "Nominal (Nominal)",
    "Nominal (Randomized Physics)": "Nominal (Randomized)",
    "Robust (Nominal Physics)":     "Robust (Nominal)",
    "Robust (Randomized Physics)":  "Robust (Randomized)",
}

C_GRAY   = "#888888"
C_GRAY_L = "#BBBBBB"
C_BLUE   = "#4C72B0"
C_BLUE_L = "#A8C4E0"
C_ORANGE = "#DD8452"
C_ORANGE_L = "#F0C89A"
C_GREEN  = "#55A868"
C_RED    = "#C44E52"

CONDITION_COLORS = {
    "FSM Baseline (Nominal)":       C_GRAY,
    "FSM Baseline (Randomized)":    C_GRAY_L,
    "Nominal (Nominal Physics)":    C_BLUE,
    "Nominal (Randomized Physics)": C_BLUE_L,
    "Robust (Nominal Physics)":     C_ORANGE,
    "Robust (Randomized Physics)":  C_ORANGE_L,
}

CONDITION_LINESTYLES = {
    "FSM Baseline (Nominal)":       "-",
    "FSM Baseline (Randomized)":    "--",
    "Nominal (Nominal Physics)":    "-",
    "Nominal (Randomized Physics)": "--",
    "Robust (Nominal Physics)":     "-",
    "Robust (Randomized Physics)":  "--",
}

SUBSAMPLE_NS = [10, 50, 100, 200, 500]


def _latest_eval_dir() -> Path:
    folders = sorted(
        (REPO_ROOT / "results").glob("mujoco_eval_protocol_*"),
        key=lambda p: p.name,
    )
    if not folders:
        raise FileNotFoundError("No mujoco_eval_protocol_* folders found in results/")
    return folders[-1]


def load_per_episode(eval_dir: Path) -> dict[str, list[int]]:
    """Return {condition: [hits_ep0, hits_ep1, ...]} in episode order."""
    data: dict[str, list] = {c: [] for c in CONDITION_ORDER}
    rows_by_cond: dict[str, list] = {c: [] for c in CONDITION_ORDER}
    with open(eval_dir / "per_episode.csv") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            if cond in rows_by_cond:
                rows_by_cond[cond].append({
                    "episode": int(row["episode"]),
                    "hits":    int(row["hits"]),
                    "reward":  float(row["reward"]),
                    "sim_time": float(row["sim_time"]),
                })
    # Sort by episode number and extract hits
    for cond in CONDITION_ORDER:
        sorted_rows = sorted(rows_by_cond[cond], key=lambda r: r["episode"])
        data[cond] = sorted_rows
    return data


def compute_stats_at_n(rows: list[dict], n: int) -> dict:
    """Compute mean/std/SE/CI for the first n episodes."""
    subset = rows[:n]
    hits = [r["hits"] for r in subset]
    rewards = [r["reward"] for r in subset]
    times = [r["sim_time"] for r in subset]
    mean_h = float(np.mean(hits))
    std_h  = float(np.std(hits))
    se_h   = std_h / np.sqrt(n)
    return {
        "n": n,
        "mean_hits":   mean_h,
        "std_hits":    std_h,
        "se_hits":     se_h,
        "ci95_hits":   1.96 * se_h,
        "mean_reward": float(np.mean(rewards)),
        "mean_time":   float(np.mean(times)),
        "survival":    float(np.mean([r["sim_time"] >= 19.9 for r in subset])),
    }


# ── Plot 1: Mean hits convergence (line plot) ─────────────────────────────────

def plot_hits_convergence(per_ep: dict, ns: list[int], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond in CONDITION_ORDER:
        rows = per_ep[cond]
        means, ci95s = [], []
        for n in ns:
            if n > len(rows):
                continue
            s = compute_stats_at_n(rows, n)
            means.append(s["mean_hits"])
            ci95s.append(s["ci95_hits"])

        valid_ns = [n for n in ns if n <= len(rows)]
        means = np.array(means)
        ci95s = np.array(ci95s)
        col = CONDITION_COLORS[cond]
        ls  = CONDITION_LINESTYLES[cond]

        ax.plot(valid_ns, means, color=col, linestyle=ls, linewidth=2,
                marker="o", markersize=6, label=SHORT_LABELS[cond])
        ax.fill_between(valid_ns, means - ci95s, means + ci95s,
                        color=col, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=11)
    ax.set_xlabel("Number of Episodes", fontsize=12)
    ax.set_ylabel("Mean Hits per Episode", fontsize=12)
    ax.set_title("Convergence of Mean Hits vs Episode Count\n(shaded = 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "convergence_hits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: Std / CI convergence (shows when estimates stabilize) ─────────────

def plot_ci_convergence(per_ep: dict, ns: list[int], out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cond in CONDITION_ORDER:
        if cond == "FSM Baseline":
            continue  # std=0, uninteresting
        rows = per_ep[cond]
        stds, ci95s = [], []
        valid_ns = []
        for n in ns:
            if n > len(rows):
                continue
            s = compute_stats_at_n(rows, n)
            stds.append(s["std_hits"])
            ci95s.append(s["ci95_hits"])
            valid_ns.append(n)

        col = CONDITION_COLORS[cond]
        ls  = CONDITION_LINESTYLES[cond]
        ax1.plot(valid_ns, stds,  color=col, linestyle=ls, linewidth=2,
                 marker="o", markersize=5, label=SHORT_LABELS[cond])
        ax2.plot(valid_ns, ci95s, color=col, linestyle=ls, linewidth=2,
                 marker="o", markersize=5, label=SHORT_LABELS[cond])

    for ax, ylabel, title in [
        (ax1, "Std Dev of Hits", "Standard Deviation vs Episode Count"),
        (ax2, "95% CI Half-Width (hits)", "95% CI Width vs Episode Count"),
    ]:
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=10)
        ax.set_xlabel("Number of Episodes", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Statistical Precision vs Sample Size", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "convergence_ci.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 3: Degradation convergence ───────────────────────────────────────────

def plot_degradation_convergence(per_ep: dict, ns: list[int], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    pairs = [
        ("Nominal SAC",  "Nominal (Nominal Physics)",  "Nominal (Randomized Physics)",  C_BLUE,   "-"),
        ("Robust SAC",   "Robust (Nominal Physics)",   "Robust (Randomized Physics)",   C_ORANGE, "-"),
    ]

    for label, nom_key, rand_key, col, ls in pairs:
        nom_rows  = per_ep[nom_key]
        rand_rows = per_ep[rand_key]
        pcts, ci95s = [], []
        valid_ns = []
        for n in ns:
            if n > min(len(nom_rows), len(rand_rows)):
                continue
            s_nom  = compute_stats_at_n(nom_rows,  n)
            s_rand = compute_stats_at_n(rand_rows, n)
            n_mean = s_nom["mean_hits"]
            r_mean = s_rand["mean_hits"]
            pct = (r_mean - n_mean) / n_mean * 100 if n_mean > 0 else 0.0
            # Propagate uncertainty via delta method (first-order approx)
            se_pct = np.sqrt(
                (s_rand["se_hits"] / n_mean) ** 2 +
                (r_mean * s_nom["se_hits"] / n_mean ** 2) ** 2
            ) * 100
            pcts.append(pct)
            ci95s.append(1.96 * se_pct)
            valid_ns.append(n)

        pcts  = np.array(pcts)
        ci95s = np.array(ci95s)
        ax.plot(valid_ns, pcts, color=col, linestyle=ls, linewidth=2,
                marker="o", markersize=6, label=label)
        ax.fill_between(valid_ns, pcts - ci95s, pcts + ci95s,
                        color=col, alpha=0.15)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=11)
    ax.set_xlabel("Number of Episodes", fontsize=12)
    ax.set_ylabel("% Change in Hits (Nominal → Randomized)", fontsize=12)
    ax.set_title("Degradation Under Domain Randomization vs Episode Count\n(shaded = propagated 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "convergence_degradation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 4: Summary table heatmap ─────────────────────────────────────────────

def plot_summary_table(per_ep: dict, ns: list[int], out_dir: Path) -> None:
    """Heatmap of mean_hits per (condition, n_episodes)."""
    conditions_short = [SHORT_LABELS[c] for c in CONDITION_ORDER]
    data = np.zeros((len(CONDITION_ORDER), len(ns)))

    for i, cond in enumerate(CONDITION_ORDER):
        rows = per_ep[cond]
        for j, n in enumerate(ns):
            if n <= len(rows):
                data[i, j] = compute_stats_at_n(rows, n)["mean_hits"]
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=70)

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns], fontsize=11)
    ax.set_yticks(range(len(CONDITION_ORDER)))
    ax.set_yticklabels(conditions_short, fontsize=10)
    ax.set_xlabel("Number of Episodes", fontsize=12)
    ax.set_title("Mean Hits: Convergence Table (Nominal → Randomized per row)",
                 fontsize=12, fontweight="bold")

    for i in range(len(CONDITION_ORDER)):
        for j in range(len(ns)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > 40 else "black")

    fig.colorbar(im, ax=ax, label="Mean Hits")
    fig.tight_layout()
    path = out_dir / "convergence_table.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    eval_dir = Path(args.dir) if args.dir else _latest_eval_dir()
    print(f"Loading results from: {eval_dir}\n")

    per_ep = load_per_episode(eval_dir)

    # Determine valid Ns based on actual episode counts
    max_eps = min(len(per_ep[c]) for c in CONDITION_ORDER)
    valid_ns = [n for n in SUBSAMPLE_NS if n <= max_eps]
    print(f"Max episodes available: {max_eps}")
    print(f"Subsampling at N = {valid_ns}\n")

    # Print convergence table to stdout
    print(f"{'Condition':<35} " + "  ".join(f"N={n:>4}" for n in valid_ns))
    print("-" * (35 + 9 * len(valid_ns)))
    for cond in CONDITION_ORDER:
        row_str = f"{SHORT_LABELS[cond]:<35}"
        for n in valid_ns:
            if n <= len(per_ep[cond]):
                s = compute_stats_at_n(per_ep[cond], n)
                row_str += f"  {s['mean_hits']:>6.1f}"
            else:
                row_str += f"  {'N/A':>6}"
        print(row_str)
    print()

    plot_hits_convergence(per_ep, valid_ns, eval_dir)
    plot_ci_convergence(per_ep, valid_ns, eval_dir)
    plot_degradation_convergence(per_ep, valid_ns, eval_dir)
    plot_summary_table(per_ep, valid_ns, eval_dir)

    print(f"\nAll convergence plots saved to: {eval_dir}")


if __name__ == "__main__":
    main()
