"""
Generate paper-quality plots from a mujoco_eval_protocol run.

Usage:
  python -m mujoco_transfer.plot_mujoco_eval                  # auto-latest
  python -m mujoco_transfer.plot_mujoco_eval --dir results/mujoco_eval_protocol_20260312_123456
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Color palette (consistent with scripts/plot_drake_eval.py) ────────────────
C_GRAY   = "#888888"
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_RED    = "#C44E52"
C_PURPLE = "#8172B2"

CONDITION_ORDER = [
    "FSM Baseline (Nominal)",
    "FSM Baseline (Randomized)",
    "Nominal (Nominal Physics)",
    "Nominal (Randomized Physics)",
    "Robust (Nominal Physics)",
    "Robust (Randomized Physics)",
]

SHORT_LABELS = {
    "FSM Baseline (Nominal)":        "FSM\n(Nominal)",
    "FSM Baseline (Randomized)":     "FSM\n(Randomized)",
    "Nominal (Nominal Physics)":     "Nominal\n(Nominal)",
    "Nominal (Randomized Physics)":  "Nominal\n(Randomized)",
    "Robust (Nominal Physics)":      "Robust\n(Nominal)",
    "Robust (Randomized Physics)":   "Robust\n(Randomized)",
}

CONDITION_COLORS = {
    "FSM Baseline (Nominal)":        C_GRAY,
    "FSM Baseline (Randomized)":     C_GRAY,
    "Nominal (Nominal Physics)":     C_BLUE,
    "Nominal (Randomized Physics)":  C_BLUE,
    "Robust (Nominal Physics)":      C_ORANGE,
    "Robust (Randomized Physics)":   C_ORANGE,
}

CONDITION_HATCHES = {
    "FSM Baseline (Nominal)":        "",
    "FSM Baseline (Randomized)":     "///",
    "Nominal (Nominal Physics)":     "",
    "Nominal (Randomized Physics)":  "///",
    "Robust (Nominal Physics)":      "",
    "Robust (Randomized Physics)":   "///",
}

# ── Hardcoded sweep data (from sweep_hyperparams.py output, 15 eps each) ─────
# Shape: rows = scales, cols = dts
SWEEP_SCALES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
SWEEP_DTS    = [0.005, 0.010, 0.020]

SWEEP_NOMINAL_HITS = np.array([
    [20.93, 33.93,  8.80],  # scale=0.005
    [25.60, 23.07,  8.87],  # scale=0.01
    [14.93, 20.67,  6.47],  # scale=0.02
    [ 7.80,  6.33,  5.40],  # scale=0.05
    [ 6.47,  6.67,  5.73],  # scale=0.10
    [ 3.07,  4.40,  5.47],  # scale=0.20
])

SWEEP_ROBUST_HITS = np.array([
    [ 9.27, 10.73,  4.73],  # scale=0.005
    [14.87, 11.93,  3.73],  # scale=0.01
    [10.33,  6.60,  3.47],  # scale=0.02
    [ 2.87,  3.07,  2.73],  # scale=0.05
    [ 2.93,  3.73,  3.13],  # scale=0.10
    [ 3.27,  3.13,  4.47],  # scale=0.20
])

# Optimal cells (scale_idx, dt_idx)
NOMINAL_BEST = (0, 1)   # scale=0.005, dt=0.01 → 33.93
ROBUST_BEST  = (1, 0)   # scale=0.01,  dt=0.005 → 14.87

FSM_NOMINAL_HITS = 63.0    # reference ceiling: FSM nominal physics


# ── Data loading ──────────────────────────────────────────────────────────────

def _latest_eval_dir() -> Path:
    folders = sorted(
        (REPO_ROOT / "results").glob("mujoco_eval_protocol_*"),
        key=lambda p: p.name,
    )
    if not folders:
        raise FileNotFoundError("No mujoco_eval_protocol_* folders found in results/")
    return folders[-1]


def load_per_episode(eval_dir: Path) -> dict[str, list[dict]]:
    """Return {condition: [row_dict, ...]}."""
    data: dict[str, list[dict]] = {c: [] for c in CONDITION_ORDER}
    with open(eval_dir / "per_episode.csv") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            if cond in data:
                data[cond].append({
                    "hits":     int(row["hits"]),
                    "reward":   float(row["reward"]),
                    "sim_time": float(row["sim_time"]),
                    "survived": row["survived"].lower() == "true",
                })
    return data


def load_summary(eval_dir: Path) -> dict[str, dict]:
    with open(eval_dir / "summary.json") as f:
        raw = json.load(f)["summary"]
    return {s["condition"]: s for s in raw}


# ── Annotation helper ─────────────────────────────────────────────────────────

def _annotate_bars(ax, bars, fmt="{:.1f}", offset=3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", fontsize=8,
        )


# ── Plot 1: Hits comparison bar chart ────────────────────────────────────────

def plot_hits_comparison(summary: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(CONDITION_ORDER))
    w = 0.6
    means = [summary[c]["mean_hits"] for c in CONDITION_ORDER]
    stds  = [summary[c]["std_hits"]  for c in CONDITION_ORDER]
    colors  = [CONDITION_COLORS[c]  for c in CONDITION_ORDER]
    hatches = [CONDITION_HATCHES[c] for c in CONDITION_ORDER]

    for i, (m, s, col, hatch) in enumerate(zip(means, stds, colors, hatches)):
        bar = ax.bar(x[i], m, w, yerr=s, color=col, hatch=hatch,
                     capsize=5, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.annotate(f"{m:.1f}", xy=(x[i], m), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(FSM_NOMINAL_HITS, color=C_GRAY, linestyle="--", linewidth=1.2,
               label=f"FSM nominal ceiling ({FSM_NOMINAL_HITS:.0f} hits)")

    solid_patch  = mpatches.Patch(color=C_GRAY,   label="FSM Baseline")
    blue_patch   = mpatches.Patch(color=C_BLUE,   label="Nominal SAC")
    orange_patch = mpatches.Patch(color=C_ORANGE, label="Robust SAC")
    hatch_patch  = mpatches.Patch(facecolor="white", edgecolor="black",
                                  hatch="///", label="Randomized Physics")
    ax.legend(handles=[solid_patch, blue_patch, orange_patch, hatch_patch], fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Mean Hits per Episode", fontsize=12)
    ax.set_title("MuJoCo Transfer: Hits per Episode Across All Conditions", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "hits_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: Hit distribution violin plot ─────────────────────────────────────

def plot_hit_distribution(per_ep: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    data_list = [[r["hits"] for r in per_ep[c]] for c in CONDITION_ORDER]
    colors    = [CONDITION_COLORS[c] for c in CONDITION_ORDER]

    parts = ax.violinplot(data_list, positions=range(len(CONDITION_ORDER)),
                          showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cmins"].set_color("gray")
    parts["cmaxes"].set_color("gray")
    parts["cbars"].set_color("gray")

    rng = np.random.default_rng(0)
    for i, (cond, col) in enumerate(zip(CONDITION_ORDER, colors)):
        hits = [r["hits"] for r in per_ep[cond]]
        jitter = rng.uniform(-0.08, 0.08, len(hits))
        ax.scatter(np.full(len(hits), i) + jitter, hits,
                   color=col, alpha=0.4, s=12, zorder=3)

    ax.axhline(FSM_NOMINAL_HITS, color=C_GRAY, linestyle="--", linewidth=1.2,
               label=f"FSM nominal ceiling ({FSM_NOMINAL_HITS:.0f} hits)")

    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([SHORT_LABELS[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Hits per Episode", fontsize=12)
    ax.set_title("Hit Distribution per Condition (Violin + Individual Episodes)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "hit_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 3: Degradation bar chart ─────────────────────────────────────────────

def plot_degradation(summary: dict, out_dir: Path) -> None:
    models = [
        ("FSM",         "FSM Baseline (Nominal)",    "FSM Baseline (Randomized)"),
        ("Nominal SAC", "Nominal (Nominal Physics)", "Nominal (Randomized Physics)"),
        ("Robust SAC",  "Robust (Nominal Physics)",  "Robust (Randomized Physics)"),
    ]
    labels, pcts, nom_hits, rand_hits = [], [], [], []
    for label, nom_key, rand_key in models:
        n = summary[nom_key]["mean_hits"]
        r = summary[rand_key]["mean_hits"]
        pct = (r - n) / n * 100 if n > 0 else 0.0
        labels.append(label)
        pcts.append(pct)
        nom_hits.append(n)
        rand_hits.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.5
    colors = [C_RED if p < 0 else C_GREEN for p in pcts]
    bars = ax.bar(x, pcts, w, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)

    for bar, pct, n, r in zip(bars, pcts, nom_hits, rand_hits):
        y_off = -18 if pct < 0 else 3
        ax.annotate(
            f"{pct:+.1f}%\n({n:.1f} → {r:.1f} hits)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, y_off), textcoords="offset points",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("% Change in Hits (Nominal → Randomized Physics)", fontsize=11)
    ax.set_title("Robustness to Domain Randomization", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "degradation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 4: Survival time ─────────────────────────────────────────────────────

def plot_survival_time(summary: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(CONDITION_ORDER))
    w = 0.6
    times   = [summary[c]["mean_sim_time"] for c in CONDITION_ORDER]
    colors  = [CONDITION_COLORS[c]  for c in CONDITION_ORDER]
    hatches = [CONDITION_HATCHES[c] for c in CONDITION_ORDER]

    for i, (t, col, hatch) in enumerate(zip(times, colors, hatches)):
        ax.bar(x[i], t, w, color=col, hatch=hatch,
               alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.text(x[i], t + 0.3, f"{t:.1f}s", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(20.0, color="gray", linestyle="--", linewidth=1.2, label="Max episode (20s)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Mean Simulation Time (s)", fontsize=12)
    ax.set_ylim(0, 23)
    ax.set_title("Episode Survival Time per Condition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "survival_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 5: Hyperparameter sweep heatmap ─────────────────────────────────────

def plot_sweep_heatmap(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, best_cell in [
        (axes[0], SWEEP_NOMINAL_HITS, "Nominal SAC — Mean Hits", NOMINAL_BEST),
        (axes[1], SWEEP_ROBUST_HITS,  "Robust SAC — Mean Hits",  ROBUST_BEST),
    ]:
        vmax = max(SWEEP_NOMINAL_HITS.max(), SWEEP_ROBUST_HITS.max())
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

        ax.set_xticks(range(len(SWEEP_DTS)))
        ax.set_xticklabels([f"{dt:.3f}s" for dt in SWEEP_DTS], fontsize=10)
        ax.set_yticks(range(len(SWEEP_SCALES)))
        ax.set_yticklabels([f"{s:.3f}" for s in SWEEP_SCALES], fontsize=10)
        ax.set_xlabel("RL Control dt", fontsize=11)
        ax.set_ylabel("Residual Scale", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(len(SWEEP_SCALES)):
            for j in range(len(SWEEP_DTS)):
                ax.text(j, i, f"{data[i, j]:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="black" if data[i, j] > vmax * 0.6 else "black")

        si, di = best_cell
        ax.add_patch(plt.Rectangle((di - 0.5, si - 0.5), 1, 1,
                                   fill=False, edgecolor="cyan", linewidth=3))
        ax.text(di, si - 0.5, "★", ha="center", va="bottom",
                fontsize=14, color="cyan", fontweight="bold")

        fig.colorbar(im, ax=ax, label="Mean Hits")

    fig.suptitle("Hyperparameter Sweep: Residual Scale × RL Control dt (15 eps/cell)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "sweep_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 6: Reward comparison ─────────────────────────────────────────────────

def plot_reward_comparison(summary: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(CONDITION_ORDER))
    w = 0.6
    means   = [summary[c]["mean_reward"] for c in CONDITION_ORDER]
    stds    = [summary[c]["std_reward"]  for c in CONDITION_ORDER]
    colors  = [CONDITION_COLORS[c]  for c in CONDITION_ORDER]
    hatches = [CONDITION_HATCHES[c] for c in CONDITION_ORDER]

    for i, (m, s, col, hatch) in enumerate(zip(means, stds, colors, hatches)):
        ax.bar(x[i], m, w, yerr=s, color=col, hatch=hatch,
               capsize=5, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.annotate(f"{m:.0f}", xy=(x[i], m), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Episode Reward per Condition", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    blue_patch   = mpatches.Patch(color=C_BLUE,   label="Nominal SAC")
    orange_patch = mpatches.Patch(color=C_ORANGE, label="Robust SAC")
    gray_patch   = mpatches.Patch(color=C_GRAY,   label="FSM Baseline")
    hatch_patch  = mpatches.Patch(facecolor="white", edgecolor="black",
                                  hatch="///", label="Randomized Physics")
    ax.legend(handles=[gray_patch, blue_patch, orange_patch, hatch_patch], fontsize=10)
    fig.tight_layout()
    path = out_dir / "reward_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to mujoco_eval_protocol directory (default: latest)")
    args = parser.parse_args()

    eval_dir = Path(args.dir) if args.dir else _latest_eval_dir()
    print(f"Loading results from: {eval_dir}\n")

    per_ep  = load_per_episode(eval_dir)
    summary = load_summary(eval_dir)

    # Verify expected conditions are present
    for cond in CONDITION_ORDER:
        if cond not in summary:
            raise ValueError(f"Missing condition in summary.json: '{cond}'")
        n = len(per_ep.get(cond, []))
        print(f"  {cond}: {n} episodes, mean_hits={summary[cond]['mean_hits']:.1f}")
    print()

    plot_hits_comparison(summary, eval_dir)
    plot_hit_distribution(per_ep, eval_dir)
    plot_degradation(summary, eval_dir)
    plot_survival_time(summary, eval_dir)
    plot_sweep_heatmap(eval_dir)
    plot_reward_comparison(summary, eval_dir)

    print(f"\nAll 6 plots saved to: {eval_dir}")


if __name__ == "__main__":
    main()
