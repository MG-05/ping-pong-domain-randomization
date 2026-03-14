"""
Generate nominal-only MuJoCo domain-validation figures.

This script intentionally ignores randomized MuJoCo conditions and focuses on
using MuJoCo as a single validation domain.

Usage:
  python -m mujoco_transfer.plot_domain_validation_figures
  python -m mujoco_transfer.plot_domain_validation_figures \
      --dir results/mujoco_eval_protocol_20260313_123354_rs001_diag \
      --report-name drake_to_mujoco_nominal_validation \
      --sweep-episodes 20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mujoco_transfer.residual_env_mujoco import MujocoResidualConfig, MujocoResidualEnv
from mujoco_transfer.sb3_compat import (
    install_numpy_pickle_compat_shims,
    make_legacy_custom_objects,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_XML_DEFAULT = str(REPO_ROOT / "mujoco_transfer/models/iiwa_wsg_paddle_ball.xml")

COND_FSM = "FSM Baseline (Nominal)"
COND_NOMINAL = "Nominal (Nominal Physics)"
COND_ROBUST = "Robust (Nominal Physics)"
NOMINAL_CONDITIONS = [COND_FSM, COND_NOMINAL, COND_ROBUST]

DISPLAY_LABELS = {
    COND_FSM: "FSM Baseline",
    COND_NOMINAL: "Nominal SAC",
    COND_ROBUST: "Robust SAC",
}

# Color psychology: neutral for stability, blue for control, green-teal for robust adaptation.
PALETTE = {
    "bg": "#F6F4EE",
    "fg": "#1F2933",
    "grid": "#D5DDE5",
    "fsm": "#5B6777",
    "nominal": "#2D6FAE",
    "robust": "#21857A",
    "accent": "#D4A33C",
    "warning": "#B03A2E",
}

MODEL_COLORS = {
    "FSM Baseline": PALETTE["fsm"],
    "Nominal SAC": PALETTE["nominal"],
    "Robust SAC": PALETTE["robust"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nominal-only MuJoCo domain-validation figures."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to mujoco_eval_protocol_<timestamp> folder (default: latest).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="results/domain_validation_reports",
        help="Root output directory for all domain-validation reports.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Report folder name under --out-root (default: eval folder name).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="figures",
        help="Figures subdirectory inside the report folder.",
    )
    parser.add_argument(
        "--run-residual-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run residual-scale sweep figure generation (default: on).",
    )
    parser.add_argument(
        "--sweep-episodes",
        type=int,
        default=5,
        help="Episodes per scale for residual sweep (default: 5).",
    )
    parser.add_argument(
        "--sweep-scales",
        type=str,
        default="0.0,0.005,0.01,0.02,0.05,0.1,0.2,0.5",
        help="Comma-separated residual scales for sweep.",
    )
    parser.add_argument(
        "--nominal-model",
        type=str,
        default=None,
        help="Override path for nominal SAC weights (.zip).",
    )
    parser.add_argument(
        "--robust-model",
        type=str,
        default=None,
        help="Override path for robust SAC weights (.zip).",
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default=MODEL_XML_DEFAULT,
        help="MuJoCo XML model path.",
    )
    parser.add_argument(
        "--rl-control-dt",
        type=float,
        default=None,
        help="Override RL control dt for sweep.",
    )
    parser.add_argument(
        "--ball-noise",
        type=float,
        default=None,
        help="Override ball init noise for sweep.",
    )
    parser.add_argument(
        "--max-residual-rad",
        type=float,
        default=None,
        help="Override max residual radians for sweep.",
    )
    parser.add_argument("--kp", type=float, default=None, help="Override PD kp for sweep.")
    parser.add_argument("--kd", type=float, default=None, help="Override PD kd for sweep.")
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=None,
        help="Override torque limit for sweep.",
    )
    return parser.parse_args()


def _latest_eval_dir() -> Path:
    folders = sorted((REPO_ROOT / "results").glob("mujoco_eval_protocol_*"), key=lambda p: p.name)
    if not folders:
        raise FileNotFoundError("No mujoco_eval_protocol_* folders found under results/.")
    return folders[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_summary(eval_dir: Path) -> dict[str, dict[str, Any]]:
    raw = _load_json(eval_dir / "summary.json")["summary"]
    return {row["condition"]: row for row in raw}


def _load_per_episode(eval_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in NOMINAL_CONDITIONS}
    with open(eval_dir / "per_episode.csv") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            if cond not in by_condition:
                continue
            by_condition[cond].append(
                {
                    "episode": int(row["episode"]),
                    "hits": int(row["hits"]),
                    "reward": float(row["reward"]),
                    "sim_time": float(row["sim_time"]),
                    "survived": row["survived"].lower() == "true",
                }
            )
    return by_condition


def _style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": PALETTE["fg"],
            "axes.labelcolor": PALETTE["fg"],
            "axes.titleweight": "bold",
            "text.color": PALETTE["fg"],
            "xtick.color": PALETTE["fg"],
            "ytick.color": PALETTE["fg"],
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "-",
            "grid.alpha": 0.35,
            "font.size": 11,
        }
    )


def plot_nominal_hits(summary: dict[str, dict[str, Any]], out_dir: Path) -> None:
    labels = [DISPLAY_LABELS[c] for c in NOMINAL_CONDITIONS]
    means = [summary[c]["mean_hits"] for c in NOMINAL_CONDITIONS]
    stds = [summary[c]["std_hits"] for c in NOMINAL_CONDITIONS]
    colors = [
        MODEL_COLORS["FSM Baseline"],
        MODEL_COLORS["Nominal SAC"],
        MODEL_COLORS["Robust SAC"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor=PALETTE["fg"],
        linewidth=0.8,
        alpha=0.9,
    )
    for bar, m in zip(bars, means):
        ax.annotate(
            f"{m:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, m),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Hits per Episode")
    ax.set_title("Nominal MuJoCo Validation: Hits")
    ax.grid(axis="y")
    fig.tight_layout()
    path = out_dir / "nominal_hits_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_nominal_distribution(per_episode: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    labels = [DISPLAY_LABELS[c] for c in NOMINAL_CONDITIONS]
    data = [[r["hits"] for r in per_episode[c]] for c in NOMINAL_CONDITIONS]
    colors = [
        MODEL_COLORS["FSM Baseline"],
        MODEL_COLORS["Nominal SAC"],
        MODEL_COLORS["Robust SAC"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": PALETTE["fg"], "linewidth": 1.4},
        whiskerprops={"color": PALETTE["fg"]},
        capprops={"color": PALETTE["fg"]},
    )
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(PALETTE["fg"])

    rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=14, alpha=0.45, color=c, zorder=3)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hits per Episode")
    ax.set_title("Nominal MuJoCo Validation: Episode Hit Distribution")
    ax.grid(axis="y")
    fig.tight_layout()
    path = out_dir / "nominal_hit_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_nominal_reward_survival(summary: dict[str, dict[str, Any]], out_dir: Path) -> None:
    labels = [DISPLAY_LABELS[c] for c in NOMINAL_CONDITIONS]
    rewards = [summary[c]["mean_reward"] for c in NOMINAL_CONDITIONS]
    reward_stds = [summary[c]["std_reward"] for c in NOMINAL_CONDITIONS]
    sim_times = [summary[c]["mean_sim_time"] for c in NOMINAL_CONDITIONS]
    survival = [100.0 * summary[c]["survival_rate"] for c in NOMINAL_CONDITIONS]
    colors = [
        MODEL_COLORS["FSM Baseline"],
        MODEL_COLORS["Nominal SAC"],
        MODEL_COLORS["Robust SAC"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))

    bars0 = axes[0].bar(
        x,
        rewards,
        yerr=reward_stds,
        capsize=5,
        color=colors,
        edgecolor=PALETTE["fg"],
        linewidth=0.8,
        alpha=0.9,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("Reward")
    axes[0].grid(axis="y")
    for bar, val in zip(bars0, rewards):
        axes[0].annotate(
            f"{val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    bars1 = axes[1].bar(
        x,
        sim_times,
        color=colors,
        edgecolor=PALETTE["fg"],
        linewidth=0.8,
        alpha=0.9,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean Survival Time (s)")
    axes[1].set_ylim(0, 22.0)
    axes[1].axhline(20.0, color=PALETTE["accent"], linestyle="--", linewidth=1.1, label="20s cap")
    axes[1].set_title("Survival")
    axes[1].grid(axis="y")
    for bar, t, s in zip(bars1, sim_times, survival):
        axes[1].annotate(
            f"{t:.1f}s\n({s:.0f}%)",
            xy=(bar.get_x() + bar.get_width() / 2.0, t),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[1].legend(loc="upper left", fontsize=9)

    fig.suptitle("Nominal MuJoCo Validation: Reward and Survival", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "nominal_reward_survival.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def _parse_scales(scales_str: str) -> list[float]:
    parts = [p.strip() for p in scales_str.split(",") if p.strip()]
    scales = sorted({float(p) for p in parts})
    if not scales:
        raise ValueError("No sweep scales provided.")
    return scales


def _coalesce(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _extract_global_config(metadata: dict[str, Any]) -> dict[str, float]:
    cfg = metadata.get("global_eval_config", {})
    return {
        "rl_control_dt": float(cfg.get("rl_control_dt", 0.01)),
        "ball_init_pos_noise": float(cfg.get("ball_init_pos_noise", 0.04)),
        "max_residual_rad": float(cfg.get("max_residual_rad", 0.15)),
        "kp": float(cfg.get("kp", 3500.0)),
        "kd": float(cfg.get("kd", 14.0)),
        "torque_limit": float(cfg.get("torque_limit", 400.0)),
    }


def _extract_model_path(metadata: dict[str, Any], condition_name: str) -> str | None:
    for cond in metadata.get("conditions", []):
        if cond.get("condition") == condition_name:
            return cond.get("model_path")
    return None


def _resolve_model_path(path_hint: str | None, fallback_name: str) -> str:
    if path_hint and path_hint != "fsm_zero":
        p = Path(path_hint).expanduser().resolve()
        if p.exists():
            return str(p)
    p = (REPO_ROOT / fallback_name).resolve()
    if p.exists():
        return str(p)
    raise FileNotFoundError(
        f"Could not resolve model path. Tried metadata and fallback: {p}"
    )


def _evaluate_condition(
    model_path: str | None,
    residual_scale: float,
    episodes: int,
    model_xml: str,
    cfg_base: dict[str, float],
) -> dict[str, float]:
    env_cfg = MujocoResidualConfig(
        randomize_dynamics=False,
        residual_scale=residual_scale,
        rl_control_dt=cfg_base["rl_control_dt"],
        ball_init_pos_noise=cfg_base["ball_init_pos_noise"],
        max_residual_rad=cfg_base["max_residual_rad"],
        kp=cfg_base["kp"],
        kd=cfg_base["kd"],
        torque_limit=cfg_base["torque_limit"],
    )
    env = MujocoResidualEnv(model_path=model_xml, config=env_cfg)

    model = None
    if model_path is not None:
        install_numpy_pickle_compat_shims()
        from stable_baselines3 import SAC

        model = SAC.load(
            model_path,
            env=env,
            custom_objects=make_legacy_custom_objects(
                observation_space=env.observation_space,
                action_space=env.action_space,
            ),
        )

    hits = []
    rewards = []
    sim_times = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            if model is None:
                action = np.zeros(7, dtype=np.float64)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
        hits.append(int(info.get("episode_hits", 0)))
        rewards.append(ep_reward)
        sim_times.append(float(info.get("sim_time", 0.0)))

    env.close()
    return {
        "mean_hits": float(np.mean(hits)),
        "std_hits": float(np.std(hits)),
        "mean_reward": float(np.mean(rewards)),
        "mean_sim_time": float(np.mean(sim_times)),
    }


def run_residual_scale_sweep(
    eval_dir: Path,
    out_dir: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    cfg_base = _extract_global_config(metadata)
    cfg_base["rl_control_dt"] = _coalesce(args.rl_control_dt, cfg_base["rl_control_dt"])
    cfg_base["ball_init_pos_noise"] = _coalesce(args.ball_noise, cfg_base["ball_init_pos_noise"])
    cfg_base["max_residual_rad"] = _coalesce(args.max_residual_rad, cfg_base["max_residual_rad"])
    cfg_base["kp"] = _coalesce(args.kp, cfg_base["kp"])
    cfg_base["kd"] = _coalesce(args.kd, cfg_base["kd"])
    cfg_base["torque_limit"] = _coalesce(args.torque_limit, cfg_base["torque_limit"])

    nominal_hint = args.nominal_model or _extract_model_path(metadata, COND_NOMINAL)
    robust_hint = args.robust_model or _extract_model_path(metadata, COND_ROBUST)
    nominal_model_path = _resolve_model_path(nominal_hint, "sac_baseline_final.zip")
    robust_model_path = _resolve_model_path(robust_hint, "sac_robust_final.zip")

    scales = _parse_scales(args.sweep_scales)
    model_xml = str(Path(args.model_xml).expanduser().resolve())
    if not Path(model_xml).exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_xml}")

    rows: list[dict[str, Any]] = []

    # FSM baseline at residual scale 0 once.
    fsm_stats = _evaluate_condition(
        model_path=None,
        residual_scale=0.0,
        episodes=args.sweep_episodes,
        model_xml=model_xml,
        cfg_base=cfg_base,
    )
    rows.append(
        {
            "model_label": "FSM Baseline",
            "residual_scale": 0.0,
            "episodes": args.sweep_episodes,
            **fsm_stats,
        }
    )

    for model_label, model_path in [("Nominal SAC", nominal_model_path), ("Robust SAC", robust_model_path)]:
        for scale in scales:
            if np.isclose(scale, 0.0):
                stats = fsm_stats
            else:
                stats = _evaluate_condition(
                    model_path=model_path,
                    residual_scale=scale,
                    episodes=args.sweep_episodes,
                    model_xml=model_xml,
                    cfg_base=cfg_base,
                )
            row = {
                "model_label": model_label,
                "residual_scale": float(scale),
                "episodes": args.sweep_episodes,
                **stats,
            }
            rows.append(row)
            print(
                f"sweep {model_label:11s} scale={scale:<6g} "
                f"mean_hits={stats['mean_hits']:.2f} std={stats['std_hits']:.2f}"
            )

    csv_path = out_dir / "residual_scale_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "residual_scale",
                "episodes",
                "mean_hits",
                "std_hits",
                "mean_reward",
                "mean_sim_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")
    return rows


def plot_residual_scale_sweep(
    sweep_rows: list[dict[str, Any]],
    out_dir: Path,
    fsm_nominal_hits_protocol: float,
) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {"Nominal SAC": [], "Robust SAC": []}
    for row in sweep_rows:
        m = row["model_label"]
        if m in by_model:
            by_model[m].append(row)

    fsm_sweep_rows = [r for r in sweep_rows if r["model_label"] == "FSM Baseline"]
    fsm_hits_sweep = (
        float(fsm_sweep_rows[0]["mean_hits"])
        if fsm_sweep_rows
        else fsm_nominal_hits_protocol
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_label in ["Nominal SAC", "Robust SAC"]:
        rows = sorted(by_model[model_label], key=lambda r: r["residual_scale"])
        scales = np.array([r["residual_scale"] for r in rows], dtype=float)
        means = np.array([r["mean_hits"] for r in rows], dtype=float)
        stds = np.array([r["std_hits"] for r in rows], dtype=float)

        color = MODEL_COLORS[model_label]
        axes[0].plot(scales, means, marker="o", linewidth=2.0, color=color, label=model_label)
        axes[0].fill_between(scales, means - stds, means + stds, color=color, alpha=0.16)

        delta = means - fsm_hits_sweep
        axes[1].plot(scales, delta, marker="o", linewidth=2.0, color=color, label=model_label)

        best_idx = int(np.argmax(means))
        axes[0].scatter([scales[best_idx]], [means[best_idx]], color=PALETTE["accent"], s=55, zorder=4)
        axes[0].annotate(
            f"best {model_label}: {scales[best_idx]:.3f}",
            xy=(scales[best_idx], means[best_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=PALETTE["fg"],
        )

    axes[0].axhline(
        fsm_hits_sweep,
        color=MODEL_COLORS["FSM Baseline"],
        linestyle="--",
        linewidth=1.5,
        label=f"FSM baseline sweep ({fsm_hits_sweep:.1f})",
    )
    if not np.isclose(fsm_nominal_hits_protocol, fsm_hits_sweep):
        axes[0].axhline(
            fsm_nominal_hits_protocol,
            color=PALETTE["warning"],
            linestyle=":",
            linewidth=1.3,
            label=f"FSM baseline protocol ({fsm_nominal_hits_protocol:.1f})",
        )
    axes[0].set_xlabel("Residual Scale")
    axes[0].set_ylabel("Mean Hits per Episode")
    axes[0].set_title("Residual-Scale Sensitivity (Nominal MuJoCo)")
    axes[0].grid(True)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].axhline(0.0, color=PALETTE["fg"], linewidth=1.0)
    axes[1].set_xlabel("Residual Scale")
    axes[1].set_ylabel("Mean Hits Gain vs FSM")
    axes[1].set_title("Performance Margin over FSM")
    axes[1].grid(True)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    path = out_dir / "residual_scale_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def write_notes(
    out_dir: Path,
    summary: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    sweep_rows: list[dict[str, Any]] | None,
) -> None:
    fsm_hits = float(summary[COND_FSM]["mean_hits"])
    robust_hits = float(summary[COND_ROBUST]["mean_hits"])
    nominal_hits = float(summary[COND_NOMINAL]["mean_hits"])
    robust_gain = robust_hits - fsm_hits
    nominal_gain = nominal_hits - fsm_hits

    notes: dict[str, Any] = {
        "scope": "nominal_only_domain_validation",
        "excluded": ["all randomized MuJoCo conditions"],
        "result_delta_vs_fsm_nominal": {
            "nominal_sac": nominal_gain,
            "robust_sac": robust_gain,
        },
        "what_changed_to_get_this_regime": [
            "Residual authority during transfer was reduced from 0.5 to 0.01.",
            "MuJoCo validation now uses the same residual wrapper for FSM and RL.",
            "Model loading uses compatibility shims for legacy SB3/numpy pickles.",
            "Nominal-only interpretation: randomized MuJoCo figures are intentionally excluded.",
        ],
        "global_eval_config": metadata.get("global_eval_config", {}),
    }

    if sweep_rows:
        best_by_model: dict[str, dict[str, Any]] = {}
        for model_label in ["Nominal SAC", "Robust SAC"]:
            rows = [r for r in sweep_rows if r["model_label"] == model_label]
            if rows:
                best_by_model[model_label] = max(rows, key=lambda r: r["mean_hits"])
        notes["best_sweep_points"] = {
            k: {
                "residual_scale": v["residual_scale"],
                "mean_hits": v["mean_hits"],
                "std_hits": v["std_hits"],
                "episodes": v["episodes"],
            }
            for k, v in best_by_model.items()
        }

    path = out_dir / "domain_validation_notes.json"
    with open(path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"Saved: {path}")


def write_report_manifest(report_dir: Path, eval_dir: Path, out_dir: Path) -> None:
    manifest = {
        "source_eval_dir": str(eval_dir),
        "figures_dir": str(out_dir),
    }
    path = report_dir / "report_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    _style_matplotlib()

    eval_dir = Path(args.dir).expanduser().resolve() if args.dir else _latest_eval_dir()
    out_root = Path(args.out_root).expanduser().resolve()
    report_name = args.report_name or eval_dir.name
    report_dir = out_root / report_name
    out_dir = report_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(eval_dir)
    per_episode = _load_per_episode(eval_dir)
    metadata = _load_json(eval_dir / "metadata.json") if (eval_dir / "metadata.json").exists() else {}

    missing = [c for c in NOMINAL_CONDITIONS if c not in summary]
    if missing:
        raise ValueError(f"Missing required nominal conditions in summary.json: {missing}")

    print(f"Loaded eval dir: {eval_dir}")
    print("Using nominal conditions only:")
    for cond in NOMINAL_CONDITIONS:
        print(f"  {DISPLAY_LABELS[cond]}: mean_hits={summary[cond]['mean_hits']:.2f}")
    print(f"Report dir: {report_dir}")
    print(f"Figure output dir: {out_dir}\n")

    plot_nominal_hits(summary, out_dir)
    plot_nominal_distribution(per_episode, out_dir)
    plot_nominal_reward_survival(summary, out_dir)

    sweep_rows: list[dict[str, Any]] | None = None
    if args.run_residual_sweep:
        print("Running residual-scale sweep (nominal physics only)...")
        sweep_rows = run_residual_scale_sweep(eval_dir, out_dir, metadata, args)
        fsm_hits = float(summary[COND_FSM]["mean_hits"])
        plot_residual_scale_sweep(sweep_rows, out_dir, fsm_hits)
    else:
        print("Skipping residual-scale sweep.")

    write_notes(out_dir, summary, metadata, sweep_rows)
    write_report_manifest(report_dir, eval_dir, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
