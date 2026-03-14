"""
Tune residual scale for robust SAC in nominal MuJoCo.

By default this script compares Robust vs FSM and Nominal SAC at each scale and
selects the best scale using transfer margin:

    margin = robust_mean_hits - max(fsm_mean_hits, nominal_mean_hits)

Usage:
  python -m mujoco_transfer.tune_robust_residual_scale \
      --episodes 200 \
      --scales "0.006,0.008,0.01,0.012,0.015,0.02" \
      --tag robust_margin_tune
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
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


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _default_robust_model() -> Path:
    return _first_existing(
        REPO_ROOT / "sac_robust_final.zip",
        REPO_ROOT / "models/sac_robust_1m/sac_final.zip",
    )


def _default_nominal_model() -> Path:
    return _first_existing(
        REPO_ROOT / "sac_baseline_final.zip",
        REPO_ROOT / "mujoco_transfer/sac_nom_1m.zip",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune residual scale for robust SAC in nominal MuJoCo."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per scale.")
    parser.add_argument(
        "--robust-model",
        type=str,
        default=str(_default_robust_model()),
        help="Path to robust SAC weights (.zip).",
    )
    parser.add_argument(
        "--nominal-model",
        type=str,
        default=str(_default_nominal_model()),
        help="Path to nominal SAC weights (.zip). Used for transfer-margin objective.",
    )
    parser.add_argument(
        "--compare-nominal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare against nominal SAC while tuning (default: on).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="robust_margin",
        choices=["robust_margin", "robust_mean"],
        help=(
            "Scale-selection objective: robust_mean (maximize robust mean hits) or "
            "robust_margin (maximize robust - max(FSM, nominal))."
        ),
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default=MODEL_XML_DEFAULT,
        help="MuJoCo XML model path.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.0,0.005,0.01,0.02,0.05,0.1,0.2,0.5",
        help="Comma-separated residual scales to evaluate.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Base seed offset (episode seeds are seed_offset + [0..episodes-1]).",
    )
    parser.add_argument("--rl-control-dt", type=float, default=0.01, help="RL control dt.")
    parser.add_argument("--ball-noise", type=float, default=0.04, help="Ball spawn noise.")
    parser.add_argument("--max-residual-rad", type=float, default=0.15, help="Max residual radians.")
    parser.add_argument("--kp", type=float, default=3500.0, help="Joint PD kp.")
    parser.add_argument("--kd", type=float, default=14.0, help="Joint PD kd.")
    parser.add_argument("--torque-limit", type=float, default=400.0, help="Joint torque limit.")
    parser.add_argument(
        "--out-root",
        type=str,
        default="results/residual_scale_tuning",
        help="Root output directory for tuning runs.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional run tag suffix.")
    return parser.parse_args()


def _parse_scales(scales_str: str) -> list[float]:
    parts = [p.strip() for p in scales_str.split(",") if p.strip()]
    scales = sorted({float(p) for p in parts})
    if not scales:
        raise ValueError("No scales provided.")
    return scales


def _make_env(args: argparse.Namespace, residual_scale: float) -> MujocoResidualEnv:
    cfg = MujocoResidualConfig(
        randomize_dynamics=False,
        residual_scale=residual_scale,
        rl_control_dt=args.rl_control_dt,
        ball_init_pos_noise=args.ball_noise,
        max_residual_rad=args.max_residual_rad,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit,
    )
    return MujocoResidualEnv(model_path=args.model_xml, config=cfg)


def _evaluate_scale(
    args: argparse.Namespace,
    model: Any | None,
    residual_scale: float,
) -> dict[str, float]:
    env = _make_env(args, residual_scale=residual_scale)

    hits: list[int] = []
    rewards: list[float] = []
    sim_times: list[float] = []
    survivors = 0

    for i in range(args.episodes):
        seed = args.seed_offset + i
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        info = {}

        while not done:
            if model is None or np.isclose(residual_scale, 0.0):
                action = np.zeros(7, dtype=np.float64)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        hits_i = int(info.get("episode_hits", 0))
        sim_time = float(info.get("sim_time", 0.0))
        hits.append(hits_i)
        rewards.append(ep_reward)
        sim_times.append(sim_time)
        if sim_time >= 19.9:
            survivors += 1

    env.close()
    return {
        "mean_hits": float(np.mean(hits)),
        "std_hits": float(np.std(hits)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_sim_time": float(np.mean(sim_times)),
        "survival_rate": float(survivors / args.episodes),
    }


def _plot_sweep(
    rows: list[dict[str, Any]],
    out_dir: Path,
    objective: str,
) -> None:
    scales = np.array([r["residual_scale"] for r in rows], dtype=float)
    robust_mean = np.array([r["robust_mean_hits"] for r in rows], dtype=float)
    robust_std = np.array([r["robust_std_hits"] for r in rows], dtype=float)
    fsm_mean = np.array([r["fsm_mean_hits"] for r in rows], dtype=float)
    nominal_mean = np.array([r["nominal_mean_hits"] for r in rows], dtype=float)
    margin = np.array([r["margin_vs_best_baseline"] for r in rows], dtype=float)

    if objective == "robust_margin":
        best_idx = int(np.argmax(margin))
    else:
        best_idx = int(np.argmax(robust_mean))
    fsm_baseline = float(fsm_mean[0])

    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F5EF",
            "axes.facecolor": "#FFFFFF",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(scales, robust_mean, marker="o", linewidth=2.2, color="#1F7A8C", label="Robust SAC")
    axes[0].fill_between(scales, robust_mean - robust_std, robust_mean + robust_std, color="#1F7A8C", alpha=0.18)
    axes[0].plot(scales, nominal_mean, marker="s", linewidth=2.0, color="#2D6FAE", label="Nominal SAC")
    axes[0].axhline(fsm_baseline, color="#5B6777", linestyle="--", linewidth=1.4, label=f"FSM baseline ({fsm_baseline:.1f})")
    axes[0].scatter([scales[best_idx]], [robust_mean[best_idx]], s=55, color="#D4A33C", zorder=4)
    axes[0].annotate(
        f"best={scales[best_idx]:.3f}\nrobust={robust_mean[best_idx]:.1f}",
        xy=(scales[best_idx], robust_mean[best_idx]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].set_xlabel("Residual Scale")
    axes[0].set_ylabel("Mean Hits per Episode")
    axes[0].set_title("Robust Transfer Sweep (Nominal MuJoCo)")
    axes[0].legend(fontsize=9)

    axes[1].plot(scales, margin, marker="o", linewidth=2.2, color="#2D6FAE")
    axes[1].axhline(0.0, color="#1F2933", linewidth=1.0)
    axes[1].set_xlabel("Residual Scale")
    axes[1].set_ylabel("Robust Margin vs max(FSM, Nominal)")
    axes[1].set_title("Transfer Margin Objective")

    fig.tight_layout()
    path = out_dir / "robust_residual_scale_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def _output_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"robust_residual_tuning_{ts}"
    if args.tag:
        run_name += f"_{args.tag}"
    out_dir = Path(args.out_root).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    args.model_xml = str(Path(args.model_xml).expanduser().resolve())
    robust_model_path = str(Path(args.robust_model).expanduser().resolve())
    nominal_model_path = str(Path(args.nominal_model).expanduser().resolve())
    scales = _parse_scales(args.scales)

    if not Path(args.model_xml).exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {args.model_xml}")
    if not Path(robust_model_path).exists():
        raise FileNotFoundError(f"Robust model not found: {robust_model_path}")
    if args.compare_nominal and not Path(nominal_model_path).exists():
        raise FileNotFoundError(f"Nominal model not found: {nominal_model_path}")

    out_dir = _output_dir(args)
    print(f"Output directory: {out_dir}")
    print(f"Robust model: {robust_model_path}")
    if args.compare_nominal:
        print(f"Nominal model: {nominal_model_path}")
    print(f"Objective: {args.objective}")
    print(f"Scales: {scales}")
    print()

    # Load models once (predict only; env will be recreated per scale).
    install_numpy_pickle_compat_shims()
    probe_env = _make_env(args, residual_scale=0.0)
    from stable_baselines3 import SAC

    robust_model = SAC.load(
        robust_model_path,
        env=probe_env,
        custom_objects=make_legacy_custom_objects(
            observation_space=probe_env.observation_space,
            action_space=probe_env.action_space,
        ),
    )
    nominal_model = None
    if args.compare_nominal:
        nominal_model = SAC.load(
            nominal_model_path,
            env=probe_env,
            custom_objects=make_legacy_custom_objects(
                observation_space=probe_env.observation_space,
                action_space=probe_env.action_space,
            ),
        )
    probe_env.close()

    fsm_stats = _evaluate_scale(args, model=None, residual_scale=0.0)
    print(
        f"FSM baseline (scale=0): mean_hits={fsm_stats['mean_hits']:.2f} "
        f"std={fsm_stats['std_hits']:.2f}"
    )

    rows: list[dict[str, Any]] = []
    for scale in scales:
        robust_stats = _evaluate_scale(args, model=robust_model, residual_scale=scale)
        if nominal_model is not None:
            nominal_stats = _evaluate_scale(args, model=nominal_model, residual_scale=scale)
        else:
            nominal_stats = {
                "mean_hits": float("nan"),
                "std_hits": float("nan"),
                "mean_reward": float("nan"),
                "std_reward": float("nan"),
                "mean_sim_time": float("nan"),
                "survival_rate": float("nan"),
            }
        best_baseline = max(fsm_stats["mean_hits"], nominal_stats["mean_hits"] if nominal_model is not None else fsm_stats["mean_hits"])
        margin_vs_best = robust_stats["mean_hits"] - best_baseline

        row = {
            "residual_scale": float(scale),
            "episodes": int(args.episodes),
            "robust_mean_hits": robust_stats["mean_hits"],
            "robust_std_hits": robust_stats["std_hits"],
            "robust_mean_reward": robust_stats["mean_reward"],
            "robust_std_reward": robust_stats["std_reward"],
            "robust_mean_sim_time": robust_stats["mean_sim_time"],
            "robust_survival_rate": robust_stats["survival_rate"],
            "nominal_mean_hits": nominal_stats["mean_hits"],
            "nominal_std_hits": nominal_stats["std_hits"],
            "nominal_mean_reward": nominal_stats["mean_reward"],
            "nominal_std_reward": nominal_stats["std_reward"],
            "nominal_mean_sim_time": nominal_stats["mean_sim_time"],
            "nominal_survival_rate": nominal_stats["survival_rate"],
            "fsm_mean_hits": fsm_stats["mean_hits"],
            "fsm_std_hits": fsm_stats["std_hits"],
            "fsm_mean_reward": fsm_stats["mean_reward"],
            "fsm_std_reward": fsm_stats["std_reward"],
            "fsm_mean_sim_time": fsm_stats["mean_sim_time"],
            "fsm_survival_rate": fsm_stats["survival_rate"],
            "margin_vs_fsm": robust_stats["mean_hits"] - fsm_stats["mean_hits"],
            "margin_vs_nominal": robust_stats["mean_hits"] - nominal_stats["mean_hits"] if nominal_model is not None else float("nan"),
            "margin_vs_best_baseline": margin_vs_best,
        }
        rows.append(row)
        nominal_txt = (
            f"nominal={nominal_stats['mean_hits']:.2f}"
            if nominal_model is not None
            else "nominal=NA"
        )
        print(
            f"scale={scale:<6g} robust={robust_stats['mean_hits']:.2f} "
            f"{nominal_txt} fsm={fsm_stats['mean_hits']:.2f} "
            f"margin={margin_vs_best:+.2f}"
        )

    if args.objective == "robust_margin":
        best = max(rows, key=lambda r: (r["margin_vs_best_baseline"], r["robust_mean_hits"]))
    else:
        best = max(rows, key=lambda r: r["robust_mean_hits"])

    csv_path = out_dir / "robust_residual_scale_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "residual_scale",
                "episodes",
                "robust_mean_hits",
                "robust_std_hits",
                "robust_mean_reward",
                "robust_std_reward",
                "robust_mean_sim_time",
                "robust_survival_rate",
                "nominal_mean_hits",
                "nominal_std_hits",
                "nominal_mean_reward",
                "nominal_std_reward",
                "nominal_mean_sim_time",
                "nominal_survival_rate",
                "fsm_mean_hits",
                "fsm_std_hits",
                "fsm_mean_reward",
                "fsm_std_reward",
                "fsm_mean_sim_time",
                "fsm_survival_rate",
                "margin_vs_fsm",
                "margin_vs_nominal",
                "margin_vs_best_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    summary = {
        "objective": args.objective,
        "compare_nominal": args.compare_nominal,
        "best_residual_scale": best["residual_scale"],
        "best_robust_mean_hits": best["robust_mean_hits"],
        "best_robust_std_hits": best["robust_std_hits"],
        "best_margin_vs_fsm": best["margin_vs_fsm"],
        "best_margin_vs_nominal": best["margin_vs_nominal"],
        "best_margin_vs_best_baseline": best["margin_vs_best_baseline"],
        "episodes_per_scale": args.episodes,
        "scales": scales,
        "robust_model_path": robust_model_path,
        "nominal_model_path": nominal_model_path if args.compare_nominal else None,
        "model_xml": args.model_xml,
        "config": {
            "rl_control_dt": args.rl_control_dt,
            "ball_noise": args.ball_noise,
            "max_residual_rad": args.max_residual_rad,
            "kp": args.kp,
            "kd": args.kd,
            "torque_limit": args.torque_limit,
            "seed_offset": args.seed_offset,
        },
    }
    json_path = out_dir / "best_robust_residual_scale.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    _plot_sweep(rows, out_dir, objective=args.objective)

    print("\n=== Best Residual Scale (Robust SAC) ===")
    print(
        f"scale={best['residual_scale']:.6g}  "
        f"robust={best['robust_mean_hits']:.2f}  "
        f"fsm={best['fsm_mean_hits']:.2f}  "
        f"nominal={best['nominal_mean_hits']:.2f}  "
        f"margin={best['margin_vs_best_baseline']:+.2f}"
    )


if __name__ == "__main__":
    main()
