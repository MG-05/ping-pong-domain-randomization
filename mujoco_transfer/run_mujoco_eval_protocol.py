"""
Domain-validation protocol for Drake-trained residual policies in MuJoCo.

This script intentionally enforces *matched evaluation conditions*:
1. All conditions run through the same residual wrapper (including FSM baseline
   via zero residual actions).
2. Action period, spawn-noise, control gains, and episode accounting are shared.
3. Policy residual authority defaults to Drake-training scale (0.5), not the
   heavily reduced transfer-only scales used in prior sweeps.

Outputs to results/mujoco_eval_protocol_YYYYMMDD_HHMMSS/:
  - metadata.json
  - per_episode.csv
  - summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_transfer.residual_env_mujoco import MujocoResidualConfig, MujocoResidualEnv
from mujoco_transfer.sb3_compat import (
    install_numpy_pickle_compat_shims,
    make_legacy_custom_objects,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = str(REPO_ROOT / "mujoco_transfer/models/iiwa_wsg_paddle_ball.xml")


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def _default_nominal_model() -> Path:
    return _first_existing(
        REPO_ROOT / "sac_baseline_final.zip",
        REPO_ROOT / "mujoco_transfer/sac_nom_1m.zip",
    )


def _default_robust_model() -> Path:
    return _first_existing(
        REPO_ROOT / "sac_robust_final.zip",
        REPO_ROOT / "models/sac_robust_1m/sac_final.zip",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched MuJoCo domain-validation protocol."
    )
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per condition.")
    parser.add_argument("--model-xml", type=str, default=MODEL_XML, help="MuJoCo XML model.")
    parser.add_argument(
        "--nominal-model",
        type=str,
        default=str(_default_nominal_model()),
        help="Path to nominal SAC weights (.zip).",
    )
    parser.add_argument(
        "--robust-model",
        type=str,
        default=str(_default_robust_model()),
        help="Path to robust SAC weights (.zip).",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.5,
        help="Residual scale used for policy evaluation (default: Drake training value).",
    )
    parser.add_argument(
        "--max-residual-rad",
        type=float,
        default=0.15,
        help="Maximum residual magnitude in radians.",
    )
    parser.add_argument(
        "--rl-control-dt",
        type=float,
        default=0.01,
        help="RL action period in seconds (default: Drake training value).",
    )
    parser.add_argument(
        "--ball-noise",
        type=float,
        default=0.04,
        help="Ball spawn position noise (default: Drake training value).",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=3500.0,
        help="Joint PD kp shared across all conditions.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=14.0,
        help="Joint PD kd shared across all conditions.",
    )
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=400.0,
        help="Joint torque limit shared across all conditions.",
    )
    parser.add_argument(
        "--out-tag",
        type=str,
        default=None,
        help="Optional suffix for output folder name.",
    )
    parser.add_argument(
        "--nominal-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only nominal-physics conditions (FSM, Nominal SAC, Robust SAC).",
    )
    return parser.parse_args()


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition": "FSM Baseline (Nominal)",
            "model_label": "FSM Baseline",
            "physics": "nominal",
            "model_path": "fsm_zero",
            "randomize": False,
            "residual_scale": 0.0,
        },
        {
            "condition": "FSM Baseline (Randomized)",
            "model_label": "FSM Baseline",
            "physics": "randomized",
            "model_path": "fsm_zero",
            "randomize": True,
            "residual_scale": 0.0,
        },
        {
            "condition": "Nominal (Nominal Physics)",
            "model_label": "Nominal SAC",
            "physics": "nominal",
            "model_path": str(Path(args.nominal_model).expanduser().resolve()),
            "randomize": False,
            "residual_scale": args.residual_scale,
        },
        {
            "condition": "Nominal (Randomized Physics)",
            "model_label": "Nominal SAC",
            "physics": "randomized",
            "model_path": str(Path(args.nominal_model).expanduser().resolve()),
            "randomize": True,
            "residual_scale": args.residual_scale,
        },
        {
            "condition": "Robust (Nominal Physics)",
            "model_label": "Robust SAC",
            "physics": "nominal",
            "model_path": str(Path(args.robust_model).expanduser().resolve()),
            "randomize": False,
            "residual_scale": args.residual_scale,
        },
        {
            "condition": "Robust (Randomized Physics)",
            "model_label": "Robust SAC",
            "physics": "randomized",
            "model_path": str(Path(args.robust_model).expanduser().resolve()),
            "randomize": True,
            "residual_scale": args.residual_scale,
        },
    ]
    if args.nominal_only:
        conditions = [c for c in conditions if c["physics"] == "nominal"]
    return conditions


def validate_inputs(args: argparse.Namespace, conditions: list[dict[str, Any]]) -> None:
    model_xml = Path(args.model_xml).expanduser().resolve()
    if not model_xml.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {model_xml}")
    for cond in conditions:
        model_path = cond["model_path"]
        if model_path == "fsm_zero":
            continue
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model weights for condition '{cond['condition']}' not found: {p}"
            )


def run_condition(
    cond: dict[str, Any],
    args: argparse.Namespace,
    n_eps: int,
) -> list[dict[str, Any]]:
    cfg = MujocoResidualConfig(
        randomize_dynamics=cond["randomize"],
        residual_scale=cond["residual_scale"],
        max_residual_rad=args.max_residual_rad,
        rl_control_dt=args.rl_control_dt,
        ball_init_pos_noise=args.ball_noise,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit,
    )
    env = MujocoResidualEnv(model_path=args.model_xml, config=cfg)

    if cond["model_path"] == "fsm_zero":
        model = None
    else:
        install_numpy_pickle_compat_shims()
        from stable_baselines3 import SAC

        model = SAC.load(
            cond["model_path"],
            env=env,
            custom_objects=make_legacy_custom_objects(
                observation_space=env.observation_space,
                action_space=env.action_space,
            ),
        )

    rows: list[dict[str, Any]] = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            if model is None:
                action = np.zeros(7)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        hits = int(info["episode_hits"])
        sim_time = float(info["sim_time"])
        rows.append(
            {
                "condition": cond["condition"],
                "model_label": cond["model_label"],
                "physics": cond["physics"],
                "episode": ep + 1,
                "seed": ep,
                "hits": hits,
                "reward": float(ep_reward),
                "sim_time": sim_time,
                "survived": sim_time >= 19.9,
            }
        )
        print(
            f"  ep {ep + 1:3d}/{n_eps}: hits={hits:3d}  "
            f"reward={ep_reward:7.2f}  sim_time={sim_time:.2f}s"
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [r["hits"] for r in rows]
    rewards = [r["reward"] for r in rows]
    times = [r["sim_time"] for r in rows]
    survived = [r["survived"] for r in rows]
    return {
        "n_episodes": len(rows),
        "mean_hits": float(np.mean(hits)),
        "std_hits": float(np.std(hits)),
        "max_hits": int(np.max(hits)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_sim_time": float(np.mean(times)),
        "survival_rate": float(np.mean(survived)),
    }


def _make_out_dir(out_tag: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{out_tag}" if out_tag else ""
    out_dir = REPO_ROOT / "results" / f"mujoco_eval_protocol_{timestamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    conditions = build_conditions(args)
    validate_inputs(args, conditions)

    out_dir = _make_out_dir(args.out_tag)
    print(f"Output directory: {out_dir}\n")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for cond in conditions:
        print(f"=== {cond['condition']} ===")
        rows = run_condition(cond, args=args, n_eps=args.episodes)
        s = summarize(rows)
        s["condition"] = cond["condition"]
        s["model_label"] = cond["model_label"]
        s["physics"] = cond["physics"]
        summaries.append(s)
        all_rows.extend(rows)
        print(
            f"  → mean_hits={s['mean_hits']:.1f} ± {s['std_hits']:.1f}  "
            f"survival={s['survival_rate'] * 100:.0f}%\n"
        )

    csv_path = out_dir / "per_episode.csv"
    fieldnames = [
        "condition",
        "model_label",
        "physics",
        "episode",
        "seed",
        "hits",
        "reward",
        "sim_time",
        "survived",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved: {csv_path}  ({len(all_rows)} rows)")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summaries}, f, indent=2)
    print(f"Saved: {summary_path}")

    model_xml = str(Path(args.model_xml).expanduser().resolve())
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "timestamp": out_dir.name,
                "profile": "domain_validation_v1",
                "n_episodes_per_condition": args.episodes,
                "seeds": f"0–{args.episodes - 1}",
                "model_xml": model_xml,
                "global_eval_config": {
                    "rl_control_dt": args.rl_control_dt,
                    "ball_init_pos_noise": args.ball_noise,
                    "residual_scale_policy": args.residual_scale,
                    "max_residual_rad": args.max_residual_rad,
                    "kp": args.kp,
                    "kd": args.kd,
                    "torque_limit": args.torque_limit,
                },
                "conditions": conditions,
            },
            f,
            indent=2,
        )
    print(f"Saved: {meta_path}")

    print("\n=== Final Summary ===")
    print(f"{'Condition':<35} {'Mean Hits':>10} {'Std':>6} {'Survival':>9}")
    print("-" * 65)
    for s in summaries:
        print(
            f"{s['condition']:<35} {s['mean_hits']:>10.1f} {s['std_hits']:>6.1f} "
            f"{s['survival_rate'] * 100:>8.0f}%"
        )


if __name__ == "__main__":
    main()
