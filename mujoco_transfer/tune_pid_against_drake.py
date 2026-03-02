from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_transfer.fsm_ik_env import MujocoFsmIkConfig, MujocoFsmIkEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "mujoco_transfer" / "models" / "iiwa_wsg_paddle_ball.xml"
DEFAULT_TARGET = REPO_ROOT / "results" / "fsm_baseline_eval.json"


def _load_drake_target(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    det = data["deterministic"]
    return {
        "hits": float(det["hits"]),
        "sim_time": float(det["sim_time"]),
    }


def _sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _score(
    mean_hits: float,
    mean_time: float,
    target_hits: float,
    target_time: float,
) -> float:
    hit_err = abs(mean_hits - target_hits) / max(1.0, target_hits)
    time_err = abs(mean_time - target_time) / max(1.0, target_time)
    # Weight hits heavily because this is the key baseline metric.
    return 3.0 * hit_err + 1.0 * time_err


def _evaluate_config(
    model_path: Path,
    cfg: MujocoFsmIkConfig,
    episodes: int,
    seed0: int,
) -> dict[str, Any]:
    hits = []
    sim_time = []
    plans = []
    ik = []
    reasons = []

    for i in range(episodes):
        env = MujocoFsmIkEnv(model_path=model_path, config=cfg)
        res = env.run_episode(seed=seed0 + i)
        hits.append(float(res["hits"]))
        sim_time.append(float(res["sim_time"]))
        plans.append(float(res["plans"]))
        ik.append(float(res["ik_successes"]))
        reasons.append(str(res["reason"]))

    return {
        "mean_hits": float(np.mean(hits)),
        "std_hits": float(np.std(hits)),
        "mean_sim_time": float(np.mean(sim_time)),
        "mean_plans": float(np.mean(plans)),
        "mean_ik_successes": float(np.mean(ik)),
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune MuJoCo joint PD gains to match Drake FSM deterministic baseline."
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--target-json", type=str, default=str(DEFAULT_TARGET))
    parser.add_argument(
        "--target-hits",
        type=float,
        default=None,
        help="Override target hit count (otherwise read deterministic target from --target-json).",
    )
    parser.add_argument(
        "--target-sim-time",
        type=float,
        default=None,
        help="Override target simulation time in seconds.",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="mujoco_transfer/pid_tuning_results.json")

    # Drake-like evaluation settings by default.
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument("--ball-x", type=float, default=0.77)
    parser.add_argument("--ball-y", type=float, default=-0.03)
    parser.add_argument("--ball-z", type=float, default=0.70)
    parser.add_argument("--ball-noise", type=float, default=0.0)

    parser.add_argument("--kp-min", type=float, default=200.0)
    parser.add_argument("--kp-max", type=float, default=5000.0)
    parser.add_argument("--kd-min", type=float, default=10.0)
    parser.add_argument("--kd-max", type=float, default=500.0)
    parser.add_argument("--tl-min", type=float, default=80.0)
    parser.add_argument("--tl-max", type=float, default=600.0)
    parser.add_argument(
        "--bias-comp",
        action=argparse.BooleanOptionalAction,
        default=MujocoFsmIkConfig().use_bias_compensation,
        help="Enable/disable MuJoCo model bias compensation in torque command.",
    )
    parser.add_argument("--bias-min", type=float, default=0.75)
    parser.add_argument("--bias-max", type=float, default=1.25)
    parser.add_argument("--contact-scale-min", type=float, default=0.75)
    parser.add_argument("--contact-scale-max", type=float, default=1.50)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    target_path = Path(args.target_json).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target = _load_drake_target(target_path)
    target_hits = float(args.target_hits) if args.target_hits is not None else target["hits"]
    target_time = (
        float(args.target_sim_time) if args.target_sim_time is not None else target["sim_time"]
    )

    print("=== PID Tuning Against Drake Deterministic Baseline ===")
    print(f"Target hits      : {target_hits:.1f}")
    print(f"Target sim time  : {target_time:.2f}s")
    print(f"Iterations       : {args.iterations}")
    print(f"Episodes/iter    : {args.episodes}")
    print(f"Model            : {model_path}")

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for i in range(args.iterations):
        kp = _sample_log_uniform(rng, args.kp_min, args.kp_max)
        kd = _sample_log_uniform(rng, args.kd_min, args.kd_max)
        tl = _sample_log_uniform(rng, args.tl_min, args.tl_max)
        bias = float(rng.uniform(args.bias_min, args.bias_max))
        contact_scale = float(rng.uniform(args.contact_scale_min, args.contact_scale_max))

        cfg = MujocoFsmIkConfig(
            physics_dt=args.physics_dt,
            control_dt=args.control_dt,
            max_episode_time=args.max_time,
            kp=kp,
            kd=kd,
            torque_limit=tl,
            use_bias_compensation=args.bias_comp,
            bias_compensation_scale=bias,
            contact_force_scale=contact_scale,
            ball_init_pos=(args.ball_x, args.ball_y, args.ball_z),
            ball_init_pos_noise=args.ball_noise,
        )
        metrics = _evaluate_config(model_path, cfg, args.episodes, seed0=10_000 + i * 100)
        score = _score(
            mean_hits=metrics["mean_hits"],
            mean_time=metrics["mean_sim_time"],
            target_hits=target_hits,
            target_time=target_time,
        )
        row = {
            "iter": i + 1,
            "kp": kp,
            "kd": kd,
            "torque_limit": tl,
            "bias_compensation_scale": bias,
            "contact_force_scale": contact_scale,
            "score": score,
            **metrics,
        }
        all_rows.append(row)

        if best is None or row["score"] < best["score"]:
            best = row
            print(
                f"[BEST {i + 1:3d}] score={row['score']:.4f} | "
                f"hits={row['mean_hits']:.2f} | "
                f"time={row['mean_sim_time']:.2f}s | "
                f"kp={kp:.1f} kd={kd:.1f} tl={tl:.1f} "
                f"bias={bias:.2f} cscale={contact_scale:.2f}"
            )
        elif (i + 1) % 10 == 0:
            print(
                f"[iter {i + 1:3d}] score={row['score']:.4f} | "
                f"hits={row['mean_hits']:.2f} | "
                f"time={row['mean_sim_time']:.2f}s"
            )

    result = {
        "target": target,
        "search": {
            "iterations": args.iterations,
            "episodes_per_iter": args.episodes,
            "seed": args.seed,
            "physics_dt": args.physics_dt,
            "control_dt": args.control_dt,
            "max_time": args.max_time,
            "ball_init_pos": [args.ball_x, args.ball_y, args.ball_z],
            "ball_noise": args.ball_noise,
        },
        "best": best,
        "all": all_rows,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved tuning results to {out_path}")
    if best is not None:
        print(
            "Best config: "
            f"kp={best['kp']:.3f}, kd={best['kd']:.3f}, torque_limit={best['torque_limit']:.3f} | "
            f"bias_compensation_scale={best['bias_compensation_scale']:.3f}, "
            f"contact_force_scale={best['contact_force_scale']:.3f} | "
            f"mean_hits={best['mean_hits']:.2f}, mean_sim_time={best['mean_sim_time']:.2f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
