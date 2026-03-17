from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import mujoco
import numpy as np

from mujoco_transfer.fsm_ik_env import (
    MujocoFsmIkConfig,
    MujocoFsmIkEnv,
    TrajectoryFrame,
)


def _default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "iiwa_wsg_paddle_ball.xml"


def _episode_seed(base_seed: int | None, episode_idx: int) -> int | None:
    if base_seed is None:
        return None
    return base_seed + episode_idx


def _run_eval_loop(
    env: MujocoFsmIkEnv,
    episodes: int,
    seed: int | None,
    viewer_handle: Any | None,
    realtime: bool,
    replay_after: bool,
    replay_loops: int,
    replay_speed: float,
    replay_stride: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ep in range(episodes):
        ep_seed = _episode_seed(seed, ep)
        trajectory: list[TrajectoryFrame] | None = (
            [] if (viewer_handle is not None and replay_after) else None
        )
        result = env.run_episode(
            seed=ep_seed,
            viewer_handle=viewer_handle,
            realtime=realtime,
            trajectory=trajectory,
            trajectory_stride=replay_stride,
        )
        results.append(result)
        print(
            f"Episode {ep + 1:3d} | "
            f"hits={result['hits']:3d} | "
            f"fsm_hits={result['fsm_hits']:3d} | "
            f"plans={result['plans']:3d} | "
            f"ik_successes={result['ik_successes']:3d} | "
            f"time={result['sim_time']:.2f}s | "
            f"reason={result['reason']}"
        )

        if viewer_handle is not None and replay_after and trajectory:
            print(
                f"  Replaying episode {ep + 1} "
                f"({len(trajectory)} frames, speed={replay_speed}x, loops={replay_loops})..."
            )
            _replay_trajectory(
                env=env,
                viewer_handle=viewer_handle,
                trajectory=trajectory,
                speed=replay_speed,
                loops=replay_loops,
            )
    return results


def _print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    hits = np.array([r["hits"] for r in results], dtype=float)
    fsm_hits = np.array([r["fsm_hits"] for r in results], dtype=float)
    plans = np.array([r["plans"] for r in results], dtype=float)
    ik_ok = np.array([r["ik_successes"] for r in results], dtype=float)
    sim_time = np.array([r["sim_time"] for r in results], dtype=float)

    print("\n=== MuJoCo FSM/IK Summary ===")
    print(f"Episodes             : {len(results)}")
    print(f"Mean hits            : {hits.mean():.2f} ± {hits.std():.2f}")
    print(f"Mean FSM hits        : {fsm_hits.mean():.2f} ± {fsm_hits.std():.2f}")
    print(f"Max hits             : {int(hits.max())}")
    print(f"Mean plan count      : {plans.mean():.2f}")
    print(f"Mean IK successes    : {ik_ok.mean():.2f}")
    print(f"Mean sim time        : {sim_time.mean():.2f}s")


def _replay_trajectory(
    env: MujocoFsmIkEnv,
    viewer_handle: Any,
    trajectory: list[TrajectoryFrame],
    speed: float,
    loops: int,
) -> None:
    if not trajectory:
        return
    if speed <= 0.0:
        raise ValueError(f"Replay speed must be > 0, got {speed}")
    loops = max(1, int(loops))

    if len(trajectory) >= 2:
        frame_dt = max(1e-6, trajectory[1][2] - trajectory[0][2])
    else:
        frame_dt = float(env.model.opt.timestep)
    sleep_dt = frame_dt / speed

    for _ in range(loops):
        for qpos, qvel, _ in trajectory:
            env.data.qpos[:] = qpos
            env.data.qvel[:] = qvel
            mujoco.mj_forward(env.model, env.data)
            viewer_handle.sync()
            if sleep_dt > 0:
                time.sleep(sleep_dt)


def main() -> int:
    default_cfg = MujocoFsmIkConfig()
    parser = argparse.ArgumentParser(
        description="Run Drake FSM/IK controller in MuJoCo transfer environment."
    )
    parser.add_argument(
        "--model",
        default=str(_default_model_path()),
        help="Path to MuJoCo XML model.",
    )
    parser.add_argument(
        "--scenario-yaml",
        default=None,
        help="Drake scenario YAML used for initializing the FSM IK model.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=default_cfg.physics_dt,
        help="Physics integration timestep (s).",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=default_cfg.control_dt,
        help="FSM/control update period (s).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=20.0,
        help="Episode horizon in seconds.",
    )
    parser.add_argument(
        "--ball-x",
        type=float,
        default=default_cfg.ball_init_pos[0],
        help="Ball spawn x (m).",
    )
    parser.add_argument(
        "--ball-y",
        type=float,
        default=default_cfg.ball_init_pos[1],
        help="Ball spawn y (m).",
    )
    parser.add_argument(
        "--ball-z",
        type=float,
        default=default_cfg.ball_init_pos[2],
        help="Ball spawn z (m).",
    )
    parser.add_argument(
        "--ball-noise",
        type=float,
        default=default_cfg.ball_init_pos_noise,
        help="Uniform position noise for ball spawn.",
    )
    parser.add_argument("--kp", type=float, default=default_cfg.kp, help="Joint PD Kp.")
    parser.add_argument("--kd", type=float, default=default_cfg.kd, help="Joint PD Kd.")
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=default_cfg.torque_limit,
        help="Per-joint torque clamp (Nm).",
    )
    parser.add_argument(
        "--bias-comp",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.use_bias_compensation,
        help="Enable/disable MuJoCo model bias compensation in torque command.",
    )
    parser.add_argument(
        "--bias-comp-scale",
        type=float,
        default=default_cfg.bias_compensation_scale,
        help="Scale factor for MuJoCo bias compensation (1.0 = full).",
    )
    parser.add_argument(
        "--contact-force-scale",
        type=float,
        default=default_cfg.contact_force_scale,
        help="Scale factor for ball contact-force proxy passed to FSM hit logic.",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=default_cfg.hit_force_threshold,
        help="Contact force threshold for MuJoCo contact-event hit counting.",
    )
    parser.add_argument(
        "--hit-min-z",
        type=float,
        default=default_cfg.hit_min_ball_height,
        help="Minimum ball height for counting a MuJoCo contact-event hit.",
    )
    parser.add_argument(
        "--hit-debounce",
        type=float,
        default=default_cfg.hit_debounce_s,
        help="Minimum time between consecutive MuJoCo contact-event hits (s).",
    )
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep to approximate real-time when rendering.",
    )
    parser.add_argument(
        "--replay-after",
        action="store_true",
        help="Replay each rendered episode after it terminates.",
    )
    parser.add_argument(
        "--replay-loops",
        type=int,
        default=2,
        help="How many times to replay each episode when --replay-after is enabled.",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.25,
        help="Replay speed multiplier (e.g., 0.25 = quarter-speed).",
    )
    parser.add_argument(
        "--replay-stride",
        type=int,
        default=1,
        help="Record every N physics steps for replay (1 = all frames).",
    )
    args = parser.parse_args()

    if args.replay_after and not args.render:
        parser.error("--replay-after requires --render.")
    if args.render and platform.system() == "Darwin":
        launched_via_mjpython = bool(os.environ.get("MJPYTHON_BIN"))
        if not launched_via_mjpython:
            raise RuntimeError(
                "On macOS, MuJoCo rendering must be launched with mjpython.\n"
                "Run:\n"
                "  .venv/bin/mjpython -m mujoco_transfer.run_fsm_ik_mujoco "
                "--episodes 1 --render --realtime"
            )

    cfg = MujocoFsmIkConfig(
        physics_dt=args.physics_dt,
        control_dt=args.control_dt,
        max_episode_time=args.max_time,
        ball_init_pos=(args.ball_x, args.ball_y, args.ball_z),
        ball_init_pos_noise=args.ball_noise,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit,
        use_bias_compensation=args.bias_comp,
        bias_compensation_scale=args.bias_comp_scale,
        contact_force_scale=args.contact_force_scale,
        hit_force_threshold=args.hit_threshold,
        hit_min_ball_height=args.hit_min_z,
        hit_debounce_s=args.hit_debounce,
    )
    env = MujocoFsmIkEnv(
        model_path=args.model,
        config=cfg,
        scenario_yaml=args.scenario_yaml,
    )

    if args.render:
        try:
            import mujoco.viewer
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo viewer is unavailable. Try running without --render."
            ) from exc

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            results = _run_eval_loop(
                env=env,
                episodes=args.episodes,
                seed=args.seed,
                viewer_handle=viewer,
                realtime=args.realtime,
                replay_after=args.replay_after,
                replay_loops=args.replay_loops,
                replay_speed=args.replay_speed,
                replay_stride=args.replay_stride,
            )
    else:
        results = _run_eval_loop(
            env=env,
            episodes=args.episodes,
            seed=args.seed,
            viewer_handle=None,
            realtime=False,
            replay_after=False,
            replay_loops=1,
            replay_speed=1.0,
            replay_stride=1,
        )

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
