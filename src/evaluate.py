"""Evaluate a trained residual SAC policy (or the pure FSM baseline).

Usage:
    # Evaluate trained policy
    python -m src.evaluate --model data/sac_nominal/sac_final.zip --episodes 20

    # Evaluate pure FSM (no residual) for baseline comparison
    python -m src.evaluate --fsm-only --episodes 20

    # Evaluate with meshcat visualisation (single episode)
    python -m src.evaluate --model data/sac_nominal/sac_final.zip --episodes 1 --render
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.envs.residual_env import EnvConfig, PingPongResidualEnv


def run_evaluation(
    env: PingPongResidualEnv,
    model: SAC | None,
    n_episodes: int,
    deterministic: bool = True,
) -> dict:
    episode_rewards: list[float] = []
    episode_hits: list[int] = []
    episode_lengths: list[int] = []
    episode_sim_times: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_hits.append(info.get("episode_hits", 0))
        episode_lengths.append(steps)
        episode_sim_times.append(info.get("sim_time", 0.0))

        print(
            f"  Episode {ep + 1:3d} | "
            f"reward={total_reward:8.2f} | "
            f"hits={info.get('episode_hits', 0):3d} | "
            f"steps={steps:5d} | "
            f"sim_time={info.get('sim_time', 0.0):.2f}s"
        )

    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_hits": float(np.mean(episode_hits)),
        "std_hits": float(np.std(episode_hits)),
        "max_hits": int(np.max(episode_hits)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_sim_time": float(np.mean(episode_sim_times)),
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained policy.")
    parser.add_argument(
        "--model", default=None, help="Path to trained SAC model (.zip)."
    )
    parser.add_argument(
        "--fsm-only",
        action="store_true",
        help="Evaluate the FSM controller with zero residual (baseline).",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes.")
    parser.add_argument(
        "--target-apex", type=float, default=0.55, help="Target apex height."
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering (single episode)."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions.",
    )
    args = parser.parse_args()

    if not args.fsm_only and args.model is None:
        parser.error("Provide --model PATH or use --fsm-only.")

    config = EnvConfig(
        target_apex_height=args.target_apex,
        ball_init_pos_noise=0.0,
    )
    env = PingPongResidualEnv(config=config)

    model = None
    mode_str = "FSM-only (zero residual)"
    if not args.fsm_only:
        model_path = args.model
        if not model_path.endswith(".zip"):
            model_path += ".zip"
        if not Path(model_path).exists():
            model_path = args.model
        model = SAC.load(model_path)
        mode_str = f"SAC residual policy ({args.model})"

    print(f"=== Evaluation: {mode_str} ===")
    print(f"  Episodes        : {args.episodes}")
    print(f"  Target apex (m) : {args.target_apex}")
    print()

    results = run_evaluation(
        env, model, args.episodes, deterministic=args.deterministic
    )

    print()
    print("=== Summary ===")
    print(f"  Mean reward     : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean hits       : {results['mean_hits']:.1f} ± {results['std_hits']:.1f}")
    print(f"  Max hits        : {results['max_hits']}")
    print(f"  Mean sim time   : {results['mean_sim_time']:.2f}s")
    print(f"  Mean ep length  : {results['mean_length']:.0f} steps")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
