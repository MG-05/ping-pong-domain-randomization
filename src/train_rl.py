from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.envs.drake_gym_env import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description="RL training stub.")
    parser.add_argument("--steps", type=int, default=0, help="Training steps.")
    parser.add_argument(
        "--save-path",
        default="data/ppo_pingpong",
        help="Where to save the trained policy.",
    )
    args = parser.parse_args()

    env = make_env()

    if args.steps <= 0:
        print("Training stub only. Pass --steps N to run PPO.")
        return 0

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.steps)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
