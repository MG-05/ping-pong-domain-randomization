"""Train a residual SAC policy on top of the FSM/IK inner-loop controller.

Usage:
    python -m src.train_rl --steps 100000
    python -m src.train_rl --steps 500000 --save-path data/sac_nominal
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.residual_env import EnvConfig, PingPongResidualEnv


class MetricsCallback(BaseCallback):
    """Log custom episode metrics to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_hits: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_hits.append(info.get("episode_hits", 0))
                self.logger.record(
                    "rollout/episode_hits", info.get("episode_hits", 0)
                )
                self.logger.record(
                    "rollout/episode_sim_time", info.get("sim_time", 0.0)
                )
        if len(self._episode_hits) >= 10:
            self.logger.record(
                "rollout/mean_hits_10ep",
                float(np.mean(self._episode_hits[-10:])),
            )
        return True


def make_train_env(config: EnvConfig | None = None) -> PingPongResidualEnv:
    env = PingPongResidualEnv(config=config)
    return Monitor(env)


def make_eval_env(config: EnvConfig | None = None) -> PingPongResidualEnv:
    cfg = config or EnvConfig()
    eval_cfg = EnvConfig(
        control_dt=cfg.control_dt,
        max_episode_steps=cfg.max_episode_steps,
        target_apex_height=cfg.target_apex_height,
        max_residual_rad=cfg.max_residual_rad,
        residual_scale=cfg.residual_scale,
        ball_init_pos=cfg.ball_init_pos,
        ball_init_pos_noise=0.0,
    )
    env = PingPongResidualEnv(config=eval_cfg)
    return Monitor(env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train residual SAC policy.")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Total training timesteps. 0 = print info and exit.",
    )
    parser.add_argument(
        "--save-path",
        default="data/sac_nominal",
        help="Directory to save models and logs.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size."
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100_000, help="Replay buffer size."
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Random exploration steps before training begins.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate policy every N steps.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.5,
        help="Scale factor for residual actions (0 = pure FSM, 1 = full range).",
    )
    parser.add_argument(
        "--max-residual",
        type=float,
        default=0.15,
        help="Max residual magnitude in radians.",
    )
    parser.add_argument(
        "--target-apex",
        type=float,
        default=0.55,
        help="Target ball apex height (metres).",
    )
    parser.add_argument(
        "--resume", default=None, help="Path to a saved model to resume from."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig(
        max_residual_rad=args.max_residual,
        residual_scale=args.residual_scale,
        target_apex_height=args.target_apex,
    )

    if args.steps <= 0:
        print("=== Ping-Pong Residual SAC Trainer ===")
        print(f"  Env observation dim : {20}")
        print(f"  Env action dim      : {7}")
        print(f"  Residual scale      : {args.residual_scale}")
        print(f"  Max residual (rad)  : {args.max_residual}")
        print(f"  Target apex (m)     : {args.target_apex}")
        print(f"  Save directory      : {save_dir}")
        print()
        print("Pass --steps N to start training.")
        return 0

    print(f"Building training environment …")
    train_env = DummyVecEnv([lambda: make_train_env(env_cfg)])
    eval_env = DummyVecEnv([lambda: make_eval_env(env_cfg)])

    policy_kwargs = dict(net_arch=[256, 256])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(
            args.resume,
            env=train_env,
            tensorboard_log=str(log_dir),
        )
        model.learning_rate = args.lr
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            verbose=1,
            seed=args.seed,
        )

    callbacks = [
        MetricsCallback(),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(save_dir / "checkpoints"),
            name_prefix="sac_residual",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "best"),
            log_path=str(save_dir / "eval_logs"),
            eval_freq=args.eval_freq,
            n_eval_episodes=3,
            deterministic=True,
        ),
    ]

    t0 = time.time()
    print(f"Starting SAC training for {args.steps} steps …")
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    final_path = save_dir / "sac_final"
    model.save(str(final_path))
    print(f"Training complete in {elapsed:.1f}s. Model saved to {final_path}")

    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
