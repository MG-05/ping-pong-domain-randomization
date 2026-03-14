import argparse
from pathlib import Path
import time
import json
import csv
import numpy as np

# Assuming Stable Baselines3 for the trained RL model
from stable_baselines3 import PPO, SAC 

import mujoco
from mujoco_transfer.residual_env_mujoco import MujocoResidualEnv, MujocoResidualConfig
from mujoco_transfer.sb3_compat import (
    install_numpy_pickle_compat_shims,
    make_legacy_custom_objects,
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Ping-Pong Policy in MuJoCo")
    parser.add_argument("--env-model", type=str, default="mujoco_transfer/models/iiwa_wsg_paddle_ball.xml", help="Path to MuJoCo XML")
    parser.add_argument("--rl-weights", type=str, required=True, help="Path to trained RL model weights (.zip)")
    parser.add_argument("--algo", type=str, default="SAC", choices=["PPO", "SAC"], help="RL Algorithm used")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the simulation visually")
    parser.add_argument("--realtime", action="store_true", help="Slow down rendering to real-time")
    parser.add_argument("--out", type=str, default=None, help="Path to save report")
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable domain randomization (--randomize / --no-randomize)")
    parser.add_argument("--residual-scale", type=float, default=None,
                        help="Residual action scale (default: environment config, currently 0.5)")
    parser.add_argument("--rl-control-dt", type=float, default=None,
                        help="RL action period in seconds (default: 0.01; use 0.005 for robust)")
    args = parser.parse_args()

    # 1. Initialize the Environment
    cfg_kwargs = dict(randomize_dynamics=args.randomize)
    if args.residual_scale is not None:
        cfg_kwargs["residual_scale"] = args.residual_scale
    if args.rl_control_dt is not None:
        cfg_kwargs["rl_control_dt"] = args.rl_control_dt
    cfg = MujocoResidualConfig(**cfg_kwargs)

    env = MujocoResidualEnv(model_path=args.env_model, config=cfg)

    # 2. Load the Trained Model
    print(f"Loading {args.algo} model from {args.rl_weights}...")
    install_numpy_pickle_compat_shims()
    custom_objects = make_legacy_custom_objects(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    if args.algo == "PPO":
        model = PPO.load(args.rl_weights, env=env, custom_objects=custom_objects)
    else:
        model = SAC.load(args.rl_weights, env=env, custom_objects=custom_objects)

    # 3. Setup rendering if requested
    viewer = None
    if args.render:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        except ImportError:
            print("Warning: mujoco.viewer not found. Running headless.")

    # 4. Evaluation Loop
    episode_stats = []
    episode_rewards = []
    episode_hits = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0

        while not done:
            # RL agent predicts the residual action based on the observation
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            done = terminated or truncated

            # Sync viewer and enforce real-time if rendering
            if viewer is not None and viewer.is_running():
                viewer.sync()
                if args.realtime:
                    dt = env.model.opt.timestep * env._substeps_per_control
                    elapsed = time.perf_counter() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

        hits = info.get('episode_hits', 0)
        sim_time = info.get('sim_time', 0)
        
        episode_rewards.append(ep_reward)
        episode_hits.append(hits)
        
        # Track data for the report
        episode_stats.append({
            "episode": ep + 1,
            "reward": float(ep_reward),
            "hits": int(hits),
            "sim_time": float(sim_time)
        })
        
        print(f"Episode {ep + 1}/{args.episodes} | "
              f"Reward: {ep_reward:.2f} | "
              f"Hits: {hits} | "
              f"Sim Time: {sim_time:.2f}s")

    if viewer is not None:
        viewer.close()

    # 5. Print Summary
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_hits = float(np.mean(episode_hits))
    max_hits = int(np.max(episode_hits))

    print("\n=== RL Evaluation Summary ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Hits:   {mean_hits:.2f}")
    print(f"Max Hits:    {max_hits}")

    # 6. Save Report to JSON or CSV
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if out_path.suffix.lower() == ".json":
            report = {
                "config": {
                    "env_model": args.env_model,
                    "rl_weights": args.rl_weights,
                    "algo": args.algo,
                    "episodes": args.episodes
                },
                "summary": {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "mean_hits": mean_hits,
                    "max_hits": max_hits
                },
                "episodes": episode_stats
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Saved JSON report to {out_path}")
            
        elif out_path.suffix.lower() == ".csv":
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "reward", "hits", "sim_time"])
                writer.writeheader()
                writer.writerows(episode_stats)
            print(f"Saved CSV report to {out_path}")
        else:
            print(f"Warning: Unrecognized file extension '{out_path.suffix}'. Please use .json or .csv.")

if __name__ == "__main__":
    main()

# To run a headless baseline sweep to test performance:
# python -m mujoco_transfer.run_rl_mujoco --rl-weights path/to/best_model.zip --episodes 100
# To visually confirm the paddle is tracking the ball and applying the residual actions correctly:
# python -m mujoco_transfer.run_rl_mujoco --rl-weights path/to/best_model.zip --episodes 5 --render --realtime

# To save a JSON file (great for nested data and metadata):
# python -m mujoco_transfer.run_rl_mujoco --rl-weights path/to/model.zip --episodes 100 --out mujoco_transfer/results_sweep1.json
# To save a CSV file (great for dropping into Excel or graphing):
# python -m mujoco_transfer.run_rl_mujoco --rl-weights path/to/model.zip --episodes 100 --out mujoco_transfer/results_sweep1.csv
