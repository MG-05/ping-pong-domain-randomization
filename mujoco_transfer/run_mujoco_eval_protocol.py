"""
Reproducible MuJoCo evaluation protocol for all 6 conditions:
  1. FSM Baseline (Nominal Physics)
  2. FSM Baseline (Randomized Physics)   ← residual_scale=0 wrapper
  3. Nominal SAC + Nominal Physics
  4. Nominal SAC + Randomized Physics
  5. Robust SAC + Nominal Physics
  6. Robust SAC + Randomized Physics

Outputs to results/mujoco_eval_protocol_YYYYMMDD_HHMMSS/:
  - metadata.json
  - per_episode.csv
  - summary.json

Usage:
  python -m mujoco_transfer.run_mujoco_eval_protocol
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from mujoco_transfer.fsm_ik_env import MujocoFsmIkEnv, MujocoFsmIkConfig
from mujoco_transfer.residual_env_mujoco import MujocoResidualEnv, MujocoResidualConfig

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODEL_XML  = str(REPO_ROOT / "mujoco_transfer/models/iiwa_wsg_paddle_ball.xml")
N_EPISODES = 500

CONDITIONS = [
    {
        "condition":    "FSM Baseline (Nominal)",
        "model_label":  "FSM Baseline",
        "physics":      "nominal",
        "model_path":   None,
        "randomize":    False,
        "residual_scale": None,
        "rl_control_dt":  None,
    },
    {
        # FSM with domain randomization: use residual_scale=0 wrapper so
        # physics randomization applies each reset, but RL adds zero residual.
        "condition":    "FSM Baseline (Randomized)",
        "model_label":  "FSM Baseline",
        "physics":      "randomized",
        "model_path":   "fsm_zero",        # sentinel handled in run_rl_condition
        "randomize":    True,
        "residual_scale": 0.0,
        "rl_control_dt":  0.001,
    },
    {
        "condition":    "Nominal (Nominal Physics)",
        "model_label":  "Nominal SAC",
        "physics":      "nominal",
        "model_path":   str(REPO_ROOT / "mujoco_transfer/sac_nom_1m.zip"),
        "randomize":    False,
        "residual_scale": 0.005,
        "rl_control_dt":  0.01,
    },
    {
        "condition":    "Nominal (Randomized Physics)",
        "model_label":  "Nominal SAC",
        "physics":      "randomized",
        "model_path":   str(REPO_ROOT / "mujoco_transfer/sac_nom_1m.zip"),
        "randomize":    True,
        "residual_scale": 0.005,
        "rl_control_dt":  0.01,
    },
    {
        "condition":    "Robust (Nominal Physics)",
        "model_label":  "Robust SAC",
        "physics":      "nominal",
        "model_path":   str(REPO_ROOT / "models/sac_robust_1m/sac_final.zip"),
        "randomize":    False,
        "residual_scale": 0.01,
        "rl_control_dt":  0.005,
    },
    {
        "condition":    "Robust (Randomized Physics)",
        "model_label":  "Robust SAC",
        "physics":      "randomized",
        "model_path":   str(REPO_ROOT / "models/sac_robust_1m/sac_final.zip"),
        "randomize":    True,
        "residual_scale": 0.01,
        "rl_control_dt":  0.005,
    },
]


def run_fsm_condition(n_eps: int) -> list[dict]:
    cfg = MujocoFsmIkConfig()
    env = MujocoFsmIkEnv(model_path=MODEL_XML, config=cfg)
    rows = []
    for ep in range(n_eps):
        result = env.run_episode(seed=ep)
        rows.append({
            "condition":    "FSM Baseline (Nominal)",
            "model_label":  "FSM Baseline",
            "physics":      "nominal",
            "episode":      ep + 1,
            "seed":         ep,
            "hits":         result["hits"],
            "reward":       0.0,  # no RL reward for FSM-only
            "sim_time":     result["sim_time"],
            "survived":     result["sim_time"] >= 19.9,
        })
        print(f"  ep {ep+1:3d}/{n_eps}: hits={result['hits']:3d}  sim_time={result['sim_time']:.2f}s")
    return rows


def run_rl_condition(cond: dict, n_eps: int) -> list[dict]:
    cfg = MujocoResidualConfig(
        randomize_dynamics=cond["randomize"],
        residual_scale=cond["residual_scale"],
        rl_control_dt=cond["rl_control_dt"],
    )
    env = MujocoResidualEnv(model_path=MODEL_XML, config=cfg)
    # "fsm_zero" sentinel: FSM-only through the RL wrapper (zero residuals)
    if cond["model_path"] == "fsm_zero":
        model = None
    else:
        model = SAC.load(cond["model_path"])
    rows = []
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
        hits     = info["episode_hits"]
        sim_time = info["sim_time"]
        rows.append({
            "condition":    cond["condition"],
            "model_label":  cond["model_label"],
            "physics":      cond["physics"],
            "episode":      ep + 1,
            "seed":         ep,
            "hits":         hits,
            "reward":       float(ep_reward),
            "sim_time":     float(sim_time),
            "survived":     sim_time >= 19.9,
        })
        print(f"  ep {ep+1:3d}/{n_eps}: hits={hits:3d}  reward={ep_reward:7.2f}  sim_time={sim_time:.2f}s")
    return rows


def summarize(rows: list[dict]) -> dict:
    hits     = [r["hits"]     for r in rows]
    rewards  = [r["reward"]   for r in rows]
    times    = [r["sim_time"] for r in rows]
    survived = [r["survived"] for r in rows]
    return {
        "n_episodes":     len(rows),
        "mean_hits":      float(np.mean(hits)),
        "std_hits":       float(np.std(hits)),
        "max_hits":       int(np.max(hits)),
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "mean_sim_time":  float(np.mean(times)),
        "survival_rate":  float(np.mean(survived)),
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results" / f"mujoco_eval_protocol_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for cond in CONDITIONS:
        print(f"=== {cond['condition']} ===")
        if cond["model_path"] is None:
            rows = run_fsm_condition(N_EPISODES)
        else:
            rows = run_rl_condition(cond, N_EPISODES)

        s = summarize(rows)
        s["condition"]    = cond["condition"]
        s["model_label"]  = cond["model_label"]
        s["physics"]      = cond["physics"]
        summaries.append(s)
        all_rows.extend(rows)

        print(f"  → mean_hits={s['mean_hits']:.1f} ± {s['std_hits']:.1f}  "
              f"survival={s['survival_rate']*100:.0f}%\n")

    # Write per_episode.csv
    csv_path = out_dir / "per_episode.csv"
    fieldnames = ["condition", "model_label", "physics", "episode", "seed",
                  "hits", "reward", "sim_time", "survived"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved: {csv_path}  ({len(all_rows)} rows)")

    # Write summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summaries}, f, indent=2)
    print(f"Saved: {summary_path}")

    # Write metadata.json
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "n_episodes_per_condition": N_EPISODES,
            "seeds": f"0–{N_EPISODES - 1}",
            "model_xml": MODEL_XML,
            "conditions": [
                {k: v for k, v in c.items() if k != "model_path"}
                | {"model_path": c["model_path"]}
                for c in CONDITIONS
            ],
        }, f, indent=2)
    print(f"Saved: {meta_path}")

    print("\n=== Final Summary ===")
    print(f"{'Condition':<35} {'Mean Hits':>10} {'Std':>6} {'Survival':>9}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['condition']:<35} {s['mean_hits']:>10.1f} {s['std_hits']:>6.1f} "
              f"{s['survival_rate']*100:>8.0f}%")


if __name__ == "__main__":
    main()
