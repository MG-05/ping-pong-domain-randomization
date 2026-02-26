"""Run all evaluations and save results to a JSON file."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs.residual_env import EnvConfig, PingPongResidualEnv
from src.evaluate import run_evaluation
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parent.parent
N_EPISODES = 20


def eval_model(model_path: str | None, label: str, randomize: bool) -> dict:
    """Evaluate a single model under given physics."""
    config = EnvConfig(
        target_apex_height=0.55,
        ball_init_pos_noise=0.0,
        use_randomization=randomize,
    )
    env = PingPongResidualEnv(config=config)

    model = None
    if model_path is not None:
        model = SAC.load(model_path)

    physics = "randomized" if randomize else "nominal"
    print(f"\n{'='*60}")
    print(f"  {label}  |  Physics: {physics}  |  Episodes: {N_EPISODES}")
    print(f"{'='*60}")

    results = run_evaluation(env, model, N_EPISODES, deterministic=True)
    env.close()

    results["label"] = label
    results["physics"] = physics
    results["model_path"] = model_path
    return results


def main():
    models = []

    # FSM baseline
    models.append(("FSM Baseline", None))

    # Nominal checkpoints from single 1M run
    nom_ckpt = REPO / "data" / "sac_nominal_1m" / "checkpoints"
    for steps in [500000, 750000]:
        p = nom_ckpt / f"sac_residual_{steps}_steps.zip"
        if p.exists():
            models.append((f"Nominal {steps // 1000}k", str(p)))
        else:
            print(f"WARNING: Missing {p}")
    nom_final = REPO / "data" / "sac_nominal_1m" / "sac_final.zip"
    if nom_final.exists():
        models.append(("Nominal 1M", str(nom_final)))

    # Robust checkpoints from 1M run (original run to 750k + resumed to 1M)
    rob_ckpt = REPO / "data" / "sac_robust_1m" / "checkpoints"
    for steps in [500000, 750000]:
        p = rob_ckpt / f"sac_residual_{steps}_steps.zip"
        if p.exists():
            models.append((f"Robust {steps // 1000}k", str(p)))
        else:
            print(f"WARNING: Missing {p}")
    rob_final = REPO / "data" / "sac_robust_1m" / "sac_final.zip"
    if rob_final.exists():
        models.append(("Robust 1M", str(rob_final)))

    all_results = []
    t0 = time.time()

    for label, model_path in models:
        for randomize in [False, True]:
            result = eval_model(model_path, label, randomize)
            all_results.append(result)
            print(f"  → Mean hits: {result['mean_hits']:.1f} ± {result['std_hits']:.1f}")

    elapsed = time.time() - t0
    print(f"\nAll evaluations complete in {elapsed:.0f}s")

    out_path = REPO / "results" / "all_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
