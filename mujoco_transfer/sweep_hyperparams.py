"""
Sweep residual_scale and rl_control_dt for nominal and robust SAC models.
Goal: find config that maximizes hits for each model.
"""
import numpy as np
from stable_baselines3 import SAC
from mujoco_transfer.residual_env_mujoco import MujocoResidualEnv, MujocoResidualConfig

MODEL_PATH = "mujoco_transfer/models/iiwa_wsg_paddle_ball.xml"
MODELS = {
    "nominal": ("mujoco_transfer/sac_nom_1m.zip", False),
    "robust":  ("models/sac_robust_1m/sac_final.zip", True),
}
N_EPS = 15

SCALES   = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
RL_DTS   = [0.005, 0.01, 0.02]


def evaluate(weights, randomize, residual_scale, rl_control_dt, n_eps):
    cfg = MujocoResidualConfig(
        randomize_dynamics=randomize,
        residual_scale=residual_scale,
        rl_control_dt=rl_control_dt,
    )
    env = MujocoResidualEnv(model_path=MODEL_PATH, config=cfg)
    model = SAC.load(weights)
    hits_list = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        hits_list.append(info["episode_hits"])
    return float(np.mean(hits_list)), float(np.std(hits_list))


print(f"{'Model':<10} {'Rand':<6} {'scale':<7} {'dt':<7} {'mean_hits':>9} {'std':>6}")
print("-" * 52)

best = {}
for name, (weights, randomize) in MODELS.items():
    key = f"{name}_{'rand' if randomize else 'nom'}"
    best[key] = (0, None, None)
    for dt in RL_DTS:
        for scale in SCALES:
            mean_h, std_h = evaluate(weights, randomize, scale, dt, N_EPS)
            rand_label = "yes" if randomize else "no"
            print(f"{name:<10} {rand_label:<6} {scale:<7.3f} {dt:<7.3f} {mean_h:>9.2f} {std_h:>6.2f}")
            if mean_h > best[key][0]:
                best[key] = (mean_h, scale, dt)

print("\n=== Best configs ===")
for key, (hits, scale, dt) in best.items():
    print(f"  {key}: mean_hits={hits:.2f}  scale={scale}  rl_dt={dt}")
