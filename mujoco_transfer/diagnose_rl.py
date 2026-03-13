"""
Diagnostic script to isolate why RL+FSM gets fewer hits than FSM-only.

Tests in order:
  1. FSM-only via MujocoFsmIkEnv (ground truth: ~63 hits)
  2. RL wrapper with residual_scale=0.0 (FSM-only through RL wrapper - infrastructure check)
  3. RL wrapper with residual_scale=0.5, rl_control_dt=0.001 (1ms, old)
  4. RL wrapper with residual_scale=0.5, rl_control_dt=0.01  (10ms, Drake-matching)
  5. RL wrapper with residual_scale=0.1, rl_control_dt=0.01  (small residuals)
"""
import sys
import numpy as np
from stable_baselines3 import SAC

from mujoco_transfer.fsm_ik_env import MujocoFsmIkEnv, MujocoFsmIkConfig
from mujoco_transfer.residual_env_mujoco import MujocoResidualEnv, MujocoResidualConfig

MODEL_PATH = "mujoco_transfer/models/iiwa_wsg_paddle_ball.xml"
WEIGHTS    = "mujoco_transfer/sac_nom_1m.zip"
EPISODES   = 10

# ---------------------------------------------------------------------------
def run_fsm_baseline(n_eps):
    cfg = MujocoFsmIkConfig()
    env = MujocoFsmIkEnv(model_path=MODEL_PATH, config=cfg)
    hits_list, times_list = [], []
    for ep in range(n_eps):
        result = env.run_episode(seed=ep)
        hits_list.append(result["hits"])
        times_list.append(result["sim_time"])
        print(f"  ep {ep+1:2d}: hits={result['hits']:3d}  sim_time={result['sim_time']:.2f}s")
    print(f"  → mean hits: {np.mean(hits_list):.1f}  mean time: {np.mean(times_list):.2f}s\n")
    return hits_list


def run_rl_eval(label, weights_path, residual_scale, rl_control_dt, n_eps):
    cfg = MujocoResidualConfig(
        randomize_dynamics=False,
        residual_scale=residual_scale,
        rl_control_dt=rl_control_dt,
    )
    env = MujocoResidualEnv(model_path=MODEL_PATH, config=cfg)

    if residual_scale == 0.0:
        # Zero-action policy: always output zeros
        def predict(obs):
            return np.zeros(7), None
    else:
        model = SAC.load(weights_path)
        def predict(obs):
            return model.predict(obs, deterministic=True)

    hits_list, rewards_list, times_list = [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        step = 0
        while not done:
            action, _ = predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated
        hits = info.get("episode_hits", 0)
        sim_time = info.get("sim_time", 0)
        hits_list.append(hits)
        rewards_list.append(ep_reward)
        times_list.append(sim_time)
        print(f"  ep {ep+1:2d}: hits={hits:3d}  reward={ep_reward:7.2f}  steps={step:5d}  sim_time={sim_time:.2f}s")

    print(f"  → mean hits: {np.mean(hits_list):.1f}  mean reward: {np.mean(rewards_list):.2f}  mean time: {np.mean(times_list):.2f}s\n")
    return hits_list


# ---------------------------------------------------------------------------
print("=" * 60)
print("TEST 1: FSM-only baseline (MujocoFsmIkEnv)")
print("=" * 60)
run_fsm_baseline(EPISODES)

print("=" * 60)
print("TEST 2: RL wrapper, residual_scale=0.0 (zero action, infrastructure check)")
print("=" * 60)
run_rl_eval("zero_action", WEIGHTS, residual_scale=0.0, rl_control_dt=0.001, n_eps=EPISODES)

print("=" * 60)
print("TEST 3: SAC model, residual_scale=0.5, rl_control_dt=0.001 (1ms, old)")
print("=" * 60)
run_rl_eval("sac_1ms", WEIGHTS, residual_scale=0.5, rl_control_dt=0.001, n_eps=EPISODES)

print("=" * 60)
print("TEST 4: SAC model, residual_scale=0.5, rl_control_dt=0.01 (10ms, Drake-matching)")
print("=" * 60)
run_rl_eval("sac_10ms", WEIGHTS, residual_scale=0.5, rl_control_dt=0.01, n_eps=EPISODES)

print("=" * 60)
print("TEST 5: SAC model, residual_scale=0.1, rl_control_dt=0.01 (small residuals)")
print("=" * 60)
run_rl_eval("sac_10ms_small", WEIGHTS, residual_scale=0.1, rl_control_dt=0.01, n_eps=EPISODES)

print("=" * 60)
print("TEST 6: SAC model, residual_scale=0.05, rl_control_dt=0.01 (tiny residuals)")
print("=" * 60)
run_rl_eval("sac_10ms_tiny", WEIGHTS, residual_scale=0.05, rl_control_dt=0.01, n_eps=EPISODES)
