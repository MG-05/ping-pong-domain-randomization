# MuJoCo Transfer (FSM/IK Only)

This folder is a MuJoCo-only transfer scaffold for baseline evaluation before RL transfer.

It includes:
- A MuJoCo world with `iiwa` + fixed `wsg` + `paddle` + `ball` + `floor`.
- Drake iiwa visual meshes (`models/assets/*.obj`) so the arm appearance matches iiwa (instead of a placeholder arm).
- Drake Schunk WSG meshes (`wsg_body.obj`, `wsg_finger_with_tip.obj`) so the end-effector is an actual gripper model.
- A runner that executes the existing Drake `FSMController` IK policy in MuJoCo.
- Episode-level metrics for controller performance (`hits`, `fsm_hits`, `plans`, `IK successes`, `sim_time`).

## Files
- `models/iiwa_wsg_paddle_ball.xml`: MuJoCo MJCF scene/model.
- `models/assets/`: iiwa mesh assets copied from Drake's iiwa description.
- `fsm_ik_env.py`: MuJoCo simulation wrapper + Drake FSM/IK policy bridge.
- `run_fsm_ik_mujoco.py`: CLI script for multi-episode evaluation.

## Run
From repo root:

```bash
pip install mujoco
python -m mujoco_transfer.run_fsm_ik_mujoco --episodes 5
```

Render in viewer:

```bash
python -m mujoco_transfer.run_fsm_ik_mujoco --episodes 1 --render --realtime
```

Replay after termination (slow-motion loops):

```bash
python -m mujoco_transfer.run_fsm_ik_mujoco \
  --episodes 1 \
  --render \
  --replay-after \
  --replay-speed 0.25 \
  --replay-loops 3
```

Tune PID against Drake deterministic baseline targets:

```bash
python -m mujoco_transfer.tune_pid_against_drake \
  --iterations 80 \
  --episodes 3 \
  --out mujoco_transfer/pid_tuning_results.json
```

## Current Tuned PID Defaults
- `kp=3500.00`
- `kd=14.00`
- `torque_limit=400.00`
- `use_bias_compensation=False`
- `ball_init_pos=(0.77, -0.03, 0.70)`
- `ball_init_pos_noise=0.0`
- `hit_debounce_s=0.008`

These are now the defaults in `MujocoFsmIkConfig` and `run_fsm_ik_mujoco.py`.
They are the best-known deterministic Drake-target PID settings found by
`tune_pid_against_drake.py` in this transfer setup.

## Important Notes
- This is intentionally **FSM/IK-only** (no RL policy loading yet).
- The IK planner still comes from Drake (`src/controllers/fsm_controller.py`).
- MuJoCo contact force is reduced to a scalar magnitude proxy for hit detection.
- `hits` reports debounced contact-event hits (`--hit-threshold`, `--hit-min-z`, `--hit-debounce`);
  `fsm_hits` reports the strict FSM impact counter.
- The current MuJoCo robot is a practical transfer model, not a strict mesh/inertia clone of Drake's station model.
- Default CLI parameters are tuned for bring-up (to ensure FSM planning is exercised in MuJoCo).

## What You Still Need To Do
1. Calibrate MuJoCo kinematics/dynamics against Drake (joint axis conventions, inertias, contact settings).
2. Verify control gains (`--kp`, `--kd`, `--torque-limit`) so the arm tracks IK trajectories without oscillation.
3. Tune ball spawn/task setup (`--ball-x --ball-y --ball-z --ball-noise`) to match your intended test distribution.
4. Validate hit detection thresholds under MuJoCo contacts (currently scalar-force proxy).
5. Run baseline sweeps and save results to a JSON/CSV report.
6. After the RL model is finalized, plug policy inference into this env (same observation/action contract as Drake residual env).
