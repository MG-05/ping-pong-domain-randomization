# MuJoCo Domain Validation

This folder provides a MuJoCo validation domain for Drake-trained residual policies.

The evaluation protocol is designed for fairness:
- FSM and RL are both evaluated through the same residual wrapper.
- FSM baseline is implemented as zero residual action (`action = 0`), not a separate control stack.
- Shared settings are enforced across all conditions (timing, spawn noise, gains, episode horizon).
- Policy residual authority defaults to Drake training values (`residual_scale=0.5`, `max_residual_rad=0.15`).

## Core Files
- `models/iiwa_wsg_paddle_ball.xml`: MuJoCo scene/model.
- `fsm_ik_env.py`: MuJoCo + Drake FSM/IK bridge.
- `residual_env_mujoco.py`: Residual RL wrapper with matched observation/action contract.
- `run_mujoco_eval_protocol.py`: Main domain-validation protocol runner.
- `plot_mujoco_eval.py`: Legacy plotting utility (includes randomized figures).
- `plot_domain_validation_figures.py`: Nominal-only figure pipeline + residual-scale sweep.
- `tune_robust_residual_scale.py`: Robust-only residual-scale tuner (find best scale).

## Run Domain Validation
From repo root:

```bash
python -m mujoco_transfer.run_mujoco_eval_protocol \
  --episodes 500 \
  --nominal-model sac_baseline_final.zip \
  --robust-model sac_robust_final.zip
```

For nominal-only comparison (FSM vs Nominal vs Robust):

```bash
python -m mujoco_transfer.run_mujoco_eval_protocol \
  --episodes 500 \
  --nominal-model sac_baseline_final.zip \
  --robust-model sac_robust_final.zip \
  --residual-scale 0.01 \
  --nominal-only
```

This writes `metadata.json`, `per_episode.csv`, and `summary.json` under:
- `results/mujoco_eval_protocol_<timestamp>/`

Generate plots:

```bash
python -m mujoco_transfer.plot_mujoco_eval --dir results/mujoco_eval_protocol_<timestamp>
```

Generate domain-validation figures (nominal MuJoCo only, randomized results ignored):

```bash
python -m mujoco_transfer.plot_domain_validation_figures \
  --dir results/mujoco_eval_protocol_<timestamp> \
  --report-name drake_to_mujoco_nominal_validation \
  --sweep-episodes 20
```

Outputs are written to:
- `results/domain_validation_reports/<report-name>/figures/`
- Includes `residual_scale_sweep.png` and `residual_scale_sweep.csv`.

Tune robust residual scale only (find best scale by mean hits):

```bash
python -m mujoco_transfer.tune_robust_residual_scale \
  --episodes 200 \
  --scales "0.006,0.008,0.01,0.012,0.015,0.02" \
  --robust-model sac_robust_final.zip \
  --tag robust_nominal_tune
```

Outputs are written to:
- `results/residual_scale_tuning/robust_residual_tuning_<timestamp>_<tag>/`
- Includes `robust_residual_scale_sweep.csv`, `best_robust_residual_scale.json`, and `robust_residual_scale_sweep.png`.
- Default selection objective is transfer margin: `robust - max(FSM, nominal)`.

## Notes
- MuJoCo is still not a perfect Drake clone; this is a validation domain, not identity simulation.
- Randomized evaluation in `residual_env_mujoco.py` now perturbs ball, paddle, and floor parameters (MuJoCo proxy mapping for restitution/dissipation).
- `run_fsm_ik_mujoco.py` remains useful for low-level bring-up/debug, but not as the primary fairness baseline for policy-vs-FSM comparisons.
