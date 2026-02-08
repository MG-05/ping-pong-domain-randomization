# Drake Ping-Pong Scaffold (iiwa + WSG + paddle + ball)

This repo is a minimal, runnable Python + Drake scaffold for simulating a KUKA iiwa with a Schunk WSG, a simple box paddle, and a free ball.

## Quickstart

```bash
./scripts/setup_env.sh
./scripts/run_sim.sh --meshcat
```

Common options:

```bash
./scripts/run_sim.sh --scenario configs/scenarios/iiwa_wsg_paddle_ball.yaml --duration 10 --realtime
```

## Notes

- Drake model URIs are in the scenario YAML as `package://drake_models/...` and are resolved by Drake.
- Local models (paddle and ball) use a `file://{{REPO_ROOT}}/...` placeholder. The loader in `src/utils/paths.py` patches this to the current repo root at runtime.
- Tune the paddle pose by editing the `add_weld` for `paddle` in `configs/scenarios/iiwa_wsg_paddle_ball.yaml` (it is welded to `wsg::body`).
- LCM status publishers are enabled by default in `main_sim.py`; use `--no-lcm` to disable.
- Meshcat recording is enabled when `--meshcat` is set and saved to `logs/meshcat.html`; use `--no-record` or `--record-path` to override.

## Next steps

- Expose the ball state output (pose/velocity) from the station.
- Implement intercept prediction + IK in `src/controllers/baseline_controller.py`.
- Add reward shaping, domain randomization, and learned residual policies in `src/envs/drake_gym_env.py` + `src/utils/randomization.py`.

## Training stub

```bash
./scripts/train_rl.sh
```

This is a placeholder that wires a `DrakeGymEnv` but does not yet define a meaningful reward or randomization.
