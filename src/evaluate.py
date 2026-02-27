"""Evaluate a trained residual SAC policy (or the pure FSM baseline).

Usage:
    # Evaluate trained policy on nominal physics
    python -m src.evaluate --model data/sac_nominal/sac_final.zip --episodes 20 --no-randomize

    # Evaluate trained policy on randomized (OOD) physics
    python -m src.evaluate --model data/sac_nominal/sac_final.zip --episodes 20 --randomize

    # Evaluate pure FSM (no residual) for baseline comparison
    python -m src.evaluate --fsm-only --episodes 20

    # Visually watch a trained model in Meshcat (single episode, real-time)
    python -m src.evaluate --model data/sac_nominal/sac_final.zip --episodes 1 --render --realtime

    # Watch a specific checkpoint (e.g. 100k steps)
    python -m src.evaluate --model data/sac_nominal/checkpoints/sac_residual_100000_steps.zip --render --realtime

    # Save a Meshcat recording to HTML
    python -m src.evaluate --model data/sac_robust/sac_final.zip --render --record-path logs/robust_demo.html

    # Compare all checkpoints under both physics conditions
    python -m src.evaluate --model-dir data/sac_nominal --episodes 5 --no-randomize
    python -m src.evaluate --model-dir data/sac_nominal --episodes 5 --randomize
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.envs.residual_env import EnvConfig, PingPongResidualEnv

if TYPE_CHECKING:
    from stable_baselines3 import SAC


def _collect_models(args) -> list[tuple[str, Path]]:
    """Build an ordered list of (label, path) pairs from CLI args."""
    if args.fsm_only:
        return [("FSM-only (zero residual)", None)]

    if args.model is not None:
        path = _resolve_model_path(args.model)
        return [(path.stem, path)]

    if args.model_dir is not None:
        return _discover_checkpoints(Path(args.model_dir))

    return []


def _resolve_model_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    if not raw.endswith(".zip") and Path(raw + ".zip").exists():
        return Path(raw + ".zip")
    return path


def _discover_checkpoints(model_dir: Path) -> list[tuple[str, Path]]:
    """Find all saved models in a training directory, sorted by step count."""
    models: list[tuple[int, str, Path]] = []

    final = model_dir / "sac_final.zip"
    if final.exists():
        models.append((float("inf"), "final", final))

    best = model_dir / "best" / "best_model.zip"
    if best.exists():
        models.append((float("inf") - 1, "best", best))

    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for f in sorted(ckpt_dir.glob("*.zip")):
            step_count = _parse_step_count(f.stem)
            label = f"{step_count // 1000}k" if step_count >= 1000 else str(step_count)
            models.append((step_count, label, f))

    models.sort(key=lambda x: x[0])
    return [(label, path) for _, label, path in models]


def _parse_step_count(stem: str) -> int:
    """Extract step count from checkpoint filenames like 'sac_residual_100000_steps'."""
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if part == "steps" and i > 0:
            try:
                return int(parts[i - 1])
            except ValueError:
                pass
    for part in parts:
        try:
            return int(part)
        except ValueError:
            continue
    return 0


def run_evaluation(
    env: PingPongResidualEnv,
    model: "SAC | None",
    n_episodes: int,
    deterministic: bool = True,
) -> dict:
    episode_rewards: list[float] = []
    episode_hits: list[int] = []
    episode_lengths: list[int] = []
    episode_sim_times: list[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_hits.append(info.get("episode_hits", 0))
        episode_lengths.append(steps)
        episode_sim_times.append(info.get("sim_time", 0.0))

        print(
            f"  Episode {ep + 1:3d} | "
            f"reward={total_reward:8.2f} | "
            f"hits={info.get('episode_hits', 0):3d} | "
            f"steps={steps:5d} | "
            f"sim_time={info.get('sim_time', 0.0):.2f}s"
        )

    return {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_hits": float(np.mean(episode_hits)),
        "std_hits": float(np.std(episode_hits)),
        "max_hits": int(np.max(episode_hits)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_sim_time": float(np.mean(episode_sim_times)),
    }


def _print_summary(results: dict) -> None:
    print(f"  Mean reward     : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean hits       : {results['mean_hits']:.1f} ± {results['std_hits']:.1f}")
    print(f"  Max hits        : {results['max_hits']}")
    print(f"  Mean sim time   : {results['mean_sim_time']:.2f}s")
    print(f"  Mean ep length  : {results['mean_length']:.0f} steps")


def _setup_meshcat():
    from pydrake.all import StartMeshcat
    try:
        meshcat = StartMeshcat()
    except Exception as exc:
        print(f"  Meshcat         : unavailable ({exc})")
        return None
    print(f"  Meshcat URL     : {meshcat.web_url()}")
    return meshcat


def _start_recording(env: PingPongResidualEnv):
    vis = env.get_meshcat_visualizer()
    if vis is not None:
        vis.StartRecording()
    return vis


def _stop_and_save_recording(env: PingPongResidualEnv, meshcat, record_path: str | None):
    vis = env.get_meshcat_visualizer()
    if vis is None:
        return
    vis.StopRecording()
    vis.PublishRecording()
    if record_path and meshcat is not None:
        output = Path(record_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(meshcat.StaticHtml(), encoding="utf-8")
        print(f"  Recording saved : {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies or FSM baseline with optional Meshcat visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default=None, help="Path to a single trained SAC model (.zip).")
    source.add_argument(
        "--model-dir", default=None,
        help="Path to a training directory. Discovers and evaluates all checkpoints (final, best, and each saved step).",
    )
    source.add_argument("--fsm-only", action="store_true", help="Evaluate pure FSM (zero residual).")

    parser.add_argument("--episodes", type=int, default=10, help="Episodes per model.")
    parser.add_argument("--target-apex", type=float, default=0.55, help="Target apex height.")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions.")
    parser.add_argument(
        "--randomize", action=argparse.BooleanOptionalAction, default=False,
        help="Enable domain randomization for out-of-distribution evaluation.",
    )

    vis_group = parser.add_argument_group("visualization")
    vis_group.add_argument("--render", action="store_true", help="Enable Meshcat 3D visualization.")
    vis_group.add_argument("--realtime", action="store_true", help="Run simulation at real-time speed (requires --render).")
    vis_group.add_argument("--record-path", default=None, help="Save Meshcat recording as a self-contained HTML file.")

    args = parser.parse_args()

    if not args.fsm_only and args.model is None and args.model_dir is None:
        parser.error("Provide --model PATH, --model-dir DIR, or --fsm-only.")

    models = _collect_models(args)
    if not models:
        parser.error("No models found. Check your --model or --model-dir path.")

    meshcat = _setup_meshcat() if args.render else None
    render_enabled = args.render and meshcat is not None

    config = EnvConfig(
        target_apex_height=args.target_apex,
        ball_init_pos_noise=0.0,
        use_randomization=args.randomize,
    )
    env = PingPongResidualEnv(config=config, meshcat=meshcat)
    if args.realtime:
        env.set_realtime_rate(1.0)

    physics_label = "Randomized (OOD)" if args.randomize else "Nominal"
    print(f"\n{'=' * 60}")
    print(f"  Physics         : {physics_label}")
    print(f"  Episodes/model  : {args.episodes}")
    print(f"  Models to eval  : {len(models)}")
    print(f"  Visualization   : {'Meshcat' if render_enabled else 'Disabled'}")
    print(f"{'=' * 60}\n")

    all_results: list[tuple[str, dict]] = []

    for label, model_path in models:
        sac_model = None
        if model_path is not None:
            if not model_path.exists():
                print(f"[SKIP] {label}: file not found ({model_path})")
                continue
            from stable_baselines3 import SAC
            sac_model = SAC.load(str(model_path))

        print(f"--- {label} ---")
        if model_path:
            print(f"  Path: {model_path}")

        if render_enabled:
            _start_recording(env)

        results = run_evaluation(env, sac_model, args.episodes, deterministic=args.deterministic)

        if render_enabled:
            per_model_record = args.record_path
            if per_model_record and len(models) > 1:
                stem = Path(per_model_record).stem
                suffix = Path(per_model_record).suffix or ".html"
                per_model_record = str(Path(per_model_record).parent / f"{stem}_{label}{suffix}")
            _stop_and_save_recording(env, meshcat, per_model_record)

        print("  Summary:")
        _print_summary(results)
        print()
        all_results.append((label, results))

    if len(all_results) > 1:
        print(f"{'=' * 60}")
        print("  COMPARISON TABLE")
        print(f"{'=' * 60}")
        header = f"{'Model':<25} {'Mean Hits':>10} {'Std':>6} {'Max':>5} {'Mean Reward':>12} {'Sim Time':>9}"
        print(header)
        print("-" * len(header))
        for label, r in all_results:
            print(
                f"{label:<25} {r['mean_hits']:>10.1f} {r['std_hits']:>6.1f} "
                f"{r['max_hits']:>5d} {r['mean_reward']:>12.2f} {r['mean_sim_time']:>8.2f}s"
            )
        print()

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
