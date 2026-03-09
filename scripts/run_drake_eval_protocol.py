#!/usr/bin/env python3
"""Run a reproducible Drake evaluation protocol with per-episode logging.

Example:
    python scripts/run_drake_eval_protocol.py --episodes 120 --physics both

    python scripts/run_drake_eval_protocol.py \
      --model "Nominal=sac_baseline_final.zip" \
      --model "Robust=sac_robust_final.zip" \
      --episodes 150 \
      --base-seed 20260307
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from src.envs.residual_env import EnvConfig, PingPongResidualEnv


DEFAULT_MODEL_CANDIDATES: tuple[tuple[str, Path], ...] = (
    ("Nominal Final", Path("sac_baseline_final.zip")),
    ("Robust Final", Path("sac_robust_final.zip")),
    ("Nominal Final (models/)", Path("models/sac_nominal_1m/sac_final.zip")),
    ("Robust Final (models/)", Path("models/sac_robust_1m/sac_final.zip")),
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path


def _resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()
    as_zip = Path(f"{raw_path}.zip").expanduser()
    if as_zip.exists():
        return as_zip.resolve()
    raise FileNotFoundError(f"Model not found: {raw_path}")


def _parse_model_specs(raw_specs: list[str] | None) -> list[ModelSpec]:
    if not raw_specs:
        return _discover_default_models()

    specs: list[ModelSpec] = []
    seen: set[Path] = set()
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --model '{raw}'. Use the form LABEL=PATH_TO_MODEL."
            )
        label, raw_path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --model '{raw}': empty label.")
        path = _resolve_model_path(raw_path.strip())
        if path in seen:
            continue
        specs.append(ModelSpec(label=label, path=path))
        seen.add(path)
    return specs


def _discover_default_models() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen: set[Path] = set()
    for label, candidate in DEFAULT_MODEL_CANDIDATES:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        specs.append(ModelSpec(label=label, path=resolved))
        seen.add(resolved)
    return specs


def _build_output_dir(raw_output_dir: str | None, overwrite: bool) -> Path:
    if raw_output_dir:
        out_dir = Path(raw_output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (Path("results") / f"drake_eval_protocol_{stamp}").resolve()

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory exists and is not empty: {out_dir}. "
            "Use --overwrite to allow this."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _z_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    sem = std / max(np.sqrt(float(values.size)), 1.0)
    margin = 1.96 * sem
    return mean - margin, mean + margin


def _summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.array([r["reward"] for r in rows], dtype=float)
    hits = np.array([r["hits"] for r in rows], dtype=float)
    lengths = np.array([r["steps"] for r in rows], dtype=float)
    sim_times = np.array([r["sim_time"] for r in rows], dtype=float)
    survived = np.array([1.0 if r["survived"] else 0.0 for r in rows], dtype=float)

    hits_ci = _z_interval(hits)
    reward_ci = _z_interval(rewards)
    time_ci = _z_interval(sim_times)

    return {
        "n_episodes": int(len(rows)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=0)),
        "mean_hits": float(np.mean(hits)),
        "std_hits": float(np.std(hits, ddof=0)),
        "max_hits": int(np.max(hits)),
        "min_hits": int(np.min(hits)),
        "mean_length": float(np.mean(lengths)),
        "mean_sim_time": float(np.mean(sim_times)),
        "survival_rate": float(np.mean(survived)),
        "mean_reward_ci95": [float(reward_ci[0]), float(reward_ci[1])],
        "mean_hits_ci95": [float(hits_ci[0]), float(hits_ci[1])],
        "mean_sim_time_ci95": [float(time_ci[0]), float(time_ci[1])],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_single_condition(
    *,
    env: "PingPongResidualEnv",
    model: Any,
    model_label: str,
    model_path: Path,
    physics: str,
    seeds: list[int],
    deterministic: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for episode_idx, seed in enumerate(seeds, start=1):
        # Ensure deterministic environment randomization across reruns.
        random.seed(seed)
        np.random.seed(seed)
        obs, info = env.reset(seed=seed)

        done = False
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        row = {
            "model_label": model_label,
            "model_path": str(model_path),
            "physics": physics,
            "episode": episode_idx,
            "seed": seed,
            "reward": total_reward,
            "hits": int(info.get("episode_hits", 0)),
            "steps": steps,
            "sim_time": float(info.get("sim_time", 0.0)),
            "survived": bool(truncated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "fsm_state_final": str(info.get("fsm_state", "")),
            "fsm_hit_count": float(info.get("fsm_metrics", {}).get("hit_count", 0.0)),
            "scenario_yaml": str(getattr(env, "_scenario_yaml", "")),
        }
        rows.append(row)

        print(
            f"  Ep {episode_idx:4d}/{len(seeds)} | seed={seed} | "
            f"hits={row['hits']:4d} | reward={row['reward']:9.2f} | "
            f"time={row['sim_time']:.2f}s | survived={'Y' if row['survived'] else 'N'}"
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible Drake evaluation protocol with fixed seeds "
            "and full per-episode logs."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec in the form LABEL=PATH. Repeat for multiple models. "
            "If omitted, the script auto-discovers default final models."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=120,
        help="Episodes per (model, physics) condition.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260307,
        help="Base episode seed. Episode i uses base_seed + i.",
    )
    parser.add_argument(
        "--physics",
        choices=("nominal", "randomized", "both"),
        default="both",
        help="Physics settings to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="SAC inference device passed to stable_baselines3 (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--target-apex",
        type=float,
        default=0.55,
        help="Target apex height for evaluation env.",
    )
    parser.add_argument(
        "--ball-init-noise",
        type=float,
        default=0.0,
        help="Initial ball position noise at reset.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2000,
        help="Episode horizon in control steps.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Defaults to results/drake_eval_protocol_<timestamp>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError(
            "stable_baselines3 is required. Run from your project virtualenv."
        ) from exc
    try:
        from src.envs.residual_env import EnvConfig, PingPongResidualEnv
    except Exception as exc:
        raise RuntimeError(
            "Could not import Drake residual environment. "
            "Run from your project virtualenv with Drake/gym dependencies installed."
        ) from exc

    model_specs = _parse_model_specs(args.model)
    if not model_specs:
        raise FileNotFoundError(
            "No models found. Provide --model LABEL=PATH or place final zips in default paths."
        )

    physics_modes = ["nominal", "randomized"] if args.physics == "both" else [args.physics]
    seeds = [args.base_seed + idx for idx in range(args.episodes)]
    out_dir = _build_output_dir(args.output_dir, overwrite=args.overwrite)
    run_started = datetime.now().isoformat(timespec="seconds")
    wall_t0 = time.time()

    metadata = {
        "script": Path(__file__).name,
        "run_started": run_started,
        "git_commit": _get_git_commit(),
        "models": [{"label": m.label, "path": str(m.path)} for m in model_specs],
        "physics_modes": physics_modes,
        "episodes_per_condition": args.episodes,
        "base_seed": args.base_seed,
        "seeds": seeds,
        "deterministic_actions": args.deterministic,
        "device": args.device,
        "env_config": {
            "target_apex_height": args.target_apex,
            "ball_init_pos_noise": args.ball_init_noise,
            "max_episode_steps": args.max_episode_steps,
        },
    }
    _write_json(out_dir / "metadata.json", metadata)

    all_episode_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    for spec in model_specs:
        print(f"\n=== Loading model: {spec.label} ===")
        print(f"Path: {spec.path}")
        model = SAC.load(str(spec.path), device=args.device)

        for physics in physics_modes:
            use_randomization = physics == "randomized"
            env_cfg = EnvConfig(
                target_apex_height=args.target_apex,
                ball_init_pos_noise=args.ball_init_noise,
                max_episode_steps=args.max_episode_steps,
                use_randomization=use_randomization,
            )
            env = PingPongResidualEnv(config=env_cfg)
            print(
                f"\n--- Evaluating {spec.label} | physics={physics} | "
                f"episodes={args.episodes} ---"
            )

            rows = _evaluate_single_condition(
                env=env,
                model=model,
                model_label=spec.label,
                model_path=spec.path,
                physics=physics,
                seeds=seeds,
                deterministic=args.deterministic,
            )
            env.close()

            summary = _summarize_episode_rows(rows)
            summary["model_label"] = spec.label
            summary["model_path"] = str(spec.path)
            summary["physics"] = physics
            all_summaries.append(summary)
            all_episode_rows.extend(rows)

            print(
                f"Summary: mean_hits={summary['mean_hits']:.2f} ± {summary['std_hits']:.2f} "
                f"(95% CI {summary['mean_hits_ci95'][0]:.2f}, "
                f"{summary['mean_hits_ci95'][1]:.2f}) | "
                f"survival={summary['survival_rate']:.1%}"
            )

        del model
        if args.device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    elapsed = time.time() - wall_t0
    report = {
        "run_started": run_started,
        "run_finished": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": elapsed,
        "n_conditions": len(all_summaries),
        "n_episode_rows": len(all_episode_rows),
        "summary": all_summaries,
    }

    _write_json(out_dir / "summary.json", report)
    _write_jsonl(out_dir / "per_episode.jsonl", all_episode_rows)
    _write_csv(out_dir / "per_episode.csv", all_episode_rows)

    print("\n=== Evaluation Complete ===")
    print(f"Output directory : {out_dir}")
    print(f"Summary JSON     : {out_dir / 'summary.json'}")
    print(f"Per-episode JSONL: {out_dir / 'per_episode.jsonl'}")
    print(f"Per-episode CSV  : {out_dir / 'per_episode.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
