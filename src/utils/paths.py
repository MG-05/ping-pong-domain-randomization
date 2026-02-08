from __future__ import annotations

from pathlib import Path
import tempfile

REPO_ROOT_TOKEN = "{{REPO_ROOT}}"


def repo_root(start: Path | None = None) -> Path:
    """Walk upward until a repo marker is found."""
    if start is None:
        start = Path(__file__).resolve()

    current = start
    for current in [current, *current.parents]:
        if (current / "README.md").exists() or (current / "pyproject.toml").exists():
            return current

    raise RuntimeError(f"Could not find repo root from: {start}")


def repo_path(*parts: str) -> Path:
    return repo_root() / Path(*parts)


def scenario_path(name: str = "iiwa_wsg_paddle_ball.yaml") -> Path:
    return repo_path("configs", "scenarios", name)


def paddle_urdf_path() -> Path:
    return repo_path("models", "paddle", "paddle.urdf")


def ball_sdf_path() -> Path:
    return repo_path("models", "ball", "ball.sdf")


def load_and_patch_yaml(path: str | Path) -> str:
    yaml_text = Path(path).read_text(encoding="utf-8")
    return yaml_text.replace(REPO_ROOT_TOKEN, str(repo_root()))


def write_patched_yaml(path: str | Path) -> Path:
    """Write a patched YAML to a temp location and return its path."""
    patched = load_and_patch_yaml(path)
    tmp_dir = Path(tempfile.gettempdir()) / "drake_scenarios"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / Path(path).name
    tmp_path.write_text(patched, encoding="utf-8")
    return tmp_path
