import os
import random
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via temp file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        os.close(fd)
        os.unlink(tmp)
        raise


class DomainRandomizer:
    def __init__(self, output_dir="/tmp/drake_randomized"):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = Path(output_dir)
        self.env = Environment(loader=FileSystemLoader(self.repo_root))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_randomized_scenario(self) -> str:
        params = self._sample_parameters()
        self._write_models(params)
        return self._write_scenario()

    def _sample_parameters(self) -> dict:
        return {
            "ball_mass": random.uniform(0.002, 0.0035),
            "ball_friction": random.uniform(0.15, 0.3),
            "ball_restitution": random.uniform(0.85, 0.95),
            "paddle_mass": random.uniform(0.08, 0.15),
            "paddle_friction": random.uniform(0.2, 0.6),
            "paddle_restitution": random.uniform(0.8, 1.0),
            "floor_dissipation": random.uniform(0.01, 0.1),
            "floor_restitution": random.uniform(0.8, 1.0),
        }

    def _write_models(self, params: dict) -> None:
        templates = [
            ("models/ball/ball.sdf.jinja", "ball_randomized.sdf",
             {k: params[k] for k in ("ball_mass", "ball_friction", "ball_restitution")}),
            ("models/paddle/paddle.urdf.jinja", "paddle_randomized.urdf",
             {k: params[k] for k in ("paddle_mass", "paddle_friction", "paddle_restitution")}),
            ("models/floor/floor.urdf.jinja", "floor_randomized.urdf",
             {k: params[k] for k in ("floor_dissipation", "floor_restitution")}),
        ]
        for tmpl_path, out_name, ctx in templates:
            xml = self.env.get_template(tmpl_path).render(**ctx)
            _atomic_write(self.output_dir / out_name, xml)

    def _write_scenario(self) -> str:
        ball_path = self.output_dir / "ball_randomized.sdf"
        paddle_path = self.output_dir / "paddle_randomized.urdf"
        floor_path = self.output_dir / "floor_randomized.urdf"

        yaml_xml = self.env.get_template(
            "configs/scenarios/iiwa_wsg_paddle_ball.yaml.jinja"
        ).render(
            generated_ball_path=f"file://{ball_path.absolute()}",
            generated_paddle_path=f"file://{paddle_path.absolute()}",
            generated_floor_path=f"file://{floor_path.absolute()}",
        )

        yaml_path = self.output_dir / "scenario_randomized.yaml"
        _atomic_write(yaml_path, yaml_xml)
        return str(yaml_path.absolute())