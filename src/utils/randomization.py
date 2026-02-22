import random
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class DomainRandomizer:
    def __init__(self, output_dir="/tmp/drake_randomized"):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = Path(output_dir)
        self.env = Environment(loader=FileSystemLoader(self.repo_root))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_randomized_scenario(self):
        # Sample parameters
        ball_mass = random.uniform(0.002, 0.0035) 
        ball_friction = random.uniform(0.15, 0.3)
        ball_restitution = random.uniform(0.85, 0.95)
        
        paddle_mass = random.uniform(0.08, 0.15)
        paddle_friction = random.uniform(0.2, 0.6)
        paddle_restitution = random.uniform(0.8, 1.0)
        
        floor_dissipation = random.uniform(0.01, 0.1)
        floor_restitution = random.uniform(0.8, 1.0)

        # Render models
        ball_xml = self.env.get_template("models/ball/ball.sdf.jinja").render(
            ball_mass=ball_mass, ball_friction=ball_friction, ball_restitution=ball_restitution)
        paddle_xml = self.env.get_template("models/paddle/paddle.urdf.jinja").render(
            paddle_mass=paddle_mass, paddle_friction=paddle_friction, paddle_restitution=paddle_restitution)
        floor_xml = self.env.get_template("models/floor/floor.urdf.jinja").render(
            floor_dissipation=floor_dissipation, floor_restitution=floor_restitution)

        # Save to /tmp
        ball_path = self.output_dir / "ball_randomized.sdf"
        paddle_path = self.output_dir / "paddle_randomized.urdf"
        floor_path = self.output_dir / "floor_randomized.urdf"

        ball_path.write_text(ball_xml)
        paddle_path.write_text(paddle_xml)
        floor_path.write_text(floor_xml)

        # Render and save scenario YAML
        yaml_xml = self.env.get_template("configs/scenarios/iiwa_wsg_paddle_ball.yaml.jinja").render(
            generated_ball_path=f"file://{ball_path.absolute()}",
            generated_paddle_path=f"file://{paddle_path.absolute()}",
            generated_floor_path=f"file://{floor_path.absolute()}"
        )
        
        yaml_path = self.output_dir / "scenario_randomized.yaml"
        yaml_path.write_text(yaml_xml)

        return str(yaml_path.absolute())