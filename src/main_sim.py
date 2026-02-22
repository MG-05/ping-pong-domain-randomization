from __future__ import annotations

import argparse
from pathlib import Path

from pydrake.all import DiagramBuilder, Simulator, StartMeshcat
from pydrake.lcm import DrakeLcm

from src.controllers.baseline_controller import BaselineController
from src.controllers.fsm_controller import FSMController
from src.station import get_iiwa_default_joint_positions, make_station
from src.utils.paths import scenario_path
from src.utils.wsg import maybe_connect_wsg_hold
from src.utils.randomization import DomainRandomizer


def _connect_optional_ball_ports(builder, station, controller) -> None:
    if hasattr(controller, "get_ball_input_port"):
        try:
            ball_state_port = station.GetOutputPort("ball.state_estimated")
            builder.Connect(ball_state_port, controller.get_ball_input_port())
        except Exception:
            pass
    if hasattr(controller, "get_ball_contact_force_input_port"):
        try:
            contact_force_port = station.GetOutputPort("ball.contact_forces")
            builder.Connect(
                contact_force_port, controller.get_ball_contact_force_input_port()
            )
        except Exception:
            pass


def build_diagram(
    scenario_yaml: str, meshcat: bool, lcm=None, controller_type: str = "fsm"
):
    builder = DiagramBuilder()
    meshcat_instance = StartMeshcat() if meshcat else None

    station = builder.AddSystem(
        make_station(scenario_yaml, meshcat=meshcat_instance, lcm=lcm)
    )
    q0 = get_iiwa_default_joint_positions(scenario_yaml)
    if controller_type == "baseline":
        controller = builder.AddSystem(BaselineController(q0=q0))
    elif controller_type == "fsm":
        controller = builder.AddSystem(FSMController(q0=q0, scenario_yaml=scenario_yaml))
    else:
        raise ValueError(
            f"Unknown controller type: {controller_type}. "
            "Expected one of: baseline, fsm."
        )

    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"), controller.get_input_port(0)
    )
    _connect_optional_ball_ports(builder=builder, station=station, controller=controller)
    builder.Connect(
        controller.get_output_port(0), station.GetInputPort("iiwa.position")
    )
    maybe_connect_wsg_hold(builder, station)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    return diagram, simulator, station, meshcat_instance


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Drake ping-pong sim.")
    parser.add_argument(
        "--scenario",
        default=str(scenario_path()),
        help="Path to a ModelDirectives scenario YAML.",
    )
    parser.add_argument("--meshcat", action="store_true", help="Enable Meshcat.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--controller",
        choices=("fsm", "baseline"),
        default="fsm",
        help="Controller to run.",
    )
    parser.add_argument(
        "--no-lcm", action="store_true", help="Disable LCM status publishers."
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable Meshcat recording when meshcat is enabled.",
    )
    parser.add_argument(
        "--record-path",
        default="logs/meshcat.html",
        help="Where to write the Meshcat HTML recording.",
    )
    
    parser.add_argument(
        "--randomize", 
        action=argparse.BooleanOptionalAction, 
        default=False,
        help="Enable domain randomization to visually debug physics changes."
    )
    
    args = parser.parse_args()

    # Swap the scenario YAML if randomization is enabled
    scenario_to_run = args.scenario
    if args.randomize:
        print("Domain Randomization is ENABLED. Generating new physics parameters...")
        randomizer = DomainRandomizer()
        scenario_to_run = randomizer.generate_randomized_scenario()
        print(f"Running randomized simulation using: {scenario_to_run}")
    else:
        print(f"Running nominal simulation using: {scenario_to_run}")

    lcm = None
    if not args.no_lcm:
        try:
            lcm = DrakeLcm()
        except Exception as exc:
            print(f"LCM unavailable ({exc}); continuing without LCM.")
            
    _, simulator, station, meshcat_instance = build_diagram(
        scenario_to_run, args.meshcat, lcm=lcm, controller_type=args.controller
    )

    record = args.meshcat and not args.no_record
    visualizer = None
    if record and meshcat_instance is not None:
        try:
            visualizer = station.GetSubsystemByName("meshcat_visualizer")
            visualizer.StartRecording()
        except Exception:
            visualizer = None

    if args.realtime:
        simulator.set_target_realtime_rate(1.0)

    simulator.Initialize()
    simulator.AdvanceTo(args.duration)

    if visualizer is not None and meshcat_instance is not None:
        visualizer.StopRecording()
        visualizer.PublishRecording()
        output_path = Path(args.record_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(meshcat_instance.StaticHtml(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())