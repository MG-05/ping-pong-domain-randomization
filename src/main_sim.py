from __future__ import annotations

import argparse
from pathlib import Path

from pydrake.all import DiagramBuilder, Simulator, StartMeshcat
from pydrake.lcm import DrakeLcm

from src.controllers.baseline_controller import BaselineController
from src.station import make_station
from src.utils.paths import scenario_path


def build_diagram(scenario_yaml: str, meshcat: bool, lcm=None):
    builder = DiagramBuilder()
    meshcat_instance = StartMeshcat() if meshcat else None

    station = builder.AddSystem(
        make_station(scenario_yaml, meshcat=meshcat_instance, lcm=lcm)
    )
    controller = builder.AddSystem(BaselineController())

    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"), controller.get_input_port(0)
    )
    builder.Connect(
        controller.get_output_port(0), station.GetInputPort("iiwa.position")
    )

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
    args = parser.parse_args()

    lcm = None
    if not args.no_lcm:
        try:
            lcm = DrakeLcm()
        except Exception as exc:
            print(f"LCM unavailable ({exc}); continuing without LCM.")
    _, simulator, station, meshcat_instance = build_diagram(
        args.scenario, args.meshcat, lcm=lcm
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
