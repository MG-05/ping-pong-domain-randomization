from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml

import pydrake.all as drake_all
from pydrake.all import (
    BasicVector,
    DiagramBuilder,
    IiwaStatusSender,
    LeafSystem,
    MakeMultibodyStateToWsgStateSystem,
    SchunkWsgStatusSender,
    Simulator,
    StartMeshcat,
)
from pydrake.geometry import MeshcatVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.systems.primitives import (
    ConstantVectorSource,
    Demultiplexer,
    PassThrough,
)

from drake import lcmt_iiwa_status, lcmt_schunk_wsg_status

from src.utils.paths import load_and_patch_yaml
from src.utils.paths import scenario_path as default_scenario_path
from src.utils.paths import write_patched_yaml

_HAS_SCENARIO = hasattr(drake_all, "MakeHardwareStation") and hasattr(
    drake_all, "LoadScenario"
)
MakeHardwareStation = getattr(drake_all, "MakeHardwareStation", None)
LoadScenario = getattr(drake_all, "LoadScenario", None)


class _DrakeYamlLoader(yaml.SafeLoader):
    pass


def _rpy_constructor(loader, node):
    values = loader.construct_sequence(node)
    return [float(v) for v in values]


_DrakeYamlLoader.add_constructor("!Rpy", _rpy_constructor)


def _passthrough_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_DrakeYamlLoader.add_constructor("!IiwaDriver", _passthrough_constructor)
_DrakeYamlLoader.add_constructor("!SchunkWsgDriver", _passthrough_constructor)


def _parse_rpy(rotation) -> RollPitchYaw:
    if rotation is None:
        rpy_deg = [0.0, 0.0, 0.0]
    elif isinstance(rotation, (list, tuple)):
        rpy_deg = rotation
    elif isinstance(rotation, dict):
        rpy_deg = rotation.get("deg") or rotation.get("rpy_deg") or rotation.get("rpy")
        if rpy_deg is None:
            raise ValueError(f"Unsupported rotation mapping: {rotation}")
    else:
        raise ValueError(f"Unsupported rotation type: {type(rotation)}")
    rpy = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    return RollPitchYaw(rpy)


def _parse_transform(data) -> RigidTransform:
    if not data:
        return RigidTransform()
    translation = np.asarray(data.get("translation", [0.0, 0.0, 0.0]), dtype=float)
    rotation = _parse_rpy(data.get("rotation"))
    return RigidTransform(rotation, translation)


def _split_scoped(name: str):
    if "::" in name:
        model_name, frame_name = name.split("::", 1)
        return model_name, frame_name
    return None, name


def _resolve_frame(plant, scoped_name: str):
    if scoped_name in ("world", "world::world"):
        return plant.world_frame()
    model_name, frame_name = _split_scoped(scoped_name)
    if model_name is None:
        raise ValueError(f"Expected scoped frame name, got: {scoped_name}")
    model_instance = plant.GetModelInstanceByName(model_name)
    try:
        return plant.GetFrameByName(frame_name, model_instance)
    except Exception:
        body = plant.GetBodyByName(frame_name, model_instance)
        return body.body_frame()


def _default_body_for_instance(plant, model_instance, model_name: str):
    body_indices = plant.GetBodyIndices(model_instance)
    if len(body_indices) == 1:
        return plant.get_body(body_indices[0])
    for candidate in ("base_link", "ball_link", model_name):
        try:
            return plant.GetBodyByName(candidate, model_instance)
        except Exception:
            continue
    return plant.get_body(body_indices[0])


def _find_drake_models_path():
    import pydrake

    share = (Path(pydrake.__file__).resolve().parent / "share" / "drake")
    candidates = [
        share / "manipulation" / "models",
        share / "models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _configure_package_map(parser: Parser) -> None:
    package_map = parser.package_map()
    if package_map.Contains("drake_models"):
        return
    models_root = _find_drake_models_path()
    if models_root is None:
        return
    package_map.Add("drake_models", str(models_root))


def _resolve_model_path(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://") :]
    if path.startswith("package://"):
        package_name, rel_path = path[len("package://") :].split("/", 1)
        if package_name == "drake_models":
            models_root = _find_drake_models_path()
            if models_root is None:
                raise FileNotFoundError("Could not locate drake_models resources.")
            return str(models_root / rel_path)
    return path


def _add_lcm_publishers(
    builder, plant, iiwa_instance, wsg_instance, position_command_port, lcm
) -> None:
    if lcm is None:
        return

    num_joints = plant.num_positions(iiwa_instance)
    demux = builder.AddSystem(Demultiplexer([num_joints, num_joints]))
    builder.Connect(plant.get_state_output_port(iiwa_instance), demux.get_input_port())

    iiwa_status = builder.AddSystem(IiwaStatusSender(num_joints))
    builder.Connect(
        position_command_port, iiwa_status.GetInputPort("position_commanded")
    )
    builder.Connect(
        demux.get_output_port(0), iiwa_status.GetInputPort("position_measured")
    )
    builder.Connect(
        demux.get_output_port(1), iiwa_status.GetInputPort("velocity_estimated")
    )
    zero_torque = builder.AddSystem(
        ConstantVectorSource(np.zeros(num_joints, dtype=float))
    )
    builder.Connect(
        zero_torque.get_output_port(), iiwa_status.GetInputPort("torque_commanded")
    )
    iiwa_pub = builder.AddSystem(
        LcmPublisherSystem.Make(
            "IIWA_STATUS",
            lcmt_iiwa_status,
            lcm,
            publish_period=0.01,
            use_cpp_serializer=True,
        )
    )
    builder.Connect(iiwa_status.get_output_port(), iiwa_pub.get_input_port())

    if wsg_instance is None:
        return
    wsg_state_to_state = builder.AddSystem(MakeMultibodyStateToWsgStateSystem())
    builder.Connect(
        plant.get_state_output_port(wsg_instance), wsg_state_to_state.get_input_port()
    )
    wsg_status = builder.AddSystem(SchunkWsgStatusSender())
    builder.Connect(wsg_state_to_state.get_output_port(), wsg_status.get_input_port(0))
    wsg_pub = builder.AddSystem(
        LcmPublisherSystem.Make(
            "SCHUNK_WSG_STATUS",
            lcmt_schunk_wsg_status,
            lcm,
            publish_period=0.01,
            use_cpp_serializer=True,
        )
    )
    builder.Connect(wsg_status.get_output_port(), wsg_pub.get_input_port())


class IiwaPositionPDCtrl(LeafSystem):
    """Simple PD controller to drive iiwa positions via joint torques."""

    def __init__(self, plant, iiwa_instance, num_joints, kp=100.0, kd=10.0):
        super().__init__()
        self._plant = plant
        self._iiwa_instance = iiwa_instance
        self._num_joints = num_joints
        self._kp = np.full(num_joints, kp, dtype=float)
        self._kd = np.full(num_joints, kd, dtype=float)

        self.DeclareVectorInputPort("iiwa_position", BasicVector(num_joints))
        self.DeclareVectorInputPort("iiwa_state", BasicVector(num_joints * 2))
        self.DeclareVectorOutputPort(
            "plant_actuation", BasicVector(plant.num_actuators()), self._calc_output
        )

    def _calc_output(self, context, output) -> None:
        q_des = self.get_input_port(0).Eval(context)
        x = self.get_input_port(1).Eval(context)
        q = x[: self._num_joints]
        v = x[self._num_joints :]
        tau = self._kp * (q_des - q) - self._kd * v

        u = np.zeros(self._plant.num_actuators(), dtype=float)
        self._plant.SetActuationInArray(self._iiwa_instance, tau, u)
        output.SetFromVector(u)


def _load_scenario_from_yaml(path: str):
    patched_path = write_patched_yaml(path)
    try:
        return LoadScenario(filename=str(patched_path))
    except TypeError:
        return LoadScenario(str(patched_path))


def _parse_directives_yaml(path: str):
    yaml_text = load_and_patch_yaml(path)
    data = yaml.load(yaml_text, Loader=_DrakeYamlLoader) or {}
    directives = data.get("directives", [])
    plant_config = data.get("plant_config", {})
    return directives, plant_config


def _coerce_scalar(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar or single-element list, got: {value}")
        value = value[0]
    return float(value)


def _iiwa_default_positions_from_mapping(mapping):
    if not mapping:
        return None
    values = []
    for i in range(1, 8):
        key = f"iiwa_joint_{i}"
        if key not in mapping:
            return None
        values.append(_coerce_scalar(mapping[key]))
    return np.asarray(values, dtype=float)


def get_iiwa_default_joint_positions(scenario_yaml: str):
    """Return iiwa default joint positions from a scenario YAML, if present."""
    directives, _ = _parse_directives_yaml(scenario_yaml)
    for directive in directives:
        add_model = directive.get("add_model")
        if not add_model:
            continue
        if add_model.get("name") != "iiwa":
            continue
        return _iiwa_default_positions_from_mapping(
            add_model.get("default_joint_positions")
        )
    return None


def _make_station_fallback(scenario_yaml: str, meshcat=None, lcm=None):
    directives, plant_config = _parse_directives_yaml(scenario_yaml)
    time_step = float(plant_config.get("time_step", 0.001))

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant, scene_graph)
    _configure_package_map(parser)

    model_instances = {}
    pending_welds = []
    pending_joint_positions = []

    for directive in directives:
        if "add_model" in directive:
            add_model = directive["add_model"]
            name = add_model["name"]
            file = _resolve_model_path(add_model["file"])
            model_instances[name] = parser.AddModelFromFile(file, name)

            if "default_free_body_pose" in add_model:
                pose = add_model["default_free_body_pose"]
                base_frame = pose.get("base_frame", "world")
                if base_frame not in ("world", "world::world"):
                    raise ValueError(
                        f"Fallback loader only supports world base_frame, got: {base_frame}"
                    )
                X_WB = _parse_transform(pose)
                body = _default_body_for_instance(
                    plant, model_instances[name], model_name=name
                )
                plant.SetDefaultFreeBodyPose(body, X_WB)
            if "default_joint_positions" in add_model:
                pending_joint_positions.append(
                    {
                        "model_instance": model_instances[name],
                        "model_name": name,
                        "positions": add_model["default_joint_positions"],
                    }
                )
        elif "add_weld" in directive:
            pending_welds.append(directive["add_weld"])

    for weld in pending_welds:
        parent_frame = _resolve_frame(plant, weld["parent"])
        child_frame = _resolve_frame(plant, weld["child"])
        X_PC = _parse_transform(weld.get("X_PC"))
        plant.WeldFrames(parent_frame, child_frame, X_PC)

    plant.Finalize()

    for entry in pending_joint_positions:
        if entry["model_name"] != "iiwa":
            continue
        q0 = _iiwa_default_positions_from_mapping(entry["positions"])
        if q0 is None:
            continue
        plant.SetDefaultPositions(entry["model_instance"], q0)

    iiwa_instance = plant.GetModelInstanceByName("iiwa")
    wsg_instance = None
    if plant.HasModelInstanceNamed("wsg"):
        wsg_instance = plant.GetModelInstanceByName("wsg")
    num_joints = plant.num_positions(iiwa_instance)

    controller = builder.AddSystem(
        IiwaPositionPDCtrl(plant, iiwa_instance, num_joints)
    )
    command_passthrough = builder.AddSystem(PassThrough(num_joints))
    builder.Connect(
        command_passthrough.get_output_port(), controller.get_input_port(0)
    )
    builder.Connect(
        plant.get_state_output_port(iiwa_instance), controller.get_input_port(1)
    )
    builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())
    builder.ExportInput(command_passthrough.get_input_port(), "iiwa.position")
    builder.ExportOutput(
        plant.get_state_output_port(iiwa_instance), "iiwa.state_estimated"
    )

    if meshcat is not None:
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        visualizer.set_name("meshcat_visualizer")

    _add_lcm_publishers(
        builder=builder,
        plant=plant,
        iiwa_instance=iiwa_instance,
        wsg_instance=wsg_instance,
        position_command_port=command_passthrough.get_output_port(),
        lcm=lcm,
    )

    return builder.Build()


def make_station(scenario_yaml: str | None = None, meshcat=None, lcm=None):
    """Create a station system from a scenario YAML."""
    yaml_path = scenario_yaml or str(default_scenario_path())
    if _HAS_SCENARIO:
        scenario = _load_scenario_from_yaml(yaml_path)
        if lcm is None:
            return MakeHardwareStation(scenario, meshcat=meshcat)
        try:
            return MakeHardwareStation(scenario, meshcat=meshcat, lcm=lcm)
        except TypeError:
            return MakeHardwareStation(scenario, meshcat=meshcat)
    return _make_station_fallback(yaml_path, meshcat=meshcat, lcm=lcm)


def make_diagram(scenario_path: str, meshcat: bool, lcm=None):
    """Build a minimal diagram containing only the station.

    Note: main_sim.py builds its own diagram to connect a controller.
    """
    meshcat_instance = StartMeshcat() if meshcat else None
    builder = DiagramBuilder()
    station = builder.AddSystem(
        make_station(scenario_path, meshcat=meshcat_instance, lcm=lcm)
    )
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    return diagram, context, simulator, station


def get_iiwa_position_port(station):
    return station.GetInputPort("iiwa.position")


def get_iiwa_state_port(station):
    return station.GetOutputPort("iiwa.state_estimated")


def get_ball_pose_port(station):
    # TODO: expose a ball pose output port via a custom subsystem.
    return None
