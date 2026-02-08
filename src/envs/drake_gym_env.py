from __future__ import annotations

from pydrake.all import DiagramBuilder, Simulator
try:
    from pydrake.systems.gym import DrakeGymEnv
except ImportError:  # pragma: no cover - fallback for older Drake layouts
    from pydrake.gym import DrakeGymEnv

from src.station import make_station
from src.utils import randomization
from src.utils.paths import scenario_path

DEFAULT_TIME_STEP = 0.01


def _build_simulator(scenario_yaml: str):
    builder = DiagramBuilder()
    station = builder.AddSystem(make_station(scenario_yaml, meshcat=None, lcm=None))

    action_port = builder.ExportInput(
        station.GetInputPort("iiwa.position"), "iiwa_position_command"
    )
    observation_port = builder.ExportOutput(
        station.GetOutputPort("iiwa.state_estimated"), "iiwa_state"
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    return simulator, action_port.get_index(), observation_port.get_index()


def _reward_fn(*_args, **_kwargs) -> float:
    # TODO: add task reward, e.g., ball return success or paddle-ball contact.
    return 0.0


def make_env(render_mode=None) -> DrakeGymEnv:
    """Create a DrakeGymEnv with placeholder ports and reward."""
    scenario_yaml = str(scenario_path())

    _, action_port_id, observation_port_id = _build_simulator(scenario_yaml)

    def simulator_factory(rng):
        simulator, _, _ = _build_simulator(scenario_yaml)
        # TODO: wire domain randomization into plant/scene graph.
        system = simulator.get_system() if hasattr(simulator, "get_system") else simulator
        randomization.apply_domain_randomization(system, rng)
        return simulator

    env = DrakeGymEnv(
        simulator_factory=simulator_factory,
        time_step=DEFAULT_TIME_STEP,
        action_port_id=action_port_id,
        observation_port_id=observation_port_id,
        reward=_reward_fn,
        render_mode=render_mode,
    )
    return env
