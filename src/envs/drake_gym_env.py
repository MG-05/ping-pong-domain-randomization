from __future__ import annotations

from pydrake.all import DiagramBuilder, Simulator
try:
    from pydrake.systems.gym import DrakeGymEnv
except ImportError:  # pragma: no cover - fallback for older Drake layouts
    from pydrake.gym import DrakeGymEnv

from src.station import make_station
from src.utils.randomization import DomainRandomizer
from src.utils.paths import scenario_path
from src.utils.wsg import maybe_connect_wsg_hold

DEFAULT_TIME_STEP = 0.01

def _build_simulator(scenario_yaml: str):
    builder = DiagramBuilder()
    station = builder.AddSystem(make_station(scenario_yaml, meshcat=None, lcm=None))
    maybe_connect_wsg_hold(builder, station)

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

def make_env(render_mode=None, use_randomization=True) -> DrakeGymEnv:
    """Create a DrakeGymEnv with placeholder ports and reward."""
    
    nominal_yaml = str(scenario_path())
    randomizer = DomainRandomizer()

    # The simulator factory is called by DrakeGymEnv on every reset
    def simulator_factory(rng):
        if use_randomization:
            # Generate new randomized files and get the path
            current_yaml = randomizer.generate_randomized_scenario()
        else:
            current_yaml = nominal_yaml
            
        # Build the simulator from scratch using the determined YAML
        simulator, _, _ = _build_simulator(current_yaml)
        return simulator

    # We still need to build it once to get the port IDs
    _, action_port_id, observation_port_id = _build_simulator(nominal_yaml)

    env = DrakeGymEnv(
        simulator_factory=simulator_factory,
        time_step=DEFAULT_TIME_STEP,
        action_port_id=action_port_id,
        observation_port_id=observation_port_id,
        reward=_reward_fn,
        render_mode=render_mode,
    )
    return env