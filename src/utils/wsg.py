from __future__ import annotations

from pydrake.systems.primitives import ConstantVectorSource

DEFAULT_WSG_HOLD_POSITION = 0.005
DEFAULT_WSG_FORCE_LIMIT = 80.0


def maybe_connect_wsg_hold(
    builder,
    station,
    position=DEFAULT_WSG_HOLD_POSITION,
    force_limit=DEFAULT_WSG_FORCE_LIMIT,
):
    """Connect constant gripper commands when WSG ports are available."""
    try:
        position_port = station.GetInputPort("wsg.position")
    except Exception:
        return False

    position_source = builder.AddSystem(ConstantVectorSource([float(position)]))
    builder.Connect(position_source.get_output_port(), position_port)

    try:
        force_port = station.GetInputPort("wsg.force_limit")
    except Exception:
        return True

    force_source = builder.AddSystem(ConstantVectorSource([float(force_limit)]))
    builder.Connect(force_source.get_output_port(), force_port)
    return True
