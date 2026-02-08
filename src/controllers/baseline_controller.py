from __future__ import annotations

import numpy as np
from pydrake.all import BasicVector, LeafSystem

DEFAULT_Q0 = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])


class BaselineController(LeafSystem):
    """Simple posture-hold controller with an optional swing mode.

    Input port:
      - iiwa_state: [q0..q6, v0..v6]
    Output port:
      - iiwa_position_command: desired joint positions

    TODO: implement intercept prediction + IK solve.
    TODO: add contact-aware paddle alignment.
    """

    def __init__(
        self,
        num_joints: int = 7,
        q0: np.ndarray | None = None,
        swing: bool = False,
        swing_joint: int = 1,
        swing_amplitude: float = 0.3,
        swing_frequency_hz: float = 0.5,
    ) -> None:
        super().__init__()
        self._num_joints = num_joints
        self._q0 = np.array(q0) if q0 is not None else DEFAULT_Q0.copy()
        self._swing = swing
        self._swing_joint = swing_joint
        self._swing_amplitude = swing_amplitude
        self._swing_frequency_hz = swing_frequency_hz

        self.DeclareVectorInputPort("iiwa_state", BasicVector(self._num_joints * 2))
        self.DeclareVectorOutputPort(
            "iiwa_position_command", BasicVector(self._num_joints), self._calc_output
        )

    def _calc_output(self, context, output) -> None:
        q_des = self._q0.copy()
        if self._swing:
            t = context.get_time()
            q_des[self._swing_joint] += self._swing_amplitude * np.sin(
                2.0 * np.pi * self._swing_frequency_hz * t
            )
        output.SetFromVector(q_des)
