"""Gymnasium environment for ping-pong with a residual RL policy on top of the FSM controller.

Architecture:
  - The FSM/IK controller runs as a standalone "inner loop", producing base joint
    commands at each control step.
  - The RL agent (outer loop) outputs small residual corrections (Δq) that are
    added to the FSM command:  q_cmd = q_base + residual_scale * clip(action).
  - The Drake simulation runs at physics_dt (0.001 s); the RL agent acts every
    control_dt (default 0.01 s).

Observation (20-D):
    [iiwa_q (7), iiwa_v (7), ball_xyz (3), ball_vxyz (3)]
    Indices: iiwa_q=0:7, iiwa_v=7:14, ball_x=14, ball_y=15, ball_z=16,
             ball_vx=17, ball_vy=18, ball_vz=19

Action (7-D):
    Residual joint-position correction, clipped to [-max_residual, +max_residual].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from pydrake.all import DiagramBuilder, Simulator
from src.utils.randomization import DomainRandomizer

from src.controllers.fsm_controller import (
    BALL_CONTACT_FORCE_SIZE,
    BALL_STATE_SIZE,
    FSMController,
    FsmTimingConfig,
)
from src.station import get_iiwa_default_joint_positions, make_station
from src.utils.paths import scenario_path
from src.utils.wsg import maybe_connect_wsg_hold

IIWA_NUM_JOINTS = 7
OBS_DIM = IIWA_NUM_JOINTS * 2 + 3 + 3  # 20

IDX_BALL_X = 14
IDX_BALL_Y = 15
IDX_BALL_Z = 16
IDX_BALL_VX = 17
IDX_BALL_VY = 18
IDX_BALL_VZ = 19


@dataclass
class EnvConfig:
    """Tunable environment hyper-parameters."""

    control_dt: float = 0.01
    max_episode_steps: int = 2000
    target_apex_height: float = 0.55
    max_residual_rad: float = 0.15
    residual_scale: float = 0.5
    ball_init_pos: tuple[float, float, float] = (0.77, -0.03, 0.70)
    ball_init_pos_noise: float = 0.04
    min_ball_height: float = 0.08
    reward_alive: float = 0.01
    reward_hit: float = 5.0
    reward_apex: float = 2.0
    penalty_drop: float = 5.0
    use_randomization: bool = True


class PingPongResidualEnv(gym.Env):
    """Drake-based ping-pong env with FSM inner loop and residual RL outer loop."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario_yaml: str | None = None,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config or EnvConfig()
        self._randomizer = DomainRandomizer()
        # Keep track of the nominal YAML for the FSM controller
        self._nominal_yaml = scenario_yaml or str(scenario_path())
        self._scenario_yaml = self._nominal_yaml
        self._render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(IIWA_NUM_JOINTS,), dtype=np.float64
        )

        self._q0_iiwa = get_iiwa_default_joint_positions(self._scenario_yaml)
        if self._q0_iiwa is None:
            self._q0_iiwa = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])

        self._fsm = FSMController(
            q0=self._q0_iiwa,
            scenario_yaml=self._scenario_yaml,
            timing=FsmTimingConfig(),
        )
        self._fsm_context = self._fsm.CreateDefaultContext()

        self._simulator: Simulator | None = None
        self._diagram = None
        self._station = None
        self._plant = None
        self._iiwa_instance = None
        self._ball_instance = None
        self._action_port_idx: int = 0
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._prev_ball_vz: float = 0.0
        self._prev_ball_z: float = 0.0
        self._episode_hits: int = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self._cfg.use_randomization:
            self._scenario_yaml = self._randomizer.generate_randomized_scenario()
        else:
            self._scenario_yaml = self._nominal_yaml
        self._build_sim()
        self._set_initial_conditions()
        self._fsm.reset()
        self._sim_time = 0.0
        self._step_count = 0
        self._episode_hits = 0
        self._prev_ball_vz = 0.0
        self._prev_ball_z = self._cfg.ball_init_pos[2]
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=float).flatten()[:IIWA_NUM_JOINTS]
        residual = (
            self._cfg.residual_scale
            * self._cfg.max_residual_rad
            * np.clip(action, -1.0, 1.0)
        )

        q_base = self._eval_fsm()
        q_cmd = np.clip(
            q_base + residual,
            self._fsm._iiwa_q_lower,
            self._fsm._iiwa_q_upper,
        )

        ctx = self._simulator.get_mutable_context()
        self._diagram.get_input_port(self._action_port_idx).FixValue(ctx, q_cmd)

        try:
            self._simulator.AdvanceTo(self._sim_time + self._cfg.control_dt)
        except RuntimeError:
            obs = self._get_obs()
            return obs, -self._cfg.penalty_drop, True, False, self._get_info()
        self._sim_time += self._cfg.control_dt
        self._step_count += 1

        obs = self._get_obs()
        reward, terminated = self._compute_reward_and_done(obs)
        truncated = self._step_count >= self._cfg.max_episode_steps

        self._prev_ball_z = float(obs[IDX_BALL_Z])
        self._prev_ball_vz = float(obs[IDX_BALL_VZ])

        return obs, reward, terminated, truncated, self._get_info()

    def close(self) -> None:
        self._simulator = None
        self._diagram = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sim(self) -> None:
        builder = DiagramBuilder()
        self._station = builder.AddSystem(
            make_station(self._scenario_yaml, meshcat=None, lcm=None)
        )
        maybe_connect_wsg_hold(builder, self._station)

        self._action_port_idx = builder.ExportInput(
            self._station.GetInputPort("iiwa.position"), "action"
        )
        builder.ExportOutput(
            self._station.GetOutputPort("iiwa.state_estimated"), "iiwa_state"
        )
        builder.ExportOutput(
            self._station.GetOutputPort("ball.state_estimated"), "ball_state"
        )
        builder.ExportOutput(
            self._station.GetOutputPort("ball.contact_forces"),
            "ball_contact_forces",
        )

        self._diagram = builder.Build()
        self._simulator = Simulator(self._diagram)

        self._plant = self._station.GetSubsystemByName("plant")
        self._iiwa_instance = self._plant.GetModelInstanceByName("iiwa")
        self._ball_instance = self._plant.GetModelInstanceByName("ball")

    def _set_initial_conditions(self) -> None:
        self._simulator.Initialize()
        ctx = self._simulator.get_mutable_context()
        plant_ctx = self._plant.GetMyMutableContextFromRoot(ctx)

        iiwa_state = np.concatenate([self._q0_iiwa, np.zeros(IIWA_NUM_JOINTS)])
        self._plant.SetPositionsAndVelocities(
            plant_ctx, self._iiwa_instance, iiwa_state
        )

        bx, by, bz = self._cfg.ball_init_pos
        noise = self._cfg.ball_init_pos_noise
        rng = self.np_random
        ball_xyz = np.array([
            bx + rng.uniform(-noise, noise),
            by + rng.uniform(-noise, noise),
            bz + rng.uniform(-noise * 0.5, noise),
        ])
        ball_quat = np.array([1.0, 0.0, 0.0, 0.0])
        ball_state = np.concatenate([ball_quat, ball_xyz, np.zeros(6)])
        self._plant.SetPositionsAndVelocities(
            plant_ctx, self._ball_instance, ball_state
        )

        self._diagram.get_input_port(self._action_port_idx).FixValue(
            ctx, self._q0_iiwa
        )

    def _get_obs(self) -> np.ndarray:
        ctx = self._simulator.get_mutable_context()
        iiwa_state = np.asarray(
            self._diagram.GetOutputPort("iiwa_state").Eval(ctx), dtype=float
        )
        ball_state_full = np.asarray(
            self._diagram.GetOutputPort("ball_state").Eval(ctx), dtype=float
        )
        ball_pos = ball_state_full[4:7]
        ball_vel = ball_state_full[10:13]
        return np.concatenate([iiwa_state, ball_pos, ball_vel])

    def _get_ball_contact_forces(self) -> np.ndarray:
        ctx = self._simulator.get_mutable_context()
        return np.asarray(
            self._diagram.GetOutputPort("ball_contact_forces").Eval(ctx),
            dtype=float,
        )

    def _eval_fsm(self) -> np.ndarray:
        """Evaluate the standalone FSM controller at the current sim state."""
        ctx = self._simulator.get_mutable_context()
        iiwa_state = np.asarray(
            self._diagram.GetOutputPort("iiwa_state").Eval(ctx), dtype=float
        )
        ball_state_full = np.asarray(
            self._diagram.GetOutputPort("ball_state").Eval(ctx), dtype=float
        )
        contact_forces = self._get_ball_contact_forces()

        self._fsm_context.SetTime(self._sim_time)
        self._fsm.get_input_port(0).FixValue(self._fsm_context, iiwa_state)
        self._fsm.get_ball_input_port().FixValue(
            self._fsm_context, ball_state_full
        )
        self._fsm.get_ball_contact_force_input_port().FixValue(
            self._fsm_context, contact_forces
        )
        return np.asarray(
            self._fsm.get_output_port(0).Eval(self._fsm_context), dtype=float
        )

    def _compute_reward_and_done(
        self, obs: np.ndarray
    ) -> tuple[float, bool]:
        ball_z = float(obs[IDX_BALL_Z])
        ball_vz = float(obs[IDX_BALL_VZ])
        cfg = self._cfg
        reward = 0.0

        if ball_z > cfg.min_ball_height:
            reward += cfg.reward_alive

        hit = self._detect_hit()
        if hit:
            self._episode_hits += 1
            reward += cfg.reward_hit

        if self._detect_apex(ball_z, ball_vz):
            apex_err = abs(ball_z - cfg.target_apex_height)
            reward += cfg.reward_apex * np.exp(-10.0 * apex_err ** 2)

        terminated = ball_z < cfg.min_ball_height
        if terminated:
            reward -= cfg.penalty_drop

        return reward, terminated

    def _detect_hit(self) -> bool:
        forces = self._get_ball_contact_forces()
        force_mag = float(np.linalg.norm(forces))
        if force_mag < 0.05:
            return False
        ctx = self._simulator.get_mutable_context()
        ball_state = np.asarray(
            self._diagram.GetOutputPort("ball_state").Eval(ctx), dtype=float
        )
        ball_z = float(ball_state[6])
        return ball_z > 0.20

    def _detect_apex(self, ball_z: float, ball_vz: float) -> bool:
        if self._prev_ball_vz > 0.05 and ball_vz <= 0.05 and ball_z > 0.25:
            return True
        return False

    def _get_info(self) -> dict:
        return {
            "sim_time": self._sim_time,
            "episode_hits": self._episode_hits,
            "fsm_state": self._fsm.get_state_name(),
            "fsm_metrics": self._fsm.get_metrics(),
        }
