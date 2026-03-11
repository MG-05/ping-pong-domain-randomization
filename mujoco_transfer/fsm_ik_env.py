from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import time
from typing import Any
import gymnasium as gym
from gymnasium import spaces

import numpy as np

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

try:
    import mujoco
except ImportError as exc:
    raise ImportError(
        "MuJoCo is required for mujoco_transfer. Install with: pip install mujoco"
    ) from exc

from src.controllers.fsm_controller import (
    BALL_CONTACT_FORCE_SIZE,
    BALL_STATE_SIZE,
    FSMController,
)
from src.station import get_iiwa_default_joint_positions
from src.utils.paths import scenario_path


IIWA_JOINT_NAMES = tuple(f"iiwa_joint_{i}" for i in range(1, 8))
IIWA_ACTUATOR_NAMES = tuple(f"iiwa_torque_{i}" for i in range(1, 8))
DEFAULT_Q0 = np.array([-0.2, 0.79, 0.32, -1.76, -0.36, -0.95, 1.63], dtype=float)


@dataclass
class MujocoFsmIkConfig:
    physics_dt: float = 0.001
    control_dt: float = 0.001  # FSM requires 1ms control for MuJoCo (Drake uses 10ms)
    max_episode_time: float = 20.0
    min_ball_height: float = 0.08
    # Tuned PID gains from pid_tuning_results_control1ms.json
    kp: float = 4017.8
    kd: float = 15.7
    torque_limit: float = 347.6
    use_bias_compensation: bool = False
    bias_compensation_scale: float = 1.0
    contact_force_scale: float = 1.0
    hit_force_threshold: float = 0.05
    hit_min_ball_height: float = 0.20
    hit_debounce_s: float = 0.1  # Debounce for proper hit counting
    ball_init_pos: tuple[float, float, float] = (0.77, -0.03, 0.70)
    ball_init_pos_noise: float = 0.04  # Match Drake's noise level
    max_residual_rad: float = 0.15
    residual_scale: float = 0.5
    reward_alive: float = 0.01
    reward_hit: float = 5.0
    reward_apex: float = 2.0
    penalty_drop: float = 5.0
    target_apex_height: float = 0.55

    # Domain Randomization Bounds
    randomize_dynamics: bool = False
    ball_mass_range: tuple[float, float] = (0.002, 0.0035)
    ball_friction_range: tuple[float, float] = (0.15, 0.30)
    # MuJoCo uses solref damping ratio for bounciness (lower = more bounce)
    ball_damping_ratio_range: tuple[float, float] = (0.05, 0.15)


TrajectoryFrame = tuple[np.ndarray, np.ndarray, float]


class MujocoFsmIkEnv(gym.Env):
    """Run Drake's FSM/IK policy inside MuJoCo dynamics."""

    def __init__(
        self,
        model_path: str | Path,
        config: MujocoFsmIkConfig | None = None,
        scenario_yaml: str | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config or MujocoFsmIkConfig()
        self._model_path = str(Path(model_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self._model_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = float(self._cfg.physics_dt)
        ratio = self._cfg.control_dt / self._cfg.physics_dt
        self._substeps_per_control = int(round(ratio))
        if not np.isclose(ratio, self._substeps_per_control):
            raise ValueError(
                "control_dt / physics_dt must be an integer. "
                f"Got control_dt={self._cfg.control_dt}, physics_dt={self._cfg.physics_dt}."
            )

        self._rng = np.random.default_rng(0)
        self._sim_time = 0.0
        self._step_count = 0
        self._physics_step_count = 0
        self._contact_hit_count = 0
        self._last_contact_hit_time = -np.inf

        self._iiwa_joint_ids = np.array(
            [self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in IIWA_JOINT_NAMES],
            dtype=int,
        )
        self._iiwa_qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self._iiwa_joint_ids], dtype=int
        )
        self._iiwa_qvel_adr = np.array(
            [self.model.jnt_dofadr[jid] for jid in self._iiwa_joint_ids], dtype=int
        )
        self._iiwa_actuator_ids = np.array(
            [self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in IIWA_ACTUATOR_NAMES],
            dtype=int,
        )

        self._ball_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._ball_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._ball_joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        self._ball_qpos_adr = int(self.model.jnt_qposadr[self._ball_joint_id])
        self._ball_qvel_adr = int(self.model.jnt_dofadr[self._ball_joint_id])

        self._scenario_yaml = scenario_yaml or str(scenario_path())
        q0 = get_iiwa_default_joint_positions(self._scenario_yaml)
        self._q_home = np.asarray(q0 if q0 is not None else DEFAULT_Q0, dtype=float).copy()
        if self._q_home.shape != (7,):
            raise ValueError(f"Expected 7 joint values for q_home, got {self._q_home.shape}")

        self._fsm = FSMController(q0=self._q_home, scenario_yaml=self._scenario_yaml)
        self._fsm_context = self._fsm.CreateDefaultContext()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64 # OBS_DIM is 20
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float64 # IIWA_NUM_JOINTS is 7
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self._iiwa_qpos_adr] = self._q_home
        self.data.qvel[self._iiwa_qvel_adr] = 0.0

        bx, by, bz = self._cfg.ball_init_pos
        noise = float(self._cfg.ball_init_pos_noise)
        ball_xyz = np.array(
            [
                bx + self._rng.uniform(-noise, noise),
                by + self._rng.uniform(-noise, noise),
                bz + self._rng.uniform(-0.5 * noise, noise),
            ],
            dtype=float,
        )
        self.data.qpos[self._ball_qpos_adr : self._ball_qpos_adr + 3] = ball_xyz
        # MuJoCo free-joint quaternion order is (w, x, y, z).
        self.data.qpos[self._ball_qpos_adr + 3 : self._ball_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        )
        self.data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 6] = 0.0

        self.data.ctrl[:] = 0.0
        self._fsm.reset()
        self._sim_time = 0.0
        self._step_count = 0
        self._physics_step_count = 0
        self._contact_hit_count = 0
        self._last_contact_hit_time = -np.inf

        # --- Dynamic Domain Randomization ---
        if self._cfg.randomize_dynamics:
            # Randomize Mass
            new_mass = self._rng.uniform(*self._cfg.ball_mass_range)
            self.model.body_mass[self._ball_body_id] = new_mass

            # Recompute inertia for the new mass (assuming solid sphere: I = 2/5 * m * r^2)
            r = 0.02 # radius from XML
            new_inertia = (2.0 / 5.0) * new_mass * (r ** 2)
            self.model.body_inertia[self._ball_body_id] = np.array([new_inertia, new_inertia, new_inertia])

            # Randomize Friction (sliding, torsional, rolling)
            # MuJoCo geom_friction is a 3-element array per geom
            new_friction = self._rng.uniform(*self._cfg.ball_friction_range)
            self.model.geom_friction[self._ball_geom_id][0] = new_friction # sliding
            self.model.geom_friction[self._ball_geom_id][1] = 0.003        # torsional (keep static or randomize)
            self.model.geom_friction[self._ball_geom_id][2] = 0.0001       # rolling (keep static or randomize)

            # Randomize Restitution (Bounciness) via solref
            # solref is [timeconst, dampratio]. We randomize the dampratio.
            new_damping = self._rng.uniform(*self._cfg.ball_damping_ratio_range)
            self.model.geom_solref[self._ball_geom_id][1] = new_damping
        # -----------------------------------------

        mujoco.mj_forward(self.model, self.data)
        self._ensure_ball_clear_of_contacts()
        self._prev_hit_count = 0
        self._prev_ball_vz = 0.0
        self._last_q_cmd = None
        return self._get_obs(), self.get_info()

    def run_episode(
        self,
        seed: int | None = None,
        viewer_handle: Any | None = None,
        realtime: bool = False,
        trajectory: list[TrajectoryFrame] | None = None,
        trajectory_stride: int = 1,
    ) -> dict[str, Any]:
        self.reset(seed=seed)
        if trajectory is not None:
            trajectory.clear()
            self._append_trajectory_frame(trajectory)
        terminated = False
        trunc_reason = "max_time"

        while self._sim_time < self._cfg.max_episode_time:
            done, reason = self.step_control(
                viewer_handle=viewer_handle,
                realtime=realtime,
                trajectory=trajectory,
                trajectory_stride=trajectory_stride,
            )
            if done:
                terminated = True
                trunc_reason = reason
                break

        metrics = self._fsm.get_metrics()
        info = self.get_info()
        fsm_hits = int(metrics.get("hit_count", 0.0))
        return {
            "terminated": terminated,
            "reason": trunc_reason,
            "sim_time": self._sim_time,
            "control_steps": self._step_count,
            "hits": int(self._contact_hit_count),
            "fsm_hits": fsm_hits,
            "plans": int(metrics.get("plan_count", 0.0)),
            "ik_successes": int(metrics.get("ik_success_count", 0.0)),
            "final_ball_z": float(info["ball_z"]),
            "fsm_state": info["fsm_state"],
            "fsm_metrics": metrics,
        }

    def step_control(
        self,
        viewer_handle: Any | None = None,
        realtime: bool = False,
        trajectory: list[TrajectoryFrame] | None = None,
        trajectory_stride: int = 1,
    ) -> tuple[bool, str]:
        # Evaluated continuously at 1000 Hz
        q, v = self._read_iiwa_state()
        q_cmd = self._eval_fsm(q=q, v=v)
        self._apply_joint_pd(q_cmd=q_cmd, q=q, v=v)

        t0 = time.perf_counter()
        mujoco.mj_step(self.model, self.data)
        self._physics_step_count += 1

        if trajectory is not None and trajectory_stride > 0 and self._physics_step_count % trajectory_stride == 0:
            self._append_trajectory_frame(trajectory)
        if viewer_handle is not None:
            viewer_handle.sync()
        if realtime:
            dt = self.model.opt.timestep
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        self._sim_time += self._cfg.control_dt
        self._step_count += 1
        self._update_contact_hit_count()

        ball_z = float(self.data.xpos[self._ball_body_id, 2])
        if ball_z < self._cfg.min_ball_height:
            return True, "ball_dropped"
        return False, "running"

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1. Scale residual action exactly like Drake
        action = np.asarray(action, dtype=float).flatten()[:7]
        residual = (
            self._cfg.residual_scale
            * self._cfg.max_residual_rad
            * np.clip(action, -1.0, 1.0)
        )

        curr_q, curr_v = self._read_iiwa_state()
        
        # 2. Get the new target at 100Hz
        fsm_q_cmd = self._eval_fsm(q=curr_q, v=curr_v)
        target_q_cmd = np.clip(
            fsm_q_cmd + residual,
            self._fsm._iiwa_q_lower,
            self._fsm._iiwa_q_upper
        )

        # 3. Initialize previous command if first step
        if getattr(self, '_last_q_cmd', None) is None:
            self._last_q_cmd = curr_q.copy()

        # 4. Step physics with Linear Interpolation (First-Order Hold)
        for i in range(self._substeps_per_control):
            frac = (i + 1) / self._substeps_per_control
            interp_q_cmd = self._last_q_cmd + frac * (target_q_cmd - self._last_q_cmd)
            
            step_q, step_v = self._read_iiwa_state()
            self._apply_joint_pd(q_cmd=interp_q_cmd, q=step_q, v=step_v)
            mujoco.mj_step(self.model, self.data)
            
            self._physics_step_count += 1
            self._sim_time += self.model.opt.timestep
            self._update_contact_hit_count()

        # Save target command for the next step's interpolation
        self._last_q_cmd = target_q_cmd.copy()
        self._step_count += 1

        # 5. Gather Returns
        obs = self._get_obs()
        reward, terminated = self._compute_reward(obs)
        truncated = self._sim_time >= self._cfg.max_episode_time

        return obs, reward, terminated, truncated, self.get_info()
    
    def _get_obs(self) -> np.ndarray:
        q, v = self._read_iiwa_state()
        
        # In MuJoCo, ball xpos is index 0:3
        ball_pos = np.asarray(self.data.xpos[self._ball_body_id], dtype=np.float64)
        
        # In MuJoCo, freejoint qvel is [vx, vy, vz, wx, wy, wz]. We just want vx, vy, vz.
        ball_vel = np.asarray(
            self.data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 3], dtype=np.float64
        )
        
        return np.concatenate([q, v, ball_pos, ball_vel])

    def _compute_reward(self, obs: np.ndarray) -> tuple[float, bool]:
        # Indices based on PingPongResidualEnv
        IDX_BALL_Z = 16
        IDX_BALL_VZ = 19
        
        ball_z = float(obs[IDX_BALL_Z])
        ball_vz = float(obs[IDX_BALL_VZ])
        cfg = self._cfg
        reward = 0.0

        if ball_z > cfg.min_ball_height:
            reward += cfg.reward_alive

        # We can use the hit logic already tracked by MuJoCo
        is_new_hit = False
        if self._contact_hit_count > self._prev_hit_count:
            is_new_hit = True
            self._prev_hit_count = self._contact_hit_count
            reward += cfg.reward_hit

        # Detect apex (ball was going up, now going down)
        if hasattr(self, '_prev_ball_vz'):
            if self._prev_ball_vz > 0.05 and ball_vz <= 0.05 and ball_z > 0.25:
                apex_err = abs(ball_z - cfg.target_apex_height)
                reward += cfg.reward_apex * np.exp(-10.0 * apex_err ** 2)

        self._prev_ball_vz = ball_vz

        terminated = ball_z < cfg.min_ball_height
        if terminated:
            reward -= cfg.penalty_drop

        return reward, terminated
    
    def get_info(self) -> dict[str, Any]:
        return {
            "sim_time": self._sim_time,
            "control_step": self._step_count,
            "episode_hits": self._contact_hit_count,
            "fsm_state": self._fsm.get_state_name(),
            "fsm_metrics": self._fsm.get_metrics(),
            "ball_z": float(self.data.xpos[self._ball_body_id, 2]),
        }

    def _eval_fsm(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        iiwa_state = np.concatenate([q, v])
        ball_state = self._read_ball_state()
        contact_forces = self._read_ball_contact_forces()

        if ball_state.shape != (BALL_STATE_SIZE,):
            raise RuntimeError(f"Expected ball state size {BALL_STATE_SIZE}, got {ball_state.shape}")
        if contact_forces.shape != (BALL_CONTACT_FORCE_SIZE,):
            raise RuntimeError(
                f"Expected ball contact force size {BALL_CONTACT_FORCE_SIZE}, got {contact_forces.shape}"
            )

        self._fsm_context.SetTime(self._sim_time)
        self._fsm.get_input_port(0).FixValue(self._fsm_context, iiwa_state)
        self._fsm.get_ball_input_port().FixValue(self._fsm_context, ball_state)
        self._fsm.get_ball_contact_force_input_port().FixValue(
            self._fsm_context, contact_forces
        )
        q_cmd = np.asarray(self._fsm.get_output_port(0).Eval(self._fsm_context), dtype=float)
        return q_cmd

    def _read_iiwa_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(self.data.qpos[self._iiwa_qpos_adr], dtype=float).copy()
        v = np.asarray(self.data.qvel[self._iiwa_qvel_adr], dtype=float).copy()
        return q, v

    def _read_ball_state(self) -> np.ndarray:
        quat_wxyz = np.asarray(self.data.xquat[self._ball_body_id], dtype=float).copy()
        pos_xyz = np.asarray(self.data.xpos[self._ball_body_id], dtype=float).copy()
        # MuJoCo free-joint qvel convention is [vx vy vz wx wy wz].
        vel_xyz = np.asarray(
            self.data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 3], dtype=float
        ).copy()
        omega_xyz = np.asarray(
            self.data.qvel[self._ball_qvel_adr + 3 : self._ball_qvel_adr + 6], dtype=float
        ).copy()
        return np.concatenate([quat_wxyz, pos_xyz, omega_xyz, vel_xyz])

    def _read_ball_contact_forces(self) -> np.ndarray:
        wrench_like = np.zeros(BALL_CONTACT_FORCE_SIZE, dtype=float)
        force_mag_sum = 0.0
        contact_force = np.zeros(6, dtype=float)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 != self._ball_geom_id and c.geom2 != self._ball_geom_id:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            force_mag_sum += float(np.linalg.norm(contact_force[0:3]))

        # Only norm is used by current FSM hit logic, so we pack it in the first slot.
        wrench_like[0] = force_mag_sum * float(self._cfg.contact_force_scale)
        return wrench_like

    def _apply_joint_pd(self, q_cmd: np.ndarray, q: np.ndarray, v: np.ndarray) -> None:
        tau = self._cfg.kp * (q_cmd - q) - self._cfg.kd * v
        if self._cfg.use_bias_compensation:
            tau += float(self._cfg.bias_compensation_scale) * np.asarray(
                self.data.qfrc_bias[self._iiwa_qvel_adr], dtype=float
            )
        tau = np.clip(tau, -self._cfg.torque_limit, self._cfg.torque_limit)
        for i, act_id in enumerate(self._iiwa_actuator_ids):
            self.data.ctrl[act_id] = tau[i]

    def _append_trajectory_frame(self, trajectory: list[TrajectoryFrame]) -> None:
        trajectory.append(
            (
                np.asarray(self.data.qpos, dtype=float).copy(),
                np.asarray(self.data.qvel, dtype=float).copy(),
                float(self.data.time),
            )
        )

    def _update_contact_hit_count(self) -> None:
        force_mag = float(np.linalg.norm(self._read_ball_contact_forces()))
        if force_mag < float(self._cfg.hit_force_threshold):
            return
        ball_z = float(self.data.xpos[self._ball_body_id, 2])
        if ball_z < float(self._cfg.hit_min_ball_height):
            return
        if (self._sim_time - self._last_contact_hit_time) < float(self._cfg.hit_debounce_s):
            return
        self._contact_hit_count += 1
        self._last_contact_hit_time = self._sim_time

    def _ensure_ball_clear_of_contacts(self) -> None:
        """Nudge the spawned ball upward until it is not intersecting any geometry."""
        max_tries = 25
        dz = 0.02
        for _ in range(max_tries):
            if not self._ball_has_contact():
                return
            self.data.qpos[self._ball_qpos_adr + 2] += dz
            self.data.qvel[self._ball_qvel_adr : self._ball_qvel_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _ball_has_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._ball_geom_id or c.geom2 == self._ball_geom_id:
                return True
        return False

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"Could not find MuJoCo object '{name}' ({obj_type}).")
        return obj_id
