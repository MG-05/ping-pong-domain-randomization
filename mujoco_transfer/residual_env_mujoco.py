from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mujoco_transfer.fsm_ik_env import MujocoFsmIkEnv, MujocoFsmIkConfig


@dataclass
class MujocoResidualConfig(MujocoFsmIkConfig):
    """Extends base FSM config with RL residual params and domain randomization."""
    # RL control rate: matches Drake training (control_dt=0.01 in residual_env.py).
    # The RL policy acts every rl_control_dt seconds; FSM is evaluated once per RL
    # step and the resulting q_cmd is held fixed for all physics substeps within
    # that window — matching Drake's AdvanceTo(sim_time + control_dt) behavior.
    rl_control_dt: float = 0.01
    # Ball init noise matches Drake training (EnvConfig.ball_init_pos_noise=0.04)
    ball_init_pos_noise: float = 0.04
    # Residual scale tuned via sweep: 0.005 optimal for nominal, 0.01 for robust.
    # Tiny scale limits interference between Drake-trained residuals and new aimed FSM.
    max_residual_rad: float = 0.15
    residual_scale: float = 0.005
    # Rewards (identical to Drake's residual_env.py)
    reward_alive: float = 0.01
    reward_hit: float = 5.0
    reward_apex: float = 2.0
    penalty_drop: float = 5.0
    target_apex_height: float = 0.55
    # Domain randomization (for robust model evaluation)
    randomize_dynamics: bool = False
    ball_mass_range: tuple[float, float] = (0.002, 0.0035)
    ball_friction_range: tuple[float, float] = (0.15, 0.30)
    ball_damping_ratio_range: tuple[float, float] = (0.05, 0.15)


class MujocoResidualEnv(gym.Env):
    """Gym wrapper: FSM-IK inner loop + RL residual outer loop.

    The FSM runs at physics_dt (1ms) and produces base joint commands.
    The RL policy outputs 7-D residuals (scaled to ±max_residual_rad) which
    are added to the FSM command before applying joint-space PD control.

    Observation (20-D): [iiwa_q(7), iiwa_v(7), ball_xyz(3), ball_vxyz(3)]
    Action (7-D): residual joint corrections in [-1, 1]
    """

    def __init__(
        self,
        model_path: str,
        config: MujocoResidualConfig | None = None,
        scenario_yaml: str | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config or MujocoResidualConfig()
        self._sim = MujocoFsmIkEnv(
            model_path=model_path,
            config=self._cfg,
            scenario_yaml=scenario_yaml,
        )

        # Expose model/data for viewer compatibility
        self.model = self._sim.model
        self.data = self._sim.data

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float64
        )

        rl_ratio = self._cfg.rl_control_dt / self._cfg.physics_dt
        self._rl_substeps = int(round(rl_ratio))
        if not np.isclose(rl_ratio, self._rl_substeps):
            raise ValueError(
                f"rl_control_dt / physics_dt must be an integer. "
                f"Got rl_control_dt={self._cfg.rl_control_dt}, physics_dt={self._cfg.physics_dt}."
            )

        self._prev_hit_count = 0
        self._prev_ball_vz = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._sim.reset(seed=seed)
        if self._cfg.randomize_dynamics:
            self._apply_domain_randomization()
            mujoco.mj_forward(self.model, self.data)
        self._prev_hit_count = 0
        self._prev_ball_vz = 0.0
        return self._get_obs(), self._sim.get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).flatten()[:7]
        residual = (
            self._cfg.residual_scale
            * self._cfg.max_residual_rad
            * np.clip(action, -1.0, 1.0)
        )

        # Evaluate FSM ONCE and fix target_q_cmd for the entire RL window.
        # This matches Drake: AdvanceTo(sim_time + control_dt) holds q_cmd fixed.
        q, v = self._sim._read_iiwa_state()
        fsm_q_cmd = self._sim._eval_fsm(q=q, v=v)
        target_q_cmd = np.clip(
            fsm_q_cmd + residual,
            self._sim._fsm._iiwa_q_lower,
            self._sim._fsm._iiwa_q_upper,
        )

        terminated = False
        for _ in range(self._rl_substeps):
            q, v = self._sim._read_iiwa_state()
            self._sim._apply_joint_pd(q_cmd=target_q_cmd, q=q, v=v)
            mujoco.mj_step(self.model, self.data)
            self._sim._physics_step_count += 1
            self._sim._sim_time += self.model.opt.timestep
            self._sim._update_contact_hit_count()

            ball_z = float(self.data.xpos[self._sim._ball_body_id, 2])
            if ball_z < self._cfg.min_ball_height:
                terminated = True
                break

        self._sim._step_count += 1
        obs = self._get_obs()
        reward, term_from_reward = self._compute_reward(obs)
        terminated = terminated or term_from_reward
        truncated = self._sim._sim_time >= self._cfg.max_episode_time
        return obs, reward, terminated, truncated, self._sim.get_info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        q, v = self._sim._read_iiwa_state()
        ball_pos = np.asarray(self.data.xpos[self._sim._ball_body_id], dtype=np.float64)
        ball_vel = np.asarray(
            self.data.qvel[self._sim._ball_qvel_adr : self._sim._ball_qvel_adr + 3],
            dtype=np.float64,
        )
        return np.concatenate([q, v, ball_pos, ball_vel])

    def _compute_reward(self, obs: np.ndarray) -> tuple[float, bool]:
        ball_z = float(obs[16])
        ball_vz = float(obs[19])
        reward = 0.0

        if ball_z > self._cfg.min_ball_height:
            reward += self._cfg.reward_alive

        if self._sim._contact_hit_count > self._prev_hit_count:
            self._prev_hit_count = self._sim._contact_hit_count
            reward += self._cfg.reward_hit

        if self._prev_ball_vz > 0.05 and ball_vz <= 0.05 and ball_z > 0.25:
            apex_err = abs(ball_z - self._cfg.target_apex_height)
            reward += self._cfg.reward_apex * np.exp(-10.0 * apex_err**2)
        self._prev_ball_vz = ball_vz

        terminated = ball_z < self._cfg.min_ball_height
        if terminated:
            reward -= self._cfg.penalty_drop
        return reward, terminated

    def _apply_domain_randomization(self) -> None:
        rng = self._sim._rng
        ball_body = self._sim._ball_body_id
        ball_geom = self._sim._ball_geom_id

        new_mass = float(rng.uniform(*self._cfg.ball_mass_range))
        self.model.body_mass[ball_body] = new_mass
        r = 0.02
        inertia = (2.0 / 5.0) * new_mass * r**2
        self.model.body_inertia[ball_body] = np.full(3, inertia)

        new_friction = float(rng.uniform(*self._cfg.ball_friction_range))
        self.model.geom_friction[ball_geom][0] = new_friction

        new_damping = float(rng.uniform(*self._cfg.ball_damping_ratio_range))
        self.model.geom_solref[ball_geom][1] = new_damping
