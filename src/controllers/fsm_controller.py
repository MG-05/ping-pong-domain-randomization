from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from pydrake.all import (
    BasicVector,
    InverseKinematics,
    LeafSystem,
    Solve,
)

from src.station import make_station

DEFAULT_Q0 = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])
BALL_STATE_SIZE = 13
BALL_CONTACT_FORCE_SIZE = 6
GRAVITY_MPS2 = 9.81


class FsmState(IntEnum):
    WAIT = 0
    PLAN = 1
    PREHIT = 2
    STRIKE = 3
    FOLLOW_THROUGH = 4
    RECOVER = 5


@dataclass(frozen=True)
class FsmTimingConfig:
    """Timing, workspace, and IK tolerances for the baseline FSM."""

    hit_height_m: float = 0.38
    min_ball_height_m: float = 0.26
    descending_velocity_threshold_mps: float = -0.12
    min_hit_horizon_s: float = 0.06
    max_hit_horizon_s: float = 1.10
    strike_lead_s: float = 0.06
    strike_tail_s: float = 0.05
    follow_through_s: float = 0.10
    plan_cooldown_s: float = 0.03
    plan_margin_s: float = 0.02
    replan_interval_s: float = 0.04
    replan_cutoff_before_strike_s: float = 0.04
    joint_speed_limit_rad_s: float = 2.40
    strike_speed_limit_rad_s: float = 3.50
    recover_position_tol_rad: float = 0.45
    recover_velocity_tol_rad_s: float = 0.25
    max_joint_deviation_rad: float = 1.40
    workspace_x_min: float = 0.40
    workspace_x_max: float = 1.00
    workspace_y_min: float = -0.55
    workspace_y_max: float = 0.55
    workspace_z_min: float = 0.22
    workspace_z_max: float = 0.90
    ball_radius_m: float = 0.02
    paddle_half_thickness_m: float = 0.005
    contact_clearance_m: float = 0.002
    prehit_distance_m: float = 0.06
    follow_distance_m: float = 0.04
    hit_target_bias_m: float = 0.02
    normal_tilt_gain: float = 0.22
    normal_tilt_max_deg: float = 18.0
    position_tolerance_m: float = 0.012
    position_tolerance_relaxed_m: float = 0.022
    orientation_tolerance_deg: float = 12.0
    orientation_tolerance_relaxed_deg: float = 24.0
    hit_contact_force_threshold: float = 0.05
    ik_seed_window_rad: float = 0.65


class FSMController(LeafSystem):
    """Finite-state paddle controller with time-gated IK strike plans.

    Inputs:
      - iiwa_state: [q0..q6, v0..v6]
      - ball_state: free-body state [qw qx qy qz x y z wx wy wz vx vy vz]
      - ball_contact_forces: generalized contact forces on the ball (size 6)

    Output:
      - iiwa_position_command: desired joint positions
    """

    def __init__(
        self,
        num_joints: int = 7,
        q0: np.ndarray | None = None,
        timing: FsmTimingConfig | None = None,
        scenario_yaml: str | None = None,
        enable_ik_fallback: bool = True,
    ) -> None:
        super().__init__()
        self._num_joints = num_joints
        self._q_home = np.array(q0) if q0 is not None else DEFAULT_Q0.copy()
        if self._q_home.shape != (self._num_joints,):
            raise ValueError(
                f"Expected q0 shape ({self._num_joints},), got {self._q_home.shape}"
            )
        self._timing = timing or FsmTimingConfig()
        self._enable_ik_fallback = enable_ik_fallback

        self._iiwa_state_port = self.DeclareVectorInputPort(
            "iiwa_state", BasicVector(self._num_joints * 2)
        )
        self._ball_state_port = self.DeclareVectorInputPort(
            "ball_state", BasicVector(BALL_STATE_SIZE)
        )
        self._ball_contact_force_port = self.DeclareVectorInputPort(
            "ball_contact_forces", BasicVector(BALL_CONTACT_FORCE_SIZE)
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", BasicVector(self._num_joints), self._calc_output
        )

        self._state = FsmState.WAIT
        self._state_enter_time = 0.0
        self._last_update_time = -np.inf
        self._last_plan_time = -np.inf

        self._q_command = self._q_home.copy()
        self._q_prehit = self._q_home.copy()
        self._q_hit = self._q_home.copy()
        self._q_follow = self._q_home.copy()
        self._q_prehit_start = self._q_home.copy()

        self._t_hit_abs = np.inf
        self._t_strike_start = np.inf
        self._t_strike_end = np.inf
        self._t_follow_end = np.inf
        self._t_prehit_start = 0.0

        self._last_hit_time = -np.inf
        self._hit_count = 0
        self._plan_count = 0
        self._ik_success_count = 0
        self._prev_ball_vz: float | None = None

        self._ik_enabled = False
        self._ik_error: str | None = None
        self._planning_station = None
        self._planning_plant = None
        self._planning_context = None
        self._world_frame = None
        self._paddle_frame = None
        self._iiwa_position_indices: np.ndarray | None = None
        self._non_iiwa_position_indices: np.ndarray | None = None
        self._q_nominal_full: np.ndarray | None = None
        self._q_non_iiwa_fixed: np.ndarray | None = None
        self._iiwa_q_lower: np.ndarray | None = None
        self._iiwa_q_upper: np.ndarray | None = None

        self._try_initialize_ik_model(scenario_yaml)

    def get_ball_input_port(self):
        return self._ball_state_port

    def get_ball_contact_force_input_port(self):
        return self._ball_contact_force_port

    def get_state_name(self) -> str:
        return FsmState(self._state).name

    def get_metrics(self) -> dict[str, float]:
        return {
            "hit_count": float(self._hit_count),
            "plan_count": float(self._plan_count),
            "ik_success_count": float(self._ik_success_count),
            "last_hit_time": float(self._last_hit_time),
            "state": float(self._state),
        }

    def _calc_output(self, context, output) -> None:
        iiwa_state = self._iiwa_state_port.Eval(context)
        q = np.asarray(iiwa_state[: self._num_joints], dtype=float)
        v = np.asarray(iiwa_state[self._num_joints :], dtype=float)
        ball_state = self._eval_ball_state(context)
        ball_contact_forces = self._eval_ball_contact_forces(context)
        t = context.get_time()

        if t > self._last_update_time + 1e-9:
            self._advance_fsm(
                t=t,
                q=q,
                v=v,
                ball_state=ball_state,
                ball_contact_forces=ball_contact_forces,
            )
            self._last_update_time = t
            if ball_state is not None and ball_state.size >= BALL_STATE_SIZE:
                self._prev_ball_vz = float(ball_state[12])

        output.SetFromVector(self._q_command)

    def _advance_fsm(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        ball_state,
        ball_contact_forces,
    ) -> None:
        if self._state == FsmState.WAIT:
            self._q_command = self._q_home.copy()
            if self._should_start_plan(t=t, ball_state=ball_state):
                self._transition_to(FsmState.PLAN, t)
            return

        if self._state == FsmState.PLAN:
            if self._attempt_plan(t=t, q=q, ball_state=ball_state):
                self._transition_to(FsmState.PREHIT, t)
            else:
                self._transition_to(FsmState.RECOVER, t)
            return

        if self._state == FsmState.PREHIT:
            if self._should_replan(t=t):
                self._attempt_plan(t=t, q=q, ball_state=ball_state)

            if t >= self._t_hit_abs:
                self._q_command = self._q_home.copy()
                self._transition_to(FsmState.RECOVER, t)
                return

            self._q_command = self._interpolate(
                q0=self._q_prehit_start,
                q1=self._q_prehit,
                t0=self._t_prehit_start,
                t1=self._t_strike_start,
                t=t,
            )
            if t >= self._t_strike_start:
                self._transition_to(FsmState.STRIKE, t)
            return

        if self._state == FsmState.STRIKE:
            self._q_command = self._strike_command(t=t)
            if self._did_register_hit(
                t=t,
                ball_state=ball_state,
                ball_contact_forces=ball_contact_forces,
            ) or self._did_detect_rebound(t=t, ball_state=ball_state):
                self._last_hit_time = t
                self._hit_count += 1
                self._transition_to(FsmState.FOLLOW_THROUGH, t)
                return
            if t >= self._t_strike_end:
                self._transition_to(FsmState.FOLLOW_THROUGH, t)
            return

        if self._state == FsmState.FOLLOW_THROUGH:
            self._q_command = self._q_follow.copy()
            if t >= self._t_follow_end:
                self._q_command = self._q_home.copy()
                self._transition_to(FsmState.WAIT, t)
            return

        if self._state == FsmState.RECOVER:
            self._q_command = self._q_home.copy()
            if self._is_recovered(q=q, v=v):
                self._transition_to(FsmState.WAIT, t)
            return

        self._q_command = self._q_home.copy()
        self._transition_to(FsmState.WAIT, t)

    def _should_start_plan(self, t: float, ball_state) -> bool:
        if ball_state is None:
            return False
        if t - self._last_plan_time < self._timing.plan_cooldown_s:
            return False

        p_B = ball_state[4:7]
        v_B = ball_state[10:13]
        z = float(p_B[2])
        vz = float(v_B[2])
        if z < self._timing.min_ball_height_m:
            return False
        if vz > self._timing.descending_velocity_threshold_mps:
            return False
        return self._predict_intercept(ball_state) is not None

    def _attempt_plan(self, t: float, q: np.ndarray, ball_state) -> bool:
        intercept = self._predict_intercept(ball_state)
        if intercept is None:
            return False
        tau_hit, p_hit, v_hit = intercept

        paddle_position_hint = self._estimate_paddle_position(q)
        targets = self._build_cartesian_strike_targets(
            p_hit=p_hit,
            v_hit=v_hit,
            tau_hit=tau_hit,
            paddle_position_hint=paddle_position_hint,
        )
        if targets is None:
            return False

        q_waypoints = self._solve_ik_waypoints(
            q_seed=q,
            p_prehit=targets["p_prehit"],
            p_hit=targets["p_hit"],
            p_follow=targets["p_follow"],
            n_hit=targets["n_hit"],
        )
        if q_waypoints is None:
            return False
        q_prehit, q_hit, q_follow = q_waypoints

        schedule = self._compute_schedule(
            tau_hit=tau_hit,
            q_now=q,
            q_prehit=q_prehit,
            q_hit=q_hit,
            q_follow=q_follow,
        )
        if schedule is None:
            return False
        strike_lead_s, strike_tail_s = schedule

        self._q_prehit_start = q.copy()
        self._q_prehit = q_prehit
        self._q_hit = q_hit
        self._q_follow = q_follow

        self._t_prehit_start = t
        self._t_hit_abs = t + tau_hit
        self._t_strike_start = self._t_hit_abs - strike_lead_s
        self._t_strike_end = self._t_hit_abs + strike_tail_s
        self._t_follow_end = self._t_strike_end + self._timing.follow_through_s

        self._last_plan_time = t
        self._plan_count += 1
        self._q_command = self._q_prehit_start.copy()
        return True

    def _should_replan(self, t: float) -> bool:
        if t - self._last_plan_time < self._timing.replan_interval_s:
            return False
        return (self._t_strike_start - t) > self._timing.replan_cutoff_before_strike_s

    def _predict_intercept(self, ball_state):
        if ball_state is None or len(ball_state) < BALL_STATE_SIZE:
            return None

        p_B = np.asarray(ball_state[4:7], dtype=float)
        v_B = np.asarray(ball_state[10:13], dtype=float)
        z0 = float(p_B[2])
        vz0 = float(v_B[2])
        z_hit = self._timing.hit_height_m

        # z(t) = z0 + vz0*t - 0.5*g*t^2 = z_hit.
        a = -0.5 * GRAVITY_MPS2
        b = vz0
        c = z0 - z_hit
        roots = np.roots([a, b, c])
        candidates = []
        for root in roots:
            if not np.isreal(root):
                continue
            t_hit = float(np.real(root))
            if t_hit < self._timing.min_hit_horizon_s:
                continue
            if t_hit > self._timing.max_hit_horizon_s:
                continue
            vz_hit = vz0 - GRAVITY_MPS2 * t_hit
            if vz_hit >= self._timing.descending_velocity_threshold_mps:
                continue
            p_hit = p_B + v_B * t_hit + np.array(
                [0.0, 0.0, -0.5 * GRAVITY_MPS2 * t_hit * t_hit], dtype=float
            )
            if not self._is_in_workspace(p_hit):
                continue
            v_hit = v_B + np.array([0.0, 0.0, -GRAVITY_MPS2 * t_hit], dtype=float)
            candidates.append((t_hit, p_hit, v_hit))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0]

    def _build_cartesian_strike_targets(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        tau_hit: float,
        paddle_position_hint: np.ndarray | None,
    ):
        n_hit = self._desired_paddle_normal(v_hit)

        contact_offset = (
            self._timing.ball_radius_m
            + self._timing.paddle_half_thickness_m
            + self._timing.contact_clearance_m
        )
        p_contact = p_hit - contact_offset * n_hit + self._timing.hit_target_bias_m * n_hit
        if paddle_position_hint is not None:
            alpha_xy = np.clip(
                (tau_hit - self._timing.min_hit_horizon_s) / 0.30, 0.35, 1.00
            )
            p_contact[:2] = paddle_position_hint[:2] + alpha_xy * (
                p_contact[:2] - paddle_position_hint[:2]
            )
        p_prehit = p_contact - self._timing.prehit_distance_m * n_hit
        p_follow = p_contact + self._timing.follow_distance_m * n_hit

        if not self._is_in_workspace(p_prehit):
            return None
        if not self._is_in_workspace(p_contact):
            return None
        if not self._is_in_workspace(p_follow):
            return None

        return {
            "p_prehit": p_prehit,
            "p_hit": p_contact,
            "p_follow": p_follow,
            "n_hit": n_hit,
        }

    def _desired_paddle_normal(self, v_hit: np.ndarray) -> np.ndarray:
        v_xy = np.array([float(v_hit[0]), float(v_hit[1]), 0.0], dtype=float)
        speed_xy = float(np.linalg.norm(v_xy))
        tilt_max = np.deg2rad(self._timing.normal_tilt_max_deg)

        if speed_xy < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=float)

        tilt = np.clip(self._timing.normal_tilt_gain * speed_xy, 0.0, tilt_max)
        direction_xy = -v_xy / speed_xy
        n = np.array(
            [
                np.sin(tilt) * direction_xy[0],
                np.sin(tilt) * direction_xy[1],
                np.cos(tilt),
            ],
            dtype=float,
        )
        return n / np.linalg.norm(n)

    def _estimate_paddle_position(self, q_iiwa: np.ndarray):
        if not self._ik_enabled:
            return None
        try:
            q_full = self._pack_q_seed_full(q_iiwa)
            self._planning_plant.SetPositions(self._planning_context, q_full)
            X_WP = self._planning_plant.CalcRelativeTransform(
                self._planning_context, self._world_frame, self._paddle_frame
            )
            return np.asarray(X_WP.translation(), dtype=float)
        except Exception:
            return None

    def _solve_ik_waypoints(
        self,
        q_seed: np.ndarray,
        p_prehit: np.ndarray,
        p_hit: np.ndarray,
        p_follow: np.ndarray,
        n_hit: np.ndarray,
    ):
        if self._ik_enabled:
            q_prehit = self._solve_single_ik(
                q_seed=q_seed,
                target_position=p_prehit,
                target_normal=n_hit,
                position_tolerance=self._timing.position_tolerance_m,
                normal_tolerance_deg=self._timing.orientation_tolerance_deg,
            )
            if q_prehit is None:
                q_prehit = self._solve_single_ik(
                    q_seed=q_seed,
                    target_position=p_prehit,
                    target_normal=n_hit,
                    position_tolerance=self._timing.position_tolerance_relaxed_m,
                    normal_tolerance_deg=self._timing.orientation_tolerance_relaxed_deg,
                )
            if q_prehit is None:
                return None

            q_hit = self._solve_single_ik(
                q_seed=q_prehit,
                target_position=p_hit,
                target_normal=n_hit,
                position_tolerance=self._timing.position_tolerance_m,
                normal_tolerance_deg=self._timing.orientation_tolerance_deg,
            )
            if q_hit is None:
                q_hit = self._solve_single_ik(
                    q_seed=q_prehit,
                    target_position=p_hit,
                    target_normal=n_hit,
                    position_tolerance=self._timing.position_tolerance_relaxed_m,
                    normal_tolerance_deg=self._timing.orientation_tolerance_relaxed_deg,
                )
            if q_hit is None:
                return None

            q_follow = self._solve_single_ik(
                q_seed=q_hit,
                target_position=p_follow,
                target_normal=n_hit,
                position_tolerance=self._timing.position_tolerance_m,
                normal_tolerance_deg=self._timing.orientation_tolerance_deg,
            )
            if q_follow is None:
                q_follow = self._solve_single_ik(
                    q_seed=q_hit,
                    target_position=p_follow,
                    target_normal=n_hit,
                    position_tolerance=self._timing.position_tolerance_relaxed_m,
                    normal_tolerance_deg=self._timing.orientation_tolerance_relaxed_deg,
                )
            if q_follow is None:
                return None

            self._ik_success_count += 1
            return q_prehit, q_hit, q_follow

        if not self._enable_ik_fallback:
            return None
        return self._build_fallback_waypoints(q_seed=q_seed, p_hit=p_hit)

    def _solve_single_ik(
        self,
        q_seed: np.ndarray,
        target_position: np.ndarray,
        target_normal: np.ndarray,
        position_tolerance: float,
        normal_tolerance_deg: float,
    ):
        if not self._ik_enabled:
            return None

        q_seed_full = self._pack_q_seed_full(q_seed)
        self._planning_plant.SetPositions(self._planning_context, q_seed_full)
        ik = InverseKinematics(self._planning_plant, self._planning_context)
        q_vars = ik.q()
        prog = ik.prog()

        if self._non_iiwa_position_indices.size > 0:
            prog.AddBoundingBoxConstraint(
                self._q_non_iiwa_fixed,
                self._q_non_iiwa_fixed,
                q_vars[self._non_iiwa_position_indices],
            )
        prog.AddBoundingBoxConstraint(
            self._iiwa_q_lower,
            self._iiwa_q_upper,
            q_vars[self._iiwa_position_indices],
        )
        seed_window = float(self._timing.ik_seed_window_rad)
        local_lower = np.maximum(self._iiwa_q_lower, q_seed - seed_window)
        local_upper = np.minimum(self._iiwa_q_upper, q_seed + seed_window)
        prog.AddBoundingBoxConstraint(
            local_lower,
            local_upper,
            q_vars[self._iiwa_position_indices],
        )

        tol_vec = np.full(3, float(position_tolerance), dtype=float)
        ik.AddPositionConstraint(
            self._paddle_frame,
            np.zeros(3),
            self._world_frame,
            np.asarray(target_position, dtype=float) - tol_vec,
            np.asarray(target_position, dtype=float) + tol_vec,
        )
        normal = np.asarray(target_normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            return None
        normal = normal / normal_norm
        ik.AddAngleBetweenVectorsConstraint(
            self._world_frame,
            normal,
            self._paddle_frame,
            np.array([0.0, 0.0, 1.0], dtype=float),
            0.0,
            float(np.deg2rad(normal_tolerance_deg)),
        )

        weights = np.full(self._planning_plant.num_positions(), 1e-4, dtype=float)
        weights[self._iiwa_position_indices] = 2.5
        prog.AddQuadraticErrorCost(np.diag(weights), q_seed_full, q_vars)
        prog.SetInitialGuess(q_vars, q_seed_full)

        result = Solve(prog)
        if not result.is_success():
            return None
        q_sol_full = result.GetSolution(q_vars)
        q_sol = np.asarray(q_sol_full[self._iiwa_position_indices], dtype=float)
        if np.any(~np.isfinite(q_sol)):
            return None
        return q_sol

    def _build_fallback_waypoints(self, q_seed: np.ndarray, p_hit: np.ndarray):
        # Fallback map only for environments where IK model init fails.
        q_prehit = self._q_home.copy()
        x_hit, y_hit, z_hit = [float(v) for v in p_hit]
        q_prehit[0] += 0.95 * np.clip(y_hit, -0.30, 0.30)
        q_prehit[1] += -0.70 * np.clip(x_hit - 0.75, -0.25, 0.25)
        q_prehit[3] += -0.70 * np.clip(z_hit - self._timing.hit_height_m, -0.20, 0.20)
        q_prehit = np.clip(q_prehit, self._iiwa_q_lower, self._iiwa_q_upper)

        q_hit = np.clip(
            q_prehit + np.array([0.0, 0.0, 0.0, 0.20, 0.0, -0.10, 0.0]),
            self._iiwa_q_lower,
            self._iiwa_q_upper,
        )
        q_follow = np.clip(
            q_hit + np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.00, 0.0]),
            self._iiwa_q_lower,
            self._iiwa_q_upper,
        )
        _ = q_seed
        return q_prehit, q_hit, q_follow

    def _compute_schedule(
        self,
        tau_hit: float,
        q_now: np.ndarray,
        q_prehit: np.ndarray,
        q_hit: np.ndarray,
        q_follow: np.ndarray,
    ):
        t_prehit_required = np.max(
            np.abs(q_prehit - q_now) / self._timing.joint_speed_limit_rad_s
        )
        t_strike_required = np.max(
            np.abs(q_hit - q_prehit) / self._timing.strike_speed_limit_rad_s
        )
        t_tail_required = np.max(
            np.abs(q_follow - q_hit) / self._timing.strike_speed_limit_rad_s
        )

        strike_lead_s = max(
            self._timing.strike_lead_s,
            t_strike_required + self._timing.plan_margin_s,
        )
        strike_tail_s = max(
            self._timing.strike_tail_s,
            t_tail_required + self._timing.plan_margin_s,
        )
        prehit_time_available = tau_hit - strike_lead_s
        if prehit_time_available <= 0.0:
            return None
        if t_prehit_required + self._timing.plan_margin_s > prehit_time_available:
            return None
        return strike_lead_s, strike_tail_s

    def _did_register_hit(self, t: float, ball_state, ball_contact_forces) -> bool:
        if ball_contact_forces is None:
            return False
        # Restrict detection around expected impact time to avoid floor-contact triggers.
        if abs(t - self._t_hit_abs) > 0.20:
            return False

        contact_norm = float(np.linalg.norm(ball_contact_forces))
        if contact_norm < self._timing.hit_contact_force_threshold:
            return False

        if ball_state is None:
            return False
        z_ball = float(ball_state[6])
        return z_ball > self._timing.workspace_z_min + 0.03

    def _did_detect_rebound(self, t: float, ball_state) -> bool:
        if ball_state is None:
            return False
        if self._prev_ball_vz is None:
            return False
        if abs(t - self._t_hit_abs) > 0.20:
            return False
        vz = float(ball_state[12])
        z_ball = float(ball_state[6])
        return (
            self._prev_ball_vz < -0.20
            and vz > 0.08
            and z_ball > self._timing.workspace_z_min + 0.03
        )

    def _is_recovered(self, q: np.ndarray, v: np.ndarray) -> bool:
        q_err = np.linalg.norm(q - self._q_home)
        v_mag = np.linalg.norm(v)
        return (
            q_err <= self._timing.recover_position_tol_rad
            and v_mag <= self._timing.recover_velocity_tol_rad_s
        )

    def _strike_command(self, t: float) -> np.ndarray:
        if t <= self._t_hit_abs:
            return self._interpolate(
                q0=self._q_prehit,
                q1=self._q_hit,
                t0=self._t_strike_start,
                t1=self._t_hit_abs,
                t=t,
            )
        return self._interpolate(
            q0=self._q_hit,
            q1=self._q_follow,
            t0=self._t_hit_abs,
            t1=self._t_strike_end,
            t=t,
        )

    def _eval_ball_state(self, context):
        try:
            raw = self._ball_state_port.Eval(context)
        except Exception:
            return None
        data = np.asarray(raw, dtype=float).reshape(-1)
        if data.size < BALL_STATE_SIZE:
            return None
        return data

    def _eval_ball_contact_forces(self, context):
        try:
            raw = self._ball_contact_force_port.Eval(context)
        except Exception:
            return None
        data = np.asarray(raw, dtype=float).reshape(-1)
        if data.size < BALL_CONTACT_FORCE_SIZE:
            return None
        return data

    def _try_initialize_ik_model(self, scenario_yaml: str | None) -> None:
        if scenario_yaml is None:
            self._ik_error = "No scenario YAML provided for IK model."
            self._set_default_joint_bounds()
            return
        try:
            station = make_station(scenario_yaml, meshcat=None, lcm=None)
            plant = station.GetSubsystemByName("plant")
            context = plant.CreateDefaultContext()
            iiwa_instance = plant.GetModelInstanceByName("iiwa")
            paddle_instance = plant.GetModelInstanceByName("paddle")
            paddle_body = plant.GetBodyByName("paddle_link", paddle_instance)

            q_nominal_full = plant.GetPositions(context).copy()
            iiwa_indices = self._infer_model_position_indices(plant, iiwa_instance)
            if iiwa_indices.size != self._num_joints:
                raise RuntimeError(
                    f"Expected {self._num_joints} iiwa positions, got {iiwa_indices.size}"
                )

            q_nominal_full[iiwa_indices] = self._q_home
            non_iiwa_indices = np.setdiff1d(
                np.arange(plant.num_positions(), dtype=int), iiwa_indices
            )

            lower_full = np.asarray(plant.GetPositionLowerLimits(), dtype=float)
            upper_full = np.asarray(plant.GetPositionUpperLimits(), dtype=float)
            deviation = self._timing.max_joint_deviation_rad
            iiwa_lower = np.maximum(lower_full[iiwa_indices], self._q_home - deviation)
            iiwa_upper = np.minimum(upper_full[iiwa_indices], self._q_home + deviation)

            self._planning_station = station
            self._planning_plant = plant
            self._planning_context = context
            self._world_frame = plant.world_frame()
            self._paddle_frame = paddle_body.body_frame()
            self._iiwa_position_indices = iiwa_indices
            self._non_iiwa_position_indices = non_iiwa_indices
            self._q_nominal_full = q_nominal_full
            self._q_non_iiwa_fixed = q_nominal_full[non_iiwa_indices]
            self._iiwa_q_lower = iiwa_lower
            self._iiwa_q_upper = iiwa_upper
            self._ik_enabled = True
            self._ik_error = None
        except Exception as exc:
            self._ik_enabled = False
            self._ik_error = str(exc)
            self._set_default_joint_bounds()
            print(f"FSMController IK disabled: {self._ik_error}")

    def _set_default_joint_bounds(self) -> None:
        deviation = self._timing.max_joint_deviation_rad
        self._iiwa_q_lower = self._q_home - deviation
        self._iiwa_q_upper = self._q_home + deviation
        self._iiwa_position_indices = np.arange(self._num_joints, dtype=int)
        self._non_iiwa_position_indices = np.array([], dtype=int)
        self._q_nominal_full = self._q_home.copy()
        self._q_non_iiwa_fixed = np.array([], dtype=float)

    @staticmethod
    def _infer_model_position_indices(plant, model_instance) -> np.ndarray:
        n_model = plant.num_positions(model_instance)
        if n_model <= 0:
            return np.array([], dtype=int)
        q_probe = np.full(plant.num_positions(), np.nan, dtype=float)
        marker = np.linspace(1.0, float(n_model), n_model, dtype=float)
        plant.SetPositionsInArray(model_instance, marker, q_probe)
        return np.where(np.isfinite(q_probe))[0].astype(int)

    def _pack_q_seed_full(self, q_iiwa: np.ndarray) -> np.ndarray:
        q_full = self._q_nominal_full.copy()
        q_full[self._iiwa_position_indices] = q_iiwa
        return q_full

    def _is_in_workspace(self, p_W: np.ndarray) -> bool:
        x, y, z = [float(v) for v in p_W]
        return (
            self._timing.workspace_x_min <= x <= self._timing.workspace_x_max
            and self._timing.workspace_y_min <= y <= self._timing.workspace_y_max
            and self._timing.workspace_z_min <= z <= self._timing.workspace_z_max
        )

    def _transition_to(self, state: FsmState, t: float) -> None:
        self._state = state
        self._state_enter_time = t

    @staticmethod
    def _interpolate(
        q0: np.ndarray, q1: np.ndarray, t0: float, t1: float, t: float
    ) -> np.ndarray:
        if t1 <= t0 + 1e-9:
            return q1.copy()
        alpha = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        return (1.0 - alpha) * q0 + alpha * q1
