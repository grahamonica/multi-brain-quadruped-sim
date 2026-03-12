"""Episode runner and ES trainer for the SNN brain.

Uses OpenAI-ES style (Salimans et al. 2017):
  Each generation perturbs weights with Gaussian noise, evaluates N candidates,
  estimates gradient from returns, updates parameters.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai.brain import DEFAULT_MOTOR_NOISE_SCALE, N_IN, SNNBrain
from quadruped.environment import QuadrupedEnvironment
from quadruped.robot import Quadruped

# ── Training hyper-parameters ──────────────────────────────────────────────
EPISODE_S   = 3000.0    # simulated seconds per episode — plenty of time to explore
BRAIN_DT    = 0.050   # brain & motor update interval (10 ms = 100 Hz)
MOTOR_SCALE = 6.0     # map [-1,1] → rad/s
GOAL_HEIGHT_M = 0.16  # = leg_length_m
FIELD_HALF  = 15.0    # field half-extent (m) — large open space
POP_SIZE    = 8       # ES population per generation
SIGMA       = 0.05    # ES noise std
LR          = 0.05    # ES learning rate
MAX_MOTOR_RAD_S = 8.0
MAX_MOTOR_NOISE_SCALE = 1.20
FAST_PROGRESS_TAU_S = 0.20
SLOW_PROGRESS_TAU_S = 0.80
DRAMATIC_PROGRESS_DROP_RATIO = 0.55
NOISE_ATTACK_TAU_S = 0.15
NOISE_RELEASE_TAU_S = 0.90
LONG_THIN_SIDE_ROLL_MIN_RAD = math.radians(65.0)
LONG_THIN_SIDE_ROLL_MAX_RAD = math.radians(115.0)
LONG_THIN_SIDE_STUCK_DELAY_S = 1.0
LONG_THIN_SIDE_PENALTY_PER_S = 3.0
STUCK_IMU_ROLL_CHANGE_REWARD_PER_RAD = 3.5
SELF_RIGHT_EXIT_BONUS = 4.0


@dataclass
class TrainingState:
    generation: int = 0
    best_reward: float = -1e9
    mean_reward: float = 0.0
    episode_reward: float = 0.0
    goal_xyz: tuple[float, float, float] = (1.0, 0.0, GOAL_HEIGHT_M)
    robot_state: dict[str, Any] = field(default_factory=dict)
    rewards_history: list[float] = field(default_factory=list)


def _make_env() -> QuadrupedEnvironment:
    robot = Quadruped.create_kt2_style()
    return QuadrupedEnvironment(robot=robot)


def _ema_alpha(dt_s: float, tau_s: float) -> float:
    return 1.0 - math.exp(-dt_s / max(tau_s, 1e-6))


def _wrap_angle_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _build_obs(env: QuadrupedEnvironment, goal: tuple[float, float, float]) -> np.ndarray:
    """Assemble the 48-element observation vector."""
    robot = env.robot
    body = robot.body

    # goal coords
    v_goal = list(goal)

    # total COM
    com = env.center_of_mass_xyz_m()
    v_com = list(com)

    # body COM (= body position)
    v_body_com = list(body.position_xyz_m)

    # leg foot world positions (4×3 = 12)
    v_feet: list[float] = []
    for leg in robot.legs:
        state = env.leg_force_states.get(leg.name)
        if state is not None:
            v_feet.extend(state.foot_position_xyz_m)
        else:
            # fallback: compute directly
            mount = env._body_point_world(leg.mount_point_xyz_m)
            foot_offset = env._body_vector_world(leg.foot_offset_from_mount_m())
            v_feet.extend([mount[0] + foot_offset[0],
                            mount[1] + foot_offset[1],
                            mount[2] + foot_offset[2]])

    # leg COM world positions (4×3 = 12)
    v_leg_com: list[float] = []
    for leg in robot.legs:
        mount = env._body_point_world(leg.mount_point_xyz_m)
        com_off = env._body_vector_world(leg.com_offset_from_mount_m())
        v_leg_com.extend([mount[0] + com_off[0],
                           mount[1] + com_off[1],
                           mount[2] + com_off[2]])

    # body IMU angles (roll, pitch, yaw)
    v_body_imu = list(body.imu.rotation_xyz_rad)

    # leg IMU angles (4×3 = 12)
    v_leg_imu: list[float] = []
    for leg in robot.legs:
        v_leg_imu.extend(leg.imu.rotation_xyz_rad)

    obs = np.array(
        v_goal + v_com + v_body_com + v_feet + v_leg_com + v_body_imu + v_leg_imu,
        dtype=np.float32,
    )
    assert obs.shape == (N_IN,), f"obs shape {obs.shape} != {N_IN}"

    # Normalize: scale positions by 1/FIELD_HALF, angles already in rad (≲π)
    obs[:33] /= FIELD_HALF
    obs[33:] /= math.pi
    return obs


def _run_episode(
    brain: SNNBrain,
    goal: tuple[float, float, float],
    on_step: Any = None,
) -> float:
    """Run one episode; return total reward. Calls on_step(state_dict) each physics step."""
    env = _make_env()
    brain.reset()
    robot = env.robot
    leg_names = [leg.name for leg in robot.legs]

    goal_xy = np.array([goal[0], goal[1]])
    initial_com = env.center_of_mass_xyz_m()
    prev_dist = float(np.linalg.norm(np.array([initial_com[0], initial_com[1]]) - goal_xy))
    total_reward = 0.0
    noise_scale = DEFAULT_MOTOR_NOISE_SCALE
    fast_closing_rate_m_s = 0.0
    slow_closing_rate_m_s = 0.0
    closing_rate_m_s = 0.0
    progress_drop_ratio = 0.0
    long_thin_side_dwell_s = 0.0
    long_thin_side_stuck = False
    prev_imu_roll_rad = _wrap_angle_pi(robot.body.imu.rotation_xyz_rad[0])

    steps = int(EPISODE_S / BRAIN_DT)        # 3000 steps at 10 ms
    report_every = 2                          # every step → frontend plays in real time

    for step_i in range(steps):
        obs = _build_obs(env, goal)
        motor_cmds = brain.step(obs, BRAIN_DT, noise_scale=noise_scale)   # (4,)

        for i, name in enumerate(leg_names):
            vel = float(np.clip(motor_cmds[i] * MOTOR_SCALE, -MAX_MOTOR_RAD_S, MAX_MOTOR_RAD_S))
            robot.set_leg_motor_velocity(name, vel)

        env.advance(BRAIN_DT)

        com = env.center_of_mass_xyz_m()
        com_xy = np.array([com[0], com[1]])
        dist = float(np.linalg.norm(com_xy - goal_xy))
        closing_rate_m_s = max(prev_dist - dist, 0.0) / BRAIN_DT

        fast_alpha = _ema_alpha(BRAIN_DT, FAST_PROGRESS_TAU_S)
        slow_alpha = _ema_alpha(BRAIN_DT, SLOW_PROGRESS_TAU_S)
        fast_closing_rate_m_s += fast_alpha * (closing_rate_m_s - fast_closing_rate_m_s)
        slow_closing_rate_m_s += slow_alpha * (closing_rate_m_s - slow_closing_rate_m_s)

        if slow_closing_rate_m_s > 1e-6:
            progress_drop_ratio = max(
                0.0,
                min(1.0, (slow_closing_rate_m_s - fast_closing_rate_m_s) / slow_closing_rate_m_s),
            )
        else:
            progress_drop_ratio = 0.0

        dramatic_progress_drop = max(
            0.0,
            (progress_drop_ratio - DRAMATIC_PROGRESS_DROP_RATIO) / max(1.0 - DRAMATIC_PROGRESS_DROP_RATIO, 1e-6),
        )
        target_noise_scale = DEFAULT_MOTOR_NOISE_SCALE + (
            dramatic_progress_drop * (MAX_MOTOR_NOISE_SCALE - DEFAULT_MOTOR_NOISE_SCALE)
        )
        noise_tau_s = NOISE_ATTACK_TAU_S if target_noise_scale > noise_scale else NOISE_RELEASE_TAU_S
        noise_scale += _ema_alpha(BRAIN_DT, noise_tau_s) * (target_noise_scale - noise_scale)

        # Dense reward: negative distance each step — maximising this = reaching goal.
        # Using -dist directly avoids any sign confusion; ES normalises across
        # candidates so the absolute scale doesn't matter.
        reward = -dist

        # Extra bonus for closing distance this step (shapes gradient toward goal)
        reward += (prev_dist - dist) * 5.0

        imu_roll_rad = _wrap_angle_pi(robot.body.imu.rotation_xyz_rad[0])
        abs_roll_rad = abs(imu_roll_rad)
        on_long_thin_side = LONG_THIN_SIDE_ROLL_MIN_RAD <= abs_roll_rad <= LONG_THIN_SIDE_ROLL_MAX_RAD
        if on_long_thin_side:
            long_thin_side_dwell_s += BRAIN_DT
        else:
            long_thin_side_dwell_s = 0.0

        if long_thin_side_dwell_s >= LONG_THIN_SIDE_STUCK_DELAY_S:
            long_thin_side_stuck = True
            reward -= LONG_THIN_SIDE_PENALTY_PER_S * BRAIN_DT

        if long_thin_side_stuck:
            imu_roll_delta_rad = abs(_wrap_angle_pi(imu_roll_rad - prev_imu_roll_rad))
            reward += imu_roll_delta_rad * STUCK_IMU_ROLL_CHANGE_REWARD_PER_RAD
            if not on_long_thin_side:
                reward += SELF_RIGHT_EXIT_BONUS
                long_thin_side_stuck = False
                long_thin_side_dwell_s = 0.0

        total_reward += reward
        prev_dist = dist
        prev_imu_roll_rad = imu_roll_rad

        if on_step is not None and step_i % report_every == 0:
            motor_vels = [float(np.clip(motor_cmds[i] * MOTOR_SCALE, -MAX_MOTOR_RAD_S, MAX_MOTOR_RAD_S))
                          for i in range(len(leg_names))]
            on_step(
                _snapshot(
                    env,
                    goal,
                    step_i,
                    steps,
                    total_reward,
                    leg_names,
                    motor_vels,
                    noise_scale,
                    closing_rate_m_s,
                    progress_drop_ratio,
                )
            )

    return total_reward


def _snapshot(
    env: QuadrupedEnvironment,
    goal: tuple[float, float, float],
    step: int,
    total_steps: int,
    reward_so_far: float,
    leg_names: list[str] | None = None,
    motor_vels: list[float] | None = None,
    noise_scale: float | None = None,
    closing_rate_m_s: float | None = None,
    progress_drop_ratio: float | None = None,
) -> dict[str, Any]:
    robot = env.robot
    body = robot.body
    legs_data = []
    for leg in robot.legs:
        state = env.leg_force_states.get(leg.name)
        foot = state.foot_position_xyz_m if state else (0.0, 0.0, 0.0)
        mount_w = env._body_point_world(leg.mount_point_xyz_m)
        com_off = env._body_vector_world(leg.com_offset_from_mount_m())
        leg_com = (mount_w[0] + com_off[0], mount_w[1] + com_off[1], mount_w[2] + com_off[2])
        legs_data.append({
            "name": leg.name,
            "mount": list(mount_w),
            "foot": list(foot),
            "com": list(leg_com),
            "angle_rad": leg.angle_rad,
            "contact_mode": state.contact_mode if state else "airborne",
        })
    com = env.center_of_mass_xyz_m()
    return {
        "type": "step",
        "step": step,
        "total_steps": total_steps,
        "reward": reward_so_far,
        "goal": list(goal),
        "body": {
            "pos": list(body.position_xyz_m),
            "rot": list(body.rotation_xyz_rad),
            "com": list(body.position_xyz_m),
            "corners": [
                list(env._body_point_world(corner))
                for corner in body.corners_body_frame()
            ],
        },
        "com": list(com),
        "legs": legs_data,
        "motors": [
            {
                "name": leg.name,
                "target_velocity_rad_s": (
                    motor_vels[index] if motor_vels is not None and index < len(motor_vels)
                    else leg.motor.target_velocity_rad_s
                ),
            }
            for index, leg in enumerate(robot.legs)
        ],
        "noise_scale": noise_scale,
        "closing_rate_m_s": closing_rate_m_s,
        "progress_drop_ratio": progress_drop_ratio,
        "time_s": env.time_s,
    }


class ESTrainer:
    """OpenAI-ES trainer."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.brain = SNNBrain(seed=seed)
        self.state = TrainingState()
        self._params = self.brain.get_params()

    def _random_goal(self) -> tuple[float, float, float]:
        angle = self.rng.uniform(0, 2 * math.pi)
        radius = self.rng.uniform(0.5, FIELD_HALF * 0.8)
        return (radius * math.cos(angle), radius * math.sin(angle), GOAL_HEIGHT_M)

    def run_generation(self, on_step: Any = None, on_gen_done: Any = None) -> None:
        goal = self._random_goal()
        self.state.goal_xyz = goal

        noise = self.rng.normal(0.0, SIGMA, (POP_SIZE, len(self._params))).astype(np.float32)
        returns = np.zeros(POP_SIZE, dtype=np.float32)

        for k in range(POP_SIZE):
            candidate = SNNBrain()
            candidate.set_params(self._params + noise[k])
            # Only stream steps for the first candidate of each generation
            cb = on_step if k == 0 else None
            returns[k] = _run_episode(candidate, goal, cb)

        # Normalize returns
        std = returns.std()
        if std > 1e-6:
            normalized = (returns - returns.mean()) / std
        else:
            normalized = np.zeros_like(returns)

        # Gradient estimate
        grad = (noise.T @ normalized) / (POP_SIZE * SIGMA)
        self._params = self._params + LR * grad

        self.brain.set_params(self._params)
        self.state.generation += 1
        self.state.mean_reward = float(returns.mean())
        self.state.best_reward = max(self.state.best_reward, float(returns.max()))
        self.state.rewards_history.append(self.state.mean_reward)

        if on_gen_done is not None:
            on_gen_done({
                "type": "generation",
                "generation": self.state.generation,
                "mean_reward": self.state.mean_reward,
                "best_reward": self.state.best_reward,
                "rewards_history": self.state.rewards_history[-100:],
                "goal": list(goal),
            })

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "params": self._params.astype(np.float32),
            "generation": self.state.generation,
            "best_reward": self.state.best_reward,
            "mean_reward": self.state.mean_reward,
            "episode_reward": self.state.episode_reward,
            "goal_xyz": np.array(self.state.goal_xyz, dtype=np.float32),
            "rewards_history": np.array(self.state.rewards_history, dtype=np.float32),
            "rng_state_json": json.dumps(self.rng.bit_generator.state),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_path, **self.checkpoint_dict())
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            self._params = checkpoint["params"].astype(np.float32)
            self.brain.set_params(self._params)
            self.state.generation = int(checkpoint["generation"])
            self.state.best_reward = float(checkpoint["best_reward"])
            self.state.mean_reward = float(checkpoint["mean_reward"])
            self.state.episode_reward = float(checkpoint["episode_reward"])
            self.state.goal_xyz = tuple(float(v) for v in checkpoint["goal_xyz"].tolist())
            self.state.rewards_history = [float(v) for v in checkpoint["rewards_history"].tolist()]
            self.rng.bit_generator.state = json.loads(checkpoint["rng_state_json"].item())
