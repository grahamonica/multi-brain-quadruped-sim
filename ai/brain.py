"""Layered spiking neural network brain for the quadruped.

Inputs (48 total):
  [0:3]   goal_xyz          – goal point world coords
  [3:6]   com_xyz           – total centre of mass world coords
  [6:9]   body_com_xyz      – body centre of mass world coords
  [9:21]  leg_foot_xyz      – world coords of each leg tip (4×3)
  [21:33] leg_com_xyz       – world coords of each leg COM  (4×3)
  [33:36] body_imu_rad      – body IMU roll/pitch/yaw
  [36:48] leg_imu_rad       – each leg IMU roll/pitch/yaw   (4×3)

Outputs (4): one target angular velocity per leg, in [-1, 1] (scaled by caller).

Architecture
------------
This brain uses 4 hidden LIF layers. Every layer contains:
  1. a shared coordination trunk used by all motors
  2. four motor-specific lanes, one per motor

All motors see the same observation, but each layer also lets motor lanes
exchange information indirectly through the shared trunk. This preserves
motor-local specialization while enabling cross-motor coordination.

Exploration noise
-----------------
Gaussian noise is added to the motor outputs each step. The caller can adjust
that scale dynamically, so exploration rises when goal progress stalls and
relaxes when progress recovers.
"""
from __future__ import annotations

import numpy as np


# Network sizes
N_IN = 48
N_OUT = 4
N_HIDDEN_LAYERS = 4
SHARED_TRUNK_WIDTH = 64
MOTOR_LANE_WIDTH = 64

# LIF parameters
TAU_MEM = 0.020
V_THRESH = 0.01
V_RESET = 0.0
DT = 0.010

# Default exploration noise added to each motor output step.
DEFAULT_MOTOR_NOISE_SCALE = 0.60


class SNNBrain:
    """Four-layer LIF network with a shared trunk and motor-specific lanes."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        rng = self._rng

        self.w_in_shared: np.ndarray = rng.normal(
            0.0,
            1.0 / N_IN**0.5,
            (SHARED_TRUNK_WIDTH, N_IN),
        ).astype(np.float32)
        self.w_in_motor: np.ndarray = rng.normal(
            0.0,
            1.0 / N_IN**0.5,
            (N_OUT, MOTOR_LANE_WIDTH, N_IN),
        ).astype(np.float32)

        self.w_shared_from_shared: np.ndarray = rng.normal(
            0.0,
            1.0 / SHARED_TRUNK_WIDTH**0.5,
            (N_HIDDEN_LAYERS - 1, SHARED_TRUNK_WIDTH, SHARED_TRUNK_WIDTH),
        ).astype(np.float32)
        self.w_shared_from_motors: np.ndarray = rng.normal(
            0.0,
            1.0 / (N_OUT * MOTOR_LANE_WIDTH) ** 0.5,
            (N_HIDDEN_LAYERS - 1, SHARED_TRUNK_WIDTH, N_OUT * MOTOR_LANE_WIDTH),
        ).astype(np.float32)
        self.w_motor_from_motor: np.ndarray = rng.normal(
            0.0,
            1.0 / MOTOR_LANE_WIDTH**0.5,
            (N_HIDDEN_LAYERS - 1, N_OUT, MOTOR_LANE_WIDTH, MOTOR_LANE_WIDTH),
        ).astype(np.float32)
        self.w_motor_from_shared: np.ndarray = rng.normal(
            0.0,
            1.0 / SHARED_TRUNK_WIDTH**0.5,
            (N_HIDDEN_LAYERS - 1, N_OUT, MOTOR_LANE_WIDTH, SHARED_TRUNK_WIDTH),
        ).astype(np.float32)

        self.b_shared: np.ndarray = rng.uniform(
            -0.2,
            0.2,
            (N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH),
        ).astype(np.float32)
        self.b_motor: np.ndarray = rng.uniform(
            -0.2,
            0.2,
            (N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH),
        ).astype(np.float32)

        self.w_out_shared: np.ndarray = np.zeros((N_OUT, SHARED_TRUNK_WIDTH), dtype=np.float32)
        self.w_out_motor: np.ndarray = np.zeros((N_OUT, MOTOR_LANE_WIDTH), dtype=np.float32)
        self.b_out: np.ndarray = np.zeros(N_OUT, dtype=np.float32)

        self._v_shared: np.ndarray = np.zeros((N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH), dtype=np.float32)
        self._trace_shared: np.ndarray = np.zeros((N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH), dtype=np.float32)
        self._v_motor: np.ndarray = np.zeros((N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH), dtype=np.float32)
        self._trace_motor: np.ndarray = np.zeros((N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH), dtype=np.float32)
        self._trace_decay: float = 0.70

    def reset(self) -> None:
        self._v_shared.fill(0.0)
        self._trace_shared.fill(0.0)
        self._v_motor.fill(0.0)
        self._trace_motor.fill(0.0)

    def step(
        self,
        obs: np.ndarray,
        dt: float = DT,
        noise: bool = True,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        """Run one timestep; returns motor commands in [-1, 1]."""
        scale = dt / TAU_MEM
        decay = 1.0 - scale

        for layer_index in range(N_HIDDEN_LAYERS):
            if layer_index == 0:
                shared_input = np.einsum("hi,i->h", self.w_in_shared, obs) + self.b_shared[layer_index]
                motor_input = np.einsum("mhi,i->mh", self.w_in_motor, obs) + self.b_motor[layer_index]
            else:
                prev_shared = self._trace_shared[layer_index - 1]
                prev_motor = self._trace_motor[layer_index - 1]
                prev_motor_flat = prev_motor.reshape(-1)

                shared_input = (
                    np.einsum(
                        "hj,j->h",
                        self.w_shared_from_shared[layer_index - 1],
                        prev_shared,
                    )
                    + np.einsum(
                        "hj,j->h",
                        self.w_shared_from_motors[layer_index - 1],
                        prev_motor_flat,
                    )
                    + self.b_shared[layer_index]
                )
                motor_input = (
                    np.einsum(
                        "mhj,mj->mh",
                        self.w_motor_from_motor[layer_index - 1],
                        prev_motor,
                    )
                    + np.einsum(
                        "mhj,j->mh",
                        self.w_motor_from_shared[layer_index - 1],
                        prev_shared,
                    )
                    + self.b_motor[layer_index]
                )

            self._v_shared[layer_index] = (self._v_shared[layer_index] * decay) + (shared_input * scale)
            shared_spikes = (self._v_shared[layer_index] >= V_THRESH).astype(np.float32)
            self._v_shared[layer_index][shared_spikes > 0] = V_RESET
            self._trace_shared[layer_index] = (
                (self._trace_shared[layer_index] * self._trace_decay)
                + (shared_spikes * (1.0 - self._trace_decay))
            )

            self._v_motor[layer_index] = (self._v_motor[layer_index] * decay) + (motor_input * scale)
            motor_spikes = (self._v_motor[layer_index] >= V_THRESH).astype(np.float32)
            self._v_motor[layer_index][motor_spikes > 0] = V_RESET
            self._trace_motor[layer_index] = (
                (self._trace_motor[layer_index] * self._trace_decay)
                + (motor_spikes * (1.0 - self._trace_decay))
            )

        out = np.tanh(
            np.einsum("mh,mh->m", self.w_out_motor, self._trace_motor[-1])
            + np.einsum("mh,h->m", self.w_out_shared, self._trace_shared[-1])
            + self.b_out
        )
        if noise:
            active_noise_scale = DEFAULT_MOTOR_NOISE_SCALE if noise_scale is None else max(noise_scale, 0.0)
            out = out + self._rng.standard_normal(N_OUT).astype(np.float32) * active_noise_scale
        return np.clip(out, -1.0, 1.0)

    def get_params(self) -> np.ndarray:
        return np.concatenate(
            [
                self.w_in_shared.ravel(),
                self.w_in_motor.ravel(),
                self.w_shared_from_shared.ravel(),
                self.w_shared_from_motors.ravel(),
                self.w_motor_from_motor.ravel(),
                self.w_motor_from_shared.ravel(),
                self.b_shared.ravel(),
                self.b_motor.ravel(),
                self.w_out_shared.ravel(),
                self.w_out_motor.ravel(),
                self.b_out.ravel(),
            ]
        )

    def set_params(self, params: np.ndarray) -> None:
        n0 = self.w_in_shared.size
        n1 = n0 + self.w_in_motor.size
        n2 = n1 + self.w_shared_from_shared.size
        n3 = n2 + self.w_shared_from_motors.size
        n4 = n3 + self.w_motor_from_motor.size
        n5 = n4 + self.w_motor_from_shared.size
        n6 = n5 + self.b_shared.size
        n7 = n6 + self.b_motor.size
        n8 = n7 + self.w_out_shared.size
        n9 = n8 + self.w_out_motor.size

        self.w_in_shared = params[:n0].reshape(self.w_in_shared.shape).astype(np.float32)
        self.w_in_motor = params[n0:n1].reshape(self.w_in_motor.shape).astype(np.float32)
        self.w_shared_from_shared = params[n1:n2].reshape(self.w_shared_from_shared.shape).astype(np.float32)
        self.w_shared_from_motors = params[n2:n3].reshape(self.w_shared_from_motors.shape).astype(np.float32)
        self.w_motor_from_motor = params[n3:n4].reshape(self.w_motor_from_motor.shape).astype(np.float32)
        self.w_motor_from_shared = params[n4:n5].reshape(self.w_motor_from_shared.shape).astype(np.float32)
        self.b_shared = params[n5:n6].reshape(self.b_shared.shape).astype(np.float32)
        self.b_motor = params[n6:n7].reshape(self.b_motor.shape).astype(np.float32)
        self.w_out_shared = params[n7:n8].reshape(self.w_out_shared.shape).astype(np.float32)
        self.w_out_motor = params[n8:n9].reshape(self.w_out_motor.shape).astype(np.float32)
        self.b_out = params[n9:].reshape(self.b_out.shape).astype(np.float32)

    def param_count(self) -> int:
        return (
            self.w_in_shared.size
            + self.w_in_motor.size
            + self.w_shared_from_shared.size
            + self.w_shared_from_motors.size
            + self.w_motor_from_motor.size
            + self.w_motor_from_shared.size
            + self.b_shared.size
            + self.b_motor.size
            + self.w_out_shared.size
            + self.w_out_motor.size
            + self.b_out.size
        )
