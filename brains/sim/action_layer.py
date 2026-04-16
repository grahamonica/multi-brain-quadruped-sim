"""Map policy outputs to motor targets.

This layer supports two actuation modes:
- motor_targets: policy outputs map directly to normalized motor velocities.
- command_primitives: policy outputs choose a scripted locomotion primitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


SUPPORTED_COMMANDS = {
    "stand",
    "trot",
    "turn_left",
    "turn_right",
    "walk",
    "skip",
    "front_flip",
    "back_flip",
    "side_roll",
    "back_up",
    "flip",
}


@dataclass(frozen=True)
class ActionProjection:
    target_velocity_rad_s: np.ndarray
    policy_output: np.ndarray
    selected_command: str | None


@dataclass
class ActionProjector:
    mode: str
    command_vocabulary: tuple[str, ...] = ()
    default_command_speed: float = 0.45
    command_update_interval_s: float = 0.60
    command_default_duration_s: float = 1.80
    command_max_duration_s: float = 4.00
    _active_command: str | None = field(default=None, init=False, repr=False)
    _active_command_speed: float = field(default=0.0, init=False, repr=False)
    _next_update_time_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"motor_targets", "command_primitives"}:
            raise ValueError("control.mode must be 'motor_targets' or 'command_primitives'.")
        if self.command_update_interval_s <= 0.0:
            raise ValueError("control.command_update_interval_s must be > 0.")
        if self.command_default_duration_s < self.command_update_interval_s:
            raise ValueError(
                "control.command_default_duration_s must be >= control.command_update_interval_s."
            )
        if self.command_max_duration_s < self.command_default_duration_s:
            raise ValueError(
                "control.command_max_duration_s must be >= control.command_default_duration_s."
            )
        if self.mode == "command_primitives":
            if len(self.command_vocabulary) == 0:
                raise ValueError("control.command_vocabulary must be non-empty in command_primitives mode.")
            known = ", ".join(sorted(SUPPORTED_COMMANDS))
            for name in self.command_vocabulary:
                if name not in SUPPORTED_COMMANDS:
                    raise ValueError(f"Unknown command {name!r}. Supported commands: {known}.")

    def reset(self) -> None:
        self._active_command = None
        self._active_command_speed = 0.0
        self._next_update_time_s = 0.0

    def project(
        self,
        policy_output: np.ndarray,
        *,
        time_s: float,
        max_motor_rad_s: float,
        motor_scale: float,
    ) -> ActionProjection:
        raw = np.asarray(policy_output, dtype=np.float32)
        if self.mode == "motor_targets":
            target = np.clip(raw * np.float32(motor_scale), -np.float32(max_motor_rad_s), np.float32(max_motor_rad_s))
            return ActionProjection(target_velocity_rad_s=target, policy_output=raw, selected_command=None)

        required = len(self.command_vocabulary)
        if raw.ndim != 1 or raw.shape[0] not in {required, required + 1}:
            raise ValueError(
                "command_primitives mode requires policy output length to equal "
                "control.command_vocabulary length or that length + 1 (duration head). "
                f"expected {required} or {required + 1}, received {raw.shape[0]}."
            )

        should_update = self._active_command is None or float(time_s) >= self._next_update_time_s
        if should_update:
            logits = raw[:required]
            command_index = int(np.argmax(logits))
            next_command = self.command_vocabulary[command_index]
            command_speed = float(np.clip(abs(float(logits[command_index])), 0.0, 1.0))
            if command_speed < 1e-3:
                command_speed = float(self.default_command_speed)

            hold_duration_s = float(self.command_default_duration_s)
            if raw.shape[0] == required + 1:
                hold_scalar = float(np.clip((float(raw[-1]) + 1.0) * 0.5, 0.0, 1.0))
                hold_duration_s = float(
                    self.command_update_interval_s
                    + (hold_scalar * (self.command_max_duration_s - self.command_update_interval_s))
                )

            self._active_command = next_command
            self._active_command_speed = command_speed
            self._next_update_time_s = float(time_s) + hold_duration_s

        assert self._active_command is not None
        target = command_target_velocity(
            self._active_command,
            time_s,
            self._active_command_speed,
            max_motor_rad_s,
        )
        return ActionProjection(
            target_velocity_rad_s=target,
            policy_output=raw,
            selected_command=self._active_command,
        )


def command_target_velocity(
    command: str,
    time_s: float,
    speed: float,
    max_motor_rad_s: float,
) -> np.ndarray:
    amplitude = np.float32(max_motor_rad_s * np.clip(speed, 0.0, 1.0))
    stride_hz = 0.35
    phase = math.tau * math.fmod(stride_hz * float(time_s), 1.0)
    wave = np.float32(math.sin(phase))
    diagonal = np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    front_back = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    left_right = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32)

    def front_flip_sequence() -> np.ndarray:
        flip_hz = 0.38
        flip_u = math.fmod(flip_hz * float(time_s), 1.0)
        phase_a = np.asarray([1.35, 1.35, -0.45, -0.45], dtype=np.float32)
        phase_b = np.asarray([-0.45, -0.45, 1.35, 1.35], dtype=np.float32)
        # Symmetric reorientation pulse brings the cycle back to neutral.
        phase_c = np.asarray([-0.85, -0.85, -0.85, -0.85], dtype=np.float32)

        if flip_u < 0.33:
            window = np.float32(math.sin(math.pi * (flip_u / 0.33)))
            return window * phase_a
        if flip_u < 0.66:
            window = np.float32(math.sin(math.pi * ((flip_u - 0.33) / 0.33)))
            return window * phase_b
        window = np.float32(math.sin(math.pi * ((flip_u - 0.66) / 0.34)))
        return window * phase_c

    if command in {"stand", "stop"}:
        return np.zeros((4,), dtype=np.float32)
    if command == "trot":
        return np.float32(0.55) * amplitude * wave * diagonal
    if command == "walk":
        return np.float32(0.35) * amplitude * wave * front_back
    if command == "skip":
        return np.float32(0.45) * amplitude * np.float32(math.sin(2.0 * phase)) * diagonal
    if command == "back_up":
        return -np.float32(0.55) * amplitude * wave * diagonal
    if command == "turn_left":
        return amplitude * ((np.float32(0.50) * wave * diagonal) + (np.float32(0.18) * wave * left_right))
    if command == "turn_right":
        return amplitude * ((np.float32(0.50) * wave * diagonal) - (np.float32(0.18) * wave * left_right))
    if command == "front_flip":
        return np.float32(1.20) * amplitude * front_flip_sequence()
    if command == "back_flip":
        return -np.float32(1.20) * amplitude * front_flip_sequence()
    if command == "side_roll":
        roll_wave = np.float32(math.sin(math.tau * 1.5 * float(time_s)))
        return np.float32(0.75) * amplitude * roll_wave * left_right
    if command == "flip":
        flip_phase = np.float32(math.sin(2.0 * math.pi * 3.8 * float(time_s)))
        return amplitude * flip_phase * np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32)

    raise ValueError(f"Unsupported command {command!r}.")
