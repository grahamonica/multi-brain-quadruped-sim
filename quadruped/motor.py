from __future__ import annotations

from dataclasses import dataclass, field
from math import copysign, exp


@dataclass
class Motor:
    target_velocity_rad_s: float = 0.0
    current_velocity_rad_s: float = 0.0
    max_angular_acceleration_rad_s2: float = 18.0
    # Back-EMF and gear friction: natural velocity decay rate when unpowered.
    viscous_damping_per_s: float = 8.0
    # First-order low-pass filter time constant on the commanded velocity.
    # Smooths out step changes from the UI slider so the motor ramps toward
    # a new setpoint gradually instead of snapping instantly, preventing the
    # impulsive torques that cause visible jolts.
    velocity_filter_tau_s: float = 0.05
    # Internal filter state — tracks the smoothed setpoint.  Reset alongside
    # target_velocity_rad_s whenever the motor is zeroed.
    smoothed_target_rad_s: float = field(default=0.0, repr=False)

    def set_velocity(self, velocity_rad_s: float) -> None:
        self.target_velocity_rad_s = velocity_rad_s

    def step(self, dt_s: float) -> float:
        if dt_s <= 0.0:
            return 0.0

        # Smooth the commanded setpoint with a first-order low-pass filter.
        # alpha = 1 - exp(-dt/tau): fraction of the gap closed each step.
        smooth_alpha = 1.0 - exp(-dt_s / max(self.velocity_filter_tau_s, 1e-6))
        self.smoothed_target_rad_s += smooth_alpha * (self.target_velocity_rad_s - self.smoothed_target_rad_s)

        prev_velocity = self.current_velocity_rad_s

        # Apply viscous friction (back-EMF + gear losses).
        self.current_velocity_rad_s *= exp(-self.viscous_damping_per_s * dt_s)

        velocity_error = self.smoothed_target_rad_s - self.current_velocity_rad_s
        max_velocity_change = self.max_angular_acceleration_rad_s2 * dt_s

        if abs(velocity_error) <= max_velocity_change:
            self.current_velocity_rad_s = self.smoothed_target_rad_s
        else:
            self.current_velocity_rad_s += copysign(max_velocity_change, velocity_error)

        # Return true angular acceleration so leg inertia forces on the body
        # are computed correctly.
        return (self.current_velocity_rad_s - prev_velocity) / dt_s
