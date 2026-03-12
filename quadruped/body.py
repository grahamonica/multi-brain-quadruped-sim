from __future__ import annotations

from dataclasses import dataclass, field

from .imu import IMU


@dataclass
class Body:
    length_m: float
    width_m: float
    height_m: float
    mass_kg: float
    imu: IMU = field(default_factory=IMU)
    position_xyz_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_xyz_m_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration_xyz_m_s2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_xyz_rad: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity_xyz_rad_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_acceleration_xyz_rad_s2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def half_extents(self) -> tuple[float, float, float]:
        return (self.length_m / 2.0, self.width_m / 2.0, self.height_m / 2.0)

    def principal_inertia_kg_m2(self) -> tuple[float, float, float]:
        lx = self.length_m
        wy = self.width_m
        hz = self.height_m
        mass = self.mass_kg
        return (
            (mass / 12.0) * ((wy * wy) + (hz * hz)),
            (mass / 12.0) * ((lx * lx) + (hz * hz)),
            (mass / 12.0) * ((lx * lx) + (wy * wy)),
        )

    def corners_body_frame(self) -> list[tuple[float, float, float]]:
        half_length, half_width, half_height = self.half_extents()
        return [
            (sx * half_length, sy * half_width, sz * half_height)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
