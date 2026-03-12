from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IMU:
    rotation_xyz_rad: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration_xyz_m_s2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def update_rotation(self, x_rad: float, y_rad: float, z_rad: float) -> None:
        self.rotation_xyz_rad[:] = [x_rad, y_rad, z_rad]

    def update_acceleration(self, x_m_s2: float, y_m_s2: float, z_m_s2: float) -> None:
        self.acceleration_xyz_m_s2[:] = [x_m_s2, y_m_s2, z_m_s2]
