"""Shared MuJoCo robot geometry helpers derived from runtime config."""

from __future__ import annotations

from brains.config import RuntimeSpec


LEG_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
LEG_ROTATION_AXIS_BODY = (0.0, 1.0, 0.0)


def body_half_extents(spec: RuntimeSpec) -> tuple[float, float, float]:
    return (
        float(spec.robot.body_length_m) * 0.5,
        float(spec.robot.body_width_m) * 0.5,
        float(spec.robot.body_height_m) * 0.5,
    )


def body_corners_body(spec: RuntimeSpec) -> tuple[tuple[float, float, float], ...]:
    hx, hy, hz = body_half_extents(spec)
    corners: list[tuple[float, float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append((sx * hx, sy * hy, sz * hz))
    return tuple(corners)


def body_principal_inertia(spec: RuntimeSpec) -> tuple[float, float, float]:
    length_m = float(spec.robot.body_length_m)
    width_m = float(spec.robot.body_width_m)
    height_m = float(spec.robot.body_height_m)
    mass_kg = float(spec.robot.body_mass_kg)
    length_sq = length_m * length_m
    width_sq = width_m * width_m
    height_sq = height_m * height_m
    return (
        (mass_kg / 12.0) * (width_sq + height_sq),
        (mass_kg / 12.0) * (length_sq + height_sq),
        (mass_kg / 12.0) * (length_sq + width_sq),
    )


def mount_points_body(spec: RuntimeSpec) -> tuple[tuple[float, float, float], ...]:
    half_length = float(spec.robot.body_length_m) * 0.5
    half_width = float(spec.robot.body_width_m) * 0.5
    return (
        (half_length, half_width, 0.0),
        (half_length, -half_width, 0.0),
        (-half_length, half_width, 0.0),
        (-half_length, -half_width, 0.0),
    )


def leg_inertia_about_mount(spec: RuntimeSpec) -> tuple[float, ...]:
    leg_inertia = float(spec.robot.leg_mass_kg) * (float(spec.robot.leg_length_m) ** 2) / 3.0
    return tuple(leg_inertia for _ in LEG_NAMES)


def leg_body_fractions(spec: RuntimeSpec) -> tuple[float, ...]:
    sample_count = int(spec.robot.leg_body_samples)
    step = 1.0 / float(sample_count + 1)
    return tuple((index + 1) * step for index in range(sample_count))


def total_robot_mass_kg(spec: RuntimeSpec) -> float:
    return float(spec.robot.body_mass_kg) + (len(LEG_NAMES) * float(spec.robot.leg_mass_kg))
