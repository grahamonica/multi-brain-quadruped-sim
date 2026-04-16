"""Generate MuJoCo MJCF directly from runtime config."""

from __future__ import annotations

from dataclasses import dataclass

from brains.config import RuntimeSpec

from .mujoco_layout import LEG_NAMES, LEG_ROTATION_AXIS_BODY, body_half_extents, mount_points_body


@dataclass(frozen=True)
class MujocoModelArtifacts:
    xml: str
    body_name: str
    freejoint_name: str
    leg_joint_names: tuple[str, ...]
    leg_body_names: tuple[str, ...]
    foot_site_names: tuple[str, ...]
    actuator_names: tuple[str, ...]


def _float_list(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def build_mujoco_model(spec: RuntimeSpec) -> MujocoModelArtifacts:
    mujoco_spec = spec.simulator.mujoco
    leg_mounts = mount_points_body(spec)
    body_half_sizes = body_half_extents(spec)
    leg_length_m = float(spec.robot.leg_length_m)
    leg_radius_m = float(spec.robot.leg_radius_m)
    body_mass_kg = float(spec.robot.body_mass_kg)
    foot_radius_m = leg_radius_m
    leg_mass_kg = float(spec.robot.leg_mass_kg)
    static_friction = float(spec.friction.foot_static)
    kinetic_friction = float(spec.friction.foot_kinetic)
    body_friction = float(spec.friction.body)

    body_name = "torso"
    freejoint_name = "root_free"
    leg_joint_names: list[str] = []
    leg_body_names: list[str] = []
    foot_site_names: list[str] = []
    actuator_names: list[str] = []
    leg_bodies: list[str] = []

    joint_min, joint_max = mujoco_spec.joint_range_rad
    for leg_name, leg_mount in zip(LEG_NAMES, leg_mounts, strict=True):
        joint_name = f"{leg_name}_hinge"
        body_name_for_leg = f"{leg_name}_leg"
        foot_site_name = f"{leg_name}_foot_site"
        actuator_name = f"{leg_name}_motor"
        leg_joint_names.append(joint_name)
        leg_body_names.append(body_name_for_leg)
        foot_site_names.append(foot_site_name)
        actuator_names.append(actuator_name)
        leg_bodies.append(
            (
                f'<body name="{body_name_for_leg}" pos="{_float_list(list(leg_mount))}">'
                f'<joint name="{joint_name}" type="hinge" axis="{_float_list(list(LEG_ROTATION_AXIS_BODY))}" '
                f'range="{joint_min:.6f} {joint_max:.6f}" damping="{spec.robot.motor_viscous_damping_per_s:.6f}"/>'
                f'<geom name="{leg_name}_capsule" type="capsule" fromto="0 0 0 0 0 {-leg_length_m:.6f}" '
                f'size="{leg_radius_m:.6f}" mass="{leg_mass_kg * 0.92:.6f}" '
                f'friction="{body_friction:.4f} 0.0012 0.00008" rgba="0.68 0.73 0.75 1"/>'
                f'<geom name="{leg_name}_foot" type="sphere" pos="0 0 {-leg_length_m:.6f}" '
                f'size="{foot_radius_m:.6f}" mass="{leg_mass_kg * 0.08:.6f}" '
                f'friction="{static_friction:.4f} 0.0012 0.00008" priority="2" rgba="0.93 0.77 0.44 1"/>'
                f'<site name="{foot_site_name}" pos="0 0 {-leg_length_m:.6f}" size="{max(foot_radius_m * 0.5, 0.003):.6f}"/>'
                f"</body>"
            )
        )

    ground_geoms = [
        (
            f'<geom name="ground" type="plane" pos="0 0 {spec.terrain.floor_height_m:.6f}" '
            f'size="{spec.terrain.field_half_m:.6f} {spec.terrain.field_half_m:.6f} 0.1" '
            f'friction="{static_friction:.4f} 0.0025 0.0001" material="ground_grid" '
            f'margin="{mujoco_spec.contact_margin_m:.6f}"/>'
        )
    ]

    actuators = [
        (
            f'<motor name="{actuator_name}" joint="{joint_name}" gear="1" '
            f'ctrllimited="true" ctrlrange="{-mujoco_spec.actuator_force_limit:.6f} {mujoco_spec.actuator_force_limit:.6f}"/>'
        )
        for actuator_name, joint_name in zip(actuator_names, leg_joint_names, strict=True)
    ]

    xml = (
        f'<mujoco model="{spec.name}">'
        '<compiler angle="radian" autolimits="true" inertiafromgeom="true"/>'
        f'<option timestep="{mujoco_spec.timestep_s:.6f}" gravity="0 0 {-spec.physics.gravity_m_s2:.6f}" '
        f'integrator="{mujoco_spec.integrator}" solver="{mujoco_spec.solver}" '
        f'iterations="{mujoco_spec.solver_iterations}" ls_iterations="{mujoco_spec.line_search_iterations}" '
        f'noslip_iterations="{mujoco_spec.noslip_iterations}"/>'
        '<asset>'
        '<texture name="ground_checker" type="2d" builtin="checker" width="512" height="512" '
        'rgb1="0.08 0.13 0.09" rgb2="0.16 0.23 0.17"/>'
        '<material name="ground_grid" texture="ground_checker" texuniform="true" texrepeat="18 18" reflectance="0.03"/>'
        '</asset>'
        '<default>'
        f'<geom contype="1" conaffinity="1" condim="4" margin="{mujoco_spec.contact_margin_m:.6f}" solref="0.005 1"/>'
        '<joint limited="true" armature="0.01"/>'
        '</default>'
        '<visual>'
        '<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>'
        '</visual>'
        '<worldbody>'
        '<light pos="0 0 5" dir="0 0 -1" directional="true"/>'
        + "".join(ground_geoms)
        + (
            f'<body name="{body_name}" pos="0 0 0">'
            f'<freejoint name="{freejoint_name}"/>'
            f'<geom name="torso_geom" type="box" size="{_float_list(list(body_half_sizes))}" '
            f'mass="{body_mass_kg:.6f}" friction="{body_friction:.4f} 0.01 0.001" '
            'rgba="0.30 0.60 0.85 1"/>'
            f'{"".join(leg_bodies)}'
            "</body>"
        )
        + '</worldbody>'
        + '<actuator>'
        + "".join(actuators)
        + '</actuator>'
        + '</mujoco>'
    )

    return MujocoModelArtifacts(
        xml=xml,
        body_name=body_name,
        freejoint_name=freejoint_name,
        leg_joint_names=tuple(leg_joint_names),
        leg_body_names=tuple(leg_body_names),
        foot_site_names=tuple(foot_site_names),
        actuator_names=tuple(actuator_names),
    )
