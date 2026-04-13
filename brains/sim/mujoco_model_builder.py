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


def _build_step_strips(spec: RuntimeSpec, friction: float, contact_margin_m: float) -> list[str]:
    terrain = spec.terrain
    floor_height = float(spec.terrain.floor_height_m)
    strips: list[str] = []
    rgba_cycle = (
        "0.16 0.21 0.15 1",
        "0.20 0.29 0.18 1",
        "0.27 0.39 0.22 1",
        "0.37 0.49 0.26 1",
        "0.49 0.59 0.31 1",
    )
    for level in range(1, terrain.step_count + 1):
        inner = terrain.center_half_m + ((level - 1) * terrain.step_width_m)
        outer = terrain.center_half_m + (level * terrain.step_width_m)
        top = floor_height + (level * terrain.step_height_m)
        half_height = max((top - floor_height) * 0.5, 1e-3)
        center_z = floor_height + half_height
        strip_half = max((outer - inner) * 0.5, 1e-3)
        color = rgba_cycle[min(level - 1, len(rgba_cycle) - 1)]
        common = (
            f'type="box" friction="{friction:.4f} 0.01 0.001" margin="{contact_margin_m:.6f}" '
            f'rgba="{color}"'
        )

        strips.append(
            f'<geom name="step_{level}_north" {common} pos="0 {(inner + outer) * 0.5:.6f} {center_z:.6f}" '
            f'size="{outer:.6f} {strip_half:.6f} {half_height:.6f}"/>'
        )
        strips.append(
            f'<geom name="step_{level}_south" {common} pos="0 {-((inner + outer) * 0.5):.6f} {center_z:.6f}" '
            f'size="{outer:.6f} {strip_half:.6f} {half_height:.6f}"/>'
        )
        strips.append(
            f'<geom name="step_{level}_east" {common} pos="{((inner + outer) * 0.5):.6f} 0 {center_z:.6f}" '
            f'size="{strip_half:.6f} {inner:.6f} {half_height:.6f}"/>'
        )
        strips.append(
            f'<geom name="step_{level}_west" {common} pos="{-((inner + outer) * 0.5):.6f} 0 {center_z:.6f}" '
            f'size="{strip_half:.6f} {inner:.6f} {half_height:.6f}"/>'
        )
    return strips


def build_mujoco_model(spec: RuntimeSpec) -> MujocoModelArtifacts:
    mujoco_spec = spec.simulator.mujoco
    leg_mounts = mount_points_body(spec)
    body_half_sizes = body_half_extents(spec)
    leg_length_m = float(spec.robot.leg_length_m)
    leg_radius_m = float(spec.robot.leg_radius_m)
    foot_radius_m = float(spec.robot.foot_radius_m)
    leg_mass_kg = float(spec.robot.leg_mass_kg)
    static_friction = float(spec.friction.foot_static)
    body_mass_kg = float(spec.robot.body_mass_kg)
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
                f'friction="{static_friction:.4f} 0.01 0.001" rgba="0.68 0.73 0.75 1"/>'
                f'<geom name="{leg_name}_foot" type="sphere" pos="0 0 {-leg_length_m:.6f}" '
                f'size="{foot_radius_m:.6f}" mass="{leg_mass_kg * 0.08:.6f}" '
                f'friction="{static_friction:.4f} 0.01 0.001" rgba="0.93 0.77 0.44 1"/>'
                f'<site name="{foot_site_name}" pos="0 0 {-leg_length_m:.6f}" size="{max(foot_radius_m * 0.5, 0.003):.6f}"/>'
                f"</body>"
            )
        )

    ground_geoms = [
        (
            f'<geom name="ground" type="plane" pos="0 0 {spec.terrain.floor_height_m:.6f}" '
            f'size="{spec.terrain.field_half_m:.6f} {spec.terrain.field_half_m:.6f} 0.1" '
            f'friction="{static_friction:.4f} 0.01 0.001" '
            f'margin="{mujoco_spec.contact_margin_m:.6f}" rgba="0.07 0.12 0.08 1"/>'
        )
    ]
    if spec.terrain.kind == "stepped_arena":
        ground_geoms.extend(
            _build_step_strips(
                spec,
                friction=static_friction,
                contact_margin_m=mujoco_spec.contact_margin_m,
            )
        )

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
