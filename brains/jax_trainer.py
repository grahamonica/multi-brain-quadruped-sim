"""JAX trainer and functional simulator backend."""
from __future__ import annotations

import math
import mujoco
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from brains.config import RuntimeSpec, canonical_config_json, config_json_matches_checkpoint, default_runtime_spec
from brains.models.registry import ModelDefinition, get_model_definition, load_policy_plugin
from brains.sim.mujoco_backend import MuJoCoBackend
from brains.sim.mujoco_layout import (
    LEG_NAMES,
    LEG_ROTATION_AXIS_BODY,
    body_corners_body,
    body_half_extents,
    body_principal_inertia,
    leg_body_fractions,
    leg_inertia_about_mount,
    mount_points_body,
    total_robot_mass_kg,
)


# Robot / physics constants mirror the current KT2-style defaults.
N_IN = 48
N_OUT = 4
N_HIDDEN_LAYERS = 4
SHARED_TRUNK_WIDTH = 32
MOTOR_LANE_WIDTH = 32
N_LEGS = len(LEG_NAMES)
N_BODY_CORNERS = 8

BODY_LENGTH_M = 0.28
BODY_WIDTH_M = 0.12
BODY_HEIGHT_M = 0.02
BODY_MASS_KG = 2.4
LEG_LENGTH_M = 0.16
LEG_MASS_KG = 2.4
FOOT_STATIC_FRICTION = 0.9
FOOT_KINETIC_FRICTION = 0.65
LEG_RADIUS_M = 0.010
FOOT_RADIUS_M = 0.010
ELASTIC_DEFORMATION_M = 0.002
N_LEG_BODY_SAMPLES = 3

TAU_MEM = 0.020
V_THRESH = 0.01
V_RESET = 0.0
DT = 0.010
TRACE_DECAY = 0.70
DEFAULT_MOTOR_NOISE_SCALE = 0.40

EPISODE_S = 30.0
SINGLE_VIEW_EPISODE_S = 120.0
BRAIN_DT = 0.050
MOTOR_SCALE = 6.0
GOAL_HEIGHT_M = 0.16
FIELD_HALF = 15.0
POSITIONAL_ENCODING = "sinusoidal"
POSITIONAL_ENCODING_GAIN = 0.35
POP_SIZE = 32
SIGMA = 0.08
LR = 0.05
PARENT_ELITE_COUNT = 5
MAX_MOTOR_RAD_S = 8.0
MAX_MOTOR_NOISE_SCALE = 1.20
FAST_PROGRESS_TAU_S = 0.20
SLOW_PROGRESS_TAU_S = 0.80
DRAMATIC_PROGRESS_DROP_RATIO = 0.55
NOISE_ATTACK_TAU_S = 0.15
NOISE_RELEASE_TAU_S = 0.90
SIDE_TIP_BAND_HALF_WIDTH_RAD = math.radians(60.0)
SIDE_TIP_DEPTH_PENALTY_SCALE = 10.0
SIDE_TIP_ESCAPE_DELTA_SCALE = 10.0
SIDE_TIP_EXIT_BONUS = 8.0
PROGRESS_REWARD_SCALE = 50.0
GOAL_REACHED_RADIUS_M = 0.5
GOAL_REACHED_BONUS = 50.0

# Lifespan / population selection
DEFAULT_LIFESPAN_S = 30.0
TIPPED_KILL_TIME_S = 5.0
SELECTION_INTERVAL_S = 15.0
LIFESPAN_BONUS_S = 20.0
SELECTION_TOP_FRAC = 0.10
SELECTION_BOT_FRAC = 0.10

GRAVITY_M_S2 = 9.81
FLOOR_HEIGHT_M = 0.0
NORMAL_STIFFNESS_N_M = 20000.0
NORMAL_DAMPING_N_S_M = 3500.0
TANGENTIAL_STIFFNESS_N_M = 7000.0
TANGENTIAL_DAMPING_N_S_M = 450.0
BODY_CONTACT_FRICTION = 0.35
ANGULAR_DAMPING_N_M_S = 8.0
LINEAR_DAMPING_N_S_M = 80.0
AIRBORNE_LINEAR_DAMPING_N_S_M = 3.0
AIRBORNE_ANGULAR_DAMPING_N_M_S = 1.0
MAX_CONTACT_FORCE_N = 120.0
MAX_SUBSTEP_S = 1.0 / 4000.0
UNLOADING_STIFFNESS_SCALE = 0.4
SLEEP_LINEAR_SPEED_THRESHOLD_M_S = 0.01
SLEEP_ANGULAR_SPEED_THRESHOLD_RAD_S = 0.06

MOTOR_MAX_ANGULAR_ACCELERATION_RAD_S2 = 18.0
MOTOR_VISCOUS_DAMPING_PER_S = 8.0
MOTOR_VELOCITY_FILTER_TAU_S = 0.05

TOTAL_MASS_KG = BODY_MASS_KG + (N_LEGS * LEG_MASS_KG)
BODY_HALF_EXTENTS = jnp.array([BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, BODY_HEIGHT_M / 2.0], dtype=jnp.float32)
BODY_PRINCIPAL_INERTIA = jnp.array(
    [
        (BODY_MASS_KG / 12.0) * ((BODY_WIDTH_M * BODY_WIDTH_M) + (BODY_HEIGHT_M * BODY_HEIGHT_M)),
        (BODY_MASS_KG / 12.0) * ((BODY_LENGTH_M * BODY_LENGTH_M) + (BODY_HEIGHT_M * BODY_HEIGHT_M)),
        (BODY_MASS_KG / 12.0) * ((BODY_LENGTH_M * BODY_LENGTH_M) + (BODY_WIDTH_M * BODY_WIDTH_M)),
    ],
    dtype=jnp.float32,
)
LEG_INERTIA_ABOUT_MOUNT = jnp.full((N_LEGS,), LEG_MASS_KG * (LEG_LENGTH_M**2) / 3.0, dtype=jnp.float32)
LEG_ROT_AXIS_BODY = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
REST_CONTACT_BUFFER_M = (TOTAL_MASS_KG * GRAVITY_M_S2) / (max(N_LEGS, 1) * NORMAL_STIFFNESS_N_M)
SUBSTEP_COUNT = int(math.ceil(BRAIN_DT / MAX_SUBSTEP_S))
SUBSTEP_DT_S = BRAIN_DT / SUBSTEP_COUNT

MOUNT_POINTS_BODY = jnp.array(
    [
        [BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, 0.0],
        [BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0, 0.0],
        [-BODY_LENGTH_M / 2.0, BODY_WIDTH_M / 2.0, 0.0],
        [-BODY_LENGTH_M / 2.0, -BODY_WIDTH_M / 2.0, 0.0],
    ],
    dtype=jnp.float32,
)
BODY_CORNERS_BODY = jnp.array(
    [
        [sx * BODY_HALF_EXTENTS[0], sy * BODY_HALF_EXTENTS[1], sz * BODY_HALF_EXTENTS[2]]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ],
    dtype=jnp.float32,
)
LEG_BODY_FRACTIONS = jnp.array([0.25, 0.50, 0.75], dtype=jnp.float32)

ACTIVE_SPEC: RuntimeSpec = default_runtime_spec()


@dataclass(frozen=True)
class _PolicyRuntime:
    model_type: str
    parameter_count: int
    init_param_vector: Any
    unflatten_params: Any
    zero_state: Any
    step: Any


ACTIVE_POLICY_RUNTIME: _PolicyRuntime | None = None


class BrainState(NamedTuple):
    v_shared: jax.Array
    trace_shared: jax.Array
    v_motor: jax.Array
    trace_motor: jax.Array


@dataclass
class TrainingState:
    generation: int = 0
    best_reward: float = -1e9
    best_single_reward: float = -1e9
    mean_reward: float = 0.0
    episode_reward: float = 0.0
    goal_xyz: tuple[float, float, float] = (1.0, 0.0, GOAL_HEIGHT_M)
    robot_state: dict[str, Any] = field(default_factory=dict)
    rewards_history: list[float] = field(default_factory=list)


def current_runtime_spec() -> RuntimeSpec:
    return ACTIVE_SPEC


def apply_runtime_spec(spec: RuntimeSpec | None = None) -> RuntimeSpec:
    global ACTIVE_SPEC
    global ACTIVE_POLICY_RUNTIME
    global PARAM_COUNT
    global BODY_LENGTH_M, BODY_WIDTH_M, BODY_HEIGHT_M, BODY_MASS_KG
    global LEG_LENGTH_M, LEG_MASS_KG, LEG_RADIUS_M, FOOT_RADIUS_M, ELASTIC_DEFORMATION_M, N_LEG_BODY_SAMPLES
    global FOOT_STATIC_FRICTION, FOOT_KINETIC_FRICTION, BODY_CONTACT_FRICTION
    global DT, EPISODE_S, SINGLE_VIEW_EPISODE_S, BRAIN_DT, DEFAULT_LIFESPAN_S, TIPPED_KILL_TIME_S
    global SELECTION_INTERVAL_S, LIFESPAN_BONUS_S, SELECTION_TOP_FRAC, SELECTION_BOT_FRAC, GOAL_REACHED_RADIUS_M
    global MOTOR_SCALE, MAX_MOTOR_RAD_S, MOTOR_MAX_ANGULAR_ACCELERATION_RAD_S2, MOTOR_VISCOUS_DAMPING_PER_S
    global MOTOR_VELOCITY_FILTER_TAU_S, GOAL_HEIGHT_M, FIELD_HALF, POSITIONAL_ENCODING, POSITIONAL_ENCODING_GAIN
    global POP_SIZE, SIGMA, LR, PARENT_ELITE_COUNT
    global DEFAULT_MOTOR_NOISE_SCALE, MAX_MOTOR_NOISE_SCALE, FAST_PROGRESS_TAU_S, SLOW_PROGRESS_TAU_S
    global DRAMATIC_PROGRESS_DROP_RATIO, NOISE_ATTACK_TAU_S, NOISE_RELEASE_TAU_S, SIDE_TIP_BAND_HALF_WIDTH_RAD
    global SIDE_TIP_DEPTH_PENALTY_SCALE, SIDE_TIP_ESCAPE_DELTA_SCALE, SIDE_TIP_EXIT_BONUS
    global PROGRESS_REWARD_SCALE, GOAL_REACHED_BONUS
    global FLOOR_HEIGHT_M
    global GRAVITY_M_S2, NORMAL_STIFFNESS_N_M, NORMAL_DAMPING_N_S_M, TANGENTIAL_STIFFNESS_N_M
    global TANGENTIAL_DAMPING_N_S_M, ANGULAR_DAMPING_N_M_S, LINEAR_DAMPING_N_S_M
    global AIRBORNE_LINEAR_DAMPING_N_S_M, AIRBORNE_ANGULAR_DAMPING_N_M_S, MAX_CONTACT_FORCE_N
    global MAX_SUBSTEP_S, UNLOADING_STIFFNESS_SCALE, SLEEP_LINEAR_SPEED_THRESHOLD_M_S
    global SLEEP_ANGULAR_SPEED_THRESHOLD_RAD_S, TOTAL_MASS_KG, BODY_HALF_EXTENTS, BODY_PRINCIPAL_INERTIA
    global LEG_INERTIA_ABOUT_MOUNT, REST_CONTACT_BUFFER_M, SUBSTEP_COUNT, SUBSTEP_DT_S, MOUNT_POINTS_BODY
    global BODY_CORNERS_BODY, LEG_BODY_FRACTIONS, LEG_ROT_AXIS_BODY

    runtime_spec = spec or default_runtime_spec()
    model_definition = get_model_definition(runtime_spec.model.type)
    runtime_spec = replace(
        runtime_spec,
        model=replace(
            runtime_spec.model,
            architecture=model_definition.architecture,
            trainer=model_definition.trainer,
            description=model_definition.description,
        ),
    )
    runtime_spec.validate()
    ACTIVE_SPEC = runtime_spec
    robot = runtime_spec.robot
    terrain = runtime_spec.terrain
    goals = runtime_spec.goals
    friction = runtime_spec.friction
    physics = runtime_spec.physics
    episode = runtime_spec.episode
    reward = runtime_spec.reward
    training = runtime_spec.training

    BODY_LENGTH_M = float(robot.body_length_m)
    BODY_WIDTH_M = float(robot.body_width_m)
    BODY_HEIGHT_M = float(robot.body_height_m)
    BODY_MASS_KG = float(robot.body_mass_kg)
    LEG_LENGTH_M = float(robot.leg_length_m)
    LEG_MASS_KG = float(robot.leg_mass_kg)
    LEG_RADIUS_M = float(robot.leg_radius_m)
    FOOT_RADIUS_M = float(robot.leg_radius_m)
    ELASTIC_DEFORMATION_M = float(robot.elastic_deformation_m)
    N_LEG_BODY_SAMPLES = int(robot.leg_body_samples)

    FOOT_STATIC_FRICTION = float(friction.foot_static)
    FOOT_KINETIC_FRICTION = float(friction.foot_kinetic)
    BODY_CONTACT_FRICTION = float(friction.body)

    MOTOR_SCALE = float(robot.motor_scale)
    MAX_MOTOR_RAD_S = float(robot.max_motor_rad_s)
    MOTOR_MAX_ANGULAR_ACCELERATION_RAD_S2 = float(robot.motor_max_angular_acceleration_rad_s2)
    MOTOR_VISCOUS_DAMPING_PER_S = float(robot.motor_viscous_damping_per_s)
    MOTOR_VELOCITY_FILTER_TAU_S = float(robot.motor_velocity_filter_tau_s)

    DT = float(episode.neuron_dt_s)
    BRAIN_DT = float(episode.brain_dt_s)
    EPISODE_S = float(episode.episode_s)
    SINGLE_VIEW_EPISODE_S = float(episode.single_view_episode_s)
    DEFAULT_LIFESPAN_S = float(episode.default_lifespan_s)
    TIPPED_KILL_TIME_S = float(episode.tipped_kill_time_s)
    SELECTION_INTERVAL_S = float(episode.selection_interval_s)
    LIFESPAN_BONUS_S = float(episode.lifespan_bonus_s)
    SELECTION_TOP_FRAC = float(episode.selection_top_frac)
    SELECTION_BOT_FRAC = float(episode.selection_bot_frac)
    GOAL_REACHED_RADIUS_M = float(episode.goal_reached_radius_m)

    GOAL_HEIGHT_M = float(goals.height_m)
    FIELD_HALF = float(terrain.field_half_m)
    POSITIONAL_ENCODING = str(runtime_spec.model.positional_encoding)
    POSITIONAL_ENCODING_GAIN = float(runtime_spec.model.positional_encoding_gain)
    POP_SIZE = int(training.population_size)
    SIGMA = float(training.sigma)
    LR = float(training.learning_rate)
    PARENT_ELITE_COUNT = int(training.parent_elite_count)

    DEFAULT_MOTOR_NOISE_SCALE = float(reward.default_motor_noise_scale)
    MAX_MOTOR_NOISE_SCALE = float(reward.max_motor_noise_scale)
    FAST_PROGRESS_TAU_S = float(reward.fast_progress_tau_s)
    SLOW_PROGRESS_TAU_S = float(reward.slow_progress_tau_s)
    DRAMATIC_PROGRESS_DROP_RATIO = float(reward.dramatic_progress_drop_ratio)
    NOISE_ATTACK_TAU_S = float(reward.noise_attack_tau_s)
    NOISE_RELEASE_TAU_S = float(reward.noise_release_tau_s)
    SIDE_TIP_BAND_HALF_WIDTH_RAD = math.radians(float(reward.side_tip_band_half_width_deg))
    SIDE_TIP_DEPTH_PENALTY_SCALE = float(reward.side_tip_depth_penalty_scale)
    SIDE_TIP_ESCAPE_DELTA_SCALE = float(reward.side_tip_escape_delta_scale)
    SIDE_TIP_EXIT_BONUS = float(reward.side_tip_exit_bonus)
    PROGRESS_REWARD_SCALE = float(reward.progress_reward_scale)
    GOAL_REACHED_BONUS = float(reward.goal_reached_bonus)

    FLOOR_HEIGHT_M = float(terrain.floor_height_m)

    GRAVITY_M_S2 = float(physics.gravity_m_s2)
    NORMAL_STIFFNESS_N_M = float(physics.normal_stiffness_n_m)
    NORMAL_DAMPING_N_S_M = float(physics.normal_damping_n_s_m)
    TANGENTIAL_STIFFNESS_N_M = float(physics.tangential_stiffness_n_m)
    TANGENTIAL_DAMPING_N_S_M = float(physics.tangential_damping_n_s_m)
    ANGULAR_DAMPING_N_M_S = float(physics.angular_damping_n_m_s)
    LINEAR_DAMPING_N_S_M = float(physics.linear_damping_n_s_m)
    AIRBORNE_LINEAR_DAMPING_N_S_M = float(physics.airborne_linear_damping_n_s_m)
    AIRBORNE_ANGULAR_DAMPING_N_M_S = float(physics.airborne_angular_damping_n_m_s)
    MAX_CONTACT_FORCE_N = float(physics.max_contact_force_n)
    MAX_SUBSTEP_S = float(physics.max_substep_s)
    UNLOADING_STIFFNESS_SCALE = float(physics.unloading_stiffness_scale)
    SLEEP_LINEAR_SPEED_THRESHOLD_M_S = float(physics.sleep_linear_speed_threshold_m_s)
    SLEEP_ANGULAR_SPEED_THRESHOLD_RAD_S = float(physics.sleep_angular_speed_threshold_rad_s)

    TOTAL_MASS_KG = float(total_robot_mass_kg(runtime_spec))
    BODY_HALF_EXTENTS = jnp.asarray(body_half_extents(runtime_spec), dtype=jnp.float32)
    BODY_PRINCIPAL_INERTIA = jnp.asarray(body_principal_inertia(runtime_spec), dtype=jnp.float32)
    LEG_INERTIA_ABOUT_MOUNT = jnp.asarray(leg_inertia_about_mount(runtime_spec), dtype=jnp.float32)
    LEG_ROT_AXIS_BODY = jnp.asarray(LEG_ROTATION_AXIS_BODY, dtype=jnp.float32)
    REST_CONTACT_BUFFER_M = (TOTAL_MASS_KG * GRAVITY_M_S2) / (max(N_LEGS, 1) * NORMAL_STIFFNESS_N_M)
    SUBSTEP_COUNT = int(math.ceil(BRAIN_DT / MAX_SUBSTEP_S))
    SUBSTEP_DT_S = BRAIN_DT / SUBSTEP_COUNT
    MOUNT_POINTS_BODY = jnp.asarray(mount_points_body(runtime_spec), dtype=jnp.float32)
    BODY_CORNERS_BODY = jnp.asarray(body_corners_body(runtime_spec), dtype=jnp.float32)
    LEG_BODY_FRACTIONS = jnp.asarray(leg_body_fractions(runtime_spec), dtype=jnp.float32)
    ACTIVE_POLICY_RUNTIME = _policy_runtime_for_spec(runtime_spec)
    PARAM_COUNT = int(ACTIVE_POLICY_RUNTIME.parameter_count)
    jax.clear_caches()
    return ACTIVE_SPEC


PARAM_SPECS = [
    ("w_in_shared", (SHARED_TRUNK_WIDTH, N_IN)),
    ("w_in_motor", (N_OUT, MOTOR_LANE_WIDTH, N_IN)),
    ("w_shared_from_shared", (N_HIDDEN_LAYERS - 1, SHARED_TRUNK_WIDTH, SHARED_TRUNK_WIDTH)),
    ("w_shared_from_motors", (N_HIDDEN_LAYERS - 1, SHARED_TRUNK_WIDTH, N_OUT * MOTOR_LANE_WIDTH)),
    ("w_motor_from_motor", (N_HIDDEN_LAYERS - 1, N_OUT, MOTOR_LANE_WIDTH, MOTOR_LANE_WIDTH)),
    ("w_motor_from_shared", (N_HIDDEN_LAYERS - 1, N_OUT, MOTOR_LANE_WIDTH, SHARED_TRUNK_WIDTH)),
    ("b_shared", (N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH)),
    ("b_motor", (N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH)),
    ("w_out_shared", (N_OUT, SHARED_TRUNK_WIDTH)),
    ("w_out_motor", (N_OUT, MOTOR_LANE_WIDTH)),
    ("b_out", (N_OUT,)),
]


def _param_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


PARAM_OFFSETS: dict[str, tuple[int, int, tuple[int, ...]]] = {}
_offset = 0
for _name, _shape in PARAM_SPECS:
    _size = _param_size(_shape)
    PARAM_OFFSETS[_name] = (_offset, _offset + _size, _shape)
    _offset += _size
PARAM_COUNT = _offset
SHARED_POLICY_PARAM_COUNT = _offset  # Frozen snapshot; PARAM_COUNT is rebound by apply_runtime_spec.


def apply_positional_encoding_np(obs: np.ndarray) -> np.ndarray:
    if POSITIONAL_ENCODING != "sinusoidal" or POSITIONAL_ENCODING_GAIN <= 0.0:
        return np.asarray(obs, dtype=np.float32)
    out = np.asarray(obs, dtype=np.float32).copy()
    # Keep goal_xyz in obs[:3] unmodified so axis semantics remain explicit.
    position_slice = out[3:33]
    sin_1x = np.sin(np.pi * position_slice).astype(np.float32)
    sin_2x = np.sin(2.0 * np.pi * position_slice).astype(np.float32)
    sinusoidal = (sin_1x + (np.float32(0.5) * sin_2x)) / np.float32(1.5)
    out[3:33] = ((np.float32(1.0) - np.float32(POSITIONAL_ENCODING_GAIN)) * position_slice) + (
        np.float32(POSITIONAL_ENCODING_GAIN) * sinusoidal
    )
    return out


def _shared_brain_zero_state() -> BrainState:
    return BrainState(
        v_shared=jnp.zeros((N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH), dtype=jnp.float32),
        trace_shared=jnp.zeros((N_HIDDEN_LAYERS, SHARED_TRUNK_WIDTH), dtype=jnp.float32),
        v_motor=jnp.zeros((N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH), dtype=jnp.float32),
        trace_motor=jnp.zeros((N_HIDDEN_LAYERS, N_OUT, MOTOR_LANE_WIDTH), dtype=jnp.float32),
    )


def _shared_init_param_vector(key: jax.Array) -> jax.Array:
    keys = jax.random.split(key, len(PARAM_SPECS))
    parts: list[jax.Array] = []
    for subkey, (name, shape) in zip(keys, PARAM_SPECS, strict=True):
        if name in {"w_in_shared", "w_in_motor"}:
            scale = 1.0 / math.sqrt(N_IN)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_shared_from_shared":
            scale = 1.0 / math.sqrt(SHARED_TRUNK_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_shared_from_motors":
            scale = 1.0 / math.sqrt(N_OUT * MOTOR_LANE_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_motor_from_motor":
            scale = 1.0 / math.sqrt(MOTOR_LANE_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name == "w_motor_from_shared":
            scale = 1.0 / math.sqrt(SHARED_TRUNK_WIDTH)
            part = jax.random.normal(subkey, shape, dtype=jnp.float32) * jnp.float32(scale)
        elif name in {"b_shared", "b_motor"}:
            part = jax.random.uniform(subkey, shape, dtype=jnp.float32, minval=-0.2, maxval=0.2)
        else:
            part = jnp.zeros(shape, dtype=jnp.float32)
        parts.append(part.reshape(-1))
    return jnp.concatenate(parts, axis=0)


def _shared_unflatten_params(flat_params: jax.Array) -> dict[str, jax.Array]:
    return {
        name: flat_params[..., start:end].reshape(flat_params.shape[:-1] + shape)
        for name, (start, end, shape) in PARAM_OFFSETS.items()
    }


def _shared_brain_step(
    params: dict[str, jax.Array],
    state: BrainState,
    obs: jax.Array,
    key: jax.Array,
    noise_scale: jax.Array,
) -> tuple[BrainState, jax.Array, jax.Array]:
    scale = jnp.float32(DT / TAU_MEM)
    decay = jnp.float32(1.0) - scale

    def _update_hidden(
        layer_index: int,
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        v_shared, trace_shared, v_motor, trace_motor = carry

        def _layer0_inputs() -> tuple[jax.Array, jax.Array]:
            shared_input = jnp.einsum("hi,i->h", params["w_in_shared"], obs) + params["b_shared"][0]
            motor_input = jnp.einsum("mhi,i->mh", params["w_in_motor"], obs) + params["b_motor"][0]
            return shared_input, motor_input

        def _later_inputs() -> tuple[jax.Array, jax.Array]:
            prev_shared = trace_shared[layer_index - 1]
            prev_motor = trace_motor[layer_index - 1]
            prev_motor_flat = prev_motor.reshape(-1)
            shared_input = (
                jnp.einsum("hj,j->h", params["w_shared_from_shared"][layer_index - 1], prev_shared)
                + jnp.einsum("hj,j->h", params["w_shared_from_motors"][layer_index - 1], prev_motor_flat)
                + params["b_shared"][layer_index]
            )
            motor_input = (
                jnp.einsum("mhj,mj->mh", params["w_motor_from_motor"][layer_index - 1], prev_motor)
                + jnp.einsum("mhj,j->mh", params["w_motor_from_shared"][layer_index - 1], prev_shared)
                + params["b_motor"][layer_index]
            )
            return shared_input, motor_input

        shared_input, motor_input = jax.lax.cond(layer_index == 0, _layer0_inputs, _later_inputs)

        new_v_shared = (v_shared[layer_index] * decay) + (shared_input * scale)
        shared_spikes = (new_v_shared >= V_THRESH).astype(jnp.float32)
        new_v_shared = jnp.where(shared_spikes > 0.0, jnp.float32(V_RESET), new_v_shared)
        new_trace_shared = (trace_shared[layer_index] * TRACE_DECAY) + (shared_spikes * (1.0 - TRACE_DECAY))

        new_v_motor = (v_motor[layer_index] * decay) + (motor_input * scale)
        motor_spikes = (new_v_motor >= V_THRESH).astype(jnp.float32)
        new_v_motor = jnp.where(motor_spikes > 0.0, jnp.float32(V_RESET), new_v_motor)
        new_trace_motor = (trace_motor[layer_index] * TRACE_DECAY) + (motor_spikes * (1.0 - TRACE_DECAY))

        v_shared = v_shared.at[layer_index].set(new_v_shared)
        trace_shared = trace_shared.at[layer_index].set(new_trace_shared)
        v_motor = v_motor.at[layer_index].set(new_v_motor)
        trace_motor = trace_motor.at[layer_index].set(new_trace_motor)
        return v_shared, trace_shared, v_motor, trace_motor

    v_shared, trace_shared, v_motor, trace_motor = jax.lax.fori_loop(
        0,
        N_HIDDEN_LAYERS,
        _update_hidden,
        (state.v_shared, state.trace_shared, state.v_motor, state.trace_motor),
    )
    out = jnp.tanh(
        jnp.einsum("mh,mh->m", params["w_out_motor"], trace_motor[-1])
        + jnp.einsum("mh,h->m", params["w_out_shared"], trace_shared[-1])
        + params["b_out"]
    )
    key, noise_key = jax.random.split(key)
    out = out + (jax.random.normal(noise_key, (N_OUT,), dtype=jnp.float32) * jnp.maximum(noise_scale, 0.0))
    out = jnp.clip(out, -1.0, 1.0)
    return BrainState(v_shared, trace_shared, v_motor, trace_motor), out, key


def _build_default_policy_runtime(model_definition: ModelDefinition) -> _PolicyRuntime:
    if model_definition.parameter_count != SHARED_POLICY_PARAM_COUNT:
        raise ValueError(
            f"Model {model_definition.type!r} expects {model_definition.parameter_count} parameters, "
            f"but the built-in shared policy has {SHARED_POLICY_PARAM_COUNT}."
        )
    if model_definition.input_size != N_IN:
        raise ValueError(
            f"Model {model_definition.type!r} input_size must be {N_IN} for the built-in shared policy; "
            f"received {model_definition.input_size}."
        )
    if model_definition.output_size != N_OUT:
        raise ValueError(
            f"Model {model_definition.type!r} output_size must be {N_OUT} for the built-in shared policy; "
            f"received {model_definition.output_size}."
        )
    return _PolicyRuntime(
        model_type=model_definition.type,
        parameter_count=SHARED_POLICY_PARAM_COUNT,
        init_param_vector=_shared_init_param_vector,
        unflatten_params=_shared_unflatten_params,
        zero_state=_shared_brain_zero_state,
        step=_shared_brain_step,
    )


def _build_plugin_policy_runtime(spec: RuntimeSpec, model_definition: ModelDefinition) -> _PolicyRuntime | None:
    plugin = load_policy_plugin(spec, model_definition)
    if plugin is None:
        return None

    seed_params = plugin.init_params(jax.random.PRNGKey(0))
    seed_flat, unravel = ravel_pytree(seed_params)
    parameter_count = int(seed_flat.shape[0])

    if parameter_count != model_definition.parameter_count:
        raise ValueError(
            f"Model {model_definition.type!r} declares parameter_count={model_definition.parameter_count}, "
            f"but plugin {model_definition.policy_entrypoint!r} produced {parameter_count}."
        )

    def _init_vector(key: jax.Array) -> jax.Array:
        params = plugin.init_params(key)
        flat, _ = ravel_pytree(params)
        return jnp.asarray(flat, dtype=jnp.float32)

    def _unflatten(flat_params: jax.Array) -> Any:
        return unravel(jnp.asarray(flat_params, dtype=jnp.float32))

    return _PolicyRuntime(
        model_type=model_definition.type,
        parameter_count=parameter_count,
        init_param_vector=_init_vector,
        unflatten_params=_unflatten,
        zero_state=plugin.zero_state,
        step=plugin.step,
    )


def _policy_runtime_for_spec(spec: RuntimeSpec) -> _PolicyRuntime:
    model_definition = get_model_definition(spec.model.type)
    if spec.control.mode == "motor_targets" and model_definition.output_size != N_OUT:
        raise ValueError(
            f"Model {model_definition.type!r} output_size must be {N_OUT} in motor_targets mode; "
            f"received {model_definition.output_size}."
        )
    if spec.control.mode == "command_primitives":
        required = len(spec.control.command_vocabulary)
        if model_definition.output_size not in {required, required + 1}:
            raise ValueError(
                f"Model {model_definition.type!r} output_size must match command_vocabulary length {required} "
                f"or {required + 1} (duration head) in command_primitives mode; "
                f"received {model_definition.output_size}."
            )
    plugin_runtime = _build_plugin_policy_runtime(spec, model_definition)
    if plugin_runtime is not None:
        return plugin_runtime
    return _build_default_policy_runtime(model_definition)


def _require_policy_runtime() -> _PolicyRuntime:
    global ACTIVE_POLICY_RUNTIME
    if ACTIVE_POLICY_RUNTIME is None:
        apply_runtime_spec(ACTIVE_SPEC)
    assert ACTIVE_POLICY_RUNTIME is not None
    return ACTIVE_POLICY_RUNTIME


def _brain_zero_state() -> Any:
    return _require_policy_runtime().zero_state()


def _brain_step(
    params: Any,
    state: Any,
    obs: jax.Array,
    key: jax.Array,
    noise_scale: jax.Array,
) -> tuple[Any, jax.Array, jax.Array]:
    return _require_policy_runtime().step(params, state, obs, key, noise_scale)


def _sample_spawn_batch(key: jax.Array, count: int) -> jax.Array:
    if ACTIVE_SPEC.spawn_policy.strategy == "origin":
        return jnp.zeros((count, 2), dtype=jnp.float32)
    if ACTIVE_SPEC.spawn_policy.strategy == "fixed_points":
        fixed_points = jnp.asarray(ACTIVE_SPEC.spawn_policy.fixed_points, dtype=jnp.float32)
        indices = jax.random.randint(key, (count,), 0, fixed_points.shape[0])
        return fixed_points[indices]
    x_min, x_max = ACTIVE_SPEC.spawn_policy.x_range_m
    y_min, y_max = ACTIVE_SPEC.spawn_policy.y_range_m
    x_key, y_key = jax.random.split(key)
    x_values = jax.random.uniform(x_key, (count,), minval=x_min, maxval=x_max, dtype=jnp.float32)
    y_values = jax.random.uniform(y_key, (count,), minval=y_min, maxval=y_max, dtype=jnp.float32)
    return jnp.stack([x_values, y_values], axis=1)


class ESTrainer:
    """Single ES trainer that keeps optimization in JAX and swaps only the rollout backend."""

    def __init__(
        self,
        seed: int = 42,
        spec: RuntimeSpec | None = None,
        *,
        model_id: str | None = None,
        log_id: str | None = None,
    ) -> None:
        self.spec = apply_runtime_spec(spec or current_runtime_spec())
        if self.spec.model.trainer != "openai_es":
            raise ValueError(
                f"ESTrainer only supports model.trainer='openai_es', received {self.spec.model.trainer!r}."
            )
        self._policy_runtime = _require_policy_runtime()
        self.model_definition = get_model_definition(self.spec.model.type)
        if self.model_definition.parameter_count != self._policy_runtime.parameter_count:
            raise ValueError(
                f"Registered model {self.spec.model.type!r} expects {self.model_definition.parameter_count} "
                f"parameters, but ESTrainer exposes {self._policy_runtime.parameter_count}."
            )
        self.model_id = model_id
        self.log_id = log_id
        self.seed = seed
        self.state = TrainingState()
        self.state.goal_xyz = (
            float(self.spec.goals.radius_m) if self.spec.goals.strategy != "fixed" else float(self.spec.goals.fixed_goal_xyz[0]),
            0.0 if self.spec.goals.strategy != "fixed" else float(self.spec.goals.fixed_goal_xyz[1]),
            float(self.spec.goals.height_m) if self.spec.goals.strategy != "fixed" else float(self.spec.goals.fixed_goal_xyz[2]),
        )
        self._key = jax.random.PRNGKey(seed)
        self._key, init_key = jax.random.split(self._key)
        self._params = self._policy_runtime.init_param_vector(init_key)
        self._top_params = np.zeros((0, self.param_count), dtype=np.float32)
        self._top_rewards = np.zeros((0,), dtype=np.float32)
        self._top_indices = np.zeros((0,), dtype=np.int32)
        self._top_generations = np.zeros((0,), dtype=np.int32)
        self._rollout_backend = MuJoCoBackend(self.spec)

    @property
    def backend(self) -> str:
        return "mujoco"

    @property
    def device_summary(self) -> str:
        return f"{self.spec.model.type} on MuJoCo {mujoco.__version__} rollout + JAX {jax.default_backend()} policy"

    @property
    def param_count(self) -> int:
        return int(self._policy_runtime.parameter_count)

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self._params, dtype=np.float32).copy()

    @property
    def top_params(self) -> np.ndarray:
        return self._top_params.copy()

    @property
    def top_rewards(self) -> np.ndarray:
        return self._top_rewards.copy()

    @property
    def top_indices(self) -> np.ndarray:
        return self._top_indices.copy()

    @property
    def top_generations(self) -> np.ndarray:
        return self._top_generations.copy()

    def _random_goal(self) -> jax.Array:
        if self.spec.goals.strategy == "fixed" and self.spec.goals.fixed_goal_xyz is not None:
            return jnp.asarray(self.spec.goals.fixed_goal_xyz, dtype=jnp.float32)
        self._key, angle_key = jax.random.split(self._key, 2)
        angle = jax.random.uniform(angle_key, (), minval=0.0, maxval=2.0 * jnp.pi)
        radius = float(self.spec.goals.radius_m)
        return jnp.array([radius * jnp.cos(angle), radius * jnp.sin(angle), GOAL_HEIGHT_M], dtype=jnp.float32)

    def _run_logged_episode(self, params_flat: jax.Array, goal_xyz: jax.Array, key: jax.Array, on_step: Any, steps: int | None = None) -> float:
        if steps is None:
            steps = int(self.spec.episode.single_view_episode_s / self.spec.episode.brain_dt_s)
        return self._rollout_backend.run_logged_episode(params_flat, goal_xyz, key, on_step, int(steps))

    def _run_population_episode(
        self,
        params_batch: jax.Array,
        goal_xyz: jax.Array,
        eval_keys: jax.Array,
        spawn_xys: jax.Array,
        on_step: Any = None,
    ) -> np.ndarray:
        """Run the full population with lifespan tracking and periodic selection."""
        params_batch_np = np.asarray(params_batch, dtype=np.float32)
        if on_step is None or params_batch_np.shape[0] == 0:
            return self._rollout_backend.run_population(
                params_batch_np,
                goal_xyz,
                eval_keys,
                int(EPISODE_S / BRAIN_DT),
                spawn_xys=spawn_xys,
            )

        returns = np.zeros((params_batch_np.shape[0],), dtype=np.float32)
        returns[0] = self._rollout_backend.run_logged_episode(
            params_batch_np[0],
            goal_xyz,
            eval_keys[0],
            on_step,
            int(EPISODE_S / BRAIN_DT),
            spawn_xy=spawn_xys[0],
        )
        if params_batch_np.shape[0] > 1:
            returns[1:] = self._rollout_backend.run_population(
                params_batch_np[1:],
                goal_xyz,
                eval_keys[1:],
                int(EPISODE_S / BRAIN_DT),
                spawn_xys=spawn_xys[1:],
            )
        return returns

    def run_generation(self, on_step: Any = None, on_gen_done: Any = None) -> None:
        goal_xyz = self._random_goal()
        self.state.goal_xyz = tuple(float(v) for v in np.asarray(goal_xyz).tolist())

        self._key, noise_key, eval_key, center_key, spawn_key = jax.random.split(self._key, 5)
        noise = jax.random.normal(noise_key, (POP_SIZE, self.param_count), dtype=jnp.float32) * jnp.float32(SIGMA)
        params_batch = self._params[None, :] + noise
        eval_keys = jax.random.split(eval_key, POP_SIZE)
        spawn_xys = _sample_spawn_batch(spawn_key, POP_SIZE)

        returns_np = self._run_population_episode(
            params_batch, goal_xyz, eval_keys, spawn_xys,
            on_step=on_step,
        )

        # Proper OpenAI ES gradient update using the full population.
        returns_std = returns_np.std()
        if returns_std > 1e-8:
            normalized_returns = (returns_np - returns_np.mean()) / returns_std
        else:
            normalized_returns = returns_np - returns_np.mean()
        noise_np = np.asarray(noise, dtype=np.float32)
        gradient = (normalized_returns[:, None] * noise_np).sum(axis=0) / (POP_SIZE * SIGMA)
        self._params = self._params + jnp.float32(LR) * jnp.asarray(gradient, dtype=jnp.float32)

        elite_count = min(PARENT_ELITE_COUNT, returns_np.shape[0])
        top_indices = np.argpartition(returns_np, -elite_count)[-elite_count:]
        top_indices = top_indices[np.argsort(returns_np[top_indices])[::-1]]
        top_params = np.asarray(jnp.take(params_batch, jnp.asarray(top_indices), axis=0), dtype=np.float32)
        top_rewards = returns_np[top_indices].astype(np.float32)
        top_generations = np.full((elite_count,), self.state.generation + 1, dtype=np.int32)

        if self._top_rewards.size > 0:
            combined_params = np.concatenate([self._top_params, top_params], axis=0)
            combined_rewards = np.concatenate([self._top_rewards, top_rewards], axis=0)
            combined_indices = np.concatenate([self._top_indices, top_indices.astype(np.int32)], axis=0)
            combined_generations = np.concatenate([self._top_generations, top_generations], axis=0)
        else:
            combined_params = top_params
            combined_rewards = top_rewards
            combined_indices = top_indices.astype(np.int32)
            combined_generations = top_generations

        leaderboard_count = min(PARENT_ELITE_COUNT, combined_rewards.shape[0])
        leaderboard_indices = np.argpartition(combined_rewards, -leaderboard_count)[-leaderboard_count:]
        leaderboard_indices = leaderboard_indices[np.argsort(combined_rewards[leaderboard_indices])[::-1]]
        self._top_params = combined_params[leaderboard_indices].astype(np.float32)
        self._top_rewards = combined_rewards[leaderboard_indices].astype(np.float32)
        self._top_indices = combined_indices[leaderboard_indices].astype(np.int32)
        self._top_generations = combined_generations[leaderboard_indices].astype(np.int32)

        # Evaluate the updated center params (no noise) so best_reward tracks the actual center.
        steps = int(EPISODE_S / BRAIN_DT)
        center_return = float(
            self._rollout_backend.run_episode(
                np.asarray(self._params, dtype=np.float32),
                goal_xyz,
                center_key,
                steps,
            )
        )

        self.state.generation += 1
        self.state.mean_reward = float(returns_np.mean())
        self.state.episode_reward = float(self._top_rewards.mean()) if self._top_rewards.size else 0.0
        self.state.best_reward = max(self.state.best_reward, center_return)
        self.state.best_single_reward = max(
            self.state.best_single_reward,
            float(self._top_rewards[0]) if self._top_rewards.size else -1e9,
        )
        self.state.rewards_history.append(self.state.mean_reward)

        if on_gen_done is not None:
            on_gen_done(
                {
                    "type": "generation",
                    "generation": self.state.generation,
                    "mean_reward": self.state.mean_reward,
                    "best_reward": self.state.best_reward,
                    "top_rewards": self._top_rewards.tolist(),
                    "rewards_history": self.state.rewards_history[-100:],
                    "goal": list(self.state.goal_xyz),
                }
            )

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "params": np.asarray(self._params, dtype=np.float32),
            "top_params": self._top_params.astype(np.float32),
            "top_rewards": self._top_rewards.astype(np.float32),
            "top_indices": self._top_indices.astype(np.int32),
            "top_generations": self._top_generations.astype(np.int32),
            "generation": np.int32(self.state.generation),
            "best_reward": np.float32(self.state.best_reward),
            "best_single_reward": np.float32(self.state.best_single_reward),
            "mean_reward": np.float32(self.state.mean_reward),
            "episode_reward": np.float32(self.state.episode_reward),
            "goal_xyz": np.asarray(self.state.goal_xyz, dtype=np.float32),
            "rewards_history": np.asarray(self.state.rewards_history, dtype=np.float32),
            "rng_key": np.asarray(self._key, dtype=np.uint32),
            "seed": np.int32(self.seed),
            "backend": np.array(self.backend),
            "compute_backend": np.array(jax.default_backend()),
            "model_type": np.array(self.spec.model.type),
            "model_architecture": np.array(self.spec.model.architecture),
            "model_trainer": np.array(self.spec.model.trainer),
            "param_count": np.int32(self.param_count),
            "model_id": np.array(self.model_id or ""),
            "log_id": np.array(self.log_id or ""),
            "config_json": np.array(canonical_config_json(self.spec)),
            "config_name": np.array(self.spec.name),
            "simulator_backend": np.array(self.spec.simulator.backend),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_path, **self.checkpoint_dict())
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "config_json" in checkpoint.files:
                checkpoint_config_json = str(checkpoint["config_json"].item())
                if not config_json_matches_checkpoint(self.spec, checkpoint_config_json):
                    raise ValueError(
                        "Checkpoint config does not match the active runtime spec aside from backend selection. "
                        "Use a compatible config file to resume training."
                    )
            elif "model_type" in checkpoint.files and str(checkpoint["model_type"].item()) != self.spec.model.type:
                raise ValueError("Checkpoint model_type does not match the active runtime spec.")
            if "param_count" in checkpoint.files and int(checkpoint["param_count"].item()) != self.param_count:
                raise ValueError(
                    "Checkpoint param_count does not match active model parameter width. "
                    f"checkpoint={int(checkpoint['param_count'].item())}, active={self.param_count}."
                )
            params = checkpoint["params"]
            if params.shape != (self.param_count,):
                raise ValueError(
                    f"Checkpoint parameter shape {params.shape} does not match active model shape {(self.param_count,)}."
                )
            self._params = jnp.asarray(checkpoint["params"], dtype=jnp.float32)
            if "top_params" in checkpoint.files:
                top_params = checkpoint["top_params"].astype(np.float32)
                if top_params.ndim != 2 or top_params.shape[1] != self.param_count:
                    raise ValueError(
                        f"Checkpoint top_params shape {top_params.shape} does not match active model width {self.param_count}."
                    )
                self._top_params = top_params
            else:
                self._top_params = np.zeros((0, self.param_count), dtype=np.float32)
            self._top_rewards = checkpoint["top_rewards"].astype(np.float32) if "top_rewards" in checkpoint.files else np.zeros((0,), dtype=np.float32)
            self._top_indices = checkpoint["top_indices"].astype(np.int32) if "top_indices" in checkpoint.files else np.zeros((0,), dtype=np.int32)
            self.state.generation = int(checkpoint["generation"])
            if "top_generations" in checkpoint.files:
                self._top_generations = checkpoint["top_generations"].astype(np.int32)
            else:
                self._top_generations = np.full(
                    (self._top_rewards.shape[0],),
                    self.state.generation,
                    dtype=np.int32,
                )
            self.state.best_reward = float(checkpoint["best_reward"])
            self.state.best_single_reward = (
                float(checkpoint["best_single_reward"])
                if "best_single_reward" in checkpoint.files
                else float(checkpoint["best_reward"])
            )
            self.state.mean_reward = float(checkpoint["mean_reward"])
            self.state.episode_reward = float(checkpoint["episode_reward"])
            self.state.goal_xyz = tuple(float(v) for v in checkpoint["goal_xyz"].tolist())
            self.state.rewards_history = [float(v) for v in checkpoint["rewards_history"].tolist()]
            self._key = jnp.asarray(checkpoint["rng_key"], dtype=jnp.uint32)
            if "model_id" in checkpoint.files:
                checkpoint_model_id = str(checkpoint["model_id"].item())
                self.model_id = checkpoint_model_id or self.model_id
            if "log_id" in checkpoint.files:
                checkpoint_log_id = str(checkpoint["log_id"].item())
                self.log_id = checkpoint_log_id or self.log_id

JaxESTrainer = ESTrainer

apply_runtime_spec(ACTIVE_SPEC)

__all__ = [
    "EPISODE_S",
    "POP_SIZE",
    "TrainingState",
    "ESTrainer",
    "JaxESTrainer",
    "apply_runtime_spec",
    "current_runtime_spec",
]
