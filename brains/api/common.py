"""Shared websocket API helpers for live viewers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import WebSocket

from brains.config import RuntimeSpec
from brains.models import list_model_definitions
from brains.sim import single_step_to_viewer_frame as translate_single_step_to_viewer_frame
from brains.sim.mujoco_layout import LEG_NAMES, LEG_ROTATION_AXIS_BODY, mount_points_body


VIEWER_RESET_SECONDS = 30.0


@dataclass(frozen=True)
class ViewerMetadata:
    mode: str
    config_name: str
    terrain: dict[str, Any]
    robot: dict[str, Any]
    model: dict[str, Any]
    goal: dict[str, Any]
    training: dict[str, Any]
    simulator: dict[str, Any]

    def to_message(self) -> dict[str, Any]:
        return {
            "type": "metadata",
            "mode": self.mode,
            "config_name": self.config_name,
            "terrain": self.terrain,
            "robot": self.robot,
            "model": self.model,
            "goal": self.goal,
            "training": self.training,
            "simulator": self.simulator,
        }


class BroadcastHub:
    """Caches the latest messages and fans them out to websocket clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._latest_messages: dict[str, dict[str, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        for message in self._latest_messages.values():
            await websocket.send_json(message)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    def publish(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type", "event"))
        self._latest_messages[message_type] = message
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


def build_viewer_metadata(spec: RuntimeSpec, mode: str) -> ViewerMetadata:
    terrain = {
        "kind": spec.terrain.kind,
        "field_half_m": spec.terrain.field_half_m,
        "center_half_m": spec.terrain.center_half_m,
        "step_count": spec.terrain.step_count,
        "step_width_m": spec.terrain.step_width_m,
        "step_height_m": spec.terrain.step_height_m,
        "floor_height_m": spec.terrain.floor_height_m,
    }
    mount_points = mount_points_body(spec)
    rotation_axes = [list(LEG_ROTATION_AXIS_BODY) for _ in LEG_NAMES]
    robot = {
        "body_length_m": spec.robot.body_length_m,
        "body_width_m": spec.robot.body_width_m,
        "body_height_m": spec.robot.body_height_m,
        "leg_length_m": spec.robot.leg_length_m,
        "leg_radius_m": spec.robot.leg_radius_m,
        "foot_radius_m": spec.robot.foot_radius_m,
        "mount_points_body": [list(point) for point in mount_points],
        "rotation_axes_body": rotation_axes,
        "leg_names": list(LEG_NAMES),
    }
    goal = {
        "strategy": spec.goals.strategy,
        "radius_m": spec.goals.radius_m,
        "height_m": spec.goals.height_m,
        "fixed_goal_xyz": list(spec.goals.fixed_goal_xyz) if spec.goals.fixed_goal_xyz is not None else None,
    }
    model = {
        "active": spec.model.type,
        "architecture": spec.model.architecture,
        "trainer": spec.model.trainer,
        "description": spec.model.description,
        "registered": [definition.to_dict() for definition in list_model_definitions()],
    }
    training = {
        "population_size": spec.training.population_size,
        "episode_s": spec.episode.episode_s,
        "selection_interval_s": spec.episode.selection_interval_s,
        "viewer_reset_s": VIEWER_RESET_SECONDS,
        "brain_dt_s": spec.episode.brain_dt_s,
    }
    simulator = {
        "backend": spec.simulator.backend,
        "render": spec.simulator.render,
        "deterministic_mode": spec.simulator.deterministic_mode,
    }
    return ViewerMetadata(
        mode=mode,
        config_name=spec.name,
        terrain=terrain,
        robot=robot,
        model=model,
        goal=goal,
        training=training,
        simulator=simulator,
    )


def viewer_reset_steps(spec: RuntimeSpec) -> int:
    return max(1, int(VIEWER_RESET_SECONDS / spec.episode.brain_dt_s))


def current_policy_params(trainer: Any) -> np.ndarray:
    return np.asarray(trainer.params, dtype=np.float32)


def single_step_to_frame(step_message: dict[str, Any], generation: int, spec: RuntimeSpec) -> dict[str, Any]:
    return translate_single_step_to_viewer_frame(step_message, generation=generation, spec=spec)
