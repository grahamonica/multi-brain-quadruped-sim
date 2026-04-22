"""VLA harness that drives the quadruped from MuJoCo head-camera frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .direction_harness import CommandPlan, HarnessRun, MotionCommand, StepCallback
from .head_camera_harness import HeadCameraHarness


class VLAAgent(Protocol):
    """Minimal agent contract for vision-language control."""

    def choose_action(
        self,
        image_rgb: np.ndarray,
        instruction: str,
        options: tuple[str, ...],
        observation: dict[str, Any],
    ) -> str | np.ndarray | list[float]: ...


@dataclass(frozen=True)
class VLAStepRecord:
    time_s: float
    action: str


class VLAHarness(HeadCameraHarness):
    """Run MuJoCo rollouts where actions come from a VLA over camera frames."""

    def _resolve_command(self, action_text: str) -> str:
        candidate = " ".join(action_text.lower().split())
        if candidate in self._options_by_name:
            return candidate
        if candidate in self._aliases:
            return self._aliases[candidate]
        for alias, option_name in self._aliases.items():
            if alias in candidate:
                return option_name
        known = ", ".join(option.name for option in self.available_options())
        raise ValueError(f"VLA action {action_text!r} did not map to a known command: {known}.")

    def _action_to_target(
        self,
        action: str | np.ndarray | list[float],
        time_s: float,
    ) -> tuple[np.ndarray, str]:
        if isinstance(action, str):
            command = self._resolve_command(action)
            return self.target_velocity(command, time_s), command

        target = np.asarray(action, dtype=np.float32)
        if target.shape != (4,):
            raise ValueError(f"VLA numeric action must have shape (4,), received {target.shape}.")
        limit = float(self.spec.robot.max_motor_rad_s)
        clipped = np.clip(target, -limit, limit).astype(np.float32)
        return clipped, "direct_motor"

    def run_vla(
        self,
        agent: VLAAgent,
        instruction: str,
        *,
        steps: int | None = None,
        goal_xyz: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        include_rgb: bool = False,
        on_step: StepCallback | None = None,
    ) -> HarnessRun:
        import mujoco

        backend = self._make_backend()
        data = backend.reset_data(spawn_xy=spawn_xy)
        goal = self.default_goal() if goal_xyz is None else np.asarray(goal_xyz, dtype=np.float32)
        metrics = backend.initial_metrics(data, goal)
        if steps is None:
            steps = int(max(1, math.ceil(self.spec.episode.episode_s / self.spec.episode.brain_dt_s)))
        total_steps = int(steps)

        records: list[VLAStepRecord] = []
        frames: list[dict[str, Any]] = []
        options = tuple(option.name for option in self.available_options())

        with mujoco.Renderer(backend.model, width=self.camera.width, height=self.camera.height) as renderer:
            camera_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera.name)
            if camera_id < 0:
                raise ValueError(f"MuJoCo model does not contain camera {self.camera.name!r}.")

            for step_index in range(total_steps):
                time_s = step_index * float(self.spec.episode.brain_dt_s)

                renderer.update_scene(data, camera=self.camera.name)
                image_rgb = np.flipud(np.asarray(renderer.render(), dtype=np.uint8))
                observation = {
                    "time_s": float(data.time),
                    "goal_xyz": goal.tolist(),
                    "body_pos": backend.body_position(data).tolist(),
                    "body_rot": backend.body_rotation(data).tolist(),
                }
                action = agent.choose_action(image_rgb, instruction, options, observation)
                target_velocity, action_name = self._action_to_target(action, time_s)

                backend._advance(data, target_velocity)
                metrics = backend._step_metrics(data, metrics, goal)
                frame = backend._snapshot(data, metrics, goal, step_index, total_steps)
                frame["harness"] = {
                    "option": action_name,
                    "target_velocity_rad_s": target_velocity.tolist(),
                }
                frame["camera"] = {
                    "name": self.camera.name,
                    "width": self.camera.width,
                    "height": self.camera.height,
                    "fovy_deg": self.camera.fovy_deg,
                    "observation": self.camera_observation_metadata(frame, action_name),
                }
                if include_rgb:
                    frame["camera"]["rgb"] = image_rgb
                frame["vla"] = {
                    "instruction": instruction,
                    "action": action_name,
                }
                records.append(VLAStepRecord(time_s=time_s, action=action_name))
                frames.append(frame)
                if on_step is not None:
                    on_step(frame)

        if records:
            commands = tuple(MotionCommand(option=record.action, duration_s=float(self.spec.episode.brain_dt_s), speed=1.0) for record in records)
        else:
            commands = (MotionCommand(option="stand", duration_s=float(self.spec.episode.brain_dt_s), speed=0.0),)
        plan = CommandPlan(commands)
        return HarnessRun(plan=plan, frames=tuple(frames), total_reward=float(metrics.total_reward))
