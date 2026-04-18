"""Head-camera harness for future VLA experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .direction_harness import CommandPlan, DirectionHarness, HarnessRun, StepCallback


@dataclass(frozen=True)
class CameraConfig:
    name: str = "head_camera"
    width: int = 160
    height: int = 120
    fovy_deg: float = 70.0
    forward_offset_m: float = 0.055
    height_offset_m: float = 0.035


class HeadCameraHarness(DirectionHarness):
    """Direction harness with a front-mounted torso camera for VLA data capture."""

    def __init__(self, spec=None, camera: CameraConfig | None = None) -> None:
        super().__init__(spec)
        self.camera = camera or CameraConfig()

    def build_camera_xml(self) -> str:
        from brains.sim.mujoco_model_builder import build_mujoco_model

        artifacts = build_mujoco_model(self.spec)
        x = (self.spec.robot.body_length_m * 0.5) + self.camera.forward_offset_m
        z = (self.spec.robot.body_height_m * 0.5) + self.camera.height_offset_m
        camera_tag = (
            f'<camera name="{self.camera.name}" mode="fixed" pos="{x:.6f} 0 {z:.6f}" '
            f'euler="0 1.570796 0" fovy="{self.camera.fovy_deg:.6f}"/>'
        )
        return artifacts.xml.replace('<freejoint name="root_free"/>', f'<freejoint name="root_free"/>{camera_tag}', 1)

    def _make_backend(self):
        import mujoco

        backend = super()._make_backend()
        backend.model_artifacts = replace(backend.model_artifacts, xml=self.build_camera_xml())
        backend.model = mujoco.MjModel.from_xml_string(backend.model_artifacts.xml)
        backend._torso_body_id = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, backend.model_artifacts.body_name)
        backend._leg_body_ids = [
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in backend.model_artifacts.leg_body_names
        ]
        backend._foot_site_ids = [
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in backend.model_artifacts.foot_site_names
        ]
        backend._joint_qpos_adr = [
            int(backend.model.joint(name).qposadr[0])
            for name in backend.model_artifacts.leg_joint_names
        ]
        backend._joint_dof_adr = [
            int(backend.model.joint(name).dofadr[0])
            for name in backend.model_artifacts.leg_joint_names
        ]
        backend._actuator_ids = [
            mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in backend.model_artifacts.actuator_names
        ]
        return backend

    def camera_observation_metadata(self, frame: dict[str, Any], command_option: str) -> dict[str, Any]:
        body = frame.get("body", {})
        return {
            "camera": self.camera.name,
            "command": command_option,
            "body_pos": body.get("pos"),
            "body_rot": body.get("rot"),
            "time_s": frame.get("time_s", 0.0),
        }

    def run(
        self,
        direction: str | CommandPlan,
        *,
        steps: int | None = None,
        goal_xyz: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        on_step: StepCallback | None = None,
    ) -> HarnessRun:
        def _annotate(frame: dict[str, Any]) -> None:
            command = frame.get("harness", {}).get("option", "unknown")
            frame["camera"] = {
                "name": self.camera.name,
                "width": self.camera.width,
                "height": self.camera.height,
                "fovy_deg": self.camera.fovy_deg,
                "observation": self.camera_observation_metadata(frame, command),
            }
            if on_step is not None:
                on_step(frame)

        return super().run(
            direction,
            steps=steps,
            goal_xyz=goal_xyz,
            spawn_xy=spawn_xy,
            on_step=_annotate,
        )
