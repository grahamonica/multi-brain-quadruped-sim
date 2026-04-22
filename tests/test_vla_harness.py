from __future__ import annotations

import unittest

import numpy as np

from brains.config import load_runtime_spec
from brains.harnesses import VLAHarness
from brains.models.hf_vla_adapter import _extract_generated_text


class _AlwaysTrotAgent:
    def choose_action(self, image_rgb, instruction, options, observation):
        del image_rgb, instruction, options, observation
        return "trot"


class _DirectMotorAgent:
    def choose_action(self, image_rgb, instruction, options, observation):
        del image_rgb, instruction, options, observation
        return np.asarray([0.5, -0.5, 0.25, -0.25], dtype=np.float32)


class VLAHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_runtime_spec("configs/smoke.yaml")

    def test_vla_harness_command_action_path(self) -> None:
        harness = VLAHarness(self.spec)
        run = harness.run_vla(_AlwaysTrotAgent(), "walk forward", steps=4)

        self.assertEqual(len(run.frames), 4)
        first = run.frames[0]
        self.assertIn("camera", first)
        self.assertIn("vla", first)
        self.assertEqual(first["vla"]["action"], "trot")
        self.assertEqual(first["camera"]["name"], "head_camera")

    def test_vla_harness_direct_motor_path(self) -> None:
        harness = VLAHarness(self.spec)
        run = harness.run_vla(_DirectMotorAgent(), "stabilize", steps=3, include_rgb=True)

        self.assertEqual(len(run.frames), 3)
        first = run.frames[0]
        self.assertEqual(first["vla"]["action"], "direct_motor")
        rgb = first["camera"]["rgb"]
        self.assertIsInstance(rgb, np.ndarray)
        self.assertEqual(rgb.shape[2], 3)

    def test_extract_generated_text_handles_common_payloads(self) -> None:
        self.assertEqual(_extract_generated_text("stand"), "stand")
        self.assertEqual(_extract_generated_text({"generated_text": "trot"}), "trot")
        self.assertEqual(
            _extract_generated_text([{"generated_text": [{"content": "turn_left"}]}]),
            "turn_left",
        )


if __name__ == "__main__":
    unittest.main()
