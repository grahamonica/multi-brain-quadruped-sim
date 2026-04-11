from __future__ import annotations

import unittest

from brains.config import load_runtime_spec
from brains.harnesses import DirectionHarness, HeadCameraHarness


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_runtime_spec("configs/smoke.yaml")

    def test_direction_harness_maps_text_to_options(self) -> None:
        harness = DirectionHarness(self.spec)
        plan = harness.compile("trot forward then turn left")
        option_names = [command.option for command in plan.commands]

        self.assertIn("trot", option_names)
        self.assertIn("turn_left", option_names)
        self.assertEqual(harness.target_velocity("trot", 0.25).shape, (4,))

    def test_head_camera_harness_injects_camera_xml(self) -> None:
        harness = HeadCameraHarness(self.spec)
        xml = harness.build_camera_xml()

        self.assertIn('camera name="head_camera"', xml)
        self.assertIn('fovy="70.000000"', xml)


if __name__ == "__main__":
    unittest.main()
