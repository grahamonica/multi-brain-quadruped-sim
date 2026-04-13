from __future__ import annotations

import unittest
from pathlib import Path

from brains.config import load_runtime_spec
import brains.jax_trainer as trainer_module
from brains.sim.mujoco_layout import LEG_NAMES, leg_body_fractions, mount_points_body, total_robot_mass_kg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DomainModelTests(unittest.TestCase):
    def test_mujoco_layout_is_built_from_runtime_spec(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "default.yaml")
        self.assertEqual(LEG_NAMES, ("front_left", "front_right", "rear_left", "rear_right"))
        self.assertEqual(mount_points_body(spec)[0], (0.14, 0.06, 0.0))
        self.assertEqual(leg_body_fractions(spec), (0.25, 0.5, 0.75))
        self.assertAlmostEqual(total_robot_mass_kg(spec), 4.8)

    def test_runtime_spec_contains_environment_fields(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        self.assertEqual(spec.name, "smoke")
        self.assertEqual(spec.terrain.step_count, 2)
        self.assertEqual(spec.spawn_policy.strategy, "uniform_box")
        self.assertEqual(spec.training.population_size, 8)

    def test_jax_runtime_uses_runtime_spec_values(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        trainer_module.apply_runtime_spec(spec)
        self.assertEqual(trainer_module.MAX_MOTOR_RAD_S, spec.robot.max_motor_rad_s)
        self.assertEqual(trainer_module.GOAL_HEIGHT_M, spec.goals.height_m)
        self.assertEqual(trainer_module.EPISODE_S, spec.episode.episode_s)


if __name__ == "__main__":
    unittest.main()
