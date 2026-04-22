from __future__ import annotations

import unittest
from dataclasses import replace

import jax

from brains.config import load_runtime_spec
from brains.jax_trainer import ESTrainer
from brains.models import ModelDefinition, refresh_model_definitions, register_model_definition


class PolicyPluginTests(unittest.TestCase):
    def tearDown(self) -> None:
        refresh_model_definitions()

    def test_linear_plugin_can_run_logged_episode(self) -> None:
        definition = ModelDefinition(
            type="plugin_linear_smoke",
            architecture="linear",
            trainer="openai_es",
            input_size=48,
            output_size=4,
            parameter_count=196,
            description="Linear plugin policy for smoke testing.",
            policy_entrypoint="brains.models.reference_linear_plugin:build_linear_policy",
        )
        register_model_definition(definition, persist=False)

        base_spec = load_runtime_spec("configs/smoke.yaml")
        spec = replace(
            base_spec,
            model=replace(
                base_spec.model,
                type=definition.type,
                architecture=definition.architecture,
                trainer=definition.trainer,
                description=definition.description,
            ),
        )

        trainer = ESTrainer(seed=5, spec=spec)
        self.assertEqual(trainer.param_count, 196)

        goal_xyz = trainer._random_goal()
        reward = trainer._run_logged_episode(trainer.params, goal_xyz, jax.random.PRNGKey(7), lambda _msg: None, steps=4)
        self.assertTrue(float(reward) == float(reward))

    def test_control_mode_contract_rejects_mismatched_output_size(self) -> None:
        definition = ModelDefinition(
            type="plugin_linear_cmd_bad",
            architecture="linear",
            trainer="openai_es",
            input_size=48,
            output_size=6,
            parameter_count=245,
            description="Bad output size contract.",
            policy_entrypoint="brains.models.reference_linear_plugin:build_linear_policy",
        )
        register_model_definition(definition, persist=False)

        base_spec = load_runtime_spec("configs/smoke.yaml")
        spec = replace(
            base_spec,
            model=replace(
                base_spec.model,
                type=definition.type,
                architecture=definition.architecture,
                trainer=definition.trainer,
                description=definition.description,
            ),
            control=replace(
                base_spec.control,
                mode="command_primitives",
                command_vocabulary=("stand", "trot", "turn_left", "turn_right"),
            ),
        )

        with self.assertRaises(ValueError):
            ESTrainer(seed=3, spec=spec)


if __name__ == "__main__":
    unittest.main()
