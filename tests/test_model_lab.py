from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from brains.config import load_runtime_spec
from brains.models import NotebookModel, apply_notebook_model, register_notebook_model
from brains.models import load_model_definitions, refresh_model_definitions


class ModelLabTests(unittest.TestCase):
    def test_apply_notebook_model_updates_control_and_model(self) -> None:
        base_spec = load_runtime_spec("configs/smoke.yaml")
        notebook_model = NotebookModel(
            model_type="notebook_cmd_model",
            architecture="shared_trunk_motor_lanes",
            description="Notebook command primitive variant",
            input_size=48,
            output_size=4,
            parameter_count=48516,
            control_mode="command_primitives",
            command_vocabulary=("stand", "trot", "turn_left", "turn_right"),
        )

        updated = apply_notebook_model(base_spec, notebook_model)

        self.assertEqual(updated.model.type, "notebook_cmd_model")
        self.assertEqual(updated.control.mode, "command_primitives")
        self.assertEqual(updated.control.command_vocabulary[1], "trot")

    def test_register_notebook_model_persists_registry(self) -> None:
        notebook_model = NotebookModel(
            model_type="notebook_registry_write",
            architecture="shared_trunk_motor_lanes",
            description="Registry write smoke test",
            input_size=48,
            output_size=4,
            parameter_count=48516,
        )

        with TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir) / "model_registry.json"
            try:
                register_notebook_model(notebook_model, persist=True, registry_path=registry_path)
                definitions = load_model_definitions(registry_path)
            finally:
                refresh_model_definitions()

        self.assertIn("notebook_registry_write", definitions)

    def test_register_notebook_model_can_infer_parameter_count_from_plugin(self) -> None:
        base_spec = load_runtime_spec("configs/smoke.yaml")
        notebook_model = NotebookModel(
            model_type="notebook_plugin_inferred_count",
            architecture="linear",
            description="Inference test for plugin parameter count.",
            input_size=48,
            output_size=4,
            parameter_count=None,
            policy_entrypoint="brains.models.reference_linear_plugin:build_linear_policy",
        )

        with TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir) / "model_registry.json"
            try:
                definition = register_notebook_model(
                    notebook_model,
                    spec=base_spec,
                    persist=True,
                    registry_path=registry_path,
                )
                definitions = load_model_definitions(registry_path)
            finally:
                refresh_model_definitions()

        self.assertEqual(definition.parameter_count, 196)
        self.assertEqual(definitions["notebook_plugin_inferred_count"].parameter_count, 196)


if __name__ == "__main__":
    unittest.main()
