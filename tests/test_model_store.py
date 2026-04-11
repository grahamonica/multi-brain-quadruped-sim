from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from brains.config import canonical_config_json, load_runtime_spec
from brains.models import ModelDefinition, load_model_definitions, refresh_model_definitions, register_model_definition
from brains.runtime import create_model_run_paths, discover_model_artifacts, write_model_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelStoreTests(unittest.TestCase):
    def test_model_run_paths_use_model_type_and_log_id(self) -> None:
        paths = create_model_run_paths("checkpoints", "shared_trunk_es", "12389748")
        self.assertEqual(paths.id, "shared_trunk_es_12389748")
        self.assertEqual(paths.latest_path, Path("checkpoints") / "shared_trunk_es_12389748" / "latest.npz")

    def test_discovery_reads_manifest_and_checkpoint_summary(self) -> None:
        spec = load_runtime_spec(PROJECT_ROOT / "configs" / "smoke.yaml")
        with TemporaryDirectory() as tmp_dir:
            paths = create_model_run_paths(tmp_dir, spec.model.type, "abc123")
            paths.run_dir.mkdir(parents=True)
            np.savez_compressed(
                paths.latest_path,
                config_json=np.array(canonical_config_json(spec)),
                generation=np.int32(7),
                best_reward=np.float32(12.5),
                mean_reward=np.float32(4.25),
                config_name=np.array(spec.name),
                model_type=np.array(spec.model.type),
                model_id=np.array(paths.id),
                log_id=np.array(paths.log_id),
            )
            write_model_manifest(
                paths,
                spec,
                paths.latest_path,
                generation=7,
                best_reward=12.5,
                mean_reward=4.25,
            )

            artifacts = discover_model_artifacts(tmp_dir)

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].id, "shared_trunk_es_abc123")
        self.assertEqual(artifacts[0].generation, 7)
        self.assertAlmostEqual(artifacts[0].best_reward or 0.0, 12.5)

    def test_model_definition_can_be_persisted_for_notebook_models(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir) / "model_registry.json"
            definition = ModelDefinition(
                type="notebook_shared_trunk",
                architecture="shared_trunk_motor_lanes",
                trainer="openai_es",
                input_size=48,
                output_size=4,
                parameter_count=48516,
                description="Notebook-created variant of the current shared trunk trainer.",
            )

            try:
                register_model_definition(definition, persist=True, registry_path=registry_path)
                definitions = load_model_definitions(registry_path)
            finally:
                refresh_model_definitions()

        self.assertIn("shared_trunk_es", definitions)
        self.assertEqual(definitions["notebook_shared_trunk"].parameter_count, 48516)


if __name__ == "__main__":
    unittest.main()
