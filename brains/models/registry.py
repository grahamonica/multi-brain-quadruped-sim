"""Registered policy model definitions.

The project currently has one trainable policy implementation. Keeping it behind
this small catalog means the training entrypoint, artifact store, and viewer can
already talk in terms of model types before additional architectures land.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CURRENT_MODEL_TYPE = "shared_trunk_es"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.json"


@dataclass(frozen=True)
class ModelDefinition:
    type: str
    architecture: str
    trainer: str
    input_size: int
    output_size: int
    parameter_count: int
    description: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "type": self.type,
            "architecture": self.architecture,
            "trainer": self.trainer,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "parameter_count": self.parameter_count,
            "description": self.description,
        }


def _default_definitions() -> dict[str, ModelDefinition]:
    return {
        CURRENT_MODEL_TYPE: ModelDefinition(
            type=CURRENT_MODEL_TYPE,
            architecture="shared_trunk_motor_lanes",
            trainer="openai_es",
            input_size=48,
            output_size=4,
            parameter_count=48516,
            description="Current JAX policy vector with shared trunk and per-motor lanes.",
        ),
    }


def model_registry_path() -> Path:
    return Path(os.environ.get("QUADRUPED_MODEL_REGISTRY", str(DEFAULT_MODEL_REGISTRY_PATH)))


def _definition_from_mapping(data: Mapping[str, Any]) -> ModelDefinition:
    definition = ModelDefinition(
        type=str(data["type"]),
        architecture=str(data["architecture"]),
        trainer=str(data["trainer"]),
        input_size=int(data["input_size"]),
        output_size=int(data["output_size"]),
        parameter_count=int(data["parameter_count"]),
        description=str(data.get("description", "")),
    )
    validate_model_definition(definition)
    return definition


def validate_model_definition(definition: ModelDefinition) -> None:
    if not definition.type:
        raise ValueError("model definition type must be non-empty.")
    if not all(ch.isalnum() or ch in {"_", "-", "."} for ch in definition.type):
        raise ValueError("model definition type may only contain letters, numbers, underscores, hyphens, and periods.")
    if not definition.architecture:
        raise ValueError("model definition architecture must be non-empty.")
    if definition.trainer != "openai_es":
        raise ValueError("only openai_es model definitions are supported by the current trainer.")
    if definition.input_size <= 0 or definition.output_size <= 0:
        raise ValueError("model definition input_size and output_size must be > 0.")
    if definition.parameter_count <= 0:
        raise ValueError("model definition parameter_count must be > 0.")


def load_model_definitions(registry_path: str | Path | None = None) -> dict[str, ModelDefinition]:
    definitions = _default_definitions()
    path = Path(registry_path) if registry_path is not None else model_registry_path()
    if not path.exists():
        return definitions
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_models = payload.get("models", []) if isinstance(payload, Mapping) else []
    if not isinstance(raw_models, list):
        raise ValueError(f"{path} must contain a 'models' list.")
    for raw_model in raw_models:
        if not isinstance(raw_model, Mapping):
            raise ValueError(f"{path} contains a non-object model definition.")
        definition = _definition_from_mapping(raw_model)
        definitions[definition.type] = definition
    return definitions


def _write_external_definitions(definitions: Mapping[str, ModelDefinition], registry_path: str | Path) -> Path:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    external = [
        definition.to_dict()
        for model_type, definition in sorted(definitions.items())
        if model_type != CURRENT_MODEL_TYPE
    ]
    path.write_text(json.dumps({"models": external}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


_MODEL_DEFINITIONS: dict[str, ModelDefinition] = load_model_definitions()


def register_model_definition(
    definition: ModelDefinition,
    *,
    persist: bool = False,
    registry_path: str | Path | None = None,
) -> ModelDefinition:
    """Register a model definition for this process and optionally persist it.

    Persisting writes `configs/model_registry.json` by default so a notebook-made
    model type is still known when the viewer or headless trainer starts later.
    """

    validate_model_definition(definition)
    _MODEL_DEFINITIONS[definition.type] = definition
    if persist:
        path = Path(registry_path) if registry_path is not None else model_registry_path()
        merged = load_model_definitions(path)
        merged[definition.type] = definition
        _write_external_definitions(merged, path)
    return definition


def refresh_model_definitions(registry_path: str | Path | None = None) -> dict[str, ModelDefinition]:
    _MODEL_DEFINITIONS.clear()
    _MODEL_DEFINITIONS.update(load_model_definitions(registry_path))
    return dict(_MODEL_DEFINITIONS)


def list_model_definitions() -> tuple[ModelDefinition, ...]:
    return tuple(_MODEL_DEFINITIONS.values())


def get_model_definition(model_type: str) -> ModelDefinition:
    try:
        return _MODEL_DEFINITIONS[model_type]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_DEFINITIONS))
        raise ValueError(f"Unknown model type {model_type!r}. Registered model types: {known}.") from exc
