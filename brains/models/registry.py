"""Registered policy model definitions.

The project currently has one trainable policy implementation. Keeping it behind
this small catalog means the training entrypoint, artifact store, and viewer can
already talk in terms of model types before additional architectures land.
"""

from __future__ import annotations

from dataclasses import dataclass


CURRENT_MODEL_TYPE = "shared_trunk_es"


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


_MODEL_DEFINITIONS: dict[str, ModelDefinition] = {
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


def list_model_definitions() -> tuple[ModelDefinition, ...]:
    return tuple(_MODEL_DEFINITIONS.values())


def get_model_definition(model_type: str) -> ModelDefinition:
    try:
        return _MODEL_DEFINITIONS[model_type]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_DEFINITIONS))
        raise ValueError(f"Unknown model type {model_type!r}. Registered model types: {known}.") from exc
