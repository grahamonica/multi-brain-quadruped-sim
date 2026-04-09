"""Model catalog for trainable quadruped policies."""

from .registry import CURRENT_MODEL_TYPE, ModelDefinition, get_model_definition, list_model_definitions

__all__ = [
    "CURRENT_MODEL_TYPE",
    "ModelDefinition",
    "get_model_definition",
    "list_model_definitions",
]
