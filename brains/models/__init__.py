"""Model catalog for trainable quadruped policies."""

from .registry import (
    CURRENT_MODEL_TYPE,
    DEFAULT_MODEL_REGISTRY_PATH,
    ModelDefinition,
    get_model_definition,
    list_model_definitions,
    load_model_definitions,
    model_registry_path,
    refresh_model_definitions,
    register_model_definition,
    validate_model_definition,
)

__all__ = [
    "CURRENT_MODEL_TYPE",
    "DEFAULT_MODEL_REGISTRY_PATH",
    "ModelDefinition",
    "get_model_definition",
    "list_model_definitions",
    "load_model_definitions",
    "model_registry_path",
    "refresh_model_definitions",
    "register_model_definition",
    "validate_model_definition",
]
