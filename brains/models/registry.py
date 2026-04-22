"""Registered policy model definitions.

The project currently has one trainable policy implementation. Keeping it behind
this small catalog means the training entrypoint, artifact store, and viewer can
already talk in terms of model types before additional architectures land.
"""

from __future__ import annotations

import json
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from brains.config import RuntimeSpec

CURRENT_MODEL_TYPE = "shared_trunk_es"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.json"
INLINE_PREFIX = "inline:"


@dataclass(frozen=True)
class ModelDefinition:
    type: str
    architecture: str
    trainer: str
    input_size: int
    output_size: int
    parameter_count: int
    description: str
    policy_entrypoint: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "type": self.type,
            "architecture": self.architecture,
            "trainer": self.trainer,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "parameter_count": self.parameter_count,
            "description": self.description,
            "policy_entrypoint": self.policy_entrypoint,
        }


@dataclass(frozen=True)
class PolicyPlugin:
    init_params: Callable[[jax.Array], Any]
    zero_state: Callable[[], Any]
    step: Callable[[Any, Any, jax.Array, jax.Array, jax.Array], tuple[Any, jax.Array, jax.Array]]


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
        policy_entrypoint=str(data["policy_entrypoint"]) if data.get("policy_entrypoint") is not None else None,
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
    if not definition.trainer:
        raise ValueError("model definition trainer must be non-empty.")
    if definition.input_size <= 0 or definition.output_size <= 0:
        raise ValueError("model definition input_size and output_size must be > 0.")
    if definition.parameter_count <= 0:
        raise ValueError("model definition parameter_count must be > 0.")
    if definition.policy_entrypoint is not None and ":" not in definition.policy_entrypoint:
        raise ValueError(
            "model definition policy_entrypoint must use 'module.path:function_name' or 'inline:name' format."
        )


_INLINE_POLICY_FACTORIES: dict[str, Callable[[RuntimeSpec, ModelDefinition], Any]] = {}


def register_inline_policy_factory(name: str, factory: Callable[[RuntimeSpec, ModelDefinition], Any]) -> str:
    """Register an in-process policy factory and return the entrypoint string."""

    if not name or any(ch in name for ch in (":", "/")):
        raise ValueError("Inline factory name must be non-empty and free of ':' and '/'.")
    if not callable(factory):
        raise ValueError("Inline factory must be callable.")
    _INLINE_POLICY_FACTORIES[name] = factory
    return f"{INLINE_PREFIX}{name}"


def _resolve_policy_entrypoint(entrypoint: str) -> Callable[[RuntimeSpec, ModelDefinition], Any]:
    if entrypoint.startswith(INLINE_PREFIX):
        name = entrypoint[len(INLINE_PREFIX):]
        factory = _INLINE_POLICY_FACTORIES.get(name)
        if factory is None:
            known = ", ".join(sorted(_INLINE_POLICY_FACTORIES)) or "<none>"
            raise ValueError(f"Inline policy factory {name!r} is not registered. Registered: {known}.")
        return factory
    if ":" not in entrypoint:
        raise ValueError(
            "policy_entrypoint must use 'module.path:function_name' or 'inline:name' format. "
            f"Received: {entrypoint!r}."
        )
    module_name, function_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Policy entrypoint {entrypoint!r} did not resolve to a callable.")
    return factory


def _policy_plugin_from_payload(payload: Any) -> PolicyPlugin:
    if isinstance(payload, PolicyPlugin):
        return payload

    if hasattr(payload, "init_params") and hasattr(payload, "zero_state") and hasattr(payload, "step"):
        return PolicyPlugin(
            init_params=payload.init_params,
            zero_state=payload.zero_state,
            step=payload.step,
        )

    if isinstance(payload, dict):
        init_params = payload.get("init_params")
        zero_state = payload.get("zero_state")
        step = payload.get("step")
        if callable(init_params) and callable(zero_state) and callable(step):
            return PolicyPlugin(
                init_params=init_params,
                zero_state=zero_state,
                step=step,
            )

    raise ValueError(
        "Policy plugin payload must provide callable init_params, zero_state, and step members."
    )


def load_policy_plugin(spec: RuntimeSpec, model_definition: ModelDefinition) -> PolicyPlugin | None:
    entrypoint = model_definition.policy_entrypoint
    if not entrypoint:
        return None
    factory = _resolve_policy_entrypoint(entrypoint)
    payload = factory(spec, model_definition)
    plugin = _policy_plugin_from_payload(payload)

    params = plugin.init_params(jax.random.PRNGKey(0))
    state = plugin.zero_state()
    obs = jnp.zeros((model_definition.input_size,), dtype=jnp.float32)
    next_state, output, next_key = plugin.step(
        params,
        state,
        obs,
        jax.random.PRNGKey(1),
        jnp.float32(0.0),
    )
    output_shape = tuple(getattr(output, "shape", ()))
    if output_shape != (model_definition.output_size,):
        raise ValueError(
            "Policy plugin output size mismatch. "
            f"Model definition expects {model_definition.output_size}, plugin produced {output_shape}."
        )
    if next_state is None:
        raise ValueError("Policy plugin step returned None state. Return state passthrough if stateless.")
    if tuple(getattr(next_key, "shape", ())) != (2,):
        raise ValueError("Policy plugin step must return a JAX PRNG key in the third position.")

    return plugin


def load_model_definitions(registry_path: str | Path | None = None) -> dict[str, ModelDefinition]:
    definitions = {
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
