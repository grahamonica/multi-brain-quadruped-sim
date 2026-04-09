"""Top-level brain/trainer package exports.

Trainer exports are resolved lazily so config and artifact tooling can import
`brains.*` modules without initializing JAX or MuJoCo.
"""

_TRAINER_EXPORTS = {
    "ESTrainer",
    "JaxESTrainer",
    "TrainingState",
    "apply_runtime_spec",
    "current_environment_model",
    "current_robot_model",
    "current_runtime_spec",
}

__all__ = sorted(_TRAINER_EXPORTS)


def __getattr__(name: str):
    if name not in _TRAINER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    trainer_module = importlib.import_module(".jax_trainer", __name__)
    return getattr(trainer_module, name)
