"""Shared notebook execution helpers.

Keep generic setup/training/saving logic here so notebooks can focus on
model-specific code and configuration.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from brains.config import RuntimeSpec
from brains.jax_trainer import ESTrainer
from brains.runtime import create_model_run_paths, discover_model_artifacts, write_model_manifest


def train_and_save_with_progress(
    *,
    spec: RuntimeSpec,
    repo_root: str | Path,
    model_type: str,
    log_id: str,
    seed: int = 11,
    generations: int = 1,
    print_step_progress: bool = True,
    population_size: int | None = None,
    sigma: float | None = None,
    learning_rate: float | None = None,
    parent_elite_count: int | None = None,
    goal_radius_m: float | None = None,
    goal_height_m: float | None = None,
    spawn_strategy: str | None = None,
    spawn_x_range_m: tuple[float, float] | None = None,
    spawn_y_range_m: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run training and save checkpoint/manifest with progress prints.

    Notebook callers can override key training/world knobs per run.
    """

    training = spec.training
    if population_size is not None:
        training = replace(training, population_size=int(population_size))
    if sigma is not None:
        training = replace(training, sigma=float(sigma))
    if learning_rate is not None:
        training = replace(training, learning_rate=float(learning_rate))
    if parent_elite_count is not None:
        training = replace(training, parent_elite_count=int(parent_elite_count))

    goals = spec.goals
    if goal_radius_m is not None:
        goals = replace(goals, radius_m=float(goal_radius_m))
    if goal_height_m is not None:
        goals = replace(goals, height_m=float(goal_height_m))

    spawn_policy = spec.spawn_policy
    if spawn_strategy is not None:
        spawn_policy = replace(spawn_policy, strategy=str(spawn_strategy))
    if spawn_x_range_m is not None:
        spawn_policy = replace(spawn_policy, x_range_m=(float(spawn_x_range_m[0]), float(spawn_x_range_m[1])))
    if spawn_y_range_m is not None:
        spawn_policy = replace(spawn_policy, y_range_m=(float(spawn_y_range_m[0]), float(spawn_y_range_m[1])))

    run_spec = replace(spec, training=training, goals=goals, spawn_policy=spawn_policy)
    run_spec.validate()

    root = Path(repo_root)
    trainer = ESTrainer(seed=seed, spec=run_spec, model_id=f"{model_type}_{log_id}", log_id=log_id)
    start = time.perf_counter()

    for generation_index in range(int(generations)):
        print(f"generation {generation_index + 1}/{generations} start", flush=True)

        def _on_step(step_message: dict[str, Any]) -> None:
            if not print_step_progress:
                return
            step = int(step_message.get("step", 0)) + 1
            total_steps = int(step_message.get("total_steps", 0))
            reward = float(step_message.get("reward", 0.0))
            selected_command = step_message.get("selected_command")
            command_suffix = f" | command={selected_command}" if selected_command else ""
            print(
                f"generation {generation_index + 1}/{generations} "
                f"step {step}/{total_steps} | reward={reward:.4f}{command_suffix}",
                flush=True,
            )

        trainer.run_generation(on_step=_on_step if print_step_progress else None)
        print(
            f"generation {generation_index + 1}/{generations} done | "
            f"mean_reward={trainer.state.mean_reward:.4f} | "
            f"best_reward={trainer.state.best_reward:.4f}",
            flush=True,
        )

    paths = create_model_run_paths(root / "checkpoints", model_type, log_id)
    trainer.model_id = paths.id
    trainer.log_id = paths.log_id
    saved = trainer.save_checkpoint(paths.latest_path)
    write_model_manifest(
        paths,
        run_spec,
        saved,
        generation=trainer.state.generation,
        best_reward=trainer.state.best_reward,
        mean_reward=trainer.state.mean_reward,
    )
    elapsed = time.perf_counter() - start
    return {
        "model_id": paths.id,
        "log_id": paths.log_id,
        "latest": str(saved),
        "generation": trainer.state.generation,
        "mean_reward": trainer.state.mean_reward,
        "best_reward": trainer.state.best_reward,
        "elapsed_s": elapsed,
        "training": {
            "population_size": run_spec.training.population_size,
            "sigma": run_spec.training.sigma,
            "learning_rate": run_spec.training.learning_rate,
            "parent_elite_count": run_spec.training.parent_elite_count,
        },
        "goals": {
            "strategy": run_spec.goals.strategy,
            "radius_m": run_spec.goals.radius_m,
            "height_m": run_spec.goals.height_m,
        },
        "spawn_policy": {
            "strategy": run_spec.spawn_policy.strategy,
            "x_range_m": list(run_spec.spawn_policy.x_range_m),
            "y_range_m": list(run_spec.spawn_policy.y_range_m),
        },
        "visible_artifacts": [artifact.id for artifact in discover_model_artifacts(root / "checkpoints")][:8],
    }
