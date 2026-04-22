"""Reference policy plugin used by notebook-defined model types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from brains.config import RuntimeSpec
from brains.models.registry import ModelDefinition


@dataclass(frozen=True)
class _LinearPolicy:
    input_size: int
    output_size: int

    def init_params(self, key: jax.Array) -> dict[str, jax.Array]:
        key_w, key_b = jax.random.split(key)
        scale = 1.0 / math.sqrt(max(self.input_size, 1))
        return {
            "w": jax.random.normal(key_w, (self.output_size, self.input_size), dtype=jnp.float32) * jnp.float32(scale),
            "b": jax.random.uniform(key_b, (self.output_size,), dtype=jnp.float32, minval=-0.1, maxval=0.1),
        }

    def zero_state(self) -> jax.Array:
        return jnp.zeros((1,), dtype=jnp.float32)

    def step(
        self,
        params: dict[str, jax.Array],
        state: jax.Array,
        obs: jax.Array,
        key: jax.Array,
        noise_scale: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits = jnp.matmul(params["w"], obs) + params["b"]
        key, noise_key = jax.random.split(key)
        noisy = logits + (jax.random.normal(noise_key, (self.output_size,), dtype=jnp.float32) * jnp.maximum(noise_scale, 0.0))
        output = jnp.tanh(noisy)
        return state, output, key


def build_linear_policy(spec: RuntimeSpec, model_definition: ModelDefinition) -> Any:
    del spec
    return _LinearPolicy(
        input_size=int(model_definition.input_size),
        output_size=int(model_definition.output_size),
    )
