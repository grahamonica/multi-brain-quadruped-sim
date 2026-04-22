"""Hugging Face vision-language adapter for command-level quadruped control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _extract_generated_text(payload: Any) -> str:
    """Extract a single text response from a transformers pipeline output.

    HF pipelines return a list-of-dicts; image-text-to-text adds a `chunks` shape.
    Recurse on lists, look up text-bearing keys on dicts, return strings as-is.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list) and payload:
        return _extract_generated_text(payload[0])
    if isinstance(payload, dict):
        for key in ("generated_text", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = [chunk["content"] if isinstance(chunk, dict) else chunk for chunk in value]
                parts = [part for part in parts if isinstance(part, str)]
                if parts:
                    return "\n".join(parts)
    raise ValueError(f"Could not extract generated text from model payload of type {type(payload).__name__}.")


@dataclass
class HuggingFaceCommandVLA:
    """Adapter that asks a HF VLM/VLA model to select one command option."""

    model_id: str
    max_new_tokens: int = 24
    device: int | str | None = None
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "transformers is required for HuggingFaceCommandVLA. "
                "Install it with: pip install transformers torch"
            ) from exc

        pipeline_kwargs: dict[str, Any] = {
            "task": "image-text-to-text",
            "model": self.model_id,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device is not None:
            pipeline_kwargs["device"] = self.device

        self._pipeline = pipeline(**pipeline_kwargs)

    def _build_prompt(self, instruction: str, options: tuple[str, ...]) -> str:
        option_list = ", ".join(options)
        return (
            f"Instruction: {instruction}\n"
            f"Available commands: {option_list}\n"
            "Select exactly one command from the available commands. "
            "Respond with only the command name."
        )

    def choose_action(
        self,
        image_rgb: np.ndarray,
        instruction: str,
        options: tuple[str, ...],
        observation: dict[str, Any],
    ) -> str:
        del observation
        if image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be uint8.")

        prompt = self._build_prompt(instruction, options)
        response = self._pipeline(image=image_rgb, text=prompt, max_new_tokens=int(self.max_new_tokens))
        generated_text = _extract_generated_text(response)
        normalized = generated_text.strip().lower()

        for option in options:
            option_normalized = option.lower()
            if normalized == option_normalized:
                return option

        for option in options:
            option_normalized = option.lower()
            if option_normalized in normalized:
                return option

        known = ", ".join(options)
        raise ValueError(
            "Model output did not include a valid command option. "
            f"Output: {generated_text!r}. Valid options: {known}."
        )
