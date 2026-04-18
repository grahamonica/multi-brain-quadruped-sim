"""Run a Hugging Face VLA/VLM against the MuJoCo head-camera harness."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from brains.config import DEFAULT_CONFIG_PATH, load_runtime_spec
from brains.harnesses import CameraConfig, VLAHarness
from brains.models import HuggingFaceCommandVLA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HF VLA model in the MuJoCo head-camera harness.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Runtime config path.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id.")
    parser.add_argument("--instruction", default="walk forward", help="Natural-language task instruction for the model.")
    parser.add_argument("--steps", type=int, default=120, help="Number of control steps to execute.")
    parser.add_argument("--camera-width", type=int, default=224, help="Head-camera width.")
    parser.add_argument("--camera-height", type=int, default=224, help="Head-camera height.")
    parser.add_argument("--camera-fovy", type=float, default=70.0, help="Head-camera vertical field of view in degrees.")
    parser.add_argument("--max-new-tokens", type=int, default=24, help="Max generated tokens per VLA inference.")
    parser.add_argument("--device", default=None, help="Transformers pipeline device (for example cpu, cuda:0, 0).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = load_runtime_spec(args.config)
    if isinstance(args.device, str) and args.device.lstrip("-").isdigit():
        resolved_device: int | str | None = int(args.device)
    else:
        resolved_device = args.device

    camera = CameraConfig(
        width=int(args.camera_width),
        height=int(args.camera_height),
        fovy_deg=float(args.camera_fovy),
    )
    harness = VLAHarness(spec=spec, camera=camera)
    agent = HuggingFaceCommandVLA(
        model_id=str(args.model_id),
        max_new_tokens=int(args.max_new_tokens),
        device=resolved_device,
    )

    run = harness.run_vla(
        agent,
        instruction=str(args.instruction),
        steps=int(args.steps),
    )

    commands = [frame.get("vla", {}).get("action", "unknown") for frame in run.frames]
    histogram = Counter(commands)

    print(f"model_id: {args.model_id}")
    print(f"instruction: {args.instruction}")
    print(f"frames: {len(run.frames)}")
    print(f"total_reward: {run.total_reward:.4f}")
    print("command_histogram:")
    for command, count in sorted(histogram.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {command}: {count}")


if __name__ == "__main__":
    main()
