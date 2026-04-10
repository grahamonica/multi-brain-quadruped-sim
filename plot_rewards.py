"""Plot mean reward over generations from a checkpoint file."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from brains.runtime import discover_model_artifacts


def _default_checkpoint() -> Path:
    artifacts = discover_model_artifacts("checkpoints")
    if artifacts:
        return artifacts[0].checkpoint_path
    return Path("checkpoints/latest.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or _default_checkpoint()

    with np.load(checkpoint_path, allow_pickle=False) as ckpt:
        history = ckpt["rewards_history"].tolist()
        generation = int(ckpt["generation"])

    if not history:
        print("No rewards history found in checkpoint.")
        return

    history = history[-100:]
    start_gen = generation - len(history) + 1
    gens = list(range(start_gen, generation + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(gens, history, color="#33cc66", linewidth=1.2)
    plt.xlabel("Generation")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Reward (last 100 gens, gen={generation}, latest={history[-1]:.1f})")
    plt.grid(True, color="#222", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
