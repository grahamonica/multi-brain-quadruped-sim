"""Plot mean reward over generations from a checkpoint file."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.npz"))
    args = parser.parse_args()

    with np.load(args.checkpoint, allow_pickle=False) as ckpt:
        history = ckpt["rewards_history"].tolist()
        generation = int(ckpt["generation"])

    if not history:
        print("No rewards history found in checkpoint.")
        return

    gens = list(range(1, len(history) + 1))

    plt.figure(figsize=(12, 5))
    plt.plot(gens, history, color="#33cc66", linewidth=1.2)
    plt.xlabel("Generation")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Reward over Generations  (gen={generation}, latest={history[-1]:.1f})")
    plt.grid(True, color="#222", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
