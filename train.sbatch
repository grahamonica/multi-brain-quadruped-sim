#!/bin/bash
#SBATCH --job-name=quadruped-train
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Activate virtual environment
source venv/bin/activate

# Run headless training (100 generations, resume from best if exists)
python train_headless.py --generations 100
