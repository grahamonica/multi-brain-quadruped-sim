# PACE Batch Jobs (Large Scale)

This folder contains SLURM batch scripts that mirror the notebook workflows,
but run at larger scale (more generations, larger populations, longer episodes).

All scripts assume they are launched from the repository root on PACE:

```bash
sbatch pace_jobs/<script>.sbatch
```

Common expectations:

- `venv` exists at `./venv` on the compute node file system.
- `train_headless.py` and configs are available from the submit directory.
- `logs/` and `checkpoints/` are created automatically.

Notebook mapping:

- `model_playground.ipynb` -> `train_notebook_shared_trunk_large.sbatch`
- `brain_command_primitives_playground.ipynb` -> `train_notebook_command_primitives_large.sbatch`
- `brain_cnn_playground.ipynb` -> `train_notebook_cnn_patch_large.sbatch`
- `brain_snn_playground.ipynb` -> `train_notebook_snn_large.sbatch`
- `brain_vla_hf_playground.ipynb` -> `run_vla_hf_harness_large.sbatch` (evaluation harness run, not ES training)

