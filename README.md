# Multi-Brain Quadruped Sim

Config-driven quadruped training stack. The repo is organized as follows: a domain layer for the robot and environment, packaged runtime code under `brains/`, declarative configs, structured run logs, regression tests, and one live viewer stack. The runtime uses JAX for policy math and optimization, and MuJoCo for rollout and playback.

**Repo Layout**

- `quadruped/`: domain models for body, leg, motor, robot, and environment/task representation.
- `brains/sim/`: runtime rollout code, MuJoCo integration, and internal simulator helpers.
- `brains/api/`: FastAPI websocket service for the viewer app.
- `brains/config/`: typed YAML/JSON runtime spec loading and validation.
- `brains/infra/`: structured logs and per-run artifact helpers.
- `brains/runtime/`: checkpoint compatibility helpers, model weight paths, and model manifests.
- `brains/models/`: registered trainable model definitions. The current registered model is `shared_trunk_es`.
- `brains/quality/`: quality gates and fixed-seed regression tooling.
- `brains/jax_trainer.py`: the current trainable brain implementation.
- `configs/`: declarative runtime specs in YAML.
- `frontend/`: React + Vite viewer that consumes websocket metadata, model lists, reward targets, and replay frame batches.
- `tests/`: config validation, smoke quality checks, and fixed-seed regression tests.
- `checkpoints/`: saved model artifacts. New runs are stored under `checkpoints/<model_type>_<log_id>/`.
- `logs/`: per-run structured artifacts and metrics. This directory is gitignored.
- `.github/workflows/quality-gates.yml`: CI job that runs the unittest suite on push and PRs.

**Architecture**

The runtime is split into five layers:

1. Domain layer: [quadruped/robot.py](quadruped/robot.py), [quadruped/leg.py](quadruped/leg.py), [quadruped/motor.py](quadruped/motor.py), and [quadruped/environment.py](quadruped/environment.py) define the robot and task in a logical, real-world shape.
2. Config layer: [brains/config/schema.py](brains/config/schema.py) resolves YAML/JSON into a typed runtime spec.
3. Model layer: [brains/models/registry.py](brains/models/registry.py) names the trainable policy types that headless jobs and the viewer can address.
4. Runtime layer: [brains/sim/mujoco_backend.py](brains/sim/mujoco_backend.py) owns rollout execution, and [brains/sim/mujoco_model_builder.py](brains/sim/mujoco_model_builder.py) is the only place that translates domain objects into MuJoCo MJCF. [brains/sim/jax_backend.py](brains/sim/jax_backend.py) remains only as an internal reference helper for quality checks.
5. Training and service layer: [brains/jax_trainer.py](brains/jax_trainer.py) contains the current ES trainer implementation. That trainer keeps optimization and policy math in JAX and always drives rollout through MuJoCo. The service layer exposes the viewer and headless workflows on top of that shared trainer.


**Runtime Backend**

The runtime uses the following backend configuration:

```yaml
simulator:
  backend: unified
  render: false
  deterministic_mode: true
  mujoco:
    timestep_s: 0.0025
    integrator: implicitfast
    solver: Newton
    solver_iterations: 100
    line_search_iterations: 50
    noslip_iterations: 4
    contact_margin_m: 0.002
    actuator_force_limit: 12.0
    velocity_servo_gain: 6.0
    joint_range_rad: [-1.1, 1.1]
```

`backend: unified` means:

- JAX owns the policy parameter vector, ES update step, and other compute-heavy math.
- MuJoCo owns environment rollout, playback, and contact dynamics.

The rollout path is built from the existing domain objects. The flow is:

1. `config` -> typed `RuntimeSpec`
2. `RuntimeSpec` -> `QuadrupedRobot` and `SimulationEnvironment`
3. domain models -> MuJoCo MJCF in [brains/sim/mujoco_model_builder.py](brains/sim/mujoco_model_builder.py)
4. compiled model -> `MuJoCoBackend`
5. runtime backend -> trainer, quality gates, APIs, and viewer app metadata/frame batches

The viewer app consumes backend metadata, available model artifacts, selected reward goals, and replay frame batches. Checkpoints remain config-aware.

**Runtime Spec**

The trainer reads a resolved environment/task spec from YAML or JSON. The main configs live at [configs/default.yaml](configs/default.yaml) and [configs/smoke.yaml](configs/smoke.yaml).

The runtime is fed through explicit domain objects in [quadruped/robot.py](quadruped/robot.py), [quadruped/leg.py](quadruped/leg.py), [quadruped/motor.py](quadruped/motor.py), and [quadruped/environment.py](quadruped/environment.py). MuJoCo uses that shared source of truth and compiles it into MJCF instead of duplicating dimensions or terrain constants in rollout code.

Supported sections:

- `model`: trainable model type, architecture label, and trainer family.
- `simulator`: runtime mode and MuJoCo solver/control settings.
- `terrain`: stepped arena or flat terrain, field bounds, step count/width/height, floor height.
- `goals`: radial random goals or a fixed goal.
- `spawn_policy`: origin, fixed points, or uniform spawn box.
- `friction`: static/kinetic foot friction and body friction.
- `robot`: body dimensions, masses, leg geometry, motor limits, and contact sampling.
- `physics`: gravity, contact stiffness/damping, substep budget, and sleep thresholds.
- `episode`: episode length, brain timestep, lifespan, selection cadence, and goal radius.
- `reward`: progress/noise/tipping/climbing reward parameters.
- `training`: ES population size, sigma, learning rate, and elite count.
- `quality_gates`: fast runtime checks and performance thresholds.
- `logging`: log level and artifact filenames.

Checkpoint resumes are config-aware. A resume attempt only fails when the resolved runtime spec differs in a meaningful way.

**Install**

```bash
python3 -m pip install -r requirements.txt
```

For the viewer app:

```bash
cd frontend
npm install
```

**Docker Setup (Optional)**

For reproducible environments and CI/CD compatibility:

```bash
# Build Docker image with pinned dependencies
docker build -t multi-brain-quadruped-sim:latest .

# Run tests in Docker (same environment as CI)
docker run --rm multi-brain-quadruped-sim:latest

# Run headless training in Docker
docker run --rm \
  -v $(pwd):/workspace \
  multi-brain-quadruped-sim:latest \
  python train_headless.py --config configs/smoke.yaml --generations 10
```

For GPU acceleration (requires `nvidia-docker2`):

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  multi-brain-quadruped-sim:latest \
  python train_headless.py --config configs/default.yaml --generations 100
```
To locally mimic the Github Actions Linux behavior run this (for a mac):
docker run --rm --platform linux/amd64 multi-brain-quadruped-sim:latest

**Quick Start**

Run the fast validation profile end to end:

```bash
python3 run_quality_gates.py --config configs/smoke.yaml
python3 train_headless.py --config configs/smoke.yaml --generations 1
python3 main.py --config configs/smoke.yaml
```

If you only want the viewer for the current model:

```bash
python3 main.py --config configs/smoke.yaml
```

**Run Quality Gates**

Fast runtime validation against a config:

```bash
python3 run_quality_gates.py --config configs/smoke.yaml
```

The built-in quality suite supports two profiles:

- `runtime`: validates the actual unified MuJoCo-backed runtime.
- `reference`: runs faster internal reference checks against the legacy JAX simulator helpers.

The smoke config uses the `runtime` profile. That suite covers:

- invalid spawn detection
- collision sanity checks
- determinism checks
- unstable-state detection
- MJCF/model compilation succeeds
- reset pose is contact-safe
- zero-action rollout stays numerically stable
- identical seed reproduces the same rollout
- the internal JAX reference path and the runtime path show bounded drift on a fixed smoke rollout
- MuJoCo rollout time stays inside the configured budget

Fixed-seed regression coverage is enforced in the test suite using the smoke config baseline at [tests/fixtures/smoke_regression_baseline.json](tests/fixtures/smoke_regression_baseline.json).

Run the repo tests locally:

```bash
python3 -m unittest discover -s tests -v
```

**Run Viewer**

Unified viewer launcher:

```bash
python3 main.py --config configs/default.yaml
```

This launcher starts the only UI stack in the repo:

- a FastAPI websocket backend, default port `8000`
- the Vite viewer app, default port `5173`

Both ports are configurable:

```bash
python3 main.py --config configs/default.yaml --api-port 8001 --viewer-port 5174
```

It uses the same runtime config pattern as the headless path, so terrain and robot geometry shown in the viewer app come from backend metadata instead of duplicated viewer app constants. The viewer discovers saved model artifacts under `checkpoints/`, lets the user select one, and runs playback-only simulation for the selected weights.

**Website Deployment**

The frontend is a static Vite build and can be hosted by any static web host:

```bash
cd frontend
VITE_WS_URL=wss://api.your-domain.example/ws npm run build
```

Deploy `frontend/dist/` to the web host. Run the backend as a separate service behind HTTPS/WSS:

```bash
QUADRUPED_CONFIG=configs/default.yaml \
QUADRUPED_CHECKPOINT_ROOT=checkpoints \
QUADRUPED_CORS_ORIGINS=https://your-domain.example \
uvicorn brains.api.live:app --host 0.0.0.0 --port 8000
```

For local development, `main.py` still starts both processes and injects the local websocket port automatically.

**Live Viewer Streaming**

The viewer is split into compute and presentation clocks:

- The backend simulates replay steps from the selected checkpoint and sends small `frame_batch` websocket messages.
- The browser stores those frames in a short queue and renders them on `requestAnimationFrame` at `episode.brain_dt_s`.
- If MuJoCo computes faster than real time, the UI already has several frames queued and playback stays smooth.
- If the user picks another model or places a reward target on the map, the viewer app sends `select_model` or `set_goal`; the backend interrupts the current replay and starts a new stream ID.

That is the efficient part: the simulation thread can compute ahead, while the browser consumes frames at a steady visual cadence instead of blocking on one websocket message per paint.

**How Config Flows Through The System**

1. A YAML or JSON file is loaded through [brains/config/schema.py](brains/config/schema.py).
2. That config is converted into domain objects in `quadruped/`.
3. The single ES trainer applies those values to its runtime constants and cached tensors.
4. The runtime backend is constructed from those same resolved objects and attached under that trainer.
5. APIs and the viewer app receive metadata derived from the same resolved config.
6. Checkpoints embed the resolved config so the runtime can reject genuinely incompatible resumes.

**Train Headless**

```bash
python3 train_headless.py --config configs/default.yaml --generations 100 --seed 42
```

Useful variants:

```bash
python3 train_headless.py --config configs/smoke.yaml --quality-only
python3 train_headless.py --config configs/default.yaml --model-type shared_trunk_es --log-id 12389748 --generations 100
python3 train_headless.py --config configs/default.yaml --resume checkpoints/shared_trunk_es_12389748/latest.npz
python3 train_headless.py --config configs/default.yaml --population-size 64 --episode-seconds 20
```

On a SLURM/PACE-style cluster:

```bash
mkdir -p logs checkpoints
CONFIG=configs/default.yaml MODEL_TYPE=shared_trunk_es LOG_ID=12389748 GENERATIONS=100 SEED=42 sbatch train.sbatch
```

By default the headless entrypoint runs the fast quality gates before training starts. Use `--skip-quality-gates` only when you intentionally want to bypass that preflight.

**Model Playground Notebook**

[notebooks/model_playground.ipynb](notebooks/model_playground.ipynb) is a Google Drive-backed Jupyter workflow for creating model variants that use the current `shared_trunk_es` trainer shape. It mounts Drive, imports `brains` from the Drive repo copy, persists a model definition to `configs/model_registry.json`, initializes or briefly trains the variant, then saves weights into:

```text
checkpoints/<model_type>_<log_id>/latest.npz
```

Those saved artifacts are discoverable by the viewer and by `train_headless.py --resume`.

**Scripted Harnesses**

The harnesses under `brains/harnesses/` are intentionally separate from trainable models for now:

- `DirectionHarness`: maps plain directions to available scripted options like `trot`, `turn_left`, `turn_right`, `back_up`, `stand`, and `stop`.
- `HeadCameraHarness`: builds on the direction harness and injects a front torso camera named `head_camera` for future VLA observation experiments.

**Run Artifacts**

Each headless run writes a timestamped artifact directory under `logs/` containing:

- `resolved_config.yaml`: the exact runtime spec used for the run
- `events.jsonl`: structured event logs
- `metrics.jsonl`: generation-level metrics
- `quality_report.json`: preflight validation results
- `summary.json`: terminal run summary

Training model weights are written to model-specific directories:

- `checkpoints/<model_type>_<log_id>/latest.npz`
- `checkpoints/<model_type>_<log_id>/best.npz`
- `checkpoints/<model_type>_<log_id>/manifest.json`

For example:

```text
checkpoints/shared_trunk_es_12389748/latest.npz
```

Old model weights with incompatible parameter shapes or genuinely mismatched configs are rejected during load instead of failing later during rollout. Reusing `--log-id` resumes that model directory when compatible; a new log ID starts a fresh experiment. Explicit `--resume` still fails loudly so you do not accidentally resume the wrong run.

**Runtime Notes**

- The runtime backend always uses MuJoCo for rollout.
- JAX is still used internally for policy math, optimizer updates, and reference/regression tooling.
- The smoke config intentionally uses `terrain.kind: flat` to keep the runtime quality gates narrow and stable.

**Plot Rewards**

```bash
python3 plot_rewards.py --checkpoint checkpoints/shared_trunk_es_12389748/latest.npz
```

**Notes**

- `train_headless.py --save-every` is retained only for CLI compatibility. Numbered generation model weights are not written.
- The launchers prefer the current Python interpreter first. That avoids silently using a stale repo-local venv when it differs from the environment you are actively running.
