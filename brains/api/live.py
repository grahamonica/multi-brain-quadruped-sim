"""Live model viewer websocket service."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from brains.config import DEFAULT_CONFIG_PATH, load_runtime_spec
from brains.runtime import discover_model_artifacts, find_model_artifact, runtime_spec_from_checkpoint

from .common import (
    BroadcastHub,
    build_viewer_metadata,
    current_policy_params,
    single_step_to_frame,
    viewer_reset_steps,
)


FRAME_BATCH_SIZE = 12


class ReplayRestart(Exception):
    """Raised inside the replay callback when the user selects a new stream."""


@dataclass(frozen=True)
class ViewerSnapshot:
    selected_model_id: str | None
    goal_xyz: tuple[float, float, float] | None
    version: int


class ViewerControls:
    def __init__(self, selected_model_id: str | None = None) -> None:
        self._lock = threading.Lock()
        self._selected_model_id = selected_model_id
        self._goal_xyz: tuple[float, float, float] | None = None
        self._version = 0

    def snapshot(self) -> ViewerSnapshot:
        with self._lock:
            return ViewerSnapshot(self._selected_model_id, self._goal_xyz, self._version)

    def set_model(self, model_id: str | None) -> int:
        with self._lock:
            self._selected_model_id = model_id
            self._version += 1
            return self._version

    def set_goal(self, goal_xyz: tuple[float, float, float]) -> int:
        with self._lock:
            self._goal_xyz = goal_xyz
            self._version += 1
            return self._version

    def version_changed(self, version: int) -> bool:
        with self._lock:
            return self._version != version


def _load_service_config() -> tuple[Path, int]:
    config_path = Path(os.environ.get("QUADRUPED_CONFIG", str(DEFAULT_CONFIG_PATH)))
    seed = int(os.environ.get("QUADRUPED_SEED", "42"))
    return config_path, seed


def _checkpoint_root() -> Path:
    return Path(os.environ.get("QUADRUPED_CHECKPOINT_ROOT", "checkpoints"))


def _cors_origins() -> list[str]:
    raw_origins = os.environ.get("QUADRUPED_CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def _path_signature(path: Path | None) -> tuple[str, int] | None:
    if path is None:
        return None
    try:
        return str(path.resolve()), path.stat().st_mtime_ns
    except OSError:
        return None


def _models_message(selected_model_id: str | None) -> dict[str, Any]:
    artifacts = discover_model_artifacts(_checkpoint_root())
    return {
        "type": "models",
        "models": [artifact.to_message() for artifact in artifacts],
        "selected_model_id": selected_model_id,
    }


def _default_selected_model_id() -> str | None:
    artifacts = discover_model_artifacts(_checkpoint_root())
    return artifacts[0].id if artifacts else None


def _load_trainer_for_selection(
    config_path: Path,
    seed: int,
    selected_model_id: str | None,
) -> tuple[Any, Any | None, Path | None, tuple[dict[str, str], ...]]:
    from brains.jax_trainer import ESTrainer, apply_runtime_spec

    artifact = find_model_artifact(selected_model_id, _checkpoint_root()) if selected_model_id is not None else None
    skipped: list[dict[str, str]] = []
    checkpoint_path = artifact.checkpoint_path if artifact is not None else None

    spec = runtime_spec_from_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    if spec is None:
        spec = load_runtime_spec(config_path)
    apply_runtime_spec(spec)
    trainer = ESTrainer(
        seed=seed,
        spec=spec,
        model_id=artifact.id if artifact is not None else None,
        log_id=artifact.log_id if artifact is not None else None,
    )

    loaded_checkpoint: Path | None = None
    if checkpoint_path is not None:
        try:
            trainer.load_checkpoint(checkpoint_path)
            trainer.model_id = artifact.id if artifact is not None else trainer.model_id
            trainer.log_id = artifact.log_id if artifact is not None else trainer.log_id
            loaded_checkpoint = checkpoint_path
        except Exception as exc:
            skipped.append({"path": str(checkpoint_path), "reason": str(exc)})
    return trainer, artifact, loaded_checkpoint, tuple(skipped)


def _viewer_thread(hub: BroadcastHub, controls: ViewerControls, config_path: Path, seed: int) -> None:
    import jax
    import jax.numpy as jnp

    active_signature: tuple[str | None, tuple[str, int] | None] | None = None
    trainer: Any | None = None
    artifact = None
    loaded_checkpoint: Path | None = None
    skipped: tuple[dict[str, str], ...] = ()
    replay_index = 0

    while True:
        snapshot = controls.snapshot()
        selected_model_id = snapshot.selected_model_id
        if selected_model_id is None:
            selected_model_id = _default_selected_model_id()
            if selected_model_id is not None:
                controls.set_model(selected_model_id)
                snapshot = controls.snapshot()

        selected_artifact = find_model_artifact(selected_model_id, _checkpoint_root()) if selected_model_id else None
        checkpoint_signature = _path_signature(selected_artifact.checkpoint_path) if selected_artifact is not None else None
        next_signature = (selected_model_id, checkpoint_signature)
        if trainer is None or next_signature != active_signature:
            trainer, artifact, loaded_checkpoint, skipped = _load_trainer_for_selection(config_path, seed, selected_model_id)
            active_signature = next_signature
            hub.publish(build_viewer_metadata(trainer.spec, mode="viewer").to_message())

        assert trainer is not None
        spec = trainer.spec
        hub.publish(_models_message(selected_model_id))

        trainer._key, episode_key = jax.random.split(trainer._key)
        if snapshot.goal_xyz is not None:
            goal_xyz = jnp.asarray(snapshot.goal_xyz, dtype=jnp.float32)
        else:
            goal_xyz = trainer._random_goal()
        trainer.state.goal_xyz = tuple(float(value) for value in goal_xyz.tolist())
        replay_steps = viewer_reset_steps(spec)
        replay_index += 1
        stream_id = f"{selected_model_id or 'scratch'}:{snapshot.version}:{replay_index}"
        hub.publish(
            {
                "type": "generation",
                "stream_id": stream_id,
                "generation": trainer.state.generation,
                "mean_reward": trainer.state.mean_reward,
                "best_reward": trainer.state.best_reward,
                "top_rewards": trainer.top_rewards.tolist(),
                "rewards_history": trainer.state.rewards_history[-100:],
                "goal": list(trainer.state.goal_xyz),
                "selected_model_id": selected_model_id,
                "model": artifact.to_message() if artifact is not None else None,
                "checkpoint_loaded": str(loaded_checkpoint) if loaded_checkpoint is not None else None,
                "playback_only": True,
                "simulator_backend": spec.simulator.backend,
                "skipped_checkpoints": list(skipped),
            }
        )

        frame_batch: list[dict[str, Any]] = []

        def _flush_batch() -> None:
            nonlocal frame_batch
            if not frame_batch:
                return
            hub.publish(
                {
                    "type": "frame_batch",
                    "stream_id": stream_id,
                    "dt_s": spec.episode.brain_dt_s,
                    "frames": frame_batch,
                }
            )
            frame_batch = []

        def _emit(step_message: dict) -> None:
            if controls.version_changed(snapshot.version):
                raise ReplayRestart()
            frame = single_step_to_frame(step_message, generation=trainer.state.generation, spec=spec)
            frame["stream_id"] = stream_id
            frame["selected_model_id"] = selected_model_id
            frame_batch.append(frame)
            if len(frame_batch) >= FRAME_BATCH_SIZE:
                _flush_batch()

        try:
            trainer._run_logged_episode(current_policy_params(trainer), goal_xyz, episode_key, _emit, steps=replay_steps)
            _flush_batch()
        except ReplayRestart:
            frame_batch = []
            continue
        time.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path, seed = _load_service_config()
    hub = BroadcastHub()
    controls = ViewerControls(selected_model_id=_default_selected_model_id())
    hub.attach_loop(asyncio.get_running_loop())
    thread = threading.Thread(target=_viewer_thread, args=(hub, controls, config_path, seed), daemon=True, name="viewer")
    app.state.hub = hub
    app.state.controls = controls
    app.state.thread = thread
    thread.start()
    yield


app = FastAPI(title="Quadruped Viewer API", lifespan=lifespan)
_allowed_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials="*" not in _allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "quadruped-viewer", "status": "ok"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict[str, Any]:
    selected_model_id = app.state.controls.snapshot().selected_model_id if hasattr(app.state, "controls") else None
    return _models_message(selected_model_id)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    hub: BroadcastHub = websocket.app.state.hub
    controls: ViewerControls = websocket.app.state.controls
    await hub.connect(websocket)
    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                continue
            message_type = message.get("type")
            if message_type == "select_model":
                model_id = message.get("model_id")
                controls.set_model(str(model_id) if model_id else None)
                hub.publish(_models_message(controls.snapshot().selected_model_id))
            elif message_type == "set_goal":
                goal = message.get("goal")
                if isinstance(goal, list) and len(goal) >= 2:
                    snapshot = controls.snapshot()
                    active_spec = load_runtime_spec(_load_service_config()[0])
                    height = float(goal[2]) if len(goal) >= 3 else float(active_spec.goals.height_m)
                    controls.set_goal((float(goal[0]), float(goal[1]), height))
                    hub.publish(
                        {
                            "type": "goal",
                            "goal": [float(goal[0]), float(goal[1]), height],
                            "version": snapshot.version + 1,
                        }
                    )
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception:
        hub.disconnect(websocket)
