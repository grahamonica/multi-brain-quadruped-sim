"""Model run artifact paths and discovery helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from brains.config import RuntimeSpec, config_json_matches_checkpoint, runtime_spec_from_dict


MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class ModelRunPaths:
    id: str
    model_type: str
    log_id: str
    run_dir: Path
    latest_path: Path
    best_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ModelArtifact:
    id: str
    model_type: str
    log_id: str
    checkpoint_path: Path
    latest_path: Path | None = None
    best_path: Path | None = None
    manifest_path: Path | None = None
    generation: int = 0
    best_reward: float | None = None
    mean_reward: float | None = None
    config_name: str | None = None
    updated_at: str | None = None

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_type": self.model_type,
            "log_id": self.log_id,
            "checkpoint_path": str(self.checkpoint_path),
            "latest_path": str(self.latest_path) if self.latest_path is not None else None,
            "best_path": str(self.best_path) if self.best_path is not None else None,
            "generation": self.generation,
            "best_reward": self.best_reward,
            "mean_reward": self.mean_reward,
            "config_name": self.config_name,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class CheckpointCompatibility:
    path: Path
    compatible: bool
    reason: str | None = None


def sanitize_model_part(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "-" for ch in value).strip("-")
    return sanitized or "model"


def new_log_id() -> str:
    now = datetime.now(tz=timezone.utc)
    suffix = time.time_ns() % 1_000_000
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}{suffix:06d}"


def model_run_id(model_type: str, log_id: str) -> str:
    return f"{sanitize_model_part(model_type)}_{sanitize_model_part(log_id)}"


def create_model_run_paths(root: str | Path, model_type: str, log_id: str | None = None) -> ModelRunPaths:
    checkpoint_root = Path(root)
    safe_model_type = sanitize_model_part(model_type)
    resolved_log_id = sanitize_model_part(log_id or new_log_id())
    run_id = model_run_id(safe_model_type, resolved_log_id)
    run_dir = checkpoint_root / run_id
    if log_id is None:
        while run_dir.exists():
            resolved_log_id = new_log_id()
            run_id = model_run_id(safe_model_type, resolved_log_id)
            run_dir = checkpoint_root / run_id
    return ModelRunPaths(
        id=run_id,
        model_type=safe_model_type,
        log_id=resolved_log_id,
        run_dir=run_dir,
        latest_path=run_dir / "latest.npz",
        best_path=run_dir / "best.npz",
        manifest_path=run_dir / MANIFEST_FILENAME,
    )


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        def scalar(name: str, default: Any = None) -> Any:
            if name not in checkpoint.files:
                return default
            value = checkpoint[name]
            return value.item() if value.shape == () else value.tolist()

        return {
            "generation": int(scalar("generation", 0)),
            "best_reward": float(scalar("best_reward")) if scalar("best_reward") is not None else None,
            "mean_reward": float(scalar("mean_reward")) if scalar("mean_reward") is not None else None,
            "config_name": str(scalar("config_name")) if scalar("config_name") is not None else None,
            "model_type": str(scalar("model_type", "")),
            "model_id": str(scalar("model_id", "")),
            "log_id": str(scalar("log_id", "")),
        }


def _safe_checkpoint_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return checkpoint_summary(path)
    except Exception:
        return {}


def _mtime_iso(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def write_model_manifest(
    paths: ModelRunPaths,
    spec: RuntimeSpec,
    checkpoint_path: str | Path,
    *,
    generation: int,
    best_reward: float,
    mean_reward: float,
    log_run_id: str | None = None,
    log_run_dir: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).isoformat()
    payload = {
        "id": paths.id,
        "model_type": paths.model_type,
        "log_id": paths.log_id,
        "created_or_updated_at": now,
        "weights": {
            "latest": str(paths.latest_path),
            "best": str(paths.best_path),
            "selected": str(checkpoint),
        },
        "metrics": {
            "generation": int(generation),
            "best_reward": float(best_reward),
            "mean_reward": float(mean_reward),
        },
        "runtime": {
            "config_name": spec.name,
            "model": asdict(spec.model),
            "simulator_backend": spec.simulator.backend,
        },
        "logs": {
            "run_id": log_run_id,
            "run_dir": str(log_run_dir) if log_run_dir is not None else None,
        },
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return paths.manifest_path


def _artifact_from_manifest(path: Path) -> ModelArtifact | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    weights = manifest.get("weights", {})
    latest = Path(weights["latest"]) if weights.get("latest") else None
    best = Path(weights["best"]) if weights.get("best") else None
    checkpoint = Path(weights.get("selected") or latest or best)
    summary = _safe_checkpoint_summary(checkpoint)
    metrics = manifest.get("metrics", {})
    return ModelArtifact(
        id=str(manifest.get("id") or summary.get("model_id") or path.parent.name),
        model_type=str(manifest.get("model_type") or summary.get("model_type") or "unknown"),
        log_id=str(manifest.get("log_id") or summary.get("log_id") or path.parent.name),
        checkpoint_path=checkpoint,
        latest_path=latest,
        best_path=best,
        manifest_path=path,
        generation=int(summary.get("generation") or metrics.get("generation") or 0),
        best_reward=summary.get("best_reward", metrics.get("best_reward")),
        mean_reward=summary.get("mean_reward", metrics.get("mean_reward")),
        config_name=summary.get("config_name") or manifest.get("runtime", {}).get("config_name"),
        updated_at=_mtime_iso(checkpoint) or manifest.get("created_or_updated_at"),
    )


def _artifact_from_run_dir(path: Path) -> ModelArtifact | None:
    manifest = path / MANIFEST_FILENAME
    if manifest.exists():
        return _artifact_from_manifest(manifest)
    latest = path / "latest.npz"
    best = path / "best.npz"
    checkpoint = latest if latest.exists() else best if best.exists() else None
    if checkpoint is None:
        return None
    summary = _safe_checkpoint_summary(checkpoint)
    model_type = summary.get("model_type") or path.name.rsplit("_", 1)[0] or "unknown"
    log_id = summary.get("log_id") or path.name.rsplit("_", 1)[-1]
    return ModelArtifact(
        id=summary.get("model_id") or path.name,
        model_type=model_type,
        log_id=log_id,
        checkpoint_path=checkpoint,
        latest_path=latest if latest.exists() else None,
        best_path=best if best.exists() else None,
        manifest_path=None,
        generation=int(summary.get("generation") or 0),
        best_reward=summary.get("best_reward"),
        mean_reward=summary.get("mean_reward"),
        config_name=summary.get("config_name"),
        updated_at=_mtime_iso(checkpoint),
    )


def _legacy_artifact(root: Path, name: str) -> ModelArtifact | None:
    path = root / name
    if not path.exists():
        return None
    summary = _safe_checkpoint_summary(path)
    stem = path.stem
    model_type = summary.get("model_type") or "legacy"
    log_id = summary.get("log_id") or stem
    return ModelArtifact(
        id=summary.get("model_id") or f"legacy_{stem}",
        model_type=model_type,
        log_id=log_id,
        checkpoint_path=path,
        latest_path=path if stem == "latest" else None,
        best_path=path if stem == "best" else None,
        generation=int(summary.get("generation") or 0),
        best_reward=summary.get("best_reward"),
        mean_reward=summary.get("mean_reward"),
        config_name=summary.get("config_name"),
        updated_at=_mtime_iso(path),
    )


def discover_model_artifacts(root: str | Path = "checkpoints") -> tuple[ModelArtifact, ...]:
    checkpoint_root = Path(root)
    artifacts: list[ModelArtifact] = []
    if checkpoint_root.exists():
        for child in sorted(checkpoint_root.iterdir()):
            if not child.is_dir():
                continue
            artifact = _artifact_from_run_dir(child)
            if artifact is not None and artifact.checkpoint_path.exists():
                artifacts.append(artifact)
        for legacy_name in ("latest.npz", "best.npz"):
            artifact = _legacy_artifact(checkpoint_root, legacy_name)
            if artifact is not None:
                artifacts.append(artifact)

    deduped: dict[str, ModelArtifact] = {}
    for artifact in artifacts:
        deduped[artifact.id] = artifact
    return tuple(
        sorted(
            deduped.values(),
            key=lambda item: item.checkpoint_path.stat().st_mtime if item.checkpoint_path.exists() else 0.0,
            reverse=True,
        )
    )


def find_model_artifact(model_id: str, root: str | Path = "checkpoints") -> ModelArtifact | None:
    for artifact in discover_model_artifacts(root):
        if artifact.id == model_id:
            return artifact
    return None


def viewer_checkpoint_candidates(root: str | Path = "checkpoints") -> tuple[Path, ...]:
    checkpoint_root = Path(root)
    names = ("latest.npz", "best.npz")
    return tuple((checkpoint_root / name) for name in names if (checkpoint_root / name).exists())


def checkpoint_matches_spec(path: str | Path, spec: RuntimeSpec) -> CheckpointCompatibility:
    checkpoint_path = Path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "config_json" in checkpoint.files:
                checkpoint_config_json = str(checkpoint["config_json"].item())
                if not config_json_matches_checkpoint(spec, checkpoint_config_json):
                    return CheckpointCompatibility(
                        path=checkpoint_path,
                        compatible=False,
                        reason="checkpoint runtime spec does not match the active config aside from backend selection",
                    )
            else:
                return CheckpointCompatibility(
                    path=checkpoint_path,
                    compatible=False,
                    reason="checkpoint is missing runtime metadata",
                )
    except Exception as exc:
        return CheckpointCompatibility(
            path=checkpoint_path,
            compatible=False,
            reason=f"failed to inspect checkpoint: {exc}",
        )
    return CheckpointCompatibility(path=checkpoint_path, compatible=True)


def runtime_spec_from_checkpoint(path: str | Path) -> RuntimeSpec | None:
    checkpoint_path = Path(path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            if "config_json" not in checkpoint.files:
                return None
            checkpoint_config_json = str(checkpoint["config_json"].item())
    except Exception:
        return None
    try:
        checkpoint_data = json.loads(checkpoint_config_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(checkpoint_data, dict):
        return None
    return runtime_spec_from_dict(checkpoint_data)


def resolve_viewer_checkpoint(root: str | Path = "checkpoints", spec: RuntimeSpec | None = None) -> Path | None:
    for path in viewer_checkpoint_candidates(root):
        if spec is None or checkpoint_matches_spec(path, spec).compatible:
            return path
    return None
