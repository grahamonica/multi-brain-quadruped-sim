"""Runtime helpers shared by the headless trainer, viewer, and tests."""

from .logging import MetricsSink, configure_logging, create_run_artifacts, write_json
from .model_store import (
    checkpoint_matches_spec,
    create_model_run_paths,
    discover_model_artifacts,
    find_model_artifact,
    resolve_viewer_checkpoint,
    runtime_spec_from_checkpoint,
    viewer_checkpoint_candidates,
    write_model_manifest,
)
from .quality_gates import (
    QualityGateRunner,
    collect_regression_metrics,
    compare_regression_to_baseline,
)

__all__ = [
    "MetricsSink",
    "QualityGateRunner",
    "collect_regression_metrics",
    "compare_regression_to_baseline",
    "checkpoint_matches_spec",
    "configure_logging",
    "create_model_run_paths",
    "create_run_artifacts",
    "discover_model_artifacts",
    "find_model_artifact",
    "resolve_viewer_checkpoint",
    "runtime_spec_from_checkpoint",
    "viewer_checkpoint_candidates",
    "write_json",
    "write_model_manifest",
]
