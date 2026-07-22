"""Reproducible experiment bookkeeping and statistical summaries."""

from .bootstrap import (
    bootstrap_experiment,
    cluster_bootstrap_interval,
    select_scene_cluster_bootstrap,
    validate_truthfulness_metadata,
)
from .experiment_store import ExperimentStore, ManifestCountMismatch
from .failure_taxonomy import classify_status, is_retryable, is_terminal
from .metrics import aggregate_metrics, export_metrics, wilson_interval

__all__ = [
    "ExperimentStore",
    "ManifestCountMismatch",
    "aggregate_metrics",
    "bootstrap_experiment",
    "classify_status",
    "cluster_bootstrap_interval",
    "export_metrics",
    "is_retryable",
    "is_terminal",
    "select_scene_cluster_bootstrap",
    "validate_truthfulness_metadata",
    "wilson_interval",
]
