"""Resumable entry point for the two-route reranking experiment matrix.

Every stage is transactional at the marker level: a stage receives a success
marker only after all declared outputs exist and its validation callback
passes.  The run-level ``_SUCCESS.json`` is deliberately stricter and is
written only after all mandatory stages complete.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch

from reranking.audit_runs import audit_canonical_runs, write_audit_bundle
from reranking.build_candidate_table import (
    BuildResult,
    build_crog_candidate_tables,
    build_modular_candidate_tables,
)
from reranking.data_contracts import (
    CROG_CANONICAL_CANDIDATES,
    CROG_CANONICAL_RUN,
    KNOWN_HASHES,
    MODULAR_CANONICAL_RUN,
    REPO_ROOT,
    streaming_sha256,
)
from reranking.completion_audit import (
    CUMULATIVE_STAGE_OUTPUT_POLICIES,
    IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES,
    DatasetExpectation,
    audit_background_activity,
    audit_run_completion,
)
from reranking.evaluate import evaluate_rankings
from reranking.extract_features import (
    audit_feature_table,
    select_crog_native_feature_columns,
    select_model_feature_columns,
    write_feature_audit_bundle,
)
from reranking.publish_modular_development import (
    PublishError,
    verify_modular_development_publication,
)


STAGES = (
    "audit-only",
    "build-features",
    "train",
    "validate",
    "lock-primary",
    "test-primary",
    "test-post-lock",
    "statistics",
    "visualize",
    "report",
)
SPECIAL_TRAIN_STAGES = ("train-worker", "train-finalize")

REQUIRED_RUN_DIRECTORIES = (
    "manifests",
    "data",
    "features",
    "configs",
    "checkpoints",
    "predictions",
    "metrics",
    "statistics",
    "figures",
    "galleries",
    "logs",
    "reports",
)

CROG_V3_ROOT = CROG_CANONICAL_RUN
CROG_V2_ROOT = (
    REPO_ROOT
    / "crog_reproduction"
    / "CROG"
    / "failure_analysis"
    / "reranking_outputs"
    / "v2_20260727T174412+0100"
)
CROG_TEST_LABELS = (
    CROG_V2_ROOT / "formal_test_primary_v2" / "labels" / "corrected" / "labels.jsonl"
)
CROG_TRAIN_FEATURES = CROG_V2_ROOT / "base_train" / "features.jsonl"
CROG_TRAIN_LABELS = CROG_V2_ROOT / "base_train" / "labels.jsonl"
CROG_VAL_FEATURES = CROG_V2_ROOT / "base_val" / "features.jsonl"
CROG_VAL_LABELS = CROG_V2_ROOT / "base_val" / "labels.jsonl"

RUN_CONFIG_SCHEMA_VERSION = 2

MODULAR_EVALUATED = (
    MODULAR_CANONICAL_RUN / "evaluation" / "hierfilm_gqcnn_per_candidate.parquet"
)
MODULAR_INPUT_MANIFEST = MODULAR_CANONICAL_RUN / "input_manifest.csv"
MODULAR_DEVELOPMENT_RUN = (
    REPO_ROOT
    / "HiFi_reproduction"
    / "runs"
    / "modular_reranking_repeatedfilm_v1_20260729_203147"
)
MODULAR_DEVELOPMENT_QUERY_MANIFESTS = (
    MODULAR_DEVELOPMENT_RUN / "compact_inputs" / "train" / "manifest.jsonl",
    MODULAR_DEVELOPMENT_RUN / "compact_inputs" / "val" / "manifest.jsonl",
)
MODULAR_DEVELOPMENT_CANDIDATE_COUNT_EVIDENCE = (
    MODULAR_DEVELOPMENT_RUN / "features" / "train" / "per_sample.parquet",
    MODULAR_DEVELOPMENT_RUN / "features" / "val" / "per_sample.parquet",
)
MODULAR_RICH_TEST_FEATURES = (
    MODULAR_DEVELOPMENT_RUN / "features" / "test" / "per_candidate.parquet"
)


class StageError(RuntimeError):
    """Raised when a mandatory stage cannot truthfully complete."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(command: Sequence[str]) -> str:
    process = subprocess.run(
        ["git", *command],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip()


def _environment() -> dict[str, Any]:
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "captured_at": _now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": (
            importlib.metadata.version("xgboost")
            if importlib.util.find_spec("xgboost") is not None
            else None
        ),
        "opencv": cv2.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "git_head": _git(("rev-parse", "HEAD")),
        "git_branch": _git(("branch", "--show-current")),
        "git_status_porcelain": _git(("status", "--porcelain=v1")),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
    }


def _safe_output_path(path: Path, *, resume: bool) -> Path:
    output = path.resolve(strict=False)
    runs_root = (REPO_ROOT / "runs").resolve(strict=False)
    if output == runs_root or not output.is_relative_to(runs_root):
        raise StageError(f"output must be below {REPO_ROOT / 'runs'}: {output}")
    protected = (MODULAR_CANONICAL_RUN.resolve(), CROG_CANONICAL_RUN.resolve())
    if any(output == source or output.is_relative_to(source) for source in protected):
        raise StageError(f"output overlaps canonical source: {output}")
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume: {output}")
    return output


def _semantic_run_config(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Return the behavior-defining configuration, excluding CLI controls.

    ``stage``, ``resume`` and the force-rerun selector decide *which* work is
    executed, not what an experiment means.  Everything that can change data,
    training, inference, or statistical interpretation is frozen here.
    """

    return {
        "project_root": str(REPO_ROOT.resolve()),
        "output": str(output.resolve()),
        "routes": list(map(str, args.routes)),
        "pools": list(map(str, args.pools)),
        "folds": int(args.folds),
        "seeds": list(map(int, args.seeds)),
        "device": str(args.device),
        "neural_query_batch_size": int(getattr(args, "neural_query_batch_size", 128)),
        "neural_batching_policy": str(
            getattr(args, "neural_batching_policy", "length_bucketed_v1")
        ),
        "train_torch_thread_count": int(getattr(args, "train_torch_thread_count", 5)),
        "train_torch_interop_thread_count": int(
            getattr(args, "train_torch_interop_thread_count", 1)
        ),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "hash_audit_enabled": not bool(args.no_hash_audit),
        "strict_no_success_before_all_stages": True,
    }


def _json_identity_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _frozen_run_config(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    semantic = _semantic_run_config(output, args)
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "created_at": _now(),
        **semantic,
        "semantic_config": semantic,
        "semantic_config_sha256": _json_identity_sha256(semantic),
        "execution_controls_excluded_from_identity": [
            "stage",
            "resume",
            "force_rerun_specific_experiment",
            "amend_prelock_device",
            "amend_prelock_training",
            "train_worker_count",
            "train_worker_index",
            "train_worker_owner",
            "train_claim_lease_seconds",
        ],
    }


def _read_run_config_identity(output: Path) -> str:
    path = output / "configs" / "run_config.json"
    if not path.is_file() or path.is_symlink():
        raise StageError(f"frozen run configuration is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError(f"frozen run configuration is unreadable: {path}") from error
    semantic = payload.get("semantic_config")
    expected = payload.get("semantic_config_sha256")
    if (
        payload.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION
        or not isinstance(semantic, Mapping)
        or not isinstance(expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
        or _json_identity_sha256(semantic) != expected
    ):
        raise StageError(f"frozen run configuration identity is invalid: {path}")
    mirrored_mismatches = [
        key for key, value in semantic.items() if payload.get(key) != value
    ]
    unsupported_matrix_controls = sorted(
        key for key in payload if str(key).startswith("matrix_")
    )
    if mirrored_mismatches or unsupported_matrix_controls:
        raise StageError(
            "frozen run configuration contains unfrozen semantic overrides: "
            + ", ".join(sorted({*mirrored_mismatches, *unsupported_matrix_controls}))
        )
    return expected


@contextmanager
def _run_config_amendment_lock(output: Path) -> Iterator[None]:
    path = output / "configs" / ".prelock_amendment.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _freeze_or_validate_run_config_unlocked(
    output: Path, args: argparse.Namespace
) -> dict[str, Any]:
    path = output / "configs" / "run_config.json"
    requested = _frozen_run_config(output, args)
    if not path.exists():
        _atomic_json(path, requested)
        return requested
    if path.is_symlink() or not path.is_file():
        raise StageError(f"run configuration must be a regular file: {path}")
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError(f"existing run configuration is unreadable: {path}") from error

    existing_semantic = existing.get("semantic_config")
    existing_identity = existing.get("semantic_config_sha256")
    if isinstance(existing_semantic, Mapping) and isinstance(existing_identity, str):
        computed = _json_identity_sha256(existing_semantic)
        if existing_identity != computed:
            raise StageError("existing frozen run configuration hash mismatch")
        mirrored_mismatches = [
            key
            for key, value in existing_semantic.items()
            if existing.get(key) != value
        ]
        unsupported_matrix_controls = sorted(
            key for key in existing if str(key).startswith("matrix_")
        )
        if mirrored_mismatches or unsupported_matrix_controls:
            raise StageError(
                "existing run configuration contains unfrozen semantic overrides: "
                + ", ".join(
                    sorted({*mirrored_mismatches, *unsupported_matrix_controls})
                )
            )
        if dict(existing_semantic) != requested["semantic_config"]:
            changed = sorted(
                key
                for key in set(existing_semantic) | set(requested["semantic_config"])
                if existing_semantic.get(key) != requested["semantic_config"].get(key)
            )
            if changed == ["device"] and bool(
                getattr(args, "amend_prelock_device", False)
            ):
                return _amend_prelock_device_config(
                    output,
                    path,
                    existing,
                    requested,
                )
            acceleration_fields = {
                "neural_query_batch_size",
                "neural_batching_policy",
                "train_torch_thread_count",
                "train_torch_interop_thread_count",
            }
            if set(changed).issubset(acceleration_fields) and bool(
                getattr(args, "amend_prelock_training", False)
            ):
                return _amend_prelock_batch_config(
                    output,
                    path,
                    existing,
                    requested,
                )
            raise StageError(
                "resume semantic configuration mismatch; refusing to change: "
                + ", ".join(changed)
            )
        return existing

    # One-time fail-closed upgrade for runs created before semantic identities
    # were introduced.  Every field that the legacy schema actually recorded
    # must already match; the previously unrecorded hash-audit policy is frozen
    # from this invocation.  Existing receipts intentionally become stale and
    # must be regenerated with the new configuration identity.
    legacy_fields = (
        "project_root",
        "output",
        "routes",
        "pools",
        "folds",
        "seeds",
        "device",
        "bootstrap_iterations",
        "strict_no_success_before_all_stages",
    )
    mismatched = [
        key
        for key in legacy_fields
        if key in existing and existing[key] != requested["semantic_config"][key]
    ]
    if mismatched:
        raise StageError(
            "resume legacy semantic configuration mismatch; refusing to change: "
            + ", ".join(sorted(mismatched))
        )
    requested["created_at"] = existing.get("created_at", requested["created_at"])
    requested["legacy_config_upgraded_at"] = _now()
    _atomic_json(path, requested)
    return requested


def _freeze_or_validate_run_config(
    output: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Freeze or validate config, serializing every pre-lock amendment."""

    if bool(getattr(args, "amend_prelock_device", False)) or bool(
        getattr(args, "amend_prelock_training", False)
    ):
        with _run_config_amendment_lock(output):
            # Read the current file only after ownership is acquired.  A peer
            # amendment may have completed while this invocation was waiting.
            return _freeze_or_validate_run_config_unlocked(output, args)
    return _freeze_or_validate_run_config_unlocked(output, args)


def _amend_prelock_device_config(
    output: Path,
    config_path: Path,
    existing: Mapping[str, Any],
    requested: dict[str, Any],
) -> dict[str, Any]:
    """Apply one evidence-backed device amendment before primary locking.

    This is deliberately narrower than a general configuration override.  It
    preserves the original frozen JSON, requires a complete four-backend local
    benchmark, and makes all old stage receipts stale through the new semantic
    identity.  Experiment-level neural identities independently include the
    effective device and therefore cannot resume an old-device checkpoint.
    """

    if (output / "manifests" / "PRIMARY_METHOD_LOCK.json").exists():
        raise StageError("device amendment is forbidden after primary locking")
    old_semantic = existing.get("semantic_config")
    new_semantic = requested.get("semantic_config")
    if not isinstance(old_semantic, Mapping) or not isinstance(new_semantic, Mapping):
        raise StageError("device amendment requires frozen semantic configurations")
    changed = sorted(
        key
        for key in set(old_semantic) | set(new_semantic)
        if old_semantic.get(key) != new_semantic.get(key)
    )
    if changed != ["device"]:
        raise StageError("device amendment may change only the device field")

    evidence_path = output / "audit" / "neural_device_benchmark.json"
    if not evidence_path.is_file() or evidence_path.is_symlink():
        raise StageError("device amendment requires audit/neural_device_benchmark.json")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError("device benchmark evidence is unreadable") from error
    rows = evidence.get("benchmarks")
    requested_device = str(new_semantic["device"])
    required_backends = {"mlp", "deepsets", "gnn", "set_transformer"}
    observed_backends = (
        {str(row.get("backend")) for row in rows if isinstance(row, Mapping)}
        if isinstance(rows, list)
        else set()
    )
    valid_rows = (
        isinstance(rows, list)
        and evidence.get("schema_version") == 1
        and evidence.get("status") == "COMPLETE"
        and old_semantic.get("device") == "mps"
        and requested_device == "cpu"
        and evidence.get("recommended_device") == requested_device
        and observed_backends == required_backends
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("cpu_seconds"), (int, float))
            and isinstance(row.get("mps_seconds"), (int, float))
            and float(row["cpu_seconds"]) > 0.0
            and float(row["mps_seconds"]) > float(row["cpu_seconds"])
            for row in rows
        )
    )
    if not valid_rows:
        raise StageError(
            "device benchmark does not justify the requested four-backend amendment"
        )

    old_identity = str(existing["semantic_config_sha256"])
    archive_path = (
        output / "configs" / f"run_config.prelock_device_{old_identity[:12]}.json"
    )
    if archive_path.exists():
        if archive_path.is_symlink() or streaming_sha256(
            archive_path
        ) != streaming_sha256(config_path):
            raise StageError("pre-lock configuration archive already differs")
    else:
        temporary = archive_path.with_name(
            f".{archive_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        shutil.copyfile(config_path, temporary)
        os.replace(temporary, archive_path)

    requested["created_at"] = existing.get("created_at", requested["created_at"])
    amendment_path = output / "audit" / "prelock_device_amendment.json"
    amendment = {
        "schema_version": 1,
        "status": "APPLIED_BEFORE_PRIMARY_LOCK",
        "amended_at": _now(),
        "changed_fields": ["device"],
        "old_device": old_semantic["device"],
        "new_device": new_semantic["device"],
        "old_semantic_config_sha256": old_identity,
        "new_semantic_config_sha256": requested["semantic_config_sha256"],
        "old_config_snapshot": str(archive_path.resolve()),
        "old_config_snapshot_sha256": streaming_sha256(archive_path),
        "benchmark_evidence": str(evidence_path.resolve()),
        "benchmark_evidence_sha256": streaming_sha256(evidence_path),
        "prior_stage_receipts_invalidated": True,
        "primary_lock_present": False,
    }
    _atomic_json(amendment_path, amendment)
    requested["prelock_device_amendment"] = amendment
    _atomic_json(config_path, requested)
    return requested


def _amend_prelock_batch_config(
    output: Path,
    config_path: Path,
    existing: Mapping[str, Any],
    requested: dict[str, Any],
) -> dict[str, Any]:
    """Freeze benchmark-backed neural acceleration before primary locking."""

    if (output / "manifests" / "PRIMARY_METHOD_LOCK.json").exists():
        raise StageError("training amendment is forbidden after primary locking")
    old_semantic = existing.get("semantic_config")
    new_semantic = requested.get("semantic_config")
    if not isinstance(old_semantic, Mapping) or not isinstance(new_semantic, Mapping):
        raise StageError("training amendment requires frozen semantic configurations")
    changed = sorted(
        key
        for key in set(old_semantic) | set(new_semantic)
        if old_semantic.get(key) != new_semantic.get(key)
    )
    allowed = {
        "neural_query_batch_size",
        "neural_batching_policy",
        "train_torch_thread_count",
        "train_torch_interop_thread_count",
    }
    if not changed or not set(changed).issubset(allowed):
        raise StageError(
            "training amendment may change only batching and Torch thread policy"
        )
    if (output / "configs" / "train_worker_protocol.json").exists():
        raise StageError("training amendment must precede worker protocol publication")
    if list((output / "logs" / "train_workers" / "claims").glob("*.json")):
        raise StageError("training amendment is forbidden while worker claims exist")

    evidence_paths: list[Path] = []
    if "neural_query_batch_size" in changed:
        batch_evidence_path = output / "audit" / "neural_batch_benchmark.json"
        if not batch_evidence_path.is_file() or batch_evidence_path.is_symlink():
            raise StageError(
                "training amendment requires audit/neural_batch_benchmark.json"
            )
        try:
            batch_evidence = json.loads(batch_evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StageError("neural batch benchmark evidence is unreadable") from error
        requested_batch = int(new_semantic["neural_query_batch_size"])
        if (
            batch_evidence.get("schema_version") != 1
            or batch_evidence.get("status") != "COMPLETE"
            or batch_evidence.get("recommended_query_batch_size") != requested_batch
            or requested_batch <= 0
            or batch_evidence.get("test_labels_opened") is not False
            or batch_evidence.get("primary_lock_present_during_benchmark") is not False
        ):
            raise StageError("neural batch benchmark does not justify the amendment")
        evidence_paths.append(batch_evidence_path)

    acceleration_fields = set(changed) - {"neural_query_batch_size"}
    if acceleration_fields:
        acceleration_evidence_path = (
            output / "audit" / "full_list_neural_acceleration_benchmark.json"
        )
        if (
            not acceleration_evidence_path.is_file()
            or acceleration_evidence_path.is_symlink()
        ):
            raise StageError(
                "acceleration amendment requires the full-list neural benchmark"
            )
        try:
            acceleration_evidence = json.loads(
                acceleration_evidence_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise StageError("neural acceleration benchmark is unreadable") from error
        recommendation = acceleration_evidence.get("recommendation")
        if (
            acceleration_evidence.get("kind")
            != "full_list_neural_acceleration_benchmark"
            or acceleration_evidence.get("status") != "COMPLETE"
            or acceleration_evidence.get("formal_training_manifests_written")
            is not False
            or not isinstance(recommendation, Mapping)
            or recommendation.get("cpu_intraop_threads")
            != new_semantic.get("train_torch_thread_count")
            or recommendation.get("cpu_interop_threads")
            != new_semantic.get("train_torch_interop_thread_count")
            or new_semantic.get("neural_batching_policy") != "length_bucketed_v1"
            or "length-aware" not in str(recommendation.get("batching", ""))
        ):
            raise StageError(
                "full-list benchmark does not justify the requested acceleration policy"
            )
        evidence_paths.append(acceleration_evidence_path)

    old_identity = str(existing["semantic_config_sha256"])
    neural_backends = {"mlp", "deepsets", "gnn", "set_transformer"}
    experiment_root = output / "manifests" / "experiments"
    superseded_root = (
        output / "manifests" / f"superseded_pre_acceleration_{old_identity[:12]}"
    )
    retained_inventory: list[dict[str, Any]] = []
    neural_archive_plan: list[dict[str, Any]] = []

    # Preflight every manifest and every neural artifact before mutating the
    # frozen configuration or stage receipts.  The old neural artifacts would
    # otherwise remain at paths that the amended experiments overwrite, making
    # the superseded manifests impossible to audit after resume.
    for source in sorted(experiment_root.glob("*.json")):
        try:
            manifest = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StageError(f"experiment manifest is unreadable: {source}") from error
        if not isinstance(manifest, Mapping):
            raise StageError(f"experiment manifest is not a JSON object: {source}")
        spec = manifest.get("spec")
        backend = spec.get("backend") if isinstance(spec, Mapping) else None
        row: dict[str, Any] = {
            "experiment_id": manifest.get("experiment_id", source.stem),
            "backend": backend,
            "status": manifest.get("status"),
            "original_path": str(source.resolve()),
            "sha256": streaming_sha256(source),
        }
        if backend not in neural_backends:
            retained_inventory.append(row)
            continue

        status = manifest.get("status")
        if status not in {"COMPLETE", "FAILED"}:
            raise StageError(
                f"neural manifest has an unsupported status {status!r}: {source}"
            )
        artifacts = manifest.get("artifacts", [])
        artifact_hashes = manifest.get("artifact_sha256")
        if not isinstance(artifacts, list) or (
            (status == "COMPLETE" or artifacts)
            and not isinstance(artifact_hashes, Mapping)
        ):
            raise StageError(f"neural manifest lacks its artifact inventory: {source}")
        if not isinstance(artifact_hashes, Mapping):
            artifact_hashes = {}
        archived_artifacts: list[dict[str, Any]] = []
        for raw_path in artifacts:
            artifact = Path(str(raw_path)).expanduser().resolve()
            if (
                not artifact.is_relative_to(output.resolve())
                or not artifact.is_file()
                or artifact.is_symlink()
            ):
                raise StageError(
                    f"neural manifest artifact is missing or unsafe: {artifact}"
                )
            declared_sha256 = artifact_hashes.get(str(artifact))
            observed_sha256 = streaming_sha256(artifact)
            if declared_sha256 != observed_sha256:
                raise StageError(f"neural manifest artifact hash mismatch: {artifact}")
            relative = artifact.relative_to(output.resolve())
            archive_artifact = superseded_root / "artifacts" / source.stem / relative
            archived_artifacts.append(
                {
                    "original_path": str(artifact),
                    "archive_path": str(archive_artifact.resolve()),
                    "sha256": observed_sha256,
                    "size_bytes": artifact.stat().st_size,
                }
            )
        neural_archive_plan.append(
            {
                "source": source,
                "row": row,
                "archived_artifacts": archived_artifacts,
            }
        )

    archive_path = (
        output / "configs" / f"run_config.prelock_training_{old_identity[:12]}.json"
    )
    if archive_path.exists():
        if archive_path.is_symlink() or streaming_sha256(
            archive_path
        ) != streaming_sha256(config_path):
            raise StageError("pre-lock training configuration archive already differs")
    else:
        temporary = archive_path.with_name(
            f".{archive_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        shutil.copyfile(config_path, temporary)
        os.replace(temporary, archive_path)

    # Copy, rather than move, the superseded artifacts.  This preserves the
    # original completed run until an amended experiment atomically replaces
    # each formal path, while keeping an immutable hash-checked audit copy.
    for planned in neural_archive_plan:
        for artifact_row in planned["archived_artifacts"]:
            source_artifact = Path(artifact_row["original_path"])
            archive_artifact = Path(artifact_row["archive_path"])
            archive_artifact.parent.mkdir(parents=True, exist_ok=True)
            if archive_artifact.exists():
                if (
                    archive_artifact.is_symlink()
                    or not archive_artifact.is_file()
                    or streaming_sha256(archive_artifact) != artifact_row["sha256"]
                ):
                    raise StageError(
                        "superseded neural artifact archive already differs: "
                        f"{archive_artifact}"
                    )
                continue
            temporary = archive_artifact.with_name(
                f".{archive_artifact.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            shutil.copyfile(source_artifact, temporary)
            if streaming_sha256(temporary) != artifact_row["sha256"]:
                temporary.unlink(missing_ok=True)
                raise StageError(
                    f"superseded neural artifact copy hash mismatch: {source_artifact}"
                )
            os.replace(temporary, archive_artifact)

    requested["created_at"] = existing.get("created_at", requested["created_at"])
    stage_archive_root = (
        output / "logs" / f"stages.superseded_pre_acceleration_{old_identity[:12]}"
    )
    archived_stage_files: list[dict[str, Any]] = []
    stage_root = output / "logs" / "stages"
    for source in sorted(stage_root.rglob("*")) if stage_root.is_dir() else []:
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(stage_root)
        destination = stage_archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sha256 = streaming_sha256(source)
        if destination.exists():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or streaming_sha256(destination) != source_sha256
            ):
                raise StageError(
                    f"superseded stage archive already differs: {destination}"
                )
        else:
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            shutil.copyfile(source, temporary)
            if streaming_sha256(temporary) != source_sha256:
                temporary.unlink(missing_ok=True)
                raise StageError(
                    f"superseded stage receipt copy hash mismatch: {source}"
                )
            os.replace(temporary, destination)
        archived_stage_files.append(
            {
                "original_path": str(source.resolve()),
                "archive_path": str(destination.resolve()),
                "sha256": source_sha256,
            }
        )

    superseded_inventory: list[dict[str, Any]] = []
    for planned in neural_archive_plan:
        source = planned["source"]
        row = planned["row"]
        destination = superseded_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or streaming_sha256(destination) != row["sha256"]
            ):
                raise StageError(
                    f"superseded neural manifest archive already differs: {destination}"
                )
        else:
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            shutil.copyfile(source, temporary)
            if streaming_sha256(temporary) != row["sha256"]:
                temporary.unlink(missing_ok=True)
                raise StageError(
                    f"superseded neural manifest copy hash mismatch: {source}"
                )
            os.replace(temporary, destination)
        superseded_inventory.append(
            {
                **row,
                "archive_path": str(destination.resolve()),
                "archived_artifacts": planned["archived_artifacts"],
            }
        )

    amendment_path = output / "audit" / "prelock_acceleration_amendment.json"
    amendment = {
        "schema_version": 1,
        "status": "APPLIED_BEFORE_PRIMARY_LOCK",
        "amended_at": _now(),
        "changed_fields": changed,
        "old_values": {field: old_semantic.get(field) for field in changed},
        "new_values": {field: new_semantic.get(field) for field in changed},
        "old_semantic_config_sha256": old_identity,
        "new_semantic_config_sha256": requested["semantic_config_sha256"],
        "old_config_snapshot": str(archive_path.resolve()),
        "old_config_snapshot_sha256": streaming_sha256(archive_path),
        "benchmark_evidence": [
            {"path": str(path.resolve()), "sha256": streaming_sha256(path)}
            for path in evidence_paths
        ],
        "prior_stage_receipts_invalidated": True,
        "archived_stage_files": archived_stage_files,
        "retained_non_neural_manifest_count": len(retained_inventory),
        "retained_non_neural_inventory_sha256": _json_identity_sha256(
            {"manifests": retained_inventory}
        ),
        "superseded_neural_manifest_count": len(superseded_inventory),
        "superseded_neural_inventory_sha256": _json_identity_sha256(
            {"manifests": superseded_inventory}
        ),
        "superseded_neural_manifests": superseded_inventory,
        "manifest_policy": (
            "retain exact non-neural identities; archive only neural manifests "
            "whose batching/thread training protocol identity changed"
        ),
        "primary_lock_present": False,
    }
    if changed == ["neural_query_batch_size"]:
        amendment["old_value"] = old_semantic.get("neural_query_batch_size")
        amendment["new_value"] = new_semantic.get("neural_query_batch_size")
    prior_training_audit = output / "audit" / "prelock_training_amendment.json"
    prior_training_audit_exists = (
        prior_training_audit.is_file() and not prior_training_audit.is_symlink()
    )
    if prior_training_audit_exists:
        prior_training_archive = (
            output
            / "audit"
            / (f"prelock_training_amendment.before_{old_identity[:12]}.json")
        )
        if prior_training_archive.exists():
            if prior_training_archive.is_symlink() or streaming_sha256(
                prior_training_archive
            ) != streaming_sha256(prior_training_audit):
                raise StageError("prior training amendment archive already differs")
        else:
            temporary = prior_training_archive.with_name(
                f".{prior_training_archive.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            shutil.copyfile(prior_training_audit, temporary)
            os.replace(temporary, prior_training_archive)
        amendment["prior_training_amendment_audit"] = {
            "path": str(prior_training_archive.resolve()),
            "sha256": streaming_sha256(prior_training_archive),
        }
    _atomic_json(amendment_path, amendment)
    if not prior_training_audit_exists:
        _atomic_json(prior_training_audit, amendment)
    requested["prelock_device_amendment"] = existing.get("prelock_device_amendment")
    previous_training_amendment = existing.get("prelock_training_amendment")
    if previous_training_amendment is not None:
        requested["prior_prelock_training_amendment"] = previous_training_amendment
    requested["prelock_training_amendment"] = amendment
    requested["prelock_acceleration_amendment"] = amendment
    _atomic_json(config_path, requested)
    return requested


def _initialize_run(output: Path, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RUN_DIRECTORIES:
        (output / name).mkdir(exist_ok=True)
    _freeze_or_validate_run_config(output, args)
    _atomic_json(output / "logs" / "environment.json", _environment())


def _stage_paths(output: Path, stage: str) -> tuple[Path, Path]:
    root = output / "logs" / "stages" / stage
    return root / "status.json", root / "_SUCCESS.json"


def _stage_output_mutation_policy(
    output: Path, stage: str, path: Path
) -> tuple[str, str | None]:
    relative = path.resolve().relative_to(output.resolve()).as_posix()
    if (stage, relative) in IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES:
        return "immutable_snapshot", None
    policy = CUMULATIVE_STAGE_OUTPUT_POLICIES.get(relative)
    if policy is None or stage not in policy["mutable_receipt_stages"]:
        return "immutable", None
    return "superseded_by_stage", str(policy["successor_stage"])


def _receipt_run_path(output: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    root = Path(os.path.abspath(output))
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = Path(os.path.abspath(path))
    if path != root and root not in path.parents:
        return None
    return path


def _stage_completed(
    output: Path,
    stage: str,
    *,
    _checking: frozenset[str] = frozenset(),
) -> bool:
    if stage in _checking:
        return False
    _, marker = _stage_paths(output, stage)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        config_identity = _read_run_config_identity(output)
    except (OSError, json.JSONDecodeError):
        return False
    except StageError:
        return False
    raw_outputs = payload.get("outputs", [])
    if not isinstance(raw_outputs, list) or not raw_outputs:
        return False
    expected_receipt = (
        output / "logs" / "stages" / stage / "outputs_at_completion.json"
    ).resolve()
    outputs: list[Path] = []
    for descriptor in raw_outputs:
        if not isinstance(descriptor, Mapping):
            return False
        path = _receipt_run_path(output, descriptor.get("path"))
        if path is None:
            return False
        expected_sha = str(descriptor.get("sha256", ""))
        expected_size = descriptor.get("size_bytes")
        if (
            not path.is_file()
            or path.is_symlink()
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
            or streaming_sha256(path) != expected_sha
        ):
            return False
        outputs.append(path)
    if len(outputs) != 1 or outputs[0].resolve() != expected_receipt:
        return False
    try:
        receipt = json.loads(expected_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    declared_outputs = receipt.get("declared_outputs")
    envelope_valid = (
        payload.get("stage") == stage
        and payload.get("status") == "SUCCESS"
        and payload.get("semantic_config_sha256") == config_identity
        and receipt.get("schema_version") == 1
        and receipt.get("stage") == stage
        and receipt.get("status") == "CAPTURED_AT_STAGE_COMPLETION"
        and receipt.get("semantic_config_sha256") == config_identity
        and isinstance(declared_outputs, list)
        and bool(declared_outputs)
        and all(path.exists() for path in outputs)
    )
    if not envelope_valid or not isinstance(declared_outputs, list):
        return False

    root = Path(os.path.abspath(output))
    seen: set[Path] = set()
    prepared: list[tuple[Path, Mapping[str, Any], str, int, str, str | None]] = []
    for descriptor in declared_outputs:
        if not isinstance(descriptor, Mapping):
            return False
        path = _receipt_run_path(root, descriptor.get("path"))
        if path is None or path in seen:
            return False
        seen.add(path)
        expected_sha = str(descriptor.get("sha256_at_stage_completion", "")).lower()
        expected_size = descriptor.get("size_bytes_at_stage_completion")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            return False
        try:
            expected_policy, expected_successor = _stage_output_mutation_policy(
                root, stage, path
            )
        except ValueError:
            return False
        if (
            descriptor.get("mutation_policy") != expected_policy
            or descriptor.get("superseded_by_stage") != expected_successor
            or path.is_symlink()
            or not path.is_file()
        ):
            return False
        if expected_policy == "immutable_snapshot":
            relative = path.relative_to(root).as_posix()
            expected_snapshot = root / str(
                IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES[(stage, relative)]
            )
            snapshot = _receipt_run_path(root, descriptor.get("snapshot_path"))
            snapshot_sha = str(descriptor.get("snapshot_sha256", "")).lower()
            snapshot_size = descriptor.get("snapshot_size_bytes")
            if (
                snapshot != expected_snapshot
                or snapshot_sha != expected_sha
                or snapshot_size != expected_size
                or snapshot.is_symlink()
                or not snapshot.is_file()
                or snapshot.stat().st_size != snapshot_size
                or streaming_sha256(snapshot) != snapshot_sha
            ):
                return False
        prepared.append(
            (
                path,
                descriptor,
                expected_sha,
                expected_size,
                expected_policy,
                expected_successor,
            )
        )

    for (
        path,
        _descriptor,
        expected_sha,
        expected_size,
        expected_policy,
        expected_successor,
    ) in prepared:
        actual_size = path.stat().st_size
        actual_sha = streaming_sha256(path)
        if expected_policy == "immutable_snapshot":
            continue
        if actual_size == expected_size and actual_sha == expected_sha:
            continue
        if expected_policy != "superseded_by_stage" or expected_successor is None:
            return False
        if not _stage_completed(
            root,
            expected_successor,
            _checking=_checking | {stage},
        ):
            return False
        successor_receipt_path = (
            root / "logs" / "stages" / expected_successor / "outputs_at_completion.json"
        )
        try:
            successor_receipt = json.loads(
                successor_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False
        successor_matches = []
        for successor_descriptor in successor_receipt.get("declared_outputs", []):
            if not isinstance(successor_descriptor, Mapping):
                return False
            successor_path = _receipt_run_path(root, successor_descriptor.get("path"))
            if successor_path == path:
                successor_matches.append(successor_descriptor)
        if len(successor_matches) != 1:
            return False
        successor_descriptor = successor_matches[0]
        if (
            successor_descriptor.get("size_bytes_at_stage_completion") != actual_size
            or str(successor_descriptor.get("sha256_at_stage_completion", "")).lower()
            != actual_sha
        ):
            return False
    return True


def _run_stage(
    output: Path,
    stage: str,
    function: Callable[[], list[Path]],
    *,
    resume: bool,
    force: set[str],
) -> None:
    config_identity = _read_run_config_identity(output)
    if resume and stage not in force and _stage_completed(output, stage):
        _append_jsonl(
            output / "logs" / "progress.jsonl",
            {"timestamp": _now(), "stage": stage, "status": "SKIPPED_COMPLETE"},
        )
        return
    status_path, success_path = _stage_paths(output, stage)
    if success_path.exists():
        success_path.unlink()
    started = time.perf_counter()
    _atomic_json(
        status_path,
        {"stage": stage, "status": "RUNNING", "started_at": _now(), "pid": os.getpid()},
    )
    _append_jsonl(
        output / "logs" / "progress.jsonl",
        {"timestamp": _now(), "stage": stage, "status": "RUNNING", "pid": os.getpid()},
    )
    try:
        outputs = function()
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise StageError(f"stage {stage} declared missing outputs: {missing}")
        declared_outputs = []
        for path in outputs:
            mutation_policy, successor_stage = _stage_output_mutation_policy(
                output, stage, path
            )
            descriptor = {
                "path": str(path.resolve()),
                "sha256_at_stage_completion": streaming_sha256(path),
                "size_bytes_at_stage_completion": path.stat().st_size,
                "mutation_policy": mutation_policy,
                "superseded_by_stage": successor_stage,
            }
            if mutation_policy == "immutable_snapshot":
                relative = path.resolve().relative_to(output.resolve()).as_posix()
                snapshot = output / str(
                    IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES[(stage, relative)]
                )
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                temporary_snapshot = snapshot.with_name(
                    f".{snapshot.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                shutil.copyfile(path, temporary_snapshot)
                os.replace(temporary_snapshot, snapshot)
                descriptor.update(
                    {
                        "snapshot_path": str(snapshot.resolve()),
                        "snapshot_sha256": streaming_sha256(snapshot),
                        "snapshot_size_bytes": snapshot.stat().st_size,
                    }
                )
            declared_outputs.append(descriptor)
        if not declared_outputs:
            raise StageError(f"stage {stage} has no hashable non-checksum outputs")
        # Several cumulative artifacts (the experiment registry, primary
        # summary, and final sanity audit) are intentionally extended by later
        # stages.  Hashing those mutable paths directly in an earlier marker
        # makes a truthful all-stage run invalidate itself.  Preserve the exact
        # stage-completion evidence in one immutable, stage-owned receipt and
        # hash that receipt in the success marker.  The run-level audit still
        # verifies the final artifacts and root checksum manifest separately.
        receipt = success_path.with_name("outputs_at_completion.json")
        _atomic_json(
            receipt,
            {
                "schema_version": 1,
                "stage": stage,
                "status": "CAPTURED_AT_STAGE_COMPLETION",
                "captured_at": _now(),
                "semantic_config_sha256": config_identity,
                "declared_outputs": declared_outputs,
                "downstream_mutation_policy": (
                    "immutable unless exact stage/path policy names a successor receipt"
                ),
            },
        )
        marker_outputs = [receipt.resolve()]
        payload = {
            "stage": stage,
            "status": "SUCCESS",
            "started_at": json.loads(status_path.read_text())["started_at"],
            "completed_at": _now(),
            "duration_seconds": time.perf_counter() - started,
            "semantic_config_sha256": config_identity,
            "outputs": [
                {
                    "path": str(path),
                    "sha256": streaming_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in marker_outputs
            ],
        }
        _atomic_json(status_path, payload)
        _atomic_json(success_path, payload)
        _append_jsonl(
            output / "logs" / "progress.jsonl",
            {"timestamp": _now(), "stage": stage, "status": "SUCCESS"},
        )
    except Exception as error:
        failure = {
            "stage": stage,
            "status": "FAILED",
            "failed_at": _now(),
            "duration_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(status_path, failure)
        _append_jsonl(output / "logs" / "progress.jsonl", failure)
        raise


def _write_checksum_manifest(entries: Mapping[str, str], path: Path) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(entries.items())]
    _atomic_text(path, "\n".join(lines) + "\n")


def _canonical_manifest(route: str) -> dict[str, Any]:
    if route == "modular":
        return {
            "route": "modular",
            "run_path": str(MODULAR_CANONICAL_RUN),
            "candidate_pool": "full_post_filter",
            "sample_count": 7675,
            "raw_candidate_count": 1466046,
            "candidate_count": 187077,
            "raw_candidate_sha256": KNOWN_HASHES[
                "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/candidates/dexnet_raw_candidates.parquet"
            ],
            "candidate_sha256": KNOWN_HASHES[
                "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/candidates/dexnet_nms_candidates.parquet"
            ],
            "score_sha256": KNOWN_HASHES[
                "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/scores/gqcnn_per_candidate.parquet"
            ],
            "evaluated_sha256": KNOWN_HASHES[
                "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/evaluation/hierfilm_gqcnn_per_candidate.parquet"
            ],
            "historical_206538_pool_allowed": False,
        }
    if route == "crog":
        return {
            "route": "crog",
            "run_path": str(CROG_CANONICAL_RUN),
            "candidate_path": str(CROG_CANONICAL_CANDIDATES),
            "candidate_pool": "frozen_top5",
            "sample_count": 17749,
            "candidate_count": 88745,
            "candidate_sha256": KNOWN_HASHES[
                "crog_reproduction/CROG/failure_analysis/reranking_outputs/full_test_17749_v1/features.jsonl"
            ],
            "run_manifest_sha256": KNOWN_HASHES[
                "crog_reproduction/CROG/failure_analysis/reranking_outputs/v3_fullchain_20260801T114301+0100/frozen_experiment_manifest.json"
            ],
            "full_list_available": False,
        }
    raise ValueError(route)


def _stream_crog_baseline(
    features_path: Path, labels_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_rows: list[dict[str, Any]] = []
    candidate_total = 0
    top1_count = 0
    oracle_count = 0
    empty = 0
    with (
        features_path.open("r", encoding="utf-8") as features,
        labels_path.open("r", encoding="utf-8") as labels,
    ):
        for index, pair in enumerate(zip(features, labels, strict=True)):
            feature = json.loads(pair[0])
            label = json.loads(pair[1])
            label_source_id = str(
                label.get("source_sample_id", label.get("sample_id", ""))
            )
            if str(feature.get("sample_id")) != label_source_id:
                raise StageError(f"CROG baseline sample mismatch at row {index}")
            candidates = feature.get("candidates", [])
            by_id = {
                str(value["candidate_id"]): value
                for value in label.get("candidate_labels", [])
            }
            if len(by_id) != len(label.get("candidate_labels", [])):
                raise StageError(f"duplicate CROG label candidate ID at row {index}")
            ordered = sorted(
                candidates,
                key=lambda value: (
                    int(value.get("q_rank", value.get("legacy_rank", 0))),
                    str(value.get("candidate_id", "")),
                ),
            )
            correctness: list[bool] = []
            for candidate in ordered:
                candidate_id = str(candidate["candidate_id"])
                candidate_label = by_id.get(candidate_id)
                if candidate_label is None:
                    raise StageError(
                        f"missing CROG label {feature['sample_id']}/{candidate_id}"
                    )
                if str(candidate.get("candidate_checksum")) != str(
                    candidate_label.get("candidate_checksum")
                ):
                    raise StageError(f"CROG candidate checksum mismatch at row {index}")
                correctness.append(bool(candidate_label.get("candidate_correct")))
            candidate_total += len(correctness)
            is_empty = len(correctness) == 0
            top1 = bool(correctness[0]) if correctness else False
            oracle = any(correctness)
            empty += int(is_empty)
            top1_count += int(top1)
            oracle_count += int(oracle)
            query_rows.append(
                {
                    "sample_id": str(label.get("sample_id", feature["sample_id"])),
                    "scene_id": str(feature.get("scene_id", feature["sample_id"])),
                    "candidate_count": len(correctness),
                    "baseline_top1_correct": top1,
                    "pool_has_positive": oracle,
                }
            )
    query_count = len(query_rows)
    return (
        {
            "route": "crog",
            "query_count": query_count,
            "candidate_total": candidate_total,
            "empty_query_count": empty,
            "q_only_j_at_1_count": top1_count,
            "q_only_j_at_1": top1_count / query_count,
            "oracle_count": oracle_count,
            "oracle": oracle_count / query_count,
            "j_at_5": oracle_count / query_count,
        },
        query_rows,
    )


def _modular_baseline() -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = pd.read_parquet(MODULAR_EVALUATED)
    universe = pd.read_csv(MODULAR_INPUT_MANIFEST)[["sample_id", "scene_id"]]
    universe["frame_id"] = universe["scene_id"]
    pool = candidates.copy()
    if {"candidate_success", "joint_success"} <= set(pool.columns):
        left = pool["candidate_success"].astype(bool).to_numpy()
        right = pool["joint_success"].astype(bool).to_numpy()
        if not np.array_equal(left, right):
            raise StageError(
                "conflicting Modular candidate_success and joint_success labels"
            )
    label_source = (
        "candidate_success" if "candidate_success" in pool.columns else "joint_success"
    )
    pool["label"] = pool[label_source].astype(bool)
    pool["score"] = pd.to_numeric(pool["gqcnn_q_value"], errors="raise")
    result = evaluate_rankings(
        pool,
        query_universe=universe,
        query_col="sample_id",
        candidate_id_col="candidate_id",
        label_col="label",
        score_col="score",
        scene_col="scene_id",
        frame_col="frame_id",
    )
    metrics = dict(result["metrics"])
    metrics.update(
        {
            "route": "modular",
            "candidate_total": int(len(pool)),
            "q_only_j_at_1": metrics["j_at_1"],
            "q_only_j_at_1_count": metrics["j_at_1_count"],
        }
    )
    return metrics, result["per_query"]


def _write_resumable_audit_bundle(
    report: Mapping[str, Any],
    audit_dir: Path,
    *,
    resume: bool,
    source_roots: Sequence[Path],
) -> dict[str, str]:
    """Refresh stage-owned provenance files without deleting peer audits."""

    if not audit_dir.exists() or not any(audit_dir.iterdir()):
        if audit_dir.exists():
            audit_dir.rmdir()
        return write_audit_bundle(report, audit_dir, source_roots=source_roots)
    if not resume:
        raise FileExistsError(audit_dir)
    staging = audit_dir.parent / (
        f".{audit_dir.name}.provenance-refresh.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        written = write_audit_bundle(report, staging, source_roots=source_roots)
        audit_dir.mkdir(parents=True, exist_ok=True)
        published: dict[str, str] = {"output_dir": str(audit_dir.resolve())}
        for key, name in (
            ("json", "audit.json"),
            ("markdown", "audit.md"),
            ("tsv", "inventory.tsv"),
        ):
            source = staging / name
            destination = audit_dir / name
            os.replace(source, destination)
            published[key] = str(destination.resolve())
        return published
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _stage_audit(output: Path, args: argparse.Namespace) -> list[Path]:
    audit = audit_canonical_runs(
        verify_hashes=not args.no_hash_audit,
        verify_all_known_artifacts=not args.no_hash_audit,
    )
    if not audit["passed"]:
        raise StageError("canonical provenance audit failed")
    audit_dir = output / "audit"
    written = _write_resumable_audit_bundle(
        audit,
        audit_dir,
        resume=bool(args.resume),
        source_roots=(MODULAR_CANONICAL_RUN, CROG_CANONICAL_RUN),
    )
    canonical_md = audit_dir / "CANONICAL_RUN_AUDIT.md"
    _atomic_text(
        canonical_md,
        "# Canonical run audit\n\n"
        f"Status: PASS\n\nModular: `{MODULAR_CANONICAL_RUN}`\n\n"
        f"CROG: `{CROG_CANONICAL_RUN}`\n\n"
        "The historic 206,538-candidate Modular source directory is absent and is "
        "excluded from every formal table. The retained 187,077-candidate run is the "
        "only physical canonical Modular test pool.\n",
    )
    discrepancy = audit_dir / "BASELINE_DISCREPANCY_REPORT.md"
    _atomic_text(
        discrepancy,
        "# Baseline discrepancy report\n\n"
        "Historical documentation mentions a 206,538-candidate Modular pool, but its "
        "candidate, score, and evaluation directories are no longer present. It cannot "
        "be frozen or ID-audited. The complete retained run contains 187,077 candidates "
        "and is therefore the only eligible canonical source. No artifact from the two "
        "versions is mixed.\n",
    )
    crog_manifest = output / "manifests" / "canonical_crog_run.json"
    modular_manifest = output / "manifests" / "canonical_modular_run.json"
    _atomic_json(crog_manifest, _canonical_manifest("crog"))
    _atomic_json(modular_manifest, _canonical_manifest("modular"))
    input_checksums = output / "manifests" / "input_checksums.sha256"
    _write_checksum_manifest(KNOWN_HASHES, input_checksums)

    crog_metrics, crog_queries = _stream_crog_baseline(
        CROG_CANONICAL_CANDIDATES, CROG_TEST_LABELS
    )
    modular_metrics, modular_queries = _modular_baseline()
    recomputation = {
        "computed_at": _now(),
        "evaluator_semantics": "strict same-GT, IoU > 0.25, 180-periodic angle <= 30 deg",
        "crog": crog_metrics,
        "modular": modular_metrics,
        "two_dimensional_annotation_consistency_not_physical_success": True,
    }
    recompute_json = audit_dir / "baseline_recomputation.json"
    _atomic_json(recompute_json, recomputation)
    pd.DataFrame(crog_queries).to_parquet(
        audit_dir / "crog_baseline_per_query.parquet", index=False, compression="zstd"
    )
    modular_queries.to_parquet(
        audit_dir / "modular_baseline_per_query.parquet",
        index=False,
        compression="zstd",
    )
    baseline_md = audit_dir / "BASELINE_RECOMPUTATION.md"
    _atomic_text(
        baseline_md,
        "# Independent baseline recomputation\n\n"
        "These are 2D annotation-consistency metrics, not empirical robot success.\n\n"
        "| route | queries | candidates | empty | q-only J@1 | Oracle |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| CROG Top-5 | {crog_metrics['query_count']} | {crog_metrics['candidate_total']} | "
        f"{crog_metrics['empty_query_count']} | {crog_metrics['q_only_j_at_1']:.9f} | "
        f"{crog_metrics['oracle']:.9f} |\n"
        f"| Modular full | {modular_metrics['query_count']} | {modular_metrics['candidate_total']} | "
        f"{modular_metrics['empty_query_count']} | {modular_metrics['q_only_j_at_1']:.9f} | "
        f"{modular_metrics['oracle']:.9f} |\n",
    )
    return [
        Path(written["json"]),
        canonical_md,
        discrepancy,
        crog_manifest,
        modular_manifest,
        input_checksums,
        recompute_json,
        baseline_md,
        audit_dir / "crog_baseline_per_query.parquet",
        audit_dir / "modular_baseline_per_query.parquet",
    ]


def _save_build_result(result: BuildResult, path: Path) -> None:
    _atomic_json(path, asdict(result))


def _verify_modular_development_source(
    output: Path,
    published_features: Path,
    *,
    query_universe_paths: Sequence[Path] = MODULAR_DEVELOPMENT_QUERY_MANIFESTS,
    candidate_count_evidence_paths: Sequence[
        Path
    ] = MODULAR_DEVELOPMENT_CANDIDATE_COUNT_EVIDENCE,
) -> Path:
    """Recompute the published development table before formal consumption.

    The formal run retains a small, immutable verification receipt rather than
    copying the large upstream Parquet.  The receipt binds both the published
    table and its independently recomputed companion manifest; completion
    auditing repeats this verification against the original sources.
    """

    try:
        manifest = verify_modular_development_publication(
            published_features,
            expected_query_universe_paths=query_universe_paths,
            expected_candidate_count_evidence_paths=(candidate_count_evidence_paths),
        )
    except PublishError as error:
        raise StageError(
            "Modular development feature publication failed independent "
            f"verification: {error}"
        ) from error
    manifest_path = published_features.with_name(
        published_features.name + ".manifest.json"
    )
    receipt = output / "manifests" / "modular_development_publication_verification.json"
    _atomic_json(
        receipt,
        {
            "schema_version": 1,
            "status": "VERIFIED_BEFORE_FORMAL_CONSUMPTION",
            "verified_at": _now(),
            "published_features_path": str(published_features.resolve()),
            "published_features_sha256": streaming_sha256(published_features),
            "publication_manifest_path": str(manifest_path.resolve()),
            "publication_manifest_sha256": streaming_sha256(manifest_path),
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "rows": int(manifest["rows"]),
            "queries": int(manifest["queries"]),
            "independent_reconstruction_exact": True,
        },
    )
    return receipt


def _concat_parquets(paths: Sequence[Path], output: Path) -> None:
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    combined.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)


def _stage_build_features(output: Path, args: argparse.Namespace) -> list[Path]:
    outputs: list[Path] = []
    feature_audits: list[dict[str, Any]] = []
    if "crog" in args.routes:
        crog_parts: list[tuple[str, Path, Path]] = [
            ("train", CROG_TRAIN_FEATURES, CROG_TRAIN_LABELS),
            ("val", CROG_VAL_FEATURES, CROG_VAL_LABELS),
            ("test", CROG_CANONICAL_CANDIDATES, CROG_TEST_LABELS),
        ]
        built_features: dict[str, Path] = {}
        built_labels: dict[str, Path] = {}
        built_queries: dict[str, Path] = {}
        for split, feature_source, label_source in crog_parts:
            feature_path = (
                output / "features" / f"candidates_crog_frozen_top5_{split}.parquet"
            )
            label_path = output / "data" / f"labels_crog_frozen_top5_{split}.parquet"
            query_path = output / "data" / f"queries_crog_frozen_top5_{split}.parquet"
            result = build_crog_candidate_tables(
                feature_source,
                feature_path,
                labels_jsonl=label_source,
                output_labels=label_path,
                output_query_universe=query_path,
                pool_type="frozen_top5",
            )
            build_manifest = (
                output / "manifests" / f"build_crog_frozen_top5_{split}.json"
            )
            _save_build_result(result, build_manifest)
            built_features[split] = feature_path
            built_labels[split] = label_path
            built_queries[split] = query_path
            outputs.extend([feature_path, label_path, query_path, build_manifest])
        development_features = (
            output / "features" / "candidates_crog_frozen_top5_development.parquet"
        )
        development_labels = (
            output / "data" / "labels_crog_frozen_top5_development.parquet"
        )
        _concat_parquets(
            [built_features["train"], built_features["val"]], development_features
        )
        _concat_parquets(
            [built_labels["train"], built_labels["val"]], development_labels
        )
        crog_development_frame = pd.read_parquet(development_features)
        crog_native_columns = select_crog_native_feature_columns(crog_development_frame)
        audit = audit_feature_table(
            crog_development_frame,
            pd.read_parquet(development_labels),
            requested=crog_native_columns,
        )
        audit["excluded_non_native_depth_evidence_columns"] = sorted(
            set(select_model_feature_columns(crog_development_frame))
            - set(crog_native_columns)
        )
        feature_audits.append({"route": "crog", "pool": "frozen_top5", **audit})
        outputs.extend([development_features, development_labels])

    if "modular" in args.routes:
        if not MODULAR_RICH_TEST_FEATURES.is_file():
            blocker = output / "data" / "modular_test_feature_upstream_status.json"
            _atomic_json(
                blocker,
                {
                    "required_path": str(MODULAR_RICH_TEST_FEATURES),
                    "status": "UPSTREAM_RETAINED_TEST_FEATURE_EXTRACTION_INCOMPLETE",
                    "canonical_labels_or_metrics_used_for_model_selection": False,
                },
            )
            raise StageError(
                "Modular retained-test leakage-safe feature extraction is "
                f"incomplete; evidence: {blocker}"
            )
        for pool in [value for value in args.pools if value in {"top5", "full"}]:
            pool_name = "frozen_top5" if pool == "top5" else "full_post_filter"
            feature_path = (
                output / "features" / f"candidates_modular_{pool_name}_test.parquet"
            )
            label_path = output / "data" / f"labels_modular_{pool_name}_test.parquet"
            result = build_modular_candidate_tables(
                MODULAR_EVALUATED,
                feature_path,
                label_path,
                pool_type=pool_name,
                query_universe=MODULAR_INPUT_MANIFEST,
                inference_features=MODULAR_RICH_TEST_FEATURES,
            )
            build_manifest = (
                output / "manifests" / f"build_modular_{pool_name}_test.json"
            )
            _save_build_result(result, build_manifest)
            outputs.extend(
                [
                    feature_path,
                    label_path,
                    Path(result.query_universe_path),
                    build_manifest,
                ]
            )

        merged_development = (
            MODULAR_DEVELOPMENT_RUN / "features" / "per_candidate.parquet"
        )
        if not merged_development.is_file():
            evidence = {
                "required_path": str(merged_development),
                "existing_partial_feature_shards": [
                    str(path)
                    for path in sorted(
                        (
                            MODULAR_DEVELOPMENT_RUN / "tmp" / "train" / "features_scene"
                        ).glob("shard_*/per_candidate.parquet")
                    )
                ],
                "status": "UPSTREAM_DEVELOPMENT_CANDIDATE_BUILD_INCOMPLETE",
                "formal_test_read": False,
            }
            blocker = output / "data" / "modular_development_upstream_status.json"
            _atomic_json(blocker, evidence)
            raise StageError(
                "Modular train/validation candidate generation, official GQ-CNN scoring, "
                f"and feature merge are incomplete; evidence: {blocker}"
            )
        outputs.append(_verify_modular_development_source(output, merged_development))
        for pool in [value for value in args.pools if value in {"top5", "full"}]:
            pool_name = "frozen_top5" if pool == "top5" else "full_post_filter"
            feature_path = (
                output
                / "features"
                / f"candidates_modular_{pool_name}_development.parquet"
            )
            label_path = (
                output / "data" / f"labels_modular_{pool_name}_development.parquet"
            )
            query_frames = [
                pd.read_json(path, lines=True)[["sample_id", "scene_id"]]
                for path in MODULAR_DEVELOPMENT_QUERY_MANIFESTS
            ]
            development_universe = pd.concat(query_frames, ignore_index=True)
            development_universe["frame_id"] = development_universe["scene_id"]
            result = build_modular_candidate_tables(
                merged_development,
                feature_path,
                label_path,
                pool_type=pool_name,
                query_universe=development_universe,
            )
            build_manifest = (
                output / "manifests" / f"build_modular_{pool_name}_development.json"
            )
            _save_build_result(result, build_manifest)
            outputs.extend(
                [
                    feature_path,
                    label_path,
                    Path(result.query_universe_path),
                    build_manifest,
                ]
            )
            audit = audit_feature_table(
                pd.read_parquet(feature_path), pd.read_parquet(label_path)
            )
            feature_audits.append({"route": "modular", "pool": pool_name, **audit})

    if feature_audits:
        combined_path = output / "audit" / "feature_audit.json"
        feature_schema = output / "features" / "feature_schema.json"
        write_feature_audit_bundle(
            feature_audits,
            json_path=combined_path,
            markdown_path=output / "audit" / "FEATURE_AUDIT.md",
            schema_path=feature_schema,
        )
        outputs.extend(
            [
                combined_path,
                output / "audit" / "FEATURE_AUDIT.md",
                feature_schema,
            ]
        )
    pool_status = output / "data" / "candidate_pool_status.json"
    _atomic_json(
        pool_status,
        {
            "schema_version": 1,
            "pools": [
                {
                    "route": "crog",
                    "pool": "frozen_top5",
                    "status": "AVAILABLE",
                    "oracle_comparison_scope": "within this frozen pool only",
                },
                {
                    "route": "crog",
                    "pool": "full_post_filter",
                    "status": "NOT_AVAILABLE_SOURCE_POOL",
                    "reason": "canonical CROG artifact exposes exactly five frozen candidates per query",
                },
                {
                    "route": "crog",
                    "pool": "pre_filter",
                    "status": "NOT_AVAILABLE_SOURCE_POOL",
                    "reason": "pre-filter CROG proposals were not retained in the canonical run",
                },
                {
                    "route": "modular",
                    "pool": "frozen_top5",
                    "status": "AVAILABLE",
                    "oracle_comparison_scope": "within Modular Top-5 only",
                },
                {
                    "route": "modular",
                    "pool": "full_post_filter",
                    "status": "AVAILABLE",
                    "oracle_comparison_scope": "within Modular post-NMS full list only",
                },
                {
                    "route": "modular",
                    "pool": "pre_filter",
                    "status": "NOT_RUN_UNSCORED_POOL",
                    "reason": (
                        "raw/mask-valid candidates are retained but do not have a "
                        "complete matching official GQ-CNN score table; post-filter "
                        "and pre-filter oracle denominators are never mixed"
                    ),
                },
            ],
        },
    )
    outputs.append(pool_status)
    return outputs


def _deferred_stage(name: str, output: Path) -> list[Path]:
    """Invoke the matrix or evidence-reporting implementation."""

    if name in {"statistics", "visualize", "report"}:
        from reranking.reporting_bridge import run_reporting_bridge

        return [Path(value) for value in run_reporting_bridge(name, output)]
    try:
        from reranking.matrix import run_matrix_stage
    except ImportError as error:
        raise StageError(
            f"matrix stage {name} is not implemented; refusing to create a success marker"
        ) from error
    return [Path(value) for value in run_matrix_stage(name, output)]


def _required_stage_names(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return STAGES
    if requested not in STAGES:
        raise ValueError(requested)
    return (requested,)


def _run_evaluator_regression_tests(output: Path) -> Path:
    log_path = output / "logs" / "evaluator_tests.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "reranking/tests/test_independent_evaluator.py",
        "reranking/tests/test_statistics_evaluate.py",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _atomic_text(log_path, completed.stdout or "pytest produced no output\n")
    passed_match = re.search(r"(\d+) passed", completed.stdout or "")
    failed_match = re.search(r"(\d+) failed", completed.stdout or "")
    error_match = re.search(r"(\d+) errors?", completed.stdout or "")
    failed = int(failed_match.group(1)) if failed_match else 0
    failed += int(error_match.group(1)) if error_match else 0
    evidence_path = output / "audit" / "evaluator_tests.json"
    _atomic_json(
        evidence_path,
        {
            "status": "PASS" if completed.returncode == 0 and failed == 0 else "FAIL",
            "exit_code": completed.returncode,
            "tests_passed": int(passed_match.group(1)) if passed_match else 0,
            "tests_failed": failed,
            "command": command,
            "completed_at": _now(),
            "log_path": str(log_path.resolve()),
            "log_sha256": streaming_sha256(log_path),
        },
    )
    if completed.returncode != 0 or failed:
        raise StageError(f"independent evaluator regression tests failed: {log_path}")
    return evidence_path


def _relevant_background_activity() -> dict[str, Any]:
    process_patterns = (
        "generate_compact_dexnet_candidates.py",
        "run_full_gqcnn_scoring.py",
        "extract_candidate_features.py",
        "reranking.matrix",
        "run_experiment_matrix.py",
    )
    active_processes: list[dict[str, Any]] = []
    process = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    for line in process.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid, parent_pid, command = parts
        if int(pid) == os.getpid():
            continue
        if any(pattern in command for pattern in process_patterns):
            active_processes.append(
                {"pid": int(pid), "ppid": int(parent_pid), "command": command}
            )
    active_containers: list[dict[str, Any]] = []
    docker = subprocess.run(
        [
            "docker",
            "ps",
            "--format",
            "{{.ID}}\t{{.Image}}\t{{.Command}}\t{{.Names}}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if docker.returncode == 0:
        for line in docker.stdout.splitlines():
            fields = line.split("\t")
            text = " ".join(fields)
            if any(
                token in text.lower()
                for token in ("gqcnn", "rerank", "dexnet", "extract_candidate")
            ):
                active_containers.append(
                    {
                        "container_id": fields[0] if fields else "",
                        "image": fields[1] if len(fields) > 1 else "",
                        "command": fields[2] if len(fields) > 2 else "",
                        "name": fields[3] if len(fields) > 3 else "",
                    }
                )
    return audit_background_activity(
        active_processes,
        active_containers,
        captured_at=_now(),
    )


def _write_complete_checksum_manifest(output: Path) -> Path:
    checksum_path = output / "checksums.sha256"
    entries: dict[str, str] = {}
    excluded = {
        checksum_path.resolve(),
        (output / "_SUCCESS.json").resolve(),
        (output / "_PARTIAL.json").resolve(),
        (output / "audit" / "run_completion_audit.json").resolve(),
    }
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve() in excluded
            or path.name.endswith(".tmp")
        ):
            continue
        entries[path.relative_to(output).as_posix()] = streaming_sha256(path)
    _write_checksum_manifest(entries, checksum_path)
    return checksum_path


def _write_run_success(output: Path, args: argparse.Namespace) -> None:
    incomplete = [stage for stage in STAGES if not _stage_completed(output, stage)]
    if incomplete:
        raise StageError(
            f"run-level success forbidden; incomplete stages: {incomplete}"
        )
    _run_evaluator_regression_tests(output)
    partial = output / "_PARTIAL.json"
    if partial.exists():
        partial.unlink()
    _write_complete_checksum_manifest(output)
    from reranking.matrix import _specs

    completion = audit_run_completion(
        output,
        [spec.key for spec in _specs("formal")],
        [
            DatasetExpectation("crog_frozen_top5", "crog", "frozen_top5"),
            DatasetExpectation("modular_frozen_top5", "modular", "frozen_top5"),
            DatasetExpectation(
                "modular_full_post_filter", "modular", "full_post_filter"
            ),
        ],
        args.seeds,
        args.folds,
        background_evidence=_relevant_background_activity(),
    )
    completion_path = output / "audit" / "run_completion_audit.json"
    _atomic_json(completion_path, completion)
    if not completion["passed"]:
        raise StageError(
            "run-level completion audit failed; see "
            f"{completion_path} ({len(completion['blockers'])} blocker(s))"
        )
    _atomic_json(
        output / "_SUCCESS.json",
        {
            "status": "SUCCESS",
            "completed_at": _now(),
            "stages": list(STAGES),
            "folds": args.folds,
            "seeds": args.seeds,
            "strict_completion_contract": True,
        },
    )


def _stage_statuses(output: Path) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for stage in STAGES:
        status, _ = _stage_paths(output, stage)
        if status.is_file():
            try:
                statuses[stage] = json.loads(status.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                statuses[stage] = {"status": "UNREADABLE"}
        else:
            statuses[stage] = {"status": "NOT_STARTED"}
    return statuses


def _write_partial(output: Path, error: Exception | None = None) -> None:
    payload: dict[str, Any] = {
        "status": "PARTIAL",
        "updated_at": _now(),
        "stages": _stage_statuses(output),
        "success_marker_written": False,
    }
    if error is None:
        payload["reason"] = "requested stages completed; full run remains incomplete"
    else:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    _atomic_json(
        output / "_PARTIAL.json",
        payload,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--routes", nargs="+", choices=("crog", "modular"), default=["crog", "modular"]
    )
    parser.add_argument(
        "--pools",
        nargs="+",
        choices=("top5", "full", "pre_filter"),
        default=["top5", "full"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--neural-query-batch-size", type=int, default=128)
    parser.add_argument(
        "--neural-batching-policy",
        choices=("length_bucketed_v1", "legacy"),
        default="length_bucketed_v1",
    )
    parser.add_argument("--train-worker-count", type=int, default=1)
    parser.add_argument("--train-worker-index", type=int, default=0)
    parser.add_argument("--train-torch-thread-count", type=int, default=5)
    parser.add_argument("--train-torch-interop-thread-count", type=int, default=1)
    parser.add_argument("--train-worker-owner")
    parser.add_argument("--train-claim-lease-seconds", type=int, default=86_400)
    parser.add_argument(
        "--stage", choices=(*STAGES, *SPECIAL_TRAIN_STAGES, "all"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-rerun-specific-experiment", action="append", default=[]
    )
    parser.add_argument(
        "--amend-prelock-device",
        action="store_true",
        help=(
            "allow a one-time, benchmark-backed device-only semantic amendment "
            "before primary locking"
        ),
    )
    parser.add_argument(
        "--amend-prelock-training",
        action="store_true",
        help=(
            "allow a one-time, benchmark-backed neural batch-size amendment "
            "before primary locking"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--no-hash-audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.project_root.resolve() != REPO_ROOT.resolve():
        raise StageError(f"project root mismatch: {args.project_root} != {REPO_ROOT}")
    if args.folds != 5:
        raise StageError("formal protocol requires exactly 5 grouped folds")
    if sorted(args.seeds) != [42, 123, 2026]:
        raise StageError("formal protocol requires seeds 42, 123, 2026")
    if args.neural_query_batch_size <= 0:
        raise StageError("neural query batch size must be positive")
    if args.train_worker_count <= 0:
        raise StageError("train worker count must be positive")
    if not 0 <= args.train_worker_index < args.train_worker_count:
        raise StageError("train worker index is outside the configured worker range")
    if args.train_torch_thread_count <= 0:
        raise StageError("train torch thread count must be positive")
    if args.train_torch_interop_thread_count <= 0:
        raise StageError("train Torch inter-op thread count must be positive")
    if args.train_claim_lease_seconds <= 0:
        raise StageError("train claim lease must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    output = _safe_output_path(args.output, resume=args.resume)
    if args.stage in SPECIAL_TRAIN_STAGES:
        if not args.resume or not output.is_dir():
            raise StageError("train workers/finalizer require an existing --resume run")
        _read_run_config_identity(output)
        if args.neural_batching_policy != "length_bucketed_v1":
            raise StageError(
                "formal train workers require neural batching policy length_bucketed_v1"
            )
        from reranking.matrix import (
            run_matrix_train_finalize,
            run_matrix_train_worker,
            train_worker_lifecycle_lock,
        )

        if args.stage == "train-worker":
            try:
                protocol_path = output / "configs" / "train_worker_protocol.json"
                if not protocol_path.exists() and (
                    args.train_worker_index != 0 or not args.amend_prelock_training
                ):
                    raise StageError(
                        "worker 0 must explicitly apply the benchmark-backed "
                        "pre-lock training amendment before other workers start"
                    )
                _freeze_or_validate_run_config(output, args)
                run_matrix_train_worker(
                    output,
                    worker_index=args.train_worker_index,
                    worker_count=args.train_worker_count,
                    torch_thread_count=args.train_torch_thread_count,
                    torch_interop_thread_count=(args.train_torch_interop_thread_count),
                    owner=args.train_worker_owner,
                    lease_seconds=args.train_claim_lease_seconds,
                )
                return 0
            except Exception as error:
                print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
        try:
            _freeze_or_validate_run_config(output, args)
            protocol_path = output / "configs" / "train_worker_protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            if (
                protocol.get("worker_count") != args.train_worker_count
                or protocol.get("torch_thread_count") != args.train_torch_thread_count
                or protocol.get("torch_interop_thread_count")
                != args.train_torch_interop_thread_count
            ):
                raise StageError(
                    "finalizer worker/thread arguments differ from immutable protocol"
                )
            with train_worker_lifecycle_lock(output, exclusive=True):
                _run_stage(
                    output,
                    "train",
                    lambda: [
                        Path(path)
                        for path in run_matrix_train_finalize(
                            output, lifecycle_lock_held=True
                        )
                    ],
                    resume=args.resume,
                    force=set(),
                )
            _write_partial(output)
            return 0
        except Exception as error:
            _write_partial(output, error)
            print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    if torch.get_num_interop_threads() != args.train_torch_interop_thread_count:
        torch.set_num_interop_threads(args.train_torch_interop_thread_count)
    torch.set_num_threads(args.train_torch_thread_count)
    _initialize_run(output, args)
    stage_functions: dict[str, Callable[[], list[Path]]] = {
        "audit-only": lambda: _stage_audit(output, args),
        "build-features": lambda: _stage_build_features(output, args),
    }
    for stage in STAGES[2:]:
        stage_functions[stage] = lambda stage=stage: _deferred_stage(stage, output)
    force = set(args.force_rerun_specific_experiment)
    try:
        for stage in _required_stage_names(args.stage):
            _run_stage(
                output,
                stage,
                stage_functions[stage],
                resume=args.resume,
                force=force,
            )
        if args.stage == "all":
            _write_run_success(output, args)
        else:
            _write_partial(output)
        return 0
    except Exception as error:
        _write_partial(output, error)
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
