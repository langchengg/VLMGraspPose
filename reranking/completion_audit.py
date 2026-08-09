"""Fail-closed run-level completion audit for the formal reranking matrix.

The orchestrator is intentionally not imported here.  This module only reads a
completed run directory and returns structured evidence suitable for deciding
whether a run-level ``_SUCCESS.json`` may be written.  Missing or malformed
evidence is always a blocker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from reranking.crog_existing_comparison import (
    CrogExistingComparisonError,
    validate_crog_existing_comparison_report,
)
from reranking.leakage_audit import (
    LeakageAuditError,
    verify_leakage_audit_bundle,
)
from reranking.publish_modular_development import (
    PublishError,
    verify_modular_development_publication,
)


FORMAL_STAGES = (
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

# Exactly five cumulative path roles are permitted to differ from an earlier
# stage-completion hash.  Each role is bound to explicit source stage(s) and to
# one successor stage whose receipt must redeclare the path at its final hash.
# No filename matching, directory wildcard, or caller-supplied exception is
# accepted by the completion audit.
CUMULATIVE_STAGE_OUTPUT_POLICIES: Mapping[str, Mapping[str, Any]] = {
    "metrics/experiment_registry.json": {
        "mutable_receipt_stages": ("train", "test-primary"),
        "successor_stage": "test-post-lock",
    },
    "metrics/experiment_registry.parquet": {
        "mutable_receipt_stages": ("train", "test-primary"),
        "successor_stage": "test-post-lock",
    },
    "metrics/primary_summary.json": {
        "mutable_receipt_stages": ("test-primary",),
        "successor_stage": "statistics",
    },
    "audit/SANITY_AUDIT.json": {
        "mutable_receipt_stages": ("validate",),
        "successor_stage": "report",
    },
    "audit/SANITY_AUDIT.md": {
        "mutable_receipt_stages": ("validate",),
        "successor_stage": "report",
    },
}

# The report stage emits ``checksums.sha256`` and the orchestrator necessarily
# rewrites that file once all stage receipts exist.  Preserve the report-stage
# bytes in this one exact stage-owned snapshot instead of broadening the five
# cumulative successor roles above.
IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES: Mapping[
    tuple[str, str], str
] = {
    ("report", "checksums.sha256"): (
        "logs/stages/report/snapshots/checksums.sha256"
    ),
}

STATISTICS_FILES = (
    "statistics/mcnemar_results.csv",
    "statistics/bootstrap_intervals.csv",
    "statistics/holm_corrected_results.csv",
)

ABLATION_FILES = (
    "metrics/feature_ablation.csv",
    "metrics/loss_ablation.csv",
    "metrics/encoder_ablation.csv",
    "metrics/gate_ablation.csv",
    "metrics/pool_ablation.csv",
)

REPORT_FILES = (
    "reports/FINAL_REPORT_ZH.md",
    "reports/FINAL_REPORT_EN.md",
)

GALLERY_CATEGORIES = (
    "recovered",
    "harmful",
    "unchanged",
    "bothwrong",
    "empty",
    "no-positive",
    "rank>5",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DatasetExpectation:
    """One route/pool dataset whose complete formal coverage is required."""

    key: str
    route: str
    pool: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return str(value)


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    rows = [
        [_json_scalar(value) for value in row]
        for row in frame[list(columns)].itertuples(index=False, name=None)
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _reranking_source_tree_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(
        root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()
    ):
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def audit_background_activity(
    active_processes: Sequence[Mapping[str, Any] | str] = (),
    active_docker_containers: Sequence[Mapping[str, Any] | str] = (),
    *,
    captured_at: str,
) -> dict[str, Any]:
    """Build injectable background-activity evidence.

    Callers are responsible for supplying only relevant training/scoring jobs
    and containers.  Keeping discovery outside this function avoids treating
    the pytest process that invokes the audit as an active training job.
    """

    processes = [dict(item) if isinstance(item, Mapping) else {"command": str(item)} for item in active_processes]
    containers = [dict(item) if isinstance(item, Mapping) else {"name": str(item)} for item in active_docker_containers]
    timestamp = _iso8601(captured_at)
    return {
        "status": "PASS" if timestamp is not None and not processes and not containers else "FAIL",
        "captured_at": captured_at,
        "active_processes": processes,
        "active_docker_containers": containers,
    }


class _AuditState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.blockers: list[dict[str, Any]] = []
        self.missing_paths: set[str] = set()
        self.checks: dict[str, dict[str, Any]] = {}
        self.stage_outputs: set[Path] = set()
        self.manifest_artifacts: set[Path] = set()
        self.critical_files: set[Path] = set()

    def fail(
        self,
        code: str,
        message: str,
        *,
        path: Path | str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        blocker: dict[str, Any] = {"code": code, "message": message}
        if path is not None:
            blocker["path"] = str(path)
        if details:
            blocker["details"] = dict(details)
        self.blockers.append(blocker)

    def missing(self, relative: str, code: str = "missing_evidence") -> None:
        path = self.root / relative
        self.missing_paths.add(str(path))
        self.fail(code, f"required evidence is missing: {relative}", path=path)

    def run(self, name: str, function: Any) -> None:
        before = len(self.blockers)
        try:
            function()
        except Exception as error:  # the completion decision must remain structured
            self.fail(
                "audit_check_exception",
                f"{name} check raised {type(error).__name__}: {error}",
            )
        self.checks[name] = {
            "passed": len(self.blockers) == before,
            "blocker_count": len(self.blockers) - before,
        }

    def regular_file(self, relative: str, *, nonempty: bool = True) -> Path | None:
        path = self.root / relative
        if not path.exists():
            self.missing(relative)
            return None
        if path.is_symlink() or not path.is_file():
            self.fail("invalid_evidence_file", "evidence must be a regular non-symlink file", path=path)
            return None
        if nonempty and path.stat().st_size == 0:
            self.fail("empty_evidence_file", "evidence file is empty", path=path)
            return None
        self.critical_files.add(path)
        return path

    def json(self, relative: str) -> Mapping[str, Any] | None:
        path = self.regular_file(relative)
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.fail("invalid_json_evidence", f"cannot parse JSON evidence: {error}", path=path)
            return None
        if not isinstance(value, Mapping):
            self.fail("invalid_json_evidence", "JSON evidence must be an object", path=path)
            return None
        return value

    def resolve_run_path(self, raw: Any, *, source: Path) -> Path | None:
        if not isinstance(raw, str) or not raw.strip():
            self.fail("invalid_artifact_path", "artifact path must be a non-empty string", path=source)
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        path = Path(os.path.abspath(path))
        if path != self.root and self.root not in path.parents:
            self.fail("artifact_path_escape", "run artifact escapes the output root", path=path)
            return None
        return path


def _normalize_datasets(
    value: Sequence[DatasetExpectation | Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> tuple[DatasetExpectation, ...]:
    rows: Iterable[Any]
    if isinstance(value, Mapping):
        rows = (
            {"key": key, **dict(item)}
            for key, item in value.items()
        )
    else:
        rows = value
    normalized: list[DatasetExpectation] = []
    for row in rows:
        if isinstance(row, DatasetExpectation):
            item = row
        elif isinstance(row, Mapping):
            item = DatasetExpectation(
                key=str(row.get("key", row.get("dataset", ""))),
                route=str(row.get("route", "")),
                pool=str(row.get("pool", "")),
            )
        else:
            raise TypeError("expected_datasets entries must be DatasetExpectation or mappings")
        if not item.key or not item.route or not item.pool:
            raise ValueError("every expected dataset requires key, route, and pool")
        normalized.append(item)
    if not normalized or len({item.key for item in normalized}) != len(normalized):
        raise ValueError("expected_datasets must be non-empty with unique keys")
    return tuple(normalized)


def _fold_values(value: int | Sequence[int]) -> tuple[int, ...]:
    folds = tuple(range(value)) if isinstance(value, int) else tuple(map(int, value))
    if not folds or len(set(folds)) != len(folds) or min(folds) < 0:
        raise ValueError("expected_folds must identify unique non-negative folds")
    return folds


def _run_config_identity(state: _AuditState) -> str | None:
    path = state.root / "configs" / "run_config.json"
    payload = state.json("configs/run_config.json")
    if payload is None:
        return None
    semantic = payload.get("semantic_config")
    expected = str(payload.get("semantic_config_sha256", "")).lower()
    if (
        not isinstance(semantic, Mapping)
        or not _SHA256.fullmatch(expected)
        or _canonical_mapping_sha256(semantic) != expected
    ):
        state.fail(
            "run_config_identity_invalid",
            "frozen run configuration has no valid semantic identity",
            path=path,
        )
        return None
    return expected


def _expected_stage_output_policy(
    state: _AuditState, stage: str, output: Path
) -> tuple[str, str | None]:
    relative = output.relative_to(state.root).as_posix()
    if (stage, relative) in IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES:
        return "immutable_snapshot", None
    policy = CUMULATIVE_STAGE_OUTPUT_POLICIES.get(relative)
    if policy is None or stage not in policy["mutable_receipt_stages"]:
        return "immutable", None
    return "superseded_by_stage", str(policy["successor_stage"])


def _check_stage_markers(state: _AuditState) -> None:
    config_identity = _run_config_identity(state)
    receipts: dict[str, dict[Path, Mapping[str, Any]]] = {}
    for stage in FORMAL_STAGES:
        relative = f"logs/stages/{stage}/_SUCCESS.json"
        marker_path = state.root / relative
        marker = state.json(relative)
        if marker is None:
            continue
        if marker.get("stage") != stage or marker.get("status") != "SUCCESS":
            state.fail("invalid_stage_marker", "stage marker does not attest this stage as SUCCESS", path=marker_path)
        if (
            config_identity is None
            or marker.get("semantic_config_sha256") != config_identity
        ):
            state.fail(
                "stage_marker_config_identity_mismatch",
                "stage marker is not bound to the frozen semantic configuration",
                path=marker_path,
            )
        outputs = marker.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            state.fail("stage_outputs_missing", "stage marker requires a non-empty outputs list", path=marker_path)
            continue
        expected_receipt = (
            state.root / "logs" / "stages" / stage / "outputs_at_completion.json"
        )
        if len(outputs) != 1 or not isinstance(outputs[0], Mapping):
            state.fail(
                "stage_receipt_marker_invalid",
                "stage marker must hash exactly its outputs_at_completion receipt",
                path=marker_path,
            )
            continue
        marker_descriptor = outputs[0]
        receipt_path = state.resolve_run_path(
            marker_descriptor.get("path"), source=marker_path
        )
        marker_sha = str(marker_descriptor.get("sha256", "")).lower()
        marker_size = marker_descriptor.get("size_bytes")
        if (
            receipt_path != expected_receipt
            or not _SHA256.fullmatch(marker_sha)
            or not isinstance(marker_size, int)
            or marker_size < 0
        ):
            state.fail(
                "stage_receipt_marker_invalid",
                "stage marker receipt path, SHA-256, or size is invalid",
                path=marker_path,
            )
            continue
        if receipt_path.is_symlink() or not receipt_path.is_file():
            if not receipt_path.exists():
                state.missing_paths.add(str(receipt_path))
            state.fail(
                "stage_receipt_missing",
                "outputs_at_completion receipt is not a regular file",
                path=receipt_path,
            )
            continue
        actual_receipt_size = receipt_path.stat().st_size
        actual_receipt_sha = _sha256(receipt_path)
        if actual_receipt_size != marker_size or actual_receipt_sha != marker_sha:
            state.fail(
                "stage_receipt_integrity_mismatch",
                "outputs_at_completion receipt no longer matches the stage marker",
                path=receipt_path,
                details={
                    "expected_size_bytes": marker_size,
                    "actual_size_bytes": actual_receipt_size,
                    "expected_sha256": marker_sha,
                    "actual_sha256": actual_receipt_sha,
                },
            )
            continue
        receipt = state.json(receipt_path.relative_to(state.root).as_posix())
        if receipt is None:
            continue
        if (
            receipt.get("schema_version") != 1
            or receipt.get("stage") != stage
            or receipt.get("status") != "CAPTURED_AT_STAGE_COMPLETION"
        ):
            state.fail(
                "stage_receipt_invalid",
                "outputs_at_completion receipt has invalid stage/status/schema",
                path=receipt_path,
            )
        if (
            config_identity is None
            or receipt.get("semantic_config_sha256") != config_identity
            or receipt.get("semantic_config_sha256")
            != marker.get("semantic_config_sha256")
        ):
            state.fail(
                "stage_receipt_config_identity_mismatch",
                "stage receipt is not bound to the marker and frozen configuration",
                path=receipt_path,
            )
        declared = receipt.get("declared_outputs")
        if not isinstance(declared, list) or not declared:
            state.fail(
                "stage_receipt_outputs_missing",
                "stage receipt requires a non-empty declared_outputs list",
                path=receipt_path,
            )
            continue
        stage_evidence: dict[Path, Mapping[str, Any]] = {}
        for index, descriptor in enumerate(declared):
            if not isinstance(descriptor, Mapping):
                state.fail(
                    "stage_receipt_output_evidence_missing",
                    "each receipt output must be an object",
                    path=receipt_path,
                    details={"index": index},
                )
                continue
            output = state.resolve_run_path(descriptor.get("path"), source=receipt_path)
            if output is None:
                continue
            expected_sha = str(
                descriptor.get("sha256_at_stage_completion", "")
            ).lower()
            expected_size = descriptor.get("size_bytes_at_stage_completion")
            if (
                not _SHA256.fullmatch(expected_sha)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                state.fail(
                    "stage_receipt_output_evidence_missing",
                    "receipt output requires stage-completion SHA-256 and size",
                    path=output,
                )
                continue
            if output in stage_evidence:
                state.fail(
                    "duplicate_stage_receipt_output",
                    "stage receipt repeats a declared output",
                    path=output,
                )
                continue
            expected_policy, expected_successor = _expected_stage_output_policy(
                state, stage, output
            )
            if (
                descriptor.get("mutation_policy") != expected_policy
                or descriptor.get("superseded_by_stage") != expected_successor
            ):
                state.fail(
                    "stage_output_mutation_policy_invalid",
                    "stage output mutation policy is not the exact audited stage/path policy",
                    path=output,
                    details={
                        "stage": stage,
                        "expected_policy": expected_policy,
                        "expected_successor_stage": expected_successor,
                    },
                )
            if expected_policy == "immutable_snapshot":
                output_relative = output.relative_to(state.root).as_posix()
                expected_snapshot = state.root / str(
                    IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES[
                        (stage, output_relative)
                    ]
                )
                snapshot = state.resolve_run_path(
                    descriptor.get("snapshot_path"), source=receipt_path
                )
                snapshot_sha = str(descriptor.get("snapshot_sha256", "")).lower()
                snapshot_size = descriptor.get("snapshot_size_bytes")
                if (
                    snapshot != expected_snapshot
                    or snapshot_sha != expected_sha
                    or snapshot_size != expected_size
                    or snapshot.is_symlink()
                    or not snapshot.is_file()
                ):
                    state.fail(
                        "stage_output_snapshot_invalid",
                        "mutable final output lacks its exact immutable stage snapshot",
                        path=expected_snapshot,
                        details={"stage": stage, "output": str(output)},
                    )
                else:
                    actual_snapshot_size = snapshot.stat().st_size
                    actual_snapshot_sha = _sha256(snapshot)
                    if (
                        actual_snapshot_size != snapshot_size
                        or actual_snapshot_sha != snapshot_sha
                    ):
                        state.fail(
                            "stage_output_snapshot_integrity_mismatch",
                            "immutable stage snapshot no longer matches its receipt",
                            path=snapshot,
                        )
                    state.stage_outputs.add(snapshot)
                    state.critical_files.add(snapshot)
            stage_evidence[output] = descriptor
        receipts[stage] = stage_evidence
        state.stage_outputs.add(receipt_path)
        state.critical_files.add(receipt_path)

    for stage, stage_evidence in receipts.items():
        for output, descriptor in stage_evidence.items():
            if output.is_symlink() or not output.is_file():
                if not output.exists():
                    state.missing_paths.add(str(output))
                state.fail(
                    "stage_declared_output_missing",
                    "receipt-declared stage output is not a regular file",
                    path=output,
                    details={"stage": stage},
                )
                continue
            expected_sha = str(
                descriptor["sha256_at_stage_completion"]
            ).lower()
            expected_size = int(descriptor["size_bytes_at_stage_completion"])
            actual_size = output.stat().st_size
            actual_sha = _sha256(output)
            if actual_size != expected_size or actual_sha != expected_sha:
                policy, successor_stage = _expected_stage_output_policy(
                    state, stage, output
                )
                successor = (
                    None
                    if successor_stage is None
                    else receipts.get(successor_stage, {}).get(output)
                )
                successor_sha = (
                    ""
                    if successor is None
                    else str(
                        successor.get("sha256_at_stage_completion", "")
                    ).lower()
                )
                successor_size = (
                    None
                    if successor is None
                    else successor.get("size_bytes_at_stage_completion")
                )
                snapshot_permits_final_rewrite = policy == "immutable_snapshot"
                successor_matches = bool(
                    policy == "superseded_by_stage"
                    and successor is not None
                    and successor_sha == actual_sha
                    and successor_size == actual_size
                )
                if not snapshot_permits_final_rewrite and not successor_matches:
                    state.fail(
                        "stage_declared_output_integrity_mismatch",
                        "receipt-declared output changed without an exact matching successor receipt",
                        path=output,
                        details={
                            "stage": stage,
                            "expected_size_bytes": expected_size,
                            "actual_size_bytes": actual_size,
                            "expected_sha256": expected_sha,
                            "actual_sha256": actual_sha,
                            "required_successor_stage": successor_stage,
                            "successor_declared": successor is not None,
                        },
                    )
            state.stage_outputs.add(output)
            state.critical_files.add(output)


def _load_experiment_manifests(state: _AuditState) -> list[tuple[Path, Mapping[str, Any]]]:
    directory = state.root / "manifests" / "experiments"
    if not directory.is_dir() or directory.is_symlink():
        state.missing("manifests/experiments", "experiment_manifest_directory_missing")
        return []
    paths = sorted(directory.glob("*.json"))
    if not paths:
        state.fail("experiment_manifests_missing", "no experiment manifests were found", path=directory)
        return []
    rows: list[tuple[Path, Mapping[str, Any]]] = []
    for path in paths:
        relative = path.relative_to(state.root).as_posix()
        value = state.json(relative)
        if value is None:
            continue
        rows.append((path, value))
        if value.get("status") == "FAILED":
            state.fail("failed_experiment_manifest", "a FAILED experiment manifest exists", path=path)
        if value.get("status") == "COMPLETE":
            artifacts = value.get("artifacts")
            artifact_sha256 = value.get("artifact_sha256")
            if not isinstance(artifacts, list) or not artifacts:
                state.fail("complete_manifest_artifacts_missing", "COMPLETE manifest has no artifacts", path=path)
                continue
            if len(artifacts) != len(set(map(str, artifacts))):
                state.fail(
                    "complete_manifest_artifacts_duplicate",
                    "COMPLETE manifest repeats an artifact path",
                    path=path,
                )
            if not isinstance(artifact_sha256, Mapping):
                state.fail(
                    "complete_manifest_artifact_hashes_missing",
                    "COMPLETE manifest has no artifact_sha256 mapping",
                    path=path,
                )
                continue
            artifact_keys = set(map(str, artifacts))
            hash_keys = set(map(str, artifact_sha256))
            if artifact_keys != hash_keys:
                state.fail(
                    "complete_manifest_artifact_hash_set_mismatch",
                    "COMPLETE manifest artifacts and artifact_sha256 keys differ",
                    path=path,
                    details={
                        "missing_hashes": sorted(artifact_keys - hash_keys),
                        "unexpected_hashes": sorted(hash_keys - artifact_keys),
                    },
                )
            for raw in artifacts:
                raw_key = str(raw)
                artifact = state.resolve_run_path(raw_key, source=path)
                if artifact is None:
                    continue
                expected_sha = str(artifact_sha256.get(raw_key, "")).lower()
                if not _SHA256.fullmatch(expected_sha):
                    state.fail(
                        "manifest_artifact_hash_invalid",
                        "COMPLETE manifest artifact has no valid SHA-256",
                        path=artifact,
                    )
                if artifact.is_symlink() or not artifact.is_file():
                    if not artifact.exists():
                        state.missing_paths.add(str(artifact))
                    state.fail("manifest_artifact_missing", "COMPLETE manifest artifact is missing", path=artifact)
                    continue
                actual_sha = _sha256(artifact)
                if expected_sha != actual_sha:
                    state.fail(
                        "manifest_artifact_integrity_mismatch",
                        "COMPLETE manifest artifact does not match artifact_sha256",
                        path=artifact,
                        details={
                            "expected_sha256": expected_sha,
                            "actual_sha256": actual_sha,
                        },
                    )
                state.manifest_artifacts.add(artifact)
                state.critical_files.add(artifact)
    return rows


def _check_manifest_coverage(
    state: _AuditState,
    manifests: Sequence[tuple[Path, Mapping[str, Any]]],
    methods: tuple[str, ...],
    datasets: tuple[DatasetExpectation, ...],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
) -> None:
    for dataset in datasets:
        dataset_rows = [(path, row) for path, row in manifests if row.get("dataset") == dataset.key]
        for path, row in dataset_rows:
            if row.get("route") != dataset.route or row.get("pool") != dataset.pool:
                state.fail(
                    "route_pool_mismatch",
                    "manifest route/pool disagrees with the frozen dataset expectation",
                    path=path,
                )
        for method in methods:
            rows = [
                (path, row)
                for path, row in dataset_rows
                if row.get("stage") == "train" and row.get("method") == method
            ]
            if not rows:
                state.fail(
                    "method_coverage_missing",
                    f"no train manifest for {dataset.key}/{method}",
                )
                continue
            learned_values = {
                row.get("spec", {}).get("learned")
                for _, row in rows
                if isinstance(row.get("spec"), Mapping)
            }
            if learned_values not in ({True}, {False}):
                state.fail(
                    "learned_status_ambiguous",
                    f"manifests do not consistently declare spec.learned for {dataset.key}/{method}",
                )
                continue
            learned = learned_values == {True}
            expected = (
                {(seed, fold) for seed in seeds for fold in folds}
                if learned
                else {(-1, -1)}
            )
            complete_coordinates = [
                (int(row.get("seed", -999)), int(row.get("fold", -999)))
                for _, row in rows
                if row.get("status") == "COMPLETE"
            ]
            actual = set(complete_coordinates)
            if len(actual) != len(complete_coordinates):
                state.fail("duplicate_coverage_coordinate", f"duplicate COMPLETE coordinate for {dataset.key}/{method}")
            if actual != expected:
                state.fail(
                    "method_coverage_incomplete",
                    f"incomplete seed/fold coverage for {dataset.key}/{method}",
                    details={
                        "missing": sorted(expected - actual),
                        "unexpected": sorted(actual - expected),
                    },
                )

        vlm = [row for _, row in dataset_rows if row.get("method") == "r11_vlm_candidate_judge"]
        if len(vlm) != 1:
            state.fail("r11_evidence_missing", f"expected one R11 status for {dataset.key}")
        elif vlm[0].get("status") not in {
            "COMPLETE",
            "NOT_RUN_CREDENTIAL_OR_BILLING_REQUIRED",
        }:
            state.fail("r11_status_invalid", f"invalid R11 status for {dataset.key}: {vlm[0].get('status')}")


def _parse_checksum_lines(
    state: _AuditState,
    path: Path,
    *,
    allow_external: bool,
) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256.fullmatch(parts[0].lower()):
            state.fail("checksum_manifest_malformed", "malformed SHA-256 manifest line", path=path, details={"line": line_number})
            continue
        name = parts[1].lstrip("* ")
        candidate = Path(name)
        if not candidate.is_absolute():
            local = Path(os.path.abspath(state.root / candidate))
            repository = Path(__file__).resolve().parents[1] / candidate
            candidate = local if local.exists() or not allow_external else repository
        candidate = Path(os.path.abspath(candidate))
        if not allow_external and candidate != state.root and state.root not in candidate.parents:
            state.fail("checksum_path_escape", "checksum entry escapes the run root", path=candidate)
            continue
        if candidate in entries:
            state.fail("checksum_duplicate_path", "checksum manifest contains a duplicate path", path=candidate)
            continue
        entries[candidate] = parts[0].lower()
        if candidate.is_symlink() or not candidate.is_file():
            if not candidate.exists():
                state.missing_paths.add(str(candidate))
            state.fail("checksummed_file_missing", "checksummed file is not a regular file", path=candidate)
        elif _sha256(candidate) != entries[candidate]:
            state.fail("checksum_mismatch", "file SHA-256 does not match checksum manifest", path=candidate)
    if not entries:
        state.fail("checksum_manifest_empty", "checksum manifest contains no entries", path=path)
    return entries


def _check_canonical_and_features(
    state: _AuditState,
    datasets: tuple[DatasetExpectation, ...],
) -> None:
    input_checksums = state.regular_file("manifests/input_checksums.sha256")
    input_entries = {} if input_checksums is None else _parse_checksum_lines(state, input_checksums, allow_external=True)
    input_hashes = set(input_entries.values())
    for route in sorted({item.route for item in datasets}):
        relative = f"manifests/canonical_{route}_run.json"
        manifest = state.json(relative)
        if manifest is None:
            continue
        if manifest.get("route") != route:
            state.fail("canonical_route_mismatch", f"canonical manifest route is not {route}", path=state.root / relative)
        hashes = {
            str(value).lower()
            for key, value in manifest.items()
            if str(key).endswith("sha256") and _SHA256.fullmatch(str(value).lower())
        }
        if not hashes:
            state.fail("canonical_checksum_missing", "canonical manifest contains no SHA-256", path=state.root / relative)
        elif input_hashes and not hashes & input_hashes:
            state.fail("canonical_checksum_unlinked", "canonical manifest hashes are absent from input_checksums.sha256", path=state.root / relative)

    baseline = state.json("audit/baseline_recomputation.json")
    if baseline is not None:
        for route in {item.route for item in datasets}:
            metrics = baseline.get(route)
            if not isinstance(metrics, Mapping):
                state.fail("baseline_recomputation_missing", f"independent baseline result missing for route {route}")
            elif not all(key in metrics for key in ("query_count", "candidate_total", "q_only_j_at_1", "oracle")):
                state.fail("baseline_recomputation_incomplete", f"baseline result lacks required metrics for route {route}")

    state.regular_file("audit/LEAKAGE_AUDIT.md")

    feature_audit = state.json("audit/feature_audit.json")
    audits = feature_audit.get("audits") if feature_audit is not None else None
    if feature_audit is not None and not isinstance(audits, list):
        state.fail("feature_audit_invalid", "feature audit must contain an audits list")
        audits = []
    pool_status = state.json("data/candidate_pool_status.json")
    pools = pool_status.get("pools") if pool_status is not None else None
    if pool_status is not None and not isinstance(pools, list):
        state.fail("candidate_pool_status_invalid", "candidate pool status must contain a pools list")
        pools = []
    for dataset in datasets:
        matches = [
            item for item in (audits or [])
            if isinstance(item, Mapping)
            and item.get("route") == dataset.route
            and item.get("pool") == dataset.pool
        ]
        if len(matches) != 1:
            state.fail("feature_audit_coverage_missing", f"feature audit missing for {dataset.route}/{dataset.pool}")
        else:
            row = matches[0]
            schema = row.get("feature_schema")
            if row.get("forbidden_column_scanner_passed") is not True or not isinstance(schema, Mapping) or not _SHA256.fullmatch(str(schema.get("feature_schema_sha256", ""))):
                state.fail("feature_audit_failed", f"feature audit is incomplete for {dataset.route}/{dataset.pool}")
            if int(row.get("candidate_count", 0)) <= 0 or int(row.get("query_count", 0)) <= 0:
                state.fail("feature_table_empty", f"feature table is empty for {dataset.route}/{dataset.pool}")
        available = [
            item for item in (pools or [])
            if isinstance(item, Mapping)
            and item.get("route") == dataset.route
            and item.get("pool") == dataset.pool
            and item.get("status") == "AVAILABLE"
        ]
        if len(available) != 1:
            state.fail("route_pool_coverage_missing", f"route/pool is not recorded AVAILABLE: {dataset.route}/{dataset.pool}")
        for relative in (
            f"features/candidates_{dataset.key}_development.parquet",
            f"data/labels_{dataset.key}_development.parquet",
            f"features/candidates_{dataset.key}_test.parquet",
            f"data/labels_{dataset.key}_test.parquet",
            f"features/feature_schema_{dataset.key}.json",
        ):
            state.regular_file(relative)


def _check_modular_development_publication(state: _AuditState) -> None:
    """Re-verify the upstream Modular development publication and receipt."""

    relative = "manifests/modular_development_publication_verification.json"
    receipt_path = state.root / relative
    receipt = state.json(relative)
    if receipt is None:
        return
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "VERIFIED_BEFORE_FORMAL_CONSUMPTION"
        or receipt.get("independent_reconstruction_exact") is not True
    ):
        state.fail(
            "modular_development_publication_receipt_invalid",
            "Modular development publication receipt is not a completed exact reconstruction",
            path=receipt_path,
        )
        return
    raw_output = receipt.get("published_features_path")
    raw_manifest = receipt.get("publication_manifest_path")
    if not isinstance(raw_output, str) or not isinstance(raw_manifest, str):
        state.fail(
            "modular_development_publication_receipt_invalid",
            "Modular publication receipt lacks source paths",
            path=receipt_path,
        )
        return
    output_path = Path(os.path.abspath(Path(raw_output).expanduser()))
    manifest_path = Path(os.path.abspath(Path(raw_manifest).expanduser()))
    if (
        output_path.is_symlink()
        or manifest_path.is_symlink()
        or not output_path.is_file()
        or not manifest_path.is_file()
    ):
        state.fail(
            "modular_development_publication_source_missing",
            "verified Modular publication or companion manifest is missing/symlinked",
            path=output_path,
        )
        return
    expected_output_sha = str(receipt.get("published_features_sha256", ""))
    expected_manifest_sha = str(receipt.get("publication_manifest_sha256", ""))
    if (
        not _SHA256.fullmatch(expected_output_sha)
        or not _SHA256.fullmatch(expected_manifest_sha)
        or _sha256(output_path) != expected_output_sha
        or _sha256(manifest_path) != expected_manifest_sha
    ):
        state.fail(
            "modular_development_publication_source_changed",
            "Modular publication bytes changed after formal build-features",
            path=output_path,
        )
        return
    try:
        verified = verify_modular_development_publication(
            output_path, manifest_path
        )
    except PublishError as error:
        state.fail(
            "modular_development_publication_verification_failed",
            f"independent Modular publication reconstruction failed: {error}",
            path=output_path,
        )
        return
    if (
        verified.get("manifest_payload_sha256")
        != receipt.get("manifest_payload_sha256")
        or int(verified.get("rows", -1)) != int(receipt.get("rows", -2))
        or int(verified.get("queries", -1)) != int(receipt.get("queries", -2))
    ):
        state.fail(
            "modular_development_publication_receipt_mismatch",
            "Modular publication receipt disagrees with independently recomputed evidence",
            path=receipt_path,
        )


def _check_entity_leakage_bundle(state: _AuditState) -> None:
    """Verify machine-readable entity and concrete fit-partition evidence."""

    bundle_relative = "audit/leakage"
    expected_names = (
        "identity_rows.jsonl",
        "identity_overlaps.jsonl",
        "fold_fit_identities.jsonl",
        "fold_fit_overlaps.jsonl",
        "fit_resolution_errors.jsonl",
        "summary.json",
        "LEAKAGE_AUDIT.md",
        "artifacts.sha256",
    )
    for name in expected_names:
        state.regular_file(f"{bundle_relative}/{name}", nonempty=False)
    root_markdown = state.regular_file("audit/LEAKAGE_AUDIT.md")
    try:
        result = verify_leakage_audit_bundle(
            state.root / bundle_relative, require_fit_evidence=True
        )
    except LeakageAuditError as error:
        state.fail(
            "entity_leakage_bundle_invalid",
            f"entity leakage evidence failed independent verification: {error}",
            path=state.root / bundle_relative,
        )
        return
    if not result.passed:
        state.fail(
            "entity_leakage_audit_failed",
            "development/test or concrete fit/test source identities overlap",
            path=state.root / bundle_relative / "summary.json",
        )
    bundle_markdown = state.root / bundle_relative / "LEAKAGE_AUDIT.md"
    if (
        root_markdown is None
        or root_markdown.read_bytes() != bundle_markdown.read_bytes()
    ):
        state.fail(
            "entity_leakage_markdown_mismatch",
            "root LEAKAGE_AUDIT.md is not the verified bundle report",
            path=root_markdown or state.root / "audit/LEAKAGE_AUDIT.md",
        )
    lock_path = state.root / "manifests/PRIMARY_METHOD_LOCK.json"
    lock = state.json("manifests/PRIMARY_METHOD_LOCK.json")
    record = lock.get("leakage_audit") if lock is not None else None
    if not isinstance(record, Mapping):
        state.fail(
            "entity_leakage_lock_record_missing",
            "primary lock does not bind the verified leakage evidence",
            path=lock_path,
        )
        return
    if (
        record.get("status") != "PASS"
        or record.get("test_labels_opened") is not False
        or record.get("audit_digest_sha256")
        != result.summary.get("audit_digest_sha256")
        or record.get("canonical_markdown_sha256") != _sha256(bundle_markdown)
    ):
        state.fail(
            "entity_leakage_lock_record_mismatch",
            "primary lock leakage digest/status/label isolation is invalid",
            path=lock_path,
        )
    declared_bundle = record.get("bundle_artifacts")
    expected_bundle_paths = {
        str((state.root / bundle_relative / name).resolve())
        for name in expected_names
    } | {str((state.root / "audit/LEAKAGE_AUDIT.md").resolve())}
    observed_bundle_paths: set[str] = set()
    if not isinstance(declared_bundle, list):
        state.fail(
            "entity_leakage_lock_artifacts_invalid",
            "primary lock lacks leakage bundle artifact descriptors",
            path=lock_path,
        )
    else:
        for descriptor in declared_bundle:
            if not isinstance(descriptor, Mapping):
                state.fail(
                    "entity_leakage_lock_artifacts_invalid",
                    "leakage bundle artifact descriptor is malformed",
                    path=lock_path,
                )
                continue
            artifact = state.resolve_run_path(
                descriptor.get("path"), source=lock_path
            )
            expected_sha = str(descriptor.get("sha256", ""))
            expected_size = descriptor.get("size_bytes")
            if artifact is not None:
                observed_bundle_paths.add(str(artifact.resolve()))
            if (
                artifact is None
                or artifact.is_symlink()
                or not artifact.is_file()
                or not _SHA256.fullmatch(expected_sha)
                or not isinstance(expected_size, int)
                or artifact.stat().st_size != expected_size
                or _sha256(artifact) != expected_sha
            ):
                state.fail(
                    "entity_leakage_lock_artifact_mismatch",
                    "locked leakage artifact is missing or hash-invalid",
                    path=artifact or lock_path,
                )
            elif artifact is not None:
                state.critical_files.add(artifact)
        if observed_bundle_paths != expected_bundle_paths:
            state.fail(
                "entity_leakage_lock_artifact_set_mismatch",
                "primary lock leakage artifact set is not exact",
                path=lock_path,
            )
    fit_evidence = record.get("development_fit_evidence")
    expected_fit_coordinates = {
        (dataset, fold)
        for dataset in result.summary.get("development_dataset_counts", {})
        for fold in range(5)
    }
    observed_fit_coordinates: set[tuple[str, int]] = set()
    if not isinstance(fit_evidence, list):
        state.fail(
            "entity_leakage_fit_lock_missing",
            "primary lock lacks concrete outer-fit artifact descriptors",
            path=lock_path,
        )
    else:
        for descriptor in fit_evidence:
            if not isinstance(descriptor, Mapping):
                state.fail(
                    "entity_leakage_fit_lock_invalid",
                    "outer-fit descriptor is malformed",
                    path=lock_path,
                )
                continue
            try:
                coordinate = (
                    str(descriptor["dataset"]),
                    int(descriptor["outer_fold"]),
                )
            except (KeyError, TypeError, ValueError):
                state.fail(
                    "entity_leakage_fit_lock_invalid",
                    "outer-fit descriptor lacks dataset/fold coordinates",
                    path=lock_path,
                )
                continue
            observed_fit_coordinates.add(coordinate)
            artifact = state.resolve_run_path(
                descriptor.get("path"), source=lock_path
            )
            expected_sha = str(descriptor.get("sha256", ""))
            if (
                artifact is None
                or artifact.is_symlink()
                or not artifact.is_file()
                or not _SHA256.fullmatch(expected_sha)
                or _sha256(artifact) != expected_sha
                or int(descriptor.get("rows", 0)) <= 0
                or int(descriptor.get("queries", 0)) <= 0
                or int(descriptor.get("groups", 0)) <= 0
                or int(descriptor.get("candidates", 0)) <= 0
            ):
                state.fail(
                    "entity_leakage_fit_lock_invalid",
                    "locked outer-fit artifact is missing, empty, or hash-invalid",
                    path=artifact or lock_path,
                )
            elif artifact is not None:
                state.critical_files.add(artifact)
        if observed_fit_coordinates != expected_fit_coordinates:
            state.fail(
                "entity_leakage_fit_coordinate_mismatch",
                "locked outer-fit evidence does not cover every dataset x five folds",
                path=lock_path,
            )


def _locked_descriptor_path(
    state: _AuditState,
    descriptor: Any,
    *,
    source: Path,
    code: str,
) -> Path | None:
    if not isinstance(descriptor, Mapping):
        state.fail(code, "locked artifact descriptor is missing or malformed", path=source)
        return None
    path = state.resolve_run_path(descriptor.get("path"), source=source)
    expected_sha = descriptor.get("sha256")
    expected_size = descriptor.get("size_bytes")
    if (
        path is None
        or not _SHA256.fullmatch(str(expected_sha or ""))
        or not isinstance(expected_size, int)
    ):
        state.fail(code, "locked artifact descriptor lacks path/hash/size", path=source)
        return None
    if path.is_symlink() or not path.is_file():
        state.fail(code, "locked artifact is missing or not a regular file", path=path)
        if not path.exists():
            state.missing_paths.add(str(path))
        return None
    actual_sha = _sha256(path)
    actual_size = path.stat().st_size
    if actual_sha != expected_sha or actual_size != expected_size:
        state.fail(
            code,
            "locked artifact changed after primary selection",
            path=path,
            details={
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
            },
        )
        return None
    state.critical_files.add(path)
    return path


def _check_primary_training_artifact_lock(
    state: _AuditState,
    primary: Mapping[str, Any],
    manifests: Sequence[tuple[Path, Mapping[str, Any]]],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
    *,
    lock_path: Path,
) -> None:
    dataset = str(primary.get("dataset", ""))
    method = str(primary.get("method", ""))
    inventory = primary.get("training_artifact_lock")
    if not isinstance(inventory, Mapping):
        state.fail(
            "primary_training_lock_missing",
            f"primary lock lacks training-artifact inventory for {dataset}/{method}",
            path=lock_path,
        )
        return
    train_rows = [
        (path, row)
        for path, row in manifests
        if row.get("status") == "COMPLETE"
        and row.get("stage") == "train"
        and row.get("dataset") == dataset
        and row.get("method") == method
    ]
    learned = {
        row.get("spec", {}).get("learned")
        for _, row in train_rows
        if isinstance(row.get("spec"), Mapping)
    } == {True}
    expected_coordinates = (
        {(seed, fold) for seed in seeds for fold in folds}
        if learned
        else {(-1, -1)}
    )
    actual_coordinates = {
        (int(row.get("seed", -999)), int(row.get("fold", -999)))
        for _, row in train_rows
    }
    if actual_coordinates != expected_coordinates:
        state.fail(
            "primary_training_lock_coverage_mismatch",
            f"selected training coordinates changed for {dataset}/{method}",
        )
    locked_rows = inventory.get("manifests")
    if (
        not isinstance(locked_rows, list)
        or inventory.get("manifest_count") != len(locked_rows)
        or len(locked_rows) != len(train_rows)
    ):
        state.fail(
            "primary_training_lock_manifest_count",
            f"locked training manifest count is invalid for {dataset}/{method}",
            path=lock_path,
        )
        return
    by_id = {str(row.get("experiment_id")): (path, row) for path, row in train_rows}
    if len(by_id) != len(train_rows):
        state.fail(
            "primary_training_lock_duplicate_identity",
            f"selected training experiment IDs are not unique for {dataset}/{method}",
        )
        return
    seen: set[str] = set()
    for locked in locked_rows:
        if not isinstance(locked, Mapping):
            state.fail("primary_training_lock_invalid", "locked training entry is malformed", path=lock_path)
            continue
        experiment_id = str(locked.get("experiment_id", ""))
        seen.add(experiment_id)
        source = by_id.get(experiment_id)
        if source is None:
            state.fail("primary_training_manifest_changed", f"locked COMPLETE manifest disappeared: {experiment_id}", path=lock_path)
            continue
        manifest_path, manifest = source
        locked_manifest_path = _locked_descriptor_path(
            state,
            locked.get("manifest"),
            source=lock_path,
            code="primary_training_manifest_integrity",
        )
        if locked_manifest_path != manifest_path:
            state.fail("primary_training_manifest_path_mismatch", f"locked manifest path changed: {experiment_id}", path=manifest_path)
        identity = str(locked.get("experiment_identity_sha256", ""))
        if (
            not _SHA256.fullmatch(identity)
            or identity != manifest.get("experiment_identity_sha256")
            or int(locked.get("seed", -999)) != int(manifest.get("seed", -998))
            or int(locked.get("fold", -999)) != int(manifest.get("fold", -998))
        ):
            state.fail("primary_training_experiment_identity_mismatch", f"locked experiment identity changed: {experiment_id}", path=manifest_path)
        artifacts = locked.get("artifacts")
        manifest_artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not isinstance(manifest_artifacts, list):
            state.fail("primary_training_artifact_inventory_invalid", "locked artifact list is malformed", path=manifest_path)
            continue
        locked_artifact_paths: set[str] = set()
        for descriptor in artifacts:
            artifact_path = _locked_descriptor_path(
                state,
                descriptor,
                source=manifest_path,
                code="primary_training_artifact_integrity",
            )
            if artifact_path is not None:
                locked_artifact_paths.add(str(artifact_path))
        expected_artifact_paths: set[str] = set()
        for raw in manifest_artifacts:
            resolved = state.resolve_run_path(raw, source=manifest_path)
            if resolved is not None:
                expected_artifact_paths.add(str(resolved))
        if locked_artifact_paths != expected_artifact_paths:
            state.fail("primary_training_artifact_inventory_mismatch", f"locked artifact set changed: {experiment_id}", path=manifest_path)
        if learned:
            bundle_path = _locked_descriptor_path(
                state,
                locked.get("bundle"),
                source=manifest_path,
                code="primary_training_bundle_integrity",
            )
            if bundle_path is None or str(bundle_path) != str(manifest.get("bundle_path")):
                state.fail("primary_training_bundle_path_mismatch", f"locked bundle path changed: {experiment_id}", path=manifest_path)
                continue
            try:
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                state.fail("primary_training_bundle_invalid", "locked bundle is unreadable", path=bundle_path)
                continue
            preprocessor = bundle.get("preprocessor")
            if (
                not isinstance(preprocessor, Mapping)
                or locked.get("preprocessor_storage") != "embedded_in_bundle"
                or locked.get("preprocessor_sha256") != _canonical_mapping_sha256(preprocessor)
            ):
                state.fail("primary_training_preprocessor_mismatch", f"locked preprocessing state changed: {experiment_id}", path=bundle_path)
            roles = locked.get("bundle_artifacts")
            role_fields = {
                "checkpoint": ("checkpoint", "checkpoint_sha256"),
                "score_calibrator": ("score_calibrator", "score_calibrator_sha256"),
                "prediction_calibrator": ("prediction_calibrator", "prediction_calibrator_sha256"),
            }
            if not isinstance(roles, Mapping):
                state.fail("primary_training_bundle_artifacts_missing", "locked bundle roles are missing", path=bundle_path)
            else:
                for role, (path_key, hash_key) in role_fields.items():
                    role_descriptor = roles.get(role)
                    role_path = _locked_descriptor_path(
                        state,
                        role_descriptor,
                        source=bundle_path,
                        code="primary_training_bundle_artifact_integrity",
                    )
                    if (
                        role_path is None
                        or str(role_path) != str(bundle.get(path_key))
                        or not isinstance(role_descriptor, Mapping)
                        or role_descriptor.get("sha256") != bundle.get(hash_key)
                    ):
                        state.fail("primary_training_bundle_artifact_mismatch", f"locked bundle {role} changed: {experiment_id}", path=bundle_path)
    if seen != set(by_id):
        state.fail("primary_training_lock_manifest_set_mismatch", f"locked COMPLETE manifest set changed for {dataset}/{method}", path=lock_path)


def _check_held_out_input_lock(
    state: _AuditState,
    lock: Mapping[str, Any],
    datasets: tuple[DatasetExpectation, ...],
    *,
    lock_path: Path,
) -> None:
    rows = lock.get("held_out_test_inputs")
    if not isinstance(rows, list):
        state.fail("held_out_input_lock_missing", "primary lock lacks held-out input identities", path=lock_path)
        return
    by_dataset = {
        str(row.get("dataset")): row for row in rows if isinstance(row, Mapping)
    }
    if set(by_dataset) != {dataset.key for dataset in datasets}:
        state.fail("held_out_input_lock_coverage", "held-out input lock has wrong dataset coverage", path=lock_path)
    for dataset in datasets:
        row = by_dataset.get(dataset.key)
        if not isinstance(row, Mapping):
            continue
        if (
            row.get("route") != dataset.route
            or row.get("pool") != dataset.pool
            or row.get("labels_opened_at_lock") is not False
        ):
            state.fail("held_out_input_lock_metadata", f"held-out lock metadata invalid for {dataset.key}", path=lock_path)
        feature_path = _locked_descriptor_path(
            state, row.get("features"), source=lock_path, code="held_out_feature_integrity"
        )
        query_path = _locked_descriptor_path(
            state, row.get("query_universe"), source=lock_path, code="held_out_query_universe_integrity"
        )
        if feature_path is None or query_path is None:
            continue
        try:
            features = pd.read_parquet(feature_path)
            universe = pd.read_parquet(query_path)
        except Exception as error:
            state.fail("held_out_input_unreadable", f"cannot read held-out identity parquet: {error}")
            continue
        schema = {
            "columns": [
                {"name": str(column), "dtype": str(features[column].dtype)}
                for column in features.columns
            ]
        }
        if (
            row.get("feature_schema") != schema
            or row.get("feature_schema_sha256") != _canonical_mapping_sha256(schema)
        ):
            state.fail("held_out_feature_schema_mismatch", f"held-out feature schema changed for {dataset.key}")
        pool_columns = row.get("candidate_identity_columns")
        query_columns = row.get("query_universe_identity_columns")
        if (
            not isinstance(pool_columns, list)
            or len(pool_columns) < 3
            or not set(pool_columns).issubset(features.columns)
            or not isinstance(query_columns, list)
            or not query_columns
            or not set(query_columns).issubset(universe.columns)
        ):
            state.fail("held_out_identity_columns_invalid", f"held-out identity columns invalid for {dataset.key}")
            continue
        candidate_keys = list(pool_columns[:2])
        features = features.copy()
        features[candidate_keys[0]] = features[candidate_keys[0]].astype(str)
        features[candidate_keys[1]] = features[candidate_keys[1]].astype(str)
        features["q_raw"] = pd.to_numeric(features["q_raw"], errors="coerce")
        if (
            features.duplicated(candidate_keys).any()
            or features["q_raw"].isna().any()
            or not np.isfinite(features["q_raw"].to_numpy(np.float64)).all()
        ):
            state.fail("held_out_candidate_identity_invalid", f"held-out candidate identity is invalid for {dataset.key}")
            continue
        canonical_features = features.sort_values(candidate_keys, kind="mergesort")
        universe = universe.copy()
        for column in query_columns:
            universe[column] = universe[column].astype(str)
        if universe[query_columns[0]].duplicated().any():
            state.fail("held_out_query_identity_duplicate", f"held-out query IDs are duplicated for {dataset.key}")
            continue
        canonical_universe = universe.sort_values(query_columns[0], kind="mergesort")
        if (
            row.get("candidate_count") != len(features)
            or row.get("candidate_pool_identity_sha256")
            != _canonical_frame_sha256(canonical_features, pool_columns)
        ):
            state.fail("held_out_candidate_pool_mismatch", f"held-out candidate identity changed for {dataset.key}")
        if (
            row.get("query_universe_count") != len(universe)
            or row.get("query_universe_identity_sha256")
            != _canonical_frame_sha256(canonical_universe, query_columns)
        ):
            state.fail("held_out_query_universe_mismatch", f"held-out query universe changed for {dataset.key}")


def _check_test_protocol(
    state: _AuditState,
    manifests: Sequence[tuple[Path, Mapping[str, Any]]],
    datasets: tuple[DatasetExpectation, ...],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
) -> None:
    lock_path = state.root / "manifests/PRIMARY_METHOD_LOCK.json"
    lock = state.json("manifests/PRIMARY_METHOD_LOCK.json")
    if lock is None:
        return
    if (
        lock.get("status") != "LOCKED_BEFORE_TEST_PRIMARY"
        or lock.get("test_inputs_materialized_before_lock") is not True
        or lock.get("test_inputs_used_for_primary_selection") is not False
        or lock.get("test_predictions_generated_before_lock") is not False
        or lock.get("test_prediction_artifacts_at_lock") != []
        or lock.get("test_feature_identity_frozen_before_prediction") is not True
        or lock.get("test_labels_opened_at_lock") is not False
    ):
        state.fail("primary_lock_invalid", "primary lock does not attest a pre-test lock", path=lock_path)
    lock_time = _iso8601(lock.get("locked_at"))
    if lock_time is None:
        state.fail("primary_lock_timestamp_invalid", "primary lock has no valid locked_at timestamp", path=lock_path)
    if not _SHA256.fullmatch(str(lock.get("evaluator_sha256", ""))):
        state.fail("primary_lock_evaluator_hash_missing", "primary lock lacks evaluator SHA-256", path=lock_path)
    if not str(lock.get("git_commit", "")).strip() or lock.get("git_commit") == "UNKNOWN":
        state.fail("primary_lock_git_missing", "primary lock lacks a concrete git commit", path=lock_path)
    if not _SHA256.fullmatch(str(lock.get("reranking_source_tree_sha256", ""))):
        state.fail(
            "primary_lock_source_hash_missing",
            "primary lock lacks the exact reranking source-tree SHA-256",
            path=lock_path,
        )
    elif lock.get("reranking_source_tree_sha256") != _reranking_source_tree_sha256():
        state.fail(
            "primary_lock_source_hash_mismatch",
            "reranking source tree changed after the primary lock",
            path=lock_path,
        )
    if not isinstance(lock.get("git_worktree_dirty"), bool):
        state.fail(
            "primary_lock_git_state_missing",
            "primary lock does not record whether reranking sources were dirty",
            path=lock_path,
        )
    primaries = lock.get("primaries")
    if not isinstance(primaries, list):
        state.fail("primary_lock_invalid", "primary lock lacks primaries list", path=lock_path)
        return
    by_dataset = {str(row.get("dataset")): row for row in primaries if isinstance(row, Mapping)}
    if set(by_dataset) != {item.key for item in datasets}:
        state.fail("primary_lock_dataset_coverage", "primary lock does not cover exactly the expected datasets", path=lock_path)
    _check_held_out_input_lock(state, lock, datasets, lock_path=lock_path)
    required_primary = {
        "route",
        "pool",
        "feature_schema_sha256",
        "model_class",
        "exact_hyperparameters",
        "checkpoint_selection_rule",
        "folds",
        "seeds",
        "score_calibration",
        "gate_type",
        "gate_thresholds",
        "q_floor",
        "candidate_manifest_sha256",
        "selection_rationale",
    }
    for dataset in datasets:
        primary = by_dataset.get(dataset.key)
        if not isinstance(primary, Mapping):
            continue
        missing = sorted(required_primary - set(primary))
        if missing:
            state.fail("primary_lock_fields_missing", f"primary lock fields missing for {dataset.key}", details={"missing": missing})
        if primary.get("route") != dataset.route or primary.get("pool") != dataset.pool:
            state.fail("primary_lock_route_pool_mismatch", f"primary lock route/pool mismatch for {dataset.key}")
        if str(primary.get("method", "")).startswith("r10_"):
            state.fail(
                "r10_primary_selection_forbidden",
                f"historical R10 method was selected as primary for {dataset.key}",
                path=lock_path,
            )
        if primary.get("folds") != len(folds) or tuple(primary.get("seeds", ())) != seeds:
            state.fail("primary_lock_protocol_mismatch", f"primary lock fold/seed protocol mismatch for {dataset.key}")
        for key in ("feature_schema_sha256", "candidate_manifest_sha256"):
            if not _SHA256.fullmatch(str(primary.get(key, ""))):
                state.fail("primary_lock_hash_missing", f"primary lock lacks valid {key} for {dataset.key}")
        if not isinstance(primary.get("gate_thresholds"), Mapping):
            state.fail("primary_lock_gate_thresholds_missing", f"primary lock lacks machine-readable gate thresholds for {dataset.key}")
        _check_primary_training_artifact_lock(
            state,
            primary,
            manifests,
            seeds,
            folds,
            lock_path=lock_path,
        )

        selected_tests = [
            row for _, row in manifests
            if row.get("dataset") == dataset.key
            and row.get("stage") == "test-primary"
            and row.get("status") == "COMPLETE"
            and row.get("method") == primary.get("method")
            and row.get("gate") == primary.get("gate")
            and row.get("prediction_designation") == "LOCKED_PRIMARY_TEST"
        ]
        baseline_tests = [
            row for _, row in manifests
            if row.get("dataset") == dataset.key
            and row.get("stage") == "test-primary"
            and row.get("status") == "COMPLETE"
            and row.get("prediction_designation") == "LOCKED_PRIMARY_TEST_BASELINE"
        ]
        if len(selected_tests) != 1 or len(baseline_tests) != 1:
            state.fail("primary_test_coverage_missing", f"baseline/locked primary test evidence incomplete for {dataset.key}")
        for row in (*selected_tests, *baseline_tests):
            completed = _iso8601(row.get("completed_at"))
            if lock_time is None or completed is None or completed < lock_time:
                state.fail("primary_lock_not_before_test", f"test completion does not follow the primary lock for {dataset.key}")

    post = state.json("metrics/post_lock_test_results.json")
    if post is not None and (
        post.get("designation") != "POST_LOCK_COMPARATIVE_ONLY"
        or post.get("primary_reselection_permitted") is not False
    ):
        state.fail("post_lock_designation_invalid", "post-lock comparisons are not irreversibly marked comparative only")
    for path, row in manifests:
        if row.get("stage") != "test-post-lock" or row.get("status") != "COMPLETE":
            continue
        designation = row.get("prediction_designation")
        if designation == "POST_LOCK_R10_EXISTING_COMPARISON":
            if (
                not str(row.get("method", "")).startswith("r10_")
                or row.get("gate") != "FROZEN_EXISTING"
            ):
                state.fail(
                    "r10_post_lock_manifest_invalid",
                    "R10 post-lock manifest is not bound to a frozen existing method",
                    path=path,
                )
        elif designation != "POST_LOCK_COMPARATIVE_ONLY":
            state.fail("post_lock_manifest_designation_invalid", "test-post-lock manifest lacks comparative-only designation", path=path)


_R10_REQUIRED_DISCOVERY_CHECKS = frozenset(
    {
        "manifest",
        "complete_oof",
        "checkpoint",
        "candidate_hash",
        "evaluator_hash",
        "independent_recomputation",
    }
)


def _r10_artifact_descriptor(
    state: _AuditState,
    descriptor: Any,
    *,
    source: Path,
    role: str,
) -> Path | None:
    if not isinstance(descriptor, Mapping):
        state.fail(
            "r10_artifact_descriptor_invalid",
            f"R10 {role} descriptor is missing or malformed",
            path=source,
        )
        return None
    path = state.resolve_run_path(descriptor.get("path"), source=source)
    expected_sha = str(descriptor.get("sha256", "")).lower()
    expected_size = descriptor.get("size_bytes")
    if (
        path is None
        or not _SHA256.fullmatch(expected_sha)
        or not isinstance(expected_size, int)
    ):
        state.fail(
            "r10_artifact_descriptor_invalid",
            f"R10 {role} descriptor lacks path/hash/size",
            path=source,
        )
        return None
    if path.is_symlink() or not path.is_file():
        state.fail(
            "r10_artifact_missing",
            f"R10 {role} artifact is missing or not a regular file",
            path=path,
        )
        if not path.exists():
            state.missing_paths.add(str(path))
        return None
    actual_sha = _sha256(path)
    actual_size = path.stat().st_size
    if actual_sha != expected_sha or actual_size != expected_size:
        state.fail(
            "r10_artifact_integrity_mismatch",
            f"R10 {role} artifact hash/size changed",
            path=path,
            details={
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
            },
        )
        return None
    state.critical_files.add(path)
    return path


def _r10_method_tuple(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("run_id", "")),
        str(row.get("run_root", "")),
        str(row.get("method_id", "")),
    )


def _r10_summary(
    state: _AuditState,
    payload: Mapping[str, Any] | None,
    *,
    source: Path,
    report_path: Path,
    report_sha256: str,
    report: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    summaries = payload.get("r10_existing_crog")
    if not isinstance(summaries, list):
        state.fail(
            "r10_summary_missing",
            "test metrics lack the R10 existing-CROG summary list",
            path=source,
        )
        return None
    matches = [
        row
        for row in summaries
        if isinstance(row, Mapping) and row.get("dataset") == "crog_frozen_top5"
    ]
    if len(matches) != 1:
        state.fail(
            "r10_summary_coverage_invalid",
            "test metrics require exactly one crog_frozen_top5 R10 summary",
            path=source,
        )
        return None
    summary = matches[0]
    declared_path = state.resolve_run_path(summary.get("report_path"), source=source)
    expected = {
        "status": report.get("status"),
        "eligible_count": report.get("eligible_count"),
        "comparison_count": report.get("comparison_count"),
        "excluded_count": report.get("excluded_count"),
    }
    observed = {key: summary.get(key) for key in expected}
    if (
        declared_path != report_path
        or summary.get("report_sha256") != report_sha256
        or observed != expected
        or summary.get("primary_reselection_permitted") is not False
        or summary.get("trained_by_current_matrix") is not False
    ):
        state.fail(
            "r10_summary_mismatch",
            "R10 summary path/hash/status/counts do not match comparison_report.json",
            path=source,
            details={"expected": expected, "observed": observed},
        )
    return summary


def _r10_result_rows(
    state: _AuditState,
    payload: Mapping[str, Any] | None,
    *,
    source: Path,
    report_path: Path,
    expected: set[tuple[str, str, str]],
) -> list[Mapping[str, Any]]:
    if payload is None:
        return []
    rows = payload.get("results")
    if not isinstance(rows, list):
        state.fail("r10_results_invalid", "test metrics lack results list", path=source)
        return []
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and (
            row.get("rung") == "R10"
            or row.get("designation") == "POST_LOCK_R10_EXISTING_COMPARISON"
        )
    ]
    observed = {
        (
            str(row.get("source_run_id", "")),
            str(row.get("source_run_root", "")),
            str(row.get("source_method", "")),
        )
        for row in selected
    }
    if observed != expected or len(selected) != len(expected):
        state.fail(
            "r10_result_coverage_mismatch",
            "R10 result rows do not exactly cover eligible existing methods",
            path=source,
        )
    for row in selected:
        declared_report = state.resolve_run_path(
            row.get("r10_comparison_report"), source=source
        )
        if (
            row.get("rung") != "R10"
            or row.get("designation") != "POST_LOCK_R10_EXISTING_COMPARISON"
            or row.get("eligible_for_primary_reselection") is not False
            or row.get("trained_by_current_matrix") is not False
            or row.get("gate") != "FROZEN_EXISTING"
            or declared_report != report_path
        ):
            state.fail(
                "r10_result_designation_invalid",
                "R10 result is not irreversibly marked post-lock/non-primary/non-trained",
                path=source,
            )
    return selected


def _audit_r10_existing_crog(state: _AuditState) -> None:
    discovery_path = state.root / "audit/r10_existing_run_discovery.json"
    discovery = state.json("audit/r10_existing_run_discovery.json")
    report_path = (
        state.root
        / "metrics/r10_existing/crog_frozen_top5/comparison_report.json"
    )
    report = state.json(
        "metrics/r10_existing/crog_frozen_top5/comparison_report.json"
    )
    post_path = state.root / "metrics/post_lock_test_results.json"
    all_path = state.root / "metrics/all_test_results.json"
    post = state.json("metrics/post_lock_test_results.json")
    all_results = state.json("metrics/all_test_results.json")
    if discovery is None:
        state.fail(
            "r10_discovery_missing",
            "formal train-stage R10 discovery evidence is missing",
            path=discovery_path,
        )
    if report is None:
        state.fail(
            "r10_comparison_report_missing",
            "formal CROG R10 comparison report is missing",
            path=report_path,
        )
    if post is None or all_results is None:
        state.fail(
            "r10_metrics_summary_file_missing",
            "R10 must be bound to post-lock and all-test metrics",
        )
    if discovery is None or report is None:
        return

    methods = discovery.get("methods")
    declared_eligible = discovery.get("eligible_methods")
    if (
        discovery.get("kind") != "existing_reranker_r10_discovery"
        or not isinstance(methods, list)
        or not isinstance(declared_eligible, list)
    ):
        state.fail(
            "r10_discovery_invalid",
            "R10 train discovery report is malformed",
            path=discovery_path,
        )
        return
    eligible_from_checks: set[tuple[str, str, str]] = set()
    ineligible: set[tuple[str, str, str]] = set()
    for method in methods:
        if not isinstance(method, Mapping):
            state.fail("r10_discovery_invalid", "R10 discovery method row is malformed", path=discovery_path)
            continue
        key = _r10_method_tuple(method)
        checks = method.get("checks")
        gaps = method.get("gaps")
        if (
            not all(key)
            or not isinstance(checks, Mapping)
            or set(checks) != set(_R10_REQUIRED_DISCOVERY_CHECKS)
        ):
            state.fail("r10_discovery_checks_invalid", "R10 discovery method checks are incomplete", path=discovery_path)
            continue
        passed = all(
            isinstance(checks[name], Mapping)
            and checks[name].get("passed") is True
            for name in _R10_REQUIRED_DISCOVERY_CHECKS
        )
        if bool(method.get("eligible")) != passed:
            state.fail("r10_discovery_eligibility_mismatch", "R10 eligible flag disagrees with checks", path=discovery_path)
        if passed:
            eligible_from_checks.add(key)
        else:
            ineligible.add(key)
            if not isinstance(gaps, list) or not gaps:
                state.fail("r10_discovery_exclusion_missing", "ineligible R10 method lacks structured gaps", path=discovery_path)
    eligible_declared = {
        _r10_method_tuple(row)
        for row in declared_eligible
        if isinstance(row, Mapping)
    }
    if eligible_declared != eligible_from_checks or len(eligible_declared) != len(
        declared_eligible
    ):
        state.fail("r10_discovery_inventory_mismatch", "R10 eligible inventory disagrees with method checks", path=discovery_path)

    try:
        validate_crog_existing_comparison_report(report)
    except CrogExistingComparisonError as error:
        state.fail(
            "r10_comparison_report_invalid",
            f"R10 comparison validator rejected the report: {error}",
            path=report_path,
        )
    if report.get("dataset") != "CROG" or report.get("scope") != "test":
        state.fail("r10_comparison_scope_invalid", "R10 comparison must be CROG test scope", path=report_path)
    status = report.get("status")
    if status not in {"complete", "complete_no_eligible"}:
        state.fail("r10_comparison_status_invalid", "R10 comparison status is not complete", path=report_path)
    comparisons = report.get("comparisons")
    exclusions = report.get("exclusions")
    if not isinstance(comparisons, list) or not isinstance(exclusions, list):
        state.fail("r10_comparison_report_invalid", "R10 report lacks comparison/exclusion lists", path=report_path)
        return
    compared = {
        _r10_method_tuple(row)
        for row in comparisons
        if isinstance(row, Mapping)
    }
    if (
        compared != eligible_from_checks
        or int(report.get("eligible_count", -1)) != len(eligible_from_checks)
        or int(report.get("comparison_count", -1)) != len(compared)
    ):
        state.fail("r10_eligible_comparison_mismatch", "R10 comparisons do not exactly cover train-discovered eligible methods", path=report_path)
    if eligible_from_checks and status != "complete":
        state.fail("r10_eligible_comparison_incomplete", "eligible R10 method lacks a complete comparison", path=report_path)
    if not eligible_from_checks:
        if status != "complete_no_eligible" or comparisons or not exclusions:
            state.fail("r10_zero_eligible_evidence_invalid", "zero-eligible R10 status lacks complete exclusion evidence", path=report_path)
        if ineligible:
            excluded = {
                _r10_method_tuple(row)
                for row in exclusions
                if isinstance(row, Mapping)
                and row.get("stage") == "eligibility"
                and isinstance(row.get("eligibility_checks"), Mapping)
                and isinstance(row.get("gaps"), list)
                and row.get("gaps")
            }
            if excluded != ineligible:
                state.fail("r10_zero_eligible_exclusions_incomplete", "zero-eligible report does not preserve every method's checks/gaps", path=report_path)
        else:
            global_exclusions = [
                row
                for row in exclusions
                if isinstance(row, Mapping)
                and isinstance(row.get("exclusion"), Mapping)
                and row["exclusion"].get("code") == "no_existing_methods_discovered"
                and isinstance(row["exclusion"].get("scan"), Mapping)
            ]
            if not global_exclusions:
                state.fail("r10_zero_eligible_exclusions_incomplete", "empty discovery lacks scan-backed exclusion evidence", path=report_path)

    artifacts = report.get("artifacts")
    if isinstance(artifacts, Mapping):
        for role, descriptor in artifacts.items():
            _r10_artifact_descriptor(state, descriptor, source=report_path, role=str(role))
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            continue
        evidence = comparison.get("comparison_evidence")
        if not isinstance(evidence, Mapping):
            continue
        _r10_artifact_descriptor(state, evidence.get("evaluator"), source=report_path, role="evaluator")
        comparison_artifacts = evidence.get("artifacts")
        if isinstance(comparison_artifacts, Mapping):
            for role, descriptor in comparison_artifacts.items():
                _r10_artifact_descriptor(state, descriptor, source=report_path, role=str(role))

    report_sha256 = _sha256(report_path)
    post_summary = _r10_summary(
        state,
        post,
        source=post_path,
        report_path=report_path,
        report_sha256=report_sha256,
        report=report,
    )
    all_summary = _r10_summary(
        state,
        all_results,
        source=all_path,
        report_path=report_path,
        report_sha256=report_sha256,
        report=report,
    )
    if post_summary is not None and all_summary is not None and dict(post_summary) != dict(all_summary):
        state.fail("r10_summary_cross_file_mismatch", "post-lock and all-test R10 summaries differ", path=all_path)
    post_rows = _r10_result_rows(
        state,
        post,
        source=post_path,
        report_path=report_path,
        expected=eligible_from_checks,
    )
    all_rows = _r10_result_rows(
        state,
        all_results,
        source=all_path,
        report_path=report_path,
        expected=eligible_from_checks,
    )
    if [dict(row) for row in post_rows] != [dict(row) for row in all_rows]:
        state.fail("r10_results_cross_file_mismatch", "post-lock and all-test R10 result rows differ", path=all_path)


def _check_independent_and_tests(
    state: _AuditState,
    datasets: tuple[DatasetExpectation, ...],
) -> None:
    tests = state.json("audit/evaluator_tests.json")
    if tests is not None:
        if tests.get("status") != "PASS" or tests.get("exit_code") != 0 or tests.get("tests_failed") != 0:
            state.fail("evaluator_tests_failed", "evaluator test evidence is not PASS")
        if not tests.get("command") or _iso8601(tests.get("completed_at")) is None:
            state.fail("evaluator_tests_evidence_incomplete", "evaluator test evidence lacks command or completed_at")
        log_raw = tests.get("log_path")
        log_path = state.resolve_run_path(log_raw, source=state.root / "audit/evaluator_tests.json")
        if log_path is None or log_path.is_symlink() or not log_path.is_file() or log_path.stat().st_size == 0:
            state.fail("evaluator_test_log_missing", "evaluator test log is missing or empty", path=log_path or "")
        else:
            state.critical_files.add(log_path)

    sanity = state.json("audit/SANITY_AUDIT.json")
    if sanity is None:
        return
    if sanity.get("stage") != "post_test_complete" or sanity.get("all_mandatory_checks_executed") is not True:
        state.fail("sanity_audit_incomplete", "post-test sanity audit is not complete")
    rows = sanity.get("datasets")
    if not isinstance(rows, list):
        state.fail("sanity_audit_invalid", "sanity audit lacks datasets list")
        return
    by_dataset = {str(row.get("dataset")): row for row in rows if isinstance(row, Mapping)}
    for dataset in datasets:
        row = by_dataset.get(dataset.key)
        if not isinstance(row, Mapping):
            state.fail("independent_recompute_missing", f"sanity audit missing dataset {dataset.key}")
            continue
        independent = row.get("independent_test_evaluator")
        if not isinstance(independent, Mapping) or independent.get("status") != "PASS" or independent.get("exact_match") is not True:
            state.fail("independent_recompute_failed", f"independent recomputation is not exact PASS for {dataset.key}")
        scanner = row.get("test_fit_scanner")
        if not isinstance(scanner, Mapping) or scanner.get("passed") is not True:
            state.fail("test_fit_scanner_failed", f"test-fit scanner is not PASS for {dataset.key}")


def _csv_has_data(state: _AuditState, relative: str) -> None:
    path = state.regular_file(relative)
    if path is None:
        return
    try:
        rows = list(csv.reader(path.open("r", encoding="utf-8", newline="")))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        state.fail("invalid_csv_evidence", f"cannot parse CSV evidence: {error}", path=path)
        return
    if len(rows) < 2 or not rows[0] or not any(cell.strip() for cell in rows[1]):
        state.fail("empty_csv_evidence", "CSV evidence must contain a header and at least one data row", path=path)


def _check_reporting(state: _AuditState) -> None:
    for relative in (*STATISTICS_FILES, *ABLATION_FILES):
        _csv_has_data(state, relative)
    for relative in REPORT_FILES:
        state.regular_file(relative)

    gallery = state.json("galleries/gallery_summary.json")
    index_path = state.regular_file("galleries/index.csv")
    gallery_rows: list[dict[str, str]] = []
    if index_path is not None:
        try:
            with index_path.open("r", encoding="utf-8", newline="") as stream:
                gallery_rows = list(csv.DictReader(stream))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            state.fail("gallery_index_invalid", f"cannot parse gallery index: {error}", path=index_path)
    if gallery is not None:
        for category in GALLERY_CATEGORIES:
            row = gallery.get(category)
            if not isinstance(row, Mapping):
                state.fail("gallery_category_missing", f"gallery category missing: {category}")
                continue
            try:
                requested = int(row["requested"])
                eligible = int(row["eligible"])
                selected = int(row["selected"])
                materialized = int(row["materialized"])
            except (KeyError, TypeError, ValueError):
                state.fail("gallery_counts_invalid", f"gallery counts invalid: {category}")
                continue
            if requested < 25:
                state.fail(
                    "gallery_quota_too_small",
                    f"gallery must request at least 25 deterministic cases per category: {category}",
                    details={"requested": requested, "minimum": 25},
                )
            per_dataset = row.get("per_dataset")
            aggregate_selection_invalid = (
                per_dataset is None and selected != min(requested, eligible)
            )
            if min(requested, eligible, selected, materialized) < 0 or aggregate_selection_invalid or materialized != selected:
                state.fail(
                    "gallery_materialization_incomplete",
                    f"every selected eligible gallery case must be materialized: {category}",
                    details={"requested": requested, "eligible": eligible, "selected": selected, "materialized": materialized},
                )
            if per_dataset is not None:
                if not isinstance(per_dataset, Mapping) or not per_dataset:
                    state.fail(
                        "gallery_per_dataset_invalid",
                        f"gallery per-dataset accounting is invalid: {category}",
                    )
                else:
                    totals = {name: 0 for name in ("requested", "eligible", "selected", "materialized")}
                    for dataset, counts in per_dataset.items():
                        if not isinstance(counts, Mapping):
                            state.fail("gallery_per_dataset_invalid", f"gallery counts missing for {category}/{dataset}")
                            continue
                        try:
                            values = {name: int(counts[name]) for name in totals}
                        except (KeyError, TypeError, ValueError):
                            state.fail("gallery_per_dataset_invalid", f"gallery counts invalid for {category}/{dataset}")
                            continue
                        if values["requested"] < 25:
                            state.fail("gallery_quota_too_small", f"gallery must request 25 cases for {category}/{dataset}")
                        if values["selected"] != min(values["requested"], values["eligible"]) or values["materialized"] != values["selected"]:
                            state.fail("gallery_materialization_incomplete", f"per-dataset gallery materialization is incomplete: {category}/{dataset}")
                        for name, value in values.items():
                            totals[name] += value
                    declared = {"requested": requested, "eligible": eligible, "selected": selected, "materialized": materialized}
                    if totals != declared:
                        state.fail("gallery_per_dataset_total_mismatch", f"gallery aggregate differs from per-dataset counts: {category}", details={"declared": declared, "computed": totals})
            indexed_assets = []
            for indexed in gallery_rows:
                if indexed.get("category") != category or not indexed.get("asset_path", "").strip():
                    continue
                asset = state.resolve_run_path(
                    str(Path("galleries") / indexed["asset_path"]),
                    source=index_path or state.root / "galleries/index.csv",
                )
                if asset is None or asset.is_symlink() or not asset.is_file() or asset.stat().st_size == 0:
                    state.fail("gallery_asset_missing", f"indexed gallery asset is missing: {category}", path=asset or "")
                else:
                    indexed_assets.append(asset)
                    state.critical_files.add(asset)
            if len(indexed_assets) != materialized:
                state.fail(
                    "gallery_materialization_unverified",
                    f"gallery index does not prove its materialized count: {category}",
                    details={"declared": materialized, "verified": len(indexed_assets)},
                )
        failure_stages = gallery.get("failure_stages")
        if not isinstance(failure_stages, Mapping) or set(failure_stages) != {
            f"F{index}" for index in range(11)
        }:
            state.fail("failure_stage_gallery_missing", "gallery must report F0 through F10")
        for stage in (f"F{index}" for index in range(11)):
            row = failure_stages.get(stage) if isinstance(failure_stages, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            try:
                requested = int(row["requested"])
                eligible = int(row["eligible"])
                selected = int(row["selected"])
                materialized = int(row["materialized"])
            except (KeyError, TypeError, ValueError):
                state.fail("failure_stage_counts_invalid", f"failure-stage gallery counts invalid: {stage}")
                continue
            per_dataset = row.get("per_dataset")
            selection_invalid = per_dataset is None and selected != min(requested, eligible)
            if requested < 25 or selection_invalid or materialized != selected:
                state.fail(
                    "failure_stage_materialization_incomplete",
                    f"failure-stage gallery is incomplete: {stage}",
                    details={"requested": requested, "eligible": eligible, "selected": selected, "materialized": materialized},
                )
            verified = sum(
                1
                for indexed in gallery_rows
                if indexed.get("category") == f"failure_stage:{stage}"
                and indexed.get("asset_path", "").strip()
            )
            if verified != materialized:
                state.fail(
                    "failure_stage_materialization_unverified",
                    f"failure-stage gallery index count differs: {stage}",
                    details={"declared": materialized, "verified": verified},
                )
        failure_groups = gallery.get("failure_groups")
        required_groups = {"grounding", "candidate-generation", "ranking"}
        if not isinstance(failure_groups, Mapping) or set(failure_groups) != required_groups:
            state.fail(
                "failure_group_gallery_missing",
                "gallery must report grounding, candidate-generation, and ranking failures",
            )
        for group_name in sorted(required_groups):
            row = failure_groups.get(group_name) if isinstance(failure_groups, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            try:
                requested = int(row["requested"])
                eligible = int(row["eligible"])
                selected = int(row["selected"])
                materialized = int(row["materialized"])
            except (KeyError, TypeError, ValueError):
                state.fail("failure_group_counts_invalid", f"failure gallery counts invalid: {group_name}")
                continue
            per_dataset = row.get("per_dataset")
            selection_invalid = per_dataset is None and selected != min(requested, eligible)
            if requested < 25 or selection_invalid or materialized != selected:
                state.fail(
                    "failure_group_materialization_incomplete",
                    f"failure gallery is incomplete: {group_name}",
                    details={"requested": requested, "eligible": eligible, "selected": selected, "materialized": materialized},
                )
            verified = sum(
                1
                for indexed in gallery_rows
                if indexed.get("category") == f"failure_group:{group_name}"
                and indexed.get("asset_path", "").strip()
            )
            if verified != materialized:
                state.fail(
                    "failure_group_materialization_unverified",
                    f"failure gallery index count differs: {group_name}",
                    details={"declared": materialized, "verified": verified},
                )
    state.regular_file("galleries/index.html")

    bundle_index = state.json("statistics/reporting_bundle_index.json")
    bundles = bundle_index.get("bundles") if bundle_index is not None else None
    if bundle_index is not None and (not isinstance(bundles, list) or not bundles):
        state.fail("reporting_bundle_index_invalid", "reporting bundle index is empty")
        return
    for row in bundles or []:
        if not isinstance(row, Mapping):
            state.fail("reporting_bundle_index_invalid", "reporting bundle entry must be an object")
            continue
        manifest_path = state.resolve_run_path(row.get("path"), source=state.root / "statistics/reporting_bundle_index.json")
        if manifest_path is not None and manifest_path.is_dir():
            manifest_path = manifest_path / "reporting_manifest.json"
        if manifest_path is None or manifest_path.is_symlink() or not manifest_path.is_file():
            state.fail("reporting_manifest_missing", "reporting bundle manifest is missing", path=manifest_path or "")
            continue
        expected = str(row.get("manifest_sha256", "")).lower()
        if not _SHA256.fullmatch(expected) or _sha256(manifest_path) != expected:
            state.fail("reporting_manifest_hash_mismatch", "reporting bundle manifest hash mismatch", path=manifest_path)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            state.fail("reporting_manifest_invalid", f"cannot parse reporting manifest: {error}", path=manifest_path)
            continue
        if manifest.get("status") != "complete" or manifest.get("run_success_marker_written") is not False:
            state.fail("reporting_manifest_incomplete", "reporting manifest is not complete and isolated", path=manifest_path)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            state.fail("reporting_traceability_missing", "reporting manifest has no checksummed artifacts", path=manifest_path)
            continue
        for descriptor in artifacts.values():
            if not isinstance(descriptor, Mapping):
                state.fail("reporting_traceability_missing", "reporting artifact descriptor is invalid", path=manifest_path)
                continue
            artifact = state.resolve_run_path(descriptor.get("path"), source=manifest_path)
            expected_sha = str(descriptor.get("sha256", "")).lower()
            expected_size = descriptor.get("size_bytes")
            if artifact is None or artifact.is_symlink() or not artifact.is_file():
                state.fail("reporting_artifact_missing", "reporting artifact is missing", path=artifact or "")
            elif not _SHA256.fullmatch(expected_sha) or expected_size != artifact.stat().st_size or expected_sha != _sha256(artifact):
                state.fail("reporting_artifact_integrity_mismatch", "reporting artifact hash/size mismatch", path=artifact)


def _check_root_checksums(state: _AuditState, experiment_paths: Sequence[Path]) -> None:
    checksum_path = state.regular_file("checksums.sha256")
    if checksum_path is None:
        return
    entries = _parse_checksum_lines(state, checksum_path, allow_external=False)
    required = set(state.critical_files) | set(state.stage_outputs) | set(state.manifest_artifacts) | set(experiment_paths)
    required.discard(checksum_path)
    missing = sorted(path for path in required if path not in entries)
    for path in missing:
        state.fail("critical_checksum_missing", "critical run file is absent from checksums.sha256", path=path)


def _check_background(state: _AuditState, evidence: Mapping[str, Any] | None) -> None:
    if evidence is None:
        state.fail("background_activity_evidence_missing", "background process/container evidence was not supplied")
        return
    processes = evidence.get("active_processes")
    containers = evidence.get("active_docker_containers")
    if _iso8601(evidence.get("captured_at")) is None or not isinstance(processes, list) or not isinstance(containers, list):
        state.fail("background_activity_evidence_invalid", "background evidence lacks timestamp/process/container lists")
        return
    if evidence.get("status") != "PASS" or processes or containers:
        state.fail(
            "background_activity_detected",
            "training/scoring processes or Docker containers remain active",
            details={"active_processes": processes, "active_docker_containers": containers},
        )


def audit_run_completion(
    output_root: str | os.PathLike[str],
    formal_spec_keys: Sequence[str],
    expected_datasets: Sequence[DatasetExpectation | Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    expected_seeds: Sequence[int],
    expected_folds: int | Sequence[int],
    *,
    background_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Audit all machine-checkable conditions required before run success.

    This function never writes to ``output_root``.  Invalid expectations and
    run evidence defects are both returned as structured ``FAIL`` results.
    """

    root = Path(os.path.abspath(Path(output_root).expanduser()))
    state = _AuditState(root)
    try:
        methods = tuple(dict.fromkeys(map(str, formal_spec_keys)))
        seeds = tuple(map(int, expected_seeds))
        folds = _fold_values(expected_folds)
        datasets = _normalize_datasets(expected_datasets)
        if not methods or any(not method for method in methods):
            raise ValueError("formal_spec_keys must be non-empty unique strings")
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError("formal completion requires at least three unique seeds")
        if len(folds) != 5:
            raise ValueError("formal completion requires exactly five grouped folds")
        required_route_pools = {
            ("crog", "frozen_top5"),
            ("modular", "frozen_top5"),
            ("modular", "full_post_filter"),
        }
        observed_route_pools = {(item.route, item.pool) for item in datasets}
        missing_route_pools = required_route_pools - observed_route_pools
        if missing_route_pools:
            raise ValueError(
                "formal completion expectations omit required route/pool tracks: "
                f"{sorted(missing_route_pools)}"
            )
    except (TypeError, ValueError) as error:
        state.fail("audit_expectations_invalid", str(error))
        return {
            "schema_version": 1,
            "status": "FAIL",
            "passed": False,
            "output_root": str(root),
            "blockers": state.blockers,
            "missing_paths": [],
            "checks": {},
        }

    if root.is_symlink() or not root.is_dir():
        state.fail("output_root_invalid", "output root must be a regular directory", path=root)
        if not root.exists():
            state.missing_paths.add(str(root))
        return {
            "schema_version": 1,
            "status": "FAIL",
            "passed": False,
            "output_root": str(root),
            "blockers": state.blockers,
            "missing_paths": sorted(state.missing_paths),
            "checks": state.checks,
        }

    state.run("stage_markers", lambda: _check_stage_markers(state))
    manifests: list[tuple[Path, Mapping[str, Any]]] = []

    def manifests_check() -> None:
        manifests.extend(_load_experiment_manifests(state))
        _check_manifest_coverage(state, manifests, methods, datasets, seeds, folds)

    state.run("experiment_coverage", manifests_check)
    state.run("canonical_baseline_features", lambda: _check_canonical_and_features(state, datasets))
    state.run(
        "modular_development_publication",
        lambda: _check_modular_development_publication(state),
    )
    state.run("entity_leakage", lambda: _check_entity_leakage_bundle(state))
    state.run("test_protocol", lambda: _check_test_protocol(state, manifests, datasets, seeds, folds))
    state.run("r10_existing_crog", lambda: _audit_r10_existing_crog(state))
    state.run("independent_recompute_and_tests", lambda: _check_independent_and_tests(state, datasets))
    state.run("statistics_ablations_reports", lambda: _check_reporting(state))
    state.run("background_activity", lambda: _check_background(state, background_evidence))
    state.run("checksums", lambda: _check_root_checksums(state, [path for path, _ in manifests]))

    status = "PASS" if not state.blockers else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "passed": status == "PASS",
        "output_root": str(root),
        "expectations": {
            "formal_spec_keys": list(methods),
            "datasets": [item.__dict__ for item in datasets],
            "seeds": list(seeds),
            "folds": list(folds),
        },
        "checks": state.checks,
        "blockers": state.blockers,
        "missing_paths": sorted(state.missing_paths),
        "critical_file_count": len(state.critical_files),
    }


__all__ = [
    "ABLATION_FILES",
    "CUMULATIVE_STAGE_OUTPUT_POLICIES",
    "DatasetExpectation",
    "FORMAL_STAGES",
    "GALLERY_CATEGORIES",
    "IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES",
    "REPORT_FILES",
    "STATISTICS_FILES",
    "audit_background_activity",
    "audit_run_completion",
]
