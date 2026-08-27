"""Fail-closed, data-driven paper artifacts for a completed GraspNet 6-DoF run.

The module is intentionally downstream of every model and evaluator stage.  It
never runs HiFi-CS, VGN, LightGBM, or graspnetAPI.  Every number and every
plotted point is read from a content-addressed CSV/JSON artifact emitted by the
formal pipeline.  A blocked, fixture-scoped, stale, split-leaking, or otherwise
incomplete run is refused before any top-level paper prose is published.

The public entry point is :func:`generate_formal_paper_artifacts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .experiment_analysis import (
    ANALYSIS_SCHEMA,
    FORMAL_SCOPE,
    FORMAL_SEEDS,
    load_analysis_input_manifest,
)
from .formal_inputs import group_artifact_slug
from .grounding_pipeline import (
    GROUNDING_METRICS_SCHEMA,
    PREDICTED_SELECTION_SCHEMA,
)
from .io import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from .provenance import RUN_MANIFEST_SCHEMA_VERSION, update_manifest
from .reporting import OKABE_ITO
from .smoke import validate_4d_regression_status
from .stages import (
    load_evaluator_geometry_contract,
    load_evaluator_parity_gate,
)


PAPER_ARTIFACT_SCHEMA = "graspnet6d_formal_paper_artifacts_v1"
FIGURE_MANIFEST_SCHEMA = "graspnet6d_quantitative_figures_v1"
TABLE_MANIFEST_SCHEMA = "graspnet6d_formal_tables_v1"
GALLERY_MANIFEST_SCHEMA = "graspnet6d_failure_gallery_v1"

CONDITIONS = (
    "oracle_gt_mask",
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)
ANALYSES = (*CONDITIONS, "combined_a8")
PREDICTED_CONDITIONS = CONDITIONS[1:]
PRIMARY_SYSTEMS = ("B0_NATIVE", "R0_RAW", "R1_GATED", "O_ORACLE")

FIGURE_FAMILIES: tuple[tuple[str, str], ...] = (
    ("01_native_vs_reranked_p_at_1", "Native versus reranked target P@1"),
    ("02_native_vs_reranked_ap", "Native versus reranked target-specific AP"),
    ("03_oracle_at_k", "Frozen-pool Oracle@K"),
    ("04_recovered_harmful_net", "Recovered, harmful, and net outcomes"),
    ("05_feature_ablation", "Feature ablation"),
    ("06_top_k_sensitivity", "Top-K sensitivity"),
    ("07_gt_vs_predicted_mask", "GT-mask versus predicted-mask"),
    ("08_candidate_coverage_by_scene", "Candidate coverage by scene"),
    ("09_first_valid_rank_distribution", "First-valid target rank distribution"),
    ("10_performance_by_target_visibility", "Performance by target visibility"),
    ("11_performance_by_mask_iou", "Performance by mask IoU bin"),
    ("12_performance_by_object_size", "Performance by object size"),
    ("13_performance_by_query_type", "Performance by query type"),
    ("14_stage_failure_decomposition", "Stage-failure decomposition"),
    ("15_runtime_breakdown", "Runtime breakdown"),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_FORMAL_TOKEN = re.compile(
    r"fixture|synthetic|dummy|placeholder|unit[-_ ]?test|smoke[-_ ]?only",
    re.IGNORECASE,
)
_REQUIRED_ANALYSIS_OUTPUTS = (
    "metrics.csv",
    "metrics.json",
    "per_group_metrics.csv",
    "paired_outcomes.csv",
    "bootstrap_results.csv",
    "significance_tests.json",
    "ablation_results.csv",
    "failure_taxonomy.csv",
    "failure_summary.json",
    "frozen_pool_audit.json",
    "model_selection.json",
    "resolved_analysis_config.json",
    "native_predictions.csv",
    "reranked_predictions.csv",
    "gated_predictions.csv",
    "oracle_predictions.csv",
)
_REQUIRED_STAGE_FILES = (
    "download",
    "prepare",
    "prepare_experiment",
    "masks",
    "candidates",
    "labels",
    "features",
    "real-ranker-smoke",
    "train-ranker",
    "evaluate",
    "ablate",
)
_REQUIRED_RUNTIME_STAGES = (
    "download",
    "prepare",
    "masks",
    "candidates",
    "labels",
    "features",
    "real-ranker-smoke",
    "smoke",
    "train-ranker",
    "evaluate",
    "ablate",
)
_TOP_LEVEL_DOCUMENTS = (
    "METHODS.md",
    "RESULTS.md",
    "CONCLUSIONS.md",
    "LIMITATIONS.md",
    "REPRODUCE.md",
    "FINAL_STATUS.md",
    "dataset_summary.json",
    "conclusion_evidence.json",
)


class PaperArtifactsRefused(RuntimeError):
    """A formal report was requested without complete, current real evidence."""


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Hash-verified saved outputs for one formal analysis arm."""

    name: str
    root: Path
    manifest: Mapping[str, Any]
    metrics: pd.DataFrame
    per_group: pd.DataFrame
    paired: pd.DataFrame
    bootstrap: pd.DataFrame
    ablations: pd.DataFrame
    failures: pd.DataFrame
    native: pd.DataFrame
    reranked: pd.DataFrame
    significance: Mapping[str, Any]
    metrics_json: Mapping[str, Any]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FormalPaperInputs:
    """All verified, immutable inputs consumed by the renderer."""

    run_dir: Path
    run_manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    target_rows: tuple[Mapping[str, Any], ...]
    language_rows: tuple[Mapping[str, Any], ...]
    grounding: pd.DataFrame
    candidate_summary: pd.DataFrame
    runtime: pd.DataFrame
    selected_condition: str
    analyses: Mapping[str, AnalysisSnapshot]
    source_hashes: Mapping[str, str]
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class PaperArtifactResult:
    """Published output record returned by the public entry point."""

    run_dir: Path
    manifest_path: Path
    manifest_sha256: str
    input_fingerprint: str
    selected_predicted_condition: str
    executed_figures: tuple[str, ...]
    unexecuted_figures: tuple[str, ...]
    outputs: Mapping[str, str]
    resumed: bool


@dataclass(frozen=True, slots=True)
class _TableRecord:
    name: str
    path: Path
    rows: int
    source_paths: tuple[Path, ...]


def _regular_file(path: Path | str, description: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise PaperArtifactsRefused(f"{description} must not be a symlink: {raw}")
    source = raw.resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise PaperArtifactsRefused(
            f"missing non-empty regular {description}: {source}"
        )
    return source


def _inside(owner: Path, path: Path, description: str) -> Path:
    try:
        path.resolve().relative_to(owner.resolve())
    except ValueError as error:
        raise PaperArtifactsRefused(
            f"{description} escapes the formal run directory: {path}"
        ) from error
    return path.resolve()


def _read_json(path: Path | str, description: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, description)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperArtifactsRefused(
            f"invalid {description} {source}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PaperArtifactsRefused(f"{description} must be a JSON object: {source}")
    return source, value


def _digest(value: Any, description: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise PaperArtifactsRefused(f"{description} is not a lowercase SHA-256 digest")
    return text


def _bound_file(
    owner: Path,
    raw_path: Any,
    raw_hash: Any,
    description: str,
    *,
    run_dir: Path | None = None,
) -> Path:
    value = Path(str(raw_path)).expanduser()
    source = value if value.is_absolute() else owner.parent / value
    source = _regular_file(source, description)
    if run_dir is not None:
        _inside(run_dir, source, description)
    expected = _digest(raw_hash, f"{description} hash")
    observed = sha256_file(source)
    if observed != expected:
        raise PaperArtifactsRefused(
            f"stale {description}: expected {expected}, observed {observed}: {source}"
        )
    return source


def _read_jsonl(path: Path | str, description: str) -> tuple[dict[str, Any], ...]:
    source = _regular_file(path, description)
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {number} is not an object")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise PaperArtifactsRefused(
            f"invalid {description} {source}: {error}"
        ) from error
    if not rows:
        raise PaperArtifactsRefused(f"{description} is empty: {source}")
    return tuple(rows)


def _read_csv(path: Path | str, description: str) -> pd.DataFrame:
    source = _regular_file(path, description)
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PaperArtifactsRefused(
            f"invalid {description} {source}: {error}"
        ) from error
    if frame.empty:
        raise PaperArtifactsRefused(f"{description} is empty: {source}")
    return frame


def _csv_records(path: Path | str, description: str) -> list[dict[str, str]]:
    source = _regular_file(path, description)
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise PaperArtifactsRefused(
            f"invalid {description} {source}: {error}"
        ) from error
    if not rows:
        raise PaperArtifactsRefused(f"{description} is empty: {source}")
    return rows


def _scope_is_formal(frame: pd.DataFrame, description: str) -> None:
    if "analysis_scope" not in frame.columns:
        raise PaperArtifactsRefused(f"{description} lacks analysis_scope")
    if frame["analysis_scope"].astype(str).ne(FORMAL_SCOPE).any():
        raise PaperArtifactsRefused(
            f"{description} contains non-formal or mixed-scope rows"
        )


def _bool_series(values: pd.Series, description: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    lowered = values.astype(str).str.strip().str.lower()
    if not lowered.isin({"true", "false", "1", "0"}).all():
        raise PaperArtifactsRefused(f"{description} contains non-boolean values")
    return lowered.isin({"true", "1"})


def _finite_column(frame: pd.DataFrame, column: str, description: str) -> pd.Series:
    if column not in frame.columns:
        raise PaperArtifactsRefused(f"{description} lacks column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise PaperArtifactsRefused(
            f"{description} column {column!r} contains non-finite values"
        )
    return values


def _normalise_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _validate_run_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path, manifest = _read_json(run_dir / "run_manifest.json", "run manifest")
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise PaperArtifactsRefused("run manifest uses an unsupported schema")
    if manifest.get("run_id") != run_dir.name:
        raise PaperArtifactsRefused("run manifest ID differs from its directory")
    if _FORBIDDEN_FORMAL_TOKEN.search(run_dir.name):
        raise PaperArtifactsRefused(
            "fixture/diagnostic-labelled run IDs are not formal"
        )
    if manifest.get("fixture_only") is True:
        raise PaperArtifactsRefused("fixture-scoped run manifests are not formal")
    if str(manifest.get("status", "")).upper() in {"BLOCKED", "FAILED"}:
        raise PaperArtifactsRefused(
            f"run status is {manifest.get('status')!r}; paper outputs are forbidden"
        )
    profile = str(manifest.get("profile", ""))
    if profile not in {"paper-lite", "paper-lite-train3", "paper-extended"}:
        raise PaperArtifactsRefused(f"profile {profile!r} is not a formal data profile")
    identity = manifest.get("immutable_identity")
    if not isinstance(identity, Mapping) or identity.get("profile") != profile:
        raise PaperArtifactsRefused("run manifest lacks its immutable identity")
    config = identity.get("resolved_config")
    if not isinstance(config, Mapping) or config.get("formal_results") is not True:
        raise PaperArtifactsRefused("resolved profile is not formal-results eligible")
    resolved = _regular_file(run_dir / "resolved_config.yaml", "resolved config")
    if sha256_file(resolved) != _digest(
        manifest.get("resolved_config_sha256"), "resolved config hash"
    ):
        raise PaperArtifactsRefused("resolved configuration is stale")
    immutable_hash = canonical_sha256(identity)
    if immutable_hash != canonical_sha256(manifest.get("immutable_identity")):
        raise PaperArtifactsRefused("run immutable identity cannot be reproduced")
    # Keep the source live while returning; this also rejects an unexpected
    # replacement between path resolution and validation.
    if not manifest_path.is_file():  # pragma: no cover - defensive
        raise PaperArtifactsRefused("run manifest disappeared during validation")
    return manifest, dict(config)


def _expected_smoke_kind(number: int) -> str:
    return "data_independent_regression" if number == 12 else "formal_smoke_gate"


def _paper_run_manifest_hash(manifest: Mapping[str, Any]) -> str:
    # Publication and strict resume attempts change these operational fields.
    # They are audit metadata, not scientific inputs.  Excluding them keeps a
    # revalidation-only resume byte-stable while binding every data/config/model
    # field consumed by the generated documents.
    operational = {
        "status",
        "ended_at_utc",
        "formal_results_emitted",
        "paper_artifact_manifest_sha256",
        "selected_predicted_condition",
        "commands_executed",
        "last_resumed_at_utc",
        "status_history",
        "blocked_stage",
        "blocked_reason",
        "failed_stage",
        "failure_type",
        "failure_message",
    }
    return canonical_sha256(
        {key: value for key, value in manifest.items() if key not in operational}
    )


def _validate_stage_gates(run_dir: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for stage in _REQUIRED_STAGE_FILES:
        path, payload = _read_json(
            run_dir / "stages" / f"{stage}.json", f"{stage} stage"
        )
        if payload.get("stage") != stage or payload.get("status") != "COMPLETE":
            raise PaperArtifactsRefused(f"required stage {stage!r} is not COMPLETE")
        sources[str(path)] = sha256_file(path)

    smoke_path, smoke = _read_json(
        run_dir / "smoke" / "smoke_checks.json", "real-data smoke evidence"
    )
    checks = smoke.get("checks")
    if smoke.get("go") is not True or not isinstance(checks, list):
        raise PaperArtifactsRefused("formal smoke gate is not GO")
    by_number = {
        int(row.get("number", -1)): row for row in checks if isinstance(row, Mapping)
    }
    if set(by_number) != set(range(1, 13)):
        raise PaperArtifactsRefused("smoke evidence does not contain checks 1..12")
    for number, row in by_number.items():
        if row.get("passed") is not True:
            raise PaperArtifactsRefused(f"smoke check {number} did not pass")
        expected_kind = _expected_smoke_kind(number)
        if row.get("kind") != expected_kind:
            raise PaperArtifactsRefused(
                f"smoke check {number} has kind {row.get('kind')!r}; "
                f"expected {expected_kind!r}"
            )
    sources[str(smoke_path)] = sha256_file(smoke_path)

    regression = validate_4d_regression_status()
    if not regression.passed:
        raise PaperArtifactsRefused(
            f"legacy 4-DoF regression gate failed: {regression.evidence}"
        )
    regression_dir = (
        run_dir.parent / "regression"
        if run_dir.parent.name == "graspnet6d"
        else run_dir / "regression"
    )
    status_path = _regular_file(
        regression_dir / "4d_regression_status.json", "4-DoF regression status"
    )
    _, status = _read_json(status_path, "4-DoF regression status")
    report_path = _regular_file(
        regression_dir / str(status["report_path"]), "4-DoF regression report"
    )
    sources[str(status_path)] = sha256_file(status_path)
    sources[str(report_path)] = sha256_file(report_path)

    geometry_path = run_dir / "geometry_validation" / "evaluator_geometry_contract.json"
    parity_path = run_dir / "evaluator_parity" / "evaluator_parity_gate.json"
    try:
        _, geometry = load_evaluator_geometry_contract(
            geometry_path, evidence_policy="formal"
        )
        _, parity = load_evaluator_parity_gate(parity_path, evidence_policy="formal")
    except Exception as error:
        raise PaperArtifactsRefused(
            f"formal geometry/evaluator parity gate failed: {type(error).__name__}: {error}"
        ) from error
    for record, members in (
        (
            geometry,
            (
                "contract_path",
                "validation_artifact",
            ),
        ),
        (
            parity,
            (
                "gate_path",
                "artifact_path",
            ),
        ),
    ):
        for member in members:
            path = _regular_file(record[member], f"formal gate {member}")
            _inside(run_dir, path, f"formal gate {member}")
            sources[str(path)] = sha256_file(path)
    return sources


def _validate_manifests(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[
    tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], dict[str, str]
]:
    target_path = _regular_file(
        run_dir / "manifests" / "target_groups.jsonl", "target manifest"
    )
    language_path = _regular_file(
        run_dir / "manifests" / "language_queries.jsonl", "language manifest"
    )
    target_rows = _read_jsonl(target_path, "target manifest")
    language_rows = _read_jsonl(language_path, "language manifest")
    target_hash = sha256_file(target_path)
    language_hash = sha256_file(language_path)
    if target_hash != _digest(
        manifest.get("dataset_manifest_hash"), "dataset manifest hash"
    ):
        raise PaperArtifactsRefused("target manifest differs from run provenance")

    report_path, report = _read_json(
        run_dir / "manifests" / "manifest_report.json", "target/language report"
    )
    if (
        report.get("target_manifest_sha256") != target_hash
        or report.get("language_manifest_sha256") != language_hash
        or int(report.get("target_group_count", -1)) != len(target_rows)
        or int(report.get("language_query_count", -1)) != len(language_rows)
    ):
        raise PaperArtifactsRefused("target/language report is stale")
    root_language = _regular_file(
        run_dir / "language_manifest.jsonl", "published language manifest"
    )
    if sha256_file(root_language) != language_hash:
        raise PaperArtifactsRefused("published language manifest is stale")

    target_ids = [str(row.get("group_id", "")) for row in target_rows]
    language_ids = [str(row.get("group_id", "")) for row in language_rows]
    if (
        any(not value for value in target_ids)
        or len(target_ids) != len(set(target_ids))
        or any(not value for value in language_ids)
        or len(language_ids) != len(set(language_ids))
        or set(target_ids) != set(language_ids)
    ):
        raise PaperArtifactsRefused(
            "target/language manifests lack one exact unique group universe"
        )
    split_by_scene: dict[str, str] = {}
    split_counts = {name: 0 for name in ("train", "validation", "test")}
    for row in target_rows:
        split = str(row.get("split", ""))
        scene = str(row.get("scene_id", ""))
        if split not in split_counts or not scene:
            raise PaperArtifactsRefused("target manifest contains invalid split/scene")
        prior = split_by_scene.setdefault(scene, split)
        if prior != split:
            raise PaperArtifactsRefused(
                f"scene {scene!r} crosses {prior!r} and {split!r}"
            )
        split_counts[split] += 1
        for field in (
            "rgb_path",
            "depth_path",
            "instance_label_path",
            "meta_path",
            "intrinsics_path",
        ):
            _regular_file(row.get(field, ""), f"target {field}")
    if any(count <= 0 for count in split_counts.values()):
        raise PaperArtifactsRefused(
            f"formal target manifest lacks a partition: {split_counts}"
        )
    language_by_id = {str(row["group_id"]): row for row in language_rows}
    target_by_id = {str(row["group_id"]): row for row in target_rows}
    for group_id, query in language_by_id.items():
        target = target_by_id[group_id]
        if (
            query.get("provenance") != "derived"
            or query.get("is_unique") is not True
            or list(query.get("resolver_result", []))
            != [int(target["target_object_id"])]
            or not str(query.get("query", "")).strip()
            or not str(query.get("template_family", "")).strip()
        ):
            raise PaperArtifactsRefused(
                f"language manifest contains a non-auditable query: {group_id}"
            )

    split_path, split_payload = _read_json(
        run_dir / "split_manifest.json", "scene split manifest"
    )
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if split_hash != _digest(manifest.get("split_hash"), "scene split hash"):
        raise PaperArtifactsRefused("scene split manifest is stale")
    return (
        target_rows,
        language_rows,
        {
            str(target_path): target_hash,
            str(language_path): language_hash,
            str(root_language): language_hash,
            str(report_path): sha256_file(report_path),
            str(split_path): sha256_file(split_path),
        },
    )


def _validate_selected_condition(run_dir: Path) -> tuple[str, dict[str, str]]:
    selection_path, payload = _read_json(
        run_dir / "predicted_condition_selection.json",
        "validation-only predicted-condition selection",
    )
    check = dict(payload)
    observed_fingerprint = check.pop("selection_fingerprint", None)
    if observed_fingerprint != canonical_sha256(check):
        raise PaperArtifactsRefused(
            "predicted-condition selection fingerprint is stale"
        )
    if (
        payload.get("schema_version") != PREDICTED_SELECTION_SCHEMA
        or payload.get("scope") != FORMAL_SCOPE
        or payload.get("fixture_only") is not False
        or payload.get("selection_split") != "validation"
        or int(payload.get("test_rows_consumed", -1)) != 0
        or payload.get("selection_metric") != "validation_mean_iou"
    ):
        raise PaperArtifactsRefused(
            "predicted condition was not selected from formal validation-only mIoU"
        )
    selected = str(payload.get("selected_condition", ""))
    if selected not in PREDICTED_CONDITIONS:
        raise PaperArtifactsRefused("selected predicted condition is invalid")
    paths = payload.get("source_paths")
    hashes = payload.get("source_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise PaperArtifactsRefused("predicted selection lacks source bindings")
    sources = {str(selection_path): sha256_file(selection_path)}
    for condition in PREDICTED_CONDITIONS:
        path = _bound_file(
            selection_path,
            paths.get(condition),
            hashes.get(condition),
            f"{condition} validation selection evidence",
            run_dir=run_dir,
        )
        _, summary = _read_json(path, f"{condition} validation metrics")
        if (
            summary.get("schema_version") != GROUNDING_METRICS_SCHEMA
            or summary.get("scope") != FORMAL_SCOPE
            or summary.get("fixture_only") is not False
            or summary.get("condition") != condition
            or summary.get("included_splits") != ["validation"]
            or summary.get("selection_scope") is not True
            or int(summary.get("test_rows_consumed_for_selection", -1)) != 0
        ):
            raise PaperArtifactsRefused(
                f"{condition} selection evidence is not validation-only formal data"
            )
        sources[str(path)] = sha256_file(path)
    return selected, sources


def _validate_grounding(run_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    aggregate_path = _regular_file(
        run_dir / "grounding_metrics.csv", "grounding metrics"
    )
    aggregate_records = _csv_records(aggregate_path, "grounding metrics")
    expected_records: list[dict[str, str]] = []
    sources = {str(aggregate_path): sha256_file(aggregate_path)}
    for condition in PREDICTED_CONDITIONS:
        summary_path, summary = _read_json(
            run_dir
            / "grounding_metrics"
            / "all_requested_splits"
            / condition
            / "summary.json",
            f"{condition} grounding summary",
        )
        if (
            summary.get("schema_version") != GROUNDING_METRICS_SCHEMA
            or summary.get("scope") != FORMAL_SCOPE
            or summary.get("fixture_only") is not False
            or summary.get("condition") != condition
            or summary.get("included_splits")
            != [
                "train",
                "validation",
                "test",
            ]
            or summary.get("selection_scope") is not False
            or int(summary.get("test_rows_consumed_for_selection", -1)) != 0
        ):
            raise PaperArtifactsRefused(
                f"{condition} grounding summary is not complete formal evidence"
            )
        raw_path = _bound_file(
            summary_path,
            summary.get("raw_metrics_path"),
            summary.get("raw_metrics_sha256"),
            f"{condition} raw grounding metrics",
            run_dir=run_dir,
        )
        source_hashes = summary.get("source_hashes")
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise PaperArtifactsRefused(
                f"{condition} grounding summary lacks input source hashes"
            )
        for raw_source, digest in source_hashes.items():
            source = _regular_file(raw_source, f"{condition} grounding source")
            observed = sha256_file(source)
            if observed != _digest(digest, f"{condition} grounding source hash"):
                raise PaperArtifactsRefused(
                    f"{condition} grounding input changed after metric computation"
                )
            sources[str(source)] = observed
        records = _csv_records(raw_path, f"{condition} raw grounding metrics")
        if any(row.get("condition") != condition for row in records):
            raise PaperArtifactsRefused(
                f"{condition} grounding rows contain another condition"
            )
        expected_records.extend(records)
        sources[str(summary_path)] = sha256_file(summary_path)
        sources[str(raw_path)] = sha256_file(raw_path)
    if aggregate_records != expected_records:
        raise PaperArtifactsRefused(
            "grounding_metrics.csv is not the exact committed predicted-arm aggregate"
        )
    frame = _read_csv(aggregate_path, "grounding metrics")
    required = {"group_id", "scene_id", "split", "condition", "iou"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PaperArtifactsRefused(f"grounding metrics lack columns: {missing}")
    _finite_column(frame, "iou", "grounding metrics")
    if set(frame["condition"].astype(str)) != set(PREDICTED_CONDITIONS):
        raise PaperArtifactsRefused("grounding metrics lack one predicted condition")
    return frame, sources


def _validate_candidate_summary(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_path = _regular_file(
        run_dir / "candidate_manifest.jsonl", "candidate manifest"
    )
    summary_path = _regular_file(
        run_dir / "candidate_pool_summary.csv", "candidate pool summary"
    )
    rows = _read_jsonl(manifest_path, "candidate manifest")
    summary_rows = _csv_records(summary_path, "candidate pool summary")
    normalised = [
        {key: _normalise_csv_value(value) for key, value in row.items()} for row in rows
    ]
    if normalised != summary_rows:
        raise PaperArtifactsRefused(
            "candidate_pool_summary.csv differs from candidate_manifest.jsonl"
        )
    if canonical_sha256(rows) != _digest(
        run_manifest.get("candidate_cache_hash"), "candidate cache hash"
    ):
        raise PaperArtifactsRefused("candidate manifest differs from run provenance")
    expected_groups = {str(row["group_id"]) for row in target_rows}
    observed_pairs: set[tuple[str, str]] = set()
    sources = {
        str(manifest_path): sha256_file(manifest_path),
        str(summary_path): sha256_file(summary_path),
    }
    for row in rows:
        group_id = str(row.get("group_id", ""))
        condition = str(row.get("grounding_condition", ""))
        pair = (condition, group_id)
        if condition not in CONDITIONS or group_id not in expected_groups:
            raise PaperArtifactsRefused(
                f"candidate manifest contains an unknown condition/group: {pair}"
            )
        if pair in observed_pairs:
            raise PaperArtifactsRefused(f"candidate manifest duplicates {pair}")
        observed_pairs.add(pair)
        bundle = _bound_file(
            manifest_path,
            row.get("bundle_path"),
            row.get("bundle_sha256"),
            f"candidate bundle {condition}/{group_id}",
            run_dir=run_dir,
        )
        _, payload = _read_json(bundle, "candidate bundle")
        check = dict(payload)
        fingerprint = check.pop("bundle_fingerprint", None)
        if fingerprint != canonical_sha256(check):
            raise PaperArtifactsRefused(
                f"candidate bundle fingerprint failed: {bundle}"
            )
        if (
            payload.get("group_id") != group_id
            or payload.get("grounding_condition") != condition
            or int(payload.get("candidate_count", -1))
            != int(row.get("candidate_count", -2))
            or payload.get("candidate_pool_fingerprint")
            != row.get("candidate_pool_fingerprint")
        ):
            raise PaperArtifactsRefused(f"candidate bundle index is stale: {bundle}")
        sources[str(bundle)] = sha256_file(bundle)
    expected_pairs = {
        (condition, group_id)
        for condition in CONDITIONS
        for group_id in expected_groups
    }
    if observed_pairs != expected_pairs:
        raise PaperArtifactsRefused(
            "candidate manifest is not the complete condition/group universe"
        )
    stage_path, stage = _read_json(
        run_dir / "stages/candidates.json", "candidate stage"
    )
    candidate_index = stage.get("candidate_index")
    if not isinstance(candidate_index, Mapping):
        raise PaperArtifactsRefused("candidate stage lacks its published index")
    if candidate_index.get("candidate_manifest_sha256") != sha256_file(
        manifest_path
    ) or candidate_index.get("candidate_pool_summary_sha256") != sha256_file(
        summary_path
    ):
        raise PaperArtifactsRefused("candidate stage index hashes are stale")
    sources[str(stage_path)] = sha256_file(stage_path)
    frame = _read_csv(summary_path, "candidate pool summary")
    _finite_column(frame, "candidate_count", "candidate pool summary")
    return frame, sources


def _validate_runtime(run_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    path = _regular_file(run_dir / "runtime.csv", "runtime table")
    frame = _read_csv(path, "runtime table")
    required = {
        "stage",
        "status",
        "wall_time_s",
        "process_rss_bytes_at_end",
        "resume_requested",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PaperArtifactsRefused(f"runtime table lacks columns: {missing}")
    if frame["stage"].astype(str).duplicated().any():
        raise PaperArtifactsRefused("runtime table duplicates a stage")
    _finite_column(frame, "wall_time_s", "runtime table")
    if (pd.to_numeric(frame["wall_time_s"]) < 0).any():
        raise PaperArtifactsRefused("runtime table contains a negative duration")
    required_runtime = set(_REQUIRED_RUNTIME_STAGES)
    completed = set(
        frame.loc[frame["status"].astype(str).eq("COMPLETE"), "stage"].astype(str)
    )
    if not required_runtime.issubset(completed):
        raise PaperArtifactsRefused(
            f"runtime evidence lacks completed stages: {sorted(required_runtime - completed)}"
        )
    # The report row is necessarily measured after the report and its runtime
    # figure are committed.  Bind the renderer to the immutable upstream slice
    # to avoid a self-invalidating paper manifest when the CLI appends that row.
    upstream = frame.loc[frame["stage"].astype(str).isin(required_runtime)].copy()
    order = {stage: index for index, stage in enumerate(_REQUIRED_RUNTIME_STAGES)}
    upstream["_paper_order"] = upstream["stage"].astype(str).map(order)
    upstream = (
        upstream.sort_values("_paper_order", kind="mergesort")
        .drop(columns="_paper_order")
        .reset_index(drop=True)
    )
    evidence_key = f"{path}#required_pre_report_rows"
    return upstream, {evidence_key: _runtime_evidence_hash(upstream)}


def _runtime_evidence_hash(frame: pd.DataFrame) -> str:
    columns = (
        "stage",
        "status",
        "wall_time_s",
        "process_rss_bytes_at_end",
        "resume_requested",
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PaperArtifactsRefused(f"runtime evidence hash lacks columns: {missing}")
    records = frame.loc[:, columns].to_dict(orient="records")
    return canonical_sha256(records)


def _validate_analysis(run_dir: Path, name: str) -> AnalysisSnapshot:
    root = run_dir / "analysis" / name
    manifest_path, manifest = _read_json(
        root / "analysis_manifest.json", f"{name} analysis manifest"
    )
    if (
        manifest.get("schema_version") != ANALYSIS_SCHEMA
        or manifest.get("status") != "COMPLETE"
        or manifest.get("analysis_scope") != FORMAL_SCOPE
        or manifest.get("fixture_only") is not False
        or manifest.get("formal_report_eligible") is not True
        or manifest.get("run_id") != run_dir.name
        or manifest.get("frozen_pool_status") != "PASS"
    ):
        raise PaperArtifactsRefused(f"{name} is not a COMPLETE formal analysis")
    guards = manifest.get("split_guards")
    if (
        not isinstance(guards, Mapping)
        or any(
            guards.get(field) is not True
            for field in (
                "scene_disjoint",
                "group_disjoint",
                "candidate_disjoint",
                "declared_partition_checked",
            )
        )
        or guards.get("test_used_for_training_or_selection") is not False
    ):
        raise PaperArtifactsRefused(f"{name} split/leakage guards did not pass")

    state_path, state = _read_json(
        root / "analysis_state.json", f"{name} analysis state"
    )
    if (
        state.get("status") != "COMPLETE"
        or state.get("analysis_fingerprint") != manifest.get("analysis_fingerprint")
        or state.get("output_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise PaperArtifactsRefused(f"{name} analysis state is stale")
    input_path = _bound_file(
        manifest_path,
        manifest.get("input_manifest_path"),
        manifest.get("input_manifest_sha256"),
        f"{name} analysis input manifest",
        run_dir=run_dir,
    )
    try:
        loaded_input = load_analysis_input_manifest(input_path)
    except Exception as error:
        raise PaperArtifactsRefused(
            f"{name} analysis input is invalid: {type(error).__name__}: {error}"
        ) from error
    if (
        loaded_input.status != "COMPLETE"
        or loaded_input.scope != FORMAL_SCOPE
        or loaded_input.fixture_only
        or loaded_input.run_id != run_dir.name
        or input_path.name != "input_manifest.json"
    ):
        raise PaperArtifactsRefused(f"{name} analysis input is not formal and complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PaperArtifactsRefused(f"{name} analysis manifest has no output hashes")
    sources = {
        str(manifest_path): sha256_file(manifest_path),
        str(state_path): sha256_file(state_path),
        str(input_path): sha256_file(input_path),
    }
    paths: dict[str, Path] = {}
    for filename in _REQUIRED_ANALYSIS_OUTPUTS:
        path = _regular_file(root / filename, f"{name}/{filename}")
        expected = _digest(outputs.get(filename), f"{name}/{filename} hash")
        if sha256_file(path) != expected:
            raise PaperArtifactsRefused(f"stale analysis output: {name}/{filename}")
        paths[filename] = path
        sources[str(path)] = expected

    metrics = _read_csv(paths["metrics.csv"], f"{name} metrics")
    per_group = _read_csv(paths["per_group_metrics.csv"], f"{name} per-group metrics")
    paired = _read_csv(paths["paired_outcomes.csv"], f"{name} paired outcomes")
    bootstrap = _read_csv(paths["bootstrap_results.csv"], f"{name} bootstrap")
    ablations = _read_csv(paths["ablation_results.csv"], f"{name} ablations")
    failures = _read_csv(paths["failure_taxonomy.csv"], f"{name} failures")
    native = _read_csv(paths["native_predictions.csv"], f"{name} native predictions")
    reranked = _read_csv(
        paths["reranked_predictions.csv"], f"{name} reranked predictions"
    )
    for description, frame in (
        ("metrics", metrics),
        ("per-group metrics", per_group),
        ("paired outcomes", paired),
        ("bootstrap", bootstrap),
        ("ablations", ablations),
        ("failures", failures),
    ):
        _scope_is_formal(frame, f"{name} {description}")
    systems = set(metrics["system"].astype(str))
    if not set(PRIMARY_SYSTEMS).issubset(systems):
        raise PaperArtifactsRefused(f"{name} metrics lack a required system")
    seed_rows = metrics.loc[metrics["system"].astype(str).eq("R0_RAW_SEED")]
    observed_seeds = set(pd.to_numeric(seed_rows["seed"]).astype(int))
    if observed_seeds != set(FORMAL_SEEDS):
        raise PaperArtifactsRefused(f"{name} metrics lack the three locked seeds")
    if (
        not bootstrap["iterations"].astype(int).eq(10_000).all()
        or not bootstrap["resampling_unit"].astype(str).eq("scene").all()
        or not bootstrap["confidence"].astype(float).eq(0.95).all()
    ):
        raise PaperArtifactsRefused(f"{name} bootstrap is not the locked formal test")
    if failures["group_id"].astype(str).duplicated().any():
        raise PaperArtifactsRefused(f"{name} failure taxonomy is not exclusive")
    pool_path, pool = _read_json(paths["frozen_pool_audit.json"], f"{name} pool audit")
    if pool.get("status") != "PASS" or pool.get("analysis_scope") != FORMAL_SCOPE:
        raise PaperArtifactsRefused(f"{name} frozen candidate audit did not pass")
    metrics_json_path, metrics_json = _read_json(
        paths["metrics.json"], f"{name} metric payload"
    )
    significance_path, significance = _read_json(
        paths["significance_tests.json"], f"{name} significance tests"
    )
    if (
        metrics_json.get("analysis_scope") != FORMAL_SCOPE
        or significance.get("analysis_scope") != FORMAL_SCOPE
    ):
        raise PaperArtifactsRefused(f"{name} JSON metrics are not formal")
    expected_conditions = set(CONDITIONS) if name == "combined_a8" else {name}
    for description, frame in (
        ("native predictions", native),
        ("reranked predictions", reranked),
    ):
        if "grounding_condition" not in frame.columns:
            raise PaperArtifactsRefused(f"{name} {description} lacks condition")
        observed = set(frame["grounding_condition"].astype(str))
        if observed != expected_conditions:
            raise PaperArtifactsRefused(
                f"{name} {description} condition universe is {sorted(observed)}"
            )
    return AnalysisSnapshot(
        name=name,
        root=root,
        manifest=manifest,
        metrics=metrics,
        per_group=per_group,
        paired=paired,
        bootstrap=bootstrap,
        ablations=ablations,
        failures=failures,
        native=native,
        reranked=reranked,
        significance=significance,
        metrics_json=metrics_json,
        source_hashes=sources,
    )


def validate_formal_paper_inputs(run_dir: Path | str) -> FormalPaperInputs:
    """Load and revalidate every source needed for formal paper artifacts.

    This function is read-only.  It is useful as a preflight independent of
    rendering and raises :class:`PaperArtifactsRefused` on the first failed
    evidence boundary.
    """

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise PaperArtifactsRefused(f"formal run directory is invalid: {root}")
    run_manifest, config = _validate_run_manifest(root)
    sources: dict[str, str] = {
        str(root / "resolved_config.yaml"): sha256_file(root / "resolved_config.yaml"),
        f"{root / 'run_manifest.json'}#upstream_fields": _paper_run_manifest_hash(
            run_manifest
        ),
    }
    sources.update(_validate_stage_gates(root))
    target_rows, language_rows, manifest_sources = _validate_manifests(
        root, run_manifest
    )
    sources.update(manifest_sources)
    selected, selection_sources = _validate_selected_condition(root)
    sources.update(selection_sources)
    grounding, grounding_sources = _validate_grounding(root)
    sources.update(grounding_sources)
    candidates, candidate_sources = _validate_candidate_summary(
        root, run_manifest, target_rows
    )
    sources.update(candidate_sources)
    runtime, runtime_sources = _validate_runtime(root)
    sources.update(runtime_sources)
    analyses = {name: _validate_analysis(root, name) for name in ANALYSES}
    for snapshot in analyses.values():
        sources.update(snapshot.source_hashes)
    identity = run_manifest["immutable_identity"]
    input_fingerprint = canonical_sha256(
        {
            "schema": PAPER_ARTIFACT_SCHEMA,
            "run_id": root.name,
            "immutable_identity_sha256": canonical_sha256(identity),
            "selected_predicted_condition": selected,
            "source_hashes": dict(sorted(sources.items())),
        }
    )
    return FormalPaperInputs(
        run_dir=root,
        run_manifest=run_manifest,
        config=config,
        target_rows=target_rows,
        language_rows=language_rows,
        grounding=grounding,
        candidate_summary=candidates,
        runtime=runtime,
        selected_condition=selected,
        analyses=analyses,
        source_hashes=dict(sorted(sources.items())),
        input_fingerprint=input_fingerprint,
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return atomic_text(
        path,
        frame.to_csv(index=False, lineterminator="\n", float_format="%.17g"),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    rows = [
        ["" if pd.isna(value) else str(value) for value in record]
        for record in frame.itertuples(index=False, name=None)
    ]

    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(escape(value) for value in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(escape(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _write_table(
    run_dir: Path,
    name: str,
    frame: pd.DataFrame,
    *,
    sources: Sequence[Path],
) -> _TableRecord:
    if frame.empty:
        raise PaperArtifactsRefused(f"refusing to publish empty table {name}")
    path = _atomic_csv(run_dir / "tables" / f"{name}.csv", frame)
    atomic_text(
        run_dir / "tables" / f"{name}.md",
        f"# {name.replace('_', ' ').title()}\n\n{_markdown_table(frame)}\n",
    )
    return _TableRecord(name, path, len(frame), tuple(sources))


def _main_result_table(inputs: FormalPaperInputs) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        snapshot = inputs.analyses[condition]
        group_lookup = {
            system: frame
            for system, frame in snapshot.per_group.groupby("system", sort=False)
        }
        for system in PRIMARY_SYSTEMS:
            selected = snapshot.metrics.loc[
                snapshot.metrics["system"].astype(str).eq(system)
            ]
            if len(selected) != 1:
                raise PaperArtifactsRefused(
                    f"{condition} has {len(selected)} aggregate rows for {system}"
                )
            row = selected.iloc[0]
            groups = group_lookup.get(system)
            if groups is None or groups.empty:
                raise PaperArtifactsRefused(
                    f"{condition} has no per-group rows for {system}"
                )
            count = len(groups)
            record: dict[str, Any] = {
                "condition": condition,
                "condition_role": (
                    "oracle_grounding_counterfactual"
                    if condition == "oracle_gt_mask"
                    else (
                        "validation_selected_complete_pipeline"
                        if condition == inputs.selected_condition
                        else "non_selected_predicted_ablation"
                    )
                ),
                "system": system,
                "test_groups": count,
                "candidate_count": int(row["candidate_count"]),
            }
            for mu in (0.4, 0.8, 1.2):
                column = f"top1_success_mu_{mu:.1f}"
                successes = int(_bool_series(groups[column], column).sum())
                record[f"p_at_1_mu_{mu:.1f}_numerator"] = successes
                record[f"p_at_1_mu_{mu:.1f}_denominator"] = count
                record[f"p_at_1_mu_{mu:.1f}"] = float(successes / count)
            record.update(
                {
                    "target_specific_ap_mean_mu_0.2_to_1.2": float(
                        row["target_graspnet_style_ap_mean_mu_0.2_to_1.2"]
                    ),
                    "mrr_mu_1.2": float(row["mrr"]),
                    "oracle_at_5": float(row["oracle_at_5"]),
                    "oracle_at_10": float(row["oracle_at_10"]),
                    "oracle_at_50": float(row["oracle_at_50"]),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def _paired_result_table(inputs: FormalPaperInputs) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        snapshot = inputs.analyses[condition]
        tests = snapshot.significance.get("tests")
        if not isinstance(tests, Mapping):
            raise PaperArtifactsRefused(f"{condition} significance tests are absent")
        for challenger in ("R0_RAW", "R1_GATED"):
            paired = snapshot.paired.loc[
                snapshot.paired["challenger"].astype(str).eq(challenger)
            ]
            if paired.empty:
                raise PaperArtifactsRefused(
                    f"{condition} paired outcomes lack {challenger}"
                )
            recovered = int(_bool_series(paired["recovered"], "recovered").sum())
            harmful = int(_bool_series(paired["harmful"], "harmful").sum())
            interval = snapshot.bootstrap.loc[
                snapshot.bootstrap["challenger"].astype(str).eq(challenger)
                & snapshot.bootstrap["metric"].astype(str).eq("delta_p_at_1")
            ]
            if len(interval) != 1 or challenger not in tests:
                raise PaperArtifactsRefused(
                    f"{condition} lacks one paired interval/test for {challenger}"
                )
            boot = interval.iloc[0]
            test = tests[challenger]
            if not isinstance(test, Mapping):
                raise PaperArtifactsRefused(
                    f"{condition} significance record is malformed"
                )
            records.append(
                {
                    "condition": condition,
                    "challenger": challenger,
                    "test_groups": len(paired),
                    "recovered": recovered,
                    "harmful": harmful,
                    "net": recovered - harmful,
                    "delta_p_at_1": float(boot["point_estimate"]),
                    "ci_low": float(boot["ci_low"]),
                    "ci_high": float(boot["ci_high"]),
                    "confidence": float(boot["confidence"]),
                    "scene_count": int(boot["scene_count"]),
                    "mcnemar_pvalue": float(test["pvalue"]),
                }
            )
    return pd.DataFrame(records)


def _grounding_result_table(inputs: FormalPaperInputs) -> pd.DataFrame:
    rows = inputs.grounding.copy()
    rows["iou"] = pd.to_numeric(rows["iou"], errors="coerce")
    rows["empty_prediction"] = _bool_series(
        rows["empty_prediction"], "grounding empty_prediction"
    )
    records: list[dict[str, Any]] = []
    for (condition, split), part in rows.groupby(["condition", "split"], sort=True):
        records.append(
            {
                "condition": str(condition),
                "split": str(split),
                "groups": len(part),
                "mean_iou": float(part["iou"].mean()),
                "empty_predictions": int(part["empty_prediction"].sum()),
                "empty_prediction_rate": float(part["empty_prediction"].mean()),
            }
        )
    return pd.DataFrame(records)


def _candidate_coverage_table(inputs: FormalPaperInputs) -> pd.DataFrame:
    rows = inputs.candidate_summary.copy()
    rows["candidate_count"] = pd.to_numeric(rows["candidate_count"], errors="coerce")
    records: list[dict[str, Any]] = []
    for (condition, split), part in rows.groupby(
        ["grounding_condition", "split"], sort=True
    ):
        records.append(
            {
                "condition": str(condition),
                "split": str(split),
                "groups": len(part),
                "non_empty_groups": int((part["candidate_count"] > 0).sum()),
                "empty_groups": int((part["candidate_count"] == 0).sum()),
                "mean_candidates": float(part["candidate_count"].mean()),
                "median_candidates": float(part["candidate_count"].median()),
            }
        )
    return pd.DataFrame(records)


def _ablation_result_table(inputs: FormalPaperInputs) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    wanted = {
        "target_p_at_1_mu_1.2",
        "target_graspnet_style_ap_mean_mu_0.2_to_1.2",
    }
    for condition in CONDITIONS:
        frame = inputs.analyses[condition].ablations.copy()
        selected = frame.loc[
            frame["metric_name"].astype(str).isin(wanted) | frame["metric_name"].isna()
        ].copy()
        selected.insert(0, "condition", condition)
        records.append(selected)
    return pd.concat(records, ignore_index=True)


def _failure_result_table(inputs: FormalPaperInputs) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        frame = inputs.analyses[condition].failures
        for category, part in frame.groupby("category", sort=True):
            records.append(
                {
                    "condition": condition,
                    "category": str(category),
                    "groups": len(part),
                    "denominator": len(frame),
                    "fraction": float(len(part) / len(frame)),
                }
            )
    return pd.DataFrame(records)


def _publish_tables(
    inputs: FormalPaperInputs,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    run_dir = inputs.run_dir
    frames = {
        "main_results": _main_result_table(inputs),
        "paired_results": _paired_result_table(inputs),
        "grounding_results": _grounding_result_table(inputs),
        "candidate_coverage": _candidate_coverage_table(inputs),
        "ablation_summary": _ablation_result_table(inputs),
        "failure_summary": _failure_result_table(inputs),
    }
    records: list[_TableRecord] = []
    analysis_sources = tuple(
        snapshot.root / "metrics.csv" for snapshot in inputs.analyses.values()
    )
    records.append(
        _write_table(
            run_dir,
            "main_results",
            frames["main_results"],
            sources=analysis_sources,
        )
    )
    records.append(
        _write_table(
            run_dir,
            "paired_results",
            frames["paired_results"],
            sources=tuple(
                path
                for snapshot in inputs.analyses.values()
                for path in (
                    snapshot.root / "paired_outcomes.csv",
                    snapshot.root / "bootstrap_results.csv",
                    snapshot.root / "significance_tests.json",
                )
            ),
        )
    )
    records.append(
        _write_table(
            run_dir,
            "grounding_results",
            frames["grounding_results"],
            sources=(run_dir / "grounding_metrics.csv",),
        )
    )
    records.append(
        _write_table(
            run_dir,
            "candidate_coverage",
            frames["candidate_coverage"],
            sources=(run_dir / "candidate_pool_summary.csv",),
        )
    )
    records.append(
        _write_table(
            run_dir,
            "ablation_summary",
            frames["ablation_summary"],
            sources=tuple(
                inputs.analyses[condition].root / "ablation_results.csv"
                for condition in CONDITIONS
            ),
        )
    )
    records.append(
        _write_table(
            run_dir,
            "failure_summary",
            frames["failure_summary"],
            sources=tuple(
                inputs.analyses[condition].root / "failure_taxonomy.csv"
                for condition in CONDITIONS
            ),
        )
    )
    payload = {
        "schema_version": TABLE_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "scope": FORMAL_SCOPE,
        "input_fingerprint": inputs.input_fingerprint,
        "tables": {
            record.name: {
                "path": str(record.path.relative_to(run_dir)),
                "sha256": sha256_file(record.path),
                "markdown_path": str(
                    (run_dir / "tables" / f"{record.name}.md").relative_to(run_dir)
                ),
                "markdown_sha256": sha256_file(
                    run_dir / "tables" / f"{record.name}.md"
                ),
                "row_count": record.rows,
                "sources": {
                    str(path): sha256_file(path) for path in record.source_paths
                },
            }
            for record in records
        },
    }
    manifest_path = atomic_json(run_dir / "tables" / "table_manifest.json", payload)
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return frames, payload


def _plot_context() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }


def _save_figure(fig: Any, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    temporary_pdf = pdf.with_name(f".{pdf.name}.{os.getpid()}.tmp")
    temporary_png = png.with_name(f".{png.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(
            temporary_pdf,
            format="pdf",
            bbox_inches="tight",
            metadata={"Creator": "graspnet6d.paper_artifacts", "CreationDate": None},
        )
        fig.savefig(
            temporary_png,
            format="png",
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "graspnet6d.paper_artifacts"},
        )
        os.replace(temporary_pdf, pdf)
        os.replace(temporary_png, png)
    finally:
        for path in (temporary_pdf, temporary_png):
            if path.exists():
                path.unlink()
    return pdf, png


def _grouped_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    hue: str,
    y: str,
    ylabel: str,
    title: str,
    rotate: bool = False,
) -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    categories = list(dict.fromkeys(frame[x].astype(str)))
    series = list(dict.fromkeys(frame[hue].astype(str)))
    fig, axis = plt.subplots(figsize=(max(4.0, len(categories) * 0.8), 2.75))
    positions = np.arange(len(categories), dtype=float)
    width = 0.78 / max(1, len(series))
    for index, value in enumerate(series):
        part = frame.loc[frame[hue].astype(str).eq(value)].set_index(x)
        heights = np.asarray(
            [float(part.loc[category, y]) for category in categories], dtype=float
        )
        offset = (index - (len(series) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            heights,
            width=width * 0.92,
            label=value,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            edgecolor="black",
            linewidth=0.4,
        )
    axis.set_xticks(positions, categories, rotation=28 if rotate else 0)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False, ncol=min(3, len(series)))
    axis.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    return fig


def _simple_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    ylabel: str,
    title: str,
    colors: Sequence[str] | None = None,
    rotate: bool = False,
) -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(max(4.0, len(frame) * 0.55), 2.75))
    palette = list(colors or [OKABE_ITO[1]])
    axis.bar(
        np.arange(len(frame)),
        frame[y].astype(float),
        color=[palette[index % len(palette)] for index in range(len(frame))],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(
        np.arange(len(frame)), frame[x].astype(str), rotation=28 if rotate else 0
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_axisbelow(True)
    fig.tight_layout(pad=0.6)
    return fig


def _record_figure(
    inputs: FormalPaperInputs,
    figure_id: str,
    title: str,
    frame: pd.DataFrame | None,
    *,
    source_paths: Sequence[Path],
    source_hashes: Mapping[str, str] | None = None,
    plotter: Callable[[pd.DataFrame], Any] | None,
    reason_code: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    bound_sources = {
        str(path): sha256_file(path) for path in source_paths if path.is_file()
    }
    if source_hashes is not None:
        for name, digest in source_hashes.items():
            bound_sources[str(name)] = _digest(
                digest, f"figure {figure_id} source hash"
            )
    if frame is None or frame.empty or plotter is None:
        return {
            "id": figure_id,
            "title": title,
            "status": reason_code or "UNEXECUTED_INPUT_MISSING",
            "reason": reason or "required saved rows or columns are absent",
            "source_files": bound_sources,
            "source_data": None,
            "outputs": {},
        }
    if frame.select_dtypes(include=[np.number]).empty:
        raise PaperArtifactsRefused(f"figure {figure_id} has no numeric source data")
    source_data = _atomic_csv(
        inputs.run_dir / "figures" / "source_data" / f"{figure_id}.csv", frame
    )
    import matplotlib.pyplot as plt

    with plt.rc_context(_plot_context()):
        figure = plotter(frame)
        pdf, png = _save_figure(figure, inputs.run_dir / "figures" / figure_id)
        plt.close(figure)
    return {
        "id": figure_id,
        "title": title,
        "status": "EXECUTED",
        "reason": None,
        "source_files": bound_sources,
        "source_data": {
            "path": str(source_data.relative_to(inputs.run_dir)),
            "sha256": sha256_file(source_data),
            "row_count": len(frame),
        },
        "outputs": {
            "pdf": {
                "path": str(pdf.relative_to(inputs.run_dir)),
                "sha256": sha256_file(pdf),
            },
            "png": {
                "path": str(png.relative_to(inputs.run_dir)),
                "sha256": sha256_file(png),
            },
        },
    }


def _performance_bins(
    inputs: FormalPaperInputs,
    *,
    value_by_group: pd.DataFrame,
    value_column: str,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
) -> pd.DataFrame:
    snapshot = inputs.analyses[inputs.selected_condition]
    systems = snapshot.per_group.loc[
        snapshot.per_group["system"].astype(str).isin(["B0_NATIVE", "R0_RAW"]),
        ["group_id", "system", "top1_success_mu_1.2"],
    ].copy()
    joined = systems.merge(value_by_group, on="group_id", validate="many_to_one")
    if joined.empty:
        return pd.DataFrame()
    joined["bin"] = pd.cut(
        pd.to_numeric(joined[value_column], errors="coerce"),
        bins=list(bin_edges),
        labels=list(bin_labels),
        include_lowest=True,
        right=True,
    )
    joined = joined.dropna(subset=["bin"])
    joined["success"] = _bool_series(
        joined["top1_success_mu_1.2"], "stratified Top-1"
    ).astype(float)
    records: list[dict[str, Any]] = []
    for (bin_name, system), part in joined.groupby(
        ["bin", "system"], observed=True, sort=True
    ):
        records.append(
            {
                "bin": str(bin_name),
                "system": str(system),
                "groups": len(part),
                "successes": int(part["success"].sum()),
                "p_at_1_mu_1.2": float(part["success"].mean()),
            }
        )
    return pd.DataFrame(records)


def _build_figure_specs(
    inputs: FormalPaperInputs, tables: Mapping[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    main = tables["main_results"]
    compact = main.loc[main["system"].isin(["B0_NATIVE", "R0_RAW", "R1_GATED"])].copy()
    compact["condition"] = compact["condition"].map(
        {
            "oracle_gt_mask": "GT mask (oracle)",
            "hifics_zero_shot_mask": "HiFi zero-shot",
            "hifics_adapted_mask": "HiFi adapted",
        }
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[0][0],
            "title": FIGURE_FAMILIES[0][1],
            "frame": compact[["condition", "system", "p_at_1_mu_1.2"]],
            "sources": tuple(
                inputs.analyses[c].root / "metrics.csv" for c in CONDITIONS
            ),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="condition",
                hue="system",
                y="p_at_1_mu_1.2",
                ylabel="Target P@1 (mu=1.2)",
                title="Frozen VGN candidate pools",
                rotate=True,
            ),
        }
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[1][0],
            "title": FIGURE_FAMILIES[1][1],
            "frame": compact[
                [
                    "condition",
                    "system",
                    "target_specific_ap_mean_mu_0.2_to_1.2",
                ]
            ],
            "sources": tuple(
                inputs.analyses[c].root / "metrics.csv" for c in CONDITIONS
            ),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="condition",
                hue="system",
                y="target_specific_ap_mean_mu_0.2_to_1.2",
                ylabel="Target-specific mean AP",
                title="Offline target-specific evaluator metric",
                rotate=True,
            ),
        }
    )

    oracle_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        source = (
            inputs.analyses[condition]
            .metrics.loc[inputs.analyses[condition].metrics["system"].eq("B0_NATIVE")]
            .iloc[0]
        )
        for k in (1, 5, 10, 20, 50):
            oracle_records.append(
                {
                    "condition": condition,
                    "k": k,
                    "oracle_rate": float(source[f"oracle_at_{k}"]),
                }
            )
    oracle_frame = pd.DataFrame(oracle_records)

    def oracle_plot(frame: pd.DataFrame) -> Any:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(4.6, 2.75))
        for index, (condition, part) in enumerate(
            frame.groupby("condition", sort=False)
        ):
            axis.plot(
                part["k"],
                part["oracle_rate"],
                marker=("o", "s", "^")[index],
                label=condition,
                color=OKABE_ITO[index + 1],
            )
        axis.set_xscale("log")
        axis.set_xticks([1, 5, 10, 20, 50], ["1", "5", "10", "20", "50"])
        axis.set_xlabel("K (native candidate order)")
        axis.set_ylabel("Oracle@K (mu=1.2)")
        axis.set_title("Frozen-pool target-grasp ceiling")
        axis.legend(frameon=False)
        figure.tight_layout(pad=0.6)
        return figure

    specs.append(
        {
            "id": FIGURE_FAMILIES[2][0],
            "title": FIGURE_FAMILIES[2][1],
            "frame": oracle_frame,
            "sources": tuple(
                inputs.analyses[c].root / "metrics.csv" for c in CONDITIONS
            ),
            "plotter": oracle_plot,
        }
    )

    paired = (
        tables["paired_results"]
        .loc[tables["paired_results"]["challenger"].eq("R0_RAW")]
        .copy()
    )
    outcome = paired.melt(
        id_vars=["condition"],
        value_vars=["recovered", "harmful", "net"],
        var_name="outcome",
        value_name="groups",
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[3][0],
            "title": FIGURE_FAMILIES[3][1],
            "frame": outcome,
            "sources": tuple(
                inputs.analyses[c].root / "paired_outcomes.csv" for c in CONDITIONS
            ),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="condition",
                hue="outcome",
                y="groups",
                ylabel="Test groups",
                title="Raw reranker versus native",
                rotate=True,
            ),
        }
    )

    selected_ablation = inputs.analyses[inputs.selected_condition].ablations.copy()
    feature = selected_ablation.loc[
        selected_ablation["ablation"]
        .astype(str)
        .isin([f"A{index}" for index in range(7)])
        & selected_ablation["status"].astype(str).eq("EXECUTED")
        & selected_ablation["metric_name"].astype(str).eq("target_p_at_1_mu_1.2")
    ][["ablation", "setting", "metric_value"]].copy()
    feature["label"] = (
        feature["ablation"].astype(str) + ": " + feature["setting"].astype(str)
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[4][0],
            "title": FIGURE_FAMILIES[4][1],
            "frame": feature[["label", "metric_value"]],
            "sources": (
                inputs.analyses[inputs.selected_condition].root
                / "ablation_results.csv",
            ),
            "plotter": lambda frame: _simple_bar(
                frame,
                x="label",
                y="metric_value",
                ylabel="Target P@1 (mu=1.2)",
                title=f"Feature contract: {inputs.selected_condition}",
                rotate=True,
            ),
        }
    )

    top_k_rows = selected_ablation.loc[
        selected_ablation["ablation"].astype(str).eq("A7")
        & selected_ablation["status"].astype(str).eq("EXECUTED")
        & selected_ablation["metric_name"].astype(str).eq("target_p_at_1_mu_1.2")
    ][["setting", "metric_value", "actual_candidate_count"]].copy()
    top_k_reason_rows = selected_ablation.loc[
        selected_ablation["ablation"].astype(str).eq("A7")
        & ~selected_ablation["status"].astype(str).eq("EXECUTED")
    ]
    specs.append(
        {
            "id": FIGURE_FAMILIES[5][0],
            "title": FIGURE_FAMILIES[5][1],
            "frame": top_k_rows if not top_k_rows.empty else None,
            "sources": (
                inputs.analyses[inputs.selected_condition].root
                / "ablation_results.csv",
            ),
            "plotter": (
                None
                if top_k_rows.empty
                else lambda frame: _simple_bar(
                    frame,
                    x="setting",
                    y="metric_value",
                    ylabel="Target P@1 (mu=1.2)",
                    title=f"Frozen pre-NMS Top-K: {inputs.selected_condition}",
                )
            ),
            "reason_code": "UNEXECUTED_INPUT_MISSING",
            "reason": (
                "; ".join(sorted(set(top_k_reason_rows["reason"].dropna().astype(str))))
                or "A7 has no executed saved rows"
            ),
        }
    )

    combined = inputs.analyses["combined_a8"].ablations
    gt_predicted = combined.loc[
        combined["ablation"].astype(str).eq("A8")
        & combined["status"].astype(str).eq("EXECUTED")
        & combined["metric_name"].astype(str).eq("target_p_at_1_mu_1.2")
    ][["setting", "metric_value", "actual_candidate_count"]].copy()
    specs.append(
        {
            "id": FIGURE_FAMILIES[6][0],
            "title": FIGURE_FAMILIES[6][1],
            "frame": gt_predicted,
            "sources": (inputs.analyses["combined_a8"].root / "ablation_results.csv",),
            "plotter": lambda frame: _simple_bar(
                frame,
                x="setting",
                y="metric_value",
                ylabel="Raw-reranker P@1 (mu=1.2)",
                title="Oracle grounding versus predicted grounding",
                rotate=True,
            ),
        }
    )

    candidates = inputs.candidate_summary.copy()
    candidates = candidates.loc[candidates["split"].astype(str).eq("test")]
    candidates["candidate_count"] = pd.to_numeric(
        candidates["candidate_count"], errors="coerce"
    )
    scene_coverage = (
        candidates.groupby(["scene_id", "grounding_condition"], sort=True)[
            "candidate_count"
        ]
        .agg(
            groups="size",
            mean_candidates="mean",
            non_empty_rate=lambda x: (x > 0).mean(),
        )
        .reset_index()
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[7][0],
            "title": FIGURE_FAMILIES[7][1],
            "frame": scene_coverage,
            "sources": (inputs.run_dir / "candidate_pool_summary.csv",),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="scene_id",
                hue="grounding_condition",
                y="non_empty_rate",
                ylabel="Non-empty pool rate",
                title="Frozen VGN pool coverage by held-out scene",
                rotate=True,
            ),
        }
    )

    selected_groups = inputs.analyses[inputs.selected_condition].per_group
    first_rank = selected_groups.loc[
        selected_groups["system"].astype(str).isin(["B0_NATIVE", "R0_RAW"]),
        ["system", "first_valid_target_rank_mu_1.2", "candidate_absent"],
    ].copy()
    first_rank["first_valid_target_rank_mu_1.2"] = pd.to_numeric(
        first_rank["first_valid_target_rank_mu_1.2"], errors="coerce"
    )

    def rank_histogram(frame: pd.DataFrame) -> Any:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(4.6, 2.75))
        finite = frame["first_valid_target_rank_mu_1.2"].dropna()
        maximum = max(1, int(finite.max()))
        bins = np.arange(0.5, maximum + 1.5, 1.0)
        for index, (system, part) in enumerate(frame.groupby("system", sort=False)):
            axis.hist(
                part["first_valid_target_rank_mu_1.2"].dropna(),
                bins=bins,
                alpha=0.55,
                label=system,
                color=OKABE_ITO[index + 1],
                edgecolor="black",
                linewidth=0.35,
            )
        absent = int(_bool_series(frame["candidate_absent"], "candidate_absent").sum())
        axis.set_xlabel("First valid target rank (mu=1.2)")
        axis.set_ylabel("Group count")
        axis.set_title(f"Absent entries excluded from rank axis (rows={absent})")
        axis.legend(frameon=False)
        figure.tight_layout(pad=0.6)
        return figure

    specs.append(
        {
            "id": FIGURE_FAMILIES[8][0],
            "title": FIGURE_FAMILIES[8][1],
            "frame": first_rank,
            "sources": (
                inputs.analyses[inputs.selected_condition].root
                / "per_group_metrics.csv",
            ),
            "plotter": rank_histogram,
        }
    )

    target = pd.DataFrame(inputs.target_rows)
    target_test = target.loc[target["split"].astype(str).eq("test")]
    visibility = _performance_bins(
        inputs,
        value_by_group=target_test[["group_id", "valid_depth_fraction"]],
        value_column="valid_depth_fraction",
        bin_edges=(0.0, 0.8, 0.9, 0.95, 1.0000001),
        bin_labels=("<=0.80", "0.80-0.90", "0.90-0.95", ">0.95"),
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[9][0],
            "title": FIGURE_FAMILIES[9][1],
            "frame": visibility,
            "sources": (
                inputs.run_dir / "manifests/target_groups.jsonl",
                inputs.analyses[inputs.selected_condition].root
                / "per_group_metrics.csv",
            ),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="bin",
                hue="system",
                y="p_at_1_mu_1.2",
                ylabel="Target P@1 (mu=1.2)",
                title="By target valid-depth fraction",
            ),
        }
    )

    grounding = inputs.grounding.loc[
        inputs.grounding["condition"].astype(str).eq(inputs.selected_condition)
        & inputs.grounding["split"].astype(str).eq("test")
    ][["group_id", "iou"]].copy()
    iou_bins = _performance_bins(
        inputs,
        value_by_group=grounding,
        value_column="iou",
        bin_edges=(-1e-12, 0.25, 0.5, 0.75, 1.0000001),
        bin_labels=("0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"),
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[10][0],
            "title": FIGURE_FAMILIES[10][1],
            "frame": iou_bins,
            "sources": (
                inputs.run_dir / "grounding_metrics.csv",
                inputs.analyses[inputs.selected_condition].root
                / "per_group_metrics.csv",
            ),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="bin",
                hue="system",
                y="p_at_1_mu_1.2",
                ylabel="Target P@1 (mu=1.2)",
                title=f"By predicted-mask IoU: {inputs.selected_condition}",
            ),
        }
    )

    object_values = target_test[["group_id", "mask_area"]].copy()
    try:
        quantiles = np.unique(
            np.quantile(
                pd.to_numeric(object_values["mask_area"]).to_numpy(float),
                [0.0, 1 / 3, 2 / 3, 1.0],
            )
        )
    except (TypeError, ValueError) as error:
        raise PaperArtifactsRefused(f"invalid object-size values: {error}") from error
    object_size = pd.DataFrame()
    if len(quantiles) == 4:
        quantiles[-1] = np.nextafter(quantiles[-1], np.inf)
        object_size = _performance_bins(
            inputs,
            value_by_group=object_values,
            value_column="mask_area",
            bin_edges=quantiles,
            bin_labels=("small", "medium", "large"),
        )
        object_size["binning"] = "test-mask-area tertiles"
        object_size["bin_edges_px"] = ",".join(f"{value:.17g}" for value in quantiles)
    specs.append(
        {
            "id": FIGURE_FAMILIES[11][0],
            "title": FIGURE_FAMILIES[11][1],
            "frame": object_size if not object_size.empty else None,
            "sources": (
                inputs.run_dir / "manifests/target_groups.jsonl",
                inputs.analyses[inputs.selected_condition].root
                / "per_group_metrics.csv",
            ),
            "plotter": (
                None
                if object_size.empty
                else lambda frame: _grouped_bar(
                    frame,
                    x="bin",
                    hue="system",
                    y="p_at_1_mu_1.2",
                    ylabel="Target P@1 (mu=1.2)",
                    title="By GT target mask area tertile",
                )
            ),
            "reason_code": "UNEXECUTED_INPUT_MISSING",
            "reason": "target test mask areas do not define three distinct tertiles",
        }
    )

    language = pd.DataFrame(inputs.language_rows)
    query_values = target_test[["group_id"]].merge(
        language[["group_id", "template_family"]],
        on="group_id",
        validate="one_to_one",
    )
    snapshot = inputs.analyses[inputs.selected_condition]
    systems = snapshot.per_group.loc[
        snapshot.per_group["system"].astype(str).isin(["B0_NATIVE", "R0_RAW"]),
        ["group_id", "system", "top1_success_mu_1.2"],
    ]
    query_join = systems.merge(query_values, on="group_id", validate="many_to_one")
    query_join["success"] = _bool_series(
        query_join["top1_success_mu_1.2"], "query-type Top-1"
    ).astype(float)
    query_records: list[dict[str, Any]] = []
    for (family, system), part in query_join.groupby(
        ["template_family", "system"], sort=True
    ):
        query_records.append(
            {
                "query_type": str(family),
                "system": str(system),
                "groups": len(part),
                "successes": int(part["success"].sum()),
                "p_at_1_mu_1.2": float(part["success"].mean()),
            }
        )
    query_frame = pd.DataFrame(query_records)
    specs.append(
        {
            "id": FIGURE_FAMILIES[12][0],
            "title": FIGURE_FAMILIES[12][1],
            "frame": query_frame,
            "sources": (
                inputs.run_dir / "manifests/language_queries.jsonl",
                inputs.analyses[inputs.selected_condition].root
                / "per_group_metrics.csv",
            ),
            "plotter": lambda frame: _grouped_bar(
                frame,
                x="query_type",
                hue="system",
                y="p_at_1_mu_1.2",
                ylabel="Target P@1 (mu=1.2)",
                title="By deterministic derived-query family",
                rotate=True,
            ),
        }
    )

    failure = (
        tables["failure_summary"]
        .loc[tables["failure_summary"]["condition"].eq(inputs.selected_condition)]
        .copy()
    )
    specs.append(
        {
            "id": FIGURE_FAMILIES[13][0],
            "title": FIGURE_FAMILIES[13][1],
            "frame": failure[["category", "groups", "denominator", "fraction"]],
            "sources": (
                inputs.analyses[inputs.selected_condition].root
                / "failure_taxonomy.csv",
            ),
            "plotter": lambda frame: _simple_bar(
                frame,
                x="category",
                y="groups",
                ylabel="Exclusive test-group count",
                title=f"Failure taxonomy: {inputs.selected_condition}",
                colors=OKABE_ITO,
                rotate=True,
            ),
        }
    )

    runtime = inputs.runtime.copy()
    runtime["wall_time_s"] = pd.to_numeric(runtime["wall_time_s"], errors="coerce")
    runtime = runtime.loc[runtime["status"].astype(str).eq("COMPLETE")]
    specs.append(
        {
            "id": FIGURE_FAMILIES[14][0],
            "title": FIGURE_FAMILIES[14][1],
            "frame": runtime[
                ["stage", "wall_time_s", "process_rss_bytes_at_end", "resume_requested"]
            ],
            "sources": (),
            "source_hashes": {
                f"{inputs.run_dir / 'runtime.csv'}#required_pre_report_rows": (
                    _runtime_evidence_hash(runtime)
                )
            },
            "plotter": lambda frame: _simple_bar(
                frame,
                x="stage",
                y="wall_time_s",
                ylabel="Wall time (s)",
                title="Measured pre-report stage runtime on the recorded Mac",
                colors=OKABE_ITO,
                rotate=True,
            ),
        }
    )
    if [spec["id"] for spec in specs] != [item[0] for item in FIGURE_FAMILIES]:
        raise AssertionError("quantitative figure specification order changed")
    return specs


def _publish_figures(
    inputs: FormalPaperInputs, tables: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for spec in _build_figure_specs(inputs, tables):
        records.append(
            _record_figure(
                inputs,
                spec["id"],
                spec["title"],
                spec.get("frame"),
                source_paths=spec["sources"],
                source_hashes=spec.get("source_hashes"),
                plotter=spec.get("plotter"),
                reason_code=spec.get("reason_code"),
                reason=spec.get("reason"),
            )
        )
    status = (
        "COMPLETE" if all(row["status"] == "EXECUTED" for row in records) else "PARTIAL"
    )
    payload = {
        "schema_version": FIGURE_MANIFEST_SCHEMA,
        "status": status,
        "scope": FORMAL_SCOPE,
        "input_fingerprint": inputs.input_fingerprint,
        "required_family_count": len(FIGURE_FAMILIES),
        "executed_count": sum(row["status"] == "EXECUTED" for row in records),
        "unexecuted_count": sum(row["status"] != "EXECUTED" for row in records),
        "figures": records,
    }
    path = atomic_json(inputs.run_dir / "figures" / "figure_manifest.json", payload)
    payload["manifest_path"] = str(path)
    payload["manifest_sha256"] = sha256_file(path)
    return payload


def _candidate_rank_rows(snapshot: AnalysisSnapshot, group_id: str) -> pd.DataFrame:
    rows = snapshot.reranked.loc[
        snapshot.reranked["group_id"].astype(str).eq(group_id)
    ].copy()
    if rows.empty:
        return rows
    required = {
        "candidate_id",
        "native_rank",
        "native_score",
        "raw_rerank_score",
        "collision",
        "pose_valid",
        "friction_required",
        "target_match",
    }
    if not required.issubset(rows.columns):
        return pd.DataFrame()
    rows["reranked_rank"] = (
        rows.sort_values(
            ["raw_rerank_score", "native_rank", "candidate_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        .add(1)
        .reindex(rows.index)
        .astype(int)
    )
    return rows.sort_values("reranked_rank", kind="mergesort")


def _project_candidate(
    candidate: Mapping[str, Any], intrinsics: np.ndarray
) -> tuple[float, float] | None:
    translation = np.asarray(candidate.get("translation_camera_m"), dtype=float)
    if (
        translation.shape != (3,)
        or not np.isfinite(translation).all()
        or translation[2] <= 0
    ):
        return None
    u = float(intrinsics[0, 0] * translation[0] / translation[2] + intrinsics[0, 2])
    v = float(intrinsics[1, 1] * translation[1] / translation[2] + intrinsics[1, 2])
    return (u, v) if np.isfinite([u, v]).all() else None


def _rotation_difference_deg(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> float | None:
    try:
        a = np.asarray(first["rotation_camera"], dtype=float)
        b = np.asarray(second["rotation_camera"], dtype=float)
        cosine = float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))
    except (KeyError, TypeError, ValueError):
        return None


def _gallery_case_groups(
    snapshot: AnalysisSnapshot,
    candidate_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    failures = snapshot.failures.copy()
    categories = {
        str(row.group_id): str(row.category) for row in failures.itertuples(index=False)
    }
    cases: dict[str, str] = {}
    reasons: dict[str, str] = {}

    def first_category(case: str, category: str) -> None:
        values = sorted(
            group for group, value in categories.items() if value == category
        )
        if values:
            cases[case] = values[0]
        else:
            reasons[case] = f"no real test group was classified as {category}"

    first_category("native_success", "S0_UNCHANGED_SUCCESS")
    first_category("candidate_absence", "F3_CANDIDATE_GENERATION_FAILURE")
    first_category("native_ordering_failure", "F4_NATIVE_ORDERING_FAILURE")
    first_category("reranking_recovery", "F5_RERANKING_RECOVERY")
    first_category("reranking_harm", "F6_RERANKING_HARM")
    first_category("predicted_mask_grounding_failure", "F1_GROUNDING_FAILURE")
    first_category(
        "single_view_geometry_failure", "F7_UNRECOVERABLE_COLLISION_GEOMETRY_FAILURE"
    )

    recovery_groups = sorted(
        group for group, value in categories.items() if value == "F5_RERANKING_RECOVERY"
    )
    collision_group: str | None = None
    orientation_group: str | None = None
    for group_id in recovery_groups:
        rows = _candidate_rank_rows(snapshot, group_id)
        if rows.empty:
            continue
        native = rows.sort_values("native_rank").iloc[0]
        reranked = rows.sort_values("reranked_rank").iloc[0]
        native_collision = bool(
            _bool_series(pd.Series([native["collision"]]), "collision").iloc[0]
        )
        reranked_collision = bool(
            _bool_series(pd.Series([reranked["collision"]]), "collision").iloc[0]
        )
        if native_collision and not reranked_collision and collision_group is None:
            collision_group = group_id
        bundle = candidate_records.get(group_id, {})
        candidate_by_id = {
            str(row.get("candidate_id")): row
            for row in bundle.get("candidate_records", [])
            if isinstance(row, Mapping)
        }
        first = candidate_by_id.get(str(native["candidate_id"]))
        second = candidate_by_id.get(str(reranked["candidate_id"]))
        difference = (
            None
            if first is None or second is None
            else _rotation_difference_deg(first, second)
        )
        if difference is not None and difference >= 5.0 and orientation_group is None:
            orientation_group = group_id
    if collision_group is None:
        reasons["collision_risk_correction"] = (
            "no real recovery changed a colliding native Top-1 to a non-colliding Top-1"
        )
    else:
        cases["collision_risk_correction"] = collision_group
    if orientation_group is None:
        reasons["orientation_correction"] = (
            "no real recovery had a verifiable >=5 degree Top-1 orientation change"
        )
    else:
        cases["orientation_correction"] = orientation_group
    return cases, reasons


def _publish_failure_gallery(inputs: FormalPaperInputs) -> dict[str, Any]:
    condition = inputs.selected_condition
    snapshot = inputs.analyses[condition]
    target_by_id = {str(row["group_id"]): row for row in inputs.target_rows}
    language_by_id = {str(row["group_id"]): row for row in inputs.language_rows}
    candidate_index = inputs.candidate_summary.loc[
        inputs.candidate_summary["grounding_condition"].astype(str).eq(condition)
    ]
    bundles: dict[str, Mapping[str, Any]] = {}
    bundle_paths: dict[str, Path] = {}
    for row in candidate_index.itertuples(index=False):
        path, payload = _read_json(
            Path(str(row.bundle_path)), "gallery candidate bundle"
        )
        bundles[str(row.group_id)] = payload
        bundle_paths[str(row.group_id)] = path
    cases, unexecuted = _gallery_case_groups(snapshot, bundles)
    if not cases:
        refusal = {
            "schema_version": GALLERY_MANIFEST_SCHEMA,
            "status": "UNEXECUTED_NO_REAL_CASES",
            "scope": FORMAL_SCOPE,
            "condition": condition,
            "reason": "the saved formal failure taxonomy contains no renderable real case",
            "unexecuted_case_types": unexecuted,
        }
        path = atomic_json(inputs.run_dir / "failure_gallery_refusal.json", refusal)
        return {
            **refusal,
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
        }

    cards: list[str] = []
    records: list[dict[str, Any]] = []
    rendered_by_group: dict[str, Path] = {}
    for case_type, group_id in cases.items():
        target = target_by_id[group_id]
        query = language_by_id[group_id]
        bundle = bundles[group_id]
        bundle_path = bundle_paths[group_id]
        rgb_path = _regular_file(target["rgb_path"], "gallery RGB")
        label_path = _regular_file(
            target["instance_label_path"], "gallery instance labels"
        )
        intrinsics_path = _regular_file(
            target["intrinsics_path"], "gallery camera intrinsics"
        )
        sidecar_path = _regular_file(
            inputs.run_dir
            / "predicted_masks"
            / condition
            / f"{group_artifact_slug(group_id)}.json",
            "gallery predicted-mask sidecar",
        )
        _, sidecar = _read_json(sidecar_path, "gallery predicted-mask sidecar")
        mask_path = _bound_file(
            sidecar_path,
            sidecar.get("mask_file"),
            sidecar.get("mask_sha256"),
            "gallery predicted mask",
            run_dir=inputs.run_dir,
        )
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(label_path) as image:
            labels = np.asarray(image)
        with Image.open(mask_path) as image:
            predicted = np.asarray(image.convert("L")) > 0
        gt = labels == int(target["target_instance_label"])
        if (
            labels.shape != rgb.shape[:2]
            or predicted.shape != rgb.shape[:2]
            or not gt.any()
        ):
            raise PaperArtifactsRefused(
                f"gallery source shape/GT mask failed: {group_id}"
            )
        if group_id not in rendered_by_group:
            gt_panel = rgb.copy()
            gt_panel[gt] = np.rint(
                0.55 * gt_panel[gt].astype(float) + 0.45 * np.array([213, 94, 0])
            ).astype(np.uint8)
            pred_panel = rgb.copy()
            pred_panel[predicted] = np.rint(
                0.55 * pred_panel[predicted].astype(float)
                + 0.45 * np.array([0, 114, 178])
            ).astype(np.uint8)
            rows = _candidate_rank_rows(snapshot, group_id)
            candidate_by_id = {
                str(row.get("candidate_id")): row
                for row in bundle.get("candidate_records", [])
                if isinstance(row, Mapping)
            }
            if not rows.empty:
                native_id = str(rows.sort_values("native_rank").iloc[0]["candidate_id"])
                reranked_id = str(
                    rows.sort_values("reranked_rank").iloc[0]["candidate_id"]
                )
                intrinsics = np.asarray(
                    np.load(intrinsics_path, allow_pickle=False), dtype=float
                )
                image = Image.fromarray(pred_panel)
                draw = ImageDraw.Draw(image)
                for candidate_id, colour in (
                    (native_id, (230, 159, 0)),
                    (reranked_id, (0, 158, 115)),
                ):
                    candidate = candidate_by_id.get(candidate_id)
                    point = (
                        None
                        if candidate is None
                        else _project_candidate(candidate, intrinsics)
                    )
                    if point is not None:
                        u, v = point
                        if 0 <= u < image.width and 0 <= v < image.height:
                            draw.ellipse(
                                (u - 8, v - 8, u + 8, v + 8),
                                outline=colour,
                                width=4,
                            )
                pred_panel = np.asarray(image)
            composite = np.concatenate([gt_panel, pred_panel], axis=1)
            output = (
                inputs.run_dir
                / "figures"
                / "qualitative"
                / f"{group_artifact_slug(group_id)}.png"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            try:
                Image.fromarray(composite).save(temporary, format="PNG")
                os.replace(temporary, output)
            finally:
                if temporary.exists():
                    temporary.unlink()
            rendered_by_group[group_id] = output
        output = rendered_by_group[group_id]
        candidate_rows = _candidate_rank_rows(snapshot, group_id)
        table_columns = [
            "candidate_id",
            "native_rank",
            "reranked_rank",
            "native_score",
            "raw_rerank_score",
            "collision",
            "pose_valid",
            "friction_required",
            "target_match",
        ]
        candidate_rows = candidate_rows[table_columns].head(10)
        candidate_html = _markdown_table(candidate_rows).replace("\n", "<br>")
        # Render a real HTML table separately; the markdown rendering above is
        # retained only in the machine record for easy plain-text inspection.
        html_rows = "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
            + "</tr>"
            for row in candidate_rows.itertuples(index=False, name=None)
        )
        header = "".join(f"<th>{html.escape(column)}</th>" for column in table_columns)
        relative = output.relative_to(inputs.run_dir).as_posix()
        cards.append(
            "<article class='card'>"
            f"<h2>{html.escape(case_type.replace('_', ' ').title())}</h2>"
            f"<p><code>{html.escape(group_id)}</code><br>{html.escape(str(query['query']))}</p>"
            f"<img src='{html.escape(relative)}' alt='GT and predicted mask with Top-1 centres'>"
            "<p>Left: GT target overlay (oracle counterfactual reference). Right: selected "
            "predicted mask; yellow circle is native Top-1 and green circle is raw-reranked Top-1.</p>"
            f"<table><thead><tr>{header}</tr></thead><tbody>{html_rows}</tbody></table>"
            "</article>"
        )
        category = snapshot.failures.loc[
            snapshot.failures["group_id"].astype(str).eq(group_id), "category"
        ].iloc[0]
        records.append(
            {
                "case_type": case_type,
                "group_id": group_id,
                "failure_category": str(category),
                "query": str(query["query"]),
                "target_object_id": int(target["target_object_id"]),
                "image_path": relative,
                "image_sha256": sha256_file(output),
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256_file(rgb_path),
                "instance_label_path": str(label_path),
                "instance_label_sha256": sha256_file(label_path),
                "predicted_mask_path": str(mask_path),
                "predicted_mask_sha256": sha256_file(mask_path),
                "predicted_mask_sidecar_path": str(sidecar_path),
                "predicted_mask_sidecar_sha256": sha256_file(sidecar_path),
                "intrinsics_path": str(intrinsics_path),
                "intrinsics_sha256": sha256_file(intrinsics_path),
                "candidate_bundle_path": str(bundle_path),
                "candidate_bundle_sha256": sha256_file(bundle_path),
                "candidate_table_markdown": candidate_html,
            }
        )
    gallery = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Formal GraspNet 6-DoF failure gallery</title>"
        "<style>body{font:14px system-ui;margin:24px;background:#f4f5f7;color:#17202a}"
        ".card{background:#fff;border:1px solid #ccd2d9;border-radius:8px;padding:14px;"
        "margin:0 0 20px}img{width:100%;height:auto}table{border-collapse:collapse;"
        "font-size:11px;display:block;overflow:auto}th,td{border:1px solid #ccd2d9;"
        "padding:4px}code{font-size:12px}</style></head><body>"
        "<h1>Formal GraspNet 6-DoF qualitative failure gallery</h1>"
        f"<p>Run <code>{html.escape(inputs.run_dir.name)}</code>; selected predicted "
        f"condition <code>{html.escape(condition)}</code>. Outcomes are offline official-"
        "evaluator labels and are not physical robot execution success.</p>"
        + "".join(cards)
        + "</body></html>"
    )
    gallery_path = atomic_text(inputs.run_dir / "failure_gallery.html", gallery)
    payload = {
        "schema_version": GALLERY_MANIFEST_SCHEMA,
        "status": "COMPLETE",
        "scope": FORMAL_SCOPE,
        "condition": condition,
        "input_fingerprint": inputs.input_fingerprint,
        "rendered_case_count": len(records),
        "records": records,
        "unexecuted_case_types": unexecuted,
        "gallery_path": str(gallery_path.relative_to(inputs.run_dir)),
        "gallery_sha256": sha256_file(gallery_path),
    }
    manifest_path = atomic_json(
        inputs.run_dir / "failure_gallery_manifest.json", payload
    )
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload


def _conclusion_record(inputs: FormalPaperInputs, condition: str) -> dict[str, Any]:
    snapshot = inputs.analyses[condition]
    paired_table = _paired_result_table(inputs)
    row = paired_table.loc[
        paired_table["condition"].eq(condition)
        & paired_table["challenger"].eq("R0_RAW")
    ].iloc[0]
    paired = snapshot.paired.loc[
        snapshot.paired["challenger"].astype(str).eq("R0_RAW")
    ].copy()
    paired["net_recovered"] = pd.to_numeric(paired["net_recovered"], errors="coerce")
    scene_nets = paired.groupby("scene_id", sort=True)["net_recovered"].sum()
    total_net = float(scene_nets.sum())
    without_largest = (
        total_net - float(scene_nets.max()) if len(scene_nets) > 1 else float("-inf")
    )
    metrics = snapshot.metrics
    native = metrics.loc[metrics["system"].eq("B0_NATIVE")].iloc[0]
    seeds = metrics.loc[metrics["system"].eq("R0_RAW_SEED")].copy()
    seed_deltas = pd.to_numeric(seeds["target_p_at_1_mu_1.2"], errors="coerce") - float(
        native["target_p_at_1_mu_1.2"]
    )
    seed_stable = bool(
        (seed_deltas >= -1e-12).all() and (seed_deltas > 1e-12).sum() >= 2
    )
    failures = snapshot.failures["category"].astype(str).value_counts()
    coverage_failures = int(
        failures.get("F3_CANDIDATE_GENERATION_FAILURE", 0)
        + failures.get("F7_UNRECOVERABLE_COLLISION_GEOMETRY_FAILURE", 0)
    )
    ordering_events = int(
        failures.get("F4_NATIVE_ORDERING_FAILURE", 0)
        + failures.get("F5_RERANKING_RECOVERY", 0)
        + failures.get("F6_RERANKING_HARM", 0)
    )
    oracle_50 = float(native["oracle_at_50"])
    coverage_dominant = bool(oracle_50 < 0.5 or coverage_failures > ordering_events)
    reliable = bool(
        float(row["delta_p_at_1"]) > 0
        and float(row["ci_low"]) >= 0
        and float(row["mcnemar_pvalue"]) <= 0.05
        and int(row["recovered"]) > int(row["harmful"])
        and len(scene_nets) >= 2
        and without_largest > 0
        and seed_stable
    )
    if coverage_dominant:
        classification = "D_CANDIDATE_COVERAGE_LIMITED"
        statement = (
            "The dominant limitation was candidate coverage rather than candidate "
            "ordering; no re-ranker can recover a valid grasp that is absent from "
            "the frozen pool."
        )
    elif reliable:
        classification = "A_RELIABLE_IMPROVEMENT"
        statement = (
            "The existing re-ranking method improved target-specific 6-DoF "
            "candidate ordering within a frozen VGN candidate pool."
        )
    elif float(row["delta_p_at_1"]) > 0:
        classification = "B_POSITIVE_BUT_UNSTABLE"
        statement = (
            "The experiment produced a positive point estimate, but did not provide "
            "sufficiently stable evidence of a general re-ranking improvement."
        )
    else:
        classification = "C_NO_RELIABLE_IMPROVEMENT"
        statement = (
            "No evidence was found that the existing re-ranker transfers reliably "
            "from frozen 4-DoF candidates to frozen VGN 6-DoF candidates under the "
            "evaluated feature contract."
        )
    return {
        "condition": condition,
        "condition_role": (
            "oracle_grounding_counterfactual"
            if condition == "oracle_gt_mask"
            else "complete_predicted_grounding_pipeline"
        ),
        "classification": classification,
        "statement": statement,
        "test_groups": int(row["test_groups"]),
        "delta_p_at_1_mu_1.2": float(row["delta_p_at_1"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "recovered": int(row["recovered"]),
        "harmful": int(row["harmful"]),
        "net": int(row["net"]),
        "mcnemar_pvalue": float(row["mcnemar_pvalue"]),
        "scene_count": len(scene_nets),
        "leave_largest_positive_scene_out_net": without_largest,
        "seed_p_at_1_deltas": [float(value) for value in seed_deltas],
        "seed_stable_by_locked_rule": seed_stable,
        "oracle_at_50": oracle_50,
        "coverage_failure_groups": coverage_failures,
        "ordering_event_groups": ordering_events,
        "coverage_dominant_by_locked_report_rule": coverage_dominant,
        "offline_evaluator_only": True,
        "physical_robot_success_claimed": False,
    }


def _conclusion_evidence(inputs: FormalPaperInputs) -> dict[str, Any]:
    oracle = _conclusion_record(inputs, "oracle_gt_mask")
    predicted = _conclusion_record(inputs, inputs.selected_condition)
    if (
        oracle["classification"] == "A_RELIABLE_IMPROVEMENT"
        and predicted["classification"] != "A_RELIABLE_IMPROVEMENT"
    ):
        cross_condition = (
            "Re-ranking improved ordering when target localisation was correct, but "
            "grounding errors reduced or eliminated the gain in the complete modular pipeline."
        )
        cross_classification = "E_PREDICTED_MASK_OFFSETS_ORACLE_GAIN"
    else:
        cross_condition = (
            "The oracle-grounding and validation-selected predicted-grounding arms "
            "must be interpreted separately; their recorded classifications are reported above."
        )
        cross_classification = "SEPARATE_ARM_INTERPRETATION"
    return {
        "schema_version": "graspnet6d_mechanical_conclusions_v1",
        "status": "COMPLETE",
        "scope": FORMAL_SCOPE,
        "run_id": inputs.run_dir.name,
        "selected_predicted_condition": inputs.selected_condition,
        "primary_intervention": "R0_RAW versus B0_NATIVE",
        "success_threshold_mu": 1.2,
        "decision_rules": {
            "reliable": (
                "delta>0; ci_low>=0; exact McNemar p<=0.05; recovered>harmful; "
                "at least two scenes; "
                "net remains positive after removing the largest positive scene; "
                "all locked seed deltas non-negative and at least two positive"
            ),
            "coverage_dominant": (
                "Oracle@50<0.5 or F3+F7 groups exceed F4+F5+F6 groups"
            ),
        },
        "oracle_counterfactual": oracle,
        "selected_predicted_pipeline": predicted,
        "cross_condition_classification": cross_classification,
        "cross_condition_statement": cross_condition,
        "metric_interpretation": (
            "offline target-specific GraspNet association/collision/friction outcomes; "
            "not physical robot execution success"
        ),
    }


def _format_fraction(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise PaperArtifactsRefused("a reported percentage has no denominator")
    return f"{numerator} / {denominator} = {100.0 * numerator / denominator:.1f}%"


def _dataset_summary(inputs: FormalPaperInputs) -> dict[str, Any]:
    target = pd.DataFrame(inputs.target_rows)
    counts = target["split"].astype(str).value_counts()
    candidates = inputs.candidate_summary.copy()
    candidates["candidate_count"] = pd.to_numeric(
        candidates["candidate_count"], errors="coerce"
    )
    return {
        "schema_version": "graspnet6d_dataset_summary_v1",
        "status": "COMPLETE",
        "scope": FORMAL_SCOPE,
        "run_id": inputs.run_dir.name,
        "profile": inputs.run_manifest["profile"],
        "downloaded_archives": sorted(inputs.run_manifest["dataset_archive_hashes"]),
        "downloaded_archive_sha256": dict(
            inputs.run_manifest["dataset_archive_hashes"]
        ),
        "scenes": int(target["scene_id"].astype(str).nunique()),
        "frames": int(
            target[["scene_id", "camera", "frame_id"]].drop_duplicates().shape[0]
        ),
        "target_groups": len(target),
        "train_groups": int(counts.get("train", 0)),
        "validation_groups": int(counts.get("validation", 0)),
        "test_groups": int(counts.get("test", 0)),
        "excluded_groups": int(inputs.run_manifest["sample_counts"]["excluded_groups"]),
        "conditions": list(CONDITIONS),
        "selected_predicted_condition": inputs.selected_condition,
        "candidate_groups": len(candidates),
        "non_empty_candidate_groups": int((candidates["candidate_count"] > 0).sum()),
        "empty_candidate_groups": int((candidates["candidate_count"] == 0).sum()),
        "target_manifest_sha256": sha256_file(
            inputs.run_dir / "manifests/target_groups.jsonl"
        ),
        "language_manifest_sha256": sha256_file(
            inputs.run_dir / "manifests/language_queries.jsonl"
        ),
        "split_manifest_sha256": sha256_file(inputs.run_dir / "split_manifest.json"),
    }


def _methods_text(inputs: FormalPaperInputs, tables: Mapping[str, pd.DataFrame]) -> str:
    del tables
    config = inputs.config
    target = pd.DataFrame(inputs.target_rows)
    split_counts = target["split"].astype(str).value_counts()
    selected_analysis = inputs.analyses[inputs.selected_condition]
    analysis_config = json.loads(
        (selected_analysis.root / "resolved_analysis_config.json").read_text(
            encoding="utf-8"
        )
    )["config"]
    geometry_contract = json.loads(
        (
            inputs.run_dir / "geometry_validation/evaluator_geometry_contract.json"
        ).read_text(encoding="utf-8")
    )
    hardware = inputs.run_manifest.get("hardware_environment") or {}
    commands = inputs.run_manifest.get("commands_executed", [])
    command_lines = [" ".join(map(str, command)) for command in commands]
    return f"""# Methods

## Evidence boundary

Run ID: `{inputs.run_dir.name}`. Profile: `{inputs.run_manifest["profile"]}`.
This document was generated from saved, hash-verified formal artifacts only. The
GT-mask arm is an oracle-grounding counterfactual. The validation-selected
`{inputs.selected_condition}` arm is the primary complete predicted-grounding
pipeline. Offline association, collision, and friction outcomes are not physical
robot grasp success.

## Data and split

The run consumed the official archives listed in `dataset_summary.json` under
their recorded terms. It retained {len(target)} target/query groups from
{target["scene_id"].astype(str).nunique()} scenes: train
{int(split_counts.get("train", 0))}, validation
{int(split_counts.get("validation", 0))}, and test
{int(split_counts.get("test", 0))}. Splits are scene-disjoint. Frames were sampled
deterministically ({config["dataset"]["frames_per_scene"]} views per scene), with
at most {config["dataset"]["max_targets_per_frame"]} targets per frame. Targets
required at least {config["target_selection"]["min_mask_pixels"]} mask pixels and
{float(config["target_selection"]["min_valid_depth_fraction"]):.2f} valid-depth
fraction. Language queries are deterministic project-derived annotations and
must resolve uniquely to one visible target; they are not official GraspNet
language labels.

## Grounding and geometry

Three pre-registered conditions were evaluated: `oracle_gt_mask`,
`hifics_zero_shot_mask`, and `hifics_adapted_mask`. Adaptation is decoder-only,
uses training scenes, selects checkpoints on validation, and never uses test
outcomes. The primary predicted arm was selected by validation mIoU, with the
zero-shot arm winning exact ties.

The target-centred TSDF used physical size
{float(config["tsdf"]["physical_size_m"]):.3f} m, resolution
{int(config["tsdf"]["resolution"])}^3, voxel size
{float(config["tsdf"]["voxel_size_m"]):.4f} m, and truncation distance
{float(config["tsdf"]["truncation_m"]):.3f} m. The target mask translated the
workspace while the full local scene depth was retained. The accepted gripper
mapping, rather than the unchecked configuration proposal, is bound by the
formal geometry contract: height {float(geometry_contract["height_m"]):.4f} m,
depth {float(geometry_contract["depth_m"]):.4f} m, with at least twenty real-data
coordinate audit figures.

## Frozen VGN candidates and evaluator

The pretrained VGN checkpoint was frozen (`{inputs.run_manifest["vgn_checkpoint_hash"]}`).
Formal inference used `{config["vgn"]["device"]}`. Candidate extraction used a
quality threshold of {float(config["vgn"]["quality_threshold"]):.3f}, at most
{int(config["vgn"]["pre_nms_max_candidates"])} pre-NMS proposals, pose NMS
(translation {float(config["nms"]["translation_threshold_m"]):.3f} m, rotation
{float(config["nms"]["rotation_threshold_deg"]):.1f} degrees, width
{float(config["nms"]["width_threshold_m"]):.3f} m), and frozen Top-
{int(config["vgn"]["frozen_top_k"])}. Candidate identity, translation, rotation,
width, geometry hashes, and membership were required to be identical for native
and reranked predictions. The evaluator adapter used low-level official
association/collision/friction math per frozen candidate and explicitly avoided
the score-dependent `eval_grasp` NMS/Top-K route. The real-data parity gate passed
before labels were admitted.

## Features, ranker, gate, and statistics

Only runtime RGB-D, selected-mask, scene-point-cloud, frozen-candidate, camera,
and table geometry sources entered the feature contract; evaluator labels and
test ground truth were barred from features. The actual re-ranker was a CPU
LightGBM LambdaMART model with graded relevance 0--6, train-only median
imputation, validation-only hyperparameter selection, and locked seeds
{list(map(int, analysis_config["seeds"]))}. Both the raw reranker and the
expected-gain gate were evaluated; the gate used train out-of-fold transitions
and a validation-only operating point and failed closed to native ordering when
new 6-DoF gate evidence was not fit-able.

Target P@1 was reported at friction thresholds 0.4, 0.8, and 1.2. The reported
target-specific AP averages fixed-denominator precision over ranks and friction
thresholds 0.2--1.2; it is not official unconditional GraspNet leaderboard AP.
MRR, NDCG, Oracle@K, recovered/harmful paired outcomes, exact two-sided McNemar,
and {int(analysis_config["bootstrap_iterations"])} scene-cluster bootstrap
replicates were computed from saved raw evaluator rows.

## Hardware, runtime, and reproducibility

Recorded hardware: `{json.dumps(hardware, sort_keys=True, ensure_ascii=False)}`.
Per-stage wall times and terminal RSS values are in `runtime.csv`; resumed work
is identified explicitly. Quantitative Figure 15 binds the eleven required
pre-report rows because the report row can only be measured after that figure is
committed. Commands recorded by the immutable run manifest:

{chr(10).join(f"- `{line}`" for line in command_lines)}
"""


def _results_text(
    inputs: FormalPaperInputs,
    tables: Mapping[str, pd.DataFrame],
    conclusion: Mapping[str, Any],
    figures: Mapping[str, Any],
    gallery: Mapping[str, Any],
) -> str:
    main = tables["main_results"]
    paired = tables["paired_results"]
    selected = inputs.selected_condition
    result_rows: list[str] = []
    for condition in ("oracle_gt_mask", selected):
        for system in ("B0_NATIVE", "R0_RAW", "R1_GATED", "O_ORACLE"):
            row = main.loc[
                main["condition"].eq(condition) & main["system"].eq(system)
            ].iloc[0]
            result_rows.append(
                "| "
                + " | ".join(
                    [
                        condition,
                        system,
                        str(int(row["test_groups"])),
                        _format_fraction(
                            int(row["p_at_1_mu_0.4_numerator"]),
                            int(row["p_at_1_mu_0.4_denominator"]),
                        ),
                        _format_fraction(
                            int(row["p_at_1_mu_0.8_numerator"]),
                            int(row["p_at_1_mu_0.8_denominator"]),
                        ),
                        _format_fraction(
                            int(row["p_at_1_mu_1.2_numerator"]),
                            int(row["p_at_1_mu_1.2_denominator"]),
                        ),
                        f"{float(row['target_specific_ap_mean_mu_0.2_to_1.2']):.6f}",
                        f"{float(row['mrr_mu_1.2']):.6f}",
                    ]
                )
                + " |"
            )
    paired_rows: list[str] = []
    for condition in ("oracle_gt_mask", selected):
        row = paired.loc[
            paired["condition"].eq(condition) & paired["challenger"].eq("R0_RAW")
        ].iloc[0]
        paired_rows.append(
            f"| {condition} | {int(row['test_groups'])} | {int(row['recovered'])} | "
            f"{int(row['harmful'])} | {int(row['net'])} | {float(row['delta_p_at_1']):.6f} | "
            f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}] | "
            f"{float(row['mcnemar_pvalue']):.6g} |"
        )
    coverage = tables["candidate_coverage"].loc[
        tables["candidate_coverage"]["split"].eq("test")
    ]
    coverage_lines = [
        f"- `{row.condition}`: {int(row.non_empty_groups)} / {int(row.groups)} non-empty; "
        f"mean {float(row.mean_candidates):.3f} candidates."
        for row in coverage.itertuples(index=False)
    ]
    grounding = tables["grounding_results"].loc[
        tables["grounding_results"]["split"].eq("test")
    ]
    grounding_lines = [
        f"- `{row.condition}`: mean IoU {float(row.mean_iou):.6f}; "
        f"{_format_fraction(int(row.empty_predictions), int(row.groups))} empty masks."
        for row in grounding.itertuples(index=False)
    ]
    ablation = tables["ablation_summary"]
    a9 = ablation.loc[ablation["ablation"].astype(str).eq("A9")]
    a9_status = ", ".join(
        f"{row.setting}={row.status}"
        for row in a9.drop_duplicates(["setting", "status"]).itertuples(index=False)
    )
    runtime_seconds = float(inputs.runtime["wall_time_s"].astype(float).sum())
    failure = tables["failure_summary"].loc[
        tables["failure_summary"]["condition"].eq(selected)
    ]
    scope_note = (
        "This is held-out evaluation on seven scenes from a locked 30-scene "
        "GraspNet training subset. Bootstrap intervals resample scenes as "
        "clusters; frames from one scene are not independent units. It is not "
        "an official GraspNet test benchmark or novel-object evaluation."
        if inputs.run_manifest["profile"] == "paper-lite-train3"
        else ""
    )
    return f"""# Results

## Evidence scope and data availability

Run `{inputs.run_dir.name}` used profile `{inputs.run_manifest["profile"]}` and
contains {len(inputs.target_rows)} target groups. All numbers below were read
from saved formal CSV/JSON artifacts and are traceable through
`paper_artifact_manifest.json`. GT-mask results are an oracle-grounding
counterfactual; `{selected}` is the validation-selected complete pipeline arm.
No value is a physical robot execution success rate.

{scope_note}

## Grounding

{chr(10).join(grounding_lines)}

## Candidate coverage

{chr(10).join(coverage_lines)}

## Native, raw reranker, gated reranker, and oracle

| Condition | System | N | P@1 mu=0.4 | P@1 mu=0.8 | P@1 mu=1.2 | target-specific mean AP | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_rows)}

## Paired intervention evidence (raw reranker versus native)

| Condition | N | Recovered | Harmful | Net | Delta P@1 | scene-bootstrap 95% CI | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_rows)}

The complete seed-level metrics, bootstrap intervals for P@1/AP/MRR/net,
expected-gain gate results, and exact tests are in `analysis/*` and `tables/`.
The raw and gated systems retain the exact frozen candidate IDs and geometry
verified by each `frozen_pool_audit.json`.

## Ablations and grounding comparison

A0--A10 results are in `tables/ablation_summary.csv`. The GT/predicted A8 plot
comes from `analysis/combined_a8`, preserving the pre-registered condition
universe. Multi-view resource status: {a9_status or "no A9 status row"}.

## Runtime and failures

The sum of the eleven required pre-report stage wall times is
{runtime_seconds:.3f} s; this is a sum of process-stage measurements rather than
end-to-end wall-clock latency. The
selected predicted arm has {len(failure)} observed exclusive taxonomy
categories across {int(failure["denominator"].max())} test groups. Exact counts
are in `tables/failure_summary.csv`. The qualitative gallery status is
`{gallery["status"]}` and contains {int(gallery.get("rendered_case_count", 0))}
actual rendered case types.

## Figures

The quantitative figure manifest reports {int(figures["executed_count"])} /
{int(figures["required_family_count"])} executed figure families. Each executed
figure has PDF and 300-dpi PNG outputs plus a plotted source-data CSV and hashes.

## Mechanical interpretation

Oracle counterfactual: {conclusion["oracle_counterfactual"]["statement"]}

Selected predicted pipeline: {conclusion["selected_predicted_pipeline"]["statement"]}

Cross-condition: {conclusion["cross_condition_statement"]}
"""


def _conclusions_text(conclusion: Mapping[str, Any]) -> str:
    oracle = conclusion["oracle_counterfactual"]
    predicted = conclusion["selected_predicted_pipeline"]

    def evidence(record: Mapping[str, Any]) -> str:
        return (
            f"N={record['test_groups']}; delta P@1={record['delta_p_at_1_mu_1.2']:.6f}; "
            f"95% CI [{record['ci_low']:.6f}, {record['ci_high']:.6f}]; "
            f"recovered={record['recovered']}; harmful={record['harmful']}; "
            f"exact McNemar p={record['mcnemar_pvalue']:.6g}; "
            f"Oracle@50={record['oracle_at_50']:.6f}."
        )

    return f"""# Conclusions

## RQ6D-1: frozen candidate ordering

{oracle["statement"]} {evidence(oracle)} This statement applies only to the
GT-mask oracle-grounding counterfactual, the recorded data profile, the frozen
VGN pool, and the offline target-specific evaluator.

## RQ6D-2: predicted grounding

{predicted["statement"]} {evidence(predicted)} The primary complete modular
pipeline used the validation-selected `{predicted["condition"]}` condition; no
test grounding or grasp outcome selected that condition.

## Cross-condition finding

{conclusion["cross_condition_statement"]}

These conclusions concern offline candidate selection under association,
collision, and friction criteria. They do not establish reachability, motion
planning feasibility, grasp execution, lifting, or physical robot success.
"""


def _limitations_text(
    inputs: FormalPaperInputs,
    figures: Mapping[str, Any],
    gallery: Mapping[str, Any],
) -> str:
    unexecuted_figures = [
        row for row in figures["figures"] if row["status"] != "EXECUTED"
    ]
    a9 = inputs.analyses[inputs.selected_condition].ablations
    multi = a9.loc[
        a9["ablation"].astype(str).eq("A9") & a9["setting"].astype(str).eq("five_view")
    ]
    multi_status = (
        str(multi.iloc[0]["status"]) if not multi.empty else "UNEXECUTED_INPUT_MISSING"
    )
    figure_note = (
        "none"
        if not unexecuted_figures
        else "; ".join(
            f"{row['id']}: {row['status']} ({row['reason']})"
            for row in unexecuted_figures
        )
    )
    gallery_note = (
        "; ".join(
            f"{key}: {value}"
            for key, value in dict(gallery.get("unexecuted_case_types", {})).items()
        )
        or "none"
    )
    return f"""# Limitations and matched future work

- **Derived language.** GraspNet did not supply the referring expressions used
  by this run; the language layer is deterministic and project-derived. Future
  work should evaluate human-authored and ambiguity-controlled instructions.
- **Oracle interpretation.** GT masks isolate candidate generation and ordering
  but are unavailable to an autonomous system. Future work should reserve them
  for diagnosis and improve the predicted grounding arm independently.
- **Detector/domain transfer.** The frozen pretrained VGN checkpoint was not
  fine-tuned in this experiment, so transfer to the recorded GraspNet views may
  limit coverage. Future work should test a separately licensed, validation-
  controlled detector adaptation while retaining a frozen comparison.
- **Single-view geometry.** The main TSDF uses one RGB-D view and therefore
  inherits occlusion and missing-surface ambiguity. The five-view A9 status was
  `{multi_status}`. Future work should execute the pre-registered multi-view arm
  when the required storage and compute are available.
- **Workspace bias.** Target-centred cropping can exclude context or reachable
  approach corridors. Future work should compare multi-scale workspaces without
  changing test-selected hyperparameters.
- **Frozen-pool ceiling.** Oracle@K bounds every ordering method; absent valid
  grasps cannot be recovered by a re-ranker. Future work should improve proposal
  coverage separately from ordering and keep pool membership audits.
- **Metric scope.** Target-specific AP here is not official unconditional
  GraspNet leaderboard AP. Future work should report both only when their task
  definitions and denominators are kept separate.
- **Offline evaluation.** Association, force-closure friction, and collision
  checks do not equal physical grasp success. Future work requires controlled
  robot trials with reachability, motion planning, swept-volume collision, and
  post-grasp lifting outcomes.
- **Mac/profile scale.** The `{inputs.run_manifest["profile"]}` profile contains
  {len(inputs.target_rows)} groups and ran on the recorded Mac environment.
  Future work should repeat the immutable protocol on the extended scene set and
  report compute/storage scaling.
- **Adaptation scope.** Decoder-only HiFi adaptation is closed-domain to the
  training scenes and validation selector. Future work should test domain-shifted
  language and scenes without changing the held-out test contract.
- **Figure availability.** Unexecuted quantitative panels: {figure_note}.
- **Qualitative availability.** Unavailable requested real-case types:
  {gallery_note}. Missing case types were recorded, not manufactured.
"""


def _reproduce_text(inputs: FormalPaperInputs) -> str:
    return f"""# Reproduce

The immutable run ID is `{inputs.run_dir.name}` and the profile is
`{inputs.run_manifest["profile"]}`. The official archives must be placed under
the paths in `resolved_config.yaml` and must match `run_manifest.json`.

From the repository root, resume the exact run:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -m graspnet6d.cli all --profile {inputs.run_manifest["profile"]} --resume --run-id {inputs.run_dir.name}
```

Re-render paper artifacts from saved CSV/JSON evidence only:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -c "from pathlib import Path; from graspnet6d.paper_artifacts import generate_formal_paper_artifacts; generate_formal_paper_artifacts(Path('artifacts/graspnet6d/{inputs.run_dir.name}'), resume=True)"
```

Validate tests:

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -m pytest -q tests/graspnet6d
```

All plotted rows are preserved under `figures/source_data/`; hashes and source
bindings are in `figures/figure_manifest.json` and
`paper_artifact_manifest.json`.
"""


def _final_status_text(
    inputs: FormalPaperInputs,
    figures: Mapping[str, Any],
    gallery: Mapping[str, Any],
) -> str:
    completion_status = (
        "COMPLETE_TRAIN3_SUBSET"
        if inputs.run_manifest.get("profile") == "paper-lite-train3"
        else "COMPLETE"
    )
    scope_lines = (
        "Source scenes: `30`  \nTrain/validation/test scenes: `18 / 5 / 7`  \n"
        "Camera: `kinect`  \nOfficial GraspNet test benchmark: `false`\n"
        if inputs.run_manifest["profile"] == "paper-lite-train3"
        else ""
    )
    return f"""# FINAL STATUS: {completion_status}

Run ID: `{inputs.run_dir.name}`  
Profile: `{inputs.run_manifest["profile"]}`  
Validation-selected predicted condition: `{inputs.selected_condition}`
{scope_lines}

All mandatory formal gates passed before publication:

1. Official-data formal profile executed.
2. Scene-disjoint train/validation/test manifests passed.
3. Real-data coordinate/geometry evidence passed with hashed audit figures.
4. Per-candidate official-evaluator parity passed.
5. Frozen candidate membership and geometry audits passed for all analyses.
6. Native, raw reranker, gated reranker, and oracle metrics are present.
7. Major ablations, paired outcomes, 10,000 scene-bootstrap replicates, exact
   McNemar tests, and exclusive failure taxonomy are present.
8. Real-data smoke checks passed 12 / 12.
9. The full legacy 4-DoF regression suite passed.
10. Quantitative figures executed {int(figures["executed_count"])} /
    {int(figures["required_family_count"])}; qualitative gallery status is
    `{gallery["status"]}`.
11. Every reported value is bound to saved CSV/JSON evidence; no missing result
    was replaced by a numerical value.

The GT-mask result is an oracle-grounding counterfactual. Only the selected
predicted-mask arm represents the complete modular pipeline. All grasp outcomes
are offline evaluator outcomes, not physical robot success. For the train3-only
profile this is held-out evaluation on seven scenes from a locked 30-scene
GraspNet training subset; it is not an official GraspNet test benchmark, full
GraspNet evaluation, or evidence of novel-object generalisation.
"""


def _publish_documents(
    inputs: FormalPaperInputs,
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, Any],
    gallery: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    if figures.get("status") != "COMPLETE":
        missing = [
            f"{row['id']}={row['status']}: {row['reason']}"
            for row in figures["figures"]
            if row["status"] != "EXECUTED"
        ]
        raise PaperArtifactsRefused(
            "top-level formal prose refused because required quantitative figure "
            f"families are incomplete: {missing}"
        )
    if gallery.get("status") not in {"COMPLETE", "UNEXECUTED_NO_REAL_CASES"}:
        raise PaperArtifactsRefused(
            "qualitative gallery has an invalid terminal status"
        )
    conclusion = _conclusion_evidence(inputs)
    conclusion_path = atomic_json(
        inputs.run_dir / "conclusion_evidence.json", conclusion
    )
    dataset_path = atomic_json(
        inputs.run_dir / "dataset_summary.json", _dataset_summary(inputs)
    )
    documents = {
        "METHODS.md": _methods_text(inputs, tables),
        "RESULTS.md": _results_text(inputs, tables, conclusion, figures, gallery),
        "CONCLUSIONS.md": _conclusions_text(conclusion),
        "LIMITATIONS.md": _limitations_text(inputs, figures, gallery),
        "REPRODUCE.md": _reproduce_text(inputs),
        "FINAL_STATUS.md": _final_status_text(inputs, figures, gallery),
    }
    outputs = {
        "conclusion_evidence.json": sha256_file(conclusion_path),
        "dataset_summary.json": sha256_file(dataset_path),
    }
    for filename, text in documents.items():
        path = atomic_text(inputs.run_dir / filename, text)
        outputs[filename] = sha256_file(path)
    return conclusion, outputs


def _collect_manifest_outputs(
    inputs: FormalPaperInputs,
    tables: Mapping[str, Any],
    figures: Mapping[str, Any],
    gallery: Mapping[str, Any],
    document_outputs: Mapping[str, str],
) -> dict[str, str]:
    outputs = dict(document_outputs)
    table_manifest = inputs.run_dir / "tables/table_manifest.json"
    figure_manifest = inputs.run_dir / "figures/figure_manifest.json"
    outputs[str(table_manifest.relative_to(inputs.run_dir))] = sha256_file(
        table_manifest
    )
    outputs[str(figure_manifest.relative_to(inputs.run_dir))] = sha256_file(
        figure_manifest
    )
    for record in tables["tables"].values():
        for path_key, hash_key in (
            ("path", "sha256"),
            ("markdown_path", "markdown_sha256"),
        ):
            outputs[str(record[path_key])] = str(record[hash_key])
    for record in figures["figures"]:
        source_data = record.get("source_data")
        if isinstance(source_data, Mapping):
            outputs[str(source_data["path"])] = str(source_data["sha256"])
        for artifact in record.get("outputs", {}).values():
            outputs[str(artifact["path"])] = str(artifact["sha256"])
    gallery_path = gallery.get("manifest_path")
    if gallery_path:
        path = Path(str(gallery_path))
        outputs[str(path.relative_to(inputs.run_dir))] = sha256_file(path)
    if gallery.get("gallery_path"):
        outputs[str(gallery["gallery_path"])] = str(gallery["gallery_sha256"])
    for record in gallery.get("records", []):
        outputs[str(record["image_path"])] = str(record["image_sha256"])
    return dict(sorted(outputs.items()))


def _resume_result(inputs: FormalPaperInputs) -> PaperArtifactResult | None:
    path = inputs.run_dir / "paper_artifact_manifest.json"
    if not path.exists():
        return None
    _, payload = _read_json(path, "paper artifact manifest")
    if (
        payload.get("schema_version") != PAPER_ARTIFACT_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("scope") != FORMAL_SCOPE
        or payload.get("run_id") != inputs.run_dir.name
        or payload.get("input_fingerprint") != inputs.input_fingerprint
        or payload.get("selected_predicted_condition") != inputs.selected_condition
    ):
        raise PaperArtifactsRefused("saved paper artifact manifest is stale")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise PaperArtifactsRefused("saved paper artifact manifest has no outputs")
    for relative, expected in outputs.items():
        source = _regular_file(inputs.run_dir / str(relative), "saved paper output")
        if sha256_file(source) != _digest(expected, "saved paper output hash"):
            raise PaperArtifactsRefused(f"saved paper output is stale: {relative}")
    figures = payload.get("figures")
    if not isinstance(figures, Mapping):
        raise PaperArtifactsRefused("saved paper manifest lacks figure status")
    return PaperArtifactResult(
        run_dir=inputs.run_dir,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        input_fingerprint=inputs.input_fingerprint,
        selected_predicted_condition=inputs.selected_condition,
        executed_figures=tuple(map(str, figures.get("executed", []))),
        unexecuted_figures=tuple(map(str, figures.get("unexecuted", []))),
        outputs={str(key): str(value) for key, value in outputs.items()},
        resumed=True,
    )


def _restore_resumed_completion(
    inputs: FormalPaperInputs, result: PaperArtifactResult
) -> None:
    """Close a revalidation attempt without touching published paper bytes."""

    manifest = inputs.run_manifest
    completion_status = (
        "COMPLETE_TRAIN3_SUBSET"
        if inputs.run_manifest.get("profile") == "paper-lite-train3"
        else "COMPLETE"
    )
    expected = {
        "status": "COMPLETE",
        "completion_status": completion_status,
        "formal_results_emitted": True,
        "paper_artifact_manifest_sha256": result.manifest_sha256,
        "selected_predicted_condition": inputs.selected_condition,
    }
    if all(manifest.get(key) == value for key, value in expected.items()):
        return
    completed_at: str | None = None
    history = manifest.get("status_history", [])
    if isinstance(history, list):
        for row in reversed(history):
            if isinstance(row, Mapping) and row.get("status") == "COMPLETE":
                value = row.get("ended_at_utc")
                if isinstance(value, str) and value:
                    completed_at = value
                    break
    update_manifest(
        inputs.run_dir,
        **expected,
        ended_at_utc=completed_at or datetime.now(timezone.utc).isoformat(),
    )


def generate_formal_paper_artifacts(
    run_dir: Path | str,
    *,
    resume: bool = False,
) -> PaperArtifactResult:
    """Generate tables, fifteen figure families, gallery, and formal prose.

    Parameters
    ----------
    run_dir:
        A formal run directory containing the four required analyses and every
        upstream evidence artifact.
    resume:
        When true, return an already complete artifact set only after
        revalidating all current inputs and every published output hash.  When
        false, an existing paper artifact manifest is refused.

    Returns
    -------
    PaperArtifactResult
        Paths, hashes, figure status, selected predicted arm, and resume state.

    Raises
    ------
    PaperArtifactsRefused
        If any gate, scope, split, hash, required figure, or evidence contract
        is incomplete.  No top-level formal prose is emitted on such a run.
    """

    inputs = validate_formal_paper_inputs(run_dir)
    existing = _resume_result(inputs)
    if existing is not None:
        if not resume:
            raise PaperArtifactsRefused(
                "paper artifacts already exist; use resume=True for strict revalidation"
            )
        _restore_resumed_completion(inputs, existing)
        return existing

    tables, table_manifest = _publish_tables(inputs)
    figures = _publish_figures(inputs, tables)
    gallery = _publish_failure_gallery(inputs)
    _, document_outputs = _publish_documents(inputs, tables, figures, gallery)
    outputs = _collect_manifest_outputs(
        inputs,
        table_manifest,
        figures,
        gallery,
        document_outputs,
    )
    executed = tuple(
        str(row["id"]) for row in figures["figures"] if row["status"] == "EXECUTED"
    )
    unexecuted = tuple(
        str(row["id"]) for row in figures["figures"] if row["status"] != "EXECUTED"
    )
    payload = {
        "schema_version": PAPER_ARTIFACT_SCHEMA,
        "status": "COMPLETE",
        "scope": FORMAL_SCOPE,
        "fixture_only": False,
        "run_id": inputs.run_dir.name,
        "profile": inputs.run_manifest["profile"],
        "input_fingerprint": inputs.input_fingerprint,
        "immutable_identity_sha256": canonical_sha256(
            inputs.run_manifest["immutable_identity"]
        ),
        "selected_predicted_condition": inputs.selected_condition,
        "source_hashes": dict(inputs.source_hashes),
        "figures": {
            "manifest_sha256": figures["manifest_sha256"],
            "required": len(FIGURE_FAMILIES),
            "executed": list(executed),
            "unexecuted": list(unexecuted),
        },
        "tables": {
            "manifest_sha256": table_manifest["manifest_sha256"],
            "names": sorted(table_manifest["tables"]),
        },
        "gallery": {
            "status": gallery["status"],
            "manifest_sha256": gallery["manifest_sha256"],
        },
        "outputs": outputs,
        "metric_interpretation": (
            "offline target-specific evaluator outcomes; not physical robot success"
        ),
    }
    manifest_path = atomic_json(
        inputs.run_dir / "paper_artifact_manifest.json", payload
    )
    manifest_hash = sha256_file(manifest_path)
    update_manifest(
        inputs.run_dir,
        status="COMPLETE",
        completion_status=(
            "COMPLETE_TRAIN3_SUBSET"
            if inputs.run_manifest["profile"] == "paper-lite-train3"
            else "COMPLETE"
        ),
        ended_at_utc=datetime.now(timezone.utc).isoformat(),
        formal_results_emitted=True,
        official_test_benchmark=False,
        source_scenes=(
            30 if inputs.run_manifest["profile"] == "paper-lite-train3" else None
        ),
        train_scenes=(
            18 if inputs.run_manifest["profile"] == "paper-lite-train3" else None
        ),
        validation_scenes=(
            5 if inputs.run_manifest["profile"] == "paper-lite-train3" else None
        ),
        test_scenes=(
            7 if inputs.run_manifest["profile"] == "paper-lite-train3" else None
        ),
        camera=inputs.run_manifest.get("camera"),
        paper_artifact_manifest_sha256=manifest_hash,
        selected_predicted_condition=inputs.selected_condition,
    )
    return PaperArtifactResult(
        run_dir=inputs.run_dir,
        manifest_path=manifest_path,
        manifest_sha256=manifest_hash,
        input_fingerprint=inputs.input_fingerprint,
        selected_predicted_condition=inputs.selected_condition,
        executed_figures=executed,
        unexecuted_figures=unexecuted,
        outputs=outputs,
        resumed=False,
    )


__all__ = [
    "ANALYSES",
    "CONDITIONS",
    "FIGURE_FAMILIES",
    "FIGURE_MANIFEST_SCHEMA",
    "GALLERY_MANIFEST_SCHEMA",
    "PAPER_ARTIFACT_SCHEMA",
    "TABLE_MANIFEST_SCHEMA",
    "FormalPaperInputs",
    "PaperArtifactResult",
    "PaperArtifactsRefused",
    "generate_formal_paper_artifacts",
    "validate_formal_paper_inputs",
]
