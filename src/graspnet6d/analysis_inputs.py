"""Strict assembly of formal post-feature analysis inputs.

The expensive pipeline commits one feature table and one official-evaluator
label bundle per target group.  This module is the only bridge from those
group-local commits to :mod:`graspnet6d.experiment_analysis`.  It revalidates
the complete frozen-candidate lineage, joins by exact ordered candidate IDs,
and writes immutable, content-addressed partition inputs.

No evaluator outcome is admitted to the versioned feature schema.  Labels are
retained only as explicit supervision columns in the assembled row tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Candidate6D
from .experiment_analysis import (
    FORMAL_SCOPE,
    INPUT_SCHEMA,
    load_analysis_input_manifest,
    validate_split_disjointness,
)
from .features import (
    assert_no_gt_leakage,
    default_feature_schema,
    feature_schema_sha256,
    load_feature_schema,
)
from .formal_inputs import (
    FORMAL_FEATURE_SCHEMA,
    GROUNDING_CONDITIONS,
    _load_candidate_bundle,
    group_artifact_slug,
    load_committed_formal_feature_table,
)
from .io import atomic_json, atomic_text, canonical_sha256, sha256_file
from .metrics import derive_graded_relevance
from .stages import (
    GEOMETRY_EVIDENCE_SCHEMA,
    LABEL_BUNDLE_SCHEMA,
    PARITY_EVIDENCE_SCHEMA,
    load_target_language_jsonl,
)


ASSEMBLY_SCHEMA = "graspnet6d_analysis_input_assembly_v1"
DEFAULT_FEATURE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "graspnet6d"
    / "feature_schema_6d_v1.json"
)
_PARTITIONS = ("train", "validation", "test")
_GROUNDING_ORDER = (
    "oracle_gt_mask",
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMAL_RUN_FORBIDDEN = re.compile(r"fixture|synthetic|dummy|unit[-_]?test", re.I)

_BASE_COLUMNS = (
    "partition",
    "scene_id",
    "group_id",
    "candidate_id",
    "geometry_sha256",
    "native_rank",
    "pre_nms_native_rank",
    "native_score",
    "collision",
    "pose_valid",
    "friction_required",
    "relevance",
    "target_object_id",
    "associated_object_id",
    "target_match",
    "grounding_condition",
)
_UNIVERSE_COLUMNS = (
    "partition",
    "scene_id",
    "group_id",
    "grounding_condition",
    "generation_status",
    "grounding_failure_reason",
)


class AnalysisInputAssemblyError(RuntimeError):
    """A required commit is missing, stale, cross-scoped, or inconsistent."""


@dataclass(frozen=True, slots=True)
class AnalysisInputAssembly:
    """Published immutable analysis-input bundle."""

    manifest_path: Path
    output_dir: Path
    assembly_fingerprint: str
    manifest_sha256: str
    condition: str
    conditions: tuple[str, ...]
    group_count: int
    candidate_count: int
    empty_group_count: int
    partition_rows: Mapping[str, Path]
    partition_group_universes: Mapping[str, Path]
    resumed: bool


@dataclass(frozen=True, slots=True)
class _GroupAssembly:
    group_id: str
    scene_id: str
    partition: str
    rows: tuple[dict[str, Any], ...]
    provenance: Mapping[str, Any]


def _digest(value: Any, description: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise AnalysisInputAssemblyError(
            f"{description} must be a lowercase SHA-256 digest"
        )
    return text


def _regular_file(path: str | os.PathLike[str], description: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AnalysisInputAssemblyError(f"{description} must not be a symlink: {raw}")
    source = raw.resolve()
    if not source.is_file():
        raise AnalysisInputAssemblyError(f"missing regular {description}: {source}")
    return source


def _read_json(
    path: str | os.PathLike[str], description: str
) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, description)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisInputAssemblyError(
            f"invalid {description} {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise AnalysisInputAssemblyError(
            f"{description} must be a JSON object: {source}"
        )
    return source, payload


def _resolve_bound_file(
    owner: Path,
    raw_path: Any,
    raw_sha256: Any,
    description: str,
) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    source = _regular_file(path, description)
    expected = _digest(raw_sha256, f"{description} sha256")
    observed = sha256_file(source)
    if observed != expected:
        raise AnalysisInputAssemblyError(
            f"stale {description}: expected {expected}, observed {observed}: {source}"
        )
    return source


def _source_hashes(raw: Any, *, owner: Path, description: str) -> dict[Path, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise AnalysisInputAssemblyError(f"{description} must be a non-empty mapping")
    result: dict[Path, str] = {}
    for raw_path, raw_digest in raw.items():
        source = _resolve_bound_file(owner, raw_path, raw_digest, description)
        if source in result:
            raise AnalysisInputAssemblyError(
                f"{description} contains duplicate resolved paths: {source}"
            )
        result[source] = str(raw_digest)
    return result


def _require_manifest_binding(
    bindings: Mapping[Path, str], source: Path, expected_sha256: str, description: str
) -> None:
    if bindings.get(source) != expected_sha256:
        raise AnalysisInputAssemblyError(
            f"{description} is not bound to the exact source manifest: {source}"
        )


def _checked_feature_schema() -> tuple[Path, tuple[Any, ...]]:
    source = _regular_file(DEFAULT_FEATURE_SCHEMA_PATH, "checked-in feature schema")
    try:
        declared = tuple(load_feature_schema(source))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise AnalysisInputAssemblyError(
            f"invalid checked-in feature schema {source}: {error}"
        ) from error
    runtime = tuple(default_feature_schema())
    if declared != runtime or feature_schema_sha256(declared) != feature_schema_sha256(
        runtime
    ):
        raise AnalysisInputAssemblyError(
            "checked-in feature schema order/content differs from the runtime extractor schema"
        )
    return source, declared


def _validate_named_bindings(
    verified: Mapping[str, Any], *, owner: Path, description: str
) -> None:
    hashes = verified.get("bindings")
    paths = verified.get("binding_paths")
    if not isinstance(hashes, Mapping) or not hashes:
        raise AnalysisInputAssemblyError(f"{description} lacks provenance bindings")
    if not isinstance(paths, Mapping) or set(paths) != set(hashes):
        raise AnalysisInputAssemblyError(
            f"{description} binding paths and hashes disagree"
        )
    for key in sorted(hashes, key=str):
        _resolve_bound_file(
            owner,
            paths[key],
            hashes[key],
            f"{description} binding {key}",
        )


def _validate_geometry_evidence(raw: Any, *, owner: Path) -> None:
    if not isinstance(raw, Mapping):
        raise AnalysisInputAssemblyError("candidate bundle lacks geometry evidence")
    if raw.get("evidence_policy") != "formal":
        raise AnalysisInputAssemblyError(
            "candidate geometry evidence is not formal_real_data evidence"
        )
    contract = _resolve_bound_file(
        owner,
        raw.get("contract_path"),
        raw.get("contract_sha256"),
        "geometry contract",
    )
    artifact = _resolve_bound_file(
        owner,
        raw.get("validation_artifact"),
        raw.get("validation_artifact_sha256"),
        "geometry validation artifact",
    )
    verified = raw.get("verified_evidence")
    if (
        not isinstance(verified, Mapping)
        or verified.get("evidence_schema") != GEOMETRY_EVIDENCE_SCHEMA
        or verified.get("fixture_only") is True
    ):
        raise AnalysisInputAssemblyError(
            "candidate geometry evidence lacks a formal real-data verification"
        )
    _validate_named_bindings(verified, owner=artifact, description="geometry evidence")
    _resolve_bound_file(
        artifact,
        verified.get("sample_metrics_path"),
        verified.get("sample_metrics_sha256"),
        "geometry sample metrics",
    )
    figures = verified.get("audit_figure_sha256")
    if not isinstance(figures, Mapping) or len(figures) < 20:
        raise AnalysisInputAssemblyError(
            "formal geometry evidence requires at least twenty hashed audit figures"
        )
    for raw_path, raw_hash in figures.items():
        _resolve_bound_file(artifact, raw_path, raw_hash, "geometry audit figure")
    # Keep both owning artifacts live until all transitive checks have passed.
    if not contract.is_file() or not artifact.is_file():  # pragma: no cover - defensive
        raise AnalysisInputAssemblyError(
            "geometry evidence disappeared during assembly"
        )


def _validate_parity_evidence(raw: Any, *, owner: Path) -> None:
    if not isinstance(raw, Mapping):
        raise AnalysisInputAssemblyError("label bundle lacks evaluator parity evidence")
    if raw.get("evidence_policy") != "formal":
        raise AnalysisInputAssemblyError(
            "label parity evidence is not formal_real_data evidence"
        )
    gate = _resolve_bound_file(
        owner, raw.get("gate_path"), raw.get("gate_sha256"), "evaluator parity gate"
    )
    artifact = _resolve_bound_file(
        owner,
        raw.get("artifact_path"),
        raw.get("artifact_sha256"),
        "evaluator parity artifact",
    )
    verified = raw.get("verified_evidence")
    if (
        not isinstance(verified, Mapping)
        or verified.get("evidence_schema") != PARITY_EVIDENCE_SCHEMA
        or verified.get("fixture_only") is True
    ):
        raise AnalysisInputAssemblyError(
            "label parity evidence lacks a formal real-data verification"
        )
    _validate_named_bindings(verified, owner=artifact, description="parity evidence")
    for stem, description in (
        ("comparison_csv", "parity comparison CSV"),
        ("report", "parity report"),
    ):
        _resolve_bound_file(
            artifact,
            verified.get(f"{stem}_path"),
            verified.get(f"{stem}_sha256"),
            description,
        )
    if not gate.is_file() or not artifact.is_file():  # pragma: no cover - defensive
        raise AnalysisInputAssemblyError("parity evidence disappeared during assembly")


def _strict_bool(value: Any, description: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise AnalysisInputAssemblyError(f"{description} must be boolean")
    return bool(value)


def _strict_int(value: Any, description: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise AnalysisInputAssemblyError(f"{description} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisInputAssemblyError(f"{description} must be an integer") from error
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisInputAssemblyError(f"{description} must be an integer") from error
    if not np.isfinite(numeric) or numeric != converted:
        raise AnalysisInputAssemblyError(f"{description} must be an integer")
    return converted


def _strict_float(value: Any, description: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisInputAssemblyError(f"{description} must be finite") from error
    if not np.isfinite(converted):
        raise AnalysisInputAssemblyError(f"{description} must be finite")
    return converted


def _validate_feature_commit(
    sidecar_path: Path,
    *,
    group_id: str,
    scene_id: str,
    partition: str,
    condition: str,
    candidates: Sequence[Candidate6D],
    candidate_pool_fingerprint: str,
    target_manifest: Path,
    target_manifest_sha256: str,
    language_manifest: Path,
    language_manifest_sha256: str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    sidecar, payload = _read_json(sidecar_path, "formal feature commit")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    try:
        frame = load_committed_formal_feature_table(
            sidecar,
            expected_group_id=group_id,
            expected_condition=condition,  # type: ignore[arg-type]
            expected_candidate_ids=candidate_ids,
        )
    except Exception as error:
        raise AnalysisInputAssemblyError(
            f"invalid formal feature commit for {group_id}: {error}"
        ) from error
    if payload.get("schema_version") != FORMAL_FEATURE_SCHEMA:
        raise AnalysisInputAssemblyError("formal feature commit uses another schema")
    if payload.get("scene_id") != scene_id or payload.get("split") != partition:
        raise AnalysisInputAssemblyError(
            f"formal features for {group_id} disagree with target scene/split"
        )
    if payload.get("condition") != condition:
        raise AnalysisInputAssemblyError(
            f"formal features for {group_id} belong to another condition"
        )
    if payload.get("candidate_pool_fingerprint") != candidate_pool_fingerprint:
        raise AnalysisInputAssemblyError(
            f"formal features for {group_id} reference another candidate pool"
        )
    contract = payload.get("input_contract")
    if not isinstance(contract, Mapping):
        raise AnalysisInputAssemblyError(
            f"formal feature commit for {group_id} lacks its input contract"
        )
    if canonical_sha256(contract) != payload.get("input_fingerprint"):
        raise AnalysisInputAssemblyError(
            f"formal feature input fingerprint is stale for {group_id}"
        )
    expected_contract = {
        "schema": FORMAL_FEATURE_SCHEMA,
        "group_id": group_id,
        "scene_id": scene_id,
        "split": partition,
        "condition": condition,
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "feature_schema_sha256": feature_schema_sha256(default_feature_schema()),
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise AnalysisInputAssemblyError(
                f"formal feature input contract mismatch for {group_id}: {key}"
            )
    bindings = _source_hashes(
        contract.get("source_hashes"),
        owner=sidecar,
        description=f"formal feature sources for {group_id}",
    )
    _require_manifest_binding(
        bindings,
        target_manifest,
        target_manifest_sha256,
        f"formal features for {group_id}",
    )
    _require_manifest_binding(
        bindings,
        language_manifest,
        language_manifest_sha256,
        f"formal features for {group_id}",
    )
    if (
        payload.get("official_evaluator_outputs_consumed") != []
        or contract.get("official_evaluator_outputs_consumed") != []
    ):
        raise AnalysisInputAssemblyError(
            f"formal features for {group_id} consumed evaluator labels"
        )
    return frame, payload


def _validate_label_bundle(
    label_path: Path,
    *,
    group_id: str,
    condition: str,
    target_object_id: int,
    candidates: Sequence[Candidate6D],
    candidate_payload: Mapping[str, Any],
    candidate_bundle_path: Path,
    candidate_bundle_sha256: str,
    candidate_pool_fingerprint: str,
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any]]:
    source, payload = _read_json(label_path, "official label bundle")
    if payload.get("schema_version") != LABEL_BUNDLE_SCHEMA:
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} use another schema"
        )
    check = dict(payload)
    observed_fingerprint = check.pop("bundle_fingerprint", None)
    if observed_fingerprint != canonical_sha256(check):
        raise AnalysisInputAssemblyError(
            f"official label bundle fingerprint mismatch for {group_id}"
        )
    if payload.get("group_id") != group_id:
        raise AnalysisInputAssemblyError(
            f"official labels at {source} belong to another group"
        )
    if payload.get("grounding_condition") != condition:
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} belong to another condition"
        )
    if _strict_int(payload.get("target_object_id"), "bundle target_object_id") != (
        target_object_id
    ):
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} target another object"
        )
    declared_candidate_path = Path(
        str(payload.get("candidate_bundle_path", ""))
    ).expanduser()
    if not declared_candidate_path.is_absolute():
        declared_candidate_path = source.parent / declared_candidate_path
    if declared_candidate_path.resolve() != candidate_bundle_path:
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} reference another candidate bundle"
        )
    if payload.get("candidate_bundle_sha256") != candidate_bundle_sha256:
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} are stale for the candidate bundle"
        )
    if payload.get("candidate_pool_fingerprint") != candidate_pool_fingerprint:
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} reference another frozen pool"
        )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if payload.get("candidate_count") != len(candidate_ids):
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} have the wrong candidate count"
        )
    recorded_ids = payload.get("candidate_ids")
    if (
        not isinstance(recorded_ids, list)
        or list(map(str, recorded_ids)) != candidate_ids
    ):
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} changed candidate order/membership"
        )
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != len(candidate_ids):
        raise AnalysisInputAssemblyError(
            f"official labels for {group_id} are incomplete"
        )
    grounding_skipped = (
        candidate_payload.get("generation_status") == "skipped_grounding_failure"
    )
    if not candidate_ids:
        new_empty_contract = (
            payload.get("label_generation_status") == "skipped_empty_pool"
        )
        if new_empty_contract:
            if (
                payload.get("evaluator_operation") != "not_called_empty_frozen_pool"
                or payload.get("evaluator_calls_for_group") != 0
                or payload.get("official_source_hashes") != {}
            ):
                raise AnalysisInputAssemblyError(
                    f"empty-pool labels for {group_id} report fabricated evaluator work"
                )
            if grounding_skipped:
                if (
                    payload.get("empty_pool_reason")
                    != "grounding_failure:"
                    + str(candidate_payload.get("grounding_failure_reason"))
                    or payload.get("grounding_failure_reason")
                    != candidate_payload.get("grounding_failure_reason")
                    or payload.get("grounding_terminal_path")
                    != candidate_payload.get("grounding_terminal_path")
                    or payload.get("grounding_terminal_sha256")
                    != candidate_payload.get("grounding_terminal_sha256")
                ):
                    raise AnalysisInputAssemblyError(
                        f"empty-pool labels for {group_id} do not match the grounding terminal"
                    )
            elif (
                candidate_payload.get("generation_status", "completed_vgn_inference")
                != "completed_vgn_inference"
                or candidate_payload.get("inference_calls_for_group", 1) != 1
                or payload.get("empty_pool_reason") != "vgn_no_candidates"
                or any(
                    key in payload
                    for key in (
                        "grounding_failure_reason",
                        "grounding_terminal_path",
                        "grounding_terminal_sha256",
                    )
                )
            ):
                raise AnalysisInputAssemblyError(
                    f"empty-pool labels for {group_id} do not prove a zero-output VGN call"
                )
        else:
            raise AnalysisInputAssemblyError(
                f"empty-pool labels for {group_id} use an unsupported contract"
            )
    else:
        if (
            payload.get("label_generation_status")
            != "completed_official_low_level_evaluation"
            or payload.get("evaluator_calls_for_group") != 1
            or payload.get("evaluator_operation")
            != "per_candidate_low_level_no_eval_grasp_no_nms_no_topk"
        ):
            raise AnalysisInputAssemblyError(
                f"official labels for {group_id} do not preserve the frozen pool"
            )
        _source_hashes(
            payload.get("official_source_hashes"),
            owner=source,
            description=f"official evaluator sources for {group_id}",
        )
    _validate_parity_evidence(payload.get("parity_gate"), owner=source)

    canonical: list[dict[str, Any]] = []
    for index, (candidate_id, raw) in enumerate(
        zip(candidate_ids, labels, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise AnalysisInputAssemblyError(
                f"official label {index} for {group_id} is not an object"
            )
        if str(raw.get("candidate_id", "")) != candidate_id:
            raise AnalysisInputAssemblyError(
                f"official labels for {group_id} changed candidate ordering"
            )
        if _strict_int(raw.get("candidate_index"), "candidate_index") != index:
            raise AnalysisInputAssemblyError(
                f"official labels for {group_id} changed evaluator row ordering"
            )
        row_target_id = _strict_int(raw.get("target_object_id"), "target_object_id")
        associated_id = _strict_int(
            raw.get("associated_object_id"), "associated_object_id"
        )
        target_match = _strict_bool(raw.get("target_match"), "target_match")
        correct_target = _strict_bool(raw.get("correct_target"), "correct_target")
        if (
            row_target_id != target_object_id
            or target_match != (associated_id == target_object_id)
            or correct_target != target_match
        ):
            raise AnalysisInputAssemblyError(
                f"official association aliases disagree for {group_id}/{candidate_id}"
            )
        pose_valid = _strict_bool(raw.get("pose_valid"), "pose_valid")
        valid_geometry = _strict_bool(raw.get("valid_geometry"), "valid_geometry")
        if pose_valid != valid_geometry:
            raise AnalysisInputAssemblyError(
                f"official pose-valid aliases disagree for {group_id}/{candidate_id}"
            )
        collision = _strict_bool(raw.get("collision"), "collision")
        _strict_bool(raw.get("empty_grasp"), "empty_grasp")
        friction = _strict_float(raw.get("friction_required"), "friction_required")
        friction_native = _strict_float(raw.get("friction_score"), "friction_score")
        if friction != friction_native:
            raise AnalysisInputAssemblyError(
                f"official friction aliases disagree for {group_id}/{candidate_id}"
            )
        relevance = _strict_int(raw.get("relevance"), "relevance")
        canonical.append(
            {
                "candidate_id": candidate_id,
                "target_object_id": target_object_id,
                "associated_object_id": associated_id,
                "target_match": target_match,
                "collision": collision,
                "pose_valid": pose_valid,
                "friction_required": friction,
                "relevance": relevance,
            }
        )
    if canonical:
        derived = derive_graded_relevance(pd.DataFrame(canonical))
        reported = np.asarray([row["relevance"] for row in canonical], dtype=np.int32)
        if not np.array_equal(derived, reported):
            raise AnalysisInputAssemblyError(
                f"official relevance disagrees with canonical outcomes for {group_id}"
            )
    return tuple(canonical), payload


def _assemble_group(
    *,
    group_id: str,
    target: Mapping[str, Any],
    condition: str,
    target_manifest: Path,
    target_manifest_sha256: str,
    language_manifest: Path,
    language_manifest_sha256: str,
    feature_directory: Path,
    label_directory: Path,
    candidate_directory: Path,
    feature_names: Sequence[str],
) -> _GroupAssembly:
    scene_id = str(target.get("scene_id", "")).strip()
    partition = str(target.get("split", "")).strip().lower()
    if not scene_id or partition not in _PARTITIONS:
        raise AnalysisInputAssemblyError(
            f"target {group_id} has invalid scene/split: {scene_id!r}/{partition!r}"
        )
    target_object_id = _strict_int(
        target.get("target_object_id"), f"target_object_id for {group_id}"
    )
    slug = group_artifact_slug(group_id)
    feature_sidecar = feature_directory / f"{slug}.json"
    label_path = label_directory / f"{slug}.json"
    candidate_path = _regular_file(
        candidate_directory / f"{slug}.json", "frozen candidate bundle"
    )
    try:
        candidates, candidate_payload, _, _ = _load_candidate_bundle(
            candidate_path, group_id
        )
    except Exception as error:
        raise AnalysisInputAssemblyError(
            f"invalid frozen candidate bundle for {group_id}: {error}"
        ) from error
    if candidate_payload.get("grounding_condition") != condition:
        raise AnalysisInputAssemblyError(
            f"frozen candidates for {group_id} belong to another condition"
        )
    _validate_geometry_evidence(
        candidate_payload.get("geometry_contract"), owner=candidate_path
    )
    ranks = [candidate.native_rank for candidate in candidates]
    if any(rank < 1 for rank in ranks) or ranks != sorted(set(ranks)):
        raise AnalysisInputAssemblyError(
            f"frozen candidates for {group_id} do not preserve increasing unique pre-NMS ranks"
        )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AnalysisInputAssemblyError(
            f"frozen candidates for {group_id} contain duplicate IDs"
        )
    pre_nms_status = candidate_payload.get("pre_nms_snapshot_status")
    pre_nms_rank_by_id: dict[str, int] = {}
    if pre_nms_status == "complete_same_inference":
        raw_pre_nms = candidate_payload.get("pre_nms_vgn_candidates")
        if not isinstance(raw_pre_nms, list):
            raise AnalysisInputAssemblyError(
                f"frozen candidates for {group_id} lack their pre-NMS snapshot"
            )
        try:
            pre_nms_rank_by_id = {
                str(record["candidate_id"]): _strict_int(
                    record["native_rank"], "pre_nms_native_rank"
                )
                for record in raw_pre_nms
            }
        except (KeyError, TypeError) as error:
            raise AnalysisInputAssemblyError(
                f"frozen candidates for {group_id} have invalid pre-NMS ranks"
            ) from error
        if len(pre_nms_rank_by_id) != len(raw_pre_nms) or any(
            pre_nms_rank_by_id.get(candidate.candidate_id) != candidate.native_rank
            for candidate in candidates
        ):
            raise AnalysisInputAssemblyError(
                f"frozen candidates for {group_id} are not bound to pre-NMS ranks"
            )
    elif pre_nms_status not in {None, "not_applicable_grounding_failure"}:
        raise AnalysisInputAssemblyError(
            f"frozen candidates for {group_id} have a non-formal pre-NMS snapshot"
        )
    pool_fingerprint = _digest(
        candidate_payload.get("candidate_pool_fingerprint"),
        f"candidate pool fingerprint for {group_id}",
    )
    candidate_sha = sha256_file(candidate_path)
    features, feature_commit = _validate_feature_commit(
        feature_sidecar,
        group_id=group_id,
        scene_id=scene_id,
        partition=partition,
        condition=condition,
        candidates=candidates,
        candidate_pool_fingerprint=pool_fingerprint,
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
        language_manifest=language_manifest,
        language_manifest_sha256=language_manifest_sha256,
    )
    labels, label_bundle = _validate_label_bundle(
        label_path,
        group_id=group_id,
        condition=condition,
        target_object_id=target_object_id,
        candidates=candidates,
        candidate_payload=candidate_payload,
        candidate_bundle_path=candidate_path,
        candidate_bundle_sha256=candidate_sha,
        candidate_pool_fingerprint=pool_fingerprint,
    )
    if len(features) != len(labels):
        raise AnalysisInputAssemblyError(
            f"feature/label candidate universes differ for {group_id}"
        )
    rows: list[dict[str, Any]] = []
    for index, (candidate, label) in enumerate(zip(candidates, labels, strict=True)):
        feature_row = features.iloc[index]
        if str(feature_row["candidate_id"]) != candidate.candidate_id:
            raise AnalysisInputAssemblyError(
                f"feature order changed for {group_id}/{candidate.candidate_id}"
            )
        feature_rank = _strict_int(feature_row["native_rank"], "feature native_rank")
        feature_score = _strict_float(
            feature_row["native_score"], "feature native_score"
        )
        if (
            feature_rank != candidate.native_rank
            or feature_score != candidate.native_score
        ):
            raise AnalysisInputAssemblyError(
                f"native score/rank changed for {group_id}/{candidate.candidate_id}"
            )
        row: dict[str, Any] = {
            "partition": partition,
            "scene_id": scene_id,
            "group_id": group_id,
            "candidate_id": candidate.candidate_id,
            "geometry_sha256": candidate.geometry_sha256,
            "native_rank": candidate.native_rank,
            "pre_nms_native_rank": pre_nms_rank_by_id.get(candidate.candidate_id),
            "native_score": candidate.native_score,
            "collision": label["collision"],
            "pose_valid": label["pose_valid"],
            "friction_required": label["friction_required"],
            "relevance": label["relevance"],
            "target_object_id": label["target_object_id"],
            "associated_object_id": label["associated_object_id"],
            "target_match": label["target_match"],
            "grounding_condition": condition,
        }
        for feature_name in feature_names:
            if feature_name not in row:
                row[feature_name] = feature_row[feature_name]
        rows.append(row)
    return _GroupAssembly(
        group_id=group_id,
        scene_id=scene_id,
        partition=partition,
        rows=tuple(rows),
        provenance={
            "group_id": group_id,
            "scene_id": scene_id,
            "partition": partition,
            "candidate_count": len(candidates),
            "candidate_ids_sha256": canonical_sha256(candidate_ids),
            "candidate_pool_fingerprint": pool_fingerprint,
            "candidate_bundle_path": str(candidate_path),
            "candidate_bundle_sha256": candidate_sha,
            "generation_status": candidate_payload.get(
                "generation_status", "completed_vgn_inference"
            ),
            "pre_nms_snapshot_status": pre_nms_status,
            "pre_nms_candidate_count": candidate_payload.get("pre_nms_candidate_count"),
            "pre_nms_pool_fingerprint": candidate_payload.get(
                "pre_nms_pool_fingerprint"
            ),
            "a7_top_k_membership_fingerprint": candidate_payload.get(
                "a7_top_k_membership_fingerprint"
            ),
            **(
                {
                    "grounding_failure_reason": candidate_payload[
                        "grounding_failure_reason"
                    ],
                    "grounding_terminal_path": candidate_payload[
                        "grounding_terminal_path"
                    ],
                    "grounding_terminal_sha256": candidate_payload[
                        "grounding_terminal_sha256"
                    ],
                }
                if candidate_payload.get("generation_status")
                == "skipped_grounding_failure"
                else {}
            ),
            "feature_commit_path": str(feature_sidecar.resolve()),
            "feature_commit_sha256": sha256_file(feature_sidecar),
            "feature_table_sha256": feature_commit["feature_sha256"],
            "label_bundle_path": str(label_path.resolve()),
            "label_bundle_sha256": sha256_file(label_path),
            "label_bundle_fingerprint": label_bundle["bundle_fingerprint"],
        },
    )


def _assert_exact_directory_membership(
    directory: Path,
    *,
    expected_json: set[str],
    expected_parquet: set[str] | None,
    description: str,
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise AnalysisInputAssemblyError(f"missing regular {description}: {directory}")
    observed_json = {path.name for path in directory.glob("*.json") if path.is_file()}
    if observed_json != expected_json:
        raise AnalysisInputAssemblyError(
            f"{description} JSON universe differs: "
            f"missing={sorted(expected_json - observed_json)[:5]}, "
            f"extra={sorted(observed_json - expected_json)[:5]}"
        )
    if expected_parquet is not None:
        observed_parquet = {
            path.name for path in directory.glob("*.parquet") if path.is_file()
        }
        if observed_parquet != expected_parquet:
            raise AnalysisInputAssemblyError(
                f"{description} Parquet universe differs: "
                f"missing={sorted(expected_parquet - observed_parquet)[:5]}, "
                f"extra={sorted(observed_parquet - expected_parquet)[:5]}"
            )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            suffix=".parquet",
            prefix=f".{destination.stem}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _resume_result(
    final_dir: Path,
    *,
    condition: str,
    conditions: Sequence[str] | None = None,
    assembly_fingerprint: str,
    group_count: int,
    candidate_count: int,
    empty_group_count: int,
) -> AnalysisInputAssembly:
    manifest_path = final_dir / "input_manifest.json"
    try:
        loaded = load_analysis_input_manifest(manifest_path)
    except Exception as error:
        raise AnalysisInputAssemblyError(
            f"content-addressed output exists but is not resumable: {final_dir}: {error}"
        ) from error
    if (
        loaded.status != "COMPLETE"
        or loaded.scope != FORMAL_SCOPE
        or loaded.fixture_only
    ):
        raise AnalysisInputAssemblyError(
            f"content-addressed output is not a complete formal input: {final_dir}"
        )
    provenance_path = _regular_file(
        final_dir / "assembly_provenance.json", "assembly provenance"
    )
    if loaded.provenance.get("assembly_provenance_sha256") != sha256_file(
        provenance_path
    ):
        raise AnalysisInputAssemblyError(
            f"content-addressed assembly provenance is stale: {final_dir}"
        )
    return AnalysisInputAssembly(
        manifest_path=manifest_path,
        output_dir=final_dir,
        assembly_fingerprint=assembly_fingerprint,
        manifest_sha256=sha256_file(manifest_path),
        condition=condition,
        conditions=tuple(conditions or (condition,)),
        group_count=group_count,
        candidate_count=candidate_count,
        empty_group_count=empty_group_count,
        partition_rows={
            name: loaded.partitions[name].rows.path for name in _PARTITIONS
        },
        partition_group_universes={
            name: loaded.partitions[name].group_universe.path for name in _PARTITIONS
        },
        resumed=True,
    )


def assemble_analysis_inputs(
    target_manifest_path: str | os.PathLike[str],
    language_manifest_path: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    condition: str,
    run_id: str,
    formal_feature_root: str | os.PathLike[str] | None = None,
    official_label_root: str | os.PathLike[str] | None = None,
    candidate_root: str | os.PathLike[str] | None = None,
) -> AnalysisInputAssembly:
    """Publish one COMPLETE ``INPUT_SCHEMA`` bundle for a grounding condition.

    ``artifact_root`` is normally the run directory.  The current strict
    producer writes formal feature commits below ``candidate_features``;
    ``formal_feature_root`` permits an explicitly named ``formal_features``
    root without weakening path or condition checks.  Root overrides refer to
    the parent of the condition directory.

    The function fails before publication unless every target group has a
    committed feature table, frozen candidate bundle, and official label
    bundle.  A zero-candidate group is legal and appears only in its partition
    group universe.  A whole partition with no candidates is not analysable
    and is rejected.
    """

    selected = str(condition).strip()
    if selected not in GROUNDING_CONDITIONS:
        raise AnalysisInputAssemblyError(
            f"condition must be one of {sorted(GROUNDING_CONDITIONS)}"
        )
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id or _FORMAL_RUN_FORBIDDEN.search(normalized_run_id):
        raise AnalysisInputAssemblyError(
            "formal run_id must be non-empty and must not be fixture-labelled"
        )
    target_manifest = _regular_file(target_manifest_path, "target manifest")
    language_manifest = _regular_file(language_manifest_path, "language manifest")
    target_manifest_sha = sha256_file(target_manifest)
    language_manifest_sha = sha256_file(language_manifest)
    try:
        groups = load_target_language_jsonl(target_manifest, language_manifest)
    except Exception as error:
        raise AnalysisInputAssemblyError(
            f"invalid target/language manifests: {error}"
        ) from error
    if not groups:
        raise AnalysisInputAssemblyError("target/language group universe is empty")

    artifacts = Path(artifact_root).expanduser().resolve()
    feature_base = (
        Path(formal_feature_root).expanduser().resolve()
        if formal_feature_root is not None
        else artifacts / "candidate_features"
    )
    label_base = (
        Path(official_label_root).expanduser().resolve()
        if official_label_root is not None
        else artifacts / "official_labels"
    )
    candidate_base = (
        Path(candidate_root).expanduser().resolve()
        if candidate_root is not None
        else artifacts / "vgn_candidates"
    )
    feature_directory = feature_base / selected
    label_directory = label_base / selected
    candidate_directory = candidate_base / selected

    expected_slugs = {group_artifact_slug(group.group_id) for group in groups}
    _assert_exact_directory_membership(
        feature_directory,
        expected_json={f"{slug}.json" for slug in expected_slugs},
        expected_parquet={f"{slug}.parquet" for slug in expected_slugs},
        description=f"formal feature directory for {selected}",
    )
    _assert_exact_directory_membership(
        label_directory,
        expected_json={f"{slug}.json" for slug in expected_slugs},
        expected_parquet=None,
        description=f"official label directory for {selected}",
    )
    _assert_exact_directory_membership(
        candidate_directory,
        expected_json={f"{slug}.json" for slug in expected_slugs},
        expected_parquet=None,
        description=f"frozen candidate directory for {selected}",
    )

    feature_schema_source, feature_specs = _checked_feature_schema()
    feature_names = tuple(spec.name for spec in feature_specs)
    assert_no_gt_leakage(feature_names)
    if set(feature_names).intersection(
        {
            "collision",
            "pose_valid",
            "friction_required",
            "relevance",
            "target_object_id",
            "associated_object_id",
            "target_match",
        }
    ):
        raise AssertionError("runtime feature schema contains evaluator supervision")

    scene_splits: dict[str, str] = {}
    split_group_counts = {partition: 0 for partition in _PARTITIONS}
    assemblies: list[_GroupAssembly] = []
    for group in groups:
        scene_id = str(group.target.get("scene_id", "")).strip()
        partition = str(group.target.get("split", "")).strip().lower()
        prior = scene_splits.setdefault(scene_id, partition)
        if prior != partition:
            raise AnalysisInputAssemblyError(
                f"scene split leakage for {scene_id}: {prior} vs {partition}"
            )
        if partition in split_group_counts:
            split_group_counts[partition] += 1
        try:
            assemblies.append(
                _assemble_group(
                    group_id=group.group_id,
                    target=group.target,
                    condition=selected,
                    target_manifest=target_manifest,
                    target_manifest_sha256=target_manifest_sha,
                    language_manifest=language_manifest,
                    language_manifest_sha256=language_manifest_sha,
                    feature_directory=feature_directory,
                    label_directory=label_directory,
                    candidate_directory=candidate_directory,
                    feature_names=feature_names,
                )
            )
        except AnalysisInputAssemblyError:
            raise
        except Exception as error:
            raise AnalysisInputAssemblyError(
                f"cannot assemble group {group.group_id}: {error}"
            ) from error
    missing_partitions = [
        partition for partition, count in split_group_counts.items() if count == 0
    ]
    if missing_partitions:
        raise AnalysisInputAssemblyError(
            f"formal target universe lacks partitions: {missing_partitions}"
        )

    all_candidate_ids = [
        str(row["candidate_id"]) for assembly in assemblies for row in assembly.rows
    ]
    if len(all_candidate_ids) != len(set(all_candidate_ids)):
        raise AnalysisInputAssemblyError(
            "candidate IDs are not globally unique across target groups"
        )
    row_columns = (
        *_BASE_COLUMNS,
        *(name for name in feature_names if name not in _BASE_COLUMNS),
    )
    row_frames: dict[str, pd.DataFrame] = {}
    universe_frames: dict[str, pd.DataFrame] = {}
    for partition in _PARTITIONS:
        selected_groups = [item for item in assemblies if item.partition == partition]
        rows = [row for item in selected_groups for row in item.rows]
        if not rows:
            raise AnalysisInputAssemblyError(
                f"partition {partition} has no candidates and cannot enter analysis"
            )
        row_frames[partition] = pd.DataFrame(rows, columns=row_columns)
        universe_frames[partition] = pd.DataFrame(
            [
                {
                    "partition": partition,
                    "scene_id": item.scene_id,
                    "group_id": item.group_id,
                    "grounding_condition": selected,
                    "generation_status": item.provenance["generation_status"],
                    "grounding_failure_reason": item.provenance.get(
                        "grounding_failure_reason"
                    ),
                }
                for item in selected_groups
            ],
            columns=_UNIVERSE_COLUMNS,
        )
    validate_split_disjointness(row_frames, universe_frames)

    group_provenance = [dict(item.provenance) for item in assemblies]
    feature_commit_index = {
        item.group_id: item.provenance["feature_commit_sha256"] for item in assemblies
    }
    label_bundle_index = {
        item.group_id: item.provenance["label_bundle_sha256"] for item in assemblies
    }
    candidate_bundle_index = {
        item.group_id: item.provenance["candidate_bundle_sha256"] for item in assemblies
    }
    feature_schema_source_sha = sha256_file(feature_schema_source)
    assembly_contract = {
        "schema_version": ASSEMBLY_SCHEMA,
        "run_id": normalized_run_id,
        "scope": FORMAL_SCOPE,
        "fixture_only": False,
        "condition": selected,
        "target_manifest_sha256": target_manifest_sha,
        "language_manifest_sha256": language_manifest_sha,
        "runtime_feature_schema_sha256": feature_schema_sha256(feature_specs),
        "checked_in_feature_schema_sha256": feature_schema_source_sha,
        "feature_commit_index": feature_commit_index,
        "label_bundle_index": label_bundle_index,
        "candidate_bundle_index": candidate_bundle_index,
    }
    assembly_fingerprint = canonical_sha256(assembly_contract)
    condition_output = (
        Path(output_root).expanduser().resolve() / "analysis_inputs" / selected
    )
    condition_output.mkdir(parents=True, exist_ok=True)
    final_dir = condition_output / assembly_fingerprint
    candidate_count = len(all_candidate_ids)
    empty_group_count = sum(not item.rows for item in assemblies)
    if final_dir.exists():
        return _resume_result(
            final_dir,
            condition=selected,
            assembly_fingerprint=assembly_fingerprint,
            group_count=len(assemblies),
            candidate_count=candidate_count,
            empty_group_count=empty_group_count,
        )

    with tempfile.TemporaryDirectory(
        dir=condition_output, prefix=".assembling-"
    ) as temporary_name:
        staging = Path(temporary_name)
        if (
            sha256_file(target_manifest) != target_manifest_sha
            or sha256_file(language_manifest) != language_manifest_sha
            or sha256_file(feature_schema_source) != feature_schema_source_sha
        ):
            raise AnalysisInputAssemblyError(
                "source manifests/schema changed during analysis-input assembly"
            )
        schema_path = atomic_text(
            staging / "feature_schema.json",
            feature_schema_source.read_text(encoding="utf-8"),
        )
        if sha256_file(schema_path) != feature_schema_source_sha:
            raise AnalysisInputAssemblyError(
                "copied feature schema differs from the checked-in source"
            )
        partition_entries: dict[str, Any] = {}
        output_hashes: dict[str, str] = {}
        for partition in _PARTITIONS:
            rows_path = _atomic_parquet(
                staging / f"{partition}_rows.parquet", row_frames[partition]
            )
            universe_path = _atomic_parquet(
                staging / f"{partition}_group_universe.parquet",
                universe_frames[partition],
            )
            rows_sha = sha256_file(rows_path)
            universe_sha = sha256_file(universe_path)
            output_hashes[rows_path.name] = rows_sha
            output_hashes[universe_path.name] = universe_sha
            partition_entries[partition] = {
                "rows": {"path": rows_path.name, "sha256": rows_sha},
                "group_universe": {
                    "path": universe_path.name,
                    "sha256": universe_sha,
                },
            }
        provenance_payload = {
            **assembly_contract,
            "assembly_fingerprint": assembly_fingerprint,
            "status": "COMPLETE",
            "target_manifest_path": str(target_manifest),
            "language_manifest_path": str(language_manifest),
            "feature_directory": str(feature_directory),
            "label_directory": str(label_directory),
            "candidate_directory": str(candidate_directory),
            "group_count": len(assemblies),
            "candidate_count": candidate_count,
            "empty_group_count": empty_group_count,
            "partition_group_counts": split_group_counts,
            "partition_candidate_counts": {
                partition: len(row_frames[partition]) for partition in _PARTITIONS
            },
            "group_commits": group_provenance,
            "output_sha256": output_hashes,
        }
        provenance_path = atomic_json(
            staging / "assembly_provenance.json", provenance_payload
        )
        manifest_payload = {
            "schema_version": INPUT_SCHEMA,
            "run_id": normalized_run_id,
            "status": "COMPLETE",
            "scope": FORMAL_SCOPE,
            "fixture_only": False,
            "feature_schema": {
                "path": schema_path.name,
                "sha256": sha256_file(schema_path),
            },
            "partitions": partition_entries,
            "provenance": {
                "target_manifest_sha256": target_manifest_sha,
                "language_manifest_sha256": language_manifest_sha,
                "feature_commit_index_sha256": canonical_sha256(feature_commit_index),
                "official_label_index_sha256": canonical_sha256(label_bundle_index),
                "candidate_pool_index_sha256": canonical_sha256(candidate_bundle_index),
                "assembly_provenance_sha256": sha256_file(provenance_path),
            },
        }
        atomic_json(staging / "input_manifest.json", manifest_payload)
        if final_dir.exists():
            return _resume_result(
                final_dir,
                condition=selected,
                assembly_fingerprint=assembly_fingerprint,
                group_count=len(assemblies),
                candidate_count=candidate_count,
                empty_group_count=empty_group_count,
            )
        os.replace(staging, final_dir)

    result = _resume_result(
        final_dir,
        condition=selected,
        assembly_fingerprint=assembly_fingerprint,
        group_count=len(assemblies),
        candidate_count=candidate_count,
        empty_group_count=empty_group_count,
    )
    return AnalysisInputAssembly(
        manifest_path=result.manifest_path,
        output_dir=result.output_dir,
        assembly_fingerprint=result.assembly_fingerprint,
        manifest_sha256=result.manifest_sha256,
        condition=result.condition,
        conditions=result.conditions,
        group_count=result.group_count,
        candidate_count=result.candidate_count,
        empty_group_count=result.empty_group_count,
        partition_rows=dict(result.partition_rows),
        partition_group_universes=dict(result.partition_group_universes),
        resumed=False,
    )


def assemble_combined_analysis_inputs(
    target_manifest_path: str | os.PathLike[str],
    language_manifest_path: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    run_id: str,
    conditions: Sequence[str] = _GROUNDING_ORDER,
    formal_feature_root: str | os.PathLike[str] | None = None,
    official_label_root: str | os.PathLike[str] | None = None,
    candidate_root: str | os.PathLike[str] | None = None,
) -> AnalysisInputAssembly:
    """Combine all formal grounding arms for the A8 analysis.

    Each source condition is first assembled and revalidated independently by
    :func:`assemble_analysis_inputs`.  Analysis group and candidate IDs are
    then deterministically namespaced by condition; geometry hashes and native
    order remain unchanged.  Namespacing is necessary because the same target
    group legitimately appears in every grounding arm, while the downstream
    split and frozen-pool contracts require globally unique identifiers.

    A COMPLETE combined bundle requires exactly the three preregistered
    grounding conditions.  Omitting an arm would make A8 silently partial.
    """

    selected_conditions = tuple(str(value).strip() for value in conditions)
    if len(selected_conditions) != len(set(selected_conditions)) or set(
        selected_conditions
    ) != set(_GROUNDING_ORDER):
        raise AnalysisInputAssemblyError(
            "combined formal inputs require exactly oracle_gt_mask, "
            "hifics_zero_shot_mask, and hifics_adapted_mask"
        )
    ordered_conditions = tuple(
        value for value in _GROUNDING_ORDER if value in selected_conditions
    )
    condition_results = {
        condition: assemble_analysis_inputs(
            target_manifest_path,
            language_manifest_path,
            artifact_root,
            output_root,
            condition=condition,
            run_id=run_id,
            formal_feature_root=formal_feature_root,
            official_label_root=official_label_root,
            candidate_root=candidate_root,
        )
        for condition in ordered_conditions
    }
    loaded = {
        condition: load_analysis_input_manifest(result.manifest_path)
        for condition, result in condition_results.items()
    }
    reference = loaded[ordered_conditions[0]]
    for condition in ordered_conditions[1:]:
        current = loaded[condition]
        if (
            current.run_id != reference.run_id
            or current.scope != FORMAL_SCOPE
            or current.status != "COMPLETE"
            or current.feature_schema.sha256 != reference.feature_schema.sha256
            or current.provenance.get("target_manifest_sha256")
            != reference.provenance.get("target_manifest_sha256")
            or current.provenance.get("language_manifest_sha256")
            != reference.provenance.get("language_manifest_sha256")
        ):
            raise AnalysisInputAssemblyError(
                f"condition {condition} does not share the formal target/schema provenance"
            )

    combined_rows: dict[str, pd.DataFrame] = {}
    combined_universes: dict[str, pd.DataFrame] = {}
    namespace_records: list[dict[str, str]] = []
    for partition in _PARTITIONS:
        row_parts: list[pd.DataFrame] = []
        universe_parts: list[pd.DataFrame] = []
        reference_universe: pd.DataFrame | None = None
        for condition in ordered_conditions:
            current = loaded[condition]
            rows = pd.read_parquet(current.partitions[partition].rows.path)
            universe = pd.read_parquet(
                current.partitions[partition].group_universe.path
            )
            if not rows["grounding_condition"].astype(str).eq(condition).all():
                raise AnalysisInputAssemblyError(
                    f"{condition}/{partition} rows contain another grounding condition"
                )
            if not universe["grounding_condition"].astype(str).eq(condition).all():
                raise AnalysisInputAssemblyError(
                    f"{condition}/{partition} universe contains another condition"
                )
            source_universe = (
                universe[["partition", "scene_id", "group_id"]]
                .astype(str)
                .sort_values(["group_id"], kind="mergesort")
                .reset_index(drop=True)
            )
            if reference_universe is None:
                reference_universe = source_universe
            elif not source_universe.equals(reference_universe):
                raise AnalysisInputAssemblyError(
                    f"condition {condition} has another {partition} target universe"
                )
            source_group_ids = universe["group_id"].astype(str)
            group_mapping = {
                source_id: f"{condition}::{source_id}" for source_id in source_group_ids
            }
            namespaced_universe = universe.copy()
            namespaced_universe["group_id"] = source_group_ids.map(group_mapping)
            namespaced_rows = rows.copy()
            namespaced_rows["group_id"] = (
                namespaced_rows["group_id"].astype(str).map(group_mapping)
            )
            source_candidate_ids = namespaced_rows["candidate_id"].astype(str)
            namespaced_rows["candidate_id"] = source_candidate_ids.map(
                lambda value: f"{condition}::{value}"
            )
            namespace_records.extend(
                {
                    "condition": condition,
                    "source_group_id": source_group_id,
                    "analysis_group_id": analysis_group_id,
                }
                for source_group_id, analysis_group_id in group_mapping.items()
            )
            row_parts.append(namespaced_rows)
            universe_parts.append(namespaced_universe)
        combined_rows[partition] = (
            pd.concat(row_parts, ignore_index=True)
            .sort_values(["group_id", "native_rank", "candidate_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        combined_universes[partition] = (
            pd.concat(universe_parts, ignore_index=True)
            .sort_values(["group_id"], kind="mergesort")
            .reset_index(drop=True)
        )
    validate_split_disjointness(combined_rows, combined_universes)
    if any(
        frame["candidate_id"].astype(str).duplicated().any()
        for frame in combined_rows.values()
    ):
        raise AnalysisInputAssemblyError(
            "combined grounding namespace did not make candidate IDs unique"
        )

    source_manifest_index = {
        condition: result.manifest_sha256
        for condition, result in condition_results.items()
    }
    namespace_sha = canonical_sha256(namespace_records)
    feature_schema_source, feature_specs = _checked_feature_schema()
    feature_schema_source_sha = sha256_file(feature_schema_source)
    assembly_contract = {
        "schema_version": ASSEMBLY_SCHEMA,
        "kind": "combined_grounding_conditions",
        "run_id": reference.run_id,
        "scope": FORMAL_SCOPE,
        "fixture_only": False,
        "conditions": list(ordered_conditions),
        "source_input_manifest_sha256": source_manifest_index,
        "target_manifest_sha256": reference.provenance["target_manifest_sha256"],
        "language_manifest_sha256": reference.provenance["language_manifest_sha256"],
        "runtime_feature_schema_sha256": feature_schema_sha256(feature_specs),
        "checked_in_feature_schema_sha256": feature_schema_source_sha,
        "identifier_namespace_sha256": namespace_sha,
    }
    assembly_fingerprint = canonical_sha256(assembly_contract)
    condition_key = "combined_grounding_conditions"
    condition_output = (
        Path(output_root).expanduser().resolve() / "analysis_inputs" / condition_key
    )
    condition_output.mkdir(parents=True, exist_ok=True)
    final_dir = condition_output / assembly_fingerprint
    group_count = sum(len(frame) for frame in combined_universes.values())
    candidate_count = sum(len(frame) for frame in combined_rows.values())
    empty_group_count = group_count - sum(
        frame["group_id"].astype(str).nunique() for frame in combined_rows.values()
    )
    if final_dir.exists():
        return _resume_result(
            final_dir,
            condition=condition_key,
            conditions=ordered_conditions,
            assembly_fingerprint=assembly_fingerprint,
            group_count=group_count,
            candidate_count=candidate_count,
            empty_group_count=empty_group_count,
        )

    with tempfile.TemporaryDirectory(
        dir=condition_output, prefix=".assembling-"
    ) as temporary_name:
        staging = Path(temporary_name)
        schema_path = atomic_text(
            staging / "feature_schema.json",
            feature_schema_source.read_text(encoding="utf-8"),
        )
        if sha256_file(schema_path) != feature_schema_source_sha:
            raise AnalysisInputAssemblyError(
                "copied feature schema differs from the checked-in source"
            )
        partition_entries: dict[str, Any] = {}
        output_hashes: dict[str, str] = {}
        for partition in _PARTITIONS:
            rows_path = _atomic_parquet(
                staging / f"{partition}_rows.parquet", combined_rows[partition]
            )
            universe_path = _atomic_parquet(
                staging / f"{partition}_group_universe.parquet",
                combined_universes[partition],
            )
            rows_sha = sha256_file(rows_path)
            universe_sha = sha256_file(universe_path)
            output_hashes[rows_path.name] = rows_sha
            output_hashes[universe_path.name] = universe_sha
            partition_entries[partition] = {
                "rows": {"path": rows_path.name, "sha256": rows_sha},
                "group_universe": {
                    "path": universe_path.name,
                    "sha256": universe_sha,
                },
            }
        provenance_payload = {
            **assembly_contract,
            "assembly_fingerprint": assembly_fingerprint,
            "status": "COMPLETE",
            "source_input_manifests": {
                condition: str(result.manifest_path)
                for condition, result in condition_results.items()
            },
            "identifier_namespace": {
                "algorithm": "condition_double_colon_prefix_v1",
                "mapping_sha256": namespace_sha,
                "mapping": namespace_records,
            },
            "group_count": group_count,
            "candidate_count": candidate_count,
            "empty_group_count": empty_group_count,
            "partition_group_counts": {
                partition: len(combined_universes[partition])
                for partition in _PARTITIONS
            },
            "partition_candidate_counts": {
                partition: len(combined_rows[partition]) for partition in _PARTITIONS
            },
            "output_sha256": output_hashes,
        }
        provenance_path = atomic_json(
            staging / "assembly_provenance.json", provenance_payload
        )
        manifest_payload = {
            "schema_version": INPUT_SCHEMA,
            "run_id": reference.run_id,
            "status": "COMPLETE",
            "scope": FORMAL_SCOPE,
            "fixture_only": False,
            "feature_schema": {
                "path": schema_path.name,
                "sha256": sha256_file(schema_path),
            },
            "partitions": partition_entries,
            "provenance": {
                "target_manifest_sha256": reference.provenance[
                    "target_manifest_sha256"
                ],
                "language_manifest_sha256": reference.provenance[
                    "language_manifest_sha256"
                ],
                "source_input_manifest_index_sha256": canonical_sha256(
                    source_manifest_index
                ),
                "identifier_namespace_sha256": namespace_sha,
                "assembly_provenance_sha256": sha256_file(provenance_path),
            },
        }
        atomic_json(staging / "input_manifest.json", manifest_payload)
        if final_dir.exists():
            return _resume_result(
                final_dir,
                condition=condition_key,
                conditions=ordered_conditions,
                assembly_fingerprint=assembly_fingerprint,
                group_count=group_count,
                candidate_count=candidate_count,
                empty_group_count=empty_group_count,
            )
        os.replace(staging, final_dir)

    result = _resume_result(
        final_dir,
        condition=condition_key,
        conditions=ordered_conditions,
        assembly_fingerprint=assembly_fingerprint,
        group_count=group_count,
        candidate_count=candidate_count,
        empty_group_count=empty_group_count,
    )
    return AnalysisInputAssembly(
        manifest_path=result.manifest_path,
        output_dir=result.output_dir,
        assembly_fingerprint=result.assembly_fingerprint,
        manifest_sha256=result.manifest_sha256,
        condition=result.condition,
        conditions=result.conditions,
        group_count=result.group_count,
        candidate_count=result.candidate_count,
        empty_group_count=result.empty_group_count,
        partition_rows=dict(result.partition_rows),
        partition_group_universes=dict(result.partition_group_universes),
        resumed=False,
    )


__all__ = [
    "ASSEMBLY_SCHEMA",
    "AnalysisInputAssembly",
    "AnalysisInputAssemblyError",
    "assemble_analysis_inputs",
    "assemble_combined_analysis_inputs",
]
