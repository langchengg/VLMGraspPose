"""Evaluation-only subgroup reporting for the frozen CROG V3 cohort.

The producer consumes already-materialized candidates, analysis metadata,
labels, and rankings.  It has no model callback and performs no inference.  In
particular, labels and GT-derived metadata are joined only while computing
metrics after all three ranking artifacts already exist.

Candidate geometry is identified by the tuple ``(stable sample ID, candidate
ID, candidate checksum)``.  Corrected and legacy labels must carry and match
the frozen checksum.  Ranking artifacts must contain the exact candidate ID
set; when they carry checksums (V3 does, legacy V2 may not), those checksums are
also required to match.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import SCHEMA_VERSION
from .analysis import subgroup_metrics
from .reporting import write_csv
from .schema import (
    artifact_identity,
    atomic_write_json,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    stable_sample_id,
)


UNKNOWN = "unknown"
TRACKS = (
    "corrected_scientific",
    "legacy_official_compatibility",
)
METHODS = (
    "q_only",
    "v2_locked_primary",
    "v3_locked_primary",
)
SUBGROUP_FIELDS = (
    "expression_family",
    "token_length",
    "query_complexity",
    "q_margin",
    "v2_switch",
    "original_first_correct_rank",
    "predicted_mask_support",
    "predicted_mask_uncertainty",
    "depth_valid_fraction",
    "object_size",
    "candidate_angle_confidence",
    "width_consistency",
    "corrected_legacy_disagreement",
    "candidate_set_failure",
    "ranking_failure",
    "grounding_failure",
)

_UNIT_INTERVAL_BINNING = {
    "edges": [0.0, 0.25, 0.5, 0.75, 1.0],
    "closed": "left except final bin",
    "labels": [
        "0.00_to_lt_0.25",
        "0.25_to_lt_0.50",
        "0.50_to_lt_0.75",
        "0.75_to_1.00",
    ],
}
BINNING = {
    "q_margin": {
        "edges": [0.0, 0.01, 0.05, 0.10, "infinity"],
        "closed": "right",
        "labels": [
            "0.00_to_0.01",
            "gt_0.01_to_0.05",
            "gt_0.05_to_0.10",
            "gt_0.10",
        ],
    },
    "predicted_mask_support": _UNIT_INTERVAL_BINNING,
    "predicted_mask_uncertainty": _UNIT_INTERVAL_BINNING,
    "depth_valid_fraction": _UNIT_INTERVAL_BINNING,
    "candidate_angle_confidence": _UNIT_INTERVAL_BINNING,
    "width_consistency": _UNIT_INTERVAL_BINNING,
}
FIELD_SOURCES = {
    "expression_family": "analysis metadata: expression_family/expression_type/template_type",
    "token_length": "analysis metadata: token_length/text_token_length",
    "query_complexity": "analysis metadata: query_complexity/complexity",
    "q_margin": "frozen q-only top-1 minus top-2 q_raw; explicit metadata q_margin fallback",
    "v2_switch": "q-only selected candidate ID compared with V2 selected candidate ID",
    "original_first_correct_rank": "first correct candidate in q-only order, separately by evaluator track",
    "predicted_mask_support": "V3-selected frozen candidate analysis evidence",
    "predicted_mask_uncertainty": "V3-selected explicit mask-uncertainty analysis evidence",
    "depth_valid_fraction": "sample-level analysis metadata",
    "object_size": "sample-level analysis metadata",
    "candidate_angle_confidence": "V3-selected frozen candidate analysis evidence",
    "width_consistency": "V3-selected frozen candidate analysis evidence",
    "corrected_legacy_disagreement": "any candidate-label disagreement between evaluator tracks",
    "candidate_set_failure": "no correct frozen candidate, separately by evaluator track",
    "ranking_failure": "q-only top candidate wrong while a correct candidate exists, separately by evaluator track",
    "grounding_failure": "explicit analysis metadata or evaluation-only mask_iou < 0.5",
}


def _record_sample_id(record: Mapping[str, Any], *, artifact_name: str) -> str:
    if "sample_id" not in record:
        raise ValueError(f"{artifact_name} record is missing sample_id")
    raw = str(record["sample_id"])
    if raw.startswith("multiple:"):
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"{artifact_name} has invalid stable sample ID: {raw}")
        try:
            canonical = stable_sample_id(parts[1], parts[2])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{artifact_name} has invalid stable sample ID: {raw}") from exc
        if raw != canonical:
            raise ValueError(
                f"{artifact_name} sample ID is not canonical: {raw!r} != {canonical!r}"
            )
        return raw
    if "split" not in record:
        raise ValueError(
            f"{artifact_name} sample ID must be stable or accompanied by official split"
        )
    return stable_sample_id(str(record["split"]), raw)


def _load_unique_records(path: str | Path, *, artifact_name: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(path):
        if not isinstance(raw, dict):
            raise TypeError(f"{artifact_name} JSONL rows must be objects")
        sample_id = _record_sample_id(raw, artifact_name=artifact_name)
        if sample_id in records:
            raise ValueError(f"duplicate {artifact_name} sample ID: {sample_id}")
        records[sample_id] = raw
    if not records:
        raise ValueError(f"{artifact_name} cohort is empty")
    return records


def _require_exact_cohort(
    observed: Mapping[str, Any], expected: set[str], *, artifact_name: str
) -> None:
    observed_ids = set(observed)
    missing = sorted(expected - observed_ids)
    extra = sorted(observed_ids - expected)
    if missing or extra:
        raise ValueError(
            f"{artifact_name} cohort differs from frozen candidates: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _frozen_candidate_identity(
    record: Mapping[str, Any], *, sample_id: str
) -> tuple[tuple[str, ...], dict[str, str], dict[str, Mapping[str, Any]]]:
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 5:
        raise ValueError(f"frozen candidate record must contain five candidates: {sample_id}")
    candidate_ids: list[str] = []
    checksums: dict[str, str] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            raise TypeError(f"candidate entries must be objects: {sample_id}")
        candidate_id = str(raw_candidate.get("candidate_id", ""))
        checksum = str(raw_candidate.get("candidate_checksum", ""))
        if not candidate_id or not checksum:
            raise ValueError(f"candidate ID/checksum must be non-empty: {sample_id}")
        if candidate_id in checksums:
            raise ValueError(f"duplicate candidate ID: {sample_id}/{candidate_id}")
        candidate_ids.append(candidate_id)
        checksums[candidate_id] = checksum
        by_id[candidate_id] = raw_candidate
    return tuple(candidate_ids), checksums, by_id


def _identity_items(record: Mapping[str, Any]) -> Sequence[Mapping[str, Any]] | None:
    for field in ("candidate_labels", "candidate_metadata"):
        value = record.get(field)
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"{field} must be a list")
            return value
    # A metadata artifact may repeat the frozen candidate records.  Ranking
    # records normally use candidate_order instead and are handled separately.
    value = record.get("candidates")
    if value is not None:
        if not isinstance(value, list):
            raise TypeError("candidates must be a list")
        return value
    return None


def _declared_checksum_map(
    record: Mapping[str, Any], *, order: Sequence[str] | None = None
) -> dict[str, str] | None:
    checksum_values = record.get("candidate_checksums")
    checksum_ids = record.get("candidate_checksum_ids")
    if checksum_values is None and checksum_ids is None:
        items = _identity_items(record)
        if items is None:
            return None
        declared: dict[str, str] = {}
        saw_checksum = False
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError("candidate identity entries must be objects")
            candidate_id = str(item.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError("candidate identity entry is missing candidate_id")
            checksum = item.get("candidate_checksum")
            if checksum is not None:
                saw_checksum = True
                checksum_value = str(checksum)
                if not checksum_value:
                    raise ValueError(f"candidate checksum is empty: {candidate_id}")
                declared[candidate_id] = checksum_value
        if not saw_checksum:
            return None
        if len(declared) != len(items):
            raise ValueError("candidate checksums must be declared for every candidate or none")
        return declared
    if checksum_values is None:
        raise ValueError("candidate_checksum_ids declared without candidate_checksums")
    if isinstance(checksum_values, Mapping):
        if checksum_ids is not None:
            raise ValueError("mapping candidate_checksums must not also declare candidate_checksum_ids")
        return {str(key): str(value) for key, value in checksum_values.items()}
    if not isinstance(checksum_values, list):
        raise TypeError("candidate_checksums must be a mapping or list")
    ids = checksum_ids if checksum_ids is not None else order
    if not isinstance(ids, (list, tuple)) or len(ids) != len(checksum_values):
        raise ValueError("candidate checksum IDs/values must be aligned")
    return dict(zip(map(str, ids), map(str, checksum_values), strict=True))


def _verify_identity(
    record: Mapping[str, Any],
    *,
    sample_id: str,
    frozen_ids: Sequence[str],
    frozen_checksums: Mapping[str, str],
    artifact_name: str,
    require_checksums: bool,
    order: Sequence[str] | None = None,
) -> bool:
    declared = _declared_checksum_map(record, order=order)
    if declared is None:
        if require_checksums:
            raise ValueError(f"{artifact_name} is missing candidate checksums: {sample_id}")
        return False
    if set(declared) != set(frozen_ids):
        raise ValueError(f"{artifact_name} candidate IDs differ: {sample_id}")
    for candidate_id in frozen_ids:
        if declared[candidate_id] != frozen_checksums[candidate_id]:
            raise ValueError(
                f"{artifact_name} candidate checksum differs: {sample_id}/{candidate_id}"
            )
    return True


def _ranking_order(
    record: Mapping[str, Any],
    *,
    method: str,
    sample_id: str,
    frozen_ids: Sequence[str],
) -> tuple[str, ...]:
    raw_order = record.get("candidate_order")
    if raw_order is None and method == "q_only":
        candidates = record.get("candidates")
        if isinstance(candidates, list) and len(candidates) == 5:
            if all(isinstance(item, Mapping) and item.get("q_rank") is not None for item in candidates):
                raw_order = [
                    item["candidate_id"]
                    for item in sorted(candidates, key=lambda value: int(value["q_rank"]))
                ]
            elif all(isinstance(item, Mapping) and item.get("q_raw") is not None for item in candidates):
                raw_order = [
                    item["candidate_id"]
                    for item in sorted(
                        candidates,
                        key=lambda value: (-float(value["q_raw"]), str(value["candidate_id"])),
                    )
                ]
    if not isinstance(raw_order, list):
        raise ValueError(f"{method} ranking is missing candidate_order: {sample_id}")
    order = tuple(map(str, raw_order))
    if len(order) != 5 or len(set(order)) != 5 or set(order) != set(frozen_ids):
        raise ValueError(f"{method} ranking changed the frozen candidate pool: {sample_id}")
    selection = record.get("selection")
    if isinstance(selection, Mapping) and selection.get("selected_candidate_id") is not None:
        if str(selection["selected_candidate_id"]) != order[0]:
            raise ValueError(f"{method} selection differs from ranking top: {sample_id}")
    return order


def _label_vector(
    record: Mapping[str, Any], *, sample_id: str, candidate_ids: Sequence[str]
) -> dict[str, bool]:
    values = record.get("candidate_labels")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(f"label artifact must contain five candidate labels: {sample_id}")
    result: dict[str, bool] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError(f"candidate labels must be objects: {sample_id}")
        candidate_id = str(value.get("candidate_id", ""))
        raw = value.get("candidate_correct")
        if candidate_id in result:
            raise ValueError(f"duplicate label candidate ID: {sample_id}/{candidate_id}")
        if isinstance(raw, (bool, np.bool_)):
            correct = bool(raw)
        elif isinstance(raw, (int, np.integer)) and int(raw) in (0, 1):
            correct = bool(raw)
        else:
            raise ValueError(
                f"candidate_correct must be boolean or integer 0/1: {sample_id}/{candidate_id}"
            )
        result[candidate_id] = correct
    if set(result) != set(candidate_ids):
        raise ValueError(f"label candidate IDs differ: {sample_id}")
    return result


def _mapping_value(value: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in value:
            return value[key]
    nested = value.get("metadata")
    if isinstance(nested, Mapping):
        for key in aliases:
            if key in nested:
                return nested[key]
    return None


def _unwrap_scalar(value: Any) -> Any:
    while isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _finite_number(value: Any, *, field: str) -> float | None:
    value = _unwrap_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or missing") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _category(value: Any) -> Any:
    value = _unwrap_scalar(value)
    if value is None or value == "":
        return UNKNOWN
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("subgroup category must be finite")
        return value
    raise ValueError(f"subgroup category must be scalar, received {type(value).__name__}")


def _unit_interval_bin(value: Any, *, field: str) -> str:
    value = _unwrap_scalar(value)
    if isinstance(value, (bool, np.bool_)):
        value = float(value)
    number = _finite_number(value, field=field)
    if number is None:
        return UNKNOWN
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0,1]")
    if number < 0.25:
        return "0.00_to_lt_0.25"
    if number < 0.50:
        return "0.25_to_lt_0.50"
    if number < 0.75:
        return "0.50_to_lt_0.75"
    return "0.75_to_1.00"


def _q_margin_bin(value: float | None) -> str:
    if value is None:
        return UNKNOWN
    if value < -1e-12:
        raise ValueError("q margin must be non-negative")
    value = max(0.0, value)
    if value <= 0.01:
        return "0.00_to_0.01"
    if value <= 0.05:
        return "gt_0.01_to_0.05"
    if value <= 0.10:
        return "gt_0.05_to_0.10"
    return "gt_0.10"


def _candidate_metadata_by_id(
    record: Mapping[str, Any], *, frozen_ids: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    values = record.get("candidate_metadata")
    if values is None:
        values = record.get("candidates")
    if values is None:
        return {}
    if not isinstance(values, list):
        raise TypeError("metadata candidate entries must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("metadata candidate entries must be objects")
        candidate_id = str(value.get("candidate_id", ""))
        if not candidate_id or candidate_id in result:
            raise ValueError("metadata candidate IDs must be non-empty and unique")
        result[candidate_id] = value
    if set(result) != set(frozen_ids):
        raise ValueError("metadata candidate IDs differ from frozen candidates")
    return result


def _aligned_candidate_value(
    record: Mapping[str, Any], key: str, *, candidate_ids: Sequence[str]
) -> dict[str, Any] | None:
    raw = record.get(key)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        values = {str(candidate_id): value for candidate_id, value in raw.items()}
    else:
        if not isinstance(raw, list) or len(raw) != len(candidate_ids):
            raise ValueError(f"{key} must contain five candidate-aligned values")
        raw_ids = record.get(f"{key}_ids", record.get("candidate_probability_ids", candidate_ids))
        if not isinstance(raw_ids, list) or len(raw_ids) != len(raw):
            raise ValueError(f"{key} candidate IDs/values must be aligned")
        values = dict(zip(map(str, raw_ids), raw, strict=True))
    if set(values) != set(candidate_ids):
        raise ValueError(f"{key} candidate IDs differ from frozen candidates")
    return values


def _grounding_failure(metadata: Mapping[str, Any]) -> str:
    explicit = _mapping_value(metadata, ("grounding_failure",))
    if explicit is not None:
        if isinstance(explicit, (bool, np.bool_)):
            return "failure" if bool(explicit) else "not_failure"
        if isinstance(explicit, (int, np.integer)) and int(explicit) in (0, 1):
            return "failure" if int(explicit) else "not_failure"
        if isinstance(explicit, str) and explicit in {"failure", "not_failure", UNKNOWN}:
            return explicit
        raise ValueError("grounding_failure must be boolean, 0/1, or an explicit category")
    mask_iou = _finite_number(_mapping_value(metadata, ("mask_iou",)), field="mask_iou")
    if mask_iou is None:
        return UNKNOWN
    if not 0.0 <= mask_iou <= 1.0:
        raise ValueError("mask_iou must be within [0,1]")
    return "failure" if mask_iou < 0.5 else "not_failure"


def _common_subgroups(
    *,
    sample_id: str,
    metadata: Mapping[str, Any],
    q_order: Sequence[str],
    v2_order: Sequence[str],
    v3_order: Sequence[str],
    v3_ranking: Mapping[str, Any],
    frozen_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata_candidates = _candidate_metadata_by_id(metadata, frozen_ids=tuple(frozen_candidates))
    # Ensure a complete candidate map for aligned V3 evidence even when the
    # metadata artifact is sample-level only.
    alignment_ids = tuple(frozen_candidates)
    evidence = _aligned_candidate_value(v3_ranking, "candidate_evidence", candidate_ids=alignment_ids)
    selected_id = str(v3_order[0])

    def selected_value(aliases: Sequence[str]) -> Any:
        sources: list[Mapping[str, Any]] = []
        if evidence is not None and isinstance(evidence[selected_id], Mapping):
            sources.append(evidence[selected_id])
        if selected_id in metadata_candidates:
            sources.append(metadata_candidates[selected_id])
        frozen = frozen_candidates[selected_id]
        features = frozen.get("features")
        sources.append(features if isinstance(features, Mapping) else frozen)
        sources.append(metadata)
        for source in sources:
            value = _mapping_value(source, aliases)
            if value is not None:
                return value
        return None

    q_values = {
        candidate_id: _finite_number(
            frozen_candidates[candidate_id].get("q_raw"), field=f"q_raw:{sample_id}/{candidate_id}"
        )
        for candidate_id in q_order[:2]
    }
    q_margin = None
    if all(value is not None for value in q_values.values()):
        q_margin = float(q_values[q_order[0]]) - float(q_values[q_order[1]])  # type: ignore[arg-type]
    if q_margin is None:
        q_margin = _finite_number(
            _mapping_value(metadata, ("q_margin",)), field="q_margin"
        )
    explicit_q_margin_bin = _mapping_value(metadata, ("q_margin_bin",))

    return {
        "expression_family": _category(
            _mapping_value(metadata, ("expression_family", "expression_type", "template_type"))
        ),
        "token_length": _category(
            _mapping_value(metadata, ("token_length", "text_token_length"))
        ),
        "query_complexity": _category(
            _mapping_value(metadata, ("query_complexity", "complexity"))
        ),
        "q_margin": (
            _category(explicit_q_margin_bin)
            if explicit_q_margin_bin is not None
            else _q_margin_bin(q_margin)
        ),
        "v2_switch": "switch" if q_order[0] != v2_order[0] else "no_switch",
        "predicted_mask_support": _unit_interval_bin(
            selected_value(("mask_support", "g2_mask_probability_mean", "mask_consistency")),
            field="predicted_mask_support",
        ),
        "predicted_mask_uncertainty": _unit_interval_bin(
            selected_value(
                (
                    "predicted_mask_uncertainty",
                    "mask_uncertainty",
                    "g2_mask_uncertainty",
                )
            ),
            field="predicted_mask_uncertainty",
        ),
        "depth_valid_fraction": _unit_interval_bin(
            _mapping_value(metadata, ("depth_valid_fraction", "depth_valid")),
            field="depth_valid_fraction",
        ),
        "object_size": _category(
            _mapping_value(
                metadata,
                ("object_size", "object_size_bin", "object_area_fraction", "mask_area_fraction"),
            )
        ),
        "candidate_angle_confidence": _unit_interval_bin(
            selected_value(
                ("angle_confidence", "g3_axial_concentration", "angle_consistency")
            ),
            field="candidate_angle_confidence",
        ),
        "width_consistency": _unit_interval_bin(
            selected_value(("width_consistency", "g4_width_consistency", "width_compatibility")),
            field="width_consistency",
        ),
        "grounding_failure": _grounding_failure(metadata),
    }


def _track_subgroups(
    common: Mapping[str, Any],
    *,
    q_order: Sequence[str],
    labels: Mapping[str, bool],
    corrected: Mapping[str, bool],
    legacy: Mapping[str, bool],
) -> dict[str, Any]:
    oracle = any(labels.values())
    first_correct = next(
        (rank for rank, candidate_id in enumerate(q_order, 1) if labels[candidate_id]),
        None,
    )
    return dict(common) | {
        "original_first_correct_rank": "none" if first_correct is None else first_correct,
        "corrected_legacy_disagreement": (
            "disagree"
            if any(corrected[candidate_id] != legacy[candidate_id] for candidate_id in q_order)
            else "agree"
        ),
        "candidate_set_failure": "failure" if not oracle else "not_failure",
        "ranking_failure": (
            "failure" if oracle and not labels[q_order[0]] else "not_failure"
        ),
    }


def _unknown_counts(
    metadata_by_track: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, int]]:
    return {
        track: {
            field: sum(value[field] == UNKNOWN for value in rows)
            for field in SUBGROUP_FIELDS
        }
        for track, rows in metadata_by_track.items()
    }


def build_subgroup_report(
    *,
    candidates_path: str | Path,
    metadata_path: str | Path,
    corrected_labels_path: str | Path,
    legacy_labels_path: str | Path,
    rankings: Mapping[str, str | Path],
    output_dir: str | Path,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Join frozen evaluation artifacts and publish subgroup metrics.

    The three ranking keys are deliberately fixed so a report cannot silently
    omit a baseline or substitute a validation-only method.
    """
    if tuple(rankings) != METHODS and set(rankings) != set(METHODS):
        raise ValueError(f"rankings must contain exactly {list(METHODS)}")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"immutable subgroup output already exists: {output}")

    candidates = _load_unique_records(candidates_path, artifact_name="candidate")
    metadata = _load_unique_records(metadata_path, artifact_name="analysis metadata")
    corrected_records = _load_unique_records(
        corrected_labels_path, artifact_name="corrected label"
    )
    legacy_records = _load_unique_records(legacy_labels_path, artifact_name="legacy label")
    ranking_records = {
        method: _load_unique_records(path, artifact_name=f"{method} ranking")
        for method, path in rankings.items()
    }
    cohort = set(candidates)
    _require_exact_cohort(metadata, cohort, artifact_name="analysis metadata")
    _require_exact_cohort(corrected_records, cohort, artifact_name="corrected label")
    _require_exact_cohort(legacy_records, cohort, artifact_name="legacy label")
    for method in METHODS:
        _require_exact_cohort(
            ranking_records[method], cohort, artifact_name=f"{method} ranking"
        )

    sample_ids = sorted(cohort)
    label_values: dict[str, dict[str, dict[str, bool]]] = {
        track: {} for track in TRACKS
    }
    orders: dict[str, dict[str, tuple[str, ...]]] = {method: {} for method in METHODS}
    common_metadata: dict[str, dict[str, Any]] = {}
    checksum_validation: dict[str, Any] = {
        "corrected_labels": True,
        "legacy_labels": True,
        "analysis_metadata": "not_declared",
        "rankings": {},
    }

    metadata_checksum_declarations: list[bool] = []
    ranking_checksum_declarations: dict[str, list[bool]] = {method: [] for method in METHODS}
    for sample_id in sample_ids:
        frozen_ids, frozen_checksums, frozen_by_id = _frozen_candidate_identity(
            candidates[sample_id], sample_id=sample_id
        )
        metadata_checksum_declarations.append(
            _verify_identity(
                metadata[sample_id],
                sample_id=sample_id,
                frozen_ids=frozen_ids,
                frozen_checksums=frozen_checksums,
                artifact_name="analysis metadata",
                require_checksums=False,
            )
        )
        for track, records, name in (
            ("corrected_scientific", corrected_records, "corrected label"),
            ("legacy_official_compatibility", legacy_records, "legacy label"),
        ):
            _verify_identity(
                records[sample_id],
                sample_id=sample_id,
                frozen_ids=frozen_ids,
                frozen_checksums=frozen_checksums,
                artifact_name=name,
                require_checksums=True,
            )
            label_values[track][sample_id] = _label_vector(
                records[sample_id], sample_id=sample_id, candidate_ids=frozen_ids
            )
        for method in METHODS:
            order = _ranking_order(
                ranking_records[method][sample_id],
                method=method,
                sample_id=sample_id,
                frozen_ids=frozen_ids,
            )
            orders[method][sample_id] = order
            ranking_checksum_declarations[method].append(
                _verify_identity(
                    ranking_records[method][sample_id],
                    sample_id=sample_id,
                    frozen_ids=frozen_ids,
                    frozen_checksums=frozen_checksums,
                    artifact_name=f"{method} ranking",
                    require_checksums=False,
                    order=order,
                )
            )

        common_metadata[sample_id] = _common_subgroups(
            sample_id=sample_id,
            metadata=metadata[sample_id],
            q_order=orders["q_only"][sample_id],
            v2_order=orders["v2_locked_primary"][sample_id],
            v3_order=orders["v3_locked_primary"][sample_id],
            v3_ranking=ranking_records["v3_locked_primary"][sample_id],
            frozen_candidates=frozen_by_id,
        )

    if any(metadata_checksum_declarations):
        if not all(metadata_checksum_declarations):
            raise ValueError("analysis metadata checksum declaration is inconsistent across cohort")
        checksum_validation["analysis_metadata"] = "verified"
    for method, declarations in ranking_checksum_declarations.items():
        if any(declarations) and not all(declarations):
            raise ValueError(f"{method} checksum declaration is inconsistent across cohort")
        checksum_validation["rankings"][method] = (
            "verified" if all(declarations) else "candidate_id_set_verified_checksum_not_declared"
        )

    metadata_by_track: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    corrected_by_id = label_values["corrected_scientific"]
    legacy_by_id = label_values["legacy_official_compatibility"]
    for track in TRACKS:
        for sample_id in sample_ids:
            metadata_by_track[track].append(
                _track_subgroups(
                    common_metadata[sample_id],
                    q_order=orders["q_only"][sample_id],
                    labels=label_values[track][sample_id],
                    corrected=corrected_by_id[sample_id],
                    legacy=legacy_by_id[sample_id],
                )
            )

    result_rows: list[dict[str, Any]] = []
    for track in TRACKS:
        metadata_columns = {
            field: [row[field] for row in metadata_by_track[track]]
            for field in SUBGROUP_FIELDS
        }
        q_correct = np.asarray(
            [
                label_values[track][sample_id][orders["q_only"][sample_id][0]]
                for sample_id in sample_ids
            ],
            dtype=bool,
        )
        for method in METHODS:
            method_correct = np.asarray(
                [
                    label_values[track][sample_id][orders[method][sample_id][0]]
                    for sample_id in sample_ids
                ],
                dtype=bool,
            )
            rows = subgroup_metrics(
                analysis_metadata=metadata_columns,
                reference_correct=q_correct,
                challenger_correct=method_correct,
                subgroup_fields=SUBGROUP_FIELDS,
                ci_method="wilson",
                confidence_level=float(confidence_level),
            )
            for row in rows:
                result_rows.append(
                    {
                        "track": track,
                        "method": method,
                        "reference_method": "q_only",
                        **row,
                        "method_correct": row["challenger_correct"],
                        "method_rate": row["challenger_rate"],
                        "delta_vs_q": row["delta"],
                        "recovered_vs_q": row["recovered"],
                        "harmful_vs_q": row["harmful"],
                    }
                )
    if not result_rows:
        raise AssertionError("subgroup analysis produced no metric rows")

    content = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v3_evaluation_only_subgroup_metrics",
        "status": "complete",
        "sample_count": len(sample_ids),
        "candidate_count": len(sample_ids) * 5,
        "tracks": list(TRACKS),
        "methods": list(METHODS),
        "reference_method": "q_only",
        "subgroup_fields": list(SUBGROUP_FIELDS),
        "field_sources": FIELD_SOURCES,
        "unknown_category": UNKNOWN,
        "unknown_counts": _unknown_counts(metadata_by_track),
        "binning": BINNING,
        "confidence_level": float(confidence_level),
        "ci_method": "wilson",
        "rows": result_rows,
    }
    content_sha256 = sha256_bytes(canonical_json(content).encode("utf-8"))
    payload = content | {
        "content_sha256": content_sha256,
        "content_hash_scope": "canonical JSON of this document excluding content_sha256 and content_hash_scope",
    }

    output.mkdir(parents=True, exist_ok=False)
    csv_path = write_csv(output / "subgroup_metrics.csv", result_rows)
    json_path = atomic_write_json(output / "subgroup_metrics.json", payload)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v3_evaluation_only_subgroup_provenance",
        "status": "complete",
        "analysis_boundary": (
            "evaluation-only join of frozen candidates, analysis metadata, labels, and "
            "precomputed rankings; no predictor or inference callback exists in this producer"
        ),
        "sample_join": "exact canonical stable sample ID cohort",
        "candidate_join": "exact candidate ID set plus checksum verification where declared",
        "field_sources": FIELD_SOURCES,
        "checksum_validation": checksum_validation,
        "sample_count": len(sample_ids),
        "candidate_count": len(sample_ids) * 5,
        "cohort_sample_ids_sha256": sha256_bytes(
            canonical_json(sample_ids).encode("utf-8")
        ),
        "subgroup_content_sha256": content_sha256,
        "inputs": {
            "candidates": artifact_identity(candidates_path),
            "analysis_metadata": artifact_identity(metadata_path),
            "corrected_labels": artifact_identity(corrected_labels_path),
            "legacy_labels": artifact_identity(legacy_labels_path),
            "rankings": {
                method: artifact_identity(rankings[method]) for method in METHODS
            },
        },
        "outputs": {
            "csv": artifact_identity(csv_path),
            "json": artifact_identity(json_path),
        },
    }
    provenance_path = atomic_write_json(output / "subgroup_provenance.json", provenance)
    return payload | {
        "artifacts": {
            "csv": artifact_identity(csv_path),
            "json": artifact_identity(json_path),
            "provenance": artifact_identity(provenance_path),
        }
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build evaluation-only CROG V3 subgroup metrics from frozen artifacts."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--corrected-labels", required=True)
    parser.add_argument("--legacy-labels", required=True)
    parser.add_argument("--q-ranking", required=True)
    parser.add_argument("--v2-ranking", required=True)
    parser.add_argument("--v3-ranking", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_subgroup_report(
        candidates_path=args.candidates,
        metadata_path=args.metadata,
        corrected_labels_path=args.corrected_labels,
        legacy_labels_path=args.legacy_labels,
        rankings={
            "q_only": args.q_ranking,
            "v2_locked_primary": args.v2_ranking,
            "v3_locked_primary": args.v3_ranking,
        },
        output_dir=args.output_dir,
        confidence_level=args.confidence_level,
    )
    print(json.dumps({"status": result["status"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests.
    raise SystemExit(main())


__all__ = [
    "BINNING",
    "FIELD_SOURCES",
    "METHODS",
    "SUBGROUP_FIELDS",
    "TRACKS",
    "UNKNOWN",
    "build_subgroup_report",
    "main",
]
