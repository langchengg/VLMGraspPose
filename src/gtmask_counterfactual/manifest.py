"""Deterministic P1 sample-manifest construction without GT pixel access."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .io import canonical_sha256


EXPECTED_SAMPLE_COUNT = 7_675
_SHA256 = re.compile(r"[0-9a-f]{64}")
ROUTES = ("g1", "c1", "d1")


class ManifestClosureError(RuntimeError):
    """The sample denominator or its identity closure is malformed."""


@dataclass(frozen=True)
class ManifestBuildResult:
    rows: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]
    audit: dict[str, Any]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalized_prompt(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def _absolute(value: Any) -> bool:
    text = _text(value)
    return bool(text) and Path(text).expanduser().is_absolute()


def _digest(value: Any) -> bool:
    return _SHA256.fullmatch(_text(value).lower()) is not None


def _integral(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return False
    return integer > 0 and (not isinstance(value, float) or value.is_integer())


def _identity_differences(
    sample: Mapping[str, Any], match: Mapping[str, Any]
) -> list[str]:
    differences: list[str] = []
    for name in ("scene_id", "frame_id", "query_id", "target_instance_id"):
        if _text(match.get(name)) and _text(match.get(name)) != _text(sample.get(name)):
            differences.append(name)
    for name in ("rgb_sha256", "depth_sha256"):
        if _text(match.get(name)) and _text(match.get(name)).lower() != _text(
            sample.get(name)
        ).lower():
            differences.append(name)
    if _text(match.get("language_prompt")) and _normalized_prompt(
        match.get("language_prompt")
    ) != _normalized_prompt(sample.get("language_prompt")):
        differences.append("language_prompt")
    return differences


def _key(row: Mapping[str, Any], names: Sequence[str]) -> tuple[str, ...] | None:
    values = tuple(_text(row.get(name)) for name in names)
    return values if all(values) else None


def _fingerprint(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    values = (
        _text(row.get("rgb_sha256")).lower(),
        _text(row.get("depth_sha256")).lower(),
        _normalized_prompt(row.get("language_prompt")),
        _text(row.get("target_instance_id")),
    )
    return values if all(values) else None


def _indices(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[tuple[str, ...], list[int]]]:
    definitions = {
        "sample_id": ("sample_id",),
        "composite": ("scene_id", "frame_id", "query_id", "target_instance_id"),
    }
    result: dict[str, dict[tuple[str, ...], list[int]]] = {}
    for name, columns in definitions.items():
        index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            value = _key(row, columns)
            if value is not None:
                index[value].append(position)
        result[name] = dict(index)
    fingerprint: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        value = _fingerprint(row)
        if value is not None:
            fingerprint[value].append(position)
    result["asset_prompt_annotation"] = dict(fingerprint)
    return result


def _route_match(
    sample: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    indices: Mapping[str, Mapping[tuple[str, ...], list[int]]],
    used: set[int],
) -> tuple[Mapping[str, Any] | None, str, str]:
    attempts = (
        ("sample_id", _key(sample, ("sample_id",))),
        (
            "composite",
            _key(sample, ("scene_id", "frame_id", "query_id", "target_instance_id")),
        ),
        ("asset_prompt_annotation", _fingerprint(sample)),
    )
    ambiguous: list[str] = []
    for method, value in attempts:
        if value is None:
            continue
        matches = indices[method].get(value, [])
        if len(matches) > 1:
            ambiguous.append(method)
            continue
        if len(matches) == 1:
            position = matches[0]
            if position in used:
                return None, "unresolved", f"{method} route row would be reused"
            used.add(position)
            return rows[position], method, ""
    if ambiguous:
        return None, "unresolved", "ambiguous route identity: " + ",".join(ambiguous)
    return None, "unresolved", "no exact route identity match"


def _base_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    required_text = (
        "sample_id",
        "scene_id",
        "frame_id",
        "query_id",
        "query_type",
        "language_prompt",
    )
    for name in required_text:
        if not _text(row.get(name)):
            reasons.append(f"missing {name}")
    if not _integral(row.get("target_instance_id")):
        reasons.append("invalid target_instance_id")
    if not _integral(row.get("gt_grasp_target_instance_id")):
        reasons.append("invalid gt_grasp_target_instance_id")
    elif _text(row.get("target_instance_id")) != _text(
        row.get("gt_grasp_target_instance_id")
    ):
        reasons.append("GT mask/grasp target identity differs")
    for prefix in (
        "rgb",
        "depth",
        "intrinsics",
        "prepared_gt_mask",
        "source_instance_mask",
        "gt_grasp_set",
    ):
        if not _absolute(row.get(f"{prefix}_path")):
            reasons.append(f"invalid absolute {prefix}_path")
        if not _digest(row.get(f"{prefix}_sha256")):
            reasons.append(f"invalid {prefix}_sha256")
    dimensions: dict[str, int] = {}
    for name in ("rgb_height", "rgb_width", "depth_height", "depth_width"):
        try:
            value = int(row.get(name))
        except (TypeError, ValueError):
            value = 0
        dimensions[name] = value
        if value <= 0:
            reasons.append(f"invalid {name}")
    if (
        dimensions["rgb_height"],
        dimensions["rgb_width"],
    ) != (
        dimensions["depth_height"],
        dimensions["depth_width"],
    ):
        reasons.append("RGB/depth declared shapes differ")
    return reasons


def build_counterfactual_manifest(
    denominator_rows: Iterable[Mapping[str, Any]],
    route_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> ManifestBuildResult:
    """Build an exact denominator using identity joins only, never row positions."""

    denominator = [dict(row) for row in denominator_rows]
    if len(denominator) != expected_count:
        raise ManifestClosureError(
            f"denominator count {len(denominator)} differs from {expected_count}"
        )
    sample_ids = [_text(row.get("sample_id")) for row in denominator]
    if any(not sample_id for sample_id in sample_ids):
        raise ManifestClosureError("denominator contains an empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ManifestClosureError("denominator sample_id values are not unique")
    if set(route_rows) != set(ROUTES):
        raise ManifestClosureError("route sources must be exactly g1, c1, and d1")

    sources = {route: [dict(row) for row in route_rows[route]] for route in ROUTES}
    indices = {route: _indices(rows) for route, rows in sources.items()}
    used = {route: set() for route in ROUTES}
    join_counts = {route: defaultdict(int) for route in ROUTES}
    output: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for source_row in sorted(denominator, key=lambda row: _text(row.get("sample_id"))):
        sample_id = _text(source_row["sample_id"])
        row = dict(source_row)
        reasons = _base_reasons(row)
        for route in ROUTES:
            match, method, reason = _route_match(
                row, sources[route], indices[route], used[route]
            )
            join_counts[route][method] += 1
            row[f"{route}_join_method"] = method
            if match is None:
                row[f"{route}_source_identity"] = ""
                reasons.append(f"{route}: {reason}")
                continue
            identity = _text(match.get("source_identity"))
            if not identity:
                reasons.append(f"{route}: missing source_identity")
            differences = _identity_differences(row, match)
            if differences:
                reasons.append(
                    f"{route}: source identity fields differ: {','.join(differences)}"
                )
            row[f"{route}_source_identity"] = identity
        row["counterfactual_evaluable"] = not reasons
        row["counterfactual_status"] = (
            "EVALUABLE"
            if not reasons
            else "technical_or_annotation_mapping_failure"
        )
        row["mapping_reason"] = "; ".join(sorted(set(reasons)))
        row["denominator_member"] = True
        output.append(row)
        if reasons:
            unresolved.append(
                {
                    "sample_id": sample_id,
                    "counterfactual_status": row["counterfactual_status"],
                    "mapping_reason": row["mapping_reason"],
                }
            )

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "P1_BASELINE_REPLAY_PASS",
        "join_policy": [
            "sample_id",
            "scene_id/frame_id/query_id/target_instance_id",
            "rgb/depth hash + normalized prompt + annotation identity",
        ],
        "row_number_join_used": False,
        "gt_pixels_read": False,
        "gt_grasp_rows_read": False,
        "sample_count": len(output),
        "counterfactual_evaluable_count": len(output) - len(unresolved),
        "unresolved_count": len(unresolved),
        "partition_total": len(output),
        "denominator_sample_ids_sha256": canonical_sha256(sorted(sample_ids)),
        "unresolved_sample_ids_sha256": canonical_sha256(
            sorted(row["sample_id"] for row in unresolved)
        ),
        "route_join_counts": {
            route: dict(sorted(counts.items()))
            for route, counts in join_counts.items()
        },
        "unused_route_rows": {
            route: len(sources[route]) - len(used[route]) for route in ROUTES
        },
    }
    if audit["counterfactual_evaluable_count"] + audit["unresolved_count"] != expected_count:
        raise AssertionError("manifest partition does not preserve the denominator")
    return ManifestBuildResult(output, unresolved, audit)


__all__ = [
    "EXPECTED_SAMPLE_COUNT",
    "ManifestBuildResult",
    "ManifestClosureError",
    "ROUTES",
    "build_counterfactual_manifest",
]
