"""Fail-closed validation for publication/reporting inputs.

This module intentionally operates only on already-produced machine artifacts.
It never reads benchmark labels or inference features.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import artifact_identity


FORMAL_METHODS = (
    "q_only",
    "v2_locked_primary",
    "v3_full_head_scalar_gate",
    "v3_fcer_native",
    "v3_fcer_rgbd",
    "v3_locked_primary",
)
FORMAL_TRACKS = ("corrected_scientific", "legacy_official_compatibility")
REQUIRED_REPORT_EVIDENCE = (
    "independent_evaluation_completion",
    "diagnostic",
    "efficiency",
    "subgroup",
    "gallery",
)


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonempty_rows(value: Any, *, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a non-empty machine table")
    rows = list(value)
    if any(not isinstance(row, Mapping) or not row for row in rows):
        raise ValueError(f"{name} rows must be non-empty mappings")
    return rows


def _formal_rows(value: Any, *, name: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = _nonempty_rows(value, name=name)
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("track", "")), str(row.get("method", "")))
        if key in indexed:
            raise ValueError(f"{name} has duplicate formal row {key}")
        indexed[key] = row
        metric = _finite(row.get("j_at_1"), field=f"{name}{key}.j_at_1")
        if not 0.0 <= metric <= 1.0:
            raise ValueError(f"{name}{key}.j_at_1 must be in [0,1]")
        try:
            sample_count = int(row["sample_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}{key}.sample_count must be a positive integer") from exc
        if sample_count <= 0:
            raise ValueError(f"{name}{key}.sample_count must be a positive integer")
    expected = {(track, method) for track in FORMAL_TRACKS for method in FORMAL_METHODS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"{name} must contain exactly six formal methods on both tracks; missing={missing}, extra={extra}")
    for track in FORMAL_TRACKS:
        counts = {int(indexed[(track, method)]["sample_count"]) for method in FORMAL_METHODS}
        if len(counts) > 1:
            raise ValueError(f"{name} has contradictory/empty sample counts for {track}")
    return indexed


def _pairwise_rows(value: Any) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows = _nonempty_rows(value, name="pairwise_rows")
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("track", "")), str(row.get("method", "")), str(row.get("reference", "")))
        if key in indexed:
            raise ValueError(f"pairwise_rows has duplicate comparison {key}")
        indexed[key] = row
        for field in (
            "effect",
            "raw_p",
            "holm_adjusted_p",
            "frame_ci_lower",
            "frame_ci_upper",
            "scene_ci_lower",
            "scene_ci_upper",
        ):
            _finite(row.get(field), field=f"pairwise_rows{key}.{field}")
        for field in ("raw_p", "holm_adjusted_p"):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"pairwise_rows{key}.{field} must be in [0,1]")
        if float(row["frame_ci_lower"]) > float(row["frame_ci_upper"]):
            raise ValueError(f"pairwise_rows{key} has reversed frame CI")
        if float(row["scene_ci_lower"]) > float(row["scene_ci_upper"]):
            raise ValueError(f"pairwise_rows{key} has reversed scene CI")
        recovered, harmful = int(row.get("recovered", -1)), int(row.get("harmful", -1))
        if recovered < 0 or harmful < 0:
            raise ValueError(f"pairwise_rows{key} has invalid switch counts")
        sample_count = int(row.get("sample_count", 0))
        if sample_count <= 0 or recovered + harmful > sample_count:
            raise ValueError(f"pairwise_rows{key} has contradictory sample/switch counts")
        expected_effect = (recovered - harmful) / sample_count
        if not math.isclose(float(row["effect"]), expected_effect, abs_tol=1e-12):
            raise ValueError(f"pairwise_rows{key}.effect contradicts recovered/harmful counts")
    required = {
        (track, "v3_locked_primary", reference)
        for track in FORMAL_TRACKS
        for reference in ("q_only", "v2_locked_primary")
    }
    if not required.issubset(indexed):
        raise ValueError(f"pairwise_rows lacks primary comparisons: {sorted(required-set(indexed))}")
    return indexed


def _assert_close(actual: Any, expected: float, *, field: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(_finite(actual, field=field), expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{field} contradicts the machine tables")


def _best_ablation(rows: Sequence[Mapping[str, Any]], *, maximum: bool) -> str:
    candidates: list[tuple[float, str]] = []
    for index, row in enumerate(rows):
        name = row.get("feature_group", row.get("ablation", row.get("name")))
        value = row.get("delta_j_at_1", row.get("delta_vs_full", row.get("effect")))
        if name is None or value is None:
            continue
        candidates.append((_finite(value, field=f"feature_ablation_rows[{index}].effect"), str(name)))
    if not candidates:
        return "n/a"
    return (max if maximum else min)(candidates)[1]


def derive_conclusion_from_machine_tables(machine_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Derive and cross-check every inferable conclusion field."""
    for table in (
        "validation_rows",
        "calibration_rows",
        "feature_ablation_rows",
        "subgroup_rows",
    ):
        _nonempty_rows(machine_inputs.get(table), name=table)
    lockcheck = _formal_rows(machine_inputs.get("lockcheck_rows"), name="lockcheck_rows")
    test = _formal_rows(machine_inputs.get("test_rows"), name="test_rows")
    pairwise = _pairwise_rows(machine_inputs.get("pairwise_rows"))

    # The same primary/reference point estimates must be represented consistently
    # in the result and paired-statistics tables.
    for track in FORMAL_TRACKS:
        primary = test[(track, "v3_locked_primary")]
        q = test[(track, "q_only")]
        v2 = test[(track, "v2_locked_primary")]
        for reference, row in (("q_only", q), ("v2_locked_primary", v2)):
            comparison = pairwise[(track, "v3_locked_primary", reference)]
            delta = float(primary["j_at_1"]) - float(row["j_at_1"])
            _assert_close(comparison["effect"], delta, field=f"pairwise {track} primary-vs-{reference} effect")
            if int(comparison["sample_count"]) != int(primary["sample_count"]):
                raise ValueError(f"pairwise {track} primary-vs-{reference} sample_count contradicts results")
        if primary.get("delta_vs_q") is not None:
            _assert_close(primary["delta_vs_q"], float(primary["j_at_1"]) - float(q["j_at_1"]), field=f"{track} primary.delta_vs_q")
        if primary.get("delta_vs_v2") is not None:
            _assert_close(primary["delta_vs_v2"], float(primary["j_at_1"]) - float(v2["j_at_1"]), field=f"{track} primary.delta_vs_v2")

    corrected = test[(FORMAL_TRACKS[0], "v3_locked_primary")]
    legacy = test[(FORMAL_TRACKS[1], "v3_locked_primary")]
    corrected_v2 = pairwise[(FORMAL_TRACKS[0], "v3_locked_primary", "v2_locked_primary")]
    corrected_q = pairwise[(FORMAL_TRACKS[0], "v3_locked_primary", "q_only")]
    native = test[(FORMAL_TRACKS[0], "v3_fcer_native")]
    rgbd = test[(FORMAL_TRACKS[0], "v3_fcer_rgbd")]
    native_delta = float(rgbd["j_at_1"]) - float(native["j_at_1"])
    native_vs_depth = "RGB-D higher" if native_delta > 0 else "CROG-native higher" if native_delta < 0 else "tie"
    derived = {
        "v3_primary_method": "v3_locked_primary",
        "v3_primary_anchor": "v2_locked_primary",
        "corrected_delta_vs_q_pp": 100.0 * corrected_q["effect"],
        "corrected_delta_vs_v2_pp": 100.0 * corrected_v2["effect"],
        "legacy_delta_vs_q_pp": 100.0 * pairwise[(FORMAL_TRACKS[1], "v3_locked_primary", "q_only")]["effect"],
        "legacy_delta_vs_v2_pp": 100.0 * pairwise[(FORMAL_TRACKS[1], "v3_locked_primary", "v2_locked_primary")]["effect"],
        "recovered_vs_v2": int(corrected_v2["recovered"]),
        "harmful_vs_v2": int(corrected_v2["harmful"]),
        "headroom_recovered_vs_v2": _finite(
            corrected.get("headroom_recovered_vs_v2"),
            field="corrected primary.headroom_recovered_vs_v2",
        ),
        "frame_bootstrap_ci": [float(corrected_v2["frame_ci_lower"]), float(corrected_v2["frame_ci_upper"])],
        "scene_bootstrap_ci": [float(corrected_v2["scene_ci_lower"]), float(corrected_v2["scene_ci_upper"])],
        "mcnemar_holm_p": float(corrected_v2["holm_adjusted_p"]),
        "native_vs_depth_conclusion": native_vs_depth,
        "most_valuable_feature_group": _best_ablation(machine_inputs["feature_ablation_rows"], maximum=True),
        "most_harmful_feature_group": _best_ablation(machine_inputs["feature_ablation_rows"], maximum=False),
        "statistically_reliable_vs_q": bool(
            corrected_q["effect"] > 0
            and corrected_q["frame_ci_lower"] > 0
            and corrected_q["scene_ci_lower"] > 0
            and corrected_q["holm_adjusted_p"] < 0.05
        ),
    }

    supplied = machine_inputs.get("conclusion_payload")
    if not isinstance(supplied, Mapping):
        raise ValueError("conclusion_payload must be a mapping used only for cross-checkable metadata")
    for field, expected in derived.items():
        if field not in supplied:
            continue
        actual = supplied[field]
        if isinstance(expected, float):
            _assert_close(actual, expected, field=f"conclusion_payload.{field}")
        elif isinstance(expected, list):
            if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)) or len(actual) != len(expected):
                raise ValueError(f"conclusion_payload.{field} contradicts the machine tables")
            for index, value in enumerate(expected):
                _assert_close(actual[index], value, field=f"conclusion_payload.{field}[{index}]")
        elif actual != expected:
            raise ValueError(f"conclusion_payload.{field} contradicts the machine tables")
    return dict(supplied) | derived


def verified_artifact(value: str | Path | Mapping[str, Any], *, name: str) -> dict[str, Any]:
    expected = value if isinstance(value, Mapping) else None
    path_value = expected.get("path") if expected is not None else value
    if not path_value:
        raise ValueError(f"{name} artifact has no path")
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{name} artifact is missing or empty: {path}")
    observed = artifact_identity(path)
    if expected is not None:
        for field in ("sha256", "size_bytes"):
            if field not in expected or expected[field] != observed[field]:
                raise ValueError(f"{name} artifact identity mismatch for {field}")
    return observed


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a readable JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def verify_independent_evaluation_completion(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    identity = verified_artifact(value, name="independent_evaluation_completion")
    payload = _json_object(identity["path"], name="independent_evaluation_completion")
    if payload.get("status") != "complete" or payload.get("stage") != "independent_evaluate":
        raise ValueError("independent evaluation completion is not a completed independent_evaluate stage")
    if payload.get("kind") != "independent_evaluate_run_complete":
        raise ValueError("independent evaluation completion kind mismatch")
    results = payload.get("result_artifacts")
    if not isinstance(results, list) or not results:
        raise ValueError("independent evaluation completion has no result artifacts")
    for index, result in enumerate(results):
        verified_artifact(result, name=f"independent_evaluation_completion.result_artifacts[{index}]")
    return identity


def verify_report_evidence(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence_artifacts must be a mapping")
    missing = set(REQUIRED_REPORT_EVIDENCE) - set(evidence)
    if missing:
        raise ValueError(f"missing required report evidence artifacts: {sorted(missing)}")
    identities = {
        name: (verify_independent_evaluation_completion(value) if name == "independent_evaluation_completion" else verified_artifact(value, name=name))
        for name, value in evidence.items()
        if name in REQUIRED_REPORT_EVIDENCE
    }
    gallery = _json_object(identities["gallery"]["path"], name="gallery")
    if gallery.get("status") != "complete" or gallery.get("kind") != "v3_failure_galleries":
        raise ValueError("gallery artifact is not a completed V3 gallery manifest")
    bound = gallery.get("independent_evaluation_completion")
    if not isinstance(bound, Mapping) or any(bound.get(key) != identities["independent_evaluation_completion"][key] for key in ("path", "sha256", "size_bytes")):
        raise ValueError("gallery is not bound to the supplied independent evaluation completion")
    inputs = gallery.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("gallery manifest has no input identities")
    for name, identity in inputs.items():
        verified_artifact(identity, name=f"gallery.inputs.{name}")
    return identities


__all__ = (
    "FORMAL_METHODS",
    "FORMAL_TRACKS",
    "REQUIRED_REPORT_EVIDENCE",
    "derive_conclusion_from_machine_tables",
    "verified_artifact",
    "verify_independent_evaluation_completion",
    "verify_report_evidence",
)
