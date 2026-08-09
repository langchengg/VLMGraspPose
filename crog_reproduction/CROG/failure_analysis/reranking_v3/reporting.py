"""Immutable, machine-readable result packaging for CROG reranking V3.

The functions in this module are deliberately post-hoc.  They accept metrics
that have already been computed by the evaluation stage and never read labels
or prediction features themselves.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .schema import artifact_identity, atomic_write_json, atomic_write_text


CONCLUSION_DEFAULTS: dict[str, Any] = {
    "v2_actual_feature_usage_complete": False,
    "v2_missing_feature_groups": [],
    "v3_primary_method": "",
    "v3_primary_anchor": "v2_locked_primary",
    "corrected_delta_vs_q_pp": 0.0,
    "corrected_delta_vs_v2_pp": 0.0,
    "legacy_delta_vs_q_pp": 0.0,
    "legacy_delta_vs_v2_pp": 0.0,
    "recovered_vs_v2": 0,
    "harmful_vs_v2": 0,
    "net_vs_v2": 0,
    "headroom_recovered_vs_v2": 0.0,
    "frame_bootstrap_ci": [0.0, 0.0],
    "scene_bootstrap_ci": [0.0, 0.0],
    "mcnemar_holm_p": 1.0,
    "native_vs_depth_conclusion": "",
    "most_valuable_feature_group": "",
    "most_harmful_feature_group": "",
    "statistically_reliable_vs_q": False,
    "statistically_reliable_vs_v2": False,
    "practically_material_vs_v2": False,
    "final_claim": "",
}

CLAIM_RELIABLE_V2 = (
    "Full-chain V3 provides statistically reliable evidence of improvement "
    "over the locked V2 reranker under the corrected benchmark evaluator."
)
CLAIM_Q_ONLY = (
    "V3 improves over q-only, but the experiment does not establish a "
    "statistically reliable improvement over V2."
)
CLAIM_NOT_SIGNIFICANT = (
    "The observed difference is not statistically reliable under clustered evaluation."
)
CLAIM_NEGATIVE = (
    "Comprehensive full-chain features did not improve the locked V2 ranking "
    "and introduced additional harmful switches."
)


def _finite_number(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _interval(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly [lower, upper]")
    result = [
        _finite_number(value[0], name=f"{name}[0]"),
        _finite_number(value[1], name=f"{name}[1]"),
    ]
    if result[0] > result[1]:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return result


def _machine_value(value: Any) -> Any:
    """Convert common scientific Python values to strict JSON values."""
    if isinstance(value, np.ndarray):
        return _machine_value(value.tolist())
    if isinstance(value, np.generic):
        return _machine_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _machine_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_machine_value(child) for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("machine-readable results cannot contain NaN or infinity")
        return value
    raise TypeError(f"unsupported machine-readable value: {type(value).__name__}")


def derive_conclusion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared four-way scientific conclusion rule.

    Reliability versus V2 requires all four declared conditions.  Reliability
    versus q-only is never inferred from a point estimate; it must be supplied
    by the paired statistical evaluation.
    """
    result = dict(CONCLUSION_DEFAULTS)
    result.update(_machine_value(payload))
    delta_v2 = _finite_number(result["corrected_delta_vs_v2_pp"], name="corrected_delta_vs_v2_pp")
    delta_q = _finite_number(result["corrected_delta_vs_q_pp"], name="corrected_delta_vs_q_pp")
    frame = _interval(result["frame_bootstrap_ci"], name="frame_bootstrap_ci")
    scene = _interval(result["scene_bootstrap_ci"], name="scene_bootstrap_ci")
    adjusted = _finite_number(result["mcnemar_holm_p"], name="mcnemar_holm_p")
    if not 0.0 <= adjusted <= 1.0:
        raise ValueError("mcnemar_holm_p must be in [0,1]")
    recovered = int(result["recovered_vs_v2"])
    harmful = int(result["harmful_vs_v2"])
    if recovered < 0 or harmful < 0:
        raise ValueError("recovered/harmful counts must be non-negative")

    reliable_v2 = delta_v2 > 0.0 and frame[0] > 0.0 and scene[0] > 0.0 and adjusted < 0.05
    reliable_q = bool(result.get("statistically_reliable_vs_q", False))
    materiality = _finite_number(result.get("materiality_threshold_pp", 0.1), name="materiality_threshold_pp")
    if materiality < 0:
        raise ValueError("materiality_threshold_pp must be non-negative")

    if reliable_v2:
        category, claim = "A_reliable_beyond_v2", CLAIM_RELIABLE_V2
    elif delta_v2 < 0.0:
        category, claim = "D_negative", CLAIM_NEGATIVE
    elif delta_q > 0.0:
        category, claim = "B_q_only", CLAIM_Q_ONLY
    else:
        category, claim = "C_not_significant", CLAIM_NOT_SIGNIFICANT

    result.update(
        {
            "corrected_delta_vs_v2_pp": delta_v2,
            "corrected_delta_vs_q_pp": delta_q,
            "frame_bootstrap_ci": frame,
            "scene_bootstrap_ci": scene,
            "mcnemar_holm_p": adjusted,
            "recovered_vs_v2": recovered,
            "harmful_vs_v2": harmful,
            "net_vs_v2": recovered - harmful,
            "statistically_reliable_vs_q": reliable_q,
            "statistically_reliable_vs_v2": reliable_v2,
            "practically_material_vs_v2": bool(reliable_v2 and delta_v2 >= materiality),
            "conclusion_category": category,
            "final_claim": claim,
            "rule_evidence": {
                "positive_corrected_delta": delta_v2 > 0.0,
                "positive_frame_ci_lower": frame[0] > 0.0,
                "positive_scene_ci_lower": scene[0] > 0.0,
                "holm_below_0_05": adjusted < 0.05,
            },
        }
    )
    return _machine_value(result)


def write_conclusion(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = derive_conclusion(payload)
    atomic_write_json(path, value)
    return value


def _csv_cell(value: Any) -> Any:
    normalized = _machine_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return normalized


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a heterogeneous table atomically without silently dropping fields."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    fields: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise TypeError("CSV rows must be mappings")
        row = {str(key): _csv_cell(value) for key, value in raw_row.items()}
        for field in row:
            if field not in fields:
                fields.append(field)
        normalized_rows.append(row)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{hashlib.sha256(str(output).encode()).hexdigest()[:8]}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
                writer.writeheader()
                writer.writerows(normalized_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


MACHINE_RESULT_FILES = (
    "results_validation.csv",
    "results_lockcheck.csv",
    "results_test.csv",
    "pairwise_statistics.csv",
    "calibration.csv",
    "feature_ablation.csv",
    "subgroup_metrics.csv",
    "feature_provenance.json",
    "conclusion.json",
    "commands.log",
    "environment.json",
    "tests.json",
)


def build_machine_results(
    output_dir: str | Path,
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    lockcheck_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    feature_ablation_rows: Sequence[Mapping[str, Any]],
    subgroup_rows: Sequence[Mapping[str, Any]],
    feature_provenance: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    conclusion_payload: Mapping[str, Any],
    commands: Sequence[str],
    environment: Mapping[str, Any],
    tests: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact machine-readable result set required by the protocol.

    Empty tables are valid and produce empty CSV files.  Existing target files
    are rejected before any output is written, which keeps reruns immutable.
    """
    output = Path(output_dir)
    targets = [output / name for name in MACHINE_RESULT_FILES]
    manifest_path = output / "machine_results_manifest.json"
    existing = [str(path) for path in [*targets, manifest_path] if path.exists()]
    if existing:
        raise FileExistsError(f"immutable report artifacts already exist: {existing[:3]}")
    if any(not isinstance(command, str) or not command.strip() for command in commands):
        raise ValueError("commands must be non-empty strings")

    # Validate all strict JSON payloads before the first write.
    provenance_value = _machine_value(feature_provenance)
    environment_value = _machine_value(environment)
    tests_value = _machine_value(tests)
    conclusion = derive_conclusion(conclusion_payload)
    tables = {
        "results_validation.csv": validation_rows,
        "results_lockcheck.csv": lockcheck_rows,
        "results_test.csv": test_rows,
        "pairwise_statistics.csv": pairwise_rows,
        "calibration.csv": calibration_rows,
        "feature_ablation.csv": feature_ablation_rows,
        "subgroup_metrics.csv": subgroup_rows,
    }
    # Exercise normalization ahead of writes so NaN/unsupported values cannot
    # leave a half-created result set.
    for rows in tables.values():
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("machine result table rows must be mappings")
            _machine_value(row)

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / name, rows)
    atomic_write_json(output / "feature_provenance.json", provenance_value)
    atomic_write_json(output / "conclusion.json", conclusion)
    atomic_write_text(output / "commands.log", "".join(f"{command.rstrip()}\n" for command in commands))
    atomic_write_json(output / "environment.json", environment_value)
    atomic_write_json(output / "tests.json", tests_value)

    artifacts = {path.name: artifact_identity(path) for path in targets}
    manifest = {
        "schema_version": "3.0.0",
        "kind": "v3_machine_readable_results",
        "status": "complete",
        "required_files": list(MACHINE_RESULT_FILES),
        "artifacts": artifacts,
        "conclusion_category": conclusion["conclusion_category"],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest | {"manifest": artifact_identity(manifest_path), "conclusion": conclusion}


def render_results_markdown(
    *,
    audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    lockcheck: Mapping[str, Any],
    formal: Mapping[str, Any],
    conclusion: Mapping[str, Any],
) -> str:
    """Render the required 19-section report from already-computed summaries."""
    disclosure = (
        "V3 was designed after aggregate exposure to previous V1/V2 test results. "
        "All V3 feature selection, architecture selection, thresholds and hyperparameters "
        "were nevertheless fixed using development, calibration and validation partitions "
        "before one immutable V3 formal-test execution."
    )
    selected = selection.get("selected", {}) if isinstance(selection.get("selected", {}), Mapping) else {}
    sections = [
        ("V2 audit conclusion", audit.get("conclusion", "See feature_provenance.json.")),
        ("Development and selection split", selection.get("split", selection.get("split_manifest", {}))),
        ("v3_lockcheck", lockcheck),
        ("Validation full table", selection.get("results", [])),
        ("Formal test full table", formal.get("results", formal)),
        ("Corrected primary results", formal.get("corrected_scientific", {})),
        ("Legacy compatibility results", formal.get("legacy_official_compatibility", {})),
        ("V3 vs q-only", formal.get("vs_q_only", {})),
        ("V3 vs V2", formal.get("vs_v2_locked_primary", {})),
        ("Native vs RGB-D", formal.get("native_vs_rgbd", {})),
        ("Recovered and harmful", {key: conclusion.get(key) for key in ("recovered_vs_v2", "harmful_vs_v2", "net_vs_v2")}),
        ("Confidence intervals and p-values", {key: conclusion.get(key) for key in ("frame_bootstrap_ci", "scene_bootstrap_ci", "mcnemar_holm_p")}),
        ("Calibration", formal.get("calibration", {})),
        ("Efficiency", formal.get("efficiency", {})),
        ("Ablations", selection.get("diagnostic_ablations", [])),
        ("Subgroups", formal.get("subgroups", [])),
        ("Galleries", formal.get("galleries", {})),
        ("Limitations", formal.get("limitations", [])),
        ("Final scientific conclusion", conclusion.get("final_claim", "")),
    ]
    body = ["# CROG Re-ranking V3 Results", "", "## Test-exposure disclosure", "", disclosure, ""]
    body.extend(["## Selected primary method", "", str(selected.get("configuration", conclusion.get("v3_primary_method", ""))), ""])
    for title, value in sections:
        body.extend([f"## {title}", ""])
        if isinstance(value, str):
            body.extend([value, ""])
        else:
            body.extend(["```json", json.dumps(_machine_value(value), indent=2, sort_keys=True, ensure_ascii=False), "```", ""])
    body.extend(
        [
            "The output remains a frozen 4-DoF planar grasp rectangle. V3 generates no new candidate; "
            "Oracle@5 is the fixed candidate-pool ceiling, and rectangle correctness is not real-robot "
            "success, force closure, or a collision-free guarantee. CROG-native inputs are RGB and text; "
            "depth is reported only as an additional RGB-D enhancement.",
            "",
        ]
    )
    return "\n".join(body)


def build_report_artifacts(
    output_dir: str | Path,
    *,
    audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    lockcheck: Mapping[str, Any],
    formal: Mapping[str, Any],
    machine_inputs: Mapping[str, Any],
    evidence_artifacts: Mapping[str, Any],
    figure_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report only from complete, mutually consistent machine evidence.

    The strict backend performs no evaluation or label IO.  It verifies the six
    frozen methods on both evaluator tracks, derives its conclusion from the
    result/statistics tables, and binds every upstream evidence artifact before
    creating the output directory.
    """
    from .report_validation import (
        derive_conclusion_from_machine_tables,
        verify_report_evidence,
    )

    output = Path(output_dir)
    # Complete validation happens before the first write, so malformed or
    # contradictory caller summaries cannot leave a report-shaped directory.
    strict_inputs = dict(machine_inputs)
    strict_inputs["conclusion_payload"] = derive_conclusion_from_machine_tables(strict_inputs)
    evidence = verify_report_evidence(evidence_artifacts)
    machine = build_machine_results(output, **strict_inputs)
    conclusion = machine["conclusion"]
    results_path = output / "RESULTS.md"
    atomic_write_text(
        results_path,
        render_results_markdown(
            audit=audit,
            selection=selection,
            lockcheck=lockcheck,
            formal=formal,
            conclusion=conclusion,
        ),
    )
    figures: dict[str, Any] = {}
    if figure_inputs is not None:
        from .plotting import build_statistical_figures

        figures = build_statistical_figures(output / "figures", **dict(figure_inputs))
    report_manifest = {
        "schema_version": "3.0.0",
        "kind": "v3_results_report",
        "status": "complete",
        "machine_manifest": machine["manifest"],
        "results_markdown": artifact_identity(results_path),
        "figures": figures,
        "evidence_artifacts": evidence,
        "conclusion_source": "derived_and_cross_checked_from_machine_tables",
        "labels_read": False,
    }
    atomic_write_json(output / "report_manifest.json", report_manifest)
    return report_manifest


__all__ = (
    "CLAIM_NEGATIVE",
    "CLAIM_NOT_SIGNIFICANT",
    "CLAIM_Q_ONLY",
    "CLAIM_RELIABLE_V2",
    "CONCLUSION_DEFAULTS",
    "MACHINE_RESULT_FILES",
    "build_machine_results",
    "build_report_artifacts",
    "derive_conclusion",
    "render_results_markdown",
    "write_conclusion",
    "write_csv",
)
