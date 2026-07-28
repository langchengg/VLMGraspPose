from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import JoinedSample, load_joined
from .metrics import evaluate_rankings, holm_adjust, risk_coverage_curve
from .schema import append_jsonl_record, atomic_write_json, read_jsonl


def q_only_rankings(samples: list[JoinedSample]) -> dict[str, list[str]]:
    return {
        sample.sample_id: [
            str(candidate["candidate_id"])
            for candidate in sample.feature["candidates"]
        ]
        for sample in samples
    }


def load_prediction_rankings(
    path: str | Path | None,
    samples: list[JoinedSample],
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """Load predictions without exposing label or GT fields to inference."""
    if path is None:
        return q_only_rankings(samples), {
            sample.sample_id: np.asarray(
                [candidate["q_raw"] for candidate in sample.feature["candidates"]],
                dtype=np.float64,
            )
            for sample in samples
        }
    records = {str(record["sample_id"]): record for record in read_jsonl(path)}
    expected = {sample.sample_id for sample in samples}
    if set(records) != expected:
        missing = sorted(expected - set(records))[:5]
        extra = sorted(set(records) - expected)[:5]
        raise ValueError(
            f"prediction cohort mismatch: missing={missing}, extra={extra}"
        )
    rankings: dict[str, list[str]] = {}
    probabilities: dict[str, np.ndarray] = {}
    have_probabilities = True
    for sample in samples:
        record = records[sample.sample_id]
        forbidden = (
            "gt",
            "ground_truth",
            "candidate_correct",
            "oracle",
            "iou",
            "angle_error",
        )
        if any(
            token in str(key).lower()
            for key in record
            if str(key) != "candidate_correctness_probabilities"
            for token in forbidden
        ):
            raise ValueError(
                f"prediction record contains evaluation-only field: {sample.sample_id}"
            )
        rankings[sample.sample_id] = list(
            map(str, record["candidate_order"])
        )
        raw = record.get(
            "candidate_correctness_probabilities",
            record.get("mean_setrank_probabilities"),
        )
        if raw is None:
            have_probabilities = False
        else:
            values = np.asarray(raw, dtype=np.float64)
            if (
                values.shape != (5,)
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
                or np.any(values > 1.0)
            ):
                raise ValueError(
                    f"invalid candidate probabilities for {sample.sample_id}"
                )
            probabilities[sample.sample_id] = values
    return rankings, probabilities if have_probabilities else None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                append_jsonl_record(handle, row)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_method(
    *,
    method: str,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    corrected_labels_path: str | Path,
    prediction_path: str | Path | None,
    output_dir: str | Path,
    expected_legacy_oracle: float | None = None,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 7301,
) -> dict[str, Any]:
    """Evaluate one frozen ranking under isolated legacy/corrected tracks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    legacy = load_joined(features_path, legacy_labels_path)
    corrected = load_joined(features_path, corrected_labels_path)
    if [sample.sample_id for sample in legacy] != [
        sample.sample_id for sample in corrected
    ]:
        raise ValueError("legacy and corrected cohorts differ")
    rankings, probabilities = load_prediction_rankings(
        prediction_path, legacy
    )
    legacy_summary, legacy_rows = evaluate_rankings(
        legacy,
        rankings,
        candidate_probabilities=probabilities,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    corrected_summary, corrected_rows = evaluate_rankings(
        corrected,
        rankings,
        candidate_probabilities=probabilities,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    if probabilities is not None:
        selected_probability_confidence = {}
        for sample in legacy:
            original_ids = [
                str(candidate["candidate_id"])
                for candidate in sample.feature["candidates"]
            ]
            selected_id = rankings[sample.sample_id][0]
            selected_index = original_ids.index(selected_id)
            selected_probability_confidence[sample.sample_id] = float(
                probabilities[sample.sample_id][selected_index]
            )
        legacy_summary["risk_coverage"] = risk_coverage_curve(
            legacy_rows, selected_probability_confidence
        )
        legacy_summary["risk_coverage_confidence"] = (
            "predicted correctness probability of the selected candidate"
        )
    if prediction_path is not None:
        prediction_records = {
            str(record["sample_id"]): record
            for record in read_jsonl(prediction_path)
        }
        if all(
            "selection" in prediction_records[sample.sample_id]
            for sample in legacy
        ):
            confidence = {
                sample.sample_id: max(
                    0.0,
                    float(
                        prediction_records[sample.sample_id]["selection"].get(
                            "gain_lower_bound",
                            prediction_records[sample.sample_id][
                                "selection"
                            ].get("best_gain", 0.0),
                        )
                    ),
                )
                for sample in legacy
            }
            legacy_summary["switch_risk_coverage"] = risk_coverage_curve(
                legacy_rows, confidence
            )
            legacy_summary["switch_risk_coverage_confidence"] = (
                "non-negative conservative gate gain lower bound"
            )
    if expected_legacy_oracle is not None and not np.isclose(
        legacy_summary["oracle_at_5"],
        float(expected_legacy_oracle),
        atol=5e-13,
        rtol=0.0,
    ):
        raise AssertionError(
            "Oracle@5 changed: "
            f"{legacy_summary['oracle_at_5']} != {expected_legacy_oracle}"
        )
    _write_rows(output_dir / "legacy_rows.jsonl", legacy_rows)
    _write_rows(output_dir / "corrected_rows.jsonl", corrected_rows)
    result = {
        "schema_version": "2.0.0",
        "kind": "reranking_evaluation",
        "method": str(method),
        "prediction_path": (
            None if prediction_path is None else str(Path(prediction_path).resolve())
        ),
        "legacy_official": legacy_summary,
        "corrected": corrected_summary,
        "oracle_invariant_passed": bool(
            expected_legacy_oracle is None
            or np.isclose(
                legacy_summary["oracle_at_5"],
                float(expected_legacy_oracle),
                atol=5e-13,
                rtol=0.0,
            )
        ),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result


def combine_method_evaluations(
    summary_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    methods = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in summary_paths
    ]
    if not methods:
        raise ValueError("at least one method summary is required")
    raw_p = {
        str(item["method"]): float(
            item["legacy_official"]["mcnemar_exact_two_sided_pvalue"]
        )
        for item in methods
        if item["method"] != "q_only"
    }
    adjusted = holm_adjust(raw_p)
    rows = []
    for item in methods:
        legacy = item["legacy_official"]
        corrected = item["corrected"]
        rows.append(
            {
                "method": item["method"],
                "sample_count": legacy["sample_count"],
                "legacy_j1": legacy["legacy_or_corrected_j1"],
                "corrected_j1": corrected["legacy_or_corrected_j1"],
                "delta_j1_pp": legacy["delta_j1_percentage_points"],
                "oracle_at_5": legacy["oracle_at_5"],
                "recovered": legacy["recovered"],
                "harmful": legacy["harmful"],
                "net_recovered": legacy["net_recovered"],
                "neutral_switch": legacy["neutral_switch"],
                "switch_coverage": legacy["switch_coverage"],
                "outcome_changing_switch_precision": legacy[
                    "outcome_changing_switch_precision"
                ],
                "mrr_at_5": legacy["mrr_at_5"],
                "ndcg_at_5": legacy["ndcg_at_5"],
                "candidate_brier": legacy.get("candidate_brier"),
                "candidate_nll": legacy.get("candidate_nll"),
                "candidate_ece": legacy.get("candidate_ece"),
                "mcnemar_p_raw": (
                    None
                    if item["method"] == "q_only"
                    else raw_p[item["method"]]
                ),
                "mcnemar_p_holm": (
                    None
                    if item["method"] == "q_only"
                    else adjusted[item["method"]]
                ),
                "frame_ci_low_pp": legacy[
                    "frame_cluster_bootstrap_delta_95ci"
                ][0],
                "frame_ci_high_pp": legacy[
                    "frame_cluster_bootstrap_delta_95ci"
                ][1],
                "sequence_ci_low_pp": legacy[
                    "sequence_cluster_bootstrap_delta_95ci"
                ][0],
                "sequence_ci_high_pp": legacy[
                    "sequence_cluster_bootstrap_delta_95ci"
                ][1],
            }
        )
    csv_path = output_dir / "method_results.csv"
    temporary_csv = csv_path.with_name(
        f".{csv_path.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary_csv.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
    result = {
        "schema_version": "2.0.0",
        "kind": "reranking_method_comparison",
        "methods": rows,
        "holm_family_size": len(raw_p),
    }
    atomic_write_json(output_dir / "method_results.json", result)
    return result
