from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from failure_analysis.gemini_crog_evidence_v1.planner import (
    load_annotation_query_types,
)


MODEL_IDS = (
    "gemini-robotics-er-2-preview",
    "gemini-3.6-flash",
)


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary)
    temporary.replace(path)


def _stable_sample_id(feature: Mapping[str, Any]) -> str:
    sample = feature.get("sample_id", feature.get("sample_index"))
    if isinstance(sample, str) and sample.startswith("multiple:"):
        return sample
    split = str(feature.get("split", "train")).lower()
    return f"multiple:{split}:{int(sample):08d}"


def _ordered_candidates(feature: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = list(feature["candidates"])
    if len(candidates) != 5:
        raise ValueError(f"frozen Top-5 changed for {_stable_sample_id(feature)}")
    return sorted(
        candidates,
        key=lambda item: (-float(item["q_raw"]), str(item["candidate_id"])),
    )


def _label_by_id(
    label: Mapping[str, Any], candidate_ids: set[str]
) -> dict[str, bool]:
    result = {
        str(item["candidate_id"]): bool(item["candidate_correct"])
        for item in label["candidate_labels"]
    }
    if set(result) != candidate_ids:
        raise ValueError(f"candidate label identity mismatch for {label['sample_id']}")
    return result


def _feature_value(candidate: Mapping[str, Any], name: str) -> tuple[float, float]:
    value = candidate.get("features", {}).get(name, {})
    if not isinstance(value, Mapping):
        return 0.0, 0.0
    raw = value.get("value")
    reliability = float(value.get("reliability", 0.0) or 0.0)
    return (float(raw) if raw is not None and math.isfinite(float(raw)) else 0.0, reliability)


def _cohort(q_correct: bool, oracle: bool) -> str:
    if q_correct:
        return "protected_correct"
    return "recoverable_error" if oracle else "unrecoverable_error"


def _quantile_edges(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(value) for value in np.quantile(np.asarray(values), [0.2, 0.4, 0.6, 0.8])]


def _bin(value: float, edges: Sequence[float]) -> str:
    return f"q{1 + sum(float(value) > float(edge) for edge in edges)}"


def build_cohort_records(
    *,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    corrected_labels_path: str | Path,
    split_manifest_path: str | Path,
    annotation_paths: Mapping[str, str | Path],
    partitions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build evaluator-only cohort rows without changing frozen candidates."""

    split_payload = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    split_rows = {str(row["sample_id"]): row for row in split_payload["rows"]}
    query_types = {
        split: load_annotation_query_types(path)
        for split, path in annotation_paths.items()
    }
    wanted = set(partitions or {"train", "calibration"})
    records: list[dict[str, Any]] = []
    for feature, legacy, corrected in zip(
        _jsonl(features_path),
        _jsonl(legacy_labels_path),
        _jsonl(corrected_labels_path),
        strict=True,
    ):
        sample_id = _stable_sample_id(feature)
        split_row = split_rows.get(sample_id)
        if split_row is None:
            raise ValueError(f"sample absent from frozen split manifest: {sample_id}")
        partition = str(split_row["development_partition"])
        if partition not in wanted:
            continue
        if str(legacy["sample_id"]) != sample_id or str(corrected["sample_id"]) != sample_id:
            raise ValueError(f"feature/evaluator identity mismatch for {sample_id}")
        ordered = _ordered_candidates(feature)
        candidate_ids = {str(item["candidate_id"]) for item in ordered}
        legacy_by_id = _label_by_id(legacy, candidate_ids)
        corrected_by_id = _label_by_id(corrected, candidate_ids)
        q_id = str(ordered[0]["candidate_id"])
        q_values = [float(item["q_raw"]) for item in ordered]
        mask_reliability, mask_feature_reliability = _feature_value(
            ordered[0], "mask_consistency"
        )
        _, depth_feature_reliability = _feature_value(ordered[0], "depth_mad_m")
        depth_available = bool(ordered[0].get("diagnostics", {}).get("depth_available"))
        depth_reliability = depth_feature_reliability if depth_available else 0.0
        official_split = str(split_row["official_split"])
        source_sample_id = int(split_row["source_sample_id"])
        query_type = query_types[official_split][source_sample_id]
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "partition": partition,
            "official_split": official_split,
            "frame_id": str(split_row["frame_id"]),
            "scene_id": str(split_row["scene_id"]),
            "group_id": str(split_row["sequence_id"]),
            "query_type": query_type,
            "q_only_candidate_id": q_id,
            "q0": q_values[0],
            "q_margin": q_values[0] - q_values[1],
            "mask_reliability": mask_reliability * mask_feature_reliability,
            "depth_reliability": depth_reliability,
            "legacy_q_correct": legacy_by_id[q_id],
            "legacy_oracle_at_5": any(legacy_by_id.values()),
            "legacy_cohort": _cohort(legacy_by_id[q_id], any(legacy_by_id.values())),
            "corrected_q_correct": corrected_by_id[q_id],
            "corrected_oracle_at_5": any(corrected_by_id.values()),
            "corrected_cohort": _cohort(
                corrected_by_id[q_id], any(corrected_by_id.values())
            ),
        }
        for rank, candidate in enumerate(ordered, start=1):
            candidate_id = str(candidate["candidate_id"])
            record[f"candidate_{rank}_id"] = candidate_id
            record[f"candidate_{rank}_legacy_correct"] = legacy_by_id[candidate_id]
            record[f"candidate_{rank}_corrected_correct"] = corrected_by_id[candidate_id]
        records.append(record)

    bin_edges: dict[str, dict[str, list[float]]] = {}
    for partition in sorted(wanted):
        subset = [row for row in records if row["partition"] == partition]
        bin_edges[partition] = {}
        for field in ("q_margin", "mask_reliability", "depth_reliability"):
            edges = _quantile_edges([float(row[field]) for row in subset])
            bin_edges[partition][field] = edges
            for row in subset:
                row[f"{field}_bin"] = _bin(float(row[field]), edges)

    summary: dict[str, Any] = {"bin_edges": bin_edges, "partitions": {}}
    for partition in sorted(wanted):
        subset = [row for row in records if row["partition"] == partition]
        partition_summary: dict[str, Any] = {
            "samples": len(subset),
            "frames": len({row["frame_id"] for row in subset}),
            "groups": len({row["group_id"] for row in subset}),
            "query_type_distribution": dict(Counter(row["query_type"] for row in subset)),
        }
        for track in ("legacy", "corrected"):
            q = sum(bool(row[f"{track}_q_correct"]) for row in subset)
            oracle = sum(bool(row[f"{track}_oracle_at_5"]) for row in subset)
            cohorts = Counter(str(row[f"{track}_cohort"]) for row in subset)
            partition_summary[track] = {
                "q_only_successes": q,
                "q_only_j1": q / len(subset) if subset else None,
                "oracle_at_5_successes": oracle,
                "oracle_at_5": oracle / len(subset) if subset else None,
                "recoverable_error_prevalence": cohorts["recoverable_error"] / len(subset)
                if subset
                else None,
                "cohorts": dict(cohorts),
            }
            for field in ("q_margin_bin", "mask_reliability_bin", "depth_reliability_bin"):
                strata: dict[str, dict[str, int | float]] = {}
                for name in sorted({str(row[field]) for row in subset}):
                    rows = [row for row in subset if row[field] == name]
                    recoverable = sum(
                        row[f"{track}_cohort"] == "recoverable_error" for row in rows
                    )
                    strata[name] = {
                        "samples": len(rows),
                        "recoverable_errors": recoverable,
                        "recoverable_prevalence": recoverable / len(rows) if rows else 0.0,
                    }
                partition_summary[track][field] = strata
        summary["partitions"][partition] = partition_summary
    return records, summary


def _direct_metric(rows: Sequence[Mapping[str, Any]], track: str) -> dict[str, Any]:
    total = len(rows)
    q = sum(bool(row[f"{track}_q_correct"]) for row in rows)
    selected = sum(bool(row[f"{track}_selected_correct"]) for row in rows)
    recovered = sum(
        not bool(row[f"{track}_q_correct"]) and bool(row[f"{track}_selected_correct"])
        for row in rows
    )
    harmful = sum(
        bool(row[f"{track}_q_correct"]) and not bool(row[f"{track}_selected_correct"])
        for row in rows
    )
    switches = sum(bool(row["switch"]) for row in rows)
    baseline_correct = q
    recoverable = sum(
        (not bool(row[f"{track}_q_correct"])) and bool(row[f"{track}_oracle_at_5"])
        for row in rows
    )
    return {
        "total": total,
        "q_only_successes": q,
        "q_only_j1": q / total if total else None,
        "selected_successes": selected,
        "selected_j1": selected / total if total else None,
        "delta_pp": 100.0 * (selected - q) / total if total else None,
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "outcome_changing_precision": recovered / (recovered + harmful)
        if recovered + harmful
        else None,
        "harm_rate": harmful / baseline_correct if baseline_correct else None,
        "recovery_recall": recovered / recoverable if recoverable else None,
        "switches": switches,
        "switch_rate": switches / total if total else None,
    }


def replay_direct_decisions(
    *,
    decision_dir: str | Path,
    cohort_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = {str(row["sample_id"]): row for row in cohort_records}
    decisions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(decision_dir).glob("*.json"))
    ]
    output: list[dict[str, Any]] = []
    for decision in decisions:
        truth = evaluation[str(decision["sample_id"])]
        valid = bool(decision.get("json_parse_valid")) and bool(
            decision.get("schema_valid")
        ) and str(decision.get("lifecycle_status")) in {"SUCCEEDED", "ABSTAIN"}
        fallback = (not valid) or bool(decision.get("abstain"))
        selected = (
            str(truth["q_only_candidate_id"])
            if fallback
            else str(decision["selected_candidate_id"])
        )
        if selected not in {str(truth[f"candidate_{rank}_id"]) for rank in range(1, 6)}:
            raise ValueError("Direct decision selected an ID outside frozen Top-5")
        rank = next(
            rank
            for rank in range(1, 6)
            if str(truth[f"candidate_{rank}_id"]) == selected
        )
        row: dict[str, Any] = {
            "sample_id": str(decision["sample_id"]),
            "model_id": str(decision["model_id"]),
            "request_hash": str(decision["request_hash"]),
            "status": str(decision.get("lifecycle_status")),
            "schema_valid": bool(decision.get("schema_valid")),
            "fallback_to_q_only": fallback,
            "selected_candidate_id": selected,
            "q_only_candidate_id": str(truth["q_only_candidate_id"]),
            "switch": selected != str(truth["q_only_candidate_id"]),
            "cache_hit": bool(decision.get("cache_hit")),
            "api_attempted": bool(decision.get("api_attempted", True)),
            "permanent_api_failure": bool(decision.get("permanent_api_failure")),
            "latency_seconds": float(decision.get("latency_seconds", 0.0) or 0.0),
            "estimated_charge_usd": float(
                decision.get("estimated_charge_usd", 0.0) or 0.0
            ),
        }
        for track in ("legacy", "corrected"):
            row[f"{track}_q_correct"] = bool(truth[f"{track}_q_correct"])
            row[f"{track}_oracle_at_5"] = bool(truth[f"{track}_oracle_at_5"])
            row[f"{track}_selected_correct"] = bool(
                truth[f"candidate_{rank}_{track}_correct"]
            )
        output.append(row)

    summary: dict[str, Any] = {
        "decision_records": len(output),
        "distinct_samples": len({row["sample_id"] for row in output}),
        "model_sample_decisions": len(
            {(row["model_id"], row["sample_id"]) for row in output}
        ),
        "models": {},
    }
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_model[str(row["model_id"])].append(row)
    for model, rows in sorted(by_model.items()):
        new_latencies = [
            float(row["latency_seconds"])
            for row in rows
            if row["api_attempted"] and not row["cache_hit"]
        ]
        summary["models"][model] = {
            "decisions": len(rows),
            "schema_valid": sum(bool(row["schema_valid"]) for row in rows),
            "fallback": sum(bool(row["fallback_to_q_only"]) for row in rows),
            "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
            "terminal_failures": sum(bool(row["permanent_api_failure"]) for row in rows),
            "latency_p50_seconds": float(np.percentile(new_latencies, 50))
            if new_latencies
            else None,
            "latency_p95_seconds": float(np.percentile(new_latencies, 95))
            if new_latencies
            else None,
            "legacy": _direct_metric(rows, "legacy"),
            "corrected": _direct_metric(rows, "corrected"),
        }
    valid_by_model = {
        model: {
            str(row["sample_id"]): row
            for row in rows
            if row["schema_valid"] and not row["fallback_to_q_only"]
        }
        for model, rows in by_model.items()
    }
    if set(valid_by_model) == set(MODEL_IDS):
        common = set(valid_by_model[MODEL_IDS[0]]) & set(valid_by_model[MODEL_IDS[1]])
        agreement = sum(
            valid_by_model[MODEL_IDS[0]][sample]["selected_candidate_id"]
            == valid_by_model[MODEL_IDS[1]][sample]["selected_candidate_id"]
            for sample in common
        )
        summary["paired_common_valid"] = len(common)
        summary["paired_selected_id_agreement"] = agreement
        summary["paired_selected_id_agreement_rate"] = (
            agreement / len(common) if common else None
        )
    return output, summary


def ledger_summary(cache_path: str | Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{Path(cache_path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        attempts = [dict(row) for row in connection.execute("SELECT * FROM request_attempts")]
        responses = [dict(row) for row in connection.execute("SELECT * FROM responses")]
    finally:
        connection.close()
    success_counts = Counter(
        str(row["request_hash"])
        for row in attempts
        if row["status"] in {"SUCCEEDED", "ABSTAIN"}
    )
    return {
        "attempts": len(attempts),
        "attempt_status": dict(Counter(str(row["status"]) for row in attempts)),
        "distinct_attempt_request_hashes": len(
            {str(row["request_hash"]) for row in attempts}
        ),
        "duplicate_successful_request_hashes": sum(
            count > 1 for count in success_counts.values()
        ),
        "attempt_estimated_charge_usd": sum(
            float(row["estimated_charge_usd"]) for row in attempts
        ),
        "response_rows": len(responses),
        "valid_response_rows": sum(bool(row["valid"]) for row in responses),
        "invalid_response_rows": sum(not bool(row["valid"]) for row in responses),
        "response_table_estimated_charge_usd": sum(
            float(row["estimated_charge_usd"]) for row in responses
        ),
    }


def replay_existing_direct(
    *,
    output_dir: str | Path,
    existing_run_root: str | Path,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    corrected_labels_path: str | Path,
    split_manifest_path: str | Path,
    annotation_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Run Stage-1 replay without importing or invoking any API client."""

    output = Path(output_dir)
    records, cohort_summary = build_cohort_records(
        features_path=features_path,
        legacy_labels_path=legacy_labels_path,
        corrected_labels_path=corrected_labels_path,
        split_manifest_path=split_manifest_path,
        annotation_paths=annotation_paths,
    )
    direct_rows, direct_summary = replay_direct_decisions(
        decision_dir=Path(existing_run_root) / "pilot" / "decisions",
        cohort_records=records,
    )
    ledger = ledger_summary(Path(existing_run_root) / "gemini_cache.sqlite")
    result = {
        "schema_version": "1.0.0",
        "mode": "offline_replay_no_api",
        "cohorts": cohort_summary,
        "direct": direct_summary,
        "ledger": ledger,
        "corrected_direct_positive_net_possible_on_observed_choices": all(
            int(value["corrected"]["recovered"]) > 0
            for value in direct_summary["models"].values()
        ),
        "direct_threshold_upper_bound_explanation": (
            "Every observed corrected outcome-changing Direct switch is harmful; "
            "rejecting all switches yields Net=0 and thresholding cannot create a "
            "beneficial candidate that was never selected."
        ),
    }
    _write_parquet(output / "per_sample_cohorts.parquet", records)
    _write_parquet(output / "direct_decision_outcomes.parquet", direct_rows)
    _write_json(output / "OFFLINE_REPLAY.json", result)
    return result
