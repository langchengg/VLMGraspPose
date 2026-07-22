"""Denominator-explicit metrics and portable experiment exports."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .bootstrap import cluster_bootstrap_interval
from .failure_taxonomy import is_candidate_outcome, is_terminal


def wilson_interval(
    positive_count: int,
    denominator: int,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Wilson score interval for a binomial proportion."""

    positives = int(positive_count)
    total = int(denominator)
    if total < 0 or positives < 0 or positives > total:
        raise ValueError("counts must satisfy 0 <= positive_count <= denominator")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if total == 0:
        return {
            "numerator": positives,
            "denominator": total,
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence": confidence,
            "method": "wilson",
        }
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = positives / total
    z_squared = z * z
    divisor = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / divisor
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / divisor
    )
    return {
        "numerator": positives,
        "denominator": total,
        "estimate": proportion,
        "ci_lower": max(0.0, center - radius),
        "ci_upper": min(1.0, center + radius),
        "confidence": confidence,
        "method": "wilson",
    }


def _finite(row: Mapping[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [value for row in rows if (value := _finite(row, key)) is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "maximum": max(values),
    }


def _status(row: Mapping[str, Any]) -> str:
    raw = row.get("outcome_status") or row.get("status") or row.get("state")
    return str(raw or "pending")


def _truthfulness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if is_candidate_outcome(_status(row))]
    missing_score_source = sum(not row.get("score_source") for row in scored)
    missing_reranking = sum("custom_reranking" not in row for row in scored)
    missing_tsdf_mode = sum(not row.get("tsdf_mode") for row in scored)
    score_sources = sorted(
        {str(row["score_source"]) for row in scored if row.get("score_source") is not None}
    )
    reranking_values = sorted(
        {bool(row["custom_reranking"]) for row in scored if "custom_reranking" in row}
    )
    tsdf_modes = sorted(
        {str(row["tsdf_mode"]) for row in scored if row.get("tsdf_mode") is not None}
    )
    return {
        "candidate_outcome_count": len(scored),
        "score_source_values": score_sources,
        "score_source_missing_count": missing_score_source,
        "all_scores_from_official_processed_quality": bool(scored)
        and score_sources == ["official_vgn_processed_quality"]
        and missing_score_source == 0,
        "custom_reranking_values": reranking_values,
        "custom_reranking_missing_count": missing_reranking,
        "any_custom_reranking": any(reranking_values),
        "all_candidate_outcomes_disable_custom_reranking": bool(scored)
        and reranking_values == [False]
        and missing_reranking == 0,
        "tsdf_mode_values": tsdf_modes,
        "tsdf_mode_missing_count": missing_tsdf_mode,
        "all_candidate_outcomes_disclose_single_view_adaptation": bool(scored)
        and tsdf_modes == ["single_view_adaptation"]
        and missing_tsdf_mode == 0,
    }


def aggregate_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest_count: int,
    confidence: float = 0.95,
    bootstrap_replicates: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate outcomes with named, auditable denominators."""

    materialized = [dict(row) for row in rows]
    total_manifest = int(manifest_count)
    if total_manifest < 0:
        raise ValueError("manifest_count must be non-negative")
    if len(materialized) > total_manifest:
        raise ValueError("row count cannot exceed manifest_count")
    sample_ids = [str(row.get("sample_id", "")) for row in materialized]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("every row must have a sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")

    terminal_rows = [row for row in materialized if is_terminal(_status(row))]
    candidate_rows = [row for row in materialized if is_candidate_outcome(_status(row))]
    official_rows = [
        row for row in candidate_rows if (_finite(row, "official_candidate_count") or 0.0) > 0
    ]
    target_rows = [
        row for row in candidate_rows if (_finite(row, "target_candidate_count") or 0.0) > 0
    ]
    for row in candidate_rows:
        official_count = _finite(row, "official_candidate_count") or 0.0
        target_count = _finite(row, "target_candidate_count") or 0.0
        if official_count < 0 or target_count < 0:
            raise ValueError("candidate counts must be non-negative")
        if target_count > official_count:
            raise ValueError(
                f"sample {row['sample_id']!r} has more target candidates than official candidates"
            )

    denominators = {
        "full_manifest": {
            "count": total_manifest,
            "description": "all rows in the immutable experiment manifest",
        },
        "terminal_samples": {
            "count": len(terminal_rows),
            "description": "rows with terminal scientific or deterministic input status",
        },
        "candidate_generation_reached": {
            "count": len(candidate_rows),
            "description": "rows where VGN candidate post-processing was reached",
        },
        "at_least_one_official_candidate": {
            "count": len(official_rows),
            "description": "candidate-generation rows containing at least one official candidate",
        },
    }
    proportions = {
        "manifest_processing_coverage": wilson_interval(
            len(terminal_rows), total_manifest, confidence=confidence
        ),
        "official_candidate_availability": wilson_interval(
            len(official_rows), len(candidate_rows), confidence=confidence
        ),
        "target_candidate_availability": wilson_interval(
            len(target_rows), len(candidate_rows), confidence=confidence
        ),
        "target_given_official_availability": wilson_interval(
            len(target_rows), len(official_rows), confidence=confidence
        ),
    }

    bootstrap: dict[str, Any] = {}
    if bootstrap_replicates > 0:
        for key in (
            "top1_vgn_quality",
            "official_candidate_count",
            "target_candidate_count",
            "support_plane_residual",
        ):
            finite_rows = [row for row in candidate_rows if _finite(row, key) is not None]
            if finite_rows and all(str(row.get("scene_id", "")).strip() for row in finite_rows):
                bootstrap[f"mean_{key}"] = cluster_bootstrap_interval(
                    finite_rows,
                    key,
                    replicates=bootstrap_replicates,
                    confidence=confidence,
                    seed=seed,
                )

    status_counts = Counter(_status(row) for row in materialized)
    return {
        "manifest_count": total_manifest,
        "registered_row_count": len(materialized),
        "status_counts": dict(sorted(status_counts.items())),
        "denominators": denominators,
        "proportions": proportions,
        "distributions": {
            key: _distribution(candidate_rows, key)
            for key in (
                "official_candidate_count",
                "target_candidate_count",
                "top1_vgn_quality",
                "top1_width_m",
                "support_plane_residual",
                "processing_time_total",
            )
        },
        "scene_cluster_bootstrap": bootstrap,
        "truthfulness": _truthfulness(materialized),
    }


_PREFERRED_COLUMNS = (
    "sample_id",
    "dataset_index",
    "scene_id",
    "instruction",
    "view",
    "state",
    "status",
    "failure_reason",
    "official_candidate_count",
    "target_candidate_count",
    "top1_vgn_quality",
    "top1_width_m",
    "support_plane_residual",
    "processing_time_total",
    "score_source",
    "custom_reranking",
    "tsdf_mode",
)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _export_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    keys = {str(key) for row in rows for key in row if key not in {"manifest_row", "result"}}
    columns = [key for key in _PREFERRED_COLUMNS if key in keys]
    columns.extend(sorted(keys - set(columns)))
    normalized = [{column: _cell(row.get(column)) for column in columns} for row in rows]
    return columns, normalized


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=".experiment-export-", suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(raw_path)


def export_metrics(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    manifest_count: int,
    confidence: float = 0.95,
    bootstrap_replicates: int = 1_000,
    seed: int = 42,
    write_parquet: bool = True,
    require_parquet: bool = False,
) -> dict[str, Any]:
    """Atomically write per-sample CSV/Parquet and aggregate JSON outputs."""

    materialized = [dict(row) for row in rows]
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_metrics(
        materialized,
        manifest_count=manifest_count,
        confidence=confidence,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    columns, normalized = _export_rows(materialized)

    csv_path = directory / "per_sample.csv"
    temporary_csv = _temporary_path(directory, ".csv")
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)

    aggregate_path = directory / "aggregate_metrics.json"
    temporary_json = _temporary_path(directory, ".json")
    try:
        with temporary_json.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_json, aggregate_path)
    finally:
        temporary_json.unlink(missing_ok=True)

    parquet_path: Path | None = None
    parquet_error: str | None = None
    if write_parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            parquet_path = directory / "per_sample.parquet"
            temporary_parquet = _temporary_path(directory, ".parquet")
            try:
                parquet_rows = [
                    {
                        column: (
                            json.dumps(value, sort_keys=True, ensure_ascii=False)
                            if isinstance(value, (dict, list, tuple, set))
                            else value
                        )
                        for column in columns
                        # SQLite/CSV rows use an empty string for absent scalar
                        # diagnostics.  Omitting it prevents PyArrow from
                        # mixing ``""`` with otherwise numeric columns.
                        if (value := row.get(column)) not in (None, "")
                    }
                    for row in materialized
                ]
                table = pa.Table.from_pylist(parquet_rows)
                pq.write_table(table, temporary_parquet)
                os.replace(temporary_parquet, parquet_path)
            finally:
                temporary_parquet.unlink(missing_ok=True)
        except ImportError as error:
            parquet_error = f"PyArrow is unavailable: {error}"
            if require_parquet:
                raise RuntimeError(parquet_error) from error

    return {
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "parquet_error": parquet_error,
        "aggregate_path": aggregate_path,
        "aggregate": aggregate,
    }
