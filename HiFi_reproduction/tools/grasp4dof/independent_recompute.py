#!/usr/bin/env python3
"""Independently recompute formal 4-DoF outcomes from stored geometry only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.candidate_decoder import rank_candidates  # noqa: E402
from src.grasping.common.evaluator import evaluate_ocid_predictions  # noqa: E402
from src.grasping.common.types import Grasp4DoF  # noqa: E402


CANDIDATE_COLUMNS = (
    "sample_id",
    "candidate_id",
    "rank",
    "center_x",
    "center_y",
    "angle_deg",
    "width_px",
    "height_px",
    "score",
)
SAMPLE_COLUMNS = (
    "sample_id",
    "scene_id",
    "j_at_1",
    "j_at_5",
    "candidate_pool_oracle",
    "first_valid_rank",
    "reciprocal_rank",
    "non_empty",
    "nms_candidate_count",
    "top1_candidate_id",
    "top5_candidate_ids_json",
)
AGGREGATE_KEYS = (
    "sample_count",
    "j_at_1",
    "j_at_5",
    "recall_at_5",
    "candidate_pool_oracle",
    "mrr",
    "mean_first_valid_rank",
    "median_first_valid_rank",
    "non_empty_rate",
    "no_grasp_rate",
    "non_empty_j_at_1",
    "non_empty_j_at_5",
    "mean_nms_candidates",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    return pd.read_parquet(path)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _strict_bool(value: Any, *, field: str, sample_id: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{sample_id}: saved {field} is not binary")


def _optional_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return str(value)


def _optional_rank(value: Any, *, sample_id: str) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"{sample_id}: saved first_valid_rank is invalid")
    return int(number)


def _strict_nonnegative_integer(value: Any, *, field: str, sample_id: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{sample_id}: {field} is not an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{sample_id}: {field} is not a non-negative integer")
    return int(number)


def _parse_id_list(value: Any, *, sample_id: str) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{sample_id}: invalid top5_candidate_ids_json") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"{sample_id}: top5_candidate_ids_json must be a string list")
    return parsed


def _load_labels(path: Path) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    labels = _read_parquet(path, label="frozen test labels")
    _require_columns(
        labels,
        ("sample_id", "scene_id", "gt_grasp_rectangles"),
        label="frozen test labels",
    )
    if labels.empty or labels[["sample_id", "scene_id", "gt_grasp_rectangles"]].isna().any().any():
        raise ValueError("frozen test labels are empty or contain nulls")
    if bool(labels.duplicated("sample_id", keep=False).any()):
        raise ValueError("duplicate sample_id in frozen test labels")
    ordered_ids = [str(value) for value in labels["sample_id"].tolist()]
    if any(not value for value in ordered_ids):
        raise ValueError("empty sample_id in frozen test labels")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in labels.itertuples(index=False):
        rectangles: list[list[list[float]]] = []
        for rectangle in row.gt_grasp_rectangles:
            points = [np.asarray(point, dtype=np.float64).reshape(-1) for point in rectangle]
            if len(points) != 4 or any(point.shape != (2,) for point in points):
                raise ValueError(f"{row.sample_id}: malformed frozen GT rectangle")
            array = np.stack(points)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{row.sample_id}: non-finite frozen GT rectangle")
            rectangles.append(array.tolist())
        if not rectangles:
            raise ValueError(f"{row.sample_id}: frozen test label has no GT grasps")
        by_id[str(row.sample_id)] = {
            "sample_id": str(row.sample_id),
            "scene_id": str(row.scene_id),
            "gt_grasp_rectangles": rectangles,
        }
    return ordered_ids, by_id


def _candidate_pool(frame: pd.DataFrame, *, sample_id: str) -> list[Grasp4DoF]:
    if frame.empty:
        return []
    if frame[list(CANDIDATE_COLUMNS)].isna().any().any():
        raise ValueError(f"{sample_id}: candidate trusted fields contain nulls")
    ranks: list[int] = []
    candidates: list[Grasp4DoF] = []
    for row in frame.itertuples(index=False):
        rank = _strict_nonnegative_integer(row.rank, field="rank", sample_id=sample_id)
        if rank <= 0:
            raise ValueError(f"{sample_id}: candidate rank must be positive")
        candidate_id = str(row.candidate_id)
        if not candidate_id:
            raise ValueError(f"{sample_id}: candidate_id cannot be empty")
        ranks.append(rank)
        candidates.append(
            Grasp4DoF(
                center_x=float(row.center_x),
                center_y=float(row.center_y),
                angle_deg=float(row.angle_deg),
                width_px=float(row.width_px),
                height_px=float(row.height_px),
                score=float(row.score),
                candidate_id=candidate_id,
            )
        )
    order = np.argsort(np.asarray(ranks), kind="stable")
    sorted_ranks = [ranks[index] for index in order]
    expected_ranks = list(range(1, len(ranks) + 1))
    if sorted_ranks != expected_ranks:
        raise ValueError(
            f"{sample_id}: candidate ranks are duplicate, missing, or non-contiguous: "
            f"{sorted_ranks[:10]}"
        )
    ordered = [candidates[index] for index in order]
    if len({item.candidate_id for item in ordered}) != len(ordered):
        raise ValueError(f"{sample_id}: duplicate candidate_id")
    reranked = rank_candidates(ordered)
    if [item.candidate_id for item in reranked] != [item.candidate_id for item in ordered]:
        raise ValueError(f"{sample_id}: saved rank disagrees with deterministic score order")
    return ordered


def _validate_saved_top_json(
    saved: Mapping[str, Any], pool: Sequence[Grasp4DoF], *, sample_id: str
) -> None:
    if "top5_json" not in saved:
        return
    try:
        rows = json.loads(str(saved["top5_json"]))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{sample_id}: invalid top5_json") from error
    if not isinstance(rows, list):
        raise ValueError(f"{sample_id}: top5_json must be a list")
    expected = [item.to_dict(rank=rank) for rank, item in enumerate(pool[:5], 1)]
    trusted_keys = (
        "candidate_id",
        "rank",
        "center_x",
        "center_y",
        "angle_deg",
        "width_px",
        "height_px",
        "score",
    )
    if len(rows) != len(expected):
        raise ValueError(f"{sample_id}: top5_json length drift")
    for rank, (observed, wanted) in enumerate(zip(rows, expected, strict=True), 1):
        if not isinstance(observed, Mapping):
            raise ValueError(f"{sample_id}: top5_json rank {rank} is not an object")
        for key in trusted_keys:
            if key not in observed or observed[key] != wanted[key]:
                raise ValueError(f"{sample_id}: top5_json {key} drift at rank {rank}")


def _recomputed_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate zero recomputed samples")
    count = len(rows)
    non_empty = [row for row in rows if bool(row["non_empty"])]
    positive_ranks = [
        int(row["first_valid_rank"])
        for row in rows
        if row["first_valid_rank"] is not None
    ]

    def mean_bool(key: str, values: Sequence[Mapping[str, Any]]) -> float:
        return 0.0 if not values else float(np.mean([bool(row[key]) for row in values]))

    return {
        "sample_count": count,
        "j_at_1": mean_bool("j_at_1", rows),
        "j_at_5": mean_bool("j_at_5", rows),
        "recall_at_5": mean_bool("j_at_5", rows),
        "candidate_pool_oracle": mean_bool("candidate_pool_oracle", rows),
        "mrr": float(np.mean([float(row["reciprocal_rank"]) for row in rows])),
        "mean_first_valid_rank": None
        if not positive_ranks
        else float(np.mean(positive_ranks)),
        "median_first_valid_rank": None
        if not positive_ranks
        else float(np.median(positive_ranks)),
        "non_empty_rate": len(non_empty) / count,
        "no_grasp_rate": 1.0 - len(non_empty) / count,
        "non_empty_j_at_1": mean_bool("j_at_1", non_empty),
        "non_empty_j_at_5": mean_bool("j_at_5", non_empty),
        "mean_nms_candidates": float(
            np.mean([int(row["nms_candidate_count"]) for row in rows])
        ),
    }


def _exact_equal(observed: Any, expected: Any) -> bool:
    if observed is None or (isinstance(observed, float) and math.isnan(observed)):
        return expected is None
    if expected is None:
        return False
    if isinstance(expected, bool):
        return isinstance(observed, (bool, np.bool_)) and bool(observed) == expected
    if isinstance(expected, int):
        if isinstance(observed, (bool, np.bool_)):
            return False
        try:
            number = float(observed)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and number.is_integer() and int(number) == expected
    if isinstance(expected, float):
        return float(observed) == expected
    return observed == expected


def recompute_method(
    *,
    method_name: str,
    method_dir: str | Path,
    ordered_label_ids: Sequence[str],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    directory = Path(method_dir).expanduser().resolve()
    candidates_path = directory / "per_candidate_predictions.parquet"
    samples_path = directory / "per_sample_predictions.parquet"
    metrics_path = directory / "metrics.json"
    candidates = _read_parquet(candidates_path, label=f"{method_name} candidates")
    samples = _read_parquet(samples_path, label=f"{method_name} samples")
    if candidates.empty and not set(CANDIDATE_COLUMNS).issubset(candidates.columns):
        # ``pyarrow.Table.from_pylist([])`` has no inferred fields.  The
        # per-sample table still proves that every frozen sample is empty.
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    else:
        _require_columns(candidates, CANDIDATE_COLUMNS, label=f"{method_name} candidates")
    _require_columns(samples, SAMPLE_COLUMNS, label=f"{method_name} samples")
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {method_name} metrics: {metrics_path}")
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(saved_metrics, Mapping):
        raise ValueError(f"{method_name}: metrics.json must contain an object")

    if bool(samples.duplicated("sample_id", keep=False).any()):
        raise ValueError(f"{method_name}: duplicate sample_id in per-sample predictions")
    saved_ids = [str(value) for value in samples["sample_id"].tolist()]
    if set(saved_ids) != set(ordered_label_ids):
        missing = sorted(set(ordered_label_ids) - set(saved_ids))
        extra = sorted(set(saved_ids) - set(ordered_label_ids))
        raise ValueError(
            f"{method_name}: sample coverage mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    candidate_sample_ids = {str(value) for value in candidates["sample_id"].tolist()}
    outside = sorted(candidate_sample_ids - set(ordered_label_ids))
    if outside:
        raise ValueError(f"{method_name}: candidates outside frozen labels: {outside[:5]}")
    candidate_ids = [str(value) for value in candidates["candidate_id"].tolist()]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{method_name}: duplicate candidate_id across method")

    saved_by_id = {
        str(row.sample_id): row._asdict() for row in samples.itertuples(index=False)
    }
    candidate_groups = {
        str(sample_id): frame
        for sample_id, frame in candidates.groupby("sample_id", sort=False)
    }
    recomputed_rows: list[dict[str, Any]] = []
    for sample_id in ordered_label_ids:
        label = labels_by_id[sample_id]
        saved = saved_by_id[sample_id]
        if str(saved["scene_id"]) != str(label["scene_id"]):
            raise ValueError(f"{method_name}/{sample_id}: scene_id mismatch")
        pool = _candidate_pool(
            candidate_groups.get(sample_id, candidates.iloc[0:0]), sample_id=sample_id
        )
        outcome = evaluate_ocid_predictions(pool, label["gt_grasp_rectangles"])
        top_ids = [item.candidate_id for item in pool[:5]]
        recomputed = {
            "sample_id": sample_id,
            "scene_id": str(label["scene_id"]),
            "j_at_1": bool(outcome.j_at_1),
            "j_at_5": bool(outcome.j_at_5),
            "candidate_pool_oracle": bool(
                any(item.candidate_success for item in outcome.candidates)
            ),
            "first_valid_rank": outcome.first_valid_rank,
            "reciprocal_rank": outcome.reciprocal_rank,
            "non_empty": bool(pool),
            "nms_candidate_count": len(pool),
            "top1_candidate_id": None if not pool else pool[0].candidate_id,
            "top5_candidate_ids": top_ids,
        }
        saved_values = {
            "j_at_1": _strict_bool(saved["j_at_1"], field="j_at_1", sample_id=sample_id),
            "j_at_5": _strict_bool(saved["j_at_5"], field="j_at_5", sample_id=sample_id),
            "candidate_pool_oracle": _strict_bool(
                saved["candidate_pool_oracle"],
                field="candidate_pool_oracle",
                sample_id=sample_id,
            ),
            "first_valid_rank": _optional_rank(
                saved["first_valid_rank"], sample_id=sample_id
            ),
            "reciprocal_rank": float(saved["reciprocal_rank"]),
            "non_empty": _strict_bool(
                saved["non_empty"], field="non_empty", sample_id=sample_id
            ),
            "nms_candidate_count": _strict_nonnegative_integer(
                saved["nms_candidate_count"],
                field="nms_candidate_count",
                sample_id=sample_id,
            ),
            "top1_candidate_id": _optional_string(saved["top1_candidate_id"]),
            "top5_candidate_ids": _parse_id_list(
                saved["top5_candidate_ids_json"], sample_id=sample_id
            ),
        }
        for key, expected in recomputed.items():
            if key in ("sample_id", "scene_id"):
                continue
            if not _exact_equal(saved_values[key], expected):
                raise RuntimeError(
                    f"{method_name}/{sample_id}: saved {key} mismatch: "
                    f"saved={saved_values[key]!r}, recomputed={expected!r}"
                )
        if not pool and not _optional_string(saved.get("empty_reason")):
            raise RuntimeError(f"{method_name}/{sample_id}: empty prediction lacks reason")
        _validate_saved_top_json(saved, pool, sample_id=sample_id)
        recomputed_rows.append(recomputed)

    metrics = _recomputed_aggregate(recomputed_rows)
    missing_metric_keys = sorted(set(AGGREGATE_KEYS) - set(saved_metrics))
    if missing_metric_keys:
        raise ValueError(
            f"{method_name}: metrics.json missing independently recomputable keys: "
            f"{missing_metric_keys}"
        )
    for key in AGGREGATE_KEYS:
        if not _exact_equal(saved_metrics[key], metrics[key]):
            raise RuntimeError(
                f"{method_name}: aggregate {key} mismatch: "
                f"saved={saved_metrics[key]!r}, recomputed={metrics[key]!r}"
            )
    return {
        "status": "EXACT_MATCH",
        "method": method_name,
        "method_dir": str(directory),
        "sample_count": len(recomputed_rows),
        "candidate_count": len(candidates),
        "empty_prediction_count": int(
            sum(not bool(row["non_empty"]) for row in recomputed_rows)
        ),
        "fewer_than_five_count": int(
            sum(int(row["nms_candidate_count"]) < 5 for row in recomputed_rows)
        ),
        "compared_sample_fields": list(SAMPLE_COLUMNS[2:]),
        "compared_aggregate_metrics": list(AGGREGATE_KEYS),
        "metrics": metrics,
        "per_candidate_predictions": str(candidates_path),
        "per_candidate_predictions_sha256": sha256_file(candidates_path),
        "per_sample_predictions": str(samples_path),
        "per_sample_predictions_sha256": sha256_file(samples_path),
        "saved_metrics": str(metrics_path),
        "saved_metrics_sha256": sha256_file(metrics_path),
    }


def parse_method_dir(value: str) -> tuple[str, Path]:
    name, separator, raw_path = str(value).partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--method-dir must be NAME=PATH")
    return name.strip(), Path(raw_path).expanduser().resolve()


def run_independent_recompute(
    *, run_dir: str | Path, method_dirs: Sequence[tuple[str, Path]]
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    labels_path = root / "manifests/test_labels.parquet"
    ordered_ids, labels = _load_labels(labels_path)
    if not method_dirs:
        raise ValueError("at least one formal method directory is required")
    names = [name for name, _ in method_dirs]
    if len(names) != len(set(names)):
        raise ValueError("formal method names must be unique")
    methods = [
        recompute_method(
            method_name=name,
            method_dir=path,
            ordered_label_ids=ordered_ids,
            labels_by_id=labels,
        )
        for name, path in method_dirs
    ]
    return {
        "schema_version": 1,
        "status": "EXACT_MATCH",
        "independent_recompute": True,
        "configuration_selection_read": False,
        "candidate_correctness_fields_trusted": False,
        "candidate_trusted_fields": list(CANDIDATE_COLUMNS),
        "evaluator": "common corrected same-GT OCID evaluator",
        "frozen_test_labels": str(labels_path),
        "frozen_test_labels_sha256": sha256_file(labels_path),
        "sample_count": len(ordered_ids),
        "method_count": len(methods),
        "methods": methods,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--method-dir",
        action="append",
        type=parse_method_dir,
        required=True,
        metavar="NAME=PATH",
    )
    args = parser.parse_args(argv)
    root = args.run_dir.expanduser().resolve()
    output = root / "independent_recompute_results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite independent recompute: {output}")
    result = run_independent_recompute(run_dir=root, method_dirs=args.method_dir)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": "EXACT_MATCH"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
