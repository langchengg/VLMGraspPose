from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from utils.grasp_metrics import (
    CORRECTED_EVALUATOR_VERSION,
    LEGACY_EVALUATOR_VERSION,
    evaluate_candidate,
)


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
                yield value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_id(sample_id: str, stable_id: str) -> str:
    prefix = f"{sample_id}/"
    if not stable_id.startswith(prefix):
        raise ValueError("selected candidate ID is not owned by its sample")
    result = stable_id[len(prefix) :]
    if not result or "/" in result:
        raise ValueError("invalid stable candidate ID")
    return result


def _stable_source_id(row: dict[str, Any]) -> str:
    split = str(row["split"]).lower()
    if split in {"validation", "val"}:
        split = "val"
    return f"multiple:{split}:{int(row['sample_id']):08d}"


def _raw_gt_index(path: str | Path) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for row in _jsonl(path):
        sample_id = _stable_source_id(row)
        if sample_id in result:
            raise ValueError("duplicate raw GT sample")
        result[sample_id] = [
            [float(value) for value in grasp]
            for grasp in row.get("gt_grasps", [])
        ]
    return result


def _evaluate_frozen_candidates(
    candidates: list[dict[str, Any]],
    gt_grasps: list[list[float]],
) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {"legacy": {}, "corrected": {}}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        result["legacy"][candidate_id] = bool(
            evaluate_candidate(
                candidate,
                gt_grasps,
                evaluator_version=LEGACY_EVALUATOR_VERSION,
            )["candidate_success"]
        )
        result["corrected"][candidate_id] = bool(
            evaluate_candidate(
                candidate,
                gt_grasps,
                evaluator_version=CORRECTED_EVALUATOR_VERSION,
            )["candidate_success"]
        )
    return result


def recompute(
    *,
    predictions_path: str | Path,
    features_path: str | Path,
    legacy_labels_path: str | Path | None = None,
    corrected_labels_path: str | Path | None = None,
    raw_predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Independently recompute correctness from stable IDs and raw GT grasps.

    Production calls must supply ``raw_predictions_path``.  The label-artifact
    fallback exists only for small backwards-compatible unit fixtures and is
    explicitly disclosed in the result.
    """

    predictions = pq.read_table(predictions_path).to_pylist()
    requested: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in predictions:
        sample_id = str(row["sample_id"])
        method = str(row["method"])
        if (sample_id, method) in seen:
            raise ValueError("duplicate saved prediction")
        seen.add((sample_id, method))
        requested[sample_id].append(row)

    per_sample: list[dict[str, Any]] = []
    matched: set[str] = set()
    if raw_predictions_path is None and (legacy_labels_path is None or corrected_labels_path is None):
        raise ValueError("raw GT predictions or both compatibility label paths are required")
    raw_gt = _raw_gt_index(raw_predictions_path) if raw_predictions_path is not None else None
    legacy_rows = list(_jsonl(legacy_labels_path)) if raw_gt is None and legacy_labels_path else []
    corrected_rows = list(_jsonl(corrected_labels_path)) if raw_gt is None and corrected_labels_path else []
    feature_rows = list(_jsonl(features_path))
    if raw_gt is None and len(feature_rows) != len(legacy_rows):
        raise ValueError("feature/legacy label length mismatch")
    if raw_gt is None and len(feature_rows) != len(corrected_rows):
        raise ValueError("feature/corrected label length mismatch")
    for index, feature in enumerate(feature_rows):
        feature_sample_id = _stable_source_id(feature)
        if raw_gt is not None:
            sample_id = feature_sample_id
            if sample_id not in raw_gt:
                raise ValueError("raw GT sample mismatch")
            evaluated = _evaluate_frozen_candidates(feature["candidates"], raw_gt[sample_id])
            legacy_correct = evaluated["legacy"]
            corrected_correct = evaluated["corrected"]
        else:
            legacy = legacy_rows[index]
            corrected = corrected_rows[index]
            sample_id = str(legacy["sample_id"])
            if sample_id != str(corrected["sample_id"]) or sample_id != feature_sample_id:
                raise ValueError("Legacy/Corrected/feature sample mismatch")
            legacy_correct = {
                str(item["candidate_id"]): bool(item["candidate_correct"])
                for item in legacy["candidate_labels"]
            }
            corrected_correct = {
                str(item["candidate_id"]): bool(item["candidate_correct"])
                for item in corrected["candidate_labels"]
            }
        if sample_id not in requested:
            continue
        candidates = feature["candidates"]
        if len(candidates) != 5:
            raise ValueError("frozen candidate count is not five")
        candidate_ids = {str(item["candidate_id"]) for item in candidates}
        if len(candidate_ids) != 5:
            raise ValueError("frozen candidate IDs are not unique")
        q_only = str(
            min(
                candidates,
                key=lambda item: (-float(item["q_raw"]), str(item["candidate_id"])),
            )["candidate_id"]
        )
        if set(legacy_correct) != candidate_ids or set(corrected_correct) != candidate_ids:
            raise ValueError("label candidate set changed")
        for prediction in requested[sample_id]:
            selected = _candidate_id(sample_id, str(prediction["selected_stable_candidate_id"]))
            if selected not in candidate_ids:
                raise ValueError("prediction is outside frozen Top-5")
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "method": str(prediction["method"]),
                    "selected_stable_candidate_id": f"{sample_id}/{selected}",
                    "legacy_q_only_correct": legacy_correct[q_only],
                    "legacy_selected_correct": legacy_correct[selected],
                    "corrected_q_only_correct": corrected_correct[q_only],
                    "corrected_selected_correct": corrected_correct[selected],
                    "legacy_oracle": any(legacy_correct.values()),
                    "corrected_oracle": any(corrected_correct.values()),
                }
            )
        matched.add(sample_id)
    missing = set(requested) - matched
    if missing:
        raise ValueError(f"saved predictions contain {len(missing)} unknown samples")

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        by_method[row["method"]].append(row)
    aggregate: dict[str, Any] = {}
    for method, rows in sorted(by_method.items()):
        result: dict[str, Any] = {"total": len(rows)}
        for track in ("legacy", "corrected"):
            q = [bool(row[f"{track}_q_only_correct"]) for row in rows]
            selected = [bool(row[f"{track}_selected_correct"]) for row in rows]
            recovered = sum((not left) and right for left, right in zip(q, selected, strict=True))
            harmful = sum(left and (not right) for left, right in zip(q, selected, strict=True))
            result.update(
                {
                    f"{track}_success_count": sum(selected),
                    f"{track}_j1": sum(selected) / len(rows),
                    f"{track}_recovered": recovered,
                    f"{track}_harmful": harmful,
                    f"{track}_net": recovered - harmful,
                    f"{track}_oracle_success_count": sum(
                        bool(row[f"{track}_oracle"]) for row in rows
                    ),
                    f"{track}_oracle_at_5": sum(
                        bool(row[f"{track}_oracle"]) for row in rows
                    )
                    / len(rows),
                }
            )
        aggregate[method] = result
    return {
        "schema_version": "1.0",
        "kind": "independent_gemini_recompute",
        "ground_truth_source_kind": (
            "raw_gt_grasps_recomputed_with_verified_evaluators"
            if raw_gt is not None
            else "precomputed_label_fallback_for_unit_compatibility"
        ),
        "legacy_evaluator_version": LEGACY_EVALUATOR_VERSION,
        "corrected_evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "input_sha256": {
            "predictions": _sha256_file(predictions_path),
            "features": _sha256_file(features_path),
            **(
                {"raw_predictions_with_gt": _sha256_file(raw_predictions_path)}
                if raw_predictions_path is not None
                else {
                    "legacy_labels": _sha256_file(legacy_labels_path),
                    "corrected_labels": _sha256_file(corrected_labels_path),
                }
            ),
            "evaluator_source": _sha256_file(Path(__file__).resolve().parents[2] / "utils" / "grasp_metrics.py"),
        },
        "prediction_count": len(per_sample),
        "per_sample": per_sample,
        "aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--legacy-labels", required=True)
    parser.add_argument("--corrected-labels", required=True)
    parser.add_argument("--raw-predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = recompute(
        predictions_path=args.predictions,
        features_path=args.features,
        legacy_labels_path=args.legacy_labels,
        corrected_labels_path=args.corrected_labels,
        raw_predictions_path=args.raw_predictions,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "prediction_count": result["prediction_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
