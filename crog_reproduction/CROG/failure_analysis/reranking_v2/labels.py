from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from utils.grasp_metrics import (
    CORRECTED_EVALUATOR_VERSION,
    LEGACY_EVALUATOR_VERSION,
    evaluate_candidate,
)

from failure_analysis.reranking.geometry import assert_candidate_set_unchanged

from .schema import (
    append_jsonl_record,
    artifact_identity,
    atomic_write_json,
    code_fingerprint,
    read_jsonl,
    stable_candidate_id,
    stable_sample_id,
)


EVALUATOR_TRACKS = {
    "legacy_official": LEGACY_EVALUATOR_VERSION,
    "corrected": CORRECTED_EVALUATOR_VERSION,
}


def build_label_record(
    feature_record: dict[str, Any],
    prediction_record: dict[str, Any],
    *,
    evaluator_track: str,
) -> dict[str, Any]:
    if evaluator_track not in EVALUATOR_TRACKS:
        raise ValueError(f"unknown evaluator track: {evaluator_track}")
    if str(feature_record["sample_id"]) != str(prediction_record["sample_id"]):
        raise ValueError("feature/prediction sample order mismatch")
    split = str(feature_record["split"])
    sample_id = stable_sample_id(split, feature_record["sample_id"])
    candidates = feature_record["candidates"]
    prediction_candidates = prediction_record.get("candidates", candidates)
    assert_candidate_set_unchanged(candidates, prediction_candidates)
    gt_grasps = prediction_record.get("gt_grasps")
    if gt_grasps is None:
        raise ValueError(f"prediction record lacks GT grasps for {sample_id}")
    version = EVALUATOR_TRACKS[evaluator_track]
    labels = []
    for candidate in candidates:
        evaluation = evaluate_candidate(
            candidate,
            gt_grasps,
            evaluator_version=version,
            iou_threshold=0.25,
            angle_threshold=30.0,
        )
        labels.append(
            {
                "candidate_id": candidate["candidate_id"],
                "stable_candidate_id": stable_candidate_id(
                    sample_id, candidate["candidate_id"]
                ),
                "candidate_checksum": candidate["candidate_checksum"],
                "candidate_correct": bool(evaluation["candidate_success"]),
                "failure_mode": evaluation["failure_mode"],
                "best_gt": evaluation["best_gt"],
                "pairwise": evaluation["pairwise"],
            }
        )
    return {
        "schema_version": "2.0.0",
        "kind": "candidate_labels",
        "evaluator_track": evaluator_track,
        "evaluator_version": version,
        "sample_id": sample_id,
        "source_sample_id": int(feature_record["sample_id"]),
        "split": split,
        "frame_id": feature_record["scene_id"],
        "sequence_id": str(feature_record["scene_id"]).split(",", 1)[0],
        "candidate_labels": labels,
        "candidate_count": len(labels),
        "original_top1_correct": bool(
            labels and labels[0]["candidate_correct"]
        ),
        "oracle_at_5": bool(
            any(item["candidate_correct"] for item in labels)
        ),
    }


def build_dual_labels(
    features_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        track: output_dir / track / "labels.jsonl"
        for track in EVALUATOR_TRACKS
    }
    completed_summary = output_dir / "summary.json"
    if resume and completed_summary.exists():
        summary = json.loads(
            completed_summary.read_text(encoding="utf-8")
        )
        if summary.get("status") != "complete":
            raise ValueError("label artifact summary is not complete")
        current_inputs = [
            artifact_identity(features_path),
            artifact_identity(predictions_path),
        ]
        if [
            identity["sha256"] for identity in current_inputs
        ] != [
            identity["sha256"] for identity in summary["inputs"]
        ]:
            raise ValueError("label artifact resume input hash mismatch")
        for track, path in paths.items():
            identity = summary["tracks"][track]
            if (
                not path.exists()
                or artifact_identity(path)["sha256"] != identity["sha256"]
            ):
                raise ValueError(f"completed label artifact changed: {track}")
        return paths
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"label output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        track: path.with_name(f".{path.name}.tmp-{os.getpid()}")
        for track, path in paths.items()
    }
    handles = {
        track: path.open("x", encoding="utf-8")
        for track, path in temporary_paths.items()
    }
    counts = {track: {"samples": 0, "top1": 0, "oracle": 0} for track in paths}
    sample_ids = set()
    succeeded = False
    try:
        feature_iter = read_jsonl(features_path)
        prediction_iter = read_jsonl(predictions_path)
        for index, (feature, prediction) in enumerate(
            zip(feature_iter, prediction_iter, strict=True)
        ):
            for track, handle in handles.items():
                record = build_label_record(
                    feature, prediction, evaluator_track=track
                )
                if track == "legacy_official":
                    if record["sample_id"] in sample_ids:
                        raise ValueError(
                            f"duplicate stable sample ID: {record['sample_id']}"
                        )
                    sample_ids.add(record["sample_id"])
                append_jsonl_record(handle, record)
                counts[track]["samples"] += 1
                counts[track]["top1"] += int(record["original_top1_correct"])
                counts[track]["oracle"] += int(record["oracle_at_5"])
        succeeded = True
    finally:
        for handle in handles.values():
            if succeeded:
                handle.flush()
                os.fsync(handle.fileno())
            handle.close()
        if not succeeded:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)
    for track, temporary_path in temporary_paths.items():
        os.replace(temporary_path, paths[track])
    input_identities = [
        artifact_identity(features_path),
        artifact_identity(predictions_path),
    ]
    track_identities = {
        track: artifact_identity(path) for track, path in paths.items()
    }
    source_code_sha256 = code_fingerprint(
        Path(__file__).resolve().parents[2]
    )
    for track, path in paths.items():
        atomic_write_json(
            path.parent / "summary.json",
            {
                "schema_version": "2.0.0",
                "kind": "candidate_label_artifact",
                "status": "complete",
                "evaluator_track": track,
                "evaluator_version": EVALUATOR_TRACKS[track],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "inputs": input_identities,
                "output": track_identities[track],
                "code_sha256": source_code_sha256,
                "unique_id_count": len(sample_ids),
                **counts[track],
            },
        )
    atomic_write_json(
        completed_summary,
        {
            "schema_version": "2.0.0",
            "kind": "dual_candidate_label_artifact",
            "status": "complete",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "inputs": input_identities,
            "tracks": track_identities,
            "code_sha256": source_code_sha256,
            "sample_count": len(sample_ids),
            "unique_id_count": len(sample_ids),
        },
    )
    return paths


def label_lookup(record: dict[str, Any]) -> dict[str, bool]:
    return {
        str(item["candidate_id"]): bool(item["candidate_correct"])
        for item in record["candidate_labels"]
    }


def validate_inference_label_join(
    feature_record: dict[str, Any], label_record: dict[str, Any]
) -> None:
    expected = stable_sample_id(feature_record["split"], feature_record["sample_id"])
    if expected != label_record["sample_id"]:
        raise ValueError(f"sample join mismatch: {expected} != {label_record['sample_id']}")
    features = {
        str(item["candidate_id"]): str(item["candidate_checksum"])
        for item in feature_record["candidates"]
    }
    labels = {
        str(item["candidate_id"]): str(item["candidate_checksum"])
        for item in label_record["candidate_labels"]
    }
    if features != labels:
        raise ValueError(f"candidate checksum join mismatch for {expected}")
