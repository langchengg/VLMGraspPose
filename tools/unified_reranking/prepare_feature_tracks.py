"""Materialize leakage-safe T1/T2 development feature tables."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.feature_tracks import (
    assemble_common_track,
    assemble_crog_native_track,
    select_crog_native_reference_columns,
)
from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verified_manifest_artifact,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log
from unified_reranking.telemetry import (
    flatten_telemetry,
    missing_feature_rate,
    telemetry_payload,
)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    parser.add_argument(
        "--track", required=True, choices=("T1_native", "T2_matched_common")
    )
    parser.add_argument(
        "--historical-crog-run",
        type=Path,
        default=ROOT / "runs" / "reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    candidate_path = (
        run_dir / "02_candidates" / f"{args.route}_{args.split}_top5.parquet"
    )
    manifest_path = run_dir / "01_manifests" / f"paired_{args.split}.parquet"
    calibration_manifest_path = (
        run_dir / "05_calibration" / f"{args.route}_calibration_manifest.json"
        if args.split != "test"
        else run_dir / "05_calibration" / f"{args.route}_test_application_manifest.json"
    )
    calibration_manifest = load_verified_json(
        calibration_manifest_path,
        name=f"{args.route}/{args.split} calibration manifest",
        statuses=("COMPLETE", "COMPLETE_LABEL_FREE"),
    )
    calibration_key = (
        "train_oof"
        if args.split == "train"
        else "validation"
        if args.split == "validation"
        else "artifact"
    )
    calibration_path = verified_manifest_artifact(
        calibration_manifest,
        key=calibration_key,
        name=f"{args.route}/{args.split} calibration artifact",
    )
    candidates = pd.read_parquet(candidate_path)
    calibration = pd.read_parquet(calibration_path)

    sources = [
        candidate_path,
        manifest_path,
        calibration_manifest_path,
        calibration_path,
        Path(__file__),
    ]
    extraction_components: dict[str, float] = {}
    if args.track == "T2_matched_common" or args.route != "crog":
        common_dir = run_dir / "03_features" / "common" / f"{args.route}_{args.split}"
        common_manifest_path = common_dir / "feature_manifest.json"
        common_manifest = load_verified_json(
            common_manifest_path,
            name=f"{args.route}/{args.split} common feature manifest",
        )
        common_path = verified_artifact_path(
            common_manifest["artifacts"]["candidate_features"],
            name=f"{args.route}/{args.split} common candidate features",
        )
        common_features = pd.read_parquet(common_path)
        extraction_components["common"] = common_manifest.get(
            "feature_extraction_latency_ms"
        )
        sources.append(common_manifest_path)
        if args.track == "T2_matched_common" or (
            args.track == "T1_native" and args.route == "g1"
        ):
            rgb_dir = run_dir / "03_features" / "rgb" / f"{args.route}_{args.split}"
            rgb_manifest_path = rgb_dir / "feature_manifest.json"
            rgb_manifest = load_verified_json(
                rgb_manifest_path,
                name=f"{args.route}/{args.split} RGB feature manifest",
            )
            rgb_path = verified_manifest_artifact(
                rgb_manifest,
                name=f"{args.route}/{args.split} RGB candidate features",
            )
            rgb = pd.read_parquet(rgb_path)
            extraction_components["rgb"] = rgb_manifest.get(
                "feature_extraction_latency_ms"
            )
            common_features = common_features.merge(
                rgb,
                on=["sample_id", "candidate_id"],
                how="left",
                validate="one_to_one",
            )
            if (
                len(common_features) != len(candidates)
                or common_features["rgb_missing"].isna().any()
            ):
                raise RuntimeError("RGB evidence does not cover frozen candidates")
            sources.append(rgb_path)
            sources.append(rgb_manifest_path)
        if args.track == "T1_native" and args.route != "crog":
            backend_dir = (
                run_dir / "03_features" / "backend_maps" / f"{args.route}_{args.split}"
            )
            backend_manifest_path = backend_dir / "feature_manifest.json"
            backend_manifest = load_verified_json(
                backend_manifest_path,
                name=f"{args.route}/{args.split} backend-map feature manifest",
            )
            backend_path = verified_manifest_artifact(
                backend_manifest,
                name=f"{args.route}/{args.split} backend-map candidate features",
            )
            backend = pd.read_parquet(backend_path)
            extraction_components["backend_maps"] = backend_manifest.get(
                "feature_extraction_latency_ms"
            )
            common_features = common_features.merge(
                backend,
                on=["sample_id", "candidate_id"],
                how="left",
                validate="one_to_one",
            )
            if (
                len(common_features) != len(candidates)
                or common_features["backend_map_missing"].isna().any()
            ):
                raise RuntimeError(
                    "backend-map evidence does not cover frozen candidates"
                )
            sources.append(backend_path)
            sources.append(backend_manifest_path)
        track = assemble_common_track(candidates, common_features, calibration)
        sources.append(common_path)
        evidence_note = (
            "matched common HiFi-CS probability/mask and RGB-D evidence"
            if args.track == "T2_matched_common"
            else "route-native G1/C1 evidence; dense backend features are joined separately"
        )
    else:
        historical_split = "val" if args.split == "validation" else args.split
        reference_path = (
            args.historical_crog_run.resolve()
            / "features"
            / f"candidates_crog_frozen_top5_{historical_split}.parquet"
        )
        parquet_file = pq.ParquetFile(reference_path)
        schema = parquet_file.read_row_group(0).slice(0, 1).to_pandas()
        selected = select_crog_native_reference_columns(schema)
        required = {
            "frame_id",
            "language_instruction",
            "candidate_id",
            "original_rank",
            "q_raw",
            "x_px",
            "y_px",
            "angle_rad",
            "width_px",
            "height_px",
            *selected,
        }
        reference = pd.read_parquet(reference_path, columns=sorted(required))
        track = assemble_crog_native_track(
            candidates,
            pd.read_parquet(manifest_path),
            reference,
            calibration,
        )
        sources.append(reference_path)
        evidence_note = "CROG native RGB/language/mask/dense-map evidence; depth-derived columns excluded"

    output = (
        run_dir / "03_features" / "tracks" / args.track / f"{args.route}_{args.split}"
    )
    feature_path = output / "candidate_features.parquet"
    _atomic_parquet(feature_path, track.frame)
    if len(track.frame) <= 0:
        raise RuntimeError("feature track contains no candidates")
    telemetry = telemetry_payload(
        phase="feature_track_assembly",
        parameter_count=None,
        ranker_latency_ms=None,
        feature_latency_ms=(time.perf_counter() - started) * 1000.0 / len(track.frame),
        missing_feature_rate_value=missing_feature_rate(
            track.frame, track.model_columns
        ),
        not_applicable_fields=("parameter_count", "ranker_latency_ms"),
    )
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in extraction_components.values()
    ):
        raise RuntimeError("feature track source extraction telemetry is incomplete")
    track_assembly_latency = float(telemetry["feature_latency_ms"])
    feature_extraction_latency = track_assembly_latency + sum(
        map(float, extraction_components.values())
    )
    result: dict[str, object] = {
        "status": "COMPLETE",
        "route": args.route,
        "split": args.split,
        "track": args.track,
        "evidence_note": evidence_note,
        "candidate_rows": len(track.frame),
        "sample_count_with_candidates": int(track.frame["sample_id"].nunique()),
        "model_feature_count": len(track.model_columns),
        "model_feature_columns": list(track.model_columns),
        "model_feature_schema_sha256": canonical_sha256(track.model_columns),
        "labels_physically_separate": True,
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "feature_extraction_latency_ms": feature_extraction_latency,
        "feature_track_assembly_latency_ms": track_assembly_latency,
        "feature_extraction_component_latency_ms": extraction_components,
        "feature_extraction_latency_protocol": (
            "hash-bound source extraction latencies plus current track assembly wall time"
        ),
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sources
        ],
        "artifact": {
            "path": str(feature_path.resolve()),
            "sha256": sha256_file(feature_path),
        },
    }
    atomic_json(output / "feature_manifest.json", result)
    if args.split == "test":
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": f"feature_track_{args.track}_{args.route}_test",
                "inputs": ["candidate_geometry", "label_free_feature_manifests"],
                "output_manifest": str((output / "feature_manifest.json").resolve()),
                "output_manifest_sha256": sha256_file(output / "feature_manifest.json"),
                "candidate_labels_opened_as_table": False,
            },
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    substage = f"feature_track_{args.track}_{args.route}_{args.split}"
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P3_P4",
        substage=substage,
        route=args.route,
        evidence_track=args.track,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(args)
        artifact = (
            run_dir
            / "03_features"
            / "tracks"
            / args.track
            / f"{args.route}_{args.split}"
            / "feature_manifest.json"
        )
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
