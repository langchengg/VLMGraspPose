"""Build symmetric T3 nearest-candidate consensus and joined feature tracks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verified_manifest_artifact,
)
from unified_reranking.feature_extractors.consensus import BACKENDS
from unified_reranking.feature_tracks import assemble_common_track
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
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test")
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    candidates = {
        route: pd.read_parquet(
            run_dir / "02_candidates" / f"{route}_{args.split}_top5.parquet"
        )
        for route in BACKENDS
    }
    calibrations: dict[str, pd.DataFrame] = {}
    calibration_manifests: dict[str, Path] = {}
    for route in BACKENDS:
        calibration_manifest_path = (
            run_dir / "05_calibration" / f"{route}_calibration_manifest.json"
            if args.split != "test"
            else run_dir / "05_calibration" / f"{route}_test_application_manifest.json"
        )
        calibration_manifest = load_verified_json(
            calibration_manifest_path,
            name=f"{route}/{args.split} calibration manifest",
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
            name=f"{route}/{args.split} calibration artifact",
        )
        calibrations[route] = pd.read_parquet(calibration_path)
        calibration_manifests[route] = calibration_manifest_path
    artifacts: dict[str, object] = {}
    dense_schema_sha256: str | None = None
    for anchor in BACKENDS:
        route_started = time.perf_counter()
        dense_dir = run_dir / "03_features" / "tri_backend_dense" / args.split / anchor
        dense_manifest_path = dense_dir / "feature_manifest.json"
        dense_manifest = load_verified_json(
            dense_manifest_path,
            name=f"{anchor}/{args.split} tri-backend dense manifest",
        )
        if (
            dense_manifest.get("route") != anchor
            or dense_manifest.get("split") != args.split
            or dense_manifest.get("dense_sampling_verified") is not True
            or dense_manifest.get("original_model_original_roundtrip_verified")
            is not True
        ):
            raise RuntimeError(f"{anchor}/{args.split} dense alignment is not verified")
        observed_dense_schema = str(
            dense_manifest.get("model_feature_schema_sha256", "")
        )
        if dense_schema_sha256 is None:
            dense_schema_sha256 = observed_dense_schema
        elif observed_dense_schema != dense_schema_sha256:
            raise RuntimeError("tri-backend dense schema differs across anchor routes")
        dense_path = verified_manifest_artifact(
            dense_manifest, name=f"{anchor}/{args.split} tri-backend dense features"
        )
        dense = pd.read_parquet(dense_path)
        dense_keys = set(
            map(tuple, dense[["sample_id", "candidate_id"]].astype(str).to_numpy())
        )
        candidate_keys = set(
            map(
                tuple,
                candidates[anchor][["sample_id", "candidate_id"]]
                .astype(str)
                .to_numpy(),
            )
        )
        if (
            dense_keys != candidate_keys
            or dense.duplicated(["sample_id", "candidate_id"]).any()
        ):
            raise RuntimeError(
                "tri-backend dense features do not preserve exact candidates"
            )
        for backend in BACKENDS:
            if not any(
                column.startswith(f"dense_{backend}_") for column in dense.columns
            ):
                raise RuntimeError(f"T3 dense features omit {backend} evidence")

        common_dir = run_dir / "03_features" / "common" / f"{anchor}_{args.split}"
        common_manifest_path = common_dir / "feature_manifest.json"
        common_manifest = load_verified_json(
            common_manifest_path, name=f"{anchor}/{args.split} common feature manifest"
        )
        common_path = verified_artifact_path(
            common_manifest["artifacts"]["candidate_features"],
            name=f"{anchor}/{args.split} common features",
        )
        common = pd.read_parquet(common_path)
        rgb_dir = run_dir / "03_features" / "rgb" / f"{anchor}_{args.split}"
        rgb_manifest_path = rgb_dir / "feature_manifest.json"
        rgb_manifest = load_verified_json(
            rgb_manifest_path, name=f"{anchor}/{args.split} RGB feature manifest"
        )
        rgb_path = verified_manifest_artifact(
            rgb_manifest, name=f"{anchor}/{args.split} RGB features"
        )
        common = common.merge(
            pd.read_parquet(rgb_path),
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        if common["rgb_missing"].isna().any():
            raise RuntimeError("T3 RGB evidence does not cover frozen candidates")
        common = common.merge(
            dense,
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        track = assemble_common_track(candidates[anchor], common, calibrations[anchor])
        output = (
            run_dir
            / "03_features"
            / "tracks"
            / "T3_tri_backend"
            / f"{anchor}_{args.split}"
        )
        feature_path = output / "candidate_features.parquet"
        _atomic_parquet(feature_path, track.frame)
        feature_columns = assert_model_feature_columns(track.model_columns)
        track_assembly_latency = (
            (time.perf_counter() - route_started) * 1000.0 / len(track.frame)
        )
        telemetry = telemetry_payload(
            phase="tri_backend_feature_track_assembly",
            parameter_count=None,
            ranker_latency_ms=None,
            feature_latency_ms=track_assembly_latency,
            missing_feature_rate_value=missing_feature_rate(
                track.frame, feature_columns
            ),
            not_applicable_fields=("parameter_count", "ranker_latency_ms"),
        )
        result = {
            "status": "COMPLETE",
            "route": anchor,
            "split": args.split,
            "track": "T3_tri_backend",
            "candidate_rows": len(track.frame),
            "model_feature_columns": list(feature_columns),
            "model_feature_schema_sha256": canonical_sha256(feature_columns),
            "telemetry": telemetry,
            **flatten_telemetry(telemetry),
            "feature_track_assembly_latency_ms": track_assembly_latency,
            "feature_extraction_latency_protocol": (
                "route-specific persisted common+RGB extraction, fixed-128 "
                "tri-backend dense benchmark, and current T3 assembly"
            ),
            "dense_sampling_verified": True,
            "original_model_original_roundtrip_verified": True,
            "dense_feature_manifest": {
                "path": str(dense_manifest_path.resolve()),
                "sha256": sha256_file(dense_manifest_path),
            },
            "dense_feature_artifact": {
                "path": str(dense_path.resolve()),
                "sha256": sha256_file(dense_path),
            },
            "dense_model_feature_schema_sha256": observed_dense_schema,
            "common_feature_manifest": {
                "path": str(common_manifest_path.resolve()),
                "sha256": sha256_file(common_manifest_path),
            },
            "rgb_feature_manifest": {
                "path": str(rgb_manifest_path.resolve()),
                "sha256": sha256_file(rgb_manifest_path),
            },
            "calibration_manifest": {
                "path": str(calibration_manifests[anchor].resolve()),
                "sha256": sha256_file(calibration_manifests[anchor]),
            },
            "artifact": {
                "path": str(feature_path.resolve()),
                "sha256": sha256_file(feature_path),
            },
            "labels_physically_separate": True,
            "candidate_test_labels_read": False if args.split == "test" else None,
        }
        atomic_json(output / "feature_manifest.json", result)
        artifacts[anchor] = result["artifact"]
    summary: dict[str, object] = {
        "status": "COMPLETE",
        "split": args.split,
        "backends": list(BACKENDS),
        "artifacts": artifacts,
        "dense_sampling_verified": True,
        "dense_model_feature_schema_sha256": dense_schema_sha256,
        "candidate_test_labels_read": False if args.split == "test" else None,
    }
    summary_path = run_dir / "03_features" / "consensus" / f"{args.split}_manifest.json"
    atomic_json(summary_path, summary)
    if args.split == "test":
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": "tri_backend_consensus_test",
                "inputs": ["verified_dense", "verified_common", "verified_rgb"],
                "output_manifest": str(summary_path.resolve()),
                "output_manifest_sha256": sha256_file(summary_path),
                "candidate_labels_opened_as_table": False,
            },
        )
    return summary


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P3_P4",
        substage=f"tri_backend_consensus_{args.split}",
        evidence_track="T3_tri_backend",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(args)
        artifact = run_dir / "03_features" / "consensus" / f"{args.split}_manifest.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
