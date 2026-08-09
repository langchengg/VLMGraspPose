"""Build the no-dedup, route-qualified Top-15 union feature track."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.artifacts import load_verified_json, verified_manifest_artifact
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log


ROUTES = ("crog", "g1", "c1")
ROUTE_TIE_ORDER = ("crog", "g1", "c1")
TRACK = "T5_cross_route"
SOURCE_TRACK = "T2_matched_common"
ROUTE_FEATURES = ("union_route_crog", "union_route_g1", "union_route_c1")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _load_route(run_dir: Path, route: str, split: str) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    feature_dir = (
        run_dir / "03_features" / "tracks" / SOURCE_TRACK / f"{route}_{split}"
    )
    manifest_path = feature_dir / "feature_manifest.json"
    manifest = load_verified_json(
        manifest_path, name=f"{route}/{split} T2 source feature manifest"
    )
    feature_path = verified_manifest_artifact(
        manifest, name=f"{route}/{split} T2 source features"
    )
    columns = assert_model_feature_columns(manifest["model_feature_columns"])
    features = pd.read_parquet(feature_path)
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "base_logit",
        "calibrated_native_probability",
        *columns,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"{route}/{split} source features miss columns: {missing}")
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_{split}_top5.parquet",
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_geometry_sha256",
        ],
    )
    source = features.merge(
        candidates,
        on=["sample_id", "candidate_id", "native_rank"],
        how="inner",
        validate="one_to_one",
    )
    if len(source) != len(features) or len(source) != len(candidates):
        raise RuntimeError(f"{route}/{split} source features do not preserve candidates")
    probability = pd.to_numeric(
        source["calibrated_native_probability"], errors="coerce"
    ).to_numpy(float)
    base_logit = pd.to_numeric(source["base_logit"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(base_logit).all()
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise ValueError(f"{route}/{split} route calibration is invalid")
    audit = {
        "feature_manifest": _record(manifest_path),
        "features": _record(feature_path),
        "candidates": _record(run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"),
        "source_model_feature_columns": list(columns),
        "route_specific_calibrated_probability": True,
    }
    return source, columns, audit


def build_union_split(
    run_dir: Path,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, tuple[str, ...], dict[str, Any]]:
    """Build one route-qualified union; Test remains strictly label-free."""

    split = str(split).lower()
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    frames: list[pd.DataFrame] = []
    source_columns: tuple[str, ...] | None = None
    audits: dict[str, Any] = {}
    label_frames: list[pd.DataFrame] = []
    for route_index, route in enumerate(ROUTE_TIE_ORDER):
        source, columns, audit = _load_route(run_dir, route, split)
        if source_columns is None:
            source_columns = columns
        elif columns != source_columns:
            raise RuntimeError("T2 route feature schemas differ; union comparison is not fair")
        work = source.copy()
        work["source_route"] = route.upper()
        work["source_candidate_id"] = work["candidate_id"].astype(str)
        work["candidate_id"] = route.upper() + ":" + work["source_candidate_id"]
        work["route_native_rank"] = work["native_rank"].astype(int)
        work["native_rank"] = (
            (work["route_native_rank"] - 1) * len(ROUTE_TIE_ORDER) + route_index + 1
        )
        for encoded_route in ROUTES:
            work[f"union_route_{encoded_route}"] = float(encoded_route == route)
        frames.append(work)
        audits[route] = audit
        if split != "test":
            label_path = run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
            labels = pd.read_parquet(
                label_path,
                columns=["sample_id", "candidate_id", "candidate_success", "jacquard_margin"],
            )
            labels["candidate_id"] = route.upper() + ":" + labels["candidate_id"].astype(str)
            label_frames.append(labels)
            audits[route]["labels"] = _record(label_path)
    if source_columns is None:
        raise RuntimeError("union source track is empty")
    features = pd.concat(frames, ignore_index=True)
    keys = ["sample_id", "candidate_id"]
    if features[keys].isna().any().any() or features.duplicated(keys).any():
        raise RuntimeError("route-qualified union candidate identities are invalid")
    counts = features.groupby("sample_id", sort=False).size()
    if counts.empty or int(counts.max()) > 15:
        raise RuntimeError("union violates the Top-15 contract")
    # Exact geometry duplicates are retained by route-qualified identity.
    model_columns = assert_model_feature_columns((*source_columns, *ROUTE_FEATURES))
    matrix = features.loc[:, model_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(matrix).all():
        # Fold preprocessing can impute missing source features, but route calibration
        # and the explicit route identity must never be missing.
        calibration = features[["base_logit", "calibrated_native_probability", *ROUTE_FEATURES]].to_numpy(float)
        if not np.isfinite(calibration).all():
            raise RuntimeError("union calibration/route identity contains non-finite values")
    labels_output: pd.DataFrame | None = None
    if split != "test":
        labels_output = pd.concat(label_frames, ignore_index=True)
        if labels_output[keys].isna().any().any() or labels_output.duplicated(keys).any():
            raise RuntimeError("union development labels have invalid keys")
        if set(map(tuple, labels_output[keys].astype(str).to_numpy())) != set(
            map(tuple, features[keys].astype(str).to_numpy())
        ):
            raise RuntimeError("union development labels do not exactly cover candidates")
    else:
        forbidden = {
            "candidate_success",
            "jacquard_margin",
            "matched_gt_index",
            "best_same_gt_iou",
        }.intersection(features.columns)
        if forbidden:
            raise PermissionError(f"union Test features contain supervision: {sorted(forbidden)}")
    audit = {
        "split": split,
        "source_track": SOURCE_TRACK,
        "primary_union_deduplication": "NONE",
        "route_tie_order": [route.upper() for route in ROUTE_TIE_ORDER],
        "route_sources": audits,
        "candidate_rows": int(len(features)),
        "candidate_bearing_samples": int(features["sample_id"].nunique()),
        "maximum_candidates_per_sample": int(counts.max()),
        "geometry_duplicate_rows_retained": int(
            features.duplicated(["sample_id", "candidate_geometry_sha256"], keep=False).sum()
        ),
        "model_feature_columns": list(model_columns),
        "feature_schema_sha256": canonical_sha256(model_columns),
        "candidate_test_labels_read": False if split == "test" else None,
    }
    return features, labels_output, model_columns, audit


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE" or value.get("signature_sha256") != signature:
        raise RuntimeError("immutable union-feature output exists with a different signature")
    for record in value.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("resumable union-feature artifact hash mismatch")
    return value


def run_split(run_dir: Path, split: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    features, labels, columns, audit = build_union_split(run_dir, split)
    sources: dict[str, Any] = {"implementation_tool": _record(Path(__file__))}
    for route, values in audit["route_sources"].items():
        for name, record in values.items():
            if isinstance(record, dict) and "sha256" in record:
                sources[f"{route}_{name}"] = record
    configuration = {
        "split": split,
        "track": TRACK,
        "source_track": SOURCE_TRACK,
        "primary_union_deduplication": "NONE",
        "route_tie_order": [route.upper() for route in ROUTE_TIE_ORDER],
        "maximum_candidates": 15,
        "model_feature_columns": list(columns),
        "candidate_test_labels_read": False if split == "test" else None,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = run_dir / "03_features" / "tracks" / TRACK / f"union_{split}"
    marker = output / "feature_manifest.json"
    resumed = _resume(marker, signature)
    if resumed is not None:
        return resumed
    feature_path = output / "candidate_features.parquet"
    _atomic_parquet(feature_path, features)
    artifacts: dict[str, Any] = {"features": _record(feature_path)}
    if labels is not None:
        label_path = output / "candidate_labels.parquet"
        _atomic_parquet(label_path, labels)
        artifacts["labels"] = _record(label_path)
    manifest = {
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "model_feature_columns": list(columns),
        "feature_schema_sha256": canonical_sha256(columns),
        "audit": audit,
        "sources": sources,
        "artifacts": artifacts,
        "test_access": "LABEL_FREE_FEATURES_ONLY" if split == "test" else "NONE",
        "candidate_test_labels_read": False if split == "test" else None,
    }
    atomic_json(marker, manifest)
    if split == "test":
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": "union_test_feature_preparation",
                "output_manifest": str(marker.resolve()),
                "output_manifest_sha256": sha256_file(marker),
                "candidate_labels_opened_as_table": False,
            },
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
    for split in splits:
        with ledger_stage(
            run_dir / "run_ledger.sqlite",
            stage="P9" if split != "test" else "P11_PRELOCK",
            substage=f"prepare_union_features_{split}",
            route="cross_route",
            evidence_track=TRACK,
            pool="primary_union_top15_no_dedup",
            method="route_specific_calibrated_features",
            command=" ".join(map(str, sys.argv)),
        ) as state:
            run_split(run_dir, split)
            marker = run_dir / "03_features" / "tracks" / TRACK / f"union_{split}" / "feature_manifest.json"
            state["artifact_path"] = str(marker.resolve())
            state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROUTE_FEATURES",
    "ROUTE_TIE_ORDER",
    "SOURCE_TRACK",
    "TRACK",
    "build_union_split",
    "run_split",
]
