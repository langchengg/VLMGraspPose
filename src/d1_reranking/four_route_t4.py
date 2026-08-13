"""Label-free D1 Top5 T4 consensus features against frozen CROG/G1/C1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.feature_extractors.common import (
    _continuous_iou,
    _corners,
    periodic_angle_error,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest
from .io import atomic_parquet


PEER_ROUTES = ("CROG", "G1", "C1")
GEOMETRY_COLUMNS = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"P12 T4 source is not regular: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _read_candidates(path: Path, *, route: str, split: str) -> pd.DataFrame:
    if split == "test":
        assert_label_free_parquet_schema(path, name=f"P12 T4 {route} Test Top5")
    frame = pd.read_parquet(path)
    score_column = "native_score_raw" if "native_score_raw" in frame else "native_score"
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        score_column,
        "candidate_geometry_sha256",
        *GEOMETRY_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"P12 T4 {route} Top5 misses columns: {missing}")
    work = frame.loc[:, list(required)].rename(
        columns={score_column: "native_score_raw"}
    )
    # The completed three-route run predates the D1 full-pose identity column.
    # Derive an extension-scoped identity from the immutable route-local raw ID
    # and its frozen geometry hash.  This is deliberately not written back to
    # or represented as the original three-route candidate identity.
    if "candidate_identity_sha256" in frame.columns:
        work["candidate_identity_sha256"] = frame[
            "candidate_identity_sha256"
        ].astype(str)
    else:
        work["candidate_identity_sha256"] = [
            canonical_sha256(
                {
                    "identity_contract": "p12_extension_route_raw_id_geometry_v1",
                    "route": route,
                    "sample_id": str(sample_id),
                    "candidate_id": str(candidate_id),
                    "candidate_geometry_sha256": str(geometry),
                }
            )
            for sample_id, candidate_id, geometry in work[
                ["sample_id", "candidate_id", "candidate_geometry_sha256"]
            ].itertuples(index=False, name=None)
        ]
    work[["sample_id", "candidate_id"]] = work[["sample_id", "candidate_id"]].astype(
        str
    )
    if work.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError(f"P12 T4 {route} has duplicate candidate IDs")
    rank = pd.to_numeric(work["native_rank"], errors="coerce")
    numeric = work[["native_score_raw", *GEOMETRY_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        rank.isna().any()
        or not np.equal(rank, np.floor(rank)).all()
        or ((rank < 1) | (rank > 5)).any()
        or not np.isfinite(numeric.to_numpy(float)).all()
        or (numeric[["width_px", "height_px"]] <= 0).any().any()
    ):
        raise ValueError(f"P12 T4 {route} numeric candidate contract differs")
    work["native_rank"] = rank.astype(int)
    work[["native_score_raw", *GEOMETRY_COLUMNS]] = numeric
    return work


def _corners_for(row: Mapping[str, Any]) -> np.ndarray:
    return _corners(
        np.asarray([float(row["cx_px"]), float(row["cy_px"])]),
        float(row["theta_deg"]),
        float(row["width_px"]),
        float(row["height_px"]),
    )


def _nearest(anchor: Mapping[str, Any], peers: pd.DataFrame) -> pd.Series:
    distance = np.hypot(
        peers["cx_px"].to_numpy(float) - float(anchor["cx_px"]),
        peers["cy_px"].to_numpy(float) - float(anchor["cy_px"]),
    )
    order = np.lexsort(
        (
            peers["candidate_id"].astype(str).to_numpy(),
            peers["native_rank"].to_numpy(int),
            distance,
        )
    )
    return peers.iloc[int(order[0])]


def _build_consensus_frame(
    d1: pd.DataFrame, peers: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Construct candidate-aligned consensus without accepting supervision."""

    if set(peers) != set(PEER_ROUTES):
        raise ValueError("P12 T4 peers must contain frozen CROG/G1/C1")
    grouped = {
        route: {
            str(sample): group
            for sample, group in frame.groupby("sample_id", sort=False)
        }
        for route, frame in peers.items()
    }
    d1_grouped = {
        str(sample): group for sample, group in d1.groupby("sample_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for anchor in d1.to_dict("records"):
        sample_id = str(anchor["sample_id"])
        output: dict[str, Any] = {
            "sample_id": sample_id,
            "candidate_id": str(anchor["candidate_id"]),
            "native_rank": int(anchor["native_rank"]),
            "native_score_raw": float(anchor["native_score_raw"]),
            "candidate_identity_sha256": str(anchor["candidate_identity_sha256"]),
            "candidate_geometry_sha256": str(anchor["candidate_geometry_sha256"]),
            **{column: float(anchor[column]) for column in GEOMETRY_COLUMNS},
        }
        agreements: list[float] = []
        disagreements: list[float] = []
        anchor_corners = _corners_for(anchor)
        for route in PEER_ROUTES:
            prefix = f"t4_{route.lower()}"
            group = grouped[route].get(sample_id)
            if group is None or group.empty:
                output.update(
                    {
                        f"{prefix}_candidate_exists": 0.0,
                        f"{prefix}_nearest_candidate_id": "",
                        f"{prefix}_nearest_cx_px": 0.0,
                        f"{prefix}_nearest_cy_px": 0.0,
                        f"{prefix}_delta_x_px": 0.0,
                        f"{prefix}_delta_y_px": 0.0,
                        f"{prefix}_center_distance_px": 0.0,
                        f"{prefix}_periodic_angle_error_deg": 90.0,
                        f"{prefix}_cos_2_angle": -1.0,
                        f"{prefix}_sin_2_angle": 0.0,
                        f"{prefix}_log_width_ratio": 0.0,
                        f"{prefix}_rotated_iou": 0.0,
                        f"{prefix}_native_score": 0.0,
                        f"{prefix}_native_rank": 0.0,
                        f"{prefix}_mutual_nearest": 0.0,
                        f"{prefix}_agreement": 0.0,
                        f"{prefix}_disagreement": 1.0,
                    }
                )
                agreements.append(0.0)
                disagreements.append(1.0)
                continue
            nearest = _nearest(anchor, group)
            dx = float(nearest["cx_px"] - anchor["cx_px"])
            dy = float(nearest["cy_px"] - anchor["cy_px"])
            distance = float(math.hypot(dx, dy))
            angle = periodic_angle_error(anchor["theta_deg"], nearest["theta_deg"])
            directed = float(anchor["theta_deg"] - nearest["theta_deg"])
            width_ratio = math.log(
                max(float(anchor["width_px"]), 1e-6)
                / max(float(nearest["width_px"]), 1e-6)
            )
            iou = _continuous_iou(anchor_corners, _corners_for(nearest))
            reverse = _nearest(nearest, d1_grouped[sample_id])
            mutual = str(reverse["candidate_id"]) == str(anchor["candidate_id"])
            scale = max(float(anchor["width_px"]), float(nearest["width_px"]), 1.0)
            agreement = float(
                math.exp(-distance / scale)
                * max(0.0, math.cos(math.radians(2.0 * angle)))
                * math.exp(-abs(width_ratio))
                * math.sqrt(max(iou, 0.0))
            )
            disagreement = 1.0 - agreement
            output.update(
                {
                    f"{prefix}_candidate_exists": 1.0,
                    f"{prefix}_nearest_candidate_id": str(nearest["candidate_id"]),
                    f"{prefix}_nearest_cx_px": float(nearest["cx_px"]),
                    f"{prefix}_nearest_cy_px": float(nearest["cy_px"]),
                    f"{prefix}_delta_x_px": dx,
                    f"{prefix}_delta_y_px": dy,
                    f"{prefix}_center_distance_px": distance,
                    f"{prefix}_periodic_angle_error_deg": angle,
                    f"{prefix}_cos_2_angle": math.cos(math.radians(2.0 * angle)),
                    f"{prefix}_sin_2_angle": math.sin(math.radians(2.0 * directed)),
                    f"{prefix}_log_width_ratio": width_ratio,
                    f"{prefix}_rotated_iou": iou,
                    f"{prefix}_native_score": float(nearest["native_score_raw"]),
                    f"{prefix}_native_rank": float(nearest["native_rank"]),
                    f"{prefix}_mutual_nearest": float(mutual),
                    f"{prefix}_agreement": agreement,
                    f"{prefix}_disagreement": disagreement,
                }
            )
            agreements.append(agreement)
            disagreements.append(disagreement)
        output["t4_agreement_mean"] = float(np.mean(agreements))
        output["t4_agreement_min"] = float(np.min(agreements))
        output["t4_disagreement_mean"] = float(np.mean(disagreements))
        output["t4_disagreement_max"] = float(np.max(disagreements))
        output["t4_peer_exists_count"] = sum(
            output[f"t4_{route.lower()}_candidate_exists"] for route in PEER_ROUTES
        )
        rows.append(output)
    result = pd.DataFrame(rows)
    if result.duplicated(["sample_id", "candidate_id"]).any() or len(result) != len(d1):
        raise RuntimeError("P12 T4 does not preserve D1 Top5 candidate identity")
    identity = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score_raw",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
        *GEOMETRY_COLUMNS,
        *{f"t4_{route.lower()}_nearest_candidate_id" for route in PEER_ROUTES},
    }
    model_columns = tuple(column for column in result.columns if column not in identity)
    if not np.isfinite(result.loc[:, model_columns].to_numpy(float)).all():
        raise RuntimeError("P12 T4 model features contain non-finite values")
    return result, model_columns


def build_t4_frame(
    d1: pd.DataFrame,
    peers: Mapping[str, pd.DataFrame],
    t3: pd.DataFrame,
    t3_model_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Widen exact T3 evidence with stable, candidate-aligned T4 consensus."""

    t3_columns = assert_model_feature_columns(tuple(map(str, t3_model_columns)))
    if not t3_columns:
        raise ValueError("P12 T4 widening requires a non-empty T3 model schema")
    keys = ["sample_id", "candidate_id"]
    frozen = [
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
        *GEOMETRY_COLUMNS,
    ]
    # T3 feature artifacts preserve candidate keys and numeric model evidence,
    # but the current D1 T3 contract does not duplicate the two SHA identity
    # columns.  Those hashes remain authoritative in the canonical Top5 pool
    # and are injected below after exact key coverage is established.
    required_t3 = [
        *keys,
        "native_rank",
        *GEOMETRY_COLUMNS,
        *t3_columns,
    ]
    missing = sorted(set(required_t3).difference(t3.columns))
    if missing:
        raise ValueError(f"P12 T3 widening source misses columns: {missing}")
    wide = t3.copy()
    wide[keys] = wide[keys].astype(str)
    if wide.duplicated(keys).any():
        raise ValueError("P12 T3 widening source has duplicate candidate keys")
    expected = d1.loc[:, [*keys, "native_score_raw", *frozen]].copy()
    expected[keys] = expected[keys].astype(str)
    observed_keys = set(map(tuple, wide[keys].to_numpy()))
    expected_keys = set(map(tuple, expected[keys].to_numpy()))
    if len(wide) != len(expected) or observed_keys != expected_keys:
        raise RuntimeError("P12 T3/T4 candidate membership differs")
    for column in ("candidate_identity_sha256", "candidate_geometry_sha256"):
        if column not in wide.columns:
            wide = wide.merge(
                expected.loc[:, [*keys, column]],
                on=keys,
                validate="one_to_one",
                sort=False,
            )
    aligned = wide.merge(
        expected.rename(
            columns={
                column: f"_frozen_{column}" for column in ["native_score_raw", *frozen]
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    for column in frozen:
        frozen_column = f"_frozen_{column}"
        if column in {"native_rank", *GEOMETRY_COLUMNS}:
            equal = np.array_equal(
                pd.to_numeric(aligned[column]).to_numpy(float),
                pd.to_numeric(aligned[frozen_column]).to_numpy(float),
            )
        else:
            equal = (
                aligned[column].astype(str).equals(aligned[frozen_column].astype(str))
            )
        if not equal:
            raise RuntimeError(f"P12 T3/T4 frozen {column} differs")
    score_column = (
        "native_score_raw" if "native_score_raw" in aligned else "native_score"
    )
    if score_column not in aligned:
        raise ValueError("P12 T3 widening source misses native score")
    if not np.array_equal(
        pd.to_numeric(aligned[score_column]).to_numpy(float),
        pd.to_numeric(aligned["_frozen_native_score_raw"]).to_numpy(float),
    ):
        raise RuntimeError("P12 T3/T4 frozen native score differs")
    if np.isinf(
        aligned.loc[:, t3_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ).any():
        raise ValueError("P12 T3 model features contain infinity")
    consensus, consensus_model_columns = _build_consensus_frame(d1, peers)
    append_columns = sorted(
        column for column in consensus.columns if column.startswith("t4_")
    )
    duplicates = sorted(set(append_columns).intersection(wide.columns))
    if duplicates:
        raise RuntimeError(f"P12 T3 already contains reserved T4 columns: {duplicates}")
    result = wide.merge(
        consensus.loc[:, [*keys, *append_columns]],
        on=keys,
        validate="one_to_one",
        sort=False,
    )
    expected_columns = [*wide.columns, *append_columns]
    result = result.loc[:, expected_columns]
    t4_model_columns = tuple(
        column for column in sorted(consensus_model_columns) if column in append_columns
    )
    model_columns = assert_model_feature_columns((*t3_columns, *t4_model_columns))
    numeric = result.loc[:, model_columns].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(float)).any():
        raise RuntimeError("P12 widened T4 model features contain infinity")
    return result, model_columns


def write_t4_features(
    *,
    split: str,
    d1_top5_path: str | Path,
    t3_manifest_path: str | Path,
    peer_top5_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    source_paths: Mapping[str, str | Path],
    resume: bool,
) -> dict[str, Any]:
    split_name = str(split).lower()
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("P12 T4 split must be Train/Validation/Test")
    if set(peer_top5_paths) != set(PEER_ROUTES):
        raise ValueError("P12 T4 peer path inventory differs")
    d1_path = Path(d1_top5_path).resolve()
    t3_manifest_source = Path(t3_manifest_path).resolve()
    t3_manifest = load_content_manifest(
        t3_manifest_source, name=f"P12 {split_name} T3 source", statuses=("COMPLETE",)
    )
    if (
        t3_manifest.get("candidate_test_labels_read") is not False
        or t3_manifest.get("configuration", {}).get("split") != split_name
        or t3_manifest.get("configuration", {}).get("pool") != "top5"
        or t3_manifest.get("configuration", {}).get("track") != "T3_route_rich"
    ):
        raise PermissionError("P12 T4 T3 source semantics differ")
    t3_columns = assert_model_feature_columns(
        tuple(map(str, t3_manifest.get("model_feature_columns", ())))
    )
    if t3_manifest.get("model_feature_schema_sha256") != canonical_sha256(t3_columns):
        raise RuntimeError("P12 T4 T3 feature schema differs")
    t3_path = Path(
        verified_artifact_path(
            t3_manifest.get("artifacts", {}).get("candidate_features", {}),
            name=f"P12 {split_name} T3 features",
        )
    )
    if split_name == "test":
        assert_label_free_parquet_schema(t3_path, name="P12 T4 Test T3 features")
    peer_paths = {
        route: Path(peer_top5_paths[route]).resolve() for route in PEER_ROUTES
    }
    reserved_sources = {"d1_top5", "t3_manifest", "t3_features", "peer_top5"}
    overlap = reserved_sources.intersection(source_paths)
    if overlap:
        raise ValueError(f"P12 T4 source names are reserved: {sorted(overlap)}")
    sources = {
        "d1_top5": _record(d1_path),
        "t3_manifest": _record(t3_manifest_source),
        "t3_features": _record(t3_path),
        "peer_top5": {route: _record(path) for route, path in peer_paths.items()},
        **{name: _record(path) for name, path in sorted(source_paths.items())},
    }
    configuration = {
        "split": split_name,
        "route": "D1",
        "pool": "top5",
        "track": "T4_four_route_consensus",
        "evidence_composition": "exact T3_route_rich columns then stable-sorted t4 consensus",
        "peer_routes": list(PEER_ROUTES),
        "matching": "deterministic nearest center; native-rank/candidate-ID tie-break",
        "coordinates": "original_image_pixels",
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = Path(output_dir).resolve()
    marker = output / "manifest.json"
    if marker.exists():
        existing = load_content_manifest(
            marker, name="P12 T4 features", statuses=("COMPLETE",)
        )
        if existing.get("signature_sha256") != signature:
            raise RuntimeError("immutable P12 T4 manifest signature differs")
        verify_artifact_records_recursive(
            existing.get("artifacts"),
            name="P12 T4 artifacts",
            require_at_least_one=True,
        )
        if resume:
            return existing
        raise FileExistsError("P12 T4 features already exist")
    d1 = _read_candidates(d1_path, route="D1", split=split_name)
    peers = {
        route: _read_candidates(path, route=route, split=split_name)
        for route, path in peer_paths.items()
    }
    t3 = pd.read_parquet(t3_path)
    frame, model_columns = build_t4_frame(d1, peers, t3, t3_columns)
    feature_path = atomic_parquet(frame, output / "candidate_features.parquet")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "model_feature_columns": list(model_columns),
        "feature_schema_sha256": canonical_sha256(model_columns),
        "candidate_rows": int(len(frame)),
        "sources": sources,
        "artifacts": {"candidate_features": _record(feature_path)},
        "candidate_test_labels_read": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


__all__ = ["PEER_ROUTES", "build_t4_frame", "write_t4_features"]
