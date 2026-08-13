"""Canonical label-free D1 candidate projection and pool freezing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    load_verified_json,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file

from .contracts import (
    CANDIDATE_GEOMETRY_COLUMNS,
    CANDIDATE_REQUIRED_COLUMNS,
    D1_ROUTE,
    FORBIDDEN_CANDIDATE_OUTPUT_COLUMNS,
    POOL_LIMITS,
    RECTANGLE_HEIGHT_PX,
)
from .io import atomic_parquet


_CANDIDATE_COLUMNS = (
    "sample_id",
    "sample_index",
    "scene_id",
    "candidate_id",
    "center_u_px",
    "center_v_px",
    "center_depth_m",
    "angle_rad",
    "width_m",
    "width_px",
    "valid",
    "candidate_json",
)
_POSE_JSON_COLUMNS = (
    "endpoints_uv_json",
    "center_camera_xyz_m_json",
    "pose_matrix_json",
)
_OPTIONAL_POSE_COLUMNS = (
    "contact_points_uv_json",
    "contact_normals_json",
)
_SCORE_COLUMNS = (
    "sample_id",
    "sample_index",
    "scene_id",
    "candidate_id",
    "candidate_identity_sha256",
    "gqcnn_q_value",
    "gqcnn_rank",
    "source_candidate_index",
    "source_candidates_npz_sha256",
    "scored_candidates_npz_sha256",
    "model_name",
    "model_commit",
    "model_config_sha256",
)


def artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} misses columns: {missing}")


_SOURCE_JSON_FLOAT64 = "source_json_float64"
_SCORER_NPZ_FLOAT32 = "scorer_npz_float32"
_IDENTITY_PRECISION_CONTRACTS = (
    _SOURCE_JSON_FLOAT64,
    _SCORER_NPZ_FLOAT32,
)


def source_pose_identity_sha256(
    record: Mapping[str, Any],
    *,
    precision_contract: str = _SOURCE_JSON_FLOAT64,
) -> str:
    """Replay one of Snapshot A's two frozen full-pose encodings.

    The historical Test score table hashes the source JSON float64 values.
    Train/Validation were compacted after the scorer started reloading the
    float32 NPZ arrays, so those hashes use float32-normalized values promoted
    back to float64 by ``candidate_identity_sha256``.  Both encodings are
    immutable source facts; callers must prove one uniform contract per split.
    """

    if precision_contract not in _IDENTITY_PRECISION_CONTRACTS:
        raise ValueError(
            f"unsupported D1 identity precision contract: {precision_contract}"
        )

    sample_id = str(record.get("sample_id", ""))
    candidate_id = str(record.get("candidate_id", ""))
    if not sample_id or not candidate_id:
        raise ValueError("D1 candidate identity requires sample_id/candidate_id")
    fields = (
        ("center_uv", [record.get("center_u_px"), record.get("center_v_px")]),
        ("center_depth_m", record.get("center_depth_m")),
        ("center_camera_xyz_m", record.get("center_camera_xyz_m")),
        ("angle_rad", record.get("angle_rad")),
        ("width_m", record.get("width_m")),
        ("width_px", record.get("width_px")),
        ("endpoints_uv", record.get("endpoints_uv")),
        ("T_camera_grasp_fixed_approach", record.get("pose_matrix")),
    )
    digest = hashlib.sha256()
    digest.update(sample_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(candidate_id.encode("utf-8"))
    for field, raw in fields:
        try:
            if (
                precision_contract == _SCORER_NPZ_FLOAT32
                and field != "T_camera_grasp_fixed_approach"
            ):
                value = np.asarray(raw, dtype="<f4").astype("<f8")
            else:
                value = np.asarray(raw, dtype="<f8")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{sample_id}/{candidate_id} has invalid immutable field {field}"
            ) from error
        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{sample_id}/{candidate_id} has non-finite immutable field {field}"
            )
        digest.update(b"\0")
        digest.update(field.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _geometry_identity(row: Any) -> str:
    return canonical_sha256(
        [
            D1_ROUTE,
            str(row.sample_id),
            str(row.candidate_id),
            *[float(getattr(row, column)) for column in CANDIDATE_GEOMETRY_COLUMNS],
        ]
    )


def _json_pose(value: Any, *, field: str, sample_id: str, candidate_id: str) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{sample_id}/{candidate_id} has invalid {field} JSON"
        ) from error


def _pose_columns_from_candidate_json(candidates: pd.DataFrame) -> pd.DataFrame:
    """Expand pose sidecars missing from Snapshot A's compact Test table."""

    missing = [column for column in _POSE_JSON_COLUMNS if column not in candidates]
    optional_missing = [
        column for column in _OPTIONAL_POSE_COLUMNS if column not in candidates
    ]
    if not missing and not optional_missing:
        return candidates
    parsed: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        try:
            value = json.loads(str(row.candidate_json))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{row.sample_id}/{row.candidate_id} has invalid candidate_json"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"{row.sample_id}/{row.candidate_id} candidate_json is not an object"
            )
        parsed.append(value)
    output = candidates.copy()
    if "endpoints_uv_json" in missing:
        output["endpoints_uv_json"] = [
            json.dumps(
                [value.get("endpoint_1_uv"), value.get("endpoint_2_uv")],
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in parsed
        ]
    if "center_camera_xyz_m_json" in missing:
        output["center_camera_xyz_m_json"] = [
            json.dumps(
                value.get("center_camera_xyz_m"),
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in parsed
        ]
    if "pose_matrix_json" in missing:
        output["pose_matrix_json"] = [
            json.dumps(
                value.get("T_camera_grasp_fixed_approach"),
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in parsed
        ]
    for column, source_name in (
        ("contact_points_uv_json", "contact_points_uv"),
        ("contact_normals_json", "contact_normals"),
    ):
        if column in optional_missing:
            output[column] = [
                json.dumps(
                    value.get(source_name),
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for value in parsed
            ]
    return output


def _validate_native_order(frame: pd.DataFrame) -> None:
    ordered = frame.sort_values(
        ["sample_id", "native_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ordered["expected_rank"] = ordered.groupby("sample_id", sort=False).cumcount() + 1
    bad = ordered.loc[ordered["native_rank"] != ordered["expected_rank"]]
    if not bad.empty:
        example = bad.iloc[0]
        raise ValueError(
            "stored GQ-CNN rank differs from full-precision q descending / "
            f"candidate-id tie order: {example.sample_id}/{example.candidate_id}"
        )


def canonicalise_candidate_rows(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Project source rows into a physically label-free canonical D1 table."""

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    _require_columns(candidates, _CANDIDATE_COLUMNS, "D1 candidate source")
    candidates = _pose_columns_from_candidate_json(candidates)
    _require_columns(candidates, _POSE_JSON_COLUMNS, "D1 candidate pose source")
    _require_columns(scores, _SCORE_COLUMNS, "D1 GQ-CNN score source")
    _require_columns(
        paired, ("sample_id", "sample_index", "scene_id", "frame_id"), "paired manifest"
    )

    candidate_columns = (
        list(_CANDIDATE_COLUMNS)
        + list(_POSE_JSON_COLUMNS)
        + [column for column in _OPTIONAL_POSE_COLUMNS if column in candidates.columns]
    )
    candidate_work = candidates.loc[:, candidate_columns].copy()
    for column in _OPTIONAL_POSE_COLUMNS:
        if column not in candidate_work:
            candidate_work[column] = None
    score_work = scores.loc[:, list(_SCORE_COLUMNS)].copy()
    if candidate_work[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("duplicate D1 candidate keys")
    if score_work[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("duplicate D1 score keys")
    candidate_keys = set(
        map(tuple, candidate_work[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    score_keys = set(
        map(tuple, score_work[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if candidate_keys != score_keys:
        raise ValueError(
            f"D1 candidate/score key mismatch: candidate_only={len(candidate_keys - score_keys)} "
            f"score_only={len(score_keys - candidate_keys)}"
        )

    joined = candidate_work.merge(
        score_work,
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_score"),
    )
    for column in ("sample_index", "scene_id"):
        left = joined[f"{column}_candidate"]
        right = joined[f"{column}_score"]
        if not left.astype(str).equals(right.astype(str)):
            raise ValueError(f"D1 candidate/score {column} mismatch")

    paired_work = paired.loc[
        :, ["sample_id", "sample_index", "scene_id", "frame_id"]
    ].copy()
    paired_work["sample_id"] = paired_work["sample_id"].astype(str)
    if (
        paired_work["sample_id"].eq("").any()
        or paired_work["sample_id"].duplicated().any()
    ):
        raise ValueError("paired manifest requires unique non-empty sample IDs")
    foreign = sorted(
        set(joined["sample_id"].astype(str)).difference(paired_work["sample_id"])
    )
    if foreign:
        raise ValueError(f"D1 candidates reference foreign samples: {foreign[:5]}")
    paired_work = paired_work.rename(
        columns={
            "sample_index": "sample_index_paired",
            "scene_id": "scene_id_paired",
        }
    )
    joined = joined.merge(
        paired_work,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    for column in ("sample_index", "scene_id"):
        source = joined[f"{column}_candidate"]
        expected = joined[f"{column}_paired"]
        if not source.astype(str).equals(expected.astype(str)):
            raise ValueError(f"D1 source differs from paired manifest: {column}")

    finite_columns = (
        "center_u_px",
        "center_v_px",
        "center_depth_m",
        "angle_rad",
        "width_m",
        "width_px",
        "gqcnn_q_value",
    )
    numeric = joined.loc[:, finite_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("D1 candidates contain NaN/Inf geometry or q")
    if (joined[["center_depth_m", "width_m", "width_px"]].to_numpy(float) <= 0).any():
        raise ValueError("D1 candidates contain non-positive depth or width")
    if not joined["valid"].astype(bool).all():
        raise ValueError("D1 NMS pool contains an invalid candidate")
    if (joined["gqcnn_rank"].astype(int) <= 0).any():
        raise ValueError("D1 GQ-CNN ranks must be positive")

    output = pd.DataFrame(
        {
            "sample_id": joined["sample_id"].astype(str),
            "sample_index": joined["sample_index_paired"].astype("int64"),
            "frame_id": joined["frame_id"].astype(str),
            "scene_id": joined["scene_id_paired"].astype(str),
            "route": D1_ROUTE,
            "split": split,
            "candidate_id": joined["candidate_id"].astype(str),
            "native_rank": joined["gqcnn_rank"].astype("int32"),
            "native_score": joined["gqcnn_q_value"].astype("float64"),
            "cx_px": joined["center_u_px"].astype("float64"),
            "cy_px": joined["center_v_px"].astype("float64"),
            "center_depth_m": joined["center_depth_m"].astype("float64"),
            "theta_deg": np.degrees(joined["angle_rad"].astype("float64")),
            "source_angle_rad": joined["angle_rad"].astype("float64"),
            "width_px": joined["width_px"].astype("float64"),
            "height_px": RECTANGLE_HEIGHT_PX,
            "width_m": joined["width_m"].astype("float64"),
            "source_candidate_identity_sha256": joined[
                "candidate_identity_sha256"
            ].astype(str),
            "source_candidate_index": joined["source_candidate_index"].astype("int32"),
            "source_candidates_npz_sha256": joined[
                "source_candidates_npz_sha256"
            ].astype(str),
            "scored_candidates_npz_sha256": joined[
                "scored_candidates_npz_sha256"
            ].astype(str),
            "gqcnn_model_name": joined["model_name"].astype(str),
            "gqcnn_model_commit": joined["model_commit"].astype(str),
            "gqcnn_model_config_sha256": joined["model_config_sha256"].astype(str),
            "candidate_json": joined["candidate_json"].astype(str),
        }
    )
    for column in _OPTIONAL_POSE_COLUMNS:
        output[column] = joined[column]
    for column in (
        "endpoints_uv_json",
        "center_camera_xyz_m_json",
        "pose_matrix_json",
    ):
        output[column] = joined[column]
    computed_identities = []
    identity_contracts: list[str] = []
    for row in joined.itertuples(index=False):
        sample_id = str(row.sample_id)
        candidate_id = str(row.candidate_id)
        identity_record = {
            "sample_id": sample_id,
            "candidate_id": candidate_id,
            "center_u_px": float(row.center_u_px),
            "center_v_px": float(row.center_v_px),
            "center_depth_m": float(row.center_depth_m),
            "center_camera_xyz_m": _json_pose(
                row.center_camera_xyz_m_json,
                field="center_camera_xyz_m",
                sample_id=sample_id,
                candidate_id=candidate_id,
            ),
            "angle_rad": float(row.angle_rad),
            "width_m": float(row.width_m),
            "width_px": float(row.width_px),
            "endpoints_uv": _json_pose(
                row.endpoints_uv_json,
                field="endpoints_uv",
                sample_id=sample_id,
                candidate_id=candidate_id,
            ),
            "pose_matrix": _json_pose(
                row.pose_matrix_json,
                field="pose_matrix",
                sample_id=sample_id,
                candidate_id=candidate_id,
            ),
        }
        expected = str(row.candidate_identity_sha256)
        matches = [
            contract
            for contract in _IDENTITY_PRECISION_CONTRACTS
            if source_pose_identity_sha256(identity_record, precision_contract=contract)
            == expected
        ]
        if not matches:
            raise ValueError(
                "D1 source full-pose identity differs from frozen score binding: "
                f"{sample_id}/{candidate_id}"
            )
        identity_contracts.append(
            matches[0] if len(matches) == 1 else "precision_invariant"
        )
        computed_identities.append(expected)
    resolved_contracts = sorted(
        set(identity_contracts).difference({"precision_invariant"})
    )
    if len(resolved_contracts) > 1:
        raise ValueError(
            "D1 frozen score binding mixes source JSON and scorer NPZ precision"
        )
    resolved_contract = (
        resolved_contracts[0] if resolved_contracts else _SOURCE_JSON_FLOAT64
    )
    output["candidate_identity_sha256"] = computed_identities
    output["candidate_identity_precision_contract"] = resolved_contract
    output["candidate_geometry_sha256"] = [
        _geometry_identity(row) for row in output.itertuples(index=False)
    ]
    output = output.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    _validate_native_order(output)
    verify_canonical_candidate_frame(output, split=split)
    return output


def verify_canonical_candidate_frame(frame: pd.DataFrame, *, split: str) -> None:
    _require_columns(frame, CANDIDATE_REQUIRED_COLUMNS, "canonical D1 candidates")
    forbidden = sorted(
        set(FORBIDDEN_CANDIDATE_OUTPUT_COLUMNS).intersection(frame.columns)
    )
    if forbidden:
        raise ValueError(
            f"canonical D1 candidates contain forbidden label columns: {forbidden}"
        )
    if frame[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("canonical D1 candidate keys are not unique")
    if set(frame["route"].astype(str)) != {D1_ROUTE} or set(
        frame["split"].astype(str)
    ) != {split}:
        raise ValueError("canonical D1 route/split contract mismatch")
    if not np.allclose(
        frame["theta_deg"].to_numpy(float),
        np.degrees(frame["source_angle_rad"].to_numpy(float)),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("canonical D1 degree/radian angle mapping mismatch")
    if "candidate_identity_precision_contract" not in frame:
        raise ValueError("canonical D1 candidates miss identity precision contract")
    contracts = set(frame["candidate_identity_precision_contract"].astype(str))
    if len(contracts) != 1 or not contracts.issubset(_IDENTITY_PRECISION_CONTRACTS):
        raise ValueError("canonical D1 identity precision contract is invalid")
    precision_contract = next(iter(contracts))
    expected_identity = []
    for row in frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        candidate_id = str(row.candidate_id)
        expected_identity.append(
            source_pose_identity_sha256(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "center_u_px": float(row.cx_px),
                    "center_v_px": float(row.cy_px),
                    "center_depth_m": float(row.center_depth_m),
                    "center_camera_xyz_m": _json_pose(
                        row.center_camera_xyz_m_json,
                        field="center_camera_xyz_m",
                        sample_id=sample_id,
                        candidate_id=candidate_id,
                    ),
                    "angle_rad": float(row.source_angle_rad),
                    "width_m": float(row.width_m),
                    "width_px": float(row.width_px),
                    "endpoints_uv": _json_pose(
                        row.endpoints_uv_json,
                        field="endpoints_uv",
                        sample_id=sample_id,
                        candidate_id=candidate_id,
                    ),
                    "pose_matrix": _json_pose(
                        row.pose_matrix_json,
                        field="pose_matrix",
                        sample_id=sample_id,
                        candidate_id=candidate_id,
                    ),
                },
                precision_contract=precision_contract,
            )
        )
    if frame["candidate_identity_sha256"].astype(str).tolist() != expected_identity:
        raise ValueError("canonical D1 full-pose identity mismatch")
    expected_geometry = [
        _geometry_identity(row) for row in frame.itertuples(index=False)
    ]
    if frame["candidate_geometry_sha256"].astype(str).tolist() != expected_geometry:
        raise ValueError("canonical D1 candidate geometry hash mismatch")
    _validate_native_order(frame)


def select_pool(frame: pd.DataFrame, pool: str) -> pd.DataFrame:
    if pool not in POOL_LIMITS:
        raise ValueError(f"unknown D1 pool: {pool}")
    limit = POOL_LIMITS[pool]
    selected = frame if limit is None else frame.loc[frame["native_rank"] <= limit]
    return selected.copy().reset_index(drop=True)


def pool_hash_rows(
    pools: Mapping[str, pd.DataFrame], paired: pd.DataFrame, *, split: str
) -> pd.DataFrame:
    sample_ids = paired["sample_id"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for pool_name, pool in pools.items():
        grouped = {
            str(sample_id): values.sort_values(
                ["native_rank", "candidate_id"], kind="mergesort"
            )
            for sample_id, values in pool.groupby("sample_id", sort=False)
        }
        for sample_id in sample_ids:
            values = grouped.get(sample_id)
            records = (
                []
                if values is None
                else [
                    {
                        "candidate_id": str(row.candidate_id),
                        "native_rank": int(row.native_rank),
                        "native_score": float(row.native_score),
                        "candidate_identity_sha256": str(row.candidate_identity_sha256),
                        "candidate_geometry_sha256": str(row.candidate_geometry_sha256),
                    }
                    for row in values.itertuples(index=False)
                ]
            )
            rows.append(
                {
                    "split": split,
                    "pool": pool_name,
                    "sample_id": sample_id,
                    "candidate_count": len(records),
                    "membership_sha256": canonical_sha256(
                        [item["candidate_id"] for item in records]
                    ),
                    "native_score_vector_sha256": canonical_sha256(
                        [item["native_score"] for item in records]
                    ),
                    "candidate_geometry_vector_sha256": canonical_sha256(
                        [item["candidate_geometry_sha256"] for item in records]
                    ),
                    "candidate_contract_sha256": canonical_sha256(records),
                }
            )
    return pd.DataFrame(rows)


def _split_output_dir(run_dir: Path, split: str) -> Path:
    return (
        run_dir / "02_candidates"
        if split == "test"
        else run_dir / "02_candidates" / split
    )


def _split_manifest_path(run_dir: Path, split: str) -> Path:
    output = _split_output_dir(run_dir, split)
    return output / ("test_manifest.json" if split == "test" else "manifest.json")


def refresh_candidate_registry(run_dir: str | Path) -> dict[str, Any]:
    """Aggregate exact split manifests into the prompt-facing candidate manifest."""

    root = Path(run_dir).resolve()
    split_records: dict[str, Any] = {}
    complete = True
    for split in ("train", "validation", "test"):
        path = _split_manifest_path(root, split)
        if not path.is_file():
            complete = False
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        expected_content = str(value.get("content_sha256", ""))
        unsigned = dict(value)
        unsigned.pop("content_sha256", None)
        if (
            value.get("status") != "COMPLETE"
            or canonical_sha256(unsigned) != expected_content
        ):
            raise RuntimeError(f"D1 {split} candidate manifest is corrupt")
        verify_artifact_records_recursive(
            value.get("artifacts", {}),
            name=f"D1 {split} candidate artifacts",
            require_at_least_one=True,
        )
        split_records[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "content_sha256": expected_content,
            "summaries": value["summaries"],
        }
    registry: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE" if complete else "PARTIAL",
        "protocol": "fair-d1-canonical-order-only-v1",
        "candidate_test_labels_read": False,
        "required_splits": ["train", "validation", "test"],
        "split_manifests": split_records,
    }
    registry["content_sha256"] = canonical_sha256(registry)
    atomic_json(root / "02_candidates" / "d1_candidate_manifest.json", registry)
    if complete:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "CANDIDATES_FROZEN"
        manifest["candidate_manifest"] = {
            "path": str(
                (root / "02_candidates" / "d1_candidate_manifest.json").resolve()
            ),
            "sha256": sha256_file(
                root / "02_candidates" / "d1_candidate_manifest.json"
            ),
        }
        atomic_json(manifest_path, manifest)
        atomic_json(
            root / "pipeline_status.json",
            {
                "schema_version": 1,
                "status": "CANDIDATES_FROZEN",
                "first_incomplete_stage": "P4_FEATURES_AND_DEVELOPMENT_LABELS",
                "formal_test_executed": False,
                "test_candidate_labels_read": False,
            },
        )
    return registry


def build_canonical_candidate_split(
    *,
    run_dir: str | Path,
    split: str,
    candidate_paths: Iterable[str | Path],
    score_paths: Iterable[str | Path],
    paired_path: str | Path,
    resume: bool = False,
    source_contract: Mapping[str, Any] | None = None,
    execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build All-NMS, Top-10 and Top-5 without opening any label artifact."""

    root = Path(run_dir).resolve()
    output_dir = _split_output_dir(root, split)
    manifest_path = _split_manifest_path(root, split)
    candidate_records = [artifact_record(path) for path in candidate_paths]
    score_records = [artifact_record(path) for path in score_paths]
    paired_record = artifact_record(paired_path)
    configuration = {
        "schema_version": 1,
        "route": D1_ROUTE,
        "split": split,
        "rectangle_height_px": RECTANGLE_HEIGHT_PX,
        "angle_mapping": "theta_deg = degrees(source angle_rad); no sign flip",
        "width_mapping": "source configured width_px; contact span is sensitivity only",
        "native_order": "full precision gqcnn_q_value descending; exact ties candidate_id ascending",
        "candidate_identity_precision": (
            "uniformly discovered and verified against the frozen score binding; "
            "source_json_float64 or scorer_npz_float32"
        ),
        "pool_limits": POOL_LIMITS,
        "candidate_sources": candidate_records,
        "score_sources": score_records,
        "paired_manifest": paired_record,
        "source_contract": dict(source_contract or {}),
    }
    source_signature = canonical_sha256(configuration)
    if manifest_path.is_file():
        existing = load_verified_json(
            manifest_path, name=f"D1 {split} candidate manifest"
        )
        unsigned = dict(existing)
        observed_content = unsigned.pop("content_sha256", None)
        if observed_content != canonical_sha256(unsigned):
            raise RuntimeError(f"D1 {split} candidate manifest content hash mismatch")
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("source_signature_sha256") == source_signature
        ):
            if not resume:
                raise FileExistsError(
                    f"D1 candidate split already exists; pass --resume: {manifest_path}"
                )
            if existing.get("configuration") != configuration:
                raise RuntimeError(f"D1 {split} candidate configuration differs")
            verify_artifact_records_recursive(
                {
                    "configuration": existing.get("configuration"),
                    "execution_contract": existing.get("execution_contract"),
                    "artifacts": existing.get("artifacts"),
                },
                name="D1 candidate resume",
                require_at_least_one=True,
            )
            refresh_candidate_registry(root)
            return existing
        raise RuntimeError(
            f"D1 candidate manifest exists with different or incomplete contract: {manifest_path}"
        )

    candidate_frames = [pd.read_parquet(record["path"]) for record in candidate_records]
    score_frames = [pd.read_parquet(record["path"]) for record in score_records]
    if not candidate_frames or not score_frames:
        raise ValueError("D1 candidate and score source lists must be non-empty")
    paired = pd.read_parquet(paired_record["path"])
    canonical = canonicalise_candidate_rows(
        pd.concat(candidate_frames, ignore_index=True),
        pd.concat(score_frames, ignore_index=True),
        paired,
        split=split,
    )
    identity_precision_contracts = sorted(
        set(canonical["candidate_identity_precision_contract"].astype(str))
    )
    if len(identity_precision_contracts) != 1:
        raise RuntimeError(f"D1 {split} candidate identity precision is not uniform")
    pools = {name: select_pool(canonical, name) for name in POOL_LIMITS}
    artifacts: dict[str, Any] = {}
    for pool_name, frame in pools.items():
        path = atomic_parquet(frame, output_dir / f"d1_{pool_name}_candidates.parquet")
        artifacts[pool_name] = {
            **artifact_record(path),
            "rows": len(frame),
            "samples_with_candidates": int(frame["sample_id"].nunique()),
        }
    hashes = pool_hash_rows(pools, paired, split=split)
    hashes_name = (
        "d1_candidate_hashes.parquet" if split == "test" else "candidate_hashes.parquet"
    )
    hashes_path = atomic_parquet(hashes, output_dir / hashes_name)
    artifacts["candidate_hashes"] = {
        **artifact_record(hashes_path),
        "rows": len(hashes),
    }
    expected_samples = int(paired["sample_id"].nunique())
    summaries = {}
    for name, frame in pools.items():
        counts = (
            frame.groupby("sample_id").size().reindex(paired["sample_id"], fill_value=0)
        )
        summaries[name] = {
            "rows": len(frame),
            "denominator_samples": expected_samples,
            "samples_with_candidates": int((counts > 0).sum()),
            "no_output_samples": int((counts == 0).sum()),
            "maximum_candidates": int(counts.max()),
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "protocol": "fair-d1-canonical-order-only-v1",
        "candidate_test_labels_read": False,
        "configuration": configuration,
        "source_signature_sha256": source_signature,
        "candidate_identity_precision_contract": identity_precision_contracts[0],
        "execution_contract": dict(execution_contract or {}),
        "artifacts": artifacts,
        "summaries": summaries,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    refresh_candidate_registry(root)
    return manifest
