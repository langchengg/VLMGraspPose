#!/usr/bin/env python3
"""Generate compact frozen Dex-Net pools from GT-free HiFi predictions.

Per-sample directories retain only the post-NMS scorer inputs.  Raw,
mask-validated, and post-NMS candidates are streamed into three ZSTD Parquet
tables.  Temporary stage sidecars and atomic-write files are confined to the
explicit current-run ``--tmp-root``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from ruamel.yaml import YAML

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # candidate generation uses the pinned legacy env
    pa = None
    pq = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "gqcnn-official"))

from src.grasping.camera_geometry import (  # noqa: E402
    CameraIntrinsicsData,
    depth_mm_to_meters,
)
from src.grasping.dexnet_adapter import (  # noqa: E402
    export_intrinsics_file,
    gqcnn_runtime,
    make_camera_intrinsics,
)
from src.grasping.dexnet_candidate_generator import generate_candidates  # noqa: E402
from src.grasping.dexnet_run_reliability import (  # noqa: E402
    SUCCESS_EMPTY,
    SUCCESS_NONEMPTY,
    canonical_json_hash,
    sha256_file,
    write_completion_marker,
)
from src.grasping.gqcnn_full_scoring import load_source_sample  # noqa: E402
from src.grasping.grasp_serialization import (  # noqa: E402
    candidates_to_records,
    save_candidates_npz,
)
from src.grasping.mask_processing import process_mask_with_diagnostics  # noqa: E402
from src.grasping.ocid_vlg_grasp_adapter import (  # noqa: E402
    OcidVlgGraspSample,
)
from src.grasping.reranking_v1.labels import evaluate_candidate_label  # noqa: E402
from tools.export_anygrasp_inputs import derive_intrinsics_from_pcd  # noqa: E402


SUMMARY_FIELDS = (
    "sample_id",
    "query",
    "mask_area_px",
    "valid_target_depth_px",
    "requested_candidate_count",
    "raw_candidate_count",
    "mask_validated_count",
    "post_nms_count",
    "scored_candidate_count",
    "best_gqcnn_q",
    "median_gqcnn_q",
    "generation_time_ms",
    "scoring_time_ms",
    "total_time_ms",
    "failure_reason",
    "status",
    "question_index",
    "scene_id",
    "sample_seed",
)

SAMPLE_SEED_MODE = "stable-sha256"
SEED_NAMESPACE = "hierfilm-modular-formal-v1"
SAMPLE_SEED_DERIVATION = (
    "uint64_be(sha256(namespace\\0base_seed\\0stable_sample_id)[:8]) "
    "mod (2**32-1)"
)
PIPELINE_NAME = "hierfilm"

CANDIDATE_STAGE_SCHEMA_SPEC = (
    ("pipeline", "string"),
    ("stage", "string"),
    ("sample_index", "int32"),
    ("sample_id", "string"),
    ("question_index", "int32"),
    ("scene_id", "string"),
    ("candidate_id", "string"),
    ("sampler_rank", "int32"),
    ("candidate_seed", "int64"),
    ("center_u_px", "float64"),
    ("center_v_px", "float64"),
    ("center_depth_m", "float64"),
    ("angle_rad", "float64"),
    ("width_m", "float64"),
    ("width_px", "float64"),
    ("endpoints_uv_json", "string"),
    ("contact_points_uv_json", "string"),
    ("contact_normals_json", "string"),
    ("center_camera_xyz_m_json", "string"),
    ("pose_matrix_json", "string"),
    ("valid", "bool"),
    ("rejection_reason", "string"),
    ("candidate_json", "string"),
)
CANDIDATE_STAGE_SCHEMA = (
    None
    if pa is None
    else pa.schema(
        [
            (
                name,
                {
                    "string": pa.string(),
                    "int32": pa.int32(),
                    "int64": pa.int64(),
                    "float64": pa.float64(),
                    "bool": pa.bool_(),
                }[kind],
            )
            for name, kind in CANDIDATE_STAGE_SCHEMA_SPEC
        ]
    )
)

STAGE_FILES = {
    "raw": "raw_candidates.parquet",
    "mask_validated": "mask_validated_candidates.parquet",
    "nms": "nms_candidates.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tmp-root",
        type=Path,
        required=True,
        help="Temporary directory below the same protected experiment run.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "finalize"),
        default="all",
        help=(
            "Use generate in the pinned Dex-Net environment and finalize in "
            "the PyArrow environment, or all when one environment has both."
        ),
    )
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--reference-candidate-root", type=Path)
    return parser.parse_args()


def derive_sample_seed(
    sample_id: str,
    *,
    base_seed: int,
    namespace: str = SEED_NAMESPACE,
) -> int:
    """Derive the exact retained repeated-FiLM per-sample sampler seed."""

    payload = f"{namespace}\0{int(base_seed)}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**32 - 1
    )


def _protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def _validate_tmp_scope(output_root: Path, tmp_root: Path) -> tuple[Path, Path]:
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = _protected_run_root(output)
    if _protected_run_root(temporary) != run_root:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary


def _atomic_text(path: Path, value: str, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any, tmp_root: Path) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        tmp_root,
    )


def _atomic_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    fields: tuple[str, ...],
    tmp_root: Path,
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def _save_scorer_bundle(
    staging: Path,
    candidates: list[dict[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    """Write only the JSON+NPZ files required by the frozen scorer.

    This local JSON writer stays compatible with the retained Python 3.7
    container, whose :class:`pathlib.Path` lacks ``write_text(newline=...)``.
    """

    records = candidates_to_records(candidates)
    text = json.dumps(
        {"metadata": _plain(metadata), "candidates": records},
        ensure_ascii=False,
        allow_nan=True,
        sort_keys=False,
        indent=2,
        separators=(",", ": "),
    ) + "\n"
    with (staging / "candidates.json").open("w", encoding="utf-8") as stream:
        stream.write(text)
    save_candidates_npz(staging / "candidates.npz", candidates)


def _stage_labels(
    candidates: list[dict[str, Any]],
    grasps: list[Any],
    evaluation_config: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    labels = []
    for candidate in candidates:
        label = evaluate_candidate_label(
            candidate, grasps, evaluation_config
        )
        labels.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                **label.to_dict(),
            }
        )
    return any(row["candidate_positive"] for row in labels), labels


def _write_jsonl(
    path: Path, rows: list[dict[str, Any]], tmp_root: Path
) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        tmp_root,
    )


def _json_scalar(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=True,
    )


def _candidate_stage_row(
    stage: str,
    sample_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in STAGE_FILES:
        raise ValueError(f"unknown candidate stage: {stage}")
    pose = candidate.get("T_camera_grasp_fixed_approach")
    return {
        "pipeline": PIPELINE_NAME,
        "stage": stage,
        "sample_index": int(sample_row["sample_index"]),
        "sample_id": str(sample_row["sample_id"]),
        "question_index": int(sample_row["question_index"]),
        "scene_id": str(sample_row["scene_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "sampler_rank": int(candidate["sampler_rank"]),
        "candidate_seed": int(candidate["seed"]),
        "center_u_px": float(candidate["center_u_px"]),
        "center_v_px": float(candidate["center_v_px"]),
        "center_depth_m": float(candidate["center_depth_m"]),
        "angle_rad": float(candidate["angle_rad"]),
        "width_m": float(candidate["width_m"]),
        "width_px": float(candidate["width_px"]),
        "endpoints_uv_json": _json_scalar(
            [
                candidate["endpoint_1_uv"],
                candidate["endpoint_2_uv"],
            ]
        ),
        "contact_points_uv_json": _json_scalar(
            candidate.get("contact_points_uv")
        ),
        "contact_normals_json": _json_scalar(candidate.get("contact_normals")),
        "center_camera_xyz_m_json": _json_scalar(
            candidate.get("center_camera_xyz_m")
        ),
        "pose_matrix_json": _json_scalar(pose),
        "valid": candidate.get("rejection_reason") is None,
        "rejection_reason": candidate.get("rejection_reason"),
        "candidate_json": _json_scalar(candidate),
    }


def _stage_sidecar_path(sidecar_root: Path, sample_id: str) -> Path:
    return sidecar_root / f"{sample_id}.json.gz"


def _write_stage_sidecar(
    path: Path,
    *,
    sample_row: Mapping[str, Any],
    sample_seed: int,
    protocol_identity_sha256: str,
    result: Any,
) -> None:
    payload = {
        "schema_version": 1,
        "sample_id": str(sample_row["sample_id"]),
        "sample_seed": int(sample_seed),
        "protocol_identity_sha256": protocol_identity_sha256,
        "stages": {
            "raw": candidates_to_records(result.raw_candidates),
            "mask_validated": candidates_to_records(
                result.mask_validated_candidates
            ),
            "nms": candidates_to_records(result.deduplicated_candidates),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
    os.replace(temporary, path)


def _read_stage_sidecar(
    path: Path,
    *,
    sample_id: str,
    sample_seed: int,
    protocol_identity_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        payload.get("schema_version") != 1
        or payload.get("sample_id") != sample_id
        or int(payload.get("sample_seed", -1)) != int(sample_seed)
        or payload.get("protocol_identity_sha256")
        != protocol_identity_sha256
    ):
        raise ValueError(f"temporary candidate-stage sidecar identity differs: {path}")
    stages = payload.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_FILES):
        raise ValueError(f"temporary candidate-stage sidecar schema differs: {path}")
    if not all(isinstance(stages[name], list) for name in STAGE_FILES):
        raise ValueError(f"temporary candidate-stage payload is invalid: {path}")
    return stages


def _candidate_stage_schema_sha256() -> str:
    return canonical_json_hash(CANDIDATE_STAGE_SCHEMA_SPEC)


def _validate_stage_primary_keys(
    stage: str, sample_id: str, rows: list[dict[str, Any]]
) -> None:
    keys = [(sample_id, str(row["candidate_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{sample_id}: duplicate {stage} candidate primary key")
    for row in rows:
        if (
            str(row.get("sample_id")) != sample_id
            or int(row.get("seed", -1)) < 0
        ):
            raise ValueError(f"{sample_id}: invalid {stage} candidate identity")


def _write_candidate_stage_parquets(
    *,
    output_root: Path,
    tmp_root: Path,
    sidecar_root: Path,
    selected: list[dict[str, Any]],
    base_seed: int,
    protocol_identity_sha256: str,
) -> dict[str, dict[str, Any]]:
    if pa is None or pq is None or CANDIDATE_STAGE_SCHEMA is None:
        raise RuntimeError(
            "PyArrow is required for --phase finalize (or --phase all)"
        )
    temporary_paths = {
        stage: tmp_root / f"{filename}.{uuid.uuid4().hex}.tmp"
        for stage, filename in STAGE_FILES.items()
    }
    writers = {
        stage: pq.ParquetWriter(
            path,
            CANDIDATE_STAGE_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        for stage, path in temporary_paths.items()
    }
    buffers: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in STAGE_FILES
    }
    counts = {stage: 0 for stage in STAGE_FILES}
    sample_ids = [str(row["sample_id"]) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("selected prediction rows contain duplicate sample IDs")
    try:
        for sample_row in selected:
            sample_id = str(sample_row["sample_id"])
            sample_seed = derive_sample_seed(
                sample_id, base_seed=base_seed
            )
            stages = _read_stage_sidecar(
                _stage_sidecar_path(sidecar_root, sample_id),
                sample_id=sample_id,
                sample_seed=sample_seed,
                protocol_identity_sha256=protocol_identity_sha256,
            )
            for stage, candidates in stages.items():
                _validate_stage_primary_keys(stage, sample_id, candidates)
                for candidate in candidates:
                    buffers[stage].append(
                        _candidate_stage_row(stage, sample_row, candidate)
                    )
                counts[stage] += len(candidates)
                if len(buffers[stage]) >= 10_000:
                    writers[stage].write_table(
                        pa.Table.from_pylist(
                            buffers[stage], schema=CANDIDATE_STAGE_SCHEMA
                        )
                    )
                    buffers[stage].clear()
        for stage, buffer in buffers.items():
            if buffer:
                writers[stage].write_table(
                    pa.Table.from_pylist(
                        buffer, schema=CANDIDATE_STAGE_SCHEMA
                    )
                )
                buffer.clear()
    finally:
        for writer in writers.values():
            writer.close()

    artifacts: dict[str, dict[str, Any]] = {}
    for stage, filename in STAGE_FILES.items():
        temporary = temporary_paths[stage]
        parquet = pq.ParquetFile(temporary)
        if (
            parquet.schema_arrow != CANDIDATE_STAGE_SCHEMA
            or parquet.metadata.num_rows != counts[stage]
        ):
            raise ValueError(f"{stage} Parquet verification failed")
        destination = output_root / filename
        os.replace(temporary, destination)
        artifacts[stage] = {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "rows": counts[stage],
            "primary_key": ["sample_id", "candidate_id"],
            "compression": "zstd",
        }
    return artifacts


def _validate_existing_stage_parquets(
    *,
    output_root: Path,
    run_config: Mapping[str, Any],
    expected_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]] | None:
    if pa is None or pq is None or CANDIDATE_STAGE_SCHEMA is None:
        raise RuntimeError(
            "PyArrow is required for --phase finalize (or --phase all)"
        )
    artifacts = run_config.get("candidate_stage_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(STAGE_FILES):
        return None
    for stage, filename in STAGE_FILES.items():
        item = artifacts[stage]
        path = output_root / filename
        if (
            not path.is_file()
            or Path(str(item.get("path"))).resolve() != path.resolve()
            or item.get("sha256") != sha256_file(path)
            or int(item.get("rows", -1)) != int(expected_counts[stage])
        ):
            return None
        parquet = pq.ParquetFile(path)
        if (
            parquet.schema_arrow != CANDIDATE_STAGE_SCHEMA
            or parquet.metadata.num_rows != int(expected_counts[stage])
        ):
            return None
    return {stage: dict(artifacts[stage]) for stage in STAGE_FILES}


def _scene_cache(
    cache_root: Path,
    row: Mapping[str, Any],
    *,
    tmp_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    scene_id = str(row["scene_id"])
    key = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:20]
    directory = cache_root / key
    depth_cache = directory / "depth_m.npy"
    intrinsics_cache = directory / "intrinsics.json"
    if depth_cache.is_file() and intrinsics_cache.is_file():
        depth_m = np.load(depth_cache, allow_pickle=False)
        intrinsics = json.loads(intrinsics_cache.read_text(encoding="utf-8"))
        depth_mm = np.asarray(
            Image.open(row["source_depth_path"]), dtype=np.uint16
        )
        return depth_mm, depth_m, intrinsics, depth_cache
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(row["source_depth_path"]) as image:
        depth_mm = np.asarray(image)
    if depth_mm.shape != (480, 640):
        raise ValueError(f"unexpected depth shape for {scene_id}: {depth_mm.shape}")
    depth_m = depth_mm_to_meters(depth_mm)
    intrinsics = derive_intrinsics_from_pcd(
        Path(row["source_pcd_path"]), depth_mm
    )
    intrinsics = {
        **intrinsics,
        "frame": "ocid_camera_optical",
        "skew": 0.0,
    }
    temporary_depth = tmp_root / f"depth_m.{uuid.uuid4().hex}.tmp"
    with temporary_depth.open("wb") as stream:
        np.save(stream, depth_m, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_depth, depth_cache)
    _atomic_json(intrinsics_cache, intrinsics, tmp_root)
    _atomic_json(
        directory / "provenance.json",
        {
            "scene_id": scene_id,
            "source_depth_path": row["source_depth_path"],
            "source_depth_sha256": row["source_depth_sha256"],
            "source_pcd_path": row["source_pcd_path"],
            "source_pcd_sha256": row["source_pcd_sha256"],
            "depth_m_sha256": sha256_file(depth_cache),
            "intrinsics_source": "derived_from_organized_pcd",
        },
        tmp_root,
    )
    return depth_mm, depth_m, intrinsics, depth_cache


def _load_sample(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    scene_cache_root: Path,
    tmp_root: Path,
) -> tuple[OcidVlgGraspSample, Path]:
    with Image.open(row["source_rgb_path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(row["native_mask_path"]) as image:
        mask = np.asarray(image.convert("L"))
    depth_mm, depth_m, intrinsics_json, depth_cache = _scene_cache(
        scene_cache_root, row, tmp_root=tmp_root
    )
    input_config = config["input"]
    processing = process_mask_with_diagnostics(
        mask,
        depth_m,
        threshold=float(input_config["mask_threshold"]),
        min_component_size_px=int(input_config["min_component_area_px"]),
        keep_largest_component=bool(input_config["retain_largest_component"]),
        erode_radius_px=int(input_config["mask_erode_px"]),
        dilate_radius_px=int(input_config["mask_dilate_px"]),
    )
    intrinsics = CameraIntrinsicsData.from_mapping(intrinsics_json)
    sample = OcidVlgGraspSample(
        sample_id=str(row["sample_id"]),
        sample_index=int(row["sample_index"]),
        question_index=int(row["question_index"]),
        scene_id=str(row["scene_id"]),
        query=str(row["query"]),
        bundle_dir=Path(row["native_mask_path"]).parent,
        rgb=rgb,
        depth_mm=depth_mm,
        depth_m=depth_m,
        mask_input=mask.copy(),
        mask_processing=processing,
        intrinsics=intrinsics,
        intrinsics_metadata=intrinsics_json,
        metadata={
            "source_depth": row["source_depth_path"],
            "source_rgb": row["source_rgb_path"],
            "source_pcd": row["source_pcd_path"],
            "prediction_mask": row["native_mask_path"],
            "split": row["split"],
        },
    )
    return sample, depth_cache


def _metadata(
    sample: OcidVlgGraspSample,
    result: Any,
    *,
    config: Mapping[str, Any],
    split: str,
    sample_seed: int,
    protocol_identity_sha256: str,
) -> dict[str, Any]:
    empty_reason = None
    if not result.deduplicated_candidates:
        if not np.any(sample.target_mask_original):
            empty_reason = "predicted_mask_empty"
        elif not np.any(sample.target_mask_processed):
            empty_reason = "no_valid_depth_in_predicted_mask"
        elif not result.raw_candidates:
            empty_reason = "official_sampler_returned_no_candidates"
        else:
            empty_reason = "no_candidates_survived_target_filtering_and_nms"
    return {
        "schema_version": 2,
        "compact_source_schema": "reranking_v1_scorer_compatible",
        "split": split,
        "sample_id": sample.sample_id,
        "sample_index": sample.sample_index,
        "question_index": sample.question_index,
        "scene_id": sample.scene_id,
        "query": sample.query,
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": sample.intrinsics.frame,
        "counts": {
            "requested": int(result.requested_candidate_count),
            "raw": len(result.raw_candidates),
            "mask_validated": len(result.mask_validated_candidates),
            "post_nms": len(result.deduplicated_candidates),
            "top_k": len(result.topk_candidates),
            "scored": 0,
        },
        "mask_area_px": int(np.count_nonzero(sample.target_mask_original)),
        "valid_target_depth_px": int(
            np.count_nonzero(sample.target_mask_processed)
        ),
        "timing_ms": {
            "generation": float(result.generation_time_ms),
            "scoring": 0.0,
            "total": float(result.generation_time_ms),
        },
        "failure_reason": empty_reason,
        "rejection_summary": result.rejection_summary,
        "seed": int(sample_seed),
        "base_seed": int(config["generation"]["seed"]),
        "seed_mode": SAMPLE_SEED_MODE,
        "seed_namespace": SEED_NAMESPACE,
        "sample_seed_derivation": SAMPLE_SEED_DERIVATION,
        "protocol_identity_sha256": protocol_identity_sha256,
        "config": _plain(config),
        "depth_source": str(sample.metadata["source_depth"]),
        "mask_source": str(sample.metadata["prediction_mask"]),
        "intrinsics_source": "derived_from_organized_pcd",
        "factory_calibration": False,
        "depth_scale": 1000.0,
        "coordinate_units": {
            "image": "pixels; u right, v down; origin top-left",
            "depth": "metres",
            "angle": "radians counter-clockwise in image (u,v) coordinates",
            "width": "metres; configured maximum jaw opening",
        },
        "pose": {
            "name": "T_camera_grasp_fixed_approach",
            "source": "official gqcnn Grasp2D.pose()",
            "transform": "from grasp frame to camera frame",
            "fixed_approach_direction_camera": [0.0, 0.0, 1.0],
            "is_freely_predicted_6dof": False,
        },
    }


def _summary_row(
    sample: OcidVlgGraspSample,
    result: Any,
    status: str,
    failure_reason: str | None,
    sample_seed: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "query": sample.query,
        "mask_area_px": int(np.count_nonzero(sample.target_mask_original)),
        "valid_target_depth_px": int(
            np.count_nonzero(sample.target_mask_processed)
        ),
        "requested_candidate_count": int(result.requested_candidate_count),
        "raw_candidate_count": len(result.raw_candidates),
        "mask_validated_count": len(result.mask_validated_candidates),
        "post_nms_count": len(result.deduplicated_candidates),
        "scored_candidate_count": 0,
        "best_gqcnn_q": "",
        "median_gqcnn_q": "",
        "generation_time_ms": float(result.generation_time_ms),
        "scoring_time_ms": 0.0,
        "total_time_ms": float(result.generation_time_ms),
        "failure_reason": failure_reason or "",
        "status": status,
        "question_index": sample.question_index,
        "scene_id": sample.scene_id,
        "sample_seed": int(sample_seed),
    }


def _compare_reference(
    result: Any,
    staging: Path,
    reference_root: Path,
    sample_id: str,
) -> None:
    reference = reference_root / sample_id
    reference_metadata = json.loads(
        (reference / "metadata.json").read_text(encoding="utf-8")
    )
    expected_counts = reference_metadata["counts"]
    actual_counts = {
        "raw": len(result.raw_candidates),
        "mask_validated": len(result.mask_validated_candidates),
        "post_nms": len(result.deduplicated_candidates),
    }
    for key, value in actual_counts.items():
        if int(expected_counts[key]) != value:
            raise AssertionError(
                f"{sample_id}: reference {key} count differs "
                f"{expected_counts[key]} != {value}"
            )
    with np.load(reference / "candidates.npz", allow_pickle=False) as left:
        with np.load(staging / "candidates.npz", allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise AssertionError(f"{sample_id}: NPZ keys differ")
            for name in left.files:
                # PCD-derived intrinsics are recomputed with the currently
                # frozen NumPy LAPACK runtime. Different NumPy builds change
                # only the float64 transform translation at ~1e-14; all
                # image-space candidate identity fields remain exact.
                equivalent = (
                    np.allclose(
                        left[name],
                        right[name],
                        rtol=0.0,
                        atol=1e-12,
                        equal_nan=True,
                    )
                    if name == "T_camera_grasp_fixed_approach"
                    else np.array_equal(left[name], right[name], equal_nan=True)
                )
                if not equivalent:
                    raise AssertionError(f"{sample_id}: NPZ {name} differs")
    expected_raw = json.loads(
        (reference / "raw_candidates.json").read_text(encoding="utf-8")
    )
    expected_mask = json.loads(
        (reference / "mask_validated_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    if [row["candidate_id"] for row in expected_raw] != [
        row["candidate_id"] for row in result.raw_candidates
    ]:
        raise AssertionError(f"{sample_id}: raw candidate IDs differ")
    if [row["candidate_id"] for row in expected_mask] != [
        row["candidate_id"] for row in result.mask_validated_candidates
    ]:
        raise AssertionError(f"{sample_id}: mask candidate IDs differ")


def main() -> int:
    args = parse_args()
    if args.status_every <= 0:
        raise ValueError("status-every must be positive")
    prediction_manifest = args.prediction_manifest.expanduser().resolve()
    annotations_path = args.annotations.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    evaluation_config_path = args.evaluation_config.expanduser().resolve()
    output_root, tmp_root = _validate_tmp_scope(
        args.output_root, args.tmp_root
    )
    reference_root = (
        None
        if args.reference_candidate_root is None
        else args.reference_candidate_root.expanduser().resolve()
    )
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume: {output_root}")
    rows = _read_jsonl(prediction_manifest)
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index are required together")
    if args.num_shards is not None:
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be positive")
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("--shard-index must be in [0, num-shards)")
        # Keep all referring expressions for a scene in the same shard. This
        # avoids duplicate depth/intrinsics caches and preserves scene-grouped
        # isolation while still balancing the official split deterministically.
        selected = [
            row
            for row in rows
            if int(
                hashlib.sha256(str(row["scene_id"]).encode("utf-8")).hexdigest(),
                16,
            )
            % args.num_shards
            == args.shard_index
        ]
    else:
        selected = rows
    selected = selected if args.limit is None else selected[: int(args.limit)]
    if not selected and args.num_shards is None:
        raise ValueError("no samples selected")
    if any(row["split"] != args.split for row in selected):
        raise ValueError("prediction manifest split mismatch")
    yaml = YAML(typ="safe")
    config = yaml.load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(
        config.get("generation"), dict
    ):
        raise ValueError("candidate config has no generation mapping")
    config["generation"] = {
        **config["generation"],
        "sample_seed_mode": SAMPLE_SEED_MODE,
        "seed_namespace": SEED_NAMESPACE,
        "sample_seed_derivation": SAMPLE_SEED_DERIVATION,
    }
    evaluation_config = yaml.load(
        evaluation_config_path.read_text(encoding="utf-8")
    )
    annotations_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if annotations_payload.get("info", {}).get("split") != args.split:
        raise ValueError("annotation split mismatch")
    annotations = {
        int(row["question_index"]): row for row in annotations_payload["data"]
    }
    preliminary_run_config_path = output_root / "run_config.json"
    preliminary_run_config = (
        json.loads(preliminary_run_config_path.read_text(encoding="utf-8"))
        if preliminary_run_config_path.is_file()
        else {}
    )
    if args.phase == "finalize":
        if not preliminary_run_config:
            raise ValueError(
                "--phase finalize requires a staged run_config.json"
            )
        runtime = dict(preliminary_run_config["sampler_runtime"])
    else:
        runtime = gqcnn_runtime()
    configuration_hash = canonical_json_hash(config)
    config_hash = sha256_file(config_path)
    prediction_manifest_hash = sha256_file(prediction_manifest)
    evaluation_config_hash = sha256_file(evaluation_config_path)
    annotations_hash = sha256_file(annotations_path)
    selected_identity_sha256 = hashlib.sha256(
        json.dumps(
            [
                {
                    "sample_index": int(row["sample_index"]),
                    "sample_id": str(row["sample_id"]),
                    "question_index": int(row["question_index"]),
                    "scene_id": str(row["scene_id"]),
                }
                for row in selected
            ],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    protocol_family_identity = {
        "schema_version": 1,
        "split": args.split,
        "prediction_manifest_sha256": prediction_manifest_hash,
        "annotations_sha256": annotations_hash,
        "config_sha256": config_hash,
        "configuration_hash": configuration_hash,
        "evaluation_config_sha256": evaluation_config_hash,
        "sample_seed_mode": SAMPLE_SEED_MODE,
        "seed_namespace": SEED_NAMESPACE,
        "sample_seed_derivation": SAMPLE_SEED_DERIVATION,
        "candidate_stage_schema_sha256": _candidate_stage_schema_sha256(),
        "sampler_commit": runtime.get("commit"),
        "sampler_version": runtime.get("version"),
    }
    protocol_family_identity_sha256 = canonical_json_hash(
        protocol_family_identity
    )
    protocol_identity = {
        **protocol_family_identity,
        "protocol_family_identity_sha256": protocol_family_identity_sha256,
        "selected_identity_sha256": selected_identity_sha256,
    }
    protocol_identity_sha256 = canonical_json_hash(protocol_identity)
    existing_run_config: dict[str, Any] = preliminary_run_config
    if existing_run_config:
        if (
            existing_run_config.get("protocol_identity_sha256")
            != protocol_identity_sha256
        ):
            raise ValueError(
                "existing candidate run protocol identity differs; "
                "use a new output root"
            )
        if (
            args.phase == "generate"
            and existing_run_config.get("status") == "COMPLETED"
        ):
            raise ValueError("candidate run is already completed")
    output_root.mkdir(parents=True, exist_ok=True)
    sidecar_root = (
        tmp_root
        / "compact_dexnet_sidecars"
        / (
            protocol_identity_sha256[:20]
            + "_"
            + hashlib.sha256(str(output_root).encode("utf-8")).hexdigest()[:12]
        )
    )
    staging_root = sidecar_root / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    labels_root = output_root / "_labels"
    labels_root.mkdir(exist_ok=True)
    scene_cache_root = output_root / "_scene_cache"
    scene_cache_root.mkdir(exist_ok=True)
    summaries: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    failures = []
    fresh = 0
    started = time.perf_counter()

    for position, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        sample_seed = derive_sample_seed(
            sample_id, base_seed=int(config["generation"]["seed"])
        )
        final = output_root / sample_id
        label_path = labels_root / f"{sample_id}.json"
        if final.is_dir() and label_path.is_file():
            try:
                source = load_source_sample(final, verify_hashes=True)
                marker = source["marker"]
                if (
                    marker["configuration_hash"] != configuration_hash
                    or marker["config_file_sha256"] != config_hash
                    or int(marker["seed"]) != sample_seed
                ):
                    raise ValueError("existing source identity differs")
                payload = json.loads(label_path.read_text(encoding="utf-8"))
                metadata = source["metadata"]
                if (
                    metadata.get("protocol_identity_sha256")
                    != protocol_identity_sha256
                    or metadata.get("seed_mode") != SAMPLE_SEED_MODE
                    or metadata.get("seed_namespace") != SEED_NAMESPACE
                    or int(metadata.get("seed", -1)) != sample_seed
                    or payload.get("protocol_identity_sha256")
                    != protocol_identity_sha256
                ):
                    raise ValueError("existing compact protocol identity differs")
                summaries.append(dict(marker["summary_row"]))
                funnel_rows.append(dict(payload["sample"]))
                continue
            except Exception:
                if not args.resume:
                    raise
                raise RuntimeError(
                    f"existing compact source is invalid; use a new output root: {final}"
                )
        if final.exists():
            raise FileExistsError(f"partial final source exists: {final}")
        if args.phase == "finalize":
            raise RuntimeError(
                f"staged candidate sample is missing: {sample_id}"
            )
        staging = (
            staging_root
            / f".{sample_id}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        staging.mkdir()
        try:
            annotation = annotations.get(int(row["question_index"]))
            if (
                annotation is None
                or annotation["image_filename"] != row["scene_id"]
                or annotation["question"] != row["query"]
            ):
                raise ValueError(f"annotation join mismatch: {sample_id}")
            grasps = annotation.get("grasps")
            if not isinstance(grasps, list) or not grasps:
                raise ValueError(f"no GT grasp rectangles: {sample_id}")
            sample, depth_cache = _load_sample(
                row,
                config=config,
                scene_cache_root=scene_cache_root,
                tmp_root=tmp_root,
            )
            result = generate_candidates(
                sample,
                dict(config["sampling"]),
                dict(config["filtering"]),
                num_samples=int(config["generation"]["num_grasp_samples"]),
                top_k=int(config["generation"]["top_k"]),
                seed=sample_seed,
                visualize_sampler=False,
            )
            for stage, candidates in (
                ("raw", result.raw_candidates),
                ("mask_validated", result.mask_validated_candidates),
                ("nms", result.deduplicated_candidates),
            ):
                _validate_stage_primary_keys(stage, sample_id, candidates)
            metadata = _metadata(
                sample,
                result,
                config=config,
                split=args.split,
                sample_seed=sample_seed,
                protocol_identity_sha256=protocol_identity_sha256,
            )
            _save_scorer_bundle(
                staging,
                result.deduplicated_candidates,
                metadata,
            )
            intrinsics = make_camera_intrinsics(
                {
                    "fx": sample.intrinsics.fx,
                    "fy": sample.intrinsics.fy,
                    "cx": sample.intrinsics.cx,
                    "cy": sample.intrinsics.cy,
                    "skew": sample.intrinsics.skew,
                    "height": sample.intrinsics.height,
                    "width": sample.intrinsics.width,
                },
                frame=sample.intrinsics.frame,
            )
            export_intrinsics_file(intrinsics, staging / "camera.intr")
            os.link(depth_cache, staging / "depth_m.npy")
            _save_mask(
                staging / "hifics_mask_processed.png",
                sample.target_mask_processed,
            )
            _atomic_json(staging / "metadata.json", metadata, tmp_root)
            status = (
                SUCCESS_NONEMPTY
                if result.deduplicated_candidates
                else SUCCESS_EMPTY
            )
            summary_row = _summary_row(
                sample,
                result,
                status,
                metadata["failure_reason"],
                sample_seed,
            )
            write_completion_marker(
                staging,
                sample_id=sample_id,
                question_index=sample.question_index,
                configuration_hash=configuration_hash,
                config_file_sha256=config_hash,
                seed=sample_seed,
                sampler_runtime=runtime,
                counts=metadata["counts"],
                status=status,
                required_files=(
                    "candidates.npz",
                    "candidates.json",
                    "metadata.json",
                    "camera.intr",
                    "depth_m.npy",
                    "hifics_mask_processed.png",
                ),
                summary_row=summary_row,
                failure_reason=metadata["failure_reason"],
            )
            raw_oracle, _ = _stage_labels(
                result.raw_candidates, grasps, evaluation_config
            )
            mask_oracle, _ = _stage_labels(
                result.mask_validated_candidates, grasps, evaluation_config
            )
            nms_oracle, nms_labels = _stage_labels(
                result.deduplicated_candidates, grasps, evaluation_config
            )
            label_payload = {
                "schema_version": 2,
                "label_only_gt_artifact": True,
                "split": args.split,
                "protocol_identity_sha256": protocol_identity_sha256,
                "sample": {
                    "sample_id": sample_id,
                    "scene_id": sample.scene_id,
                    "question_index": sample.question_index,
                    "raw_candidate_count": len(result.raw_candidates),
                    "mask_validated_candidate_count": len(
                        result.mask_validated_candidates
                    ),
                    "nms_candidate_count": len(
                        result.deduplicated_candidates
                    ),
                    "raw_oracle": raw_oracle,
                    "mask_validated_oracle": mask_oracle,
                    "nms_oracle": nms_oracle,
                    "valid_empty": not result.deduplicated_candidates,
                },
                "candidate_labels": [
                    {
                        "sample_id": sample_id,
                        "candidate_id": item["candidate_id"],
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "candidate_id"
                        },
                    }
                    for item in nms_labels
                ],
                "annotations_path": str(annotations_path),
                "annotations_sha256": annotations_hash,
                "evaluation_config_path": str(evaluation_config_path),
                "evaluation_config_sha256": evaluation_config_hash,
                "inference_artifact_dependency": False,
            }
            if reference_root is not None:
                _compare_reference(
                    result, staging, reference_root, sample_id
                )
            _write_stage_sidecar(
                _stage_sidecar_path(sidecar_root, sample_id),
                sample_row=row,
                sample_seed=sample_seed,
                protocol_identity_sha256=protocol_identity_sha256,
                result=result,
            )
            # Publishing the label first makes an interruption recoverable:
            # an orphan label is harmless and is replaced on the next attempt.
            _atomic_json(label_path, label_payload, tmp_root)
            os.replace(staging, final)
            summaries.append(summary_row)
            funnel_rows.append(label_payload["sample"])
            fresh += 1
        except Exception as error:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if staging.exists():
                shutil.rmtree(staging)
        if position % args.status_every == 0 or position == len(selected):
            print(
                f"Dex-Net {args.split}: {position}/{len(selected)} "
                f"fresh={fresh} failures={len(failures)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    if failures:
        _atomic_json(output_root / "failures.json", failures, tmp_root)
        raise RuntimeError(f"candidate generation failures: {len(failures)}")
    summaries.sort(key=lambda item: int(item["question_index"]))
    # Preserve the prediction-manifest order, not question-index sorting alone.
    summary_by_id = {row["sample_id"]: row for row in summaries}
    summaries = [summary_by_id[str(row["sample_id"])] for row in selected]
    _atomic_csv(output_root / "summary.csv", summaries, SUMMARY_FIELDS, tmp_root)
    _write_jsonl(output_root / "funnel_labels.jsonl", funnel_rows, tmp_root)
    manifest_rows = [
        {
            "sample_index": int(row["sample_index"]),
            "sample_id": str(row["sample_id"]),
            "question_index": int(row["question_index"]),
            "scene_id": str(row["scene_id"]),
            "query": str(row["query"]),
            "source_status": summary_by_id[str(row["sample_id"])]["status"],
        }
        for row in selected
    ]
    _write_jsonl(output_root / "run_manifest.jsonl", manifest_rows, tmp_root)
    counts = {
        "samples": len(summaries),
        "nonempty_samples": sum(
            row["status"] == SUCCESS_NONEMPTY for row in summaries
        ),
        "empty_samples": sum(row["status"] == SUCCESS_EMPTY for row in summaries),
        "raw_candidates": sum(int(row["raw_candidate_count"]) for row in summaries),
        "mask_validated_candidates": sum(
            int(row["mask_validated_count"]) for row in summaries
        ),
        "nms_candidates": sum(int(row["post_nms_count"]) for row in summaries),
        "raw_oracle": sum(bool(row["raw_oracle"]) for row in funnel_rows),
        "mask_validated_oracle": sum(
            bool(row["mask_validated_oracle"]) for row in funnel_rows
        ),
        "nms_oracle": sum(bool(row["nms_oracle"]) for row in funnel_rows),
        "execution_failures": 0,
    }
    stage_expected_counts = {
        "raw": counts["raw_candidates"],
        "mask_validated": counts["mask_validated_candidates"],
        "nms": counts["nms_candidates"],
    }
    if args.phase == "generate":
        stage_artifacts: dict[str, dict[str, Any]] = {}
        run_status = "CANDIDATES_STAGED"
    else:
        stage_artifacts = _validate_existing_stage_parquets(
            output_root=output_root,
            run_config=existing_run_config,
            expected_counts=stage_expected_counts,
        )
        if stage_artifacts is None:
            missing_sidecars = [
                str(_stage_sidecar_path(sidecar_root, str(row["sample_id"])))
                for row in selected
                if not _stage_sidecar_path(
                    sidecar_root, str(row["sample_id"])
                ).is_file()
            ]
            if missing_sidecars:
                raise RuntimeError(
                    "cannot rebuild candidate-stage Parquets: verified temporary "
                    f"sidecars are missing ({missing_sidecars[:3]})"
                )
            stage_artifacts = _write_candidate_stage_parquets(
                output_root=output_root,
                tmp_root=tmp_root,
                sidecar_root=sidecar_root,
                selected=selected,
                base_seed=int(config["generation"]["seed"]),
                protocol_identity_sha256=protocol_identity_sha256,
            )
        run_status = "COMPLETED"
    run_config = {
        "schema_version": 2,
        "status": run_status,
        "split": args.split,
        "counts": counts,
        "prediction_manifest": str(prediction_manifest),
        "prediction_manifest_sha256": prediction_manifest_hash,
        "annotations": str(annotations_path),
        "annotations_sha256": annotations_hash,
        "config": str(config_path),
        "config_sha256": config_hash,
        "configuration_hash": configuration_hash,
        "evaluation_config": str(evaluation_config_path),
        "evaluation_config_sha256": evaluation_config_hash,
        "sampler_runtime": runtime,
        "sample_seed_mode": SAMPLE_SEED_MODE,
        "seed_namespace": SEED_NAMESPACE,
        "sample_seed_derivation": SAMPLE_SEED_DERIVATION,
        "protocol_identity": protocol_identity,
        "protocol_identity_sha256": protocol_identity_sha256,
        "protocol_family_identity": protocol_family_identity,
        "protocol_family_identity_sha256": protocol_family_identity_sha256,
        "candidate_stage_schema": [
            list(item) for item in CANDIDATE_STAGE_SCHEMA_SPEC
        ],
        "candidate_stage_schema_sha256": _candidate_stage_schema_sha256(),
        "candidate_stage_artifacts": stage_artifacts,
        "compact_storage": {
            "raw_candidates_persisted": True,
            "mask_validated_candidates_persisted": True,
            "nms_candidates_persisted": True,
            "candidate_stage_format": "zstd_parquet",
            "candidate_stage_primary_key": ["sample_id", "candidate_id"],
            "per_sample_raw_json_persisted": False,
            "per_sample_mask_validated_json_persisted": False,
            "per_sample_nms_csv_persisted": False,
            "depth_m_scene_cache_hardlinked": True,
            "gt_labels_separate": True,
            "temporary_root": str(tmp_root),
            "automatic_deletion_scope": "current_run_tmp_only",
        },
        "fresh_samples": (
            int(existing_run_config.get("fresh_samples", 0))
            if args.phase == "finalize"
            else fresh
        ),
        "elapsed_seconds": float(time.perf_counter() - started)
        + (
            float(existing_run_config.get("elapsed_seconds", 0.0))
            if args.phase == "finalize"
            else 0.0
        ),
        "generation_command": (
            existing_run_config.get("generation_command")
            if args.phase == "finalize"
            else " ".join([sys.executable, *sys.argv])
        ),
        "finalization_command": (
            " ".join([sys.executable, *sys.argv])
            if args.phase != "generate"
            else None
        ),
        "selection": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "scene_grouped": args.num_shards is not None,
            "limit": args.limit,
        },
    }
    _atomic_json(output_root / "run_config.json", run_config, tmp_root)
    _atomic_text(
        output_root / "run_command.txt",
        " ".join([sys.executable, *sys.argv]) + "\n",
        tmp_root,
    )
    if run_status == "COMPLETED":
        # Parquets are hash-verified and contain every stage record.  Only the
        # current invocation's temporary sidecars/staging tree is removed.
        resolved_sidecar = sidecar_root.resolve()
        if tmp_root.resolve() not in resolved_sidecar.parents:
            raise RuntimeError("refusing to clean sidecars outside tmp-root")
        shutil.rmtree(sidecar_root)
    print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
