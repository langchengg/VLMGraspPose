"""Strict loader for the two frozen full-run VGN candidate pools."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_SAMPLE_COUNT = 7_675
NPZ_FIELDS = (
    "official_selection_index",
    "score_rank",
    "vgn_quality",
    "voxel_index_ijk",
    "position_task_m",
    "quaternion_task_xyzw",
    "width_m",
    "T_task_grasp",
    "T_camera_grasp",
    "inside_dilated_target_mask",
)


class AnalysisIntegrityError(RuntimeError):
    """Raised when a frozen input run cannot support unbiased analysis."""


@dataclass(frozen=True)
class LoadedAnalysisTables:
    samples: pd.DataFrame
    predicted_candidates: pd.DataFrame
    gt_regenerated_candidates: pd.DataFrame
    integrity: Mapping[str, Any]


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisIntegrityError(f"missing required JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisIntegrityError(f"cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisIntegrityError(f"expected JSON object: {path}")
    return value


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnalysisIntegrityError(
                    f"invalid manifest JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise AnalysisIntegrityError(
                    f"manifest line {line_number} is not an object"
                )
            rows.append(row)
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise AnalysisIntegrityError(
            f"manifest has {len(rows)} rows, expected exactly {EXPECTED_SAMPLE_COUNT}"
        )
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not value for value in sample_ids):
        raise AnalysisIntegrityError("manifest contains an empty sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise AnalysisIntegrityError("manifest contains duplicate sample_id values")
    return rows


def _summary_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "summary.csv"
    if not path.is_file():
        raise AnalysisIntegrityError(f"missing summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise AnalysisIntegrityError(
            f"{path} has {len(rows)} rows, expected {EXPECTED_SAMPLE_COUNT}"
        )
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise AnalysisIntegrityError(f"duplicate/empty sample_id in {path}: {sample_id!r}")
        result[sample_id] = row
    return result


def _validate_run_config(root: Path, expected_mask_source: str) -> dict[str, Any]:
    config = _json(root / "run_config.json")
    required = {
        "repository_commit": "d7af0622433f52ae88ebe81533f12b46b33e951a",
        "checkpoint_sha256": (
            "ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b"
        ),
        "score_source": "official_vgn_processed_quality",
        "custom_reranking": False,
        "tsdf_mode": "single_view_adaptation",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise AnalysisIntegrityError(
                f"{root.name} config {key}={config.get(key)!r}, expected {expected!r}"
            )
    cli = config.get("cli_arguments")
    if not isinstance(cli, dict) or cli.get("mask_source") != expected_mask_source:
        raise AnalysisIntegrityError(
            f"{root.name} mask_source is not {expected_mask_source!r}"
        )
    if int(config.get("expected_manifest_count", -1)) != EXPECTED_SAMPLE_COUNT:
        raise AnalysisIntegrityError(f"{root.name} expected manifest count is not 7,675")
    return config


def _require_terminal(root: Path, sample_id: str, summary: Mapping[str, Any]) -> None:
    if str(summary.get("state")) != "terminal":
        raise AnalysisIntegrityError(f"{root.name}/{sample_id} is not terminal")
    if not (root / "samples" / sample_id / "result.json").is_file():
        raise AnalysisIntegrityError(f"missing result.json for {root.name}/{sample_id}")


def _compare_float_array(
    left: np.ndarray, right: np.ndarray, *, label: str, atol: float = 1e-6
) -> None:
    if left.shape != right.shape or not np.allclose(left, right, atol=atol, rtol=0.0):
        raise AnalysisIntegrityError(f"candidate JSON/NPZ mismatch for {label}")


def _validate_npz(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_inside: bool | None = None,
) -> None:
    if not path.is_file():
        raise AnalysisIntegrityError(f"missing candidate NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(NPZ_FIELDS) - set(payload.files))
        if missing:
            raise AnalysisIntegrityError(f"{path} missing arrays: {missing}")
        count = len(payload["official_selection_index"])
        if count != len(records):
            raise AnalysisIntegrityError(
                f"{path} stores {count} candidates but JSON stores {len(records)}"
            )
        if not records:
            return
        indices = np.asarray(
            [record["official_selection_index"] for record in records], dtype=np.int64
        )
        if not np.array_equal(indices, payload["official_selection_index"]):
            raise AnalysisIntegrityError(f"candidate index mismatch in {path}")
        qualities = np.asarray([record["vgn_quality"] for record in records])
        widths = np.asarray([record["width_m"] for record in records])
        _compare_float_array(qualities, payload["vgn_quality"], label=f"{path}:quality")
        _compare_float_array(widths, payload["width_m"], label=f"{path}:width")
        if expected_inside is not None and not np.all(
            payload["inside_dilated_target_mask"] == expected_inside
        ):
            raise AnalysisIntegrityError(f"unexpected filter membership in {path}")


def _rank_map(records: Iterable[Mapping[str, Any]]) -> dict[int, int]:
    ordered = sorted(
        records,
        key=lambda record: (
            -float(record["vgn_quality"]),
            int(record["official_selection_index"]),
        ),
    )
    return {
        int(record["official_selection_index"]): rank
        for rank, record in enumerate(ordered, start=1)
    }


def _candidate_row(
    record: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    pool_source: str,
    rank_all: int,
    rank_filtered: int | None,
) -> dict[str, Any]:
    task = list(record["position_task_m"])
    camera = list(record["position_camera_m"])
    quaternion = list(record["quaternion_camera_xyzw"])
    projected = record.get("projected_uv")
    return {
        "sample_id": sample["sample_id"],
        "scene_id": sample["scene_id"],
        "dataset_index": int(sample["dataset_index"]),
        "instruction": sample["instruction"],
        "query_type": sample.get("query_type", "unknown"),
        "target_category": sample.get("target_category", "unknown"),
        "pool_source": pool_source,
        "candidate_index_original": int(record["official_selection_index"]),
        "rank_vgn_all": int(rank_all),
        "rank_vgn_pred_filtered": rank_filtered,
        "vgn_quality": float(record["vgn_quality"]),
        "width_m": float(record["width_m"]),
        "position_task_x": float(task[0]),
        "position_task_y": float(task[1]),
        "position_task_z": float(task[2]),
        "position_camera_x": float(camera[0]),
        "position_camera_y": float(camera[1]),
        "position_camera_z": float(camera[2]),
        "quaternion_camera_xyzw": [float(value) for value in quaternion],
        "projected_u_saved": None if projected is None else float(projected[0]),
        "projected_v_saved": None if projected is None else float(projected[1]),
        "pred_filter_pass": bool(record.get("inside_dilated_target_mask", False)),
        "inside_raw_pool_mask": bool(record.get("inside_raw_target_mask", False)),
        "is_baseline_top1": bool(
            sample.get("baseline_hard_filter_index") is not None
            and int(record["official_selection_index"])
            == int(sample["baseline_hard_filter_index"])
        ),
        "is_existing_target_candidate": bool(
            int(record["official_selection_index"])
            in sample.get("target_candidate_indices", set())
        ),
    }


def _load_pool_sample(
    root: Path,
    sample: dict[str, Any],
    *,
    pool_source: str,
) -> list[dict[str, Any]]:
    directory = root / "samples" / str(sample["sample_id"])
    candidates = _json(directory / "candidates.json")
    top1 = _json(directory / "top1.json")
    all_records = candidates.get("all_official_vgn_candidates")
    target_records = candidates.get("target_filtered_vgn_candidates")
    if not isinstance(all_records, list) or not isinstance(target_records, list):
        raise AnalysisIntegrityError(f"invalid candidate collections in {directory}")
    _validate_npz(directory / "candidates_all.npz", all_records)
    _validate_npz(
        directory / "candidates_target.npz", target_records, expected_inside=True
    )
    if len(all_records) != int(candidates.get("official_candidate_count", -1)):
        raise AnalysisIntegrityError(f"official candidate count mismatch in {directory}")
    if len(target_records) != int(candidates.get("target_filtered_candidate_count", -1)):
        raise AnalysisIntegrityError(f"target candidate count mismatch in {directory}")

    indices = [int(record["official_selection_index"]) for record in all_records]
    if len(indices) != len(set(indices)):
        raise AnalysisIntegrityError(f"duplicate candidate index in {directory}")
    target_indices = {int(record["official_selection_index"]) for record in target_records}
    if not target_indices.issubset(set(indices)):
        raise AnalysisIntegrityError(f"target candidates not contained in official pool: {directory}")
    sample["target_candidate_indices"] = target_indices

    all_ranks = _rank_map(all_records)
    filtered = [record for record in all_records if int(record["official_selection_index"]) in target_indices]
    filtered_ranks = _rank_map(filtered)
    expected_top1 = None
    if filtered:
        expected_top1 = min(filtered, key=lambda record: (all_ranks[int(record["official_selection_index"])],))
        expected_top1 = int(expected_top1["official_selection_index"])
    actual_candidate = top1.get("candidate")
    actual_top1 = (
        int(actual_candidate["official_selection_index"])
        if isinstance(actual_candidate, dict)
        else None
    )
    if actual_top1 != expected_top1:
        raise AnalysisIntegrityError(
            f"recomputed baseline top1 mismatch for {directory}: {expected_top1} != {actual_top1}"
        )
    if pool_source == "predicted_mask":
        sample["baseline_hard_filter_index"] = actual_top1
        sample["baseline_vgn_all_index"] = (
            min(all_records, key=lambda record: (all_ranks[int(record["official_selection_index"])],))["official_selection_index"]
            if all_records
            else None
        )
    return [
        _candidate_row(
            record,
            sample=sample,
            pool_source=pool_source,
            rank_all=all_ranks[int(record["official_selection_index"])],
            rank_filtered=filtered_ranks.get(int(record["official_selection_index"])),
        )
        for record in all_records
    ]


def load_analysis_tables(
    pred_output: Path | str,
    gt_oracle_output: Path | str,
    manifest: Path | str,
) -> LoadedAnalysisTables:
    """Load and cross-check every sample and official candidate from both runs."""

    pred_root = Path(pred_output).expanduser().resolve()
    gt_root = Path(gt_oracle_output).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_rows = _manifest_rows(manifest_path)
    pred_config = _validate_run_config(pred_root, "predicted")
    gt_config = _validate_run_config(gt_root, "gt-oracle")
    pred_summary = _summary_rows(pred_root)
    gt_summary = _summary_rows(gt_root)
    manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
    if manifest_ids != set(pred_summary) or manifest_ids != set(gt_summary):
        raise AnalysisIntegrityError("manifest/predicted/oracle sample_id sets differ")

    comparable_keys = (
        "repository_commit",
        "checkpoint_sha256",
        "tsdf_mode",
        "official_postprocessing",
    )
    differences = {
        key: (pred_config.get(key), gt_config.get(key))
        for key in comparable_keys
        if pred_config.get(key) != gt_config.get(key)
    }
    if differences:
        raise AnalysisIntegrityError(f"run configs differ beyond mask source: {differences}")

    sample_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    observed_gt_paths: set[str] = set()
    for manifest_row in sorted(manifest_rows, key=lambda row: int(row["sample_index"])):
        sample_id = str(manifest_row["sample_id"])
        pred = pred_summary[sample_id]
        oracle = gt_summary[sample_id]
        _require_terminal(pred_root, sample_id, pred)
        _require_terminal(gt_root, sample_id, oracle)
        expected_scene = str(manifest_row["scene_id"])
        expected_index = int(manifest_row["sample_index"])
        for label, summary in (("predicted", pred), ("oracle", oracle)):
            if str(summary.get("scene_id")) != expected_scene:
                raise AnalysisIntegrityError(f"{sample_id} {label} scene_id mismatch")
            if int(float(summary.get("dataset_index", -1))) != expected_index:
                raise AnalysisIntegrityError(f"{sample_id} {label} dataset_index mismatch")
            if str(summary.get("instruction")) != str(manifest_row["query"]):
                raise AnalysisIntegrityError(f"{sample_id} {label} instruction mismatch")
        pred_result = _json(pred_root / "samples" / sample_id / "result.json")
        oracle_result = _json(gt_root / "samples" / sample_id / "result.json")
        bundle = Path(str(manifest_row["output_dir"])).expanduser().resolve()
        metadata = _json(bundle / "metadata.json")
        workspace = _json(pred_root / "samples" / sample_id / "workspace_frame.json")
        depth_metadata = workspace.get("depth")
        if not isinstance(depth_metadata, dict) or depth_metadata.get("depth_unit") != "mm":
            raise AnalysisIntegrityError(
                f"{sample_id} does not explicitly record raw OCID depth unit as mm"
            )
        gt_mask_path = str(Path(str(pred_result["gt_mask_path"])).resolve())
        if gt_mask_path in observed_gt_paths:
            raise AnalysisIntegrityError(f"GT mask mapping is not unique: {gt_mask_path}")
        if not Path(gt_mask_path).is_file():
            raise AnalysisIntegrityError(f"missing mapped GT mask: {gt_mask_path}")
        observed_gt_paths.add(gt_mask_path)
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": expected_scene,
            "dataset_index": expected_index,
            "instruction": str(manifest_row["query"]),
            "query_type": str(pred_result.get("query_type") or "unknown"),
            "target_category": str(pred_result.get("target_category") or "unknown"),
            "target_name": str(pred_result.get("target_name") or "unknown"),
            "view": str(pred_result.get("view") or metadata.get("camera_view_from_sequence_path") or "unknown"),
            "pred_status": str(pred_result["status"]),
            "gt_regenerated_status": str(oracle_result["status"]),
            "mask_iou": float(pred_result.get("mask_iou") or 0.0),
            "pred_mask_area_px": int(float(pred_result.get("pred_mask_area_px") or 0)),
            "gt_mask_area_px": int(float(pred_result.get("gt_mask_area_px") or 0)),
            "valid_target_depth_points": int(float(pred_result.get("valid_target_depth_points") or 0)),
            "fit_rmse_px": float(pred_result.get("fit_rmse_px") or 0.0),
            "fit_p95_px": float(pred_result.get("fit_p95_px") or 0.0),
            "support_plane_residual": float(pred_result.get("support_plane_residual") or 0.0),
            "processing_time_total": float(pred_result.get("processing_time_total") or 0.0),
            "gt_regenerated_processing_time_total": float(oracle_result.get("processing_time_total") or 0.0),
            "bundle_dir": str(bundle),
            "rgb_path": str(metadata["source_rgb"]),
            "depth_path": str(metadata["source_depth"]),
            "pred_mask_path": str(bundle / "target_mask.png"),
            "gt_mask_path": gt_mask_path,
            "workspace_frame_path": str(pred_root / "samples" / sample_id / "workspace_frame.json"),
            "support_plane_path": str(pred_root / "samples" / sample_id / "support_plane.json"),
            "intrinsics": workspace["intrinsics"],
            "T_camera_task": workspace["T_camera_task"],
            "target_cloud": workspace["target_cloud"],
            "depth_unit": "mm",
            "depth_scale": float(depth_metadata["depth_scale"]),
            "depth_metric_p50_m": float(depth_metadata["metric_p50_m"]),
            "target_mask_valid_depth_ratio": float(
                workspace["mask"]["valid_depth_ratio"]
            ),
            "intrinsics_source": str(workspace["intrinsics"].get("source", "unknown")),
            "baseline_hard_filter_index": None,
            "baseline_vgn_all_index": None,
            "target_candidate_indices": set(),
        }
        pred_candidates = _load_pool_sample(
            pred_root, sample, pool_source="predicted_mask"
        )
        pred_count = len(pred_candidates)
        pred_target_count = sum(bool(row["pred_filter_pass"]) for row in pred_candidates)
        if pred_count != int(float(pred_result.get("official_candidate_count") or 0)):
            raise AnalysisIntegrityError(f"{sample_id} predicted summary candidate mismatch")
        if pred_target_count != int(float(pred_result.get("target_candidate_count") or 0)):
            raise AnalysisIntegrityError(f"{sample_id} predicted target count mismatch")

        oracle_sample = dict(sample)
        oracle_sample["baseline_hard_filter_index"] = None
        oracle_sample["target_candidate_indices"] = set()
        regenerated = _load_pool_sample(
            gt_root, oracle_sample, pool_source="gt_regenerated"
        )
        if len(regenerated) != int(float(oracle_result.get("official_candidate_count") or 0)):
            raise AnalysisIntegrityError(f"{sample_id} oracle summary candidate mismatch")

        sample.update(
            n_official_candidates=pred_count,
            n_pred_filtered_candidates=pred_target_count,
            n_gt_regenerated_official_candidates=len(regenerated),
            n_gt_regenerated_filtered_candidates=sum(
                bool(row["pred_filter_pass"]) for row in regenerated
            ),
        )
        # Set baseline flags after the top-1 index is known.
        for row in pred_candidates:
            row["is_baseline_top1"] = (
                sample["baseline_hard_filter_index"] is not None
                and row["candidate_index_original"] == sample["baseline_hard_filter_index"]
            )
        sample["target_candidate_indices"] = sorted(sample["target_candidate_indices"])
        sample_rows.append(sample)
        predicted_rows.extend(pred_candidates)
        oracle_rows.extend(regenerated)

    samples = pd.DataFrame(sample_rows).sort_values("dataset_index").reset_index(drop=True)
    predicted_candidates = pd.DataFrame(predicted_rows)
    gt_candidates = pd.DataFrame(oracle_rows)
    integrity = {
        "manifest_count": len(samples),
        "scene_count": int(samples["scene_id"].nunique()),
        "predicted_candidate_count": len(predicted_candidates),
        "gt_regenerated_candidate_count": len(gt_candidates),
        "predicted_signature": pred_config.get("run_signature_sha256"),
        "gt_regenerated_signature": gt_config.get("run_signature_sha256"),
        "comparable_except_mask_source": True,
        "top1_recomputed_mismatch_count": 0,
        "candidate_npz_mismatch_count": 0,
    }
    return LoadedAnalysisTables(samples, predicted_candidates, gt_candidates, integrity)


__all__ = [
    "AnalysisIntegrityError",
    "EXPECTED_SAMPLE_COUNT",
    "LoadedAnalysisTables",
    "load_analysis_tables",
]
