#!/usr/bin/env python3
"""Create a rigorously checked 20-sample AnyGrasp input review subset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from tools.export_anygrasp_inputs import _parse_pcd, sha256_file
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from export_anygrasp_inputs import _parse_pcd, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZE = 5
EXPECTED_SELECTION = 20
SPATIAL_TERMS = (
    " left ",
    " right ",
    " above ",
    " below ",
    " under ",
    " over ",
    " behind ",
    " in front ",
    " next to ",
    " beside ",
    " between ",
    " nearest ",
    " closest ",
    " farthest ",
    " middle ",
    " center ",
)
REQUIRED_BUNDLE_ARTIFACTS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
    "checksums.sha256",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid_jsonl:{path}:{line_number}:{error}") from error
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _index_unique(
    rows: list[dict[str, Any]], field: str, source: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if field not in row:
            raise ValueError(f"{source}_missing_{field}")
        key = str(row[field])
        if key in result:
            raise ValueError(f"{source}_duplicate_{field}:{key}")
        result[key] = row
    return result


def _prediction_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("samples")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("prediction_manifest_samples_not_list")
    return rows


def _normalise_query(query: str) -> str:
    return " " + " ".join(str(query).lower().split()) + " "


def _has_spatial_language(query: str) -> bool:
    normalised = _normalise_query(query)
    return any(term in normalised for term in SPATIAL_TERMS)


def _target_category(target: str | None) -> str:
    value = str(target or "unknown")
    prefix, separator, suffix = value.rpartition("_")
    return prefix if separator and suffix.isdigit() else value


def _source_paths(
    source_root: Path, original_record: dict[str, Any]
) -> dict[str, Path]:
    sequence, image_name = str(original_record["image_filename"]).split(",", 1)
    return {
        "rgb": source_root / sequence / "rgb" / image_name,
        "depth": source_root / sequence / "depth" / image_name,
        "instance": source_root
        / sequence
        / "seg_mask_instances_combi"
        / image_name,
        "pcd": source_root
        / sequence
        / "pcd"
        / Path(image_name).with_suffix(".pcd").name,
    }


def _instance_count(
    path: Path, cache: dict[str, tuple[int, str | None]]
) -> tuple[int, str | None]:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    try:
        instance_map = np.asarray(Image.open(path))
        if instance_map.ndim != 2:
            result = (0, f"instance_map_not_2d:{instance_map.shape}")
        else:
            result = (int(np.count_nonzero(np.unique(instance_map) > 0)), None)
    except (OSError, ValueError) as error:
        result = (0, f"instance_map_unreadable:{type(error).__name__}")
    cache[key] = result
    return result


def _join_sources(
    metrics_rows: list[dict[str, str]],
    frozen_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    anygrasp_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    source_root: Path,
) -> list[dict[str, Any]]:
    frozen_by_index = {str(index): row for index, row in enumerate(frozen_rows)}
    prediction_by_index = _index_unique(
        prediction_rows, "sample_index", "prediction_manifest"
    )
    anygrasp_by_id = _index_unique(anygrasp_rows, "sample_id", "anygrasp_manifest")
    original_by_question = _index_unique(
        original_rows, "question_index", "original_manifest"
    )
    instance_cache: dict[str, tuple[int, str | None]] = {}
    joined = []
    seen_metric_indices = set()
    for metric in metrics_rows:
        sample_index = str(metric.get("sample_index"))
        if sample_index in seen_metric_indices:
            raise ValueError(f"metrics_duplicate_sample_index:{sample_index}")
        seen_metric_indices.add(sample_index)
        frozen = frozen_by_index.get(sample_index)
        prediction = prediction_by_index.get(sample_index)
        if frozen is None or prediction is None:
            continue
        if str(metric.get("sample_id")) != str(frozen.get("num")):
            raise ValueError(f"metrics_frozen_sample_id_mismatch:{sample_index}")
        if str(metric.get("text")) != str(frozen.get("text")):
            raise ValueError(f"metrics_frozen_query_mismatch:{sample_index}")
        stable_id = str(prediction.get("stable_sample_id"))
        anygrasp = anygrasp_by_id.get(stable_id)
        if anygrasp is None or not bool(
            anygrasp.get("ready_for_anygrasp", anygrasp.get("ready", False))
        ):
            continue
        question_index = str(frozen.get("question_index"))
        original = original_by_question.get(question_index)
        if original is None:
            raise ValueError(f"original_question_missing:{question_index}")
        scene_id = str(frozen.get("scene_id"))
        query = str(frozen.get("text"))
        if str(prediction.get("scene_id")) != scene_id:
            raise ValueError(f"prediction_scene_mismatch:{stable_id}")
        if str(prediction.get("query")) != query:
            raise ValueError(f"prediction_query_mismatch:{stable_id}")
        if str(prediction.get("question_index")) != question_index:
            raise ValueError(f"prediction_question_index_mismatch:{stable_id}")
        if str(anygrasp.get("scene_id")) != scene_id:
            raise ValueError(f"anygrasp_manifest_scene_mismatch:{stable_id}")
        if str(anygrasp.get("query")) != query:
            raise ValueError(f"anygrasp_manifest_query_mismatch:{stable_id}")
        if str(anygrasp.get("question_index")) != question_index:
            raise ValueError(f"anygrasp_manifest_question_index_mismatch:{stable_id}")
        if str(original.get("image_filename")) != scene_id:
            raise ValueError(f"original_scene_mismatch:{stable_id}")
        if str(original.get("question")) != query:
            raise ValueError(f"original_query_mismatch:{stable_id}")
        paths = _source_paths(source_root, original)
        clutter_count, clutter_error = _instance_count(paths["instance"], instance_cache)
        joined.append(
            {
                "sample_index": int(sample_index),
                "evaluation_sample_id": str(metric["sample_id"]),
                "sample_id": stable_id,
                "question_index": int(question_index),
                "scene_id": scene_id,
                "query": query,
                "iou": float(metric["iou"]),
                "target_category": _target_category(original.get("target")),
                "scene_instance_count": clutter_count,
                "clutter_count_error": clutter_error,
                "spatial_language": _has_spatial_language(query),
                "prediction_manifest_row": prediction,
                "anygrasp_manifest_row": anygrasp,
                "original_record": original,
                "source_paths": paths,
            }
        )
    return joined


def _take_group(
    ordered: list[dict[str, Any]],
    selected_ids: set[str],
    group: str,
) -> list[dict[str, Any]]:
    result = []
    for candidate in ordered:
        if candidate["sample_id"] in selected_ids:
            continue
        row = dict(candidate)
        row["selection_group"] = group
        row["selection_rank"] = len(result) + 1
        result.append(row)
        selected_ids.add(row["sample_id"])
        if len(result) == GROUP_SIZE:
            return result
    return result


def _select_diverse(
    candidates: list[dict[str, Any]], selected: list[dict[str, Any]], selected_ids: set[str]
) -> list[dict[str, Any]]:
    used_scenes = {row["scene_id"] for row in selected}
    used_categories = {row["target_category"] for row in selected}
    ordered = sorted(
        (row for row in candidates if row["sample_id"] not in selected_ids),
        key=lambda row: (
            -int(row["spatial_language"]),
            -int(row["scene_instance_count"]),
            row["sample_index"],
        ),
    )
    result = []
    remaining = list(ordered)
    while remaining and len(result) < GROUP_SIZE:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                int(remaining[index]["spatial_language"]),
                int(remaining[index]["scene_id"] not in used_scenes),
                int(remaining[index]["target_category"] not in used_categories),
                int(remaining[index]["scene_instance_count"]),
                -int(remaining[index]["sample_index"]),
            ),
        )
        candidate = remaining.pop(best_index)
        row = dict(candidate)
        row["selection_group"] = "diverse_clutter_spatial"
        row["selection_rank"] = len(result) + 1
        result.append(row)
        selected_ids.add(row["sample_id"])
        used_scenes.add(row["scene_id"])
        used_categories.add(row["target_category"])
    return result


def select_verified_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < EXPECTED_SELECTION:
        return []
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    by_highest = sorted(candidates, key=lambda row: (-row["iou"], row["sample_index"]))
    selected.extend(_take_group(by_highest, selected_ids, "highest_iou"))
    median_iou = float(np.median([row["iou"] for row in candidates]))
    by_median = sorted(
        candidates,
        key=lambda row: (abs(row["iou"] - median_iou), row["sample_index"]),
    )
    selected.extend(_take_group(by_median, selected_ids, "nearest_median"))
    by_lowest = sorted(candidates, key=lambda row: (row["iou"], row["sample_index"]))
    selected.extend(_take_group(by_lowest, selected_ids, "lowest_iou"))
    selected.extend(_select_diverse(candidates, selected, selected_ids))
    return selected if len(selected) == EXPECTED_SELECTION else []


def _verify_checksum_manifest(bundle: Path) -> list[str]:
    blockers = []
    checksum_path = bundle / "checksums.sha256"
    if not checksum_path.is_file():
        return ["checksums_missing"]
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            blockers.append(f"checksum_line_invalid:{line_number}")
            continue
        expected, name = parts[0], parts[1].strip()
        artifact = bundle / name
        if not artifact.is_file():
            blockers.append(f"checksum_artifact_missing:{name}")
        elif sha256_file(artifact) != expected:
            blockers.append(f"checksum_mismatch:{name}")
    return blockers


def _validate_selected(
    row: dict[str, Any],
    run_dir: Path,
    anygrasp_root: Path,
) -> dict[str, Any]:
    result = {
        key: row[key]
        for key in (
            "sample_index",
            "evaluation_sample_id",
            "sample_id",
            "question_index",
            "scene_id",
            "query",
            "iou",
            "target_category",
            "scene_instance_count",
            "spatial_language",
            "selection_group",
            "selection_rank",
        )
    }
    blockers = []
    if row.get("clutter_count_error"):
        blockers.append(row["clutter_count_error"])
    bundle = anygrasp_root / row["sample_id"]
    for name in REQUIRED_BUNDLE_ARTIFACTS:
        path = bundle / name
        if not path.is_file():
            blockers.append(f"bundle_artifact_missing:{name}")
        elif path.is_symlink():
            blockers.append(f"bundle_artifact_is_symlink:{name}")
    prediction_metadata_path = run_dir / "predictions" / row["sample_id"] / "sample_metadata.json"
    if not prediction_metadata_path.is_file():
        blockers.append("prediction_metadata_missing")
        prediction_metadata = {}
    else:
        try:
            prediction_metadata = _read_json(prediction_metadata_path)
        except (OSError, json.JSONDecodeError) as error:
            prediction_metadata = {}
            blockers.append(f"prediction_metadata_unreadable:{type(error).__name__}")
    if prediction_metadata.get("stable_sample_id") != row["sample_id"]:
        blockers.append("prediction_metadata_sample_id_mismatch")
    if str(prediction_metadata.get("scene_id")) != row["scene_id"]:
        blockers.append("prediction_metadata_scene_mismatch")
    if str(prediction_metadata.get("query")) != row["query"]:
        blockers.append("prediction_metadata_query_mismatch")
    if str(prediction_metadata.get("question_index")) != str(row["question_index"]):
        blockers.append("prediction_metadata_question_index_mismatch")

    if blockers:
        result.update(
            {
                "verified": False,
                "blockers": blockers,
                "target_point_count": 0,
                "target_mask_pixel_count": 0,
                "projection_rmse_px": None,
                "projection_p95_px": None,
                "point_cloud_depth_p95_mm": None,
            }
        )
        return result

    blockers.extend(_verify_checksum_manifest(bundle))
    try:
        rgb = np.asarray(Image.open(bundle / "color.png"))
        depth = np.asarray(Image.open(bundle / "depth.png"))
        mask = np.asarray(Image.open(bundle / "target_mask.png"))
        probability = np.load(
            bundle / "target_probability.npy", allow_pickle=False, mmap_mode="r"
        )
        intrinsics = _read_json(bundle / "intrinsics.json")
        metadata = _read_json(bundle / "metadata.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        blockers.append(f"bundle_unreadable:{type(error).__name__}")
        result.update(
            {
                "verified": False,
                "blockers": blockers,
                "target_point_count": 0,
                "target_mask_pixel_count": 0,
                "projection_rmse_px": None,
                "projection_p95_px": None,
                "point_cloud_depth_p95_mm": None,
            }
        )
        return result

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        blockers.append(f"rgb_not_uint8_rgb:{rgb.dtype}:{rgb.shape}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        blockers.append(f"depth_not_raw_uint16:{depth.dtype}:{depth.shape}")
    if mask.dtype != np.uint8 or mask.ndim != 2:
        blockers.append(f"target_mask_not_uint8_2d:{mask.dtype}:{mask.shape}")
    if probability.dtype != np.float32 or probability.ndim != 2:
        blockers.append(f"probability_not_float32_2d:{probability.dtype}:{probability.shape}")
    elif not np.isfinite(probability).all():
        blockers.append("probability_nonfinite")
    shapes = {
        tuple(rgb.shape[:2]) if rgb.ndim >= 2 else None,
        tuple(depth.shape) if depth.ndim == 2 else None,
        tuple(mask.shape) if mask.ndim == 2 else None,
        tuple(probability.shape) if probability.ndim == 2 else None,
    }
    shapes.discard(None)
    if len(shapes) != 1:
        blockers.append("bundle_dimension_mismatch")
    mask_values = set(np.unique(mask).tolist()) if mask.ndim == 2 else set()
    if not mask_values <= {0, 1, 255}:
        blockers.append(f"target_mask_not_binary:{sorted(mask_values)[:20]}")
    target_mask = mask > 0 if mask.ndim == 2 else np.zeros(depth.shape, dtype=bool)
    target_mask_count = int(target_mask.sum())
    if target_mask_count == 0:
        blockers.append("target_mask_empty")

    if metadata.get("ready_for_anygrasp") is not True:
        blockers.append("bundle_metadata_not_ready_for_anygrasp")
    if str(metadata.get("sample_id")) != row["sample_id"]:
        blockers.append("bundle_metadata_sample_id_mismatch")
    if str(metadata.get("scene_id")) != row["scene_id"]:
        blockers.append("bundle_metadata_scene_mismatch")
    if str(metadata.get("query")) != row["query"]:
        blockers.append("bundle_metadata_query_mismatch")
    if str(metadata.get("question_index")) != str(row["question_index"]):
        blockers.append("bundle_metadata_question_index_mismatch")
    if metadata.get("anygrasp_inference_ran") is not False:
        blockers.append("bundle_metadata_inference_claim_invalid")
    if metadata.get("oracle_artifacts_exported") is not False:
        blockers.append("bundle_metadata_oracle_flag_invalid")
    if intrinsics.get("source") != "derived_from_organized_pcd":
        blockers.append("intrinsics_source_invalid")
    if intrinsics.get("factory_calibration") is not False:
        blockers.append("factory_calibration_claim_invalid")
    if float(intrinsics.get("depth_scale", 0)) != 1000.0:
        blockers.append("depth_scale_not_1000")
    if intrinsics.get("depth_unit") != "millimetres":
        blockers.append("depth_unit_not_millimetres")
    if intrinsics.get("pcd_coordinate_unit") != "metres":
        blockers.append("pcd_unit_not_metres")
    if intrinsics.get("depth_scale_verified") is not True:
        blockers.append("bundle_depth_scale_not_verified")
    if float(intrinsics.get("fit_p95_px", float("inf"))) > 2.0:
        blockers.append("bundle_intrinsics_fit_p95_exceeds_2px")
    if (bundle / "language.txt").read_text() != row["query"]:
        blockers.append("bundle_language_query_mismatch")

    projection_rmse = projection_p95 = depth_p95 = None
    target_point_count = 0
    try:
        records, pcd_header = _parse_pcd(row["source_paths"]["pcd"])
        if depth.shape != (pcd_header["height"], pcd_header["width"]):
            raise ValueError("pcd_bundle_shape_mismatch")
        x = records["x"].astype(np.float64)
        y = records["y"].astype(np.float64)
        z = records["z"].astype(np.float64)
        yy, xx = np.indices(depth.shape, dtype=np.float64)
        valid_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)
        valid = valid_xyz & (depth > 0)
        if not np.any(valid):
            raise ValueError("pcd_no_valid_depth_correspondences")
        depth_delta = np.abs(z[valid] * 1000.0 - depth[valid].astype(np.float64))
        depth_p95 = float(np.percentile(depth_delta, 95))
        if depth_p95 > 1.1:
            blockers.append(f"point_cloud_metric_scale_unverified:{depth_p95:.6f}mm")
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        residual = np.sqrt(
            (fx * x[valid] / z[valid] + cx - xx[valid]) ** 2
            + (fy * y[valid] / z[valid] + cy - yy[valid]) ** 2
        )
        projection_rmse = float(np.sqrt(np.mean(residual**2)))
        projection_p95 = float(np.percentile(residual, 95))
        if projection_p95 > 2.0:
            blockers.append(f"projection_p95_exceeds_2px:{projection_p95:.6f}")
        target_point_count = int((target_mask & valid).sum())
        if target_point_count == 0:
            blockers.append("target_point_count_zero")
        # Consumer depth images legitimately contain sparse zero/invalid pixels.
        # Keep their coverage as a diagnostic, but only block when the predicted
        # target contains no reconstructable 3D point at all.
    except (OSError, ValueError, KeyError) as error:
        blockers.append(f"pcd_alignment_validation_failed:{error}")

    result.update(
        {
            "verified": not blockers,
            "blockers": blockers,
            "target_point_count": target_point_count,
            "target_mask_pixel_count": target_mask_count,
            "target_valid_point_fraction": (
                target_point_count / target_mask_count if target_mask_count else 0.0
            ),
            "projection_rmse_px": projection_rmse,
            "projection_p95_px": projection_p95,
            "point_cloud_depth_p95_mm": depth_p95,
            "bundle_fit_rmse_px": intrinsics.get("fit_rmse_px"),
            "bundle_fit_p95_px": intrinsics.get("fit_p95_px"),
            "bundle_path": str(bundle),
        }
    )
    return result


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hard_link"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"source_paths"}}


def _write_manifests(output: Path, rows: list[dict[str, Any]]) -> None:
    clean_rows = [_serializable_row(row) for row in rows]
    (output / "selection_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in clean_rows)
    )
    fields = (
        "selection_group",
        "selection_rank",
        "sample_index",
        "sample_id",
        "question_index",
        "scene_id",
        "query",
        "iou",
        "target_category",
        "scene_instance_count",
        "spatial_language",
        "verified",
        "blockers",
        "target_point_count",
        "target_mask_pixel_count",
        "target_valid_point_fraction",
        "bundle_fit_rmse_px",
        "bundle_fit_p95_px",
        "projection_rmse_px",
        "projection_p95_px",
        "point_cloud_depth_p95_mm",
    )
    with (output / "selection_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in clean_rows:
            csv_row = {field: row.get(field) for field in fields}
            csv_row["blockers"] = ";".join(row.get("blockers", []))
            writer.writerow(csv_row)


def _write_index(output: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        sample_id = html.escape(str(row["sample_id"]))
        cards.append(
            "<article>"
            f"<h2>{html.escape(row['selection_group'])} #{row['selection_rank']}</h2>"
            f"<img src=\"{sample_id}/color.png\" alt=\"Source RGB for {sample_id}\">"
            f"<img src=\"{sample_id}/target_mask.png\" alt=\"Predicted mask for {sample_id}\">"
            f"<p><code>{sample_id}</code> · IoU {row['iou']:.6f}</p>"
            f"<p>{html.escape(str(row['query']))}</p>"
            "</article>"
        )
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>AnyGrasp verified subset review index</title>"
        "<style>body{font-family:sans-serif}main{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem}"
        "article{border:1px solid #bbb;padding:1rem}img{width:48%;image-rendering:auto}</style>"
        "</head><body><h1>AnyGrasp verified subset review index</h1>"
        "<p>Predicted target visual correspondence requires human inspection.</p><main>"
        + "".join(cards)
        + "</main></body></html>"
    )
    (output / "index.html").write_text(document)


def _report_text(
    status: str,
    rows: list[dict[str, Any]],
    group_counts: dict[str, int],
    global_blockers: list[str],
) -> str:
    lines = [
        "# AnyGrasp verified subset report",
        "",
        f"- Status: **{status}**",
        f"- Selected samples: {len(rows)}",
        f"- Verified samples: {sum(bool(row.get('verified')) for row in rows)}",
        "- AnyGrasp inference ran: false",
        "",
        "## Group counts",
        "",
    ]
    for group in (
        "highest_iou",
        "nearest_median",
        "lowest_iou",
        "diverse_clutter_spatial",
    ):
        lines.append(f"- {group}: {group_counts.get(group, 0)}")
    if global_blockers:
        lines.extend(("", "## Global blockers", ""))
        lines.extend(f"- {blocker}" for blocker in global_blockers)
    lines.extend(
        (
            "",
            "## Selected samples and geometric diagnostics",
            "",
            "| Group | Sample ID | IoU | Fit p95 px | Projection p95 px | Target valid points | Valid fraction | Clutter instances |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in rows:
        lines.append(
            f"| {row['selection_group']} | `{row['sample_id']}` | {row['iou']:.6f} | "
            f"{row.get('bundle_fit_p95_px')} | {row.get('projection_p95_px')} | "
            f"{row.get('target_point_count', 0)} | "
            f"{row.get('target_valid_point_fraction', 0.0):.6f} | "
            f"{row['scene_instance_count']} |"
        )
    lines.extend(
        (
            "",
            "## Caveats",
            "",
            "- Target visual correspondence requires human inspection; this tool does not certify semantic or visual correctness automatically.",
            "- Sparse zero/invalid depth pixels inside a predicted mask are reported through the valid-point fraction; a sample is blocked only when it has no reconstructable target point.",
            "- Factory calibration and camera/robot extrinsics are unavailable; stored intrinsics are effective pinhole fits derived from organized PCDs.",
            "- Full-DoF AnyGrasp generation remains blocked until the licensed SDK, checkpoint, compatible CUDA runtime, and required extrinsic frame mapping are available.",
            "- No AnyGrasp inference was run and no full-DoF grasp poses were generated.",
            "",
        )
    )
    return "\n".join(lines)


def _materialize_subset(
    output_dir: Path,
    anygrasp_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing subset: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary subset path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for row in rows:
            source_bundle = anygrasp_root / row["sample_id"]
            target_bundle = temporary / row["sample_id"]
            target_bundle.mkdir()
            modes = {}
            for source in sorted(source_bundle.iterdir()):
                if not source.is_file() or source.is_symlink():
                    continue
                modes[source.name] = _link_or_copy(source, target_bundle / source.name)
            row["materialization"] = modes
        _write_manifests(temporary, rows)
        _write_index(temporary, rows)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def create_verified_subset(
    *,
    run_dir: Path,
    source_root: Path,
    metrics: Path | None = None,
    frozen: Path | None = None,
    prediction_manifest: Path | None = None,
    anygrasp_manifest: Path | None = None,
    original: Path | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    validation_only: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    source_root = Path(source_root).resolve()
    metrics = Path(metrics or run_dir / "evaluation" / "per_sample_metrics.csv").resolve()
    frozen = Path(frozen or run_dir / "ocid_vlg_test.json").resolve()
    prediction_manifest = Path(
        prediction_manifest or run_dir / "predictions" / "export_manifest.json"
    ).resolve()
    anygrasp_root = run_dir / "anygrasp_input_predicted_mask"
    anygrasp_manifest = Path(
        anygrasp_manifest or anygrasp_root / "manifest.jsonl"
    ).resolve()
    original = Path(
        original or source_root / "refer" / "unique" / "test_expressions.json"
    ).resolve()
    output_dir = Path(output_dir or run_dir / "anygrasp_verified_subset").resolve()
    report_path = Path(
        report_path or REPO_ROOT / "reports" / "anygrasp_verified_subset_report.md"
    ).resolve()
    required_sources = {
        "formal_metrics": metrics,
        "frozen_manifest": frozen,
        "prediction_manifest": prediction_manifest,
        "anygrasp_manifest": anygrasp_manifest,
        "original_manifest": original,
    }
    missing = [f"{name}_missing:{path}" for name, path in required_sources.items() if not path.is_file()]
    if missing:
        return {
            "status": "BLOCKED",
            "selected": 0,
            "verified": 0,
            "group_counts": {},
            "global_blockers": missing,
            "rows": [],
            "validation_only": validation_only,
            "anygrasp_inference_ran": False,
        }

    metrics_rows = _read_csv(metrics)
    frozen_rows = _read_json(frozen)
    prediction_rows = _prediction_rows(_read_json(prediction_manifest))
    anygrasp_rows = _read_jsonl(anygrasp_manifest)
    original_payload = _read_json(original)
    original_rows = original_payload.get("data")
    if not isinstance(frozen_rows, list) or not isinstance(original_rows, list):
        raise ValueError("frozen_or_original_manifest_not_list")
    candidates = _join_sources(
        metrics_rows,
        frozen_rows,
        prediction_rows,
        anygrasp_rows,
        original_rows,
        source_root,
    )
    selected = select_verified_candidates(candidates)
    global_blockers = []
    if len(selected) != EXPECTED_SELECTION:
        global_blockers.append(
            f"insufficient_ready_unique_candidates:{len(candidates)};required:{EXPECTED_SELECTION}"
        )
        validated = []
    else:
        validated = [
            _validate_selected(row, run_dir, anygrasp_root) for row in selected
        ]
    group_counts = dict(Counter(row["selection_group"] for row in validated))
    if len(validated) != EXPECTED_SELECTION or any(
        not row["verified"] for row in validated
    ):
        status = "BLOCKED"
    else:
        status = "DONE"

    if not validation_only:
        if status == "DONE":
            _materialize_subset(output_dir, anygrasp_root, validated)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            _report_text(status, validated, group_counts, global_blockers)
        )
    return {
        "status": status,
        "selected": len(validated),
        "verified": sum(bool(row.get("verified")) for row in validated),
        "group_counts": group_counts,
        "global_blockers": global_blockers,
        "rows": validated,
        "validation_only": validation_only,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "anygrasp_inference_ran": False,
        "visual_correspondence_automatically_certified": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG",
    )
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--frozen-manifest", dest="frozen", type=Path)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--anygrasp-manifest", type=Path)
    parser.add_argument("--original-manifest", dest="original", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_verified_subset(
        run_dir=args.run_dir,
        source_root=args.source_root,
        metrics=args.metrics,
        frozen=args.frozen,
        prediction_manifest=args.prediction_manifest,
        anygrasp_manifest=args.anygrasp_manifest,
        original=args.original,
        output_dir=args.output_dir,
        report_path=args.report_path,
        validation_only=args.validation_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
