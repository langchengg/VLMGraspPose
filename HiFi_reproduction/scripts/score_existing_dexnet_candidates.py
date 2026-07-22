#!/usr/bin/env python3
"""Score exact saved Dex-Net candidates with official GQ-CNN v1.3.0.

This file is intentionally Python 3.7-compatible and self-contained so it can
run only inside the isolated linux/amd64 TensorFlow 1.15 image.  It never calls
an image grasp sampler or a policy; the official GQCnnQualityFunction receives
the exact centers, angles, depths, and widths loaded from ``candidates.npz``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
GQCNN_COMMIT = "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21"
SUMMARY_FIELDS = (
    "sample_id",
    "query",
    "candidate_count",
    "gqcnn_top1_candidate_id",
    "gqcnn_top1_q_value",
    "geometric_top1_candidate_id",
    "top1_agreement",
    "spearman_rank_correlation",
    "top5_overlap_count",
    "top5_overlap_fraction",
    "official_quality_function_time_ms",
    "milliseconds_per_candidate",
    "source_candidates_npz_sha256",
    "source_npz_unchanged",
    "failure_reason",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_roots",
        nargs="*",
        type=Path,
        default=[
            REPO_ROOT / "outputs" / "dexnet_candidates_one_sample",
            REPO_ROOT / "outputs" / "dexnet_candidates_ten_samples",
        ],
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-name", default="GQCNN-2.1")
    parser.add_argument("--crop-height", type=int, default=96)
    parser.add_argument("--crop-width", type=int, default=96)
    parser.add_argument("--inpaint-rescale-factor", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verify-runtime-only", action="store_true")
    parser.add_argument("--model-load-only", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def strict_value(value):
    if isinstance(value, np.ndarray):
        return strict_value(value.tolist())
    if isinstance(value, np.generic):
        return strict_value(value.item())
    if isinstance(value, dict):
        return {str(key): strict_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(strict_value(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(path), mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo("%s.npy" % name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return path


def load_candidate_sidecar(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("candidate JSON must contain a candidates list")
    return payload["candidates"], payload.get("metadata", {})


def assert_same(name, actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("NPZ/JSON shape or dtype mismatch for %s" % name)
    if not np.array_equal(actual, expected, equal_nan=True):
        raise ValueError("NPZ/JSON numeric mismatch for %s" % name)


def load_frozen_candidates(sample_dir):
    npz_path = sample_dir / "candidates.npz"
    json_path = sample_dir / "candidates.json"
    records, metadata = load_candidate_sidecar(json_path)
    with np.load(str(npz_path), allow_pickle=False) as archive:
        required = (
            "center_uv",
            "center_depth_m",
            "center_camera_xyz_m",
            "angle_rad",
            "width_m",
            "width_px",
            "endpoints_uv",
            "valid",
            "T_camera_grasp_fixed_approach",
        )
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError("candidate NPZ is missing %s" % missing)
        arrays = {name: np.asarray(archive[name]) for name in required}
    count = arrays["center_uv"].shape[0]
    if len(records) != count:
        raise ValueError("NPZ/JSON candidate count mismatch")
    ids = [record.get("candidate_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != count:
        raise ValueError("saved candidate IDs are missing or non-unique")
    assert_same(
        "center_uv",
        arrays["center_uv"],
        np.asarray(
            [[record["center_u_px"], record["center_v_px"]] for record in records],
            dtype=arrays["center_uv"].dtype,
        ),
    )
    for array_name, field_name in (
        ("center_depth_m", "center_depth_m"),
        ("angle_rad", "angle_rad"),
        ("width_m", "width_m"),
        ("width_px", "width_px"),
    ):
        assert_same(
            array_name,
            arrays[array_name],
            np.asarray([record[field_name] for record in records], dtype=arrays[array_name].dtype),
        )
    assert_same(
        "center_camera_xyz_m",
        arrays["center_camera_xyz_m"],
        np.asarray([record["center_camera_xyz_m"] for record in records], dtype=arrays["center_camera_xyz_m"].dtype),
    )
    assert_same(
        "endpoints_uv",
        arrays["endpoints_uv"],
        np.asarray(
            [[record["endpoint_1_uv"], record["endpoint_2_uv"]] for record in records],
            dtype=arrays["endpoints_uv"].dtype,
        ),
    )
    if not np.all(arrays["valid"]):
        raise ValueError("only frozen valid post-NMS candidates may be scored")
    return records, metadata, arrays, {
        "candidates_npz_sha256": sha256_file(npz_path),
        "candidates_json_sha256": sha256_file(json_path),
    }


def load_intrinsics_values(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "frame": values.get("frame", values.get("_frame")),
        "fx": values.get("fx", values.get("_fx")),
        "fy": values.get("fy", values.get("_fy")),
        "cx": values.get("cx", values.get("_cx")),
        "cy": values.get("cy", values.get("_cy")),
        "skew": values.get("skew", values.get("_skew", 0.0)),
        "height": values.get("height", values.get("_height")),
        "width": values.get("width", values.get("_width")),
    }


def make_official_state_and_grasps(sample_dir, arrays, records, inpaint_rescale_factor):
    from autolab_core import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
    from gqcnn.grasping import Grasp2D, RgbdImageState

    intr = load_intrinsics_values(sample_dir / "camera.intr")
    camera = CameraIntrinsics(
        intr["frame"],
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        skew=float(intr["skew"]),
        height=int(intr["height"]),
        width=int(intr["width"]),
    )
    depth = np.load(str(sample_dir / "depth_m.npy"), allow_pickle=False).astype(np.float32)
    mask = np.asarray(Image.open(str(sample_dir / "hifics_mask_processed.png")).convert("L"), dtype=np.uint8)
    if depth.shape != mask.shape or depth.shape != (int(intr["height"]), int(intr["width"])):
        raise ValueError("depth, mask, and intrinsics dimensions disagree")
    depth_im = DepthImage(depth, frame=intr["frame"])
    color_im = ColorImage(np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8), frame=intr["frame"])
    segmask = BinaryImage(mask, frame=intr["frame"])
    segmask = segmask.mask_binary(depth_im.invalid_pixel_mask().inverse())
    inpainted_depth = depth_im.inpaint(rescale_factor=float(inpaint_rescale_factor))
    state = RgbdImageState(
        RgbdImage.from_color_and_depth(color_im, inpainted_depth),
        camera,
        segmask=segmask,
    )
    grasps = []
    for index, record in enumerate(records):
        contacts = record.get("contact_points_uv")
        normals = record.get("contact_normals")
        grasps.append(
            Grasp2D(
                arrays["center_uv"][index].astype(np.float64),
                angle=float(arrays["angle_rad"][index]),
                depth=float(arrays["center_depth_m"][index]),
                width=float(arrays["width_m"][index]),
                camera_intr=camera,
                contact_points=None if contacts is None else [np.asarray(item, dtype=np.float64) for item in contacts],
                contact_normals=None if normals is None else [np.asarray(item, dtype=np.float64) for item in normals],
            )
        )
    return state, grasps, intr


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or np.std(first) <= 0.0 or np.std(second) <= 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def compare_geometric(sample_dir, scored, top_k):
    path = sample_dir / "geometrically_ranked_candidates.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometric = payload["candidates"] if isinstance(payload, dict) else payload
    geometric_rank = {item["candidate_id"]: int(item["geometric_rank"]) for item in geometric}
    neural_rank = {item["candidate_id"]: int(item["gqcnn_rank"]) for item in scored}
    if set(geometric_rank) != set(neural_rank):
        raise ValueError("geometric and GQ-CNN candidate ID sets differ")
    ids = sorted(geometric_rank)
    rank_correlation = pearson(
        [geometric_rank[candidate_id] for candidate_id in ids],
        [neural_rank[candidate_id] for candidate_id in ids],
    )
    geometric_top = [item["candidate_id"] for item in geometric[:top_k]]
    neural_top = [item["candidate_id"] for item in scored[:top_k]]
    overlap = sorted(set(geometric_top) & set(neural_top))
    denominator = max(1, min(top_k, len(scored)))
    return {
        "candidate_count": len(scored),
        "rank_correlation_type": "Spearman (Pearson correlation of deterministic integer ranks)",
        "spearman_rank_correlation": rank_correlation,
        "geometric_top1_candidate_id": geometric[0]["candidate_id"],
        "gqcnn_top1_candidate_id": scored[0]["candidate_id"],
        "top1_agreement": geometric[0]["candidate_id"] == scored[0]["candidate_id"],
        "top_k": top_k,
        "geometric_topk_candidate_ids": geometric_top,
        "gqcnn_topk_candidate_ids": neural_top,
        "topk_overlap_candidate_ids": overlap,
        "topk_overlap_count": len(overlap),
        "topk_overlap_fraction": float(len(overlap) / denominator),
    }


def save_scored_csv(path, records):
    preferred = [
        "gqcnn_rank",
        "candidate_id",
        "sample_id",
        "query",
        "gqcnn_q_value",
        "center_u_px",
        "center_v_px",
        "center_depth_m",
        "angle_rad",
        "angle_deg",
        "width_m",
        "width_px",
    ]
    fields = preferred + sorted(set().union(*[set(record) for record in records]) - set(preferred))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {}
            for field in fields:
                value = strict_value(record.get(field))
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
                row[field] = "" if value is None else value
            writer.writerow(row)


def save_overlay(sample_dir, top1):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rgb = np.asarray(Image.open(str(sample_dir / "rgb.png")).convert("RGB"))
    mask = np.asarray(Image.open(str(sample_dir / "hifics_mask_processed.png")).convert("L")) > 0
    endpoints = np.asarray([top1["endpoint_1_uv"], top1["endpoint_2_uv"]], dtype=np.float64)
    center = np.asarray([top1["center_u_px"], top1["center_v_px"]], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    axis.imshow(rgb)
    if np.any(mask) and not np.all(mask):
        axis.contour(mask.astype(np.uint8), levels=[0.5], colors=["#00e676"], linewidths=1.2)
    axis.plot(endpoints[:, 0], endpoints[:, 1], color="#ff2d55", linewidth=3.0)
    axis.scatter(endpoints[:, 0], endpoints[:, 1], marker="|", s=100, linewidths=2.5, color="#ff2d55")
    axis.scatter(center[0], center[1], s=35, color="white", edgecolors="#ff2d55")
    axis.text(
        center[0] + 4,
        center[1] - 4,
        "%s q=%.4f" % (top1["candidate_id"], top1["gqcnn_q_value"]),
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
    )
    axis.set_title("Official GQ-CNN Top-1 (q is not calibrated for OCID-VLG)")
    axis.set_xlim(-0.5, rgb.shape[1] - 0.5)
    axis.set_ylim(rgb.shape[0] - 0.5, -0.5)
    axis.set_axis_off()
    figure.tight_layout(pad=0.1)
    figure.savefig(str(sample_dir / "gqcnn_top1_overlay.png"), dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def empty_row(sample_id):
    row = {field: "" for field in SUMMARY_FIELDS}
    row["sample_id"] = sample_id
    return row


def score_sample(sample_dir, quality_fn, args, runtime_metadata):
    source_npz = sample_dir / "candidates.npz"
    records, metadata, arrays, hashes = load_frozen_candidates(sample_dir)
    file_metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata and metadata.get("sample_id") != file_metadata.get("sample_id"):
        raise ValueError("candidate and sample metadata disagree")
    state, grasps, intr = make_official_state_and_grasps(
        sample_dir, arrays, records, args.inpaint_rescale_factor
    )
    started = time.perf_counter()
    q_values = np.asarray(
        quality_fn(
            state,
            grasps,
            params={"vis": {"tf_images": False, "k": min(25, len(grasps))}},
        ),
        dtype=np.float64,
    )
    quality_time_ms = (time.perf_counter() - started) * 1000.0
    if q_values.shape != (len(records),) or not np.all(np.isfinite(q_values)):
        raise RuntimeError("official GQ-CNN returned non-finite or wrong-shaped q-values")
    scored = []
    for index, source in enumerate(records):
        record = dict(source)
        record.update(
            {
                "center_u_px": float(arrays["center_uv"][index, 0]),
                "center_v_px": float(arrays["center_uv"][index, 1]),
                "center_depth_m": float(arrays["center_depth_m"][index]),
                "center_camera_xyz_m": arrays["center_camera_xyz_m"][index].astype(float).tolist(),
                "angle_rad": float(arrays["angle_rad"][index]),
                "angle_deg": float(np.degrees(arrays["angle_rad"][index])),
                "width_m": float(arrays["width_m"][index]),
                "width_px": float(arrays["width_px"][index]),
                "endpoint_1_uv": arrays["endpoints_uv"][index, 0].astype(float).tolist(),
                "endpoint_2_uv": arrays["endpoints_uv"][index, 1].astype(float).tolist(),
                "gqcnn_q_value": float(q_values[index]),
                "gqcnn_model_name": args.model_name,
                "gqcnn_q_is_calibrated_success_probability_for_ocid_vlg": False,
            }
        )
        scored.append(record)
    scored.sort(key=lambda item: (-round(item["gqcnn_q_value"], 12), item["candidate_id"]))
    for rank, record in enumerate(scored, start=1):
        record["gqcnn_rank"] = rank

    comparison = compare_geometric(sample_dir, scored, args.top_k)
    after_hash = sha256_file(source_npz)
    if after_hash != hashes["candidates_npz_sha256"]:
        raise RuntimeError("frozen candidates.npz changed during scoring")
    output_metadata = {
        "schema_version": 1,
        "sample_id": file_metadata["sample_id"],
        "query": file_metadata["query"],
        "model_name": args.model_name,
        "model_dir": str(args.model_dir),
        "official_gqcnn_release": "v1.3.0",
        "official_gqcnn_commit": GQCNN_COMMIT,
        "candidate_source": "exact saved post-NMS candidates.npz; no sampling or pose alteration",
        "preprocessing": "official GQCnnQualityFunction.grasps_to_tensors",
        "quality_interface": "official GraspQualityFunctionFactory quality_function('gqcnn', ...)",
        "q_value_warning": "Not a calibrated success probability for OCID-VLG camera/gripper domain",
        "runtime": runtime_metadata,
        "camera_intrinsics": intr,
        "source_hashes": dict(hashes, candidates_npz_sha256_after_scoring=after_hash, source_npz_unchanged=True),
        "timing_ms": {
            "official_quality_function_total": quality_time_ms,
            "per_candidate": quality_time_ms / len(scored),
        },
        "comparison_with_geometric_rank": comparison,
    }
    save_json(
        sample_dir / "gqcnn_scored_candidates.json",
        {"metadata": output_metadata, "candidates": scored},
    )
    save_scored_csv(sample_dir / "gqcnn_scored_candidates.csv", scored)
    rank_by_id = {record["candidate_id"]: record["gqcnn_rank"] for record in scored}
    q_by_id = {record["candidate_id"]: record["gqcnn_q_value"] for record in scored}
    npz_ids = [record["candidate_id"] for record in records]
    save_npz(
        sample_dir / "gqcnn_scored_candidates.npz",
        {
            "candidate_id": np.asarray(npz_ids, dtype="<U64"),
            "gqcnn_q_value": np.asarray([q_by_id[value] for value in npz_ids], dtype=np.float64),
            "gqcnn_rank": np.asarray([rank_by_id[value] for value in npz_ids], dtype=np.int32),
            "center_uv": arrays["center_uv"],
            "center_depth_m": arrays["center_depth_m"],
            "center_camera_xyz_m": arrays["center_camera_xyz_m"],
            "angle_rad": arrays["angle_rad"],
            "width_m": arrays["width_m"],
            "width_px": arrays["width_px"],
            "endpoints_uv": arrays["endpoints_uv"],
            "T_camera_grasp_fixed_approach": arrays["T_camera_grasp_fixed_approach"],
        },
    )
    top1 = scored[0]
    top1_payload = {
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "candidate_id": top1["candidate_id"],
        "sample_id": top1.get("sample_id"),
        "query": top1.get("query"),
        "center_pixel_uv": [top1["center_u_px"], top1["center_v_px"]],
        "center_depth_m": top1["center_depth_m"],
        "center_camera_xyz_m": top1["center_camera_xyz_m"],
        "angle_rad": top1["angle_rad"],
        "angle_deg": top1["angle_deg"],
        "width_m": top1["width_m"],
        "width_px": top1["width_px"],
        "gqcnn_q_value": top1["gqcnn_q_value"],
        "gqcnn_rank": 1,
        "gqcnn_model_name": args.model_name,
        "gqcnn_q_is_calibrated_success_probability_for_ocid_vlg": False,
        "camera_frame": intr["frame"],
        "calibration_warning": "Pretrained GQ-CNN camera and gripper domain differs from OCID-VLG.",
        "T_camera_grasp_fixed_approach": top1["T_camera_grasp_fixed_approach"],
    }
    save_json(sample_dir / "gqcnn_top1.json", top1_payload)
    if comparison is not None:
        save_json(sample_dir / "geometric_vs_gqcnn_comparison.json", comparison)
    if not args.skip_overlay:
        save_overlay(sample_dir, top1)

    row = empty_row(str(file_metadata["sample_id"]))
    row.update(
        {
            "query": file_metadata["query"],
            "candidate_count": len(scored),
            "gqcnn_top1_candidate_id": top1["candidate_id"],
            "gqcnn_top1_q_value": top1["gqcnn_q_value"],
            "official_quality_function_time_ms": quality_time_ms,
            "milliseconds_per_candidate": quality_time_ms / len(scored),
            "source_candidates_npz_sha256": hashes["candidates_npz_sha256"],
            "source_npz_unchanged": 1,
            "failure_reason": "",
        }
    )
    if comparison is not None:
        row.update(
            {
                "geometric_top1_candidate_id": comparison["geometric_top1_candidate_id"],
                "top1_agreement": int(comparison["top1_agreement"]),
                "spearman_rank_correlation": "" if comparison["spearman_rank_correlation"] is None else comparison["spearman_rank_correlation"],
                "top5_overlap_count": comparison["topk_overlap_count"],
                "top5_overlap_fraction": comparison["topk_overlap_fraction"],
            }
        )
    return row


def sample_dirs(root):
    if not root.is_dir():
        raise FileNotFoundError("output root does not exist: %s" % root)
    samples = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "candidates.npz").is_file()
    )
    if not samples:
        raise FileNotFoundError("no candidate samples below %s" % root)
    return samples


def write_summary(root, rows):
    with (root / "gqcnn_scoring_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_runtime():
    import tensorflow as tf
    import gqcnn
    from gqcnn.version import __version__ as gqcnn_version

    metadata = {
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tensorflow_gpu_available": bool(tf.test.is_gpu_available()),
        "gqcnn_version": gqcnn_version,
        "gqcnn_commit_expected": GQCNN_COMMIT,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if platform.machine() != "x86_64":
        raise RuntimeError("scoring must run in linux/amd64, got %s" % platform.machine())
    if tf.__version__ != "1.15.0":
        raise RuntimeError("TensorFlow 1.15.0 required, got %s" % tf.__version__)
    if tf.test.is_gpu_available():
        raise RuntimeError("CPU-only scoring requested but TensorFlow reports a GPU")
    if gqcnn_version != "1.3.0":
        raise RuntimeError("official GQ-CNN v1.3.0 required, got %s" % gqcnn_version)
    return metadata


def main():
    args = parse_args()
    runtime = verify_runtime()
    print(json.dumps({"status": "RUNTIME_OK", "runtime": runtime}, sort_keys=True))
    if args.verify_runtime_only:
        return 0
    if args.model_dir is None:
        raise ValueError("--model-dir is required unless --verify-runtime-only is used")
    args.model_dir = args.model_dir.expanduser().resolve()
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError("official model config missing: %s" % (args.model_dir / "config.json"))

    from gqcnn.grasping import GraspQualityFunctionFactory

    model_started = time.perf_counter()
    quality_fn = GraspQualityFunctionFactory.quality_function(
        "gqcnn",
        {
            "gqcnn_model": str(args.model_dir),
            "crop_height": int(args.crop_height),
            "crop_width": int(args.crop_width),
        },
    )
    runtime["model_load_time_ms"] = (time.perf_counter() - model_started) * 1000.0
    runtime["model_im_height"] = int(quality_fn.gqcnn.im_height)
    runtime["model_im_width"] = int(quality_fn.gqcnn.im_width)
    runtime["model_pose_dim"] = int(quality_fn.gqcnn.pose_dim)
    print(json.dumps({"status": "MODEL_LOAD_OK", "runtime": runtime}, sort_keys=True))
    if args.model_load_only:
        quality_fn.gqcnn.close_session()
        return 0

    failures = 0
    total = 0
    for root_arg in args.output_roots:
        root = root_arg.expanduser().resolve()
        rows = []
        for sample_dir in sample_dirs(root):
            total += 1
            try:
                row = score_sample(sample_dir, quality_fn, args, runtime)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            except Exception as error:
                failures += 1
                row = empty_row(sample_dir.name)
                row["failure_reason"] = "%s: %s" % (type(error).__name__, error)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            rows.append(row)
        write_summary(root, rows)
    quality_fn.gqcnn.close_session()
    print(json.dumps({"status": "DONE" if failures == 0 else "PARTIAL", "samples": total, "failures": failures}, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
