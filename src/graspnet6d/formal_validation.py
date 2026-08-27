"""Data-backed geometry and evaluator-parity evidence producers.

These runners create the strict evidence consumed by :mod:`graspnet6d.stages`.
Formal execution is the default.  Callback injection and reduced galleries are
accepted only when the caller explicitly marks the run as a test fixture.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import Candidate6D, candidate_pool_fingerprint
from .evaluator import (
    DEFAULT_GRASPNET_API_ROOT,
    FRICTION_DESCENDING,
    CandidateEvaluation,
    ensure_graspnetapi_source,
    evaluate_frozen_candidates,
    validate_graspnet_rows,
)
from .geometry import (
    CameraIntrinsics,
    as_rotation_matrix,
    as_transform,
    backproject_pixels,
    invert_transform,
    project_points,
    transform_points,
)
from .io import atomic_json, atomic_text, canonical_sha256, sha256_file
from .stages import (
    GEOMETRY_EVIDENCE_SCHEMA,
    GEOMETRY_SCHEMA,
    PARITY_EVIDENCE_SCHEMA,
    PARITY_SCHEMA,
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    GroupManifest,
    _official_group_evaluator_inputs,
    load_jsonl_records,
)


VISUAL_REVIEW_SCHEMA = "graspnet6d_ai_assisted_geometry_visual_review_v2"
GEOMETRY_RENDER_SCHEMA = "graspnet6d_geometry_render_manifest_v1"
RAW_VGN_VALIDATION_SCHEMA = "graspnet6d_raw_vgn_validation_candidates_v1"
FORMAL_GEOMETRY_MINIMUM_GROUPS = 20
AI_ASSISTED_REVIEW_KIND = "ai_assisted_geometry_review"
AI_ASSISTED_REVIEW_DISCLAIMER = (
    "AI-assisted geometry review; not an independent human review."
)
AI_ASSISTED_GROUP_CHECKS = (
    "approach_axis_consistent",
    "table_not_inverted",
    "candidate_overlay_consistent",
)
ORACLE_GROUNDING_CONDITION = "oracle_gt_mask"

_GEOMETRY_BINDINGS = {
    "data_manifest_sha256",
    "group_manifest_sha256",
    "tsdf_config_sha256",
    "extraction_config_sha256",
    "upstream_versions_sha256",
}


class FormalValidationError(RuntimeError):
    """A formal evidence gate could not be truthfully passed."""


@dataclass(frozen=True, slots=True)
class GeometryValidationArtifacts:
    metrics_csv_path: str
    evidence_path: str
    contract_path: str
    render_manifest_path: str
    audit_figure_paths: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["audit_figure_paths"] = list(self.audit_figure_paths)
        return result


@dataclass(frozen=True, slots=True)
class ParityValidationArtifacts:
    comparison_csv_path: str
    report_path: str
    evidence_path: str
    gate_path: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _scope(*, scope: str, fixture_only: bool) -> None:
    if fixture_only:
        if scope != "test_fixture":
            raise FormalValidationError(
                "fixture_only runs must use scope='test_fixture'"
            )
    elif scope != "formal_real_data":
        raise FormalValidationError(
            "non-fixture validation must use scope='formal_real_data'"
        )


def _regular_file(path: Path | str, description: str) -> Path:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FormalValidationError(f"missing regular {description}: {source}")
    return source


def _read_json(path: Path | str, description: str) -> dict[str, Any]:
    source = _regular_file(path, description)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalValidationError(f"invalid {description}: {source}: {error}") from error
    if not isinstance(value, dict):
        raise FormalValidationError(f"{description} must be a JSON object: {source}")
    return value


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _index_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl_records(path, description="geometry/parity target manifest")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        if not group_id or group_id in result:
            raise FormalValidationError("target manifest has blank or duplicate group_id")
        result[group_id] = dict(row)
    return result


def _selected_groups(
    manifest: Mapping[str, dict[str, Any]], group_ids: Sequence[str]
) -> tuple[str, ...]:
    selected = tuple(str(value).strip() for value in group_ids)
    if not selected or any(not value for value in selected) or len(set(selected)) != len(selected):
        raise FormalValidationError("selected_group_ids must be non-empty and unique")
    missing = sorted(set(selected) - set(manifest))
    if missing:
        raise FormalValidationError(f"selected groups are absent from the manifest: {missing}")
    return selected


def _reject_fixture_record(value: Any, description: str) -> None:
    if isinstance(value, Mapping):
        if value.get("fixture_only") is True:
            raise FormalValidationError(f"formal {description} is marked fixture_only")
        for child in value.values():
            _reject_fixture_record(child, description)
    elif isinstance(value, list):
        for child in value:
            _reject_fixture_record(child, description)
    elif isinstance(value, str) and "fixture" in value.casefold():
        raise FormalValidationError(f"formal {description} contains a fixture marker")


def _binding_records(
    evidence_path: Path,
    binding_paths: Mapping[str, Path | str],
    *,
    required: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    if not required.issubset(binding_paths):
        raise FormalValidationError(
            f"missing evidence binding paths: {sorted(required - set(binding_paths))}"
        )
    hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    for key in sorted(required):
        source = _regular_file(binding_paths[key], f"binding {key}")
        hashes[key] = sha256_file(source)
        paths[key] = _relative(source, evidence_path.parent)
    return hashes, paths


def _load_tsdf_cache(path: Path, group_id: str) -> dict[str, np.ndarray]:
    source = _regular_file(path, "TSDF cache")
    sidecar = _read_json(source.with_suffix(".json"), "TSDF sidecar")
    if sidecar.get("schema_version") != TSDF_SCHEMA or sidecar.get("group_id") != group_id:
        raise FormalValidationError(f"TSDF sidecar schema/group mismatch: {source}")
    if sidecar.get("output_sha256") != sha256_file(source):
        raise FormalValidationError(f"TSDF sidecar hash mismatch: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, KeyError, ValueError) as error:
        raise FormalValidationError(f"invalid TSDF cache {source}: {error}") from error
    required = {
        "tsdf",
        "T_local_to_camera",
        "T_local_to_table",
        "physical_size",
        "full_scene_depth_integrated",
    }
    if not required.issubset(values):
        raise FormalValidationError(f"TSDF cache lacks arrays: {sorted(required - set(values))}")
    if values["tsdf"].shape != (1, 40, 40, 40) or not np.isfinite(values["tsdf"]).all():
        raise FormalValidationError("TSDF cache violates the frozen 40^3 finite contract")
    if not bool(np.asarray(values["full_scene_depth_integrated"]).item()):
        raise FormalValidationError("TSDF cache was not integrated from full scene depth")
    for name in ("T_local_to_camera", "T_local_to_table"):
        values[name] = as_transform(values[name])
    physical_size = float(np.asarray(values["physical_size"]).item())
    if not np.isfinite(physical_size) or physical_size <= 0:
        raise FormalValidationError("TSDF physical_size must be finite and positive")
    values["physical_size"] = np.asarray(physical_size)
    return values


def _load_raw_candidates(
    path: Path, group_id: str, tsdf_path: Path, *, fixture_only: bool
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    source = _regular_file(path, "raw VGN candidate bundle")
    payload = _read_json(source, "raw VGN candidate bundle")
    if payload.get("schema_version") not in {
        RAW_VGN_VALIDATION_SCHEMA,
        VGN_BUNDLE_SCHEMA,
    }:
        raise FormalValidationError(f"unsupported raw VGN candidate schema: {source}")
    if payload.get("group_id") != group_id:
        raise FormalValidationError(f"raw VGN candidates belong to another group: {source}")
    raw = payload.get("raw_vgn_candidates")
    if not isinstance(raw, list) or not raw:
        raise FormalValidationError(f"raw VGN candidate bundle is empty: {source}")
    if int(payload.get("candidate_count", -1)) != len(raw):
        raise FormalValidationError(f"raw VGN candidate count mismatch: {source}")
    if int(payload.get("inference_calls_for_group", -1)) != 1:
        raise FormalValidationError("raw candidate evidence must record exactly one VGN inference")
    check = dict(payload)
    observed_fingerprint = check.pop("bundle_fingerprint", None)
    if observed_fingerprint != canonical_sha256(check):
        raise FormalValidationError(f"raw VGN bundle fingerprint mismatch: {source}")
    recorded_tsdf = payload.get("tsdf_sha256")
    if recorded_tsdf is not None and recorded_tsdf != sha256_file(tsdf_path):
        raise FormalValidationError("raw VGN candidates were generated from another TSDF cache")
    if not fixture_only:
        required_formal = {
            "tsdf_path",
            "tsdf_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "device",
            "extraction_config",
            "raw_vgn_pool_fingerprint",
        }
        if not required_formal.issubset(payload):
            raise FormalValidationError(
                f"formal raw VGN bundle lacks provenance: {sorted(required_formal - set(payload))}"
            )
        bound_tsdf = _regular_file(payload["tsdf_path"], "raw-candidate source TSDF")
        if bound_tsdf != tsdf_path or payload["tsdf_sha256"] != sha256_file(bound_tsdf):
            raise FormalValidationError("formal raw VGN bundle TSDF path/hash is stale")
        checkpoint = _regular_file(payload["checkpoint_path"], "raw-candidate VGN checkpoint")
        if payload["checkpoint_sha256"] != sha256_file(checkpoint):
            raise FormalValidationError("formal raw VGN checkpoint hash is stale")
        if payload["raw_vgn_pool_fingerprint"] != canonical_sha256(raw):
            raise FormalValidationError("formal raw VGN pool fingerprint mismatch")
    candidate_ids = [str(item.get("candidate_id", "")) for item in raw if isinstance(item, dict)]
    if len(candidate_ids) != len(raw) or any(not item for item in candidate_ids):
        raise FormalValidationError("raw VGN candidate records lack candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise FormalValidationError("raw VGN candidate IDs are duplicated")
    return payload, tuple(dict(item) for item in raw)


def _intrinsics(record: Mapping[str, Any], image_shape: tuple[int, int]) -> CameraIntrinsics:
    path = _regular_file(record["intrinsics_path"], "camera intrinsics")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.shape != (3, 3):
        raise FormalValidationError(f"camera intrinsics must be 3x3: {path}")
    return CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=int(image_shape[1]),
        height=int(image_shape[0]),
    )


def _image_inputs(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    from PIL import Image
    from scipy.io import loadmat

    rgb_path = _regular_file(record["rgb_path"], "RGB image")
    depth_path = _regular_file(record["depth_path"], "depth image")
    label_path = _regular_file(record["instance_label_path"], "instance-label image")
    meta_path = _regular_file(record["meta_path"], "frame metadata")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(depth_path))
    label = np.asarray(Image.open(label_path))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.ndim != 2 or label.ndim != 2:
        raise FormalValidationError("RGB/depth/label images have invalid dimensions")
    if rgb.shape[:2] != depth.shape or label.shape != depth.shape:
        raise FormalValidationError("RGB/depth/label image shapes differ")
    meta = loadmat(meta_path)
    raw_scale = np.asarray(meta.get("factor_depth", []), dtype=np.float64).reshape(-1)
    if raw_scale.size != 1 or not np.isfinite(raw_scale[0]) or raw_scale[0] <= 0:
        raise FormalValidationError("frame metadata lacks one positive factor_depth")
    return rgb, depth, label, float(raw_scale[0])


def _depth_reprojection_metrics(
    depth: np.ndarray,
    target_mask: np.ndarray,
    depth_scale: float,
    intrinsics: CameraIntrinsics,
) -> dict[str, float | bool | int]:
    """Recompute a deterministic target-depth 2D -> 3D -> 2D round trip."""

    valid = target_mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    rows, columns = np.nonzero(valid)
    if len(rows) == 0:
        raise FormalValidationError("target mask contains no valid positive depth")
    # Bound audit cost without changing its deterministic spatial coverage.
    if len(rows) > 4096:
        selected = np.linspace(0, len(rows) - 1, num=4096, dtype=int)
        rows = rows[selected]
        columns = columns[selected]
    depth_m = np.asarray(depth[rows, columns], dtype=np.float64) / float(depth_scale)
    if not np.isfinite(depth_m).all() or np.any(depth_m <= 0.0):
        raise FormalValidationError("target depth does not convert to positive finite metres")
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    reconstructed = project_points(
        backproject_pixels(pixels, depth_m, intrinsics), intrinsics
    )
    errors = np.linalg.norm(reconstructed - pixels, axis=1)
    depth_unit_valid = bool(float(depth_m.min()) > 0.0 and float(depth_m.max()) <= 10.0)
    return {
        "depth_reprojection_sample_count": int(len(errors)),
        "depth_reprojection_median_error_px": float(np.median(errors)),
        "depth_reprojection_p95_error_px": float(np.quantile(errors, 0.95)),
        "depth_m_min": float(depth_m.min()),
        "depth_m_max": float(depth_m.max()),
        "depth_unit_valid": depth_unit_valid,
    }


def _candidate_metrics(
    *,
    group_id: str,
    candidates: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    intrinsics: CameraIntrinsics,
    target_mask: np.ndarray,
    conversion: np.ndarray,
    depth_metrics: Mapping[str, float | bool | int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local_to_camera = as_transform(cache["T_local_to_camera"])
    local_to_table = as_transform(cache["T_local_to_table"])
    camera_to_table = as_transform(local_to_table @ invert_transform(local_to_camera))
    table_to_camera = invert_transform(camera_to_table)
    table_to_local = invert_transform(local_to_table)
    size = float(np.asarray(cache["physical_size"]).item())
    # Target workspaces are defined to be table-axis aligned.  A flipped or
    # permuted table basis therefore fails deterministically, even if it is a
    # mathematically valid right-handed rotation.
    table_frame_not_inverted = bool(
        np.allclose(local_to_table[:3, :3], np.eye(3), atol=1e-6, rtol=0.0)
    )
    mask_rows, mask_columns = np.nonzero(target_mask)
    if len(mask_rows) == 0:
        raise FormalValidationError(f"target mask is empty for {group_id}")
    margin = 25.0
    x_min, x_max = float(mask_columns.min()) - margin, float(mask_columns.max()) + margin
    y_min, y_max = float(mask_rows.min()) - margin, float(mask_rows.max()) + margin
    metrics: list[dict[str, Any]] = []
    render: list[dict[str, Any]] = []
    for raw in candidates:
        candidate_id = str(raw["candidate_id"])
        projection_gate_marker = raw.get(
            "official_evaluator_valid_visible_target", False
        )
        if not isinstance(projection_gate_marker, bool):
            raise FormalValidationError(
                f"candidate {candidate_id} has a malformed "
                "official_evaluator_valid_visible_target marker"
            )
        projection_gate_required = projection_gate_marker
        local_t = np.asarray(raw.get("translation_local_m"), dtype=np.float64)
        camera_t = np.asarray(raw.get("translation_camera_m"), dtype=np.float64)
        table_t = np.asarray(raw.get("translation_table_m"), dtype=np.float64)
        local_r = as_rotation_matrix(raw.get("rotation_local_vgn"))
        camera_r = as_rotation_matrix(raw.get("rotation_camera_vgn"))
        table_r = as_rotation_matrix(raw.get("rotation_table_vgn"))
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in (local_t, camera_t, table_t)):
            raise FormalValidationError(f"candidate {candidate_id} has invalid translations")
        expected_camera_t = transform_points(local_to_camera, local_t)
        expected_table_t = transform_points(local_to_table, local_t)
        camera_round_trip = transform_points(table_to_camera, transform_points(camera_to_table, camera_t))
        local_round_trip = transform_points(table_to_local, transform_points(local_to_table, local_t))
        camera_table_error = max(
            float(np.linalg.norm(camera_round_trip - camera_t)),
            float(np.linalg.norm(expected_camera_t - camera_t)),
            float(np.linalg.norm(transform_points(camera_to_table, camera_t) - table_t)),
        )
        local_table_error = max(
            float(np.linalg.norm(local_round_trip - local_t)),
            float(np.linalg.norm(expected_table_t - table_t)),
        )
        converted_camera_r = camera_r @ conversion
        converted_local_r = local_r @ conversion
        converted_table_r = table_r @ conversion
        orthogonality = max(
            float(np.max(np.abs(value.T @ value - np.eye(3))))
            for value in (converted_camera_r, converted_local_r, converted_table_r)
        )
        rotation_consistency = max(
            float(
                np.max(
                    np.abs(local_to_camera[:3, :3] @ converted_local_r - converted_camera_r)
                )
            ),
            float(
                np.max(
                    np.abs(local_to_table[:3, :3] @ converted_local_r - converted_table_r)
                )
            ),
            float(
                np.max(
                    np.abs(camera_to_table[:3, :3] @ converted_camera_r - converted_table_r)
                )
            ),
        )
        determinant = float(np.linalg.det(converted_camera_r))
        width_m = float(raw.get("width_m", np.nan))
        width_m_valid = bool(np.isfinite(width_m) and 0.0 < width_m <= size)
        try:
            pixel = np.asarray(project_points(camera_t, intrinsics), dtype=np.float64)
            reconstructed = backproject_pixels(pixel, camera_t[2], intrinsics)
            reprojection = np.asarray(project_points(reconstructed, intrinsics), dtype=np.float64)
            reprojection_error = float(np.linalg.norm(reprojection - pixel))
            in_image = bool(
                0 <= pixel[0] < int(intrinsics.width or 0)
                and 0 <= pixel[1] < int(intrinsics.height or 0)
            )
            near_target = bool(x_min <= pixel[0] <= x_max and y_min <= pixel[1] <= y_max)
        except ValueError:
            pixel = np.array([np.nan, np.nan], dtype=np.float64)
            reprojection_error = float("inf")
            in_image = False
            near_target = False
        inside = bool(np.all(local_t >= -1e-9) and np.all(local_t <= size + 1e-9))
        metrics.append(
            {
                "group_id": group_id,
                "candidate_id": candidate_id,
                "reprojection_error_px": reprojection_error,
                "camera_table_round_trip_error_m": camera_table_error,
                "local_table_round_trip_error_m": local_table_error,
                "rotation_orthogonality_error": orthogonality,
                "rotation_determinant": determinant,
                "rotation_frame_consistency_error": rotation_consistency,
                "candidate_center_inside_workspace": inside,
                # Association is unavailable when this pre-evaluator geometry
                # probe runs.  Full-pool projection remains diagnostic, while
                # image bounds become a formal gate only when an explicit
                # official-evaluator/visibility marker is present.  Target-bbox
                # proximity is never a gate: distractors legitimately belong
                # to the identical frozen pool.
                "projected_center_in_image": in_image,
                "projected_center_near_target_bbox": near_target,
                "projection_image_bounds_gate_required": projection_gate_required,
                "projected_center_reasonable": (
                    not projection_gate_required or in_image
                ),
                "width_m": width_m,
                "width_m_valid": width_m_valid,
                "table_frame_not_inverted": table_frame_not_inverted,
                **depth_metrics,
                "approach_visual_audit_passed": False,
            }
        )
        render.append(
            {
                "candidate_id": candidate_id,
                "native_rank": int(raw.get("native_rank", 0)),
                "camera_t": camera_t,
                "camera_r": converted_camera_r,
                "pixel": pixel,
            }
        )
    return metrics, render


def _render_figure(
    path: Path,
    *,
    group_id: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    target_mask: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    intrinsics: CameraIntrinsics,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Official RGB")
    depth_view = axes[0, 1].imshow(depth, cmap="viridis")
    axes[0, 1].set_title("Official depth")
    figure.colorbar(depth_view, ax=axes[0, 1], fraction=0.046)
    axes[1, 0].imshow(rgb)
    overlay = np.zeros((*target_mask.shape, 4), dtype=np.float32)
    overlay[target_mask] = (1.0, 0.0, 0.0, 0.45)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Target instance mask")
    axes[1, 1].imshow(rgb)
    height_px, width_px = rgb.shape[:2]
    in_frame_count = 0
    ordered_candidates = sorted(candidates, key=lambda item: item["native_rank"])
    for index, candidate in enumerate(ordered_candidates):
        pixel = np.asarray(candidate["pixel"], dtype=np.float64)
        if not np.isfinite(pixel).all():
            continue
        in_frame = bool(
            0 <= pixel[0] < width_px and 0 <= pixel[1] < height_px
        )
        in_frame_count += int(in_frame)
        display_pixel = np.array(
            [
                np.clip(pixel[0], 0, width_px - 1),
                np.clip(pixel[1], 0, height_px - 1),
            ],
            dtype=np.float64,
        )
        center = np.asarray(candidate["camera_t"], dtype=np.float64)
        approach = np.asarray(candidate["camera_r"], dtype=np.float64)[:, 0]
        endpoint = center + 0.05 * approach
        try:
            endpoint_pixel = np.asarray(project_points(endpoint, intrinsics), dtype=np.float64)
        except ValueError:
            endpoint_pixel = pixel
        colour = plt.cm.tab10(index % 10)
        axes[1, 1].scatter(
            display_pixel[0],
            display_pixel[1],
            s=28,
            color=colour,
            marker="o" if in_frame else "x",
        )
        if in_frame and np.isfinite(endpoint_pixel).all():
            axes[1, 1].annotate(
                "",
                xy=(endpoint_pixel[0], endpoint_pixel[1]),
                xytext=(pixel[0], pixel[1]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "linewidth": 1.5,
                },
            )
    axes[1, 1].set_xlim(-0.5, width_px - 0.5)
    axes[1, 1].set_ylim(height_px - 0.5, -0.5)
    axes[1, 1].set_title(
        "Full frozen pool: converted +X axes "
        f"({in_frame_count}/{len(ordered_candidates)} centres in frame; "
        "edge x = off-frame)"
    )
    for axis in axes.reshape(-1):
        axis.axis("off")
    figure.suptitle(f"Real geometry audit: {group_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=140)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _render_or_reuse(
    *,
    output: Path,
    render_input_fingerprint: str,
    render_inputs: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[Mapping[str, Any]], CameraIntrinsics]],
) -> tuple[Path, tuple[Path, ...]]:
    manifest_path = output / "geometry_render_manifest.json"
    figure_paths = tuple(output / "figures" / f"{index:03d}_{canonical_sha256(group_id)[:16]}.png" for index, group_id in enumerate(render_inputs))
    if manifest_path.is_file() and not manifest_path.is_symlink():
        payload = _read_json(manifest_path, "geometry render manifest")
        if payload.get("schema_version") == GEOMETRY_RENDER_SCHEMA and payload.get(
            "input_fingerprint"
        ) == render_input_fingerprint:
            expected = payload.get("figure_sha256")
            if isinstance(expected, Mapping) and all(
                path.is_file()
                and not path.is_symlink()
                and expected.get(_relative(path, output)) == sha256_file(path)
                for path in figure_paths
            ):
                return manifest_path, figure_paths
    for path, (group_id, values) in zip(figure_paths, render_inputs.items(), strict=True):
        rgb, depth, mask, candidates, intrinsics = values
        _render_figure(
            path,
            group_id=group_id,
            rgb=rgb,
            depth=depth,
            target_mask=mask,
            candidates=candidates,
            intrinsics=intrinsics,
        )
    atomic_json(
        manifest_path,
        {
            "schema_version": GEOMETRY_RENDER_SCHEMA,
            "input_fingerprint": render_input_fingerprint,
            "group_ids": list(render_inputs),
            "figure_sha256": {
                _relative(path, output): sha256_file(path) for path in figure_paths
            },
        },
    )
    return manifest_path, figure_paths


def _visual_review(
    path: Path | str | None,
    *,
    output: Path,
    group_ids: Sequence[str],
    figure_paths: Sequence[Path],
    fixture_only: bool,
) -> tuple[dict[str, bool], Path, dict[str, Any]]:
    if path is None:
        raise FormalValidationError(
            "audit figures were rendered but no AI-assisted visual review was supplied; "
            f"inspect the immutable figures with visual capability, create a "
            f"{VISUAL_REVIEW_SCHEMA} review JSON, and rerun"
        )
    source = _regular_file(path, "geometry visual review")
    payload = _read_json(source, "geometry visual review")
    if payload.get("schema_version") != VISUAL_REVIEW_SCHEMA:
        raise FormalValidationError("geometry visual review schema mismatch")
    expected_scope = "test_fixture" if fixture_only else "formal_real_data"
    if payload.get("scope") != expected_scope or payload.get("fixture_only") is not fixture_only:
        raise FormalValidationError("geometry visual review scope/fixture marker mismatch")
    if str(payload.get("status", "")).upper() != "PASSED":
        raise FormalValidationError("geometry visual review did not pass")
    if payload.get("review_kind") != AI_ASSISTED_REVIEW_KIND:
        raise FormalValidationError("geometry review is not explicitly AI-assisted")
    if not str(payload.get("reviewer_system", "")).strip():
        raise FormalValidationError("AI-assisted geometry review lacks reviewer_system")
    if payload.get("independent_human_review") is not False:
        raise FormalValidationError(
            "AI-assisted geometry review must explicitly set independent_human_review:false"
        )
    if payload.get("disclaimer") != AI_ASSISTED_REVIEW_DISCLAIMER:
        raise FormalValidationError("AI-assisted geometry review disclaimer mismatch")
    reviewed = payload.get("reviewed_group_ids")
    if not isinstance(reviewed, list) or set(map(str, reviewed)) != set(group_ids):
        raise FormalValidationError("visual review group IDs differ from rendered groups")
    decisions = payload.get("checks_by_group")
    if not isinstance(decisions, Mapping) or set(map(str, decisions)) != set(group_ids):
        raise FormalValidationError("visual review lacks one check record per group")
    checked_decisions: dict[str, bool] = {}
    for group_id in group_ids:
        group_checks = decisions[group_id]
        if not isinstance(group_checks, Mapping) or set(group_checks) != set(
            AI_ASSISTED_GROUP_CHECKS
        ):
            raise FormalValidationError(
                f"AI-assisted review checks are incomplete for group {group_id!r}"
            )
        checked_decisions[group_id] = all(
            group_checks[name] is True for name in AI_ASSISTED_GROUP_CHECKS
        )
    if not all(checked_decisions.values()):
        raise FormalValidationError("one or more AI-assisted visual geometry checks failed")
    claimed_hashes = payload.get("figure_sha256")
    if not isinstance(claimed_hashes, Mapping):
        raise FormalValidationError("visual review lacks figure hashes")
    observed = {_relative(item, output): sha256_file(item) for item in figure_paths}
    if dict(claimed_hashes) != observed:
        raise FormalValidationError("visual review figure hashes differ from rendered figures")
    return checked_decisions, source, payload


def run_geometry_validation(
    target_manifest_path: Path | str,
    selected_group_ids: Sequence[str],
    tsdf_cache_paths: Mapping[str, Path | str],
    raw_candidate_paths: Mapping[str, Path | str],
    output_root: Path | str,
    *,
    R_vgn_gripper_to_graspnet_gripper: np.ndarray,
    height_m: float,
    depth_m: float,
    binding_paths: Mapping[str, Path | str],
    visual_review_path: Path | str | None = None,
    scope: str = "formal_real_data",
    fixture_only: bool = False,
    minimum_figures: int = FORMAL_GEOMETRY_MINIMUM_GROUPS,
    renderer: Callable[..., None] | None = None,
) -> GeometryValidationArtifacts:
    """Render, AI-assist review, and publish strict geometry evidence.

    The first call may omit ``visual_review_path``.  It renders and hashes the
    gallery, then fails closed so a visual-capable AI can inspect the exact
    immutable PNGs without making an independent-human-review claim.
    A second call supplies the separately-authored review record.
    """

    _scope(scope=scope, fixture_only=fixture_only)
    if renderer is not None and not fixture_only:
        raise FormalValidationError("renderer injection is permitted only for test fixtures")
    if minimum_figures < FORMAL_GEOMETRY_MINIMUM_GROUPS and not fixture_only:
        raise FormalValidationError(
            "formal geometry validation requires at least "
            f"{FORMAL_GEOMETRY_MINIMUM_GROUPS} real groups/figures"
        )
    if minimum_figures <= 0:
        raise ValueError("minimum_figures must be positive")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = _regular_file(target_manifest_path, "target manifest")
    manifest = _index_manifest(manifest_path)
    selected = _selected_groups(manifest, selected_group_ids)
    if len(selected) < minimum_figures:
        raise FormalValidationError(
            f"geometry validation requires {minimum_figures} groups/figures, got {len(selected)}"
        )
    if set(selected) - set(tsdf_cache_paths) or set(selected) - set(raw_candidate_paths):
        raise FormalValidationError("every selected group requires explicit TSDF and candidate paths")
    if Path(binding_paths.get("group_manifest_sha256", "")).expanduser().resolve() != manifest_path:
        raise FormalValidationError("group_manifest binding must be the consumed target manifest")
    conversion = as_rotation_matrix(R_vgn_gripper_to_graspnet_gripper)
    if not np.isfinite(height_m) or not np.isfinite(depth_m) or height_m <= 0 or depth_m <= 0:
        raise FormalValidationError("proposed GraspNet height/depth must be finite and positive")
    rows: list[dict[str, Any]] = []
    render_inputs: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[Mapping[str, Any]], CameraIntrinsics],
    ] = {}
    input_hashes: dict[str, Any] = {}
    for group_id in selected:
        record = manifest[group_id]
        if not fixture_only:
            _reject_fixture_record(record, "target manifest record")
        tsdf_path = _regular_file(tsdf_cache_paths[group_id], "TSDF cache")
        candidate_path = _regular_file(raw_candidate_paths[group_id], "raw candidates")
        cache = _load_tsdf_cache(tsdf_path, group_id)
        candidate_payload, candidates = _load_raw_candidates(
            candidate_path, group_id, tsdf_path, fixture_only=fixture_only
        )
        if not fixture_only:
            _reject_fixture_record(candidate_payload, "raw candidate bundle")
        rgb, depth, label, depth_scale = _image_inputs(record)
        target_mask = label == int(record["target_instance_label"])
        intrinsics = _intrinsics(record, depth.shape)
        depth_metrics = _depth_reprojection_metrics(
            depth, target_mask, depth_scale, intrinsics
        )
        candidate_rows, render_candidates = _candidate_metrics(
            group_id=group_id,
            candidates=candidates,
            cache=cache,
            intrinsics=intrinsics,
            target_mask=target_mask,
            conversion=conversion,
            depth_metrics=depth_metrics,
        )
        rows.extend(candidate_rows)
        render_inputs[group_id] = (
            rgb,
            depth,
            target_mask,
            render_candidates,
            intrinsics,
        )
        input_hashes[group_id] = {
            "tsdf_path": str(tsdf_path),
            "tsdf_sha256": sha256_file(tsdf_path),
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "rgb_sha256": sha256_file(_regular_file(record["rgb_path"], "RGB image")),
            "depth_sha256": sha256_file(_regular_file(record["depth_path"], "depth image")),
            "label_sha256": sha256_file(
                _regular_file(record["instance_label_path"], "instance-label image")
            ),
        }
    if not rows:
        raise FormalValidationError("selected geometry groups produced no candidates")
    render_fingerprint = canonical_sha256(
        {
            "manifest_sha256": sha256_file(manifest_path),
            "selected_group_ids": selected,
            "input_hashes": input_hashes,
            "conversion": conversion,
            "height_m": float(height_m),
            "depth_m": float(depth_m),
        }
    )
    if renderer is None:
        render_manifest_path, figures = _render_or_reuse(
            output=output,
            render_input_fingerprint=render_fingerprint,
            render_inputs=render_inputs,
        )
    else:
        figures_list: list[Path] = []
        for index, (group_id, values) in enumerate(render_inputs.items()):
            path = output / "figures" / f"{index:03d}_{canonical_sha256(group_id)[:16]}.png"
            renderer(
                path,
                group_id=group_id,
                rgb=values[0],
                depth=values[1],
                target_mask=values[2],
                candidates=values[3],
                intrinsics=values[4],
            )
            _regular_file(path, "injected-renderer audit figure")
            figures_list.append(path)
        figures = tuple(figures_list)
        render_manifest_path = output / "geometry_render_manifest.json"
        atomic_json(
            render_manifest_path,
            {
                "schema_version": GEOMETRY_RENDER_SCHEMA,
                "input_fingerprint": render_fingerprint,
                "group_ids": list(render_inputs),
                "figure_sha256": {
                    _relative(path, output): sha256_file(path) for path in figures
                },
                "fixture_only": True,
            },
        )
    decisions, review_path, review_payload = _visual_review(
        visual_review_path,
        output=output,
        group_ids=selected,
        figure_paths=figures,
        fixture_only=fixture_only,
    )
    for row in rows:
        row["approach_visual_audit_passed"] = decisions[str(row["group_id"])]
    frame = pd.DataFrame(rows)
    metrics_path = output / "geometry_sample_metrics.csv"
    _atomic_csv(metrics_path, frame)
    reprojection = frame["reprojection_error_px"].to_numpy(float)
    depth_reprojection_median = frame[
        "depth_reprojection_median_error_px"
    ].to_numpy(float)
    depth_reprojection_p95 = frame[
        "depth_reprojection_p95_error_px"
    ].to_numpy(float)
    numeric_finite = np.isfinite(
        frame[
            [
                "reprojection_error_px",
                "camera_table_round_trip_error_m",
                "local_table_round_trip_error_m",
                "rotation_orthogonality_error",
                "rotation_determinant",
                "rotation_frame_consistency_error",
            ]
        ].to_numpy(float)
    ).all()
    passed = bool(
        numeric_finite
        and float(np.median(reprojection)) < 1.0
        and float(np.quantile(reprojection, 0.95)) < 2.0
        and float(np.max(depth_reprojection_median)) < 1.0
        and float(np.max(depth_reprojection_p95)) < 2.0
        and float(frame["camera_table_round_trip_error_m"].max()) <= 1e-6
        and float(frame["local_table_round_trip_error_m"].max()) <= 1e-6
        and float(frame["rotation_orthogonality_error"].max()) <= 1e-5
        and float(frame["rotation_frame_consistency_error"].max()) <= 1e-5
        and float(np.max(np.abs(frame["rotation_determinant"].to_numpy(float) - 1.0))) <= 1e-5
        and frame["candidate_center_inside_workspace"].astype(bool).all()
        and frame["projected_center_reasonable"].astype(bool).all()
        and frame["depth_unit_valid"].astype(bool).all()
        and frame["width_m_valid"].astype(bool).all()
        and frame["table_frame_not_inverted"].astype(bool).all()
        and frame["approach_visual_audit_passed"].astype(bool).all()
    )
    evidence_path = output / "geometry_validation_evidence.json"
    bindings, checked_binding_paths = _binding_records(
        evidence_path, binding_paths, required=_GEOMETRY_BINDINGS
    )
    figure_values = [_relative(path, evidence_path.parent) for path in figures]
    evidence = {
        "schema_version": GEOMETRY_EVIDENCE_SCHEMA,
        "scope": scope,
        "fixture_only": fixture_only,
        "status": "PASSED" if passed else "FAILED",
        "validated_group_ids": list(selected),
        "bindings": bindings,
        "binding_paths": checked_binding_paths,
        "R_vgn_gripper_to_graspnet_gripper": conversion.tolist(),
        "height_m": float(height_m),
        "depth_m": float(depth_m),
        "sample_metrics_path": _relative(metrics_path, evidence_path.parent),
        "sample_metrics_sha256": sha256_file(metrics_path),
        "audit_figure_paths": figure_values,
        "audit_figure_sha256": {
            raw: sha256_file(path) for raw, path in zip(figure_values, figures, strict=True)
        },
        "render_manifest_path": _relative(render_manifest_path, evidence_path.parent),
        "render_manifest_sha256": sha256_file(render_manifest_path),
        "visual_review_path": _relative(review_path, evidence_path.parent),
        "visual_review_sha256": sha256_file(review_path),
        "visual_review_kind": AI_ASSISTED_REVIEW_KIND,
        "visual_review_reviewer_system": str(review_payload["reviewer_system"]),
        "independent_human_review": False,
        "visual_review_disclaimer": AI_ASSISTED_REVIEW_DISCLAIMER,
        "sample_source_hashes": input_hashes,
        "recomputed_summary": {
            "candidate_count": len(frame),
            "reprojection_median_error_px": float(np.median(reprojection)),
            "reprojection_p95_error_px": float(np.quantile(reprojection, 0.95)),
            "depth_reprojection_median_error_px": float(
                np.max(depth_reprojection_median)
            ),
            "depth_reprojection_p95_error_px": float(np.max(depth_reprojection_p95)),
            "depth_m_min": float(frame["depth_m_min"].min()),
            "depth_m_max": float(frame["depth_m_max"].max()),
            "all_widths_valid_metres": bool(frame["width_m_valid"].astype(bool).all()),
            "full_pool_candidate_count": len(frame),
            "projected_center_in_image_count": int(
                frame["projected_center_in_image"].astype(bool).sum()
            ),
            "projected_center_near_target_bbox_diagnostic_count": int(
                frame["projected_center_near_target_bbox"].astype(bool).sum()
            ),
            "projection_image_bounds_gate_required_count": int(
                frame["projection_image_bounds_gate_required"].astype(bool).sum()
            ),
            "table_frame_not_inverted": bool(
                frame["table_frame_not_inverted"].astype(bool).all()
            ),
            "camera_table_round_trip_max_error_m": float(
                frame["camera_table_round_trip_error_m"].max()
            ),
            "local_table_round_trip_max_error_m": float(
                frame["local_table_round_trip_error_m"].max()
            ),
        },
    }
    atomic_json(evidence_path, evidence)
    if not passed:
        raise FormalValidationError(
            f"geometry validation thresholds failed; inspect {metrics_path} and {evidence_path}"
        )
    contract_path = output / "evaluator_geometry_contract.json"
    atomic_json(
        contract_path,
        {
            "schema_version": GEOMETRY_SCHEMA,
            "validated": True,
            "validation_artifact": _relative(evidence_path, contract_path.parent),
            "R_vgn_gripper_to_graspnet_gripper": conversion.tolist(),
            "height_m": float(height_m),
            "depth_m": float(depth_m),
        },
    )
    return GeometryValidationArtifacts(
        metrics_csv_path=str(metrics_path),
        evidence_path=str(evidence_path),
        contract_path=str(contract_path),
        render_manifest_path=str(render_manifest_path),
        audit_figure_paths=tuple(map(str, figures)),
    )


def _load_frozen_bundle(
    path: Path, group_id: str, *, fixture_only: bool
) -> tuple[list[str], np.ndarray, str]:
    source = _regular_file(path, "frozen candidate bundle")
    payload = _read_json(source, "frozen candidate bundle")
    if payload.get("schema_version") != VGN_BUNDLE_SCHEMA or payload.get("group_id") != group_id:
        raise FormalValidationError(f"frozen candidate schema/group mismatch: {source}")
    if payload.get("grounding_condition") != ORACLE_GROUNDING_CONDITION:
        raise FormalValidationError(
            "evaluator parity requires frozen oracle_gt_mask candidate bundles"
        )
    if not fixture_only:
        _reject_fixture_record(payload, "frozen candidate bundle")
    check = dict(payload)
    observed = check.pop("bundle_fingerprint", None)
    if observed != canonical_sha256(check):
        raise FormalValidationError(f"frozen candidate bundle fingerprint mismatch: {source}")
    records = payload.get("candidate_records")
    if not isinstance(records, list) or not records:
        raise FormalValidationError(f"frozen candidate pool is empty: {source}")
    if int(payload.get("candidate_count", -1)) != len(records):
        raise FormalValidationError(f"frozen candidate count mismatch: {source}")
    raw_records = payload.get("raw_vgn_candidates")
    if not isinstance(raw_records, list) or len(raw_records) != len(records):
        raise FormalValidationError(f"raw/converted frozen candidate counts differ: {source}")
    if int(payload.get("inference_calls_for_group", -1)) != 1:
        raise FormalValidationError("frozen bundle must record exactly one VGN inference")
    candidates = [Candidate6D.from_dict(record) for record in records]
    if any(candidate.group_id != group_id for candidate in candidates):
        raise FormalValidationError("frozen candidate record belongs to another group")
    pool = candidate_pool_fingerprint(candidates)
    if pool != payload.get("candidate_pool_fingerprint"):
        raise FormalValidationError(f"frozen candidate pool fingerprint mismatch: {source}")
    rows = np.asarray(payload.get("graspnet_rows"), dtype=np.float64)
    if rows.shape != (len(candidates), 17):
        raise FormalValidationError(f"frozen candidate evaluator rows have wrong shape: {source}")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise FormalValidationError("frozen candidate IDs are duplicated")
    if [str(record.get("candidate_id", "")) for record in raw_records] != identifiers:
        raise FormalValidationError("raw/converted frozen candidate ordering or membership differs")
    ranks = [candidate.native_rank for candidate in candidates]
    if ranks != sorted(set(ranks)):
        raise FormalValidationError("frozen candidates are not in deterministic native-rank order")
    for candidate, row in zip(candidates, rows, strict=True):
        expected = np.concatenate(
            (
                [
                    candidate.native_score,
                    candidate.width_m,
                    candidate.height_m,
                    candidate.depth_m,
                ],
                np.asarray(candidate.rotation_camera, dtype=np.float64).reshape(-1),
                np.asarray(candidate.translation_camera_m, dtype=np.float64),
            )
        )
        if not np.allclose(row[:16], expected, atol=1e-12, rtol=0):
            raise FormalValidationError("candidate record and GraspNet row geometry differ")
    return identifiers, rows, pool


def _official_reference_no_pruning(
    rows: np.ndarray,
    *,
    models_object_m: Sequence[np.ndarray],
    dexnet_models: Sequence[Any],
    poses_object_to_camera: Sequence[np.ndarray],
    object_ids: Sequence[int],
    dexnet_config: dict[str, Any],
    table_points_camera_m: np.ndarray,
    api_root: Path | str,
    **_: Any,
) -> list[dict[str, Any]]:
    """Faithful official low-level reference that never calls ``eval_grasp``."""

    ensure_graspnetapi_source(api_root)
    try:
        from graspnetAPI.utils import eval_utils as utils
    except (ImportError, ModuleNotFoundError) as error:
        raise FormalValidationError(f"official evaluator reference is unavailable: {error}") from error
    values = np.asarray(rows, dtype=np.float64)
    valid = validate_graspnet_rows(values)
    if not models_object_m or len(models_object_m) != len(dexnet_models):
        raise FormalValidationError("official reference requires one Dex-Net model per object")
    if len(models_object_m) != len(poses_object_to_camera) or len(models_object_m) != len(object_ids):
        raise FormalValidationError("official reference object/model/pose counts differ")
    sampled = [utils.voxel_sample_points(np.asarray(model), 0.008) for model in models_object_m]
    transformed = [
        utils.transform_points(model, np.asarray(pose))
        for model, pose in zip(sampled, poses_object_to_camera, strict=True)
    ]
    scene = np.concatenate(transformed, axis=0)
    instance_by_point = np.concatenate(
        [np.full(len(model), index, dtype=np.int64) for index, model in enumerate(transformed)]
    )
    instances = np.full(len(values), -1, dtype=np.int64)
    associable = np.flatnonzero(np.all(np.isfinite(values[:, 13:16]), axis=1))
    closest = utils.compute_closest_points(values[associable, 13:16], scene)
    instances[associable] = instance_by_point[closest]
    ids = np.full(len(values), -1, dtype=np.int64)
    object_id_array = np.asarray(object_ids, dtype=np.int64)
    ids[associable] = object_id_array[instances[associable]]
    indices_by_model = [
        np.flatnonzero((instances == index) & valid) for index in range(len(sampled))
    ]
    grasps_by_model = [values[indices].copy() for indices in indices_by_model]
    table = np.asarray(table_points_camera_m, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 3 or not np.isfinite(table).all() or len(table) == 0:
        raise FormalValidationError("official reference requires finite non-empty table geometry")
    scene_with_table = np.concatenate((scene, table), axis=0)
    collisions, empty, dex_grasps = utils.collision_detection(
        grasps_by_model,
        transformed,
        dexnet_models,
        poses_object_to_camera,
        scene_with_table,
        outlier=0.05,
        empty_thresh=10,
        return_dexgrasps=True,
    )
    config = copy.deepcopy(dexnet_config)
    try:
        metric = config["metrics"]["force_closure"]
    except (KeyError, TypeError) as error:
        raise FormalValidationError("Dex-Net config lacks metrics.force_closure") from error
    force_closure: dict[float, Any] = {}
    for raw_friction in FRICTION_DESCENDING:
        friction = round(float(raw_friction), 2)
        metric["friction_coef"] = friction
        force_closure[friction] = utils.GraspQualityConfigFactory.create_config(metric)
    collision = np.ones(len(values), dtype=bool)
    friction_scores = np.full(len(values), -1.0, dtype=np.float64)
    for model_index, candidate_indices in enumerate(indices_by_model):
        model_collision = np.asarray(collisions[model_index], dtype=bool)
        model_empty = np.asarray(empty[model_index], dtype=bool)
        if len(model_collision) != len(candidate_indices) or len(model_empty) != len(candidate_indices):
            raise FormalValidationError("official collision helper changed candidate membership")
        collision[candidate_indices] = model_collision
        for local_index, candidate_index in enumerate(candidate_indices):
            if model_collision[local_index]:
                continue
            grasp = dex_grasps[model_index][local_index]
            if grasp is not None:
                friction_scores[candidate_index] = utils.get_grasp_score(
                    grasp,
                    dexnet_models[model_index],
                    FRICTION_DESCENDING,
                    force_closure,
                )
    return [
        {
            "candidate_index": index,
            "associated_object_id": int(ids[index]),
            "collision": bool(collision[index]),
            "friction_score": float(friction_scores[index]),
            "valid_geometry": bool(valid[index]),
        }
        for index in range(len(values))
    ]


def _normalise_evaluations(
    values: Sequence[Any], *, expected_count: int, description: str
) -> list[dict[str, Any]]:
    if len(values) != expected_count:
        raise FormalValidationError(
            f"{description} changed candidate membership: expected {expected_count}, got {len(values)}"
        )
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, CandidateEvaluation):
            record = value.to_record()
        elif isinstance(value, Mapping):
            record = dict(value)
        else:
            raise FormalValidationError(f"{description} returned an unsupported result type")
        if int(record.get("candidate_index", -1)) != index:
            raise FormalValidationError(f"{description} changed candidate ordering")
        required = {
            "associated_object_id",
            "collision",
            "friction_score",
            "valid_geometry",
        }
        if not required.issubset(record):
            raise FormalValidationError(f"{description} omitted fields: {sorted(required - set(record))}")
        friction = float(record["friction_score"])
        if not np.isfinite(friction):
            raise FormalValidationError(f"{description} returned non-finite friction")
        result.append(
            {
                "associated_object_id": int(record["associated_object_id"]),
                "collision": bool(record["collision"]),
                "friction_score": friction,
                "binary_valid": bool(record["valid_geometry"]),
            }
        )
    return result


def run_evaluator_parity_validation(
    target_manifest_path: Path | str,
    selected_group_ids: Sequence[str],
    candidate_bundle_paths: Mapping[str, Path | str],
    output_root: Path | str,
    *,
    dataset_root: Path | str,
    api_root: Path | str = DEFAULT_GRASPNET_API_ROOT,
    scope: str = "formal_real_data",
    fixture_only: bool = False,
    friction_atol: float = 1e-9,
    input_loader: Callable[..., Mapping[str, Any]] | None = None,
    official_reference: Callable[..., Sequence[Any]] | None = None,
    adapter_evaluator: Callable[..., Sequence[Any]] | None = None,
    binding_paths: Mapping[str, Path | str] | None = None,
) -> ParityValidationArtifacts:
    """Compare every frozen candidate with an unpruned official reference."""

    _scope(scope=scope, fixture_only=fixture_only)
    injected = any(value is not None for value in (input_loader, official_reference, adapter_evaluator))
    if injected and not fixture_only:
        raise FormalValidationError("parity callback injection is permitted only for test fixtures")
    if not np.isfinite(friction_atol) or not 0 <= friction_atol <= 1e-9:
        raise FormalValidationError("friction_atol must be in [0, 1e-9]")
    manifest_path = _regular_file(target_manifest_path, "target manifest")
    manifest = _index_manifest(manifest_path)
    selected = _selected_groups(manifest, selected_group_ids)
    if set(selected) - set(candidate_bundle_paths):
        raise FormalValidationError("every selected parity group requires a candidate bundle")
    data_root = Path(dataset_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FormalValidationError(f"dataset root is absent: {data_root}")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    loader = input_loader or _official_group_evaluator_inputs
    reference = official_reference or _official_reference_no_pruning
    adapter = adapter_evaluator or evaluate_frozen_candidates
    comparison_rows: list[dict[str, Any]] = []
    pool_fingerprints: dict[str, str] = {}
    candidate_sources: dict[str, str] = {}
    all_candidate_ids: set[str] = set()
    for group_id in selected:
        record = manifest[group_id]
        if not fixture_only:
            _reject_fixture_record(record, "parity target manifest record")
        candidate_path = _regular_file(candidate_bundle_paths[group_id], "candidate bundle")
        candidate_ids, rows, pool = _load_frozen_bundle(
            candidate_path, group_id, fixture_only=fixture_only
        )
        overlap = all_candidate_ids & set(candidate_ids)
        if overlap:
            raise FormalValidationError(f"candidate IDs repeat across parity groups: {sorted(overlap)}")
        all_candidate_ids.update(candidate_ids)
        pool_fingerprints[group_id] = pool
        candidate_sources[group_id] = sha256_file(candidate_path)
        group = GroupManifest(group_id=group_id, target=record, language={})
        inputs = dict(loader(group, dataset_root=data_root, api_root=api_root))
        required_inputs = {
            "models_object_m",
            "dexnet_models",
            "poses_object_to_camera",
            "object_ids",
            "dexnet_config",
            "table_points_camera_m",
            "source_hashes",
            "dexnet_source_kind",
        }
        if not required_inputs.issubset(inputs):
            raise FormalValidationError(
                f"parity input loader omitted fields: {sorted(required_inputs - set(inputs))}"
            )
        if not fixture_only and inputs["dexnet_source_kind"] != "official_textured_obj_and_sdf_no_pickle":
            raise FormalValidationError("formal parity did not load official OBJ/SDF Dex-Net sources")
        source_hashes = inputs["source_hashes"]
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise FormalValidationError("parity inputs lack official source hashes")
        for raw_path, expected in source_hashes.items():
            source = _regular_file(raw_path, "official parity source")
            if sha256_file(source) != expected:
                raise FormalValidationError(f"official parity source hash is stale: {source}")
        evaluator_kwargs = {
            key: inputs[key]
            for key in (
                "models_object_m",
                "dexnet_models",
                "poses_object_to_camera",
                "object_ids",
                "dexnet_config",
                "table_points_camera_m",
            )
        }
        evaluator_kwargs.update(
            {"target_object_id": int(record["target_object_id"]), "api_root": api_root}
        )
        official_values = _normalise_evaluations(
            list(reference(rows.copy(), **evaluator_kwargs)),
            expected_count=len(rows),
            description="official unpruned reference",
        )
        if adapter_evaluator is None:
            adapter_raw = adapter(
                rows.copy(),
                **evaluator_kwargs,
                parity_gate=None,
                validation_probe=True,
            )
        else:
            adapter_raw = adapter(rows.copy(), **evaluator_kwargs)
        adapter_values = _normalise_evaluations(
            list(adapter_raw),
            expected_count=len(rows),
            description="frozen-candidate adapter",
        )
        for candidate_id, official, adapted in zip(
            candidate_ids, official_values, adapter_values, strict=True
        ):
            comparison_rows.append(
                {
                    "group_id": group_id,
                    "candidate_id": candidate_id,
                    "official_associated_object_id": official["associated_object_id"],
                    "adapter_associated_object_id": adapted["associated_object_id"],
                    "official_collision": official["collision"],
                    "adapter_collision": adapted["collision"],
                    "official_friction_score": official["friction_score"],
                    "adapter_friction_score": adapted["friction_score"],
                    "official_binary_valid": official["binary_valid"],
                    "adapter_binary_valid": adapted["binary_valid"],
                }
            )
    if not comparison_rows:
        raise FormalValidationError("parity selection contains no frozen candidates")
    frame = pd.DataFrame(comparison_rows)
    if frame["candidate_id"].astype(str).duplicated().any():
        raise FormalValidationError("parity comparison candidate IDs are duplicated")
    association_mismatches = int(
        np.count_nonzero(
            frame["official_associated_object_id"].to_numpy(int)
            != frame["adapter_associated_object_id"].to_numpy(int)
        )
    )
    collision_mismatches = int(
        np.count_nonzero(
            frame["official_collision"].to_numpy(bool)
            != frame["adapter_collision"].to_numpy(bool)
        )
    )
    validity_mismatches = int(
        np.count_nonzero(
            frame["official_binary_valid"].to_numpy(bool)
            != frame["adapter_binary_valid"].to_numpy(bool)
        )
    )
    friction_error = np.abs(
        frame["official_friction_score"].to_numpy(float)
        - frame["adapter_friction_score"].to_numpy(float)
    )
    if not np.isfinite(friction_error).all():
        raise FormalValidationError("parity comparison contains non-finite friction errors")
    friction_max_abs_error = float(np.max(friction_error))
    passed = bool(
        association_mismatches == 0
        and collision_mismatches == 0
        and validity_mismatches == 0
        and friction_max_abs_error <= friction_atol
    )
    comparison_path = output / "evaluator_parity_comparison.csv"
    _atomic_csv(comparison_path, frame)
    report_path = output / "evaluator_parity_report.md"
    atomic_text(
        report_path,
        "\n".join(
            [
                "# Frozen-candidate evaluator parity",
                "",
                f"Status: **{'PASSED' if passed else 'FAILED'}**",
                f"Candidates compared without NMS/Top-K pruning: {len(frame)}",
                f"Association mismatches: {association_mismatches}",
                f"Collision mismatches: {collision_mismatches}",
                f"Binary-validity mismatches: {validity_mismatches}",
                f"Maximum friction absolute error: {friction_max_abs_error:.17g}",
                f"Required friction tolerance: {friction_atol:.17g}",
                "",
                "The reference calls official low-level association, collision, and "
                "force-closure helpers for every input row. It never calls eval_grasp.",
            ]
        )
        + "\n",
    )
    evidence_path = output / "evaluator_parity_evidence.json"
    api_source = Path(api_root).expanduser().resolve() / "graspnetAPI/utils/eval_utils.py"
    adapter_source = Path(__file__).resolve().with_name("evaluator.py")
    parity_bindings: dict[str, Path | str] = {
        "official_api_source_sha256": api_source,
        "adapter_source_sha256": adapter_source,
        "dataset_manifest_sha256": manifest_path,
    }
    if binding_paths is not None and not fixture_only:
        raise FormalValidationError("formal parity binding paths are fixed by the consumed sources")
    if binding_paths is not None:
        parity_bindings.update(binding_paths)
    bindings, checked_paths = _binding_records(
        evidence_path,
        parity_bindings,
        required={
            "official_api_source_sha256",
            "adapter_source_sha256",
            "dataset_manifest_sha256",
        },
    )
    aggregate_pool = canonical_sha256(
        [{"group_id": group_id, "pool_fingerprint": pool_fingerprints[group_id]} for group_id in selected]
    )
    summary = {
        "association_mismatches": association_mismatches,
        "collision_mismatches": collision_mismatches,
        "binary_validity_mismatches": validity_mismatches,
        "friction_max_abs_error": friction_max_abs_error,
    }
    evidence = {
        "schema_version": PARITY_EVIDENCE_SCHEMA,
        "scope": scope,
        "fixture_only": fixture_only,
        "status": "PASSED" if passed else "FAILED",
        "group_id": selected[0] if len(selected) == 1 else f"aggregate:{aggregate_pool[:16]}",
        "validated_group_ids": list(selected),
        "grounding_condition": ORACLE_GROUNDING_CONDITION,
        "candidate_count": len(frame),
        "candidate_pool_fingerprint": aggregate_pool,
        "candidate_pool_fingerprints": pool_fingerprints,
        "candidate_bundle_sha256": candidate_sources,
        "friction_atol": float(friction_atol),
        "bindings": bindings,
        "binding_paths": checked_paths,
        "comparison_csv_path": _relative(comparison_path, evidence_path.parent),
        "comparison_csv_sha256": sha256_file(comparison_path),
        "report_path": _relative(report_path, evidence_path.parent),
        "report_sha256": sha256_file(report_path),
        "reference_operation": "official_low_level_per_candidate_no_eval_grasp_no_nms_no_topk",
        "adapter_operation": "frozen_candidate_low_level_no_nms_no_topk",
        "recomputed_summary": summary,
    }
    atomic_json(evidence_path, evidence)
    if not passed:
        raise FormalValidationError(
            f"evaluator parity failed; inspect {comparison_path} and {evidence_path}"
        )
    gate_path = output / "evaluator_parity_gate.json"
    atomic_json(
        gate_path,
        {
            "schema_version": PARITY_SCHEMA,
            "validated": True,
            "artifact_path": _relative(evidence_path, gate_path.parent),
        },
    )
    return ParityValidationArtifacts(
        comparison_csv_path=str(comparison_path),
        report_path=str(report_path),
        evidence_path=str(evidence_path),
        gate_path=str(gate_path),
    )


__all__ = [
    "AI_ASSISTED_GROUP_CHECKS",
    "AI_ASSISTED_REVIEW_DISCLAIMER",
    "AI_ASSISTED_REVIEW_KIND",
    "FORMAL_GEOMETRY_MINIMUM_GROUPS",
    "FormalValidationError",
    "GEOMETRY_RENDER_SCHEMA",
    "GeometryValidationArtifacts",
    "ParityValidationArtifacts",
    "RAW_VGN_VALIDATION_SCHEMA",
    "VISUAL_REVIEW_SCHEMA",
    "run_evaluator_parity_validation",
    "run_geometry_validation",
]
