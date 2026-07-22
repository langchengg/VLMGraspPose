"""Research visualizations for target-filtered official VGN candidates.

All colors and drawing conventions live in this module; they never influence
candidate generation, filtering, or selection.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


COLORS = {
    "mask": (0.10, 0.85, 0.35, 0.35),
    "accepted": "#25d366",
    "rejected": "#ff453a",
    "top1": "#ffd60a",
    "approach": "#00c7ff",
    "jaw": "#ff9f0a",
}


def _record(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        return candidate
    converter = getattr(candidate, "to_record", None)
    if callable(converter):
        return converter()
    return vars(candidate)


def _atomic_figure(figure: Any, path: Path | str, *, dpi: int = 160) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    figure.savefig(temporary, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    os.replace(temporary, destination)
    return destination


def _base_axis(rgb: np.ndarray) -> tuple[Any, Any]:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must be HxWx3, got {image.shape}")
    figure, axis = plt.subplots(
        figsize=(max(6.4, image.shape[1] / 90), max(4.8, image.shape[0] / 90))
    )
    axis.imshow(image)
    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    axis.set_axis_off()
    return figure, axis


def _draw_mask(axis: Any, mask: np.ndarray) -> None:
    binary = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
    rgba[binary] = COLORS["mask"]
    axis.imshow(rgba)
    boundary = binary ^ ndimage.binary_erosion(binary)
    if np.any(boundary):
        axis.contour(boundary.astype(np.uint8), levels=[0.5], colors=[COLORS["accepted"]], linewidths=0.7)


def save_rgb_mask_overlay(
    rgb: np.ndarray, mask: np.ndarray, path: Path | str, *, title: str | None = None
) -> Path:
    figure, axis = _base_axis(rgb)
    if np.asarray(mask).shape != np.asarray(rgb).shape[:2]:
        raise ValueError("RGB/mask shape mismatch")
    _draw_mask(axis, mask)
    if title:
        axis.set_title(title)
    return _atomic_figure(figure, path)


def save_candidates_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    candidates: Iterable[Any],
    path: Path | str,
    *,
    top_k: int | None = None,
    target_only: bool = False,
    title: str | None = None,
) -> Path:
    figure, axis = _base_axis(rgb)
    _draw_mask(axis, mask)
    records = [_record(candidate) for candidate in candidates]
    if target_only:
        records = [r for r in records if bool(r.get("inside_dilated_target_mask", False))]
    if top_k is not None:
        records = records[: max(0, int(top_k))]
    for index, record in enumerate(records):
        uv = record.get("projected_uv")
        if uv is None or len(uv) != 2 or not np.all(np.isfinite(uv)):
            continue
        accepted = bool(record.get("inside_dilated_target_mask", False))
        color = COLORS["accepted"] if accepted else COLORS["rejected"]
        marker = "o" if accepted else "x"
        axis.scatter(uv[0], uv[1], c=color, marker=marker, s=24, linewidths=1.0)
        quality = float(record.get("vgn_quality", np.nan))
        label_index = record.get("score_rank", record.get("official_selection_index", index))
        axis.text(
            float(uv[0]) + 3,
            float(uv[1]) - 3,
            f"{label_index}:{quality:.3f}",
            fontsize=5.5,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 0.8},
        )
    if title:
        axis.set_title(title)
    return _atomic_figure(figure, path)


def _project(point: np.ndarray, intrinsics: Any) -> np.ndarray | None:
    xyz = np.asarray(point, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or xyz[2] <= 0:
        return None
    return np.array(
        [intrinsics.fx * xyz[0] / xyz[2] + intrinsics.cx,
         intrinsics.fy * xyz[1] / xyz[2] + intrinsics.cy],
        dtype=np.float64,
    )


def save_top1_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    candidate: Any,
    intrinsics: Any,
    path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    record = _record(candidate)
    transform = np.asarray(record["T_camera_grasp"], dtype=np.float64).reshape(4, 4)
    center = transform[:3, 3]
    rotation = transform[:3, :3]
    approach_tip = center + rotation[:, 2] * 0.05
    half_width = 0.5 * float(record["width_m"])
    jaws = (center - rotation[:, 1] * half_width, center + rotation[:, 1] * half_width)
    center_uv = _project(center, intrinsics)
    approach_uv = _project(approach_tip, intrinsics)
    jaw_uv = [_project(point, intrinsics) for point in jaws]

    figure, axis = _base_axis(rgb)
    _draw_mask(axis, mask)
    if center_uv is not None:
        axis.scatter(*center_uv, c=COLORS["top1"], marker="*", s=110, edgecolors="black")
    if center_uv is not None and approach_uv is not None:
        axis.annotate(
            "",
            xy=approach_uv,
            xytext=center_uv,
            arrowprops={"arrowstyle": "->", "color": COLORS["approach"], "lw": 2.0},
        )
    if all(point is not None for point in jaw_uv):
        points = np.stack(jaw_uv)
        axis.plot(points[:, 0], points[:, 1], color=COLORS["jaw"], linewidth=2.4)
        axis.scatter(points[:, 0], points[:, 1], c=COLORS["jaw"], marker="|", s=75)
    if title:
        axis.set_title(title)
    return _atomic_figure(figure, path)


def save_quality_max_projection(quality: np.ndarray, path: Path | str) -> Path:
    volume = np.asarray(quality, dtype=np.float32)
    if volume.shape != (40, 40, 40):
        raise ValueError(f"quality volume must be 40^3, got {volume.shape}")
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for axis_index, axis in enumerate(axes):
        projection = np.max(volume, axis=axis_index)
        image = axis.imshow(projection.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(f"max over task {'xyz'[axis_index]}")
        axis.set_xlabel("voxel")
        axis.set_ylabel("voxel")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle("Official VGN processed quality max projections")
    figure.tight_layout()
    return _atomic_figure(figure, path)


def atomic_write_point_cloud(path: Path | str, points: np.ndarray, colors: np.ndarray | None = None) -> Path:
    import open3d as o3d

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    if not o3d.io.write_point_cloud(str(temporary), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Open3D failed to write {temporary}")
    os.replace(temporary, destination)
    return destination


def save_grasps_3d_ply(
    path: Path | str,
    scene_points_camera: np.ndarray,
    target_points_camera: np.ndarray,
    candidates: Iterable[Any],
    *,
    top_k: int = 20,
    T_camera_task: np.ndarray | None = None,
    workspace_size_m: float = 0.30,
    table_height_m: float = 0.05,
    selected_official_index: int | None = None,
) -> Path:
    """Save a colored point-sampled 3-D diagnostic without requiring a GUI."""
    scene = np.asarray(scene_points_camera, dtype=np.float64)[:: max(1, len(scene_points_camera) // 40000)]
    target = np.asarray(target_points_camera, dtype=np.float64)
    point_sets = [scene, target]
    color_sets = [
        np.tile(np.array([0.55, 0.58, 0.62]), (len(scene), 1)),
        np.tile(np.array([0.12, 0.90, 0.34]), (len(target), 1)),
    ]
    if T_camera_task is not None:
        transform = np.asarray(T_camera_task, dtype=np.float64).reshape(4, 4)

        def task_to_camera(points: np.ndarray) -> np.ndarray:
            return points @ transform[:3, :3].T + transform[:3, 3]

        size = float(workspace_size_m)
        corners = np.array(
            [[x, y, z] for x in (0.0, size) for y in (0.0, size) for z in (0.0, size)],
            dtype=np.float64,
        )
        edges = []
        for first in range(8):
            for second in range(first + 1, 8):
                if np.count_nonzero(corners[first] != corners[second]) == 1:
                    edges.append(
                        np.linspace(corners[first], corners[second], 25, dtype=np.float64)
                    )
        cube_points = task_to_camera(np.vstack(edges))
        point_sets.append(cube_points)
        color_sets.append(np.tile(np.array([0.75, 0.28, 1.0]), (len(cube_points), 1)))

        plane_axis = np.linspace(0.0, size, 28)
        xx, yy = np.meshgrid(plane_axis, plane_axis, indexing="xy")
        plane_task = np.column_stack(
            (xx.ravel(), yy.ravel(), np.full(xx.size, float(table_height_m)))
        )
        plane_points = task_to_camera(plane_task)
        point_sets.append(plane_points)
        color_sets.append(np.tile(np.array([0.15, 0.75, 0.90]), (len(plane_points), 1)))
    candidates_list = list(candidates)
    displayed = candidates_list[: max(0, int(top_k))]
    if selected_official_index is not None and not any(
        int(_record(candidate)["official_selection_index"])
        == int(selected_official_index)
        for candidate in displayed
    ):
        selected = next(
            (
                candidate
                for candidate in candidates_list
                if int(_record(candidate)["official_selection_index"])
                == int(selected_official_index)
            ),
            None,
        )
        if selected is not None:
            displayed.append(selected)
    for candidate in displayed:
        record = _record(candidate)
        transform = np.asarray(record["T_camera_grasp"], dtype=np.float64).reshape(4, 4)
        center, rotation = transform[:3, 3], transform[:3, :3]
        width = float(record["width_m"])
        samples = np.linspace(-0.5, 0.5, 25)
        jaw_line = center[None, :] + samples[:, None] * width * rotation[:, 1][None, :]
        approach = center[None, :] - np.linspace(0.0, 0.05, 20)[:, None] * rotation[:, 2][None, :]
        points = np.vstack((jaw_line, approach))
        is_selected = (
            selected_official_index is not None
            and int(record["official_selection_index"]) == int(selected_official_index)
        )
        color = np.array([1.0, 0.82, 0.05]) if is_selected else np.array([0.05, 0.65, 1.0])
        point_sets.append(points)
        color_sets.append(np.tile(color, (len(points), 1)))
    return atomic_write_point_cloud(path, np.vstack(point_sets), np.vstack(color_sets))
