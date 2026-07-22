"""Headless, publication-friendly overlays for planar grasp candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .camera_geometry import grasp_endpoints_uv


def _get(candidate: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(candidate, Mapping) and name in candidate:
            return candidate[name]
        if hasattr(candidate, name):
            return getattr(candidate, name)
    return default


def _optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _center(candidate: Any) -> np.ndarray:
    value = _get(candidate, "center_uv")
    if value is None:
        value = [_get(candidate, "center_u_px", "u"), _get(candidate, "center_v_px", "v")]
    center = np.asarray(value, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("candidate has no finite center_uv")
    return center


def _endpoints(candidate: Any) -> np.ndarray:
    value = _get(candidate, "endpoints_uv")
    if value is None:
        first = _get(candidate, "endpoint_1_uv")
        second = _get(candidate, "endpoint_2_uv")
        if first is not None and second is not None:
            value = [first, second]
    if value is not None:
        endpoints = np.asarray(value, dtype=np.float64)
        if endpoints.shape == (2, 2) and np.all(np.isfinite(endpoints)):
            return endpoints
    width = float(_get(candidate, "width_px"))
    angle = float(_get(candidate, "angle_rad", "angle"))
    return grasp_endpoints_uv(_center(candidate), width, angle)


def _rgb_array(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"RGB image must be HxWx3 or HxWx4, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        finite = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if finite.size and np.max(finite) > 1.0:
            finite = finite / 255.0
        return np.clip(finite, 0.0, 1.0)
    return image


def _draw_mask_contour(axis: Any, mask: Optional[np.ndarray]) -> None:
    if mask is None:
        return
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be HxW")
    if np.any(binary) and not np.all(binary):
        axis.contour(binary.astype(np.uint8), levels=[0.5], colors=["#00e676"], linewidths=1.2)


def draw_candidates(
    axis: Any,
    candidates: Iterable[Any],
    *,
    show_ids: bool = True,
    show_scores: bool = True,
    color: str = "#00b7ff",
    rejected_color: str = "#ff3b30",
    score_range: Optional[Tuple[float, float]] = None,
    score_field: str = "gqcnn_q_value",
    score_label: str = "q",
) -> None:
    """Draw candidate centres, axes, jaw endpoints, IDs and optional scores."""

    candidates = list(candidates)
    score_map = plt.get_cmap("viridis")
    if score_range is None:
        scores = [
            _optional_float(
                _get(
                    item,
                    score_field,
                    *(("quality",) if score_field == "gqcnn_q_value" else ()),
                    default=np.nan,
                )
            )
            for item in candidates
        ]
        finite_scores = np.asarray([score for score in scores if np.isfinite(score)])
        if finite_scores.size:
            low, high = float(np.min(finite_scores)), float(np.max(finite_scores))
            score_range = (low, high if high > low else low + 1.0)

    for index, candidate in enumerate(candidates):
        center = _center(candidate)
        endpoints = _endpoints(candidate)
        rejection = _get(candidate, "rejection_reason")
        score = _optional_float(
            _get(
                candidate,
                score_field,
                *(("quality",) if score_field == "gqcnn_q_value" else ()),
                default=np.nan,
            )
        )
        candidate_color = rejected_color if rejection not in (None, "") else color
        if rejection in (None, "") and np.isfinite(score) and score_range is not None:
            low, high = score_range
            candidate_color = score_map(np.clip((score - low) / max(high - low, 1e-12), 0.0, 1.0))
        axis.plot(endpoints[:, 0], endpoints[:, 1], color=candidate_color, linewidth=1.5, zorder=3)
        axis.scatter(
            endpoints[:, 0], endpoints[:, 1], marker="|", s=36, linewidths=1.7,
            color=candidate_color, zorder=4,
        )
        axis.scatter(center[0], center[1], marker="o", s=14, color="white", edgecolors=candidate_color, linewidths=1.0, zorder=5)
        label_parts = []
        if show_ids:
            label_parts.append(str(_get(candidate, "candidate_id", default=index)))
        if show_scores and np.isfinite(score):
            label_parts.append(f"{score_label}={score:.3f}")
        if label_parts:
            axis.text(
                center[0] + 3, center[1] - 3, " ".join(label_parts), color="white",
                fontsize=6, zorder=6,
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.0},
            )


def _finish_figure(figure: Any, axis: Any, output_path: Path | str, title: Optional[str], dpi: int) -> Path:
    if title:
        axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout(pad=0.1)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return destination


def save_candidate_overlay(
    rgb: np.ndarray,
    candidates: Iterable[Any],
    output_path: Path | str,
    *,
    mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    show_ids: bool = True,
    show_scores: bool = True,
    score_range: Optional[Tuple[float, float]] = None,
    score_field: str = "gqcnn_q_value",
    score_label: str = "q",
    dpi: int = 150,
) -> Path:
    """Save RGB + optional target contour + planar candidates."""

    image = _rgb_array(rgb)
    if mask is not None and np.asarray(mask).shape != image.shape[:2]:
        raise ValueError("RGB/mask shape mismatch")
    figure, axis = plt.subplots(
        figsize=(max(4.0, image.shape[1] / 100), max(3.0, image.shape[0] / 100)),
        dpi=100,
    )
    axis.imshow(image)
    _draw_mask_contour(axis, mask)
    draw_candidates(
        axis,
        candidates,
        show_ids=show_ids,
        show_scores=show_scores,
        score_range=score_range,
        score_field=score_field,
        score_label=score_label,
    )
    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    return _finish_figure(figure, axis, output_path, title, dpi)


def save_depth_visualization(
    depth_m: np.ndarray,
    output_path: Path | str,
    *,
    candidates: Iterable[Any] = (),
    mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """Save metric depth (invalid values transparent) with optional candidates."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be HxW")
    if mask is not None and np.asarray(mask).shape != depth.shape:
        raise ValueError("depth/mask shape mismatch")
    valid = np.isfinite(depth) & (depth > 0)
    display = np.ma.masked_where(~valid, depth)
    figure, axis = plt.subplots(
        figsize=(max(4.0, depth.shape[1] / 100), max(3.0, depth.shape[0] / 100)),
        dpi=100,
    )
    image = axis.imshow(display, cmap="magma")
    image.cmap.set_bad(color="#222222")
    _draw_mask_contour(axis, mask)
    draw_candidates(axis, candidates)
    axis.set_xlim(-0.5, depth.shape[1] - 0.5)
    axis.set_ylim(depth.shape[0] - 0.5, -0.5)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_label("Depth (m)")
    return _finish_figure(figure, axis, output_path, title, dpi)


def save_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path | str,
    *,
    candidates: Iterable[Any] = (),
    alpha: float = 0.35,
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """Save a translucent target mask over RGB with optional valid candidates."""

    image = _rgb_array(rgb)
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != image.shape[:2]:
        raise ValueError("RGB/mask shape mismatch")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    overlay = np.zeros((*binary.shape, 4), dtype=np.float32)
    overlay[..., 1] = 1.0
    overlay[..., 3] = binary.astype(np.float32) * float(alpha)
    figure, axis = plt.subplots(
        figsize=(max(4.0, image.shape[1] / 100), max(3.0, image.shape[0] / 100)),
        dpi=100,
    )
    axis.imshow(image)
    axis.imshow(overlay)
    _draw_mask_contour(axis, binary)
    draw_candidates(axis, candidates)
    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    return _finish_figure(figure, axis, output_path, title, dpi)
