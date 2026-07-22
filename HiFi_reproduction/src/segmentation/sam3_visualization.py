"""Headless visualizations for prompts, hypotheses, and selected masks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from .sam3_prompt_builder import VisualPrompt


def _save(figure, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return path


def save_prompt_visualization(rgb: np.ndarray, prompt: VisualPrompt, path: Path | str) -> Path:
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.imshow(rgb)
    x1, y1, x2, y2 = prompt.tight_box_xyxy
    axis.add_patch(plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, color="cyan", linewidth=1.5))
    x1, y1, x2, y2 = prompt.expanded_box_xyxy
    axis.add_patch(plt.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1, fill=False, color="yellow", linewidth=2.0))
    if prompt.positive_points_xy:
        x, y = zip(*prompt.positive_points_xy)
        axis.scatter(x, y, c="#00e676", marker="*", s=70, edgecolors="black")
    if prompt.negative_points_xy:
        x, y = zip(*prompt.negative_points_xy)
        axis.scatter(x, y, c="#ff1744", marker="x", s=55)
    axis.contour(prompt.cleaned_mask.astype(np.uint8), levels=[0.5], colors=["white"], linewidths=0.8)
    axis.set_title(f"SAM 3 prompt: {prompt.strategy}")
    axis.axis("off")
    return _save(figure, path)


def save_candidate_grid(
    rgb: np.ndarray,
    masks: Sequence[np.ndarray],
    metrics: Sequence[Mapping[str, object]],
    path: Path | str,
) -> Path:
    count = max(1, len(masks))
    columns = min(3, count)
    rows = int(math.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, mask, record in zip(axes.flat, masks, metrics):
        axis.imshow(rgb)
        axis.imshow(np.ma.masked_where(~np.asarray(mask, dtype=bool), mask), cmap="spring", alpha=0.45)
        axis.set_title(
            f"{record['candidate_id']} sam={float(record['sam_quality']):.3f} "
            f"score={float(record['refinement_score']):.3f}"
        )
        axis.axis("off")
    return _save(figure, path)


def save_mask_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path | str, *, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.imshow(rgb)
    axis.imshow(np.ma.masked_where(~np.asarray(mask, dtype=bool), mask), cmap="spring", alpha=0.45)
    axis.contour(np.asarray(mask, dtype=np.uint8), levels=[0.5], colors=["yellow"], linewidths=1.2)
    axis.set_title(title)
    axis.axis("off")
    return _save(figure, path)


def save_coarse_vs_refined(
    rgb: np.ndarray,
    coarse: np.ndarray,
    refined: np.ndarray,
    path: Path | str,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, mask, title in zip(axes[:2], (coarse, refined), ("HiFi-CS coarse", "Selected refined")):
        axis.imshow(rgb)
        axis.imshow(np.ma.masked_where(~np.asarray(mask, dtype=bool), mask), cmap="spring", alpha=0.45)
        axis.set_title(title)
        axis.axis("off")
    delta = np.zeros((*coarse.shape, 3), dtype=np.uint8)
    delta[np.asarray(coarse, dtype=bool) & ~np.asarray(refined, dtype=bool)] = (255, 64, 64)
    delta[~np.asarray(coarse, dtype=bool) & np.asarray(refined, dtype=bool)] = (64, 255, 64)
    axes[2].imshow(rgb)
    unchanged = np.repeat(np.all(delta == 0, axis=2)[..., None], 3, axis=2)
    axes[2].imshow(np.ma.masked_where(unchanged, delta), alpha=0.65)
    axes[2].set_title("Delta: removed red / added green")
    axes[2].axis("off")
    return _save(figure, path)
