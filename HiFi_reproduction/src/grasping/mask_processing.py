"""Dependency-light processing of target masks for grasp sampling.

The functions in this module never modify their input arrays.  Processed masks
are returned as two-dimensional boolean arrays so callers have to opt in before
converting them to an SDK-specific image representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise ValueError(f"{name} must be HxW (or HxWx1), got {result.shape}")
    return result


def to_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a binary or probability mask to a finite boolean mask.

    Values greater than or equal to ``threshold`` are foreground.  NaN and
    infinite values are always background.  This also handles conventional
    uint8 masks containing 0 and 255 without special casing.
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    values = _as_2d(mask, "mask")
    if values.dtype == np.bool_:
        return values.copy()
    finite = np.isfinite(values)
    return finite & (values >= threshold)


def resize_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize a mask to ``(height, width)`` using nearest-neighbour only."""

    source = to_binary_mask(mask)
    if len(target_shape) != 2:
        raise ValueError("target_shape must contain (height, width)")
    height, width = (int(target_shape[0]), int(target_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("target dimensions must be positive")
    if source.shape == (height, width):
        return source.copy()
    image = Image.fromarray(source.astype(np.uint8) * 255, mode="L")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 (4-neighbour) or 2 (8-neighbour)")
    return ndimage.generate_binary_structure(2, connectivity)


def remove_small_components(
    mask: np.ndarray,
    min_size_px: int,
    *,
    connectivity: int = 2,
) -> np.ndarray:
    """Remove connected foreground components smaller than ``min_size_px``."""

    binary = to_binary_mask(mask)
    min_size_px = int(min_size_px)
    if min_size_px < 0:
        raise ValueError("min_size_px cannot be negative")
    if min_size_px <= 1 or not np.any(binary):
        return binary.copy()
    labels, count = ndimage.label(binary, structure=_connectivity_structure(connectivity))
    if count == 0:
        return np.zeros_like(binary)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_size_px
    keep[0] = False
    return keep[labels]


def retain_relevant_component(
    mask: np.ndarray,
    reference_uv: Optional[Sequence[float]] = None,
    *,
    connectivity: int = 2,
) -> np.ndarray:
    """Retain a mapped component, falling back deterministically to the largest.

    If ``reference_uv=(u, v)`` rounds to a foreground pixel, its component is
    selected.  Otherwise the largest component is selected.  Equal-size ties
    are resolved by scan order, making the result deterministic.
    """

    binary = to_binary_mask(mask)
    labels, count = ndimage.label(binary, structure=_connectivity_structure(connectivity))
    if count == 0:
        return np.zeros_like(binary)

    selected = 0
    if reference_uv is not None:
        if len(reference_uv) != 2 or not np.all(np.isfinite(reference_uv)):
            raise ValueError("reference_uv must contain two finite values")
        u, v = np.rint(reference_uv).astype(int)
        if 0 <= v < binary.shape[0] and 0 <= u < binary.shape[1]:
            selected = int(labels[v, u])

    if selected == 0:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        selected = int(np.argmax(sizes))
    return labels == selected


def retain_largest_component(mask: np.ndarray, *, connectivity: int = 2) -> np.ndarray:
    """Retain only the largest connected foreground component."""

    return retain_relevant_component(mask, connectivity=connectivity)


def disk_structure(radius_px: int) -> np.ndarray:
    """Return a circular binary structuring element with integer radius."""

    radius_px = int(radius_px)
    if radius_px < 0:
        raise ValueError("radius_px cannot be negative")
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def binary_erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = to_binary_mask(mask)
    if int(radius_px) == 0:
        return binary.copy()
    return ndimage.binary_erosion(binary, structure=disk_structure(radius_px), border_value=0)


def binary_dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = to_binary_mask(mask)
    if int(radius_px) == 0:
        return binary.copy()
    return ndimage.binary_dilation(binary, structure=disk_structure(radius_px), border_value=0)


def valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    """Return pixels containing finite, strictly positive metric depth."""

    depth = _as_2d(depth_m, "depth_m")
    return np.isfinite(depth) & (depth > 0)


def intersect_valid_depth(mask: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Intersect an aligned binary mask with finite positive depth."""

    binary = to_binary_mask(mask)
    valid = valid_depth_mask(depth_m)
    if binary.shape != valid.shape:
        raise ValueError(f"mask/depth shape mismatch: {binary.shape} versus {valid.shape}")
    return binary & valid


@dataclass(frozen=True)
class MaskProcessingResult:
    """Diagnostic snapshots from target-mask processing."""

    original_binary: np.ndarray
    resized_binary: np.ndarray
    valid_depth: np.ndarray
    processed: np.ndarray


def process_mask_with_diagnostics(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    threshold: float = 0.5,
    min_component_size_px: int = 0,
    keep_largest_component: bool = False,
    reference_uv: Optional[Sequence[float]] = None,
    erode_radius_px: int = 0,
    dilate_radius_px: int = 0,
) -> MaskProcessingResult:
    """Process a predicted mask while retaining immutable diagnostic stages.

    Morphology is applied after component selection.  The final mask is
    intersected with valid depth again, so dilation can never introduce pixels
    for which no depth measurement exists.
    """

    depth = _as_2d(depth_m, "depth_m")
    original = to_binary_mask(mask, threshold)
    resized = resize_mask_nearest(original, depth.shape)
    processed = remove_small_components(resized, min_component_size_px)
    if keep_largest_component or reference_uv is not None:
        processed = retain_relevant_component(processed, reference_uv)
    valid = valid_depth_mask(depth)
    processed = processed & valid
    processed = binary_erode(processed, erode_radius_px)
    processed = binary_dilate(processed, dilate_radius_px)
    processed &= valid
    return MaskProcessingResult(
        original_binary=original.copy(),
        resized_binary=resized.copy(),
        valid_depth=valid.copy(),
        processed=processed.copy(),
    )


def process_mask(mask: np.ndarray, depth_m: np.ndarray, **kwargs: object) -> np.ndarray:
    """Return only the processed mask; see :func:`process_mask_with_diagnostics`."""

    return process_mask_with_diagnostics(mask, depth_m, **kwargs).processed

