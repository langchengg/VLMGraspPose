"""Coordinate and rectangle geometry for the unified 4-DoF protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.draw import polygon as draw_polygon


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize a parallel-jaw angle to ``[-90, 90)`` degrees."""

    value = float(angle_deg)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    normalized = (value + 90.0) % 180.0 - 90.0
    return 0.0 if normalized == 0.0 else float(normalized)


def periodic_angle_difference_deg(first: float, second: float) -> float:
    """Smallest parallel-jaw angular difference in ``[0, 90]``."""

    return abs(normalize_angle_deg(float(first) - float(second)))


def transform_oriented_length(
    length: float,
    angle_deg: float,
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    """Transform an oriented length and its angle under anisotropic scaling."""

    values = (float(length), float(scale_x), float(scale_y))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("length and scales must be finite")
    if values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0:
        raise ValueError("length and scales must be positive")
    radians = math.radians(float(angle_deg))
    dx = math.cos(radians) * values[1]
    dy = math.sin(radians) * values[2]
    factor = math.hypot(dx, dy)
    return values[0] * factor, normalize_angle_deg(math.degrees(math.atan2(dy, dx)))


@dataclass(frozen=True, slots=True)
class CropTransform:
    """Axis-aligned crop resized to a model tensor, with reversible mapping."""

    crop_x: float
    crop_y: float
    crop_width: float
    crop_height: float
    model_width: int
    model_height: int
    native_width: int
    native_height: int

    def __post_init__(self) -> None:
        numeric = (
            self.crop_x,
            self.crop_y,
            self.crop_width,
            self.crop_height,
            self.model_width,
            self.model_height,
            self.native_width,
            self.native_height,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("crop transform values must be finite")
        if min(
            self.crop_width,
            self.crop_height,
            self.model_width,
            self.model_height,
            self.native_width,
            self.native_height,
        ) <= 0:
            raise ValueError("crop and image dimensions must be positive")

    def native_to_model_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            (float(x) - self.crop_x) * self.model_width / self.crop_width,
            (float(y) - self.crop_y) * self.model_height / self.crop_height,
        )

    def model_to_native_point(
        self, x: float, y: float, *, clip: bool = False
    ) -> tuple[float, float]:
        native_x = self.crop_x + float(x) * self.crop_width / self.model_width
        native_y = self.crop_y + float(y) * self.crop_height / self.model_height
        if clip:
            native_x = float(np.clip(native_x, 0.0, self.native_width - 1.0))
            native_y = float(np.clip(native_y, 0.0, self.native_height - 1.0))
        return native_x, native_y

    def native_to_model_pose(
        self, x: float, y: float, angle_deg: float, width_px: float
    ) -> tuple[float, float, float, float]:
        model_x, model_y = self.native_to_model_point(x, y)
        width, angle = transform_oriented_length(
            width_px,
            angle_deg,
            scale_x=self.model_width / self.crop_width,
            scale_y=self.model_height / self.crop_height,
        )
        return model_x, model_y, angle, width

    def model_to_native_pose(
        self,
        x: float,
        y: float,
        angle_deg: float,
        width_px: float,
        *,
        clip: bool = False,
    ) -> tuple[float, float, float, float]:
        native_x, native_y = self.model_to_native_point(x, y, clip=clip)
        width, angle = transform_oriented_length(
            width_px,
            angle_deg,
            scale_x=self.crop_width / self.model_width,
            scale_y=self.crop_height / self.model_height,
        )
        return native_x, native_y, angle, width


def rectangle_corners(
    center_x: float,
    center_y: float,
    width_px: float,
    height_px: float,
    angle_deg: float,
) -> np.ndarray:
    """Return four clockwise ``[x, y]`` vertices."""

    center = np.asarray([center_x, center_y], dtype=np.float64)
    radians = math.radians(normalize_angle_deg(angle_deg))
    axis = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    return np.stack(
        [
            center + sx * width_px * 0.5 * axis + sy * height_px * 0.5 * normal
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
        ]
    )


def rotated_rectangle_iou(first: object, second: object) -> float:
    """Continuous rotated IoU for objects exposing the Grasp4DoF fields."""

    first_polygon = rectangle_corners(
        first.center_x, first.center_y, first.width_px, first.height_px, first.angle_deg
    ).astype(np.float32)
    second_polygon = rectangle_corners(
        second.center_x,
        second.center_y,
        second.width_px,
        second.height_px,
        second.angle_deg,
    ).astype(np.float32)
    first_area = float(abs(cv2.contourArea(first_polygon)))
    second_area = float(abs(cv2.contourArea(second_polygon)))
    intersection, _ = cv2.intersectConvexConvex(first_polygon, second_polygon)
    union = first_area + second_area - float(intersection)
    return 0.0 if union <= 0.0 else float(intersection / union)


def _rectangle_pixels(grasp: object, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image shape must be positive")
    box = rectangle_corners(
        grasp.center_x,
        grasp.center_y,
        grasp.width_px,
        grasp.height_px,
        grasp.angle_deg,
    ).astype(np.intp)
    rows, columns = draw_polygon(box[:, 1], box[:, 0], shape=(height, width))
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(rows.astype(np.int64) * width + columns.astype(np.int64))


def rasterized_rectangle_iou(
    first: object, second: object, *, shape: tuple[int, int] = (480, 640)
) -> float:
    """Corrected CROG-style clipped pixel IoU (vertices x/y, raster rows y/x)."""

    first_pixels = _rectangle_pixels(first, shape)
    second_pixels = _rectangle_pixels(second, shape)
    if first_pixels.size == 0 and second_pixels.size == 0:
        return 0.0
    intersection = np.intersect1d(first_pixels, second_pixels, assume_unique=True).size
    union = int(first_pixels.size + second_pixels.size - intersection)
    return 0.0 if union == 0 else float(intersection / union)
