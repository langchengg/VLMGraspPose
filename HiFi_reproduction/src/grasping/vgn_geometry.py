"""Strict metric geometry utilities for the HiFi-CS to VGN adapter.

This module deliberately contains no VGN network code.  It establishes the
metric camera/task-frame contract needed by the official VGN implementation.
Open3D and PyYAML are imported lazily so pure geometry and unit tests remain
usable when those optional runtime dependencies are not installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STATUS_MISSING_CAMERA_INTRINSICS = "missing_camera_intrinsics"
STATUS_AMBIGUOUS_DEPTH_UNIT = "ambiguous_depth_unit"
STATUS_INVALID_DEPTH = "invalid_depth"
STATUS_EMPTY_MASK = "empty_mask"
STATUS_MASK_TOO_SMALL = "mask_too_small"
STATUS_MASK_TOO_LARGE = "mask_too_large"
STATUS_INSUFFICIENT_MASKED_DEPTH = "insufficient_masked_depth"
STATUS_MASK_DEPTH_SHAPE_ERROR = "mask_depth_shape_error"
STATUS_SUPPORT_PLANE_FAILED = "support_plane_failed"


class GeometryError(RuntimeError):
    """A sample-local geometry failure with a machine-readable status code."""

    def __init__(
        self, status: str, message: str, *, details: Mapping[str, Any] | None = None
    ):
        super().__init__(message)
        self.status = str(status)
        self.status_code = self.status
        self.details = dict(details or {})


class MissingCameraIntrinsicsError(GeometryError):
    def __init__(self, message: str, **details: Any):
        super().__init__(STATUS_MISSING_CAMERA_INTRINSICS, message, details=details)


class AmbiguousDepthUnitError(GeometryError):
    def __init__(self, message: str, **details: Any):
        super().__init__(STATUS_AMBIGUOUS_DEPTH_UNIT, message, details=details)


class MaskGeometryError(GeometryError):
    pass


class SupportPlaneError(GeometryError):
    def __init__(self, message: str, **details: Any):
        super().__init__(STATUS_SUPPORT_PLANE_FAILED, message, details=details)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, str) and value.strip().upper() == "REQUIRED_REAL_VALUE":
        raise MissingCameraIntrinsicsError(
            f"{name} is still REQUIRED_REAL_VALUE", field=name
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MissingCameraIntrinsicsError(
            f"{name} must be a real number", field=name
        ) from error
    if not np.isfinite(result):
        raise MissingCameraIntrinsicsError(f"{name} must be finite", field=name)
    return result


@dataclass(frozen=True)
class CameraIntrinsics:
    """Validated pinhole intrinsics; no value is synthesized or guessed."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    view: str | None = None
    source: str | None = None
    factory_calibration: bool | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            width, height = int(self.width), int(self.height)
        except (TypeError, ValueError) as error:
            raise MissingCameraIntrinsicsError(
                "width and height must be integers"
            ) from error
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        if width <= 0 or height <= 0:
            raise MissingCameraIntrinsicsError("width and height must be positive")
        for name in ("fx", "fy", "cx", "cy"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.fx <= 0 or self.fy <= 0:
            raise MissingCameraIntrinsicsError("fx and fy must be positive")
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def validate_image_shape(self, shape: Sequence[int]) -> None:
        if len(shape) < 2:
            raise MissingCameraIntrinsicsError(
                f"image shape must have at least two axes: {shape}"
            )
        actual = (int(shape[0]), int(shape[1]))
        expected = (self.height, self.width)
        if actual != expected:
            raise MissingCameraIntrinsicsError(
                f"intrinsics/image resolution mismatch: calibration={expected}, image={actual}",
                calibration_shape=list(expected),
                image_shape=list(actual),
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "view": self.view,
            "source": self.source,
            "factory_calibration": self.factory_calibration,
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        view: str | None = None,
        image_shape: Sequence[int] | None = None,
    ) -> "CameraIntrinsics":
        required = ("width", "height", "fx", "fy", "cx", "cy")
        missing = [
            name for name in required if name not in values or values[name] is None
        ]
        if missing:
            raise MissingCameraIntrinsicsError(
                f"camera intrinsics missing required fields: {missing}",
                missing_fields=missing,
            )
        metadata = {key: value for key, value in values.items() if key not in required}
        intrinsics = cls(
            width=values["width"],
            height=values["height"],
            fx=values["fx"],
            fy=values["fy"],
            cx=values["cx"],
            cy=values["cy"],
            view=view
            or _none_or_str(
                values.get("view") or values.get("camera_view_from_sequence_path")
            ),
            source=_none_or_str(
                values.get("source") or values.get("intrinsics_source")
            ),
            factory_calibration=_none_or_bool(
                values.get(
                    "factory_calibration", values.get("factory_calibration_claimed")
                )
            ),
            metadata=metadata,
        )
        if image_shape is not None:
            intrinsics.validate_image_shape(image_shape)
        return intrinsics


def validate_intrinsics_image_shape(intrinsics: Any, shape: Sequence[int]) -> None:
    """Validate image size for this or the repository's existing intrinsics type."""

    if len(shape) < 2:
        raise MissingCameraIntrinsicsError(
            f"image shape must have at least two axes: {shape}"
        )
    actual = (int(shape[0]), int(shape[1]))
    try:
        expected = (int(intrinsics.height), int(intrinsics.width))
    except (AttributeError, TypeError, ValueError) as error:
        raise MissingCameraIntrinsicsError(
            "intrinsics object must expose integer height and width"
        ) from error
    if actual != expected:
        raise MissingCameraIntrinsicsError(
            f"intrinsics/image resolution mismatch: calibration={expected}, image={actual}",
            calibration_shape=list(expected),
            image_shape=list(actual),
        )


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _none_or_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise MissingCameraIntrinsicsError("factory_calibration metadata must be boolean")


def _load_json_or_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise MissingCameraIntrinsicsError(
            f"camera calibration file not found: {path}", path=str(path)
        )
    try:
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise MissingCameraIntrinsicsError(
                    "PyYAML is required to read YAML camera calibration", path=str(path)
                ) from error
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise MissingCameraIntrinsicsError(
                "camera calibration must be JSON or YAML", path=str(path)
            )
    except GeometryError:
        raise
    except Exception as error:
        raise MissingCameraIntrinsicsError(
            f"failed to parse camera calibration {path}: {error}", path=str(path)
        ) from error
    if not isinstance(loaded, Mapping):
        raise MissingCameraIntrinsicsError(
            "camera calibration root must be a mapping", path=str(path)
        )
    return loaded


def select_intrinsics_mapping(
    config: Mapping[str, Any], view: str | None = None
) -> tuple[Mapping[str, Any], str | None]:
    """Select direct, per-view, or ``default`` intrinsics without guessing a view."""

    required = {"width", "height", "fx", "fy", "cx", "cy"}
    # Any direct intrinsic field indicates a direct mapping, even if incomplete;
    # ``from_mapping`` will then report the exact missing fields.
    if required.intersection(config.keys()):
        return config, view
    if view is not None and view in config:
        selected = config[view]
        if not isinstance(selected, Mapping):
            raise MissingCameraIntrinsicsError(
                f"intrinsics entry for view {view!r} is not a mapping"
            )
        return selected, view
    if "default" in config:
        selected = config["default"]
        if not isinstance(selected, Mapping):
            raise MissingCameraIntrinsicsError(
                "default intrinsics entry is not a mapping"
            )
        return selected, view or "default"
    available = sorted(
        str(key) for key, value in config.items() if isinstance(value, Mapping)
    )
    if view is None:
        raise MissingCameraIntrinsicsError(
            "calibration contains per-view entries but no view/default was selected",
            available_views=available,
        )
    raise MissingCameraIntrinsicsError(
        f"no camera intrinsics for view {view!r} and no default entry",
        requested_view=view,
        available_views=available,
    )


def load_intrinsics_config(
    source: Path | str | Mapping[str, Any],
    *,
    view: str | None = None,
    image_shape: Sequence[int] | None = None,
) -> CameraIntrinsics:
    """Load direct/per-view JSON/YAML camera calibration with strict validation."""

    if isinstance(source, Mapping):
        config, source_path = source, None
    else:
        source_path = Path(source).expanduser().resolve()
        config = _load_json_or_yaml(source_path)
    selected, selected_view = select_intrinsics_mapping(config, view)
    selected_values = dict(selected)
    for provenance_key in (
        "source",
        "intrinsics_source",
        "factory_calibration",
        "factory_calibration_claimed",
    ):
        if provenance_key not in selected_values and provenance_key in config:
            selected_values[provenance_key] = config[provenance_key]
    intrinsics = CameraIntrinsics.from_mapping(
        selected_values, view=selected_view, image_shape=image_shape
    )
    if intrinsics.source is None and source_path is not None:
        # This is provenance, not a claim about how calibration was obtained.
        object.__setattr__(intrinsics, "source", str(source_path))
    return intrinsics


_UNIT_ALIASES = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
}


def _normalize_depth_unit(value: Any, *, allow_auto: bool = False) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if allow_auto and normalized == "auto":
        return "auto"
    if normalized not in _UNIT_ALIASES:
        raise AmbiguousDepthUnitError(
            f"unsupported depth unit {value!r}", supplied_unit=str(value)
        )
    return _UNIT_ALIASES[normalized]


@dataclass(frozen=True)
class DepthStatistics:
    shape: tuple[int, int]
    dtype: str
    nonfinite_ratio: float
    raw_positive_count: int
    raw_p1: float | None
    raw_p50: float | None
    raw_p99: float | None
    metric_positive_count: int
    metric_p1_m: float | None
    metric_p50_m: float | None
    metric_p99_m: float | None
    masked_area: int
    masked_valid_count: int
    masked_valid_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


@dataclass(frozen=True)
class DepthConversionResult:
    depth_m: np.ndarray
    source_unit: str
    depth_scale: float
    decision_reason: str
    stats: DepthStatistics

    def log_dict(self) -> dict[str, Any]:
        return {
            "depth_unit": self.source_unit,
            "depth_scale": self.depth_scale,
            "depth_unit_reason": self.decision_reason,
            **self.stats.to_dict(),
        }


def _quantiles(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    p1, p50, p99 = np.percentile(values.astype(np.float64), [1.0, 50.0, 99.0])
    return float(p1), float(p50), float(p99)


def _metadata_depth_units(metadata: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    if not metadata:
        return []
    found: list[tuple[str, str]] = []
    for key in ("depth_unit", "depth_units", "unit"):
        if key in metadata and metadata[key] is not None:
            found.append((key, _normalize_depth_unit(metadata[key]) or ""))
    nested = metadata.get("depth")
    if isinstance(nested, Mapping):
        for key in ("unit", "depth_unit"):
            if key in nested and nested[key] is not None:
                found.append((f"depth.{key}", _normalize_depth_unit(nested[key]) or ""))
    return found


def _sanity_candidates(
    raw_positive: np.ndarray, depth_scale: float
) -> tuple[bool, bool]:
    """Return plausibility as metres and millimetres for ordinary RGB-D ranges."""

    if raw_positive.size == 0:
        return False, False
    p1, p50, p99 = np.percentile(raw_positive.astype(np.float64), [1.0, 50.0, 99.0])
    plausible_m = 0.02 <= p50 <= 20.0 and 0.005 <= p1 and p99 <= 50.0
    mm = np.array([p1, p50, p99], dtype=np.float64) / depth_scale
    plausible_mm = 0.02 <= mm[1] <= 20.0 and 0.005 <= mm[0] and mm[2] <= 50.0
    return bool(plausible_m), bool(plausible_mm)


def resolve_depth_m(
    depth: np.ndarray,
    *,
    unit: str = "auto",
    depth_scale: float = 1000.0,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
    target_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    configured_unit: str | None = None,
) -> DepthConversionResult:
    """Convert explicitly sourced depth to metres and report the decision.

    ``auto`` consults manifest metadata and explicit configuration first, then
    requires the numeric sanity check to be unambiguous.  The dtype is never
    used as a unit signal.
    """

    source = np.asarray(depth)
    if source.ndim != 2:
        raise GeometryError(
            STATUS_INVALID_DEPTH, f"depth must be HxW, got {source.shape}"
        )
    requested = _normalize_depth_unit(unit, allow_auto=True)
    scale = float(depth_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise GeometryError(
            STATUS_INVALID_DEPTH, "depth_scale must be finite and positive"
        )
    minimum = float(min_depth_m)
    maximum = None if max_depth_m is None else float(max_depth_m)
    if not np.isfinite(minimum) or minimum < 0:
        raise GeometryError(
            STATUS_INVALID_DEPTH, "min_depth_m must be finite and non-negative"
        )
    if maximum is not None and (not np.isfinite(maximum) or maximum <= minimum):
        raise GeometryError(STATUS_INVALID_DEPTH, "max_depth_m must exceed min_depth_m")

    finite = np.isfinite(source)
    raw_positive = source[finite & (source > 0)]
    raw_q = _quantiles(raw_positive)
    plausible_m, plausible_mm = _sanity_candidates(raw_positive, scale)

    if requested != "auto":
        selected = requested
        reason = f"explicit --depth-unit={selected}"
    else:
        evidence = _metadata_depth_units(metadata)
        if (
            configured_unit is not None
            and str(configured_unit).strip().lower() != "auto"
        ):
            evidence.append(
                ("configured_unit", _normalize_depth_unit(configured_unit) or "")
            )
        evidence_units = {value for _, value in evidence}
        if len(evidence_units) > 1:
            raise AmbiguousDepthUnitError(
                "conflicting depth-unit metadata/configuration", evidence=evidence
            )
        selected = next(iter(evidence_units), None)
        if selected is not None:
            plausible = plausible_m if selected == "m" else plausible_mm
            if not plausible:
                raise AmbiguousDepthUnitError(
                    "declared depth unit conflicts with numeric range sanity check",
                    declared_unit=selected,
                    evidence=evidence,
                    raw_p1=raw_q[0],
                    raw_p50=raw_q[1],
                    raw_p99=raw_q[2],
                )
            reason = "metadata/config: " + ", ".join(
                f"{key}={value}" for key, value in evidence
            )
        elif plausible_m != plausible_mm:
            selected = "m" if plausible_m else "mm"
            reason = "unambiguous numeric range sanity check (no unit metadata)"
        else:
            raise AmbiguousDepthUnitError(
                "numeric range does not identify a unique depth unit",
                plausible_m=plausible_m,
                plausible_mm=plausible_mm,
                raw_p1=raw_q[0],
                raw_p50=raw_q[1],
                raw_p99=raw_q[2],
            )
    assert selected in {"m", "mm"}
    metric = source.astype(np.float32, copy=True)
    if selected == "mm":
        metric /= np.float32(scale)
    valid = np.isfinite(metric) & (metric > minimum)
    if maximum is not None:
        valid &= metric <= maximum
    metric[~valid] = np.float32(0.0)
    metric_values = metric[metric > 0]
    metric_q = _quantiles(metric_values)

    masked_area = 0
    masked_valid_count = 0
    masked_ratio: float | None = None
    if target_mask is not None:
        mask = _binary_mask_2d(target_mask)
        if mask.shape != source.shape:
            raise GeometryError(
                STATUS_MASK_DEPTH_SHAPE_ERROR,
                f"target mask/depth mismatch while computing depth stats: {mask.shape} vs {source.shape}",
            )
        masked_area = int(mask.sum())
        masked_valid_count = int(np.count_nonzero(mask & valid))
        masked_ratio = float(masked_valid_count / masked_area) if masked_area else 0.0
    stats = DepthStatistics(
        shape=(int(source.shape[0]), int(source.shape[1])),
        dtype=str(source.dtype),
        nonfinite_ratio=(
            float(1.0 - np.count_nonzero(finite) / source.size) if source.size else 1.0
        ),
        raw_positive_count=int(raw_positive.size),
        raw_p1=raw_q[0],
        raw_p50=raw_q[1],
        raw_p99=raw_q[2],
        metric_positive_count=int(metric_values.size),
        metric_p1_m=metric_q[0],
        metric_p50_m=metric_q[1],
        metric_p99_m=metric_q[2],
        masked_area=masked_area,
        masked_valid_count=masked_valid_count,
        masked_valid_ratio=masked_ratio,
    )
    if metric_values.size == 0:
        raise GeometryError(
            STATUS_INVALID_DEPTH,
            "depth has no valid metric samples",
            details=stats.to_dict(),
        )
    return DepthConversionResult(metric, selected, scale, reason, stats)


def _binary_mask_2d(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise MaskGeometryError(
            STATUS_MASK_DEPTH_SHAPE_ERROR,
            f"mask must be HxW or HxWx1, got {values.shape}",
        )
    if values.dtype == np.bool_:
        return values.copy()
    return np.isfinite(values) & (values >= float(threshold))


def resize_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize a binary mask using only nearest-neighbour sampling."""

    binary = _binary_mask_2d(mask)
    if len(target_shape) != 2:
        raise MaskGeometryError(
            STATUS_MASK_DEPTH_SHAPE_ERROR, "target shape must be (height, width)"
        )
    out_h, out_w = int(target_shape[0]), int(target_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise MaskGeometryError(
            STATUS_MASK_DEPTH_SHAPE_ERROR, "target shape must be positive"
        )
    if binary.shape == (out_h, out_w):
        return binary.copy()
    # Pixel-centre nearest-neighbour mapping, deterministic and dependency-free.
    row = np.minimum(
        np.floor(
            (np.arange(out_h, dtype=np.float64) + 0.5) * binary.shape[0] / out_h
        ).astype(int),
        binary.shape[0] - 1,
    )
    col = np.minimum(
        np.floor(
            (np.arange(out_w, dtype=np.float64) + 0.5) * binary.shape[1] / out_w
        ).astype(int),
        binary.shape[1] - 1,
    )
    return binary[row[:, None], col[None, :]].copy()


def _disk(radius_px: int) -> np.ndarray:
    radius = int(radius_px)
    if radius < 0:
        raise ValueError("morphology radius cannot be negative")
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for mask connected-component processing"
        ) from error
    return ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))


def dilate_mask(mask: np.ndarray, radius_px: int = 3) -> np.ndarray:
    binary = _binary_mask_2d(mask)
    radius = int(radius_px)
    if radius == 0:
        return binary
    try:
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError("SciPy is required for mask dilation") from error
    return np.asarray(
        ndimage.binary_dilation(binary, structure=_disk(radius)), dtype=bool
    )


@dataclass(frozen=True)
class MaskDiagnostics:
    original_shape: tuple[int, int]
    depth_shape: tuple[int, int]
    resized: bool
    resize_interpolation: str | None
    cleanup: str
    area_px: int
    area_fraction: float
    connected_component_count: int
    valid_depth_points: int
    valid_depth_ratio: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["original_shape"] = list(self.original_shape)
        result["depth_shape"] = list(self.depth_shape)
        return result


@dataclass(frozen=True)
class MaskResult:
    raw_resized_mask: np.ndarray
    mask: np.ndarray
    diagnostics: MaskDiagnostics


def prepare_target_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    cleanup: str = "none",
    threshold: float = 0.5,
    min_area_px: int = 25,
    max_area_fraction: float = 0.80,
    min_valid_depth_points: int = 20,
    close_radius_px: int = 2,
) -> MaskResult:
    """Align and validate a predicted target mask without masking scene depth."""

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise MaskGeometryError(
            STATUS_MASK_DEPTH_SHAPE_ERROR, f"depth must be HxW, got {depth.shape}"
        )
    try:
        original = _binary_mask_2d(mask, threshold)
        resized = resize_mask_nearest(original, depth.shape)
    except GeometryError:
        raise
    except Exception as error:
        raise MaskGeometryError(STATUS_MASK_DEPTH_SHAPE_ERROR, str(error)) from error
    cleanup = str(cleanup).strip().lower()
    if cleanup not in {"none", "largest-component", "close"}:
        raise ValueError(f"unsupported mask cleanup: {cleanup}")
    processed = resized.copy()
    if cleanup == "largest-component" and processed.any():
        labels, count = _label_components(processed)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        processed = labels == int(np.argmax(sizes))
    elif cleanup == "close" and processed.any():
        try:
            from scipy import ndimage
        except ImportError as error:
            raise RuntimeError("SciPy is required for mask closing") from error
        processed = np.asarray(
            ndimage.binary_closing(processed, structure=_disk(close_radius_px)),
            dtype=bool,
        )
    _, component_count = _label_components(processed)
    area = int(processed.sum())
    area_fraction = float(area / processed.size) if processed.size else 0.0
    valid_depth = np.isfinite(depth) & (depth > 0)
    valid_count = int(np.count_nonzero(processed & valid_depth))
    valid_ratio = float(valid_count / area) if area else 0.0
    diagnostics = MaskDiagnostics(
        original_shape=(int(original.shape[0]), int(original.shape[1])),
        depth_shape=(int(depth.shape[0]), int(depth.shape[1])),
        resized=original.shape != depth.shape,
        resize_interpolation="nearest" if original.shape != depth.shape else None,
        cleanup=cleanup,
        area_px=area,
        area_fraction=area_fraction,
        connected_component_count=int(component_count),
        valid_depth_points=valid_count,
        valid_depth_ratio=valid_ratio,
    )
    details = diagnostics.to_dict()
    if area == 0:
        raise MaskGeometryError(
            STATUS_EMPTY_MASK, "predicted target mask is empty", details=details
        )
    if area < int(min_area_px):
        raise MaskGeometryError(
            STATUS_MASK_TOO_SMALL,
            f"target mask has {area} pixels; minimum is {int(min_area_px)}",
            details=details,
        )
    if area_fraction > float(max_area_fraction):
        raise MaskGeometryError(
            STATUS_MASK_TOO_LARGE,
            f"target mask covers {area_fraction:.3%}; maximum is {float(max_area_fraction):.3%}",
            details=details,
        )
    if valid_count < int(min_valid_depth_points):
        raise MaskGeometryError(
            STATUS_INSUFFICIENT_MASKED_DEPTH,
            f"target mask has {valid_count} valid depth points; minimum is {int(min_valid_depth_points)}",
            details=details,
        )
    return MaskResult(resized, processed, diagnostics)


def backproject_depth(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project finite positive pixels into the camera optical frame."""

    depth = np.asarray(depth_m, dtype=np.float64)
    validate_intrinsics_image_shape(intrinsics, depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        binary = _binary_mask_2d(mask)
        if binary.shape != depth.shape:
            raise MaskGeometryError(
                STATUS_MASK_DEPTH_SHAPE_ERROR,
                f"mask/depth mismatch: {binary.shape} vs {depth.shape}",
            )
        valid &= binary
    vv, uu = np.nonzero(valid)
    z = depth[vv, uu]
    x = (uu.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (vv.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z)).astype(np.float64, copy=False)


@dataclass(frozen=True)
class TargetCloud:
    points_camera_m: np.ndarray
    raw_points_camera_m: np.ndarray
    median_camera_m: np.ndarray
    mad_camera_m: np.ndarray
    centroid_camera_m: np.ndarray
    aabb_min_camera_m: np.ndarray
    aabb_max_camera_m: np.ndarray
    raw_point_count: int
    point_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_camera_m": self.median_camera_m.tolist(),
            "mad_camera_m": self.mad_camera_m.tolist(),
            "centroid_camera_m": self.centroid_camera_m.tolist(),
            "aabb_min_camera_m": self.aabb_min_camera_m.tolist(),
            "aabb_max_camera_m": self.aabb_max_camera_m.tolist(),
            "raw_point_count": self.raw_point_count,
            "point_count": self.point_count,
        }


def robust_target_cloud(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mad_threshold: float = 4.5,
    min_points: int = 20,
) -> TargetCloud:
    """Back-project target depth, then reject coordinate-wise median/MAD outliers."""

    raw = backproject_depth(depth_m, intrinsics, mask=target_mask)
    if raw.shape[0] < int(min_points):
        raise MaskGeometryError(
            STATUS_INSUFFICIENT_MASKED_DEPTH,
            f"target point cloud has {raw.shape[0]} points; minimum is {int(min_points)}",
        )
    median = np.median(raw, axis=0)
    mad = np.median(np.abs(raw - median), axis=0)
    scale = 1.4826 * mad
    # A constant coordinate provides no evidence for rejecting a point.
    robust_deviation = np.zeros_like(raw)
    variable = scale > np.finfo(np.float64).eps
    robust_deviation[:, variable] = (
        np.abs(raw[:, variable] - median[variable]) / scale[variable]
    )
    keep = np.all(robust_deviation <= float(mad_threshold), axis=1)
    filtered = raw[keep]
    if filtered.shape[0] < int(min_points):
        raise MaskGeometryError(
            STATUS_INSUFFICIENT_MASKED_DEPTH,
            f"only {filtered.shape[0]} target points remain after MAD filtering",
            details={
                "raw_point_count": int(raw.shape[0]),
                "mad_threshold": float(mad_threshold),
            },
        )
    return TargetCloud(
        points_camera_m=filtered,
        raw_points_camera_m=raw,
        median_camera_m=median,
        mad_camera_m=mad,
        centroid_camera_m=np.mean(filtered, axis=0),
        aabb_min_camera_m=np.min(filtered, axis=0),
        aabb_max_camera_m=np.max(filtered, axis=0),
        raw_point_count=int(raw.shape[0]),
        point_count=int(filtered.shape[0]),
    )


@dataclass(frozen=True)
class SupportPlane:
    equation_camera: np.ndarray
    inlier_count: int
    candidate_point_count: int
    residual_rmse_m: float
    candidate_source: str
    distance_threshold_m: float
    seed: int

    @property
    def normal_camera(self) -> np.ndarray:
        return self.equation_camera[:3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_equation_camera": self.equation_camera.tolist(),
            "inlier_count": self.inlier_count,
            "candidate_point_count": self.candidate_point_count,
            "residual_rmse_m": self.residual_rmse_m,
            "candidate_source": self.candidate_source,
            "distance_threshold_m": self.distance_threshold_m,
            "seed": self.seed,
        }


def orient_plane_normal_toward_camera(equation: Sequence[float]) -> np.ndarray:
    plane = np.asarray(equation, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise SupportPlaneError("plane equation must contain four finite values")
    norm = float(np.linalg.norm(plane[:3]))
    if norm <= np.finfo(np.float64).eps:
        raise SupportPlaneError("plane normal is degenerate")
    plane = plane / norm
    closest = -plane[3] * plane[:3]
    if np.dot(plane[:3], -closest) < 0.0:
        plane = -plane
    return plane


def estimate_support_plane(
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    target_cloud: TargetCloud,
    *,
    local_radius_m: float = 0.25,
    target_exclusion_dilation_px: int = 5,
    distance_threshold_m: float = 0.008,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    min_candidate_points: int = 100,
    min_inliers: int = 80,
    seed: int = 42,
) -> SupportPlane:
    """Estimate a local support plane with deterministic Open3D RANSAC.

    Nearby scene points outside a dilated target mask are preferred.  If that
    set is too small, all valid off-target scene points are used and the source
    is recorded explicitly.
    """

    try:
        import open3d as o3d
    except ImportError as error:
        raise SupportPlaneError(
            "Open3D is required for support-plane RANSAC", dependency="open3d"
        ) from error
    depth = np.asarray(depth_m, dtype=np.float64)
    validate_intrinsics_image_shape(intrinsics, depth.shape)
    mask = _binary_mask_2d(target_mask)
    if mask.shape != depth.shape:
        raise MaskGeometryError(
            STATUS_MASK_DEPTH_SHAPE_ERROR,
            f"mask/depth mismatch: {mask.shape} vs {depth.shape}",
        )
    exclusion = dilate_mask(mask, target_exclusion_dilation_px)
    valid = np.isfinite(depth) & (depth > 0) & ~exclusion
    all_off_target = backproject_depth(depth, intrinsics, mask=valid)
    if all_off_target.shape[0] < int(min_candidate_points):
        raise SupportPlaneError(
            "not enough valid off-target scene points for plane estimation",
            candidate_count=int(all_off_target.shape[0]),
        )
    distances = np.linalg.norm(all_off_target - target_cloud.centroid_camera_m, axis=1)
    local = all_off_target[distances <= float(local_radius_m)]
    if local.shape[0] >= int(min_candidate_points):
        candidates, candidate_source = local, "near_target_off_mask"
    else:
        candidates, candidate_source = all_off_target, "all_scene_off_mask"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(candidates)
    try:
        o3d.utility.random.seed(int(seed))
    except AttributeError:
        pass
    try:
        equation, inlier_indices = pcd.segment_plane(
            distance_threshold=float(distance_threshold_m),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
            probability=1.0,
        )
    except TypeError:
        # Compatibility with older Open3D versions lacking ``probability``.
        equation, inlier_indices = pcd.segment_plane(
            distance_threshold=float(distance_threshold_m),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
    if len(inlier_indices) < int(min_inliers):
        raise SupportPlaneError(
            f"support plane has only {len(inlier_indices)} inliers; minimum is {int(min_inliers)}",
            candidate_source=candidate_source,
            candidate_count=int(candidates.shape[0]),
        )
    plane = orient_plane_normal_toward_camera(equation)
    inliers = candidates[np.asarray(inlier_indices, dtype=np.int64)]
    residuals = inliers @ plane[:3] + plane[3]
    return SupportPlane(
        equation_camera=plane,
        inlier_count=int(inliers.shape[0]),
        candidate_point_count=int(candidates.shape[0]),
        residual_rmse_m=float(np.sqrt(np.mean(np.square(residuals)))),
        candidate_source=candidate_source,
        distance_threshold_m=float(distance_threshold_m),
        seed=int(seed),
    )


@dataclass(frozen=True)
class TaskFrame:
    T_camera_task: np.ndarray
    target_centroid_camera_m: np.ndarray
    target_projection_camera_m: np.ndarray
    workspace_size_m: float
    table_height_m: float
    non_official_geometry_fallback: bool
    camera_axis_used_for_x: str

    @property
    def T_task_camera(self) -> np.ndarray:
        return invert_transform(self.T_camera_task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_semantics": "T_camera_task maps task-frame coordinates into camera frame",
            "T_camera_task": self.T_camera_task.tolist(),
            "T_task_camera": self.T_task_camera.tolist(),
            "target_centroid_camera_m": self.target_centroid_camera_m.tolist(),
            "target_projection_camera_m": self.target_projection_camera_m.tolist(),
            "workspace_size_m": self.workspace_size_m,
            "table_height_m": self.table_height_m,
            "non_official_geometry_fallback": self.non_official_geometry_fallback,
            "camera_axis_used_for_x": self.camera_axis_used_for_x,
        }


def build_task_frame(
    target_centroid_camera_m: Sequence[float],
    support_plane: SupportPlane | Sequence[float] | None,
    *,
    workspace_size_m: float = 0.30,
    table_height_m: float = 0.05,
    allow_camera_aligned_fallback: bool = False,
) -> TaskFrame:
    """Construct the required right-handed z-up ``T_camera_task`` transform."""

    centroid = np.asarray(target_centroid_camera_m, dtype=np.float64)
    if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
        raise SupportPlaneError(
            "target centroid must contain three finite camera coordinates"
        )
    size, table_height = float(workspace_size_m), float(table_height_m)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("workspace_size_m must be finite and positive")
    if not np.isfinite(table_height) or not 0 <= table_height < size:
        raise ValueError("table_height_m must be in [0, workspace_size_m)")
    fallback = support_plane is None
    if fallback and not allow_camera_aligned_fallback:
        raise SupportPlaneError(
            "support plane unavailable and camera-aligned fallback is disabled"
        )
    if fallback:
        plane = None
        z_task = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        projection = centroid.copy()
    else:
        raw_equation = (
            support_plane.equation_camera
            if isinstance(support_plane, SupportPlane)
            else support_plane
        )
        plane = orient_plane_normal_toward_camera(raw_equation)  # type: ignore[arg-type]
        z_task = plane[:3]
        projection = centroid - (float(np.dot(z_task, centroid)) + plane[3]) * z_task
    camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_projected = camera_x - float(np.dot(camera_x, z_task)) * z_task
    axis_used = "camera_x"
    if np.linalg.norm(x_projected) < 1e-8:
        camera_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_projected = camera_y - float(np.dot(camera_y, z_task)) * z_task
        axis_used = "camera_y_degenerate_camera_x_fallback"
    x_task = x_projected / np.linalg.norm(x_projected)
    y_task = np.cross(z_task, x_task)
    y_task /= np.linalg.norm(y_task)
    # Recompute x to suppress accumulated floating-point non-orthogonality.
    x_task = np.cross(y_task, z_task)
    x_task /= np.linalg.norm(x_task)
    rotation = np.column_stack((x_task, y_task, z_task))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise SupportPlaneError("constructed task rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
        raise SupportPlaneError("constructed task rotation is not right-handed")
    desired_projection_task = np.array([size / 2.0, size / 2.0, table_height])
    translation = projection - rotation @ desired_projection_task
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return TaskFrame(
        T_camera_task=transform,
        target_centroid_camera_m=centroid,
        target_projection_camera_m=projection,
        workspace_size_m=size,
        table_height_m=table_height,
        non_official_geometry_fallback=fallback,
        camera_axis_used_for_x=axis_used,
    )


def validate_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("transform last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-7
    ):
        raise ValueError("transform rotation must be right-handed and orthonormal")
    return matrix


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    values = np.asarray(points, dtype=np.float64)
    single = values.shape == (3,)
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("points must be finite with shape (N, 3) or (3,)")
    transformed = values @ matrix[:3, :3].T + matrix[:3, 3]
    return transformed[0] if single else transformed


@dataclass(frozen=True)
class ProjectionResult:
    uv: np.ndarray
    positive_depth: np.ndarray
    inside_image: np.ndarray


def project_camera_points(
    points_camera_m: np.ndarray, intrinsics: CameraIntrinsics
) -> ProjectionResult:
    points = np.asarray(points_camera_m, dtype=np.float64)
    single = points.shape == (3,)
    if single:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_m must have shape (N, 3) or (3,)")
    positive = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    uv[positive, 0] = (
        intrinsics.fx * points[positive, 0] / points[positive, 2] + intrinsics.cx
    )
    uv[positive, 1] = (
        intrinsics.fy * points[positive, 1] / points[positive, 2] + intrinsics.cy
    )
    inside = (
        positive
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < intrinsics.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < intrinsics.height)
    )
    if single:
        return ProjectionResult(uv[0], positive[0:1], inside[0:1])
    return ProjectionResult(uv, positive, inside)


def project_camera_point(
    point_camera_m: Sequence[float], intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, bool]:
    """Project one camera-frame point and return ``(uv, inside_image)``."""

    result = project_camera_points(
        np.asarray(point_camera_m, dtype=np.float64), intrinsics
    )
    return result.uv, bool(result.inside_image[0])


def nearest_target_point_distances(
    query_points_camera_m: np.ndarray,
    target_points_camera_m: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    query = np.asarray(query_points_camera_m, dtype=np.float64)
    target = np.asarray(target_points_camera_m, dtype=np.float64)
    single = query.shape == (3,)
    if single:
        query = query[None, :]
    if (
        query.ndim != 2
        or query.shape[1] != 3
        or target.ndim != 2
        or target.shape[1] != 3
    ):
        raise ValueError("query and target points must have shape (N, 3)")
    if target.shape[0] == 0:
        raise ValueError("target point cloud cannot be empty")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(target)):
        raise ValueError("query and target points must be finite")
    result = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], int(chunk_size)):
        block = query[start : start + int(chunk_size)]
        squared = np.sum(np.square(block[:, None, :] - target[None, :, :]), axis=2)
        result[start : start + block.shape[0]] = np.sqrt(np.min(squared, axis=1))
    return result[0:1] if single else result


def nearest_target_point_distance(
    query_point_camera_m: Sequence[float], target_points_camera_m: np.ndarray
) -> float:
    """Scalar convenience wrapper for one candidate centre."""

    return float(
        nearest_target_point_distances(
            np.asarray(query_point_camera_m, dtype=np.float64), target_points_camera_m
        )[0]
    )


def projected_depth_difference_m(
    point_camera_m: Sequence[float], depth_m: np.ndarray, intrinsics: CameraIntrinsics
) -> float | None:
    point = np.asarray(point_camera_m, dtype=np.float64)
    projection = project_camera_points(point, intrinsics)
    if not bool(projection.inside_image[0]):
        return None
    u, v = np.rint(projection.uv).astype(int)
    # Rounding can reach width/height for a sub-pixel point just inside the image.
    u = min(max(int(u), 0), intrinsics.width - 1)
    v = min(max(int(v), 0), intrinsics.height - 1)
    observed = float(np.asarray(depth_m)[v, u])
    if not np.isfinite(observed) or observed <= 0:
        return None
    return float(point[2] - observed)


def plane_z_in_task(
    plane_equation_camera: Sequence[float],
    T_camera_task: np.ndarray,
    points_camera: np.ndarray,
) -> np.ndarray:
    """Diagnostic helper returning task-z for supplied points on a camera plane."""

    plane = orient_plane_normal_toward_camera(plane_equation_camera)
    points = np.asarray(points_camera, dtype=np.float64)
    residual = points @ plane[:3] + plane[3]
    if not np.allclose(residual, 0.0, atol=1e-5):
        raise ValueError("points_camera are not on the supplied plane")
    return transform_points(invert_transform(T_camera_task), points)[:, 2]
