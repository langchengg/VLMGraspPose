"""GT-free visual inputs for one-request-per-sample local VLM re-ranking."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.grasping.camera_geometry import depth_mm_to_meters
from src.grasping.mask_processing import process_mask_with_diagnostics


NEUTRAL_CANDIDATE_COLORS = (
    (58, 134, 255),
    (255, 159, 64),
    (175, 82, 222),
    (48, 176, 199),
    (255, 99, 132),
)
VLM_VISUALIZATION_RECIPE_SCHEMA_VERSION = 1
VLM_VISUALIZATION_RENDERER = "reranking_v1_gt_free_visualizer"
_RECIPE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VLMVisualizationResult:
    full_scene_overlay_path: Path
    candidate_contact_sheet_path: Path
    manifest_path: Path
    depth_normalization: Mapping[str, Any]


def canonical_recipe_sha256(value: Mapping[str, Any]) -> str:
    """Hash a visualization recipe independently of JSON whitespace."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _load_rgb(value: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(value, (str, Path)):
        array = np.asarray(Image.open(value).convert("RGB"))
    else:
        array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"RGB must have shape HxWx3/4, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _load_2d(
    value: np.ndarray | str | Path, *, dtype: np.dtype[Any]
) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            array = np.asarray(Image.open(path))
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"input must be two-dimensional, got {array.shape}")
    return array.astype(dtype)


def _candidate_geometry(candidate: Mapping[str, Any]) -> tuple[np.ndarray, float, float, float]:
    center_value = candidate.get(
        "center_uv",
        [candidate.get("center_u_px"), candidate.get("center_v_px")],
    )
    center = np.asarray(center_value, dtype=np.float64)
    angle = float(candidate["angle_rad"])
    width = float(candidate["width_px"])
    height = float(
        candidate.get("rectangle_height_px", candidate.get("height_px", 20.0))
    )
    if (
        center.shape != (2,)
        or not np.all(np.isfinite(center))
        or not all(math.isfinite(value) for value in (angle, width, height))
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"candidate {candidate.get('candidate_id')} has invalid geometry")
    return center, angle, width, height


def _render_candidate_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Project a scored candidate onto the minimal GT-free drawing contract."""

    candidate_id = str(candidate.get("candidate_id", ""))
    if not candidate_id:
        raise ValueError("every visualization candidate requires a candidate_id")
    center, angle, width, height = _candidate_geometry(candidate)
    return {
        "candidate_id": candidate_id,
        "center_uv": [float(center[0]), float(center[1])],
        "angle_rad": float(angle),
        "width_px": float(width),
        "rectangle_height_px": float(height),
    }


def _normalized_mask_processing(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "mask_threshold",
        "min_component_area_px",
        "retain_largest_component",
        "mask_erode_px",
        "mask_dilate_px",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"visualization mask-processing recipe missing: {missing}")
    integer_fields = (
        "min_component_area_px",
        "mask_erode_px",
        "mask_dilate_px",
    )
    if (
        isinstance(value["mask_threshold"], bool)
        or not isinstance(value["mask_threshold"], numbers.Real)
        or not isinstance(value["retain_largest_component"], bool)
        or any(
            isinstance(value[key], bool)
            or not isinstance(value[key], numbers.Integral)
            for key in integer_fields
        )
    ):
        raise ValueError("visualization mask-processing types are invalid")
    normalized = {
        "mask_threshold": float(value["mask_threshold"]),
        "min_component_area_px": int(value["min_component_area_px"]),
        "retain_largest_component": bool(value["retain_largest_component"]),
        "mask_erode_px": int(value["mask_erode_px"]),
        "mask_dilate_px": int(value["mask_dilate_px"]),
    }
    if (
        not math.isfinite(normalized["mask_threshold"])
        or normalized["min_component_area_px"] < 0
        or normalized["mask_erode_px"] < 0
        or normalized["mask_dilate_px"] < 0
    ):
        raise ValueError("visualization mask-processing recipe is invalid")
    return normalized


def build_vlm_visualization_recipe(
    *,
    sample_id: str,
    rgb_path: str | Path,
    depth_mm_path: str | Path,
    predicted_mask_path: str | Path,
    candidates_q_order: Sequence[Mapping[str, Any]],
    mask_processing: Mapping[str, Any],
    crop_size: int = 160,
    mask_alpha: float = 0.30,
) -> dict[str, Any]:
    """Create a compact, GT-free recipe without rendering persistent PNGs."""

    if not sample_id:
        raise ValueError("visualization recipe sample_id must be non-empty")
    if crop_size < 64:
        raise ValueError("crop_size must be at least 64 pixels")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("mask_alpha must be in [0,1]")
    paths = {
        "rgb": Path(rgb_path).expanduser().resolve(),
        "depth_mm": Path(depth_mm_path).expanduser().resolve(),
        "predicted_hifi_mask": Path(predicted_mask_path).expanduser().resolve(),
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"visualization recipe source files missing: {missing_paths}"
        )
    candidates = [
        _render_candidate_record(candidate) for candidate in candidates_q_order
    ]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("visualization recipe candidate IDs must be non-empty and unique")
    candidate_hash = hashlib.sha256(
        json.dumps(
            candidates,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    recipe = {
        "schema_version": VLM_VISUALIZATION_RECIPE_SCHEMA_VERSION,
        "recipe_kind": "gt_free_vlm_visualization",
        "renderer": VLM_VISUALIZATION_RENDERER,
        "sample_id": sample_id,
        "gt_free": True,
        "candidate_ids": candidate_ids,
        "candidate_records": candidates,
        "candidate_records_sha256": candidate_hash,
        "candidate_order": (
            "full-precision GQ-CNN descending; candidate_id tie-break"
        ),
        "inputs": {
            "rgb": {
                "path": str(paths["rgb"]),
                "sha256": _sha256_file(paths["rgb"]),
                "representation": "RGB image",
            },
            "depth_mm": {
                "path": str(paths["depth_mm"]),
                "sha256": _sha256_file(paths["depth_mm"]),
                "representation": (
                    "OCID uint16 millimetres; float32 metres = value * 0.001"
                ),
            },
            "predicted_hifi_mask": {
                "path": str(paths["predicted_hifi_mask"]),
                "sha256": _sha256_file(paths["predicted_hifi_mask"]),
                "representation": "HiFi predicted native-resolution mask",
            },
        },
        "mask_processing": _normalized_mask_processing(mask_processing),
        "render_config": {
            "crop_size_px": int(crop_size),
            "mask_alpha": float(mask_alpha),
        },
        "allowed_modalities": [
            "rgb",
            "metric_depth_derived_from_depth_mm",
            "predicted_hifi_mask",
            "frozen_candidate_pose",
        ],
        "forbidden_inputs_absent": [
            "ground-truth mask",
            "ground-truth grasp",
            "candidate correctness",
            "evaluation metrics",
        ],
    }
    validate_vlm_visualization_recipe(recipe, verify_sources=True)
    return recipe


def validate_vlm_visualization_recipe(
    recipe: Mapping[str, Any], *, verify_sources: bool
) -> dict[str, Any]:
    """Fail closed on recipe semantics and optionally re-hash source files."""

    expected_recipe_keys = {
        "schema_version",
        "recipe_kind",
        "renderer",
        "sample_id",
        "gt_free",
        "candidate_ids",
        "candidate_records",
        "candidate_records_sha256",
        "candidate_order",
        "inputs",
        "mask_processing",
        "render_config",
        "allowed_modalities",
        "forbidden_inputs_absent",
    }
    if (
        set(recipe) != expected_recipe_keys
        or recipe.get("schema_version")
        != VLM_VISUALIZATION_RECIPE_SCHEMA_VERSION
        or recipe.get("recipe_kind") != "gt_free_vlm_visualization"
        or recipe.get("renderer") != VLM_VISUALIZATION_RENDERER
        or recipe.get("gt_free") is not True
        or not str(recipe.get("sample_id", ""))
        or recipe.get("candidate_order")
        != "full-precision GQ-CNN descending; candidate_id tie-break"
        or recipe.get("allowed_modalities")
        != [
            "rgb",
            "metric_depth_derived_from_depth_mm",
            "predicted_hifi_mask",
            "frozen_candidate_pose",
        ]
        or recipe.get("forbidden_inputs_absent")
        != [
            "ground-truth mask",
            "ground-truth grasp",
            "candidate correctness",
            "evaluation metrics",
        ]
    ):
        raise ValueError("unsupported or non-GT-free VLM visualization recipe")
    candidates_raw = recipe.get("candidate_records")
    candidate_ids_raw = recipe.get("candidate_ids")
    if (
        not isinstance(candidates_raw, Sequence)
        or isinstance(candidates_raw, (str, bytes, bytearray))
        or not isinstance(candidate_ids_raw, Sequence)
        or isinstance(candidate_ids_raw, (str, bytes, bytearray))
    ):
        raise ValueError("visualization recipe candidates are malformed")
    expected_candidate_keys = {
        "candidate_id",
        "center_uv",
        "angle_rad",
        "width_px",
        "rectangle_height_px",
    }
    if any(
        not isinstance(candidate, Mapping)
        or set(candidate) != expected_candidate_keys
        or not isinstance(candidate["candidate_id"], str)
        or not isinstance(candidate["center_uv"], list)
        or len(candidate["center_uv"]) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            for value in (
                *candidate["center_uv"],
                candidate["angle_rad"],
                candidate["width_px"],
                candidate["rectangle_height_px"],
            )
        )
        for candidate in candidates_raw
    ):
        raise ValueError("visualization recipe candidate schema is invalid")
    candidates = [
        _render_candidate_record(candidate) for candidate in candidates_raw
    ]
    candidate_ids = [str(value) for value in candidate_ids_raw]
    if (
        not candidate_ids
        or candidate_ids != [item["candidate_id"] for item in candidates]
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("visualization recipe candidate identity is invalid")
    expected_candidate_hash = hashlib.sha256(
        json.dumps(
            candidates,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recipe.get("candidate_records_sha256") != expected_candidate_hash:
        raise ValueError("visualization recipe candidate records changed")
    inputs = recipe.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "rgb",
        "depth_mm",
        "predicted_hifi_mask",
    }:
        raise ValueError("visualization recipe input schema is invalid")
    expected_representations = {
        "rgb": "RGB image",
        "depth_mm": (
            "OCID uint16 millimetres; float32 metres = value * 0.001"
        ),
        "predicted_hifi_mask": "HiFi predicted native-resolution mask",
    }
    resolved_inputs: dict[str, Path] = {}
    for role, representation in expected_representations.items():
        item = inputs[role]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256", "representation"}
            or item.get("representation") != representation
            or _RECIPE_SHA256_PATTERN.fullmatch(str(item.get("sha256", "")))
            is None
        ):
            raise ValueError(f"visualization recipe {role} provenance is invalid")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if verify_sources and (
            not path.is_file() or _sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"visualization recipe {role} source changed")
        resolved_inputs[role] = path
    mask_processing = _normalized_mask_processing(
        recipe.get("mask_processing", {})
    )
    if set(recipe.get("mask_processing", {})) != {
        "mask_threshold",
        "min_component_area_px",
        "retain_largest_component",
        "mask_erode_px",
        "mask_dilate_px",
    }:
        raise ValueError("visualization mask-processing schema is invalid")
    render_config = recipe.get("render_config")
    if (
        not isinstance(render_config, Mapping)
        or set(render_config) != {"crop_size_px", "mask_alpha"}
        or isinstance(render_config.get("crop_size_px"), bool)
        or not isinstance(
            render_config.get("crop_size_px"), numbers.Integral
        )
        or isinstance(render_config.get("mask_alpha"), bool)
        or not isinstance(
            render_config.get("mask_alpha"), numbers.Real
        )
    ):
        raise ValueError("visualization recipe render config is missing")
    crop_size = int(render_config.get("crop_size_px", -1))
    mask_alpha = float(render_config.get("mask_alpha", float("nan")))
    if crop_size < 64 or not math.isfinite(mask_alpha) or not 0 <= mask_alpha <= 1:
        raise ValueError("visualization recipe render config is invalid")
    return {
        "sample_id": str(recipe["sample_id"]),
        "candidate_ids": candidate_ids,
        "candidate_records": candidates,
        "candidate_records_sha256": expected_candidate_hash,
        "inputs": resolved_inputs,
        "mask_processing": mask_processing,
        "crop_size_px": crop_size,
        "mask_alpha": mask_alpha,
        "recipe_sha256": canonical_recipe_sha256(recipe),
    }


def render_vlm_visualization_recipe(
    recipe: Mapping[str, Any], *, output_dir: str | Path
) -> VLMVisualizationResult:
    """Materialize one recipe into a caller-owned temporary directory."""

    validated = validate_vlm_visualization_recipe(
        recipe, verify_sources=True
    )
    with Image.open(validated["inputs"]["depth_mm"]) as image:
        depth_mm = np.asarray(image)
    depth_m = depth_mm_to_meters(depth_mm)
    with Image.open(validated["inputs"]["predicted_hifi_mask"]) as image:
        mask_input = np.asarray(image)
    processing = validated["mask_processing"]
    predicted_mask = process_mask_with_diagnostics(
        mask_input,
        depth_m,
        threshold=processing["mask_threshold"],
        min_component_size_px=processing["min_component_area_px"],
        keep_largest_component=processing["retain_largest_component"],
        erode_radius_px=processing["mask_erode_px"],
        dilate_radius_px=processing["mask_dilate_px"],
    ).processed
    return build_vlm_visualizations(
        sample_id=validated["sample_id"],
        rgb=validated["inputs"]["rgb"],
        depth_m=depth_m,
        predicted_mask=predicted_mask,
        candidates_q_order=validated["candidate_records"],
        output_dir=output_dir,
        crop_size=validated["crop_size_px"],
        mask_alpha=validated["mask_alpha"],
        source_recipe_sha256=validated["recipe_sha256"],
    )


def _rectangle_points(candidate: Mapping[str, Any]) -> list[tuple[float, float]]:
    center, angle, width, height = _candidate_geometry(candidate)
    axis = np.asarray([math.cos(angle), math.sin(angle)])
    normal = np.asarray([-axis[1], axis[0]])
    return [
        tuple(center + width * 0.5 * sx * axis + height * 0.5 * sy * normal)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def _contact_points(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    center, angle, width, _ = _candidate_geometry(candidate)
    axis = np.asarray([math.cos(angle), math.sin(angle)])
    return center - width * 0.5 * axis, center + width * 0.5 * axis


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _draw_candidate(
    draw: ImageDraw.ImageDraw,
    candidate: Mapping[str, Any],
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    color: tuple[int, int, int] = (50, 145, 220),
    width: int = 3,
    draw_contacts: bool = True,
) -> None:
    ox, oy = offset

    def transform(point: tuple[float, float] | np.ndarray) -> tuple[float, float]:
        return (
            (float(point[0]) - ox) * scale,
            (float(point[1]) - oy) * scale,
        )

    polygon = [transform(point) for point in _rectangle_points(candidate)]
    draw.line(polygon + [polygon[0]], fill=color, width=width, joint="curve")
    if draw_contacts:
        for point in _contact_points(candidate):
            x, y = transform(point)
            radius = max(3, width + 1)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color,
                width=max(2, width - 1),
            )


def _crop_bounds(
    center: np.ndarray, size: int, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    half = size // 2
    left = int(round(float(center[0]))) - half
    top = int(round(float(center[1]))) - half
    return left, top, left + size, top + size


def _crop_with_padding(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    *,
    fill: tuple[int, ...] | int,
) -> Image.Image:
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    output = Image.new(image.mode, (width, height), fill)
    source = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source[2] > source[0] and source[3] > source[1]:
        output.paste(
            image.crop(source), (source[0] - left, source[1] - top)
        )
    return output


def _depth_color(
    depth: np.ndarray, valid: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    normalized = np.clip(
        (np.nan_to_num(depth, nan=minimum) - minimum) / max(maximum - minimum, 1e-9),
        0.0,
        1.0,
    )
    # Fixed blue->cyan->yellow metric-depth palette, shared by all candidates.
    red = np.clip(2.0 * normalized - 0.2, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * np.abs(normalized - 0.5), 0.0, 1.0)
    blue = np.clip(1.2 - 2.0 * normalized, 0.0, 1.0)
    color = np.stack([red, green, blue], axis=-1)
    color[~valid] = 0.08
    return np.round(color * 255.0).astype(np.uint8)


def _atomic_save_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{time.time_ns()}.tmp.png"
    )
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, destination)


def _atomic_write_json(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def build_vlm_visualizations(
    *,
    sample_id: str,
    rgb: np.ndarray | str | Path,
    depth_m: np.ndarray | str | Path,
    predicted_mask: np.ndarray | str | Path,
    candidates_q_order: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    crop_size: int = 160,
    mask_alpha: float = 0.30,
    source_recipe_sha256: str | None = None,
) -> VLMVisualizationResult:
    """Build the two candidate-complete, GT-free VLM images and provenance."""

    if crop_size < 64:
        raise ValueError("crop_size must be at least 64 pixels")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("mask_alpha must be in [0,1]")
    if (
        source_recipe_sha256 is not None
        and _RECIPE_SHA256_PATTERN.fullmatch(source_recipe_sha256) is None
    ):
        raise ValueError("source_recipe_sha256 must be a lowercase SHA-256")
    rgb_array = _load_rgb(rgb)
    depth = _load_2d(depth_m, dtype=np.dtype(np.float32))
    mask = _load_2d(predicted_mask, dtype=np.dtype(bool))
    if depth.shape != rgb_array.shape[:2] or mask.shape != rgb_array.shape[:2]:
        raise ValueError("RGB, metric depth, and predicted mask shapes must match")
    candidates = list(candidates_q_order)
    candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if not candidate_ids or any(not value for value in candidate_ids):
        raise ValueError("every VLM candidate requires a candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")

    destination = Path(output_dir)
    full_path = destination / "full_scene_overlay.png"
    sheet_path = destination / "candidate_contact_sheet.png"
    manifest_path = destination / "vlm_visualization_manifest.json"

    overlay = rgb_array.astype(np.float32)
    tint = np.asarray([45.0, 170.0, 190.0], dtype=np.float32)
    overlay[mask] = (1.0 - mask_alpha) * overlay[mask] + mask_alpha * tint
    scene_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    legend_width = 176
    full_image = Image.new(
        "RGB", (scene_image.width + legend_width, scene_image.height), (24, 24, 24)
    )
    full_image.paste(scene_image, (0, 0))
    full_draw = ImageDraw.Draw(full_image)
    boundary_y, boundary_x = np.nonzero(_mask_boundary(mask))
    for x, y in zip(boundary_x.tolist(), boundary_y.tolist()):
        full_draw.point((x, y), fill=(235, 235, 235))
    for index, candidate in enumerate(candidates):
        color = NEUTRAL_CANDIDATE_COLORS[
            index % len(NEUTRAL_CANDIDATE_COLORS)
        ]
        if index == 0:
            _draw_candidate(
                full_draw, candidate, color=(245, 245, 245), width=5
            )
        _draw_candidate(full_draw, candidate, color=color, width=3)
        center, _, _, _ = _candidate_geometry(candidate)
        legend_y = 34 + index * 34
        full_draw.line(
            [
                (float(center[0]), float(center[1])),
                (scene_image.width - 4, legend_y + 7),
                (scene_image.width + 12, legend_y + 7),
            ],
            fill=color,
            width=2,
        )
        full_draw.rectangle(
            (
                scene_image.width + 16,
                legend_y,
                scene_image.width + 30,
                legend_y + 14,
            ),
            fill=color,
        )
        full_draw.text(
            (scene_image.width + 36, legend_y + 1),
            f"{index + 1}: {candidate_ids[index]}",
            fill=(245, 245, 245),
            font=ImageFont.load_default(),
        )
    full_draw.text(
        (scene_image.width + 12, 8),
        "Neutral candidate key",
        fill=(245, 245, 245),
        font=ImageFont.load_default(),
    )
    full_draw.text(
        (scene_image.width + 12, min(scene_image.height - 28, 38 + len(candidates) * 34)),
        "White outline = original q Top-1",
        fill=(215, 215, 215),
        font=ImageFont.load_default(),
    )
    _atomic_save_png(full_image, full_path)

    valid_depth = np.isfinite(depth) & (depth > 0)
    if np.any(valid_depth):
        depth_min, depth_max = np.percentile(
            depth[valid_depth].astype(np.float64), [2.0, 98.0]
        ).tolist()
        if depth_max <= depth_min:
            depth_max = depth_min + 1e-6
    else:
        depth_min, depth_max = 0.0, 1.0
    depth_rgb = _depth_color(depth, valid_depth, depth_min, depth_max)
    filled_depth = np.where(valid_depth, depth, np.nan)
    fill_value = float(np.nanmedian(filled_depth)) if np.any(valid_depth) else 0.0
    stable_depth = np.nan_to_num(filled_depth, nan=fill_value)
    grad_y, grad_x = np.gradient(stable_depth)
    edge = np.hypot(grad_x, grad_y)
    edge_scale = (
        float(np.percentile(edge[valid_depth], 98.0))
        if np.any(valid_depth)
        else 1.0
    )
    edge_scale = max(edge_scale, 1e-9)
    edge_gray = np.round(np.clip(edge / edge_scale, 0, 1) * 255).astype(np.uint8)
    edge_rgb = np.repeat(edge_gray[..., None], 3, axis=2)
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask] = np.asarray([45, 190, 205], dtype=np.uint8)

    rgb_image = Image.fromarray(rgb_array)
    depth_image = Image.fromarray(depth_rgb)
    edge_image = Image.fromarray(edge_rgb)
    mask_image = Image.fromarray(mask_rgb)
    label_width = 136
    gutter = 8
    header_height = 26
    row_height = crop_size + gutter
    sheet_width = label_width + 4 * (crop_size + gutter)
    sheet_height = header_height + len(candidates) * row_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    headings = ("RGB + grasp", "metric depth", "depth edge", "predicted mask")
    for panel, heading in enumerate(headings):
        x = label_width + panel * (crop_size + gutter)
        draw.text((x + 4, 6), heading, fill=(230, 230, 230))
    for row_index, candidate in enumerate(candidates):
        center, _, _, _ = _candidate_geometry(candidate)
        bounds = _crop_bounds(center, crop_size, rgb_image.width, rgb_image.height)
        y = header_height + row_index * row_height
        draw.text(
            (6, y + 8),
            f"{candidate_ids[row_index]}\nq-rank {row_index + 1}",
            fill=(245, 245, 245),
        )
        crops = [
            _crop_with_padding(rgb_image, bounds, fill=(0, 0, 0)),
            _crop_with_padding(depth_image, bounds, fill=(20, 20, 20)),
            _crop_with_padding(edge_image, bounds, fill=(0, 0, 0)),
            _crop_with_padding(mask_image, bounds, fill=(0, 0, 0)),
        ]
        for panel, crop in enumerate(crops):
            crop_draw = ImageDraw.Draw(crop)
            color = NEUTRAL_CANDIDATE_COLORS[
                row_index % len(NEUTRAL_CANDIDATE_COLORS)
            ]
            if row_index == 0:
                _draw_candidate(
                    crop_draw,
                    candidate,
                    offset=(float(bounds[0]), float(bounds[1])),
                    color=(245, 245, 245),
                    width=5,
                )
            _draw_candidate(
                crop_draw,
                candidate,
                offset=(float(bounds[0]), float(bounds[1])),
                color=color,
                width=3,
            )
            x = label_width + panel * (crop_size + gutter)
            sheet.paste(crop, (x, y))
    _atomic_save_png(sheet, sheet_path)

    normalization = {
        "method": "sample-global valid metric-depth percentiles",
        "valid_rule": "finite and > 0 metres",
        "percentile_low": 2.0,
        "percentile_high": 98.0,
        "depth_min_m": float(depth_min),
        "depth_max_m": float(depth_max),
        "same_scale_for_all_candidates": True,
        "depth_edge": "Euclidean np.gradient magnitude; clipped at sample-global "
        "valid-pixel p98",
        "depth_edge_scale_m_per_pixel": float(edge_scale),
    }
    manifest = {
        "schema_version": 1,
        "sample_id": sample_id,
        "gt_free": True,
        "source_recipe_sha256": source_recipe_sha256,
        "candidate_ids": candidate_ids,
        "candidate_records_sha256": hashlib.sha256(
            json.dumps(
                candidates,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "candidate_order": "full-precision GQ-CNN descending; candidate_id tie-break",
        "original_top1_neutral_marker": candidate_ids[0],
        "candidate_color_semantics": (
            "fixed categorical identity colors only; colors do not encode "
            "correctness, quality, safety, or ground truth"
        ),
        "inputs": {
            "rgb_sha256": _sha256_array(rgb_array),
            "metric_depth_sha256": _sha256_array(depth),
            "predicted_hifi_mask_sha256": _sha256_array(mask),
            "allowed_modalities": [
                "rgb",
                "metric_depth",
                "predicted_hifi_mask",
                "frozen_candidate_pose",
            ],
        },
        "forbidden_inputs_absent": [
            "ground-truth mask",
            "ground-truth grasp",
            "candidate correctness",
            "evaluation metrics",
        ],
        "depth_normalization": normalization,
        "crop_size_px": crop_size,
        "output_images": [
            {"path": str(full_path.resolve()), "sha256": _sha256_file(full_path)},
            {"path": str(sheet_path.resolve()), "sha256": _sha256_file(sheet_path)},
        ],
    }
    _atomic_write_json(manifest, manifest_path)
    return VLMVisualizationResult(
        full_scene_overlay_path=full_path,
        candidate_contact_sheet_path=sheet_path,
        manifest_path=manifest_path,
        depth_normalization=normalization,
    )
