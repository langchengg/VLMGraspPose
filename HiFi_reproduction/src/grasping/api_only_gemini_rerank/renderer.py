"""Two-board, GT-free renderer with display-order randomisation."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.grasping.common.geometry import rectangle_corners
from src.grasping.common.sample_io import CompactSampleLoader

from .contracts import assert_no_gt_payload, validate_display_mapping
from .evidence import contact_points
from .io import sha256_json


RENDERER_VERSION = "api_only_two_board_v3"
PALETTE = ((45, 110, 180), (180, 120, 40), (130, 80, 170), (35, 145, 145), (190, 155, 45))


def renderer_hash() -> str:
    return sha256_json({
        "version": RENDERER_VERSION, "palette": PALETTE, "overview": [640, 480],
        "crop_side": 160, "max_long_edge": 2048, "success_colours": False,
    })


def display_mapping(candidate_ids: Sequence[str], *, seed: int) -> dict[str, str]:
    ids = list(map(str, candidate_ids))
    rng = np.random.default_rng(int(seed))
    display = np.asarray([chr(ord("A") + index) for index in range(len(ids))], dtype=object)
    rng.shuffle(display)
    mapping = dict(zip(ids, map(str, display), strict=True))
    validate_display_mapping(mapping, ids)
    return mapping


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_candidate(image: Image.Image, candidate: Mapping[str, Any], label: str, colour: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    corners = rectangle_corners(
        float(candidate["center_x"]), float(candidate["center_y"]), float(candidate["width_px"]),
        float(candidate["height_px"]), float(candidate["angle_deg"]),
    )
    points = [tuple(map(float, point)) for point in corners]
    draw.line(points + [points[0]], fill=colour, width=3)
    centre = (float(candidate["center_x"]), float(candidate["center_y"]))
    contacts = contact_points(candidate)
    draw.ellipse((centre[0]-4, centre[1]-4, centre[0]+4, centre[1]+4), fill=colour)
    draw.line([tuple(contacts[0]), tuple(contacts[1])], fill=colour, width=2)
    for point in contacts:
        draw.ellipse((point[0]-4, point[1]-4, point[0]+4, point[1]+4), outline=colour, width=2)
    draw.text((centre[0]+5, centre[1]-22), label, fill=colour, font=_font(18), stroke_width=2, stroke_fill=(255,255,255))


def _crop(array: np.ndarray, centre: tuple[float, float], side: int = 160, *, nearest: bool = False) -> np.ndarray:
    height, width = array.shape[:2]
    x, y = map(int, map(round, centre))
    half = side // 2
    x0, y0 = x - half, y - half
    x1, y1 = x0 + side, y0 + side
    source_x0, source_y0 = max(0, x0), max(0, y0)
    source_x1, source_y1 = min(width, x1), min(height, y1)
    shape = (side, side) if array.ndim == 2 else (side, side, *array.shape[2:])
    crop = np.zeros(shape, dtype=array.dtype)
    if source_x0 < source_x1 and source_y0 < source_y1:
        target_x0, target_y0 = source_x0 - x0, source_y0 - y0
        target_x1 = target_x0 + source_x1 - source_x0
        target_y1 = target_y0 + source_y1 - source_y0
        crop[target_y0:target_y1, target_x0:target_x1] = array[source_y0:source_y1, source_x0:source_x1]
    return crop


def _depth_rgb(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.rint(np.clip((depth[valid]-low)/max(high-low, 1e-9), 0, 1)*255).astype(np.uint8)
    rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_CIVIDIS)[:, :, ::-1]
    rgb[~valid] = 0
    return rgb


def _png(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True, pnginfo=None)
    return stream.getvalue()


def render_boards(
    candidate_rows: Sequence[Mapping[str, Any]],
    deployment: Mapping[str, Any],
    *,
    seed: int,
    evidence_variant: str,
    output_dir: Path | None = None,
) -> tuple[bytes, bytes, dict[str, Any]]:
    if not 2 <= len(candidate_rows) <= 5:
        raise ValueError("boards are only generated for API-eligible K=2..5")
    loader = CompactSampleLoader()
    arrays = loader.load(deployment, mask_source="predicted", load_intrinsics=False)
    frozen_candidates = sorted(candidate_rows, key=lambda row: (int(row["original_rank"]), str(row["candidate_id"])))
    mapping = display_mapping([str(row["candidate_id"]) for row in frozen_candidates], seed=seed)
    # Render in display-ID order. Because internal->display assignment is seeded and
    # random, neither the grid position nor colour encodes original backend rank.
    candidates = sorted(frozen_candidates, key=lambda row: mapping[str(row["candidate_id"])])
    overview = Image.fromarray(arrays.rgb.copy())
    if evidence_variant != "E0_RGB_ONLY":
        overlay = arrays.rgb.copy()
        contours, _ = cv2.findContours(arrays.binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (110, 110, 110), 2)
        overview = Image.fromarray(overlay)
    for candidate in candidates:
        display_id = mapping[str(candidate["candidate_id"])]
        colour = PALETTE[ord(display_id) - ord("A")]
        _draw_candidate(overview, candidate, display_id, colour)
    draw = ImageDraw.Draw(overview)
    draw.rectangle((0,0,640,32), fill=(245,245,245))
    draw.text((8,7), "Frozen candidates; colours encode identity only, never correctness", fill=(20,20,20), font=_font(15))

    valid = arrays.depth_m[np.isfinite(arrays.depth_m) & (arrays.depth_m > 0)]
    low, high = (0.0, 1.0) if not len(valid) else tuple(map(float, np.quantile(valid, [0.02, 0.98])))
    rows: list[Image.Image] = []
    transforms: list[dict[str, Any]] = []
    for candidate in candidates:
        display_id = mapping[str(candidate["candidate_id"])]
        colour = PALETTE[ord(display_id) - ord("A")]
        centre = (float(candidate["center_x"]), float(candidate["center_y"]))
        rgb_crop = _crop(arrays.rgb, centre)
        layers = [rgb_crop]
        labels = ["RGB"]
        if evidence_variant != "E0_RGB_ONLY":
            probability = _crop(arrays.probability, centre)
            layers.append(np.repeat(np.rint(np.clip(probability,0,1)*255).astype(np.uint8)[:,:,None],3,axis=2))
            labels.append("Predicted probability")
        if evidence_variant in {"E2_RGBD_GEOMETRY_SCORE_BLIND", "E3_RGBD_GEOMETRY_SCORE_AWARE"}:
            layers.append(_depth_rgb(_crop(arrays.depth_m, centre, nearest=True), low, high))
            labels.append(f"Depth m [{low:.2f},{high:.2f}]")
        panel = Image.new("RGB", (160*len(layers), 190), "white")
        for column, (layer, title) in enumerate(zip(layers, labels, strict=True)):
            tile = Image.fromarray(layer.astype(np.uint8))
            # Candidate geometry is drawn in crop coordinates using an offset copy.
            shifted = dict(candidate)
            shifted["center_x"] = 80.0
            shifted["center_y"] = 80.0
            _draw_candidate(tile, shifted, display_id, colour)
            panel.paste(tile, (column*160, 30))
            ImageDraw.Draw(panel).text((column*160+5, 7), title, fill=(20,20,20), font=_font(13))
        rows.append(panel)
        transforms.append({"candidate_id": str(candidate["candidate_id"]), "display_id": display_id, "crop_center_xy": list(centre), "crop_side_px": 160})
    grid_width = max(panel.width for panel in rows)
    grid = Image.new("RGB", (grid_width, 190*len(rows)+34), "white")
    ImageDraw.Draw(grid).text((8,7), "Equal crop, scale, font, line width and sample-global metric-depth range", fill=(20,20,20), font=_font(14))
    for index, panel in enumerate(rows):
        grid.paste(panel, (0, 34+190*index))
    overview_bytes, grid_bytes = _png(overview), _png(grid)
    metadata = {
        "renderer_version": RENDERER_VERSION,
        "renderer_hash": renderer_hash(),
        "sample_id": str(deployment["sample_id"]),
        "source_rgb_sha256": str(deployment["source_rgb_sha256"]),
        "source_depth_sha256": str(deployment["source_depth_sha256"]),
        "predicted_mask_sha256": str(deployment["predicted_mask_sha256"]),
        "predicted_probability_sha256": str(deployment["predicted_probability_sha256"]),
        "candidate_set_sha256": str(frozen_candidates[0]["candidate_set_sha256"]),
        "display_mapping": mapping,
        "palette_by_display_id": {
            chr(ord("A") + index): list(PALETTE[index]) for index in range(len(candidates))
        },
        "palette_semantics": "identity only; never correctness",
        "evidence_variant": evidence_variant,
        "crop_transforms": transforms,
        "depth_display_range_m": [low, high],
        "scene_overview_sha256": hashlib.sha256(overview_bytes).hexdigest(),
        "candidate_evidence_grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
    }
    assert_no_gt_payload(metadata)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scene_overview.png").write_bytes(overview_bytes)
        (output_dir / "candidate_evidence_grid.png").write_bytes(grid_bytes)
        (output_dir / "board_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return overview_bytes, grid_bytes, metadata
