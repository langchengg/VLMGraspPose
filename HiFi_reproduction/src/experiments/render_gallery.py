"""Lightweight WebP artifacts and a self-contained experiment gallery.

This module is deliberately presentation-only.  It never changes candidate
scores, target filtering, or sample outcomes.  Existing PNG diagnostics are
converted losslessly with respect to experiment semantics; an optional GT mask
is used only to draw a magenta boundary on the rendered image.
"""

from __future__ import annotations

import csv
import html
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


WEBP_ARTIFACTS = {
    "rgb_mask_overlay.webp": "rgb_mask_overlay.png",
    "candidates_2d_overlay.webp": "candidates_2d_overlay.png",
    "top1_2d_overlay.webp": "top1_2d_overlay.png",
}


def _atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def _resolve_optional_path(value: Any, *, relative_to: Path) -> Path | None:
    if value in (None, ""):
        return None
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    candidate = candidate.resolve()
    return candidate if candidate.is_file() else None


def _gt_boundary(mask_path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 0
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def atomic_save_webp(
    image: Image.Image,
    destination: str | Path,
    *,
    quality: int = 82,
    method: int = 6,
) -> Path:
    """Write a WebP via fsync + atomic rename."""

    path = Path(destination).expanduser().resolve()
    if not 0 <= int(quality) <= 100:
        raise ValueError("WebP quality must be in [0, 100]")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".webp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.convert("RGB").save(
            temporary,
            format="WEBP",
            quality=int(quality),
            method=int(method),
            exact=True,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def convert_sample_artifacts(
    sample_dir: str | Path,
    *,
    gt_mask_path: str | Path | None = None,
    quality: int = 82,
    overwrite: bool = False,
) -> dict[str, str]:
    """Convert available PNG diagnostics to lightweight, atomic WebP files.

    ``gt_mask_path`` affects pixels only: a magenta GT boundary is added when
    the path exists.  It does not affect any metric or filtering result.
    """

    directory = Path(sample_dir).expanduser().resolve()
    explicit_gt = _resolve_optional_path(gt_mask_path, relative_to=directory)
    converted: dict[str, str] = {}
    for output_name, input_name in WEBP_ARTIFACTS.items():
        source = directory / input_name
        destination = directory / output_name
        if destination.is_file() and not overwrite:
            converted[output_name] = str(destination)
            continue
        if not source.is_file():
            continue
        with Image.open(source) as loaded:
            rendered = loaded.convert("RGB")
            if explicit_gt is not None:
                pixels = np.asarray(rendered).copy()
                boundary = _gt_boundary(explicit_gt, rendered.size)
                pixels[boundary] = np.array([255, 46, 196], dtype=np.uint8)
                rendered = Image.fromarray(pixels, mode="RGB")
            atomic_save_webp(rendered, destination, quality=quality)
        converted[output_name] = str(destination)
    return converted


def _sample_mapping(sample: Any) -> dict[str, Any]:
    if isinstance(sample, Mapping):
        return dict(sample)
    if is_dataclass(sample):
        return asdict(sample)
    return dict(vars(sample))


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    interior = binary.copy()
    interior[1:, :] &= binary[:-1, :]
    interior[:-1, :] &= binary[1:, :]
    interior[:, 1:] &= binary[:, :-1]
    interior[:, :-1] &= binary[:, 1:]
    return binary & ~interior


def _load_binary_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize(size, Image.Resampling.NEAREST)
        ) > 0


def _header(
    image: Image.Image,
    *,
    sample_id: str,
    instruction: str,
    status: str,
    subtitle: str,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 48), fill=(0, 0, 0, 185))
    draw.text((7, 5), f"{sample_id}  [{status}]", fill=(255, 255, 255, 255))
    clipped = instruction if len(instruction) <= 78 else f"{instruction[:75]}..."
    draw.text((7, 19), clipped, fill=(235, 240, 247, 255))
    draw.text((7, 33), subtitle, fill=(163, 205, 255, 255))


def _render_base(
    rgb: Image.Image,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
) -> Image.Image:
    rendered = rgb.convert("RGB")
    pixels = np.asarray(rendered).copy()
    mask = np.asarray(pred_mask, dtype=bool)
    pixels[mask] = np.clip(
        0.72 * pixels[mask] + 0.28 * np.array([30, 225, 92]), 0, 255
    ).astype(np.uint8)
    pixels[_boundary(mask)] = np.array([37, 211, 102], dtype=np.uint8)
    if gt_mask is not None:
        pixels[_boundary(gt_mask)] = np.array([255, 46, 196], dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _candidate_payload(sample_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates_payload = _read_json(sample_dir / "candidates.json")
    raw_candidates = candidates_payload.get("all_official_vgn_candidates", [])
    candidates = [dict(value) for value in raw_candidates if isinstance(value, Mapping)]
    top1 = _read_json(sample_dir / "top1.json")
    if not top1:
        # Full-run failure records may have no grasp-specific top1.json.  Their
        # atomic result record still carries the terminal status for rendering.
        top1 = _read_json(sample_dir / "result.json")
    return candidates, top1


def _draw_candidates(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for index, candidate in enumerate(candidates[: max(0, int(top_k))]):
        uv = candidate.get("projected_uv")
        if not isinstance(uv, (list, tuple)) or len(uv) != 2:
            continue
        try:
            u, v = float(uv[0]), float(uv[1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite([u, v]).all():
            continue
        accepted = bool(candidate.get("inside_dilated_target_mask", False))
        color = (52, 226, 108, 255) if accepted else (255, 76, 76, 255)
        if accepted:
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=color, width=2)
        else:
            draw.line((u - 4, v - 4, u + 4, v + 4), fill=color, width=2)
            draw.line((u - 4, v + 4, u + 4, v - 4), fill=color, width=2)
        quality = _float(candidate.get("vgn_quality"))
        label = f"{index}:{quality:.3f}" if quality is not None else str(index)
        draw.text((u + 5, v - 7), label, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 220))


def _load_intrinsics(sample_dir: Path, sample: Mapping[str, Any]) -> dict[str, float] | None:
    candidates: list[Path] = [sample_dir / "intrinsics.json"]
    raw_path = sample.get("intrinsics_path")
    if raw_path not in (None, ""):
        candidates.insert(0, Path(str(raw_path)).expanduser())
    workspace = _read_json(sample_dir / "workspace_frame.json")
    if isinstance(workspace.get("intrinsics"), Mapping):
        values = workspace["intrinsics"]
    else:
        values = next((_read_json(path) for path in candidates if path.is_file()), {})
    try:
        return {key: float(values[key]) for key in ("fx", "fy", "cx", "cy")}
    except (KeyError, TypeError, ValueError):
        return None


def _project_xyz(point: np.ndarray, intrinsics: Mapping[str, float]) -> tuple[float, float] | None:
    xyz = np.asarray(point, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or xyz[2] <= 0:
        return None
    return (
        intrinsics["fx"] * xyz[0] / xyz[2] + intrinsics["cx"],
        intrinsics["fy"] * xyz[1] / xyz[2] + intrinsics["cy"],
    )


def _draw_top1(
    image: Image.Image,
    candidate: Mapping[str, Any] | None,
    intrinsics: Mapping[str, float] | None,
) -> None:
    if not candidate:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    uv = candidate.get("projected_uv")
    center_uv: tuple[float, float] | None = None
    if isinstance(uv, (list, tuple)) and len(uv) == 2:
        try:
            center_uv = (float(uv[0]), float(uv[1]))
        except (TypeError, ValueError):
            pass
    transform_raw = candidate.get("T_camera_grasp")
    if transform_raw is not None and intrinsics is not None:
        try:
            transform = np.asarray(transform_raw, dtype=np.float64).reshape(4, 4)
            center = transform[:3, 3]
            rotation = transform[:3, :3]
            center_uv = _project_xyz(center, intrinsics) or center_uv
            approach_uv = _project_xyz(center + 0.05 * rotation[:, 2], intrinsics)
            width = float(candidate.get("width_m", 0.0))
            left_uv = _project_xyz(center - 0.5 * width * rotation[:, 1], intrinsics)
            right_uv = _project_xyz(center + 0.5 * width * rotation[:, 1], intrinsics)
            if center_uv is not None and approach_uv is not None:
                draw.line((*center_uv, *approach_uv), fill=(0, 199, 255, 255), width=3)
            if left_uv is not None and right_uv is not None:
                draw.line((*left_uv, *right_uv), fill=(255, 168, 35, 255), width=4)
        except (TypeError, ValueError):
            pass
    if center_uv is not None and np.isfinite(center_uv).all():
        u, v = center_uv
        draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=(255, 214, 10, 255), outline=(0, 0, 0, 255), width=2)


def render_sample_webp(
    sample_dir: Path,
    sample: Any,
    *,
    top_k: int = 50,
    gt_mask_path: Path | None = None,
    quality: int = 82,
) -> dict[str, Path]:
    """Render the three required WebPs directly from RGB/mask/result JSON.

    Terminal failure samples are intentionally renderable: absent candidates or
    top-1 poses produce annotated RGB/mask panels carrying the failure status.
    """

    directory = Path(sample_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    values = _sample_mapping(sample)
    rgb_path = Path(str(values["rgb_path"])).expanduser().resolve()
    mask_path = Path(str(values["mask_path"])).expanduser().resolve()
    with Image.open(rgb_path) as loaded_rgb:
        rgb = loaded_rgb.convert("RGB")
    pred_mask = _load_binary_mask(mask_path, rgb.size)
    gt_mask = (
        _load_binary_mask(Path(gt_mask_path).expanduser().resolve(), rgb.size)
        if gt_mask_path is not None and Path(gt_mask_path).expanduser().is_file()
        else None
    )
    candidates, top1 = _candidate_payload(directory)
    status = str(top1.get("status") or values.get("status") or values.get("outcome_status") or "unknown")
    sample_id = str(values.get("sample_id") or directory.name)
    instruction = str(values.get("instruction") or values.get("query") or "")

    mask_overlay = _render_base(rgb, pred_mask, gt_mask)
    _header(
        mask_overlay,
        sample_id=sample_id,
        instruction=instruction,
        status=status,
        subtitle="HiFi-CS predicted boundary (green); GT boundary (magenta, when available)",
    )
    candidate_overlay = _render_base(rgb, pred_mask, gt_mask)
    _draw_candidates(candidate_overlay, candidates, top_k=top_k)
    _header(
        candidate_overlay,
        sample_id=sample_id,
        instruction=instruction,
        status=status,
        subtitle=f"official candidates={len(candidates)}; green=target-filtered; red=off-target",
    )
    top1_overlay = _render_base(rgb, pred_mask, gt_mask)
    candidate = top1.get("candidate") if isinstance(top1.get("candidate"), Mapping) else None
    _draw_top1(top1_overlay, candidate, _load_intrinsics(directory, values))
    quality_value = _float(candidate.get("vgn_quality")) if candidate else None
    quality_text = f"{quality_value:.6f}" if quality_value is not None else "not available"
    _header(
        top1_overlay,
        sample_id=sample_id,
        instruction=instruction,
        status=status,
        subtitle=f"top-1 official VGN quality={quality_text}; cyan=approach; orange=closing width",
    )

    rendered = {
        "rgb_mask_overlay": atomic_save_webp(
            mask_overlay, directory / "rgb_mask_overlay.webp", quality=quality
        ),
        "candidates_2d_overlay": atomic_save_webp(
            candidate_overlay, directory / "candidates_2d_overlay.webp", quality=quality
        ),
        "top1_2d_overlay": atomic_save_webp(
            top1_overlay, directory / "top1_2d_overlay.webp", quality=quality
        ),
    }
    return rendered


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_rows(output: Path) -> list[dict[str, Any]]:
    for candidate in (
        output / "metrics" / "per_sample.csv",
        output / "per_sample.csv",
        output / "summary.csv",
    ):
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8", newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
    samples = output / "samples"
    rows: list[dict[str, Any]] = []
    if samples.is_dir():
        for directory in sorted(path for path in samples.iterdir() if path.is_dir()):
            payload = _read_json(directory / "result.json")
            top1 = _read_json(directory / "top1.json")
            rows.append({"sample_id": directory.name, **payload, **top1})
    return rows


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _find_gt_mask(row: Mapping[str, Any], sample_dir: Path) -> Path | None:
    for key in (
        "gt_mask_path",
        "ground_truth_mask_path",
        "ground_truth_mask_original_resolution_path",
    ):
        if path := _resolve_optional_path(row.get(key), relative_to=sample_dir):
            return path
    for metadata_name in ("result.json", "sample_metadata.json", "metadata.json"):
        metadata = _read_json(sample_dir / metadata_name)
        for key in (
            "gt_mask_path",
            "ground_truth_mask_path",
            "ground_truth_mask_original_resolution_path",
        ):
            if path := _resolve_optional_path(metadata.get(key), relative_to=sample_dir):
                return path
    return None


def _artifact_link(report_dir: Path, path: Path) -> str | None:
    if not path.is_file():
        return None
    return Path(os.path.relpath(path, report_dir)).as_posix()


def build_gallery(
    rows: Iterable[Mapping[str, Any]],
    samples_root: str | Path,
    output_path: str | Path,
    *,
    webp_quality: int = 82,
) -> Path:
    """Build a portable gallery whose local links are guaranteed to exist."""

    samples = Path(samples_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    report = destination.parent
    report.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in rows]
    cards: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        sample_dir = samples / sample_id
        raw_alternate = row.get("sample_output_dir")
        if not sample_dir.is_dir() and raw_alternate not in (None, ""):
            alternate = Path(str(raw_alternate)).expanduser()
            if not alternate.is_absolute():
                alternate = samples / alternate
            if alternate.is_dir():
                sample_dir = alternate.resolve()
        gt_mask = _find_gt_mask(row, sample_dir) if sample_dir.is_dir() else None
        if sample_dir.is_dir():
            convert_sample_artifacts(
                sample_dir, gt_mask_path=gt_mask, quality=webp_quality
            )
        status = str(row.get("outcome_status") or row.get("status") or row.get("state") or "unknown")
        query_type = str(row.get("query_type") or "unknown")
        category = str(row.get("target_category") or row.get("category") or "unknown")
        top1_quality = _float(row.get("top1_vgn_quality"))
        if top1_quality is None and sample_dir.is_dir():
            top1_payload = _read_json(sample_dir / "top1.json")
            candidate = top1_payload.get("candidate")
            if isinstance(candidate, Mapping):
                top1_quality = _float(candidate.get("vgn_quality"))
        images = [
            link
            for name in WEBP_ARTIFACTS
            if (link := _artifact_link(report, sample_dir / name)) is not None
        ]
        top1_link = _artifact_link(report, sample_dir / "top1.json")
        view_link = next(
            (
                link
                for name in ("grasps_3d.html", "grasps_3d.ply")
                if (link := _artifact_link(report, sample_dir / name)) is not None
            ),
            None,
        )
        cards.append(
            {
                "sample_id": sample_id,
                "instruction": str(row.get("instruction") or row.get("query") or ""),
                "status": status,
                "query_type": query_type,
                "category": category,
                "mask_iou": _float(row.get("pred_mask_iou") or row.get("mask_iou")),
                "vgn_quality": top1_quality,
                "candidate_count": _int(
                    row.get("official_candidate_count")
                    or row.get("candidate_count_before_target_filter")
                ),
                "images": images,
                "top1_link": top1_link,
                "view_link": view_link,
            }
        )

    statuses = sorted({card["status"] for card in cards})
    query_types = sorted({card["query_type"] for card in cards})
    categories = sorted({card["category"] for card in cards})

    def options(values: list[str]) -> str:
        return "".join(
            f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
            for value in values
        )

    card_markup: list[str] = []
    for card in cards:
        images = "".join(
            f'<img loading="lazy" src="{html.escape(source, quote=True)}" '
            f'alt="{html.escape(card["sample_id"], quote=True)} diagnostic {index + 1}">'
            for index, source in enumerate(card["images"])
        ) or '<div class="empty">No rendered image</div>'
        links = ""
        if card["top1_link"]:
            links += f'<a href="{html.escape(card["top1_link"], quote=True)}">top1.json</a>'
        if card["view_link"]:
            links += f'<a href="{html.escape(card["view_link"], quote=True)}">3D view</a>'
        mask_iou = (
            f'{card["mask_iou"]:.3f}' if card["mask_iou"] is not None else "—"
        )
        quality_text = (
            f'{card["vgn_quality"]:.3f}'
            if card["vgn_quality"] is not None
            else "—"
        )
        card_markup.append(
            f'<article data-status="{html.escape(card["status"], quote=True)}" '
            f'data-query="{html.escape(card["query_type"], quote=True)}" '
            f'data-category="{html.escape(card["category"], quote=True)}" '
            f'data-mask-iou="{card["mask_iou"] if card["mask_iou"] is not None else ""}" '
            f'data-vgn-quality="{card["vgn_quality"] if card["vgn_quality"] is not None else ""}" '
            f'data-candidate-count="{card["candidate_count"]}">'
            f'<div class="images">{images}</div><div class="meta">'
            f'<div class="title">{html.escape(card["sample_id"])}</div>'
            f'<div class="instruction">{html.escape(card["instruction"])}</div>'
            f'<div class="tags">status: {html.escape(card["status"])} · '
            f'query: {html.escape(card["query_type"])} · category: {html.escape(card["category"])}</div>'
            f'<div class="metrics">mask IoU {mask_iou} · quality {quality_text} · '
            f'official candidates {card["candidate_count"]}</div><div>{links}</div></div></article>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCID-VLG × HiFi-CS × VGN gallery</title>
<style>
:root{{--bg:#11141a;--panel:#1b2029;--text:#eef2f8;--muted:#aeb8c7;--ok:#49d17d;--bad:#ff6b6b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:16px;background:#11141af2;border-bottom:1px solid #343b48}}
h1{{margin:0 0 12px;font-size:20px}}.controls{{display:flex;gap:8px;flex-wrap:wrap}}select{{padding:7px;background:var(--panel);color:var(--text);border:1px solid #465063;border-radius:5px}}
#count{{color:var(--muted);align-self:center}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px;padding:12px}}
article{{background:var(--panel);border:1px solid #333b49;border-radius:8px;overflow:hidden}}article[data-status="ok"]{{border-color:#2b7449}}
.images{{display:grid;grid-auto-flow:column;grid-auto-columns:100%;overflow-x:auto;scroll-snap-type:x mandatory}}.images img{{width:100%;aspect-ratio:4/3;object-fit:contain;background:#080a0e;scroll-snap-align:start}}
.meta{{padding:10px}}.title{{font-weight:700;overflow-wrap:anywhere}}.instruction{{margin:6px 0;color:#dce4f0;min-height:2.4em}}.tags{{color:var(--muted);font-size:12px}}.metrics{{margin-top:7px;font-variant-numeric:tabular-nums}}a{{color:#8cc8ff;margin-right:10px}}.empty{{padding:4px;color:var(--muted)}}
</style></head><body>
<header><h1>OCID-VLG × HiFi-CS × official VGN — qualitative gallery</h1>
<div class="controls">
<select id="status"><option value="">all statuses</option>{options(statuses)}</select>
<select id="query"><option value="">all query types</option>{options(query_types)}</select>
<select id="category"><option value="">all categories</option>{options(categories)}</select>
<select id="sort"><option value="dataset">manifest order</option><option value="mask_iou_desc">mask IoU ↓</option><option value="vgn_quality_desc">VGN quality ↓</option><option value="candidate_count_desc">official candidates ↓</option></select>
<span id="count"></span></div></header><main id="grid">{''.join(card_markup)}</main>
<script>
const grid=document.querySelector('#grid');const cards=[...grid.querySelectorAll('article')];
function render(){{
 const status=document.querySelector('#status').value,query=document.querySelector('#query').value,category=document.querySelector('#category').value,sort=document.querySelector('#sort').value;
 let visible=cards.filter(card=>(!status||card.dataset.status===status)&&(!query||card.dataset.query===query)&&(!category||card.dataset.category===category));
 const key={{mask_iou_desc:'maskIou',vgn_quality_desc:'vgnQuality',candidate_count_desc:'candidateCount'}}[sort];
 if(key)visible.sort((a,b)=>(Number(b.dataset[key]||'-Infinity')-Number(a.dataset[key]||'-Infinity'))||a.querySelector('.title').textContent.localeCompare(b.querySelector('.title').textContent));
 cards.forEach(card=>card.hidden=!visible.includes(card));visible.forEach(card=>grid.appendChild(card));
 document.querySelector('#count').textContent=`${{visible.length}} / ${{cards.length}} samples`;
}}
for(const id of ['status','query','category','sort'])document.querySelector('#'+id).addEventListener('change',render);render();
</script></body></html>
"""
    return _atomic_text(destination, document)


__all__ = [
    "WEBP_ARTIFACTS",
    "atomic_save_webp",
    "build_gallery",
    "convert_sample_artifacts",
    "render_sample_webp",
]
