#!/usr/bin/env python3
"""Build a deterministic visual inspection bundle from a completed candidate run."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.grasp_visualization import (  # noqa: E402
    save_candidate_overlay,
    save_depth_visualization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-successes", type=int, default=20)
    parser.add_argument("--tail-count", type=int, default=10)
    parser.add_argument("--small-percentile", type=float, default=1.0)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Allow a verifier-clean run that still has terminal failed samples.",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_review_paths(candidate_root: Path, review_root: Path) -> None:
    if (
        review_root == candidate_root
        or review_root.is_relative_to(candidate_root)
        or candidate_root.is_relative_to(review_root)
    ):
        raise ValueError("review root and candidate root must be separate, non-nested paths")


def verified_summary_rows(summary: Any, *, allow_failures: bool) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain an object")
    verification = summary.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("summary.json has no verifier report")
    blocking_fields = (
        "missing_samples",
        "corrupt_samples",
    )
    blocking = {name: verification.get(name) for name in blocking_fields if verification.get(name)}
    for name in (
        "duplicate_sample_ids",
        "unexpected_output_directories",
        "configuration_hash_mismatches",
        "seed_mismatches",
    ):
        if verification.get(name):
            blocking[name] = verification[name]
    if verification.get("accounting_identity") != verification.get("accounting_identity_expected"):
        blocking["accounting_identity"] = {
            "actual": verification.get("accounting_identity"),
            "expected": verification.get("accounting_identity_expected"),
        }
    if blocking:
        raise ValueError(f"candidate verifier report is not clean: {blocking}")
    if verification.get("failed_samples") and not allow_failures:
        raise ValueError("terminal failures present; pass --allow-failures to include them")
    rows = summary.get("samples")
    if not isinstance(rows, list) or len(rows) != int(verification.get("expected_samples", -1)):
        raise ValueError("summary sample list does not cover the expected manifest")
    return rows


def copy_json(source: Path, destination: Path) -> bool:
    if source.is_file():
        shutil.copy2(source, destination)
        return True
    return False


def select_cases(
    rows: list[dict[str, Any]], *, seed: int, random_count: int, tail_count: int, percentile: float
) -> tuple[dict[str, set[str]], dict[str, float | None]]:
    nonempty = [row for row in rows if row.get("status") == "success_nonempty"]
    empty = [row for row in rows if row.get("status") == "success_empty"]
    failed = [row for row in rows if row.get("status") == "failed"]
    ordered = sorted(nonempty, key=lambda row: (int(row["post_nms_count"]), str(row["sample_id"])))
    rng = random.Random(seed)
    random_rows = rng.sample(nonempty, min(random_count, len(nonempty)))
    categories: dict[str, set[str]] = {
        "random_success": {str(row["sample_id"]) for row in random_rows},
        "smallest_nonempty": {str(row["sample_id"]) for row in ordered[:tail_count]},
        "largest": {str(row["sample_id"]) for row in ordered[-tail_count:]},
        "empty": {str(row["sample_id"]) for row in empty},
        "failed": {str(row["sample_id"]) for row in failed},
    }
    mask_values = np.asarray(
        [float(row["mask_area_px"]) for row in rows if str(row.get("mask_area_px", ""))],
        dtype=np.float64,
    )
    depth_values = np.asarray(
        [float(row["valid_target_depth_px"]) for row in rows if str(row.get("valid_target_depth_px", ""))],
        dtype=np.float64,
    )
    mask_threshold = float(np.percentile(mask_values, percentile)) if mask_values.size else None
    depth_threshold = float(np.percentile(depth_values, percentile)) if depth_values.size else None
    categories["very_small_mask"] = {
        str(row["sample_id"])
        for row in rows
        if mask_threshold is not None
        and str(row.get("mask_area_px", ""))
        and float(row["mask_area_px"]) <= mask_threshold
    }
    categories["sparse_valid_depth"] = {
        str(row["sample_id"])
        for row in rows
        if depth_threshold is not None
        and str(row.get("valid_target_depth_px", ""))
        and float(row["valid_target_depth_px"]) <= depth_threshold
    }
    # Explicit minimum, lower median, and maximum candidate-count cases.
    if ordered:
        categories["candidate_count_minimum"] = {str(ordered[0]["sample_id"])}
        categories["candidate_count_median"] = {str(ordered[(len(ordered) - 1) // 2]["sample_id"])}
        categories["candidate_count_maximum"] = {str(ordered[-1]["sample_id"])}
    return categories, {
        "very_small_mask_max_px": mask_threshold,
        "sparse_valid_depth_max_px": depth_threshold,
    }


def _input_arrays(mask_root: Path, sample_id: str, sample_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = mask_root / sample_id
    rgb = np.asarray(Image.open(bundle / "color.png").convert("RGB"), dtype=np.uint8)
    predicted_mask = np.asarray(Image.open(bundle / "target_mask.png").convert("L"), dtype=np.uint8) > 0
    if (sample_dir / "depth_m.npy").is_file():
        depth = np.load(sample_dir / "depth_m.npy", allow_pickle=False).astype(np.float32)
    else:
        depth = np.asarray(Image.open(bundle / "depth.png"), dtype=np.float32) / 1000.0
    return rgb, predicted_mask, depth


def _placeholder(path: Path, sample_id: str, reason: str) -> None:
    image = Image.new("RGB", (960, 540), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.multiline_text(
        (30, 30),
        f"{sample_id}\nVisualization unavailable\n{reason}",
        fill=(90, 20, 20),
        spacing=8,
    )
    image.save(path)


def render_case(
    mask_root: Path, sample_dir: Path, destination: Path, row: dict[str, Any]
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    destination.mkdir(parents=True)
    artifact_names: list[str] = []
    errors: list[str] = []
    for name in (
        "raw_candidates.json",
        "filtered_candidates.json",
        "topk_candidates.json",
        "metadata.json",
        "rejection_summary.json",
        "failure.json",
    ):
        try:
            if copy_json(sample_dir / name, destination / name):
                artifact_names.append(name)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")

    image_names = (
        "rgb.png",
        "predicted_hifics_mask.png",
        "depth.png",
        "raw_candidates.png",
        "filtered_candidates.png",
        "topk_candidates.png",
    )
    try:
        rgb, predicted_mask, depth = _input_arrays(mask_root, sample_id, sample_dir)
    except Exception as error:
        reason = f"input unreadable: {type(error).__name__}: {error}"
        errors.append(reason)
        for name in image_names:
            _placeholder(destination / name, sample_id, reason)
            artifact_names.append(name)
        record = {"summary": row, "status": str(row.get("status")), "render_errors": errors}
        (destination / "case.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_names.append("case.json")
        return {"artifacts": sorted(artifact_names), "render_errors": errors}

    Image.fromarray(rgb, mode="RGB").save(destination / "rgb.png")
    Image.fromarray(predicted_mask.astype(np.uint8) * 255, mode="L").save(
        destination / "predicted_hifics_mask.png"
    )
    artifact_names.extend(("rgb.png", "predicted_hifics_mask.png"))
    status = str(row.get("status"))
    stages = (
        ("raw_candidates.json", "raw_candidates.png", "Raw official antipodal candidates"),
        ("filtered_candidates.json", "filtered_candidates.png", "Target-valid post-NMS candidates"),
        ("topk_candidates.json", "topk_candidates.png", "Post-NMS Top-K candidates"),
    )
    for source_name, image_name, title in stages:
        try:
            source = sample_dir / source_name
            candidates = load_json(source) if source.is_file() else []
            save_candidate_overlay(
                rgb,
                candidates,
                destination / image_name,
                mask=predicted_mask,
                title=f"{sample_id}: {title}",
                show_scores=False,
            )
        except Exception as error:
            reason = f"{source_name}: {type(error).__name__}: {error}"
            errors.append(reason)
            _placeholder(destination / image_name, sample_id, reason)
        artifact_names.append(image_name)
    topk_path = sample_dir / "topk_candidates.json"
    topk = load_json(topk_path) if topk_path.is_file() else []
    try:
        save_depth_visualization(
            depth,
            destination / "depth.png",
            candidates=topk,
            mask=predicted_mask,
            title=f"{sample_id}: metric depth + Top-K",
        )
    except Exception as error:
        reason = f"depth visualization: {type(error).__name__}: {error}"
        errors.append(reason)
        _placeholder(destination / "depth.png", sample_id, reason)
    artifact_names.append("depth.png")
    (destination / "case.json").write_text(
        json.dumps(
            {"summary": row, "status": status, "render_errors": errors},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_names.append("case.json")
    return {"artifacts": sorted(artifact_names), "render_errors": errors}


def html_index(cases: list[dict[str, Any]], thresholds: dict[str, Any]) -> str:
    cards = []
    for case in cases:
        sample_id = str(case["sample_id"])
        categories = ", ".join(case["categories"])
        artifacts = set(case.get("artifacts", []))
        links = []
        for name, label in (
            ("raw_candidates.json", "raw JSON"),
            ("filtered_candidates.json", "filtered JSON"),
            ("topk_candidates.json", "Top-K JSON"),
            ("metadata.json", "metadata"),
            ("rejection_summary.json", "rejections"),
            ("failure.json", "failure"),
            ("case.json", "case record"),
        ):
            if name in artifacts:
                links.append(f"<a href='{sample_id}/{name}'>{label}</a>")
        error_note = ""
        if case.get("render_errors"):
            error_note = "<p><b>Render warnings:</b> " + html.escape(
                " | ".join(str(value) for value in case["render_errors"])
            ) + "</p>"
        cards.append(
            f"<article><h2>{html.escape(sample_id)}</h2>"
            f"<p><b>Status:</b> {html.escape(str(case['status']))} · "
            f"<b>Categories:</b> {html.escape(categories)} · "
            f"<b>post-NMS:</b> {html.escape(str(case.get('post_nms_count', '')))}</p>"
            f"<p>{html.escape(str(case.get('query', '')))}</p>"
            f"<div class='grid'>"
            f"<figure><img src='{sample_id}/rgb.png'><figcaption>RGB</figcaption></figure>"
            f"<figure><img src='{sample_id}/predicted_hifics_mask.png'><figcaption>HiFi-CS mask</figcaption></figure>"
            f"<figure><img src='{sample_id}/depth.png'><figcaption>Depth + Top-K</figcaption></figure>"
            f"<figure><img src='{sample_id}/raw_candidates.png'><figcaption>Raw</figcaption></figure>"
            f"<figure><img src='{sample_id}/filtered_candidates.png'><figcaption>Filtered</figcaption></figure>"
            f"<figure><img src='{sample_id}/topk_candidates.png'><figcaption>Top-K</figcaption></figure>"
            f"</div>{error_note}<p>{' · '.join(links)}</p></article>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'>
<title>Dex-Net Full HiFi-CS Review</title><style>
body{font-family:system-ui;margin:2rem;background:#f5f5f5;color:#202020}article{background:white;padding:1rem;margin:1rem 0;border-radius:.5rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.75rem}img{width:100%;height:auto}figure{margin:0}figcaption{font-size:.85rem;color:#555}code{white-space:pre-wrap}
</style></head><body><h1>Dex-Net Full HiFi-CS Review</h1><p>Deterministic selection thresholds: <code>""" + html.escape(json.dumps(thresholds, sort_keys=True)) + "</code></p>" + "".join(cards) + "</body></html>"


def main() -> int:
    args = parse_args()
    candidate_root = args.candidate_root.expanduser().resolve()
    review_root = args.review_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    validate_review_paths(candidate_root, review_root)
    rows = verified_summary_rows(
        load_json(candidate_root / "summary.json"), allow_failures=args.allow_failures
    )
    categories, thresholds = select_cases(
        rows,
        seed=args.seed,
        random_count=args.random_successes,
        tail_count=args.tail_count,
        percentile=args.small_percentile,
    )
    selected = sorted(set().union(*categories.values()))
    by_id = {str(row["sample_id"]): row for row in rows}
    if review_root.exists() and not args.overwrite_existing:
        raise FileExistsError(f"review root exists: {review_root}")
    staging = review_root.parent / f".{review_root.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    try:
        for sample_id in selected:
            row = dict(by_id[sample_id])
            row["categories"] = sorted(name for name, members in categories.items() if sample_id in members)
            render_result = render_case(
                mask_root, candidate_root / sample_id, staging / sample_id, row
            )
            row.update(render_result)
            cases.append(row)
        manifest = {
            "seed": args.seed,
            "selection_thresholds": thresholds,
            "category_counts": {name: len(members) for name, members in categories.items()},
            "selected_count": len(cases),
            "cases": cases,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "index.html").write_text(html_index(cases, thresholds), encoding="utf-8")
        if review_root.exists():
            backup = review_root.parent / f".{review_root.name}.backup.{uuid.uuid4().hex}"
            os.replace(review_root, backup)
            os.replace(staging, review_root)
            try:
                shutil.rmtree(backup)
            except OSError:
                pass
        else:
            os.replace(staging, review_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": "REVIEW_READY", "selected": len(cases), "output": str(review_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
