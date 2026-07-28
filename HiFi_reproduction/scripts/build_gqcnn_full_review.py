#!/usr/bin/env python3
"""Build a bounded, deterministic visual review bundle for full GQ-CNN scores."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.gqcnn_full_scoring import atomic_write_csv, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-ten-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--extreme-count", type=int, default=10)
    parser.add_argument("--many-count", type=int, default=10)
    parser.add_argument("--max-one-candidate", type=int, default=100)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_ranked(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("metadata", {})), list(payload.get("candidates", []))


def depth_image(source_dir: Path) -> Image.Image:
    depth = np.load(source_dir / "depth_m.npy", allow_pickle=False).astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size:
        lo, hi = np.percentile(finite, [2, 98])
        gray = np.clip((depth - lo) / max(float(hi - lo), 1e-9), 0, 1)
    else:
        gray = np.zeros_like(depth)
    return Image.fromarray(np.uint8(gray * 255), mode="L").convert("RGB")


def base_image(source_dir: Path, metadata: dict) -> Image.Image:
    bundle = Path(str(metadata.get("input_bundle", "")))
    color = bundle / "color.png"
    if color.is_file():
        return Image.open(color).convert("RGB")
    return depth_image(source_dir)


def masked_image(image: Image.Image, source_dir: Path) -> Image.Image:
    mask = Image.open(source_dir / "hifics_mask_processed.png").convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    tint = Image.new("RGB", image.size, (0, 255, 110))
    alpha = mask.point(lambda value: 55 if value else 0)
    return Image.composite(Image.blend(image, tint, 0.35), image, alpha)


def draw_grasps(image: Image.Image, records: list[dict], limit: int) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = [(255, 60, 45), (255, 190, 0), (0, 220, 255), (190, 90, 255), (60, 255, 80)]
    for index, record in enumerate(records[:limit]):
        endpoints = record.get("endpoints_uv") or [
            record.get("endpoint_1_uv"), record.get("endpoint_2_uv")
        ]
        if not endpoints or endpoints[0] is None or endpoints[1] is None:
            continue
        p1 = tuple(float(value) for value in endpoints[0])
        p2 = tuple(float(value) for value in endpoints[1])
        color = colors[index % len(colors)]
        width = 5 if index == 0 else 3
        draw.line([p1, p2], fill=color, width=width)
        radius = 4 if index == 0 else 3
        for u, v in (p1, p2):
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)
        label = "#%s %s q=%.8g" % (
            record.get("gqcnn_rank"),
            record.get("candidate_id"),
            float(record["gqcnn_q_value"]),
        )
        label_y = 8 + index * 15
        bounds = draw.textbbox((8, label_y), label, font=ImageFont.load_default())
        draw.rectangle((bounds[0] - 2, bounds[1] - 1, bounds[2] + 2, bounds[3] + 1), fill=(0, 0, 0))
        draw.text((8, label_y), label, fill=color, font=ImageFont.load_default())
    return canvas


def score_panel(image: Image.Image, records: list[dict], title: str) -> Image.Image:
    panel_height = max(150, 24 * min(10, len(records)) + 50)
    panel = Image.new("RGB", (image.width, panel_height), (24, 27, 33))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(240, 240, 240), font=font)
    shown = records[:10]
    max_q = max([float(row["gqcnn_q_value"]) for row in shown] or [1.0])
    for index, row in enumerate(shown):
        y = 32 + index * 22
        q_value = float(row["gqcnn_q_value"])
        draw.text((10, y), "%2d %s %.8g" % (index + 1, row["candidate_id"], q_value), fill=(230, 230, 230), font=font)
        bar_x = min(250, max(135, image.width // 2))
        bar_width = int(max(0.0, q_value) / max(max_q, 1e-15) * max(20, image.width - bar_x - 15))
        draw.rectangle((bar_x, y + 2, bar_x + bar_width, y + 13), fill=(67, 176, 255))
    combined = Image.new("RGB", (image.width, image.height + panel.height))
    combined.paste(image, (0, 0))
    combined.paste(panel, (0, image.height))
    return combined


def add_category(target: dict[str, set[str]], rows: list[dict], category: str) -> None:
    for row in rows:
        target[row["sample_id"]].add(category)


def select_review(rows: list[dict], args: argparse.Namespace) -> tuple[dict[str, set[str]], dict]:
    selected: dict[str, set[str]] = defaultdict(set)
    nonempty = [row for row in rows if row["scoring_status"] == "scored_nonempty"]
    empty = [row for row in rows if row["scoring_status"] == "skipped_valid_empty"]
    rng = random.Random(args.seed)
    random_rows = rng.sample(nonempty, min(args.random_count, len(nonempty)))
    add_category(selected, random_rows, "random_nonempty")
    add_category(selected, sorted(nonempty, key=lambda row: float(row["top1_q_value"]), reverse=True)[: args.extreme_count], "highest_top1_q")
    add_category(selected, sorted(nonempty, key=lambda row: float(row["top1_q_value"]))[: args.extreme_count], "lowest_top1_q")
    add_category(selected, sorted(nonempty, key=lambda row: float(row["q_value_range"]), reverse=True)[: args.extreme_count], "largest_q_spread")
    add_category(selected, sorted(nonempty, key=lambda row: float(row["q_value_range"]))[: args.extreme_count], "smallest_q_spread")
    one_all = [row for row in nonempty if int(row["source_candidate_count"]) == 1]
    one_kept = one_all[: args.max_one_candidate]
    add_category(selected, one_kept, "one_candidate")
    add_category(selected, sorted(nonempty, key=lambda row: int(row["source_candidate_count"]), reverse=True)[: args.many_count], "many_candidates")
    add_category(selected, empty, "valid_empty")
    previous_ids: list[str] = []
    if args.previous_ten_root and args.previous_ten_root.is_dir():
        previous_ids = sorted(path.name for path in args.previous_ten_root.glob("q*") if path.is_dir())
        by_id = {row["sample_id"]: row for row in rows}
        add_category(selected, [by_id[value] for value in previous_ids if value in by_id], "previous_ten")
    details = {
        "seed": args.seed,
        "category_counts_before_deduplication": {
            "random_nonempty": len(random_rows),
            "highest_top1_q": min(args.extreme_count, len(nonempty)),
            "lowest_top1_q": min(args.extreme_count, len(nonempty)),
            "largest_q_spread": min(args.extreme_count, len(nonempty)),
            "smallest_q_spread": min(args.extreme_count, len(nonempty)),
            "one_candidate_kept": len(one_kept),
            "one_candidate_total": len(one_all),
            "many_candidates": min(args.many_count, len(nonempty)),
            "valid_empty": len(empty),
            "previous_ten": len(previous_ids),
        },
        "all_one_candidate_sample_ids": [row["sample_id"] for row in one_all],
    }
    return selected, details


def main() -> int:
    args = parse_args()
    candidate_root = args.candidate_root.expanduser().resolve()
    scored_root = args.scored_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(scored_root / "summary.csv")
    selected, details = select_review(rows, args)
    by_id = {row["sample_id"]: row for row in rows}
    manifest_rows = []
    html_cards = []
    for sample_id in sorted(selected):
        source_dir = candidate_root / sample_id
        sample_out = output_dir / "samples" / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        image = masked_image(base_image(source_dir, source_metadata), source_dir)
        row = by_id[sample_id]
        status = row["scoring_status"]
        records: list[dict] = []
        if status == "scored_nonempty":
            _, records = load_ranked(scored_root / sample_id / "gqcnn_scored_candidates.json")
            top1 = draw_grasps(image, records, 1)
            top5 = draw_grasps(depth_image(source_dir), records, 5)
            panel = score_panel(top5, records, "%s | %s" % (sample_id, source_metadata.get("query", "")))
        else:
            top1 = image.copy()
            top5 = image.copy()
            draw = ImageDraw.Draw(top1)
            draw.text((10, 10), "VALID EMPTY: %s" % source_metadata.get("failure_reason"), fill=(255, 80, 60), font=ImageFont.load_default())
            top5 = top1.copy()
            panel = score_panel(top5, [], "%s | valid empty" % sample_id)
        top1.save(sample_out / "top1.png")
        top5.save(sample_out / "top5.png")
        panel.save(sample_out / "score_panel.png")
        review = {
            "sample_id": sample_id,
            "query": source_metadata.get("query", ""),
            "categories": sorted(selected[sample_id]),
            "scoring_status": status,
            "source_candidate_count": int(row["source_candidate_count"]),
            "top1_candidate_id": row.get("top1_candidate_id") or None,
            "top1_q_value": None if not row.get("top1_q_value") else float(row["top1_q_value"]),
            "top5_candidate_ids": [record["candidate_id"] for record in records[:5]],
            "full_ranked_output": str(scored_root / sample_id / "gqcnn_scored_candidates.json") if records else None,
        }
        atomic_write_json(sample_out / "review.json", review)
        manifest_rows.append({
            "sample_id": sample_id,
            "categories": sorted(selected[sample_id]),
            "scoring_status": status,
            "candidate_count": int(row["source_candidate_count"]),
            "top1_q_value": review["top1_q_value"],
            "relative_directory": "samples/%s" % sample_id,
        })
        html_cards.append(
            '<article><h3>{}</h3><p>{}</p><p>{}</p><a href="samples/{}/review.json">JSON</a><img src="samples/{}/top5.png" loading="lazy"><img src="samples/{}/score_panel.png" loading="lazy"></article>'.format(
                html.escape(sample_id), html.escape(", ".join(sorted(selected[sample_id]))),
                html.escape(str(source_metadata.get("query", ""))), sample_id, sample_id, sample_id
            )
        )
    details["selected_unique_samples"] = len(manifest_rows)
    details["candidate_root"] = str(candidate_root)
    details["scored_root"] = str(scored_root)
    atomic_write_json(output_dir / "selection_manifest.json", {"selection": details, "samples": manifest_rows})
    atomic_write_csv(output_dir / "selection_manifest.csv", manifest_rows, ("sample_id", "categories", "scoring_status", "candidate_count", "top1_q_value", "relative_directory"))
    page = """<!doctype html><html><head><meta charset="utf-8"><title>GQ-CNN full review</title><style>body{{font-family:system-ui;background:#101217;color:#eee;margin:24px}}article{{background:#1b1f27;padding:16px;margin:18px 0;border-radius:10px}}img{{max-width:48%;height:auto;margin:8px}}a{{color:#62b5ff}}</style></head><body><h1>GQ-CNN full deterministic review</h1><p>{} unique samples. Selection details are in selection_manifest.json.</p>{}</body></html>""".format(len(manifest_rows), "\n".join(html_cards))
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(details, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
