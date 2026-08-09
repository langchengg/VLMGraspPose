#!/usr/bin/env python3
"""Build readable contact sheets for the required 50-grid pilot inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--record-review", action="store_true")
    parser.add_argument("--review-notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    pilot = pd.read_parquet(experiment / "pilot/pilot_manifest.parquet")
    sort_columns = [
        column
        for column in ("query_type", "baseline_iou_bin", "target_size_bin", "sample_id")
        if column in pilot
    ]
    ordered = pilot.sort_values(sort_columns, kind="stable")
    strata = [
        list(frame.index)
        for _, frame in ordered.groupby(
            [column for column in sort_columns if column != "sample_id"],
            sort=True,
            dropna=False,
        )
    ]
    selected_indices = []
    depth = 0
    while len(selected_indices) < int(args.count):
        added = False
        for indices in strata:
            if depth < len(indices):
                selected_indices.append(indices[depth])
                added = True
                if len(selected_indices) == int(args.count):
                    break
        if not added:
            break
        depth += 1
    selected = pilot.loc[selected_indices].copy()
    if len(selected) < int(args.count):
        raise ValueError(f"only {len(selected)} pilot rows are available")
    output = experiment / "pilot/manual_grid_inspection"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    per_sheet = 10
    tile_width, tile_height = 600, 430
    for sheet_index in range((len(selected) + per_sheet - 1) // per_sheet):
        rows = selected.iloc[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        canvas = Image.new("RGB", (tile_width * 2, tile_height * 5), "white")
        for local_index, row in enumerate(rows.itertuples(index=False)):
            grid_path = experiment / "proposals" / str(row.sample_id) / "proposal_grid.png"
            if not grid_path.is_file():
                raise FileNotFoundError(grid_path)
            grid = Image.open(grid_path).convert("RGB")
            grid.thumbnail((tile_width - 8, tile_height - 48), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_width, tile_height), "white")
            tile.paste(grid, ((tile_width - grid.width) // 2, 44))
            draw = ImageDraw.Draw(tile)
            baseline_band = getattr(
                row, "baseline_iou_band", getattr(row, "baseline_iou_bin", "?")
            )
            size_bin = getattr(
                row, "target_area_bin", getattr(row, "target_size_bin", "?")
            )
            label = f"{row.sample_id} | {getattr(row, 'query_type', '?')} | {baseline_band} | {size_bin}"
            draw.text((6, 5), label[:92], fill="black")
            query = str(getattr(row, "query", ""))
            draw.text((6, 23), query[:92], fill="black")
            tile = ImageOps.expand(tile, border=1, fill="black")
            x = (local_index % 2) * tile_width
            y = (local_index // 2) * tile_height
            canvas.paste(tile.crop((0, 0, tile_width, tile_height)), (x, y))
            records.append(
                {
                    "sample_id": str(row.sample_id),
                    "proposal_grid_path": str(grid_path),
                    "contact_sheet": f"proposal_inspection_{sheet_index + 1:02d}.png",
                    "query_type": str(getattr(row, "query_type", "")),
                    "baseline_iou_bin": str(baseline_band),
                    "target_size_bin": str(size_bin),
                    "manual_status": (
                        "VISUALLY_REVIEWED" if args.record_review else "PENDING_VISUAL_REVIEW"
                    ),
                }
            )
        canvas.save(output / f"proposal_inspection_{sheet_index + 1:02d}.png")
    pd.DataFrame(records).to_csv(output / "inspection_manifest.csv", index=False)
    (output / "inspection_summary.json").write_text(
        json.dumps(
            {
                "requested_grids": int(args.count),
                "rendered_grids": len(records),
                "contact_sheets": (len(records) + per_sheet - 1) // per_sheet,
                "manual_review_complete": bool(args.record_review),
                "manual_review_notes": args.review_notes if args.record_review else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rendered_grids": len(records), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
