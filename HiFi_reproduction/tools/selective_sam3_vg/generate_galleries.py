#!/usr/bin/env python3
"""Generate six audited qualitative HTML galleries after formal evaluation."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import load_binary_mask, resize_binary_mask  # noqa: E402
from segmentation.selective_sam3_vg.metrics import evaluator_iou_float32  # noqa: E402


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    result[mask] = result[mask] * 0.45 + np.asarray(color, dtype=np.float32) * 0.55
    return np.clip(result, 0, 255).astype(np.uint8)


def _candidate_oracle(formal: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_lookup = baseline.set_index("sample_id")
    rows = []
    triggered = formal[formal["sam_invoked"]]
    for number, row in enumerate(triggered.itertuples(index=False), start=1):
        directory = ROOT / "outputs/selective_sam3_vg/formal_test_masks" / row.sample_id
        archive = np.load(directory / "all_sam_hypotheses.npz", allow_pickle=False)
        try:
            masks = archive["masks"]
        finally:
            archive.close()
        source = baseline_lookup.loc[row.sample_id]
        gt = load_binary_mask(source["gt_mask_path"], expected_shape=(352, 352))
        values = [evaluator_iou_float32(resize_binary_mask(mask.astype(bool), gt.shape), gt) for mask in masks]
        areas = [int(np.count_nonzero(mask)) for mask in masks]
        coarse_area = int(np.count_nonzero(load_binary_mask(directory / "coarse_mask.png")))
        best_index = int(np.argmax(values))
        rows.append(
            {
                "sample_id": row.sample_id,
                "best_sam_iou": float(values[best_index]),
                "best_sam_candidate_id": f"sam3_{best_index:03d}",
                "best_sam_delta": float(values[best_index] - row.baseline_iou),
                "maximum_sam_area_ratio": max(areas) / max(coarse_area, 1),
            }
        )
        if number % 500 == 0:
            print(f"gallery candidate audit {number}/{len(triggered)}", flush=True)
    return pd.DataFrame(rows)


def _render_case(row: pd.Series, baseline_row: pd.Series, oracle_row: pd.Series | None, output: Path) -> None:
    directory = ROOT / "outputs/selective_sam3_vg/formal_test_masks" / row["sample_id"]
    rgb = np.asarray(Image.open(baseline_row["rgb_path"]).convert("RGB"), dtype=np.uint8)
    coarse = load_binary_mask(directory / "coarse_mask.png", expected_shape=rgb.shape[:2])
    final = load_binary_mask(directory / "final_mask.png", expected_shape=rgb.shape[:2])
    gt_small = load_binary_mask(baseline_row["gt_mask_path"], expected_shape=(352, 352))
    gt = resize_binary_mask(gt_small, rgb.shape[:2])
    hypotheses: list[np.ndarray] = []
    if (directory / "all_sam_hypotheses.npz").is_file():
        archive = np.load(directory / "all_sam_hypotheses.npz", allow_pickle=False)
        try:
            hypotheses = [np.asarray(mask, dtype=bool) for mask in archive["masks"]]
        finally:
            archive.close()
    panels: list[tuple[str, np.ndarray]] = [
        ("RGB", rgb), ("GT (evaluation only)", _overlay(rgb, gt, (24, 180, 95))),
        ("HiFi coarse", _overlay(rgb, coarse, (220, 60, 60))),
    ]
    panels.extend((f"SAM hypothesis {index}", _overlay(rgb, mask, (50, 120, 230))) for index, mask in enumerate(hypotheses))
    panels.append(("Final selected", _overlay(rgb, final, (190, 70, 210))))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0), squeeze=False)
    for axis, (title, image) in zip(axes[0], panels, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    oracle_text = "N/A" if oracle_row is None else f"{oracle_row['best_sam_iou']:.4f}"
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    fig.suptitle(
        f"{row['sample_id']} | query: {row['query']}\n"
        f"baseline={row['baseline_iou']:.4f}, final={row['hybrid_iou']:.4f}, delta={row['delta_iou']:+.4f}, "
        f"best SAM={oracle_text}, trigger={row['trigger_score']:.4f}, selected={row['selected_source']}, "
        f"reason={provenance['fallback_reason']}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_gallery(
    name: str,
    cases: pd.DataFrame,
    formal: pd.DataFrame,
    baseline: pd.DataFrame,
    oracle: pd.DataFrame,
    gallery_root: Path,
) -> None:
    assets = gallery_root / "assets" / name
    assets.mkdir(parents=True, exist_ok=True)
    formal_lookup = formal.set_index("sample_id")
    baseline_lookup = baseline.set_index("sample_id")
    oracle_lookup = oracle.set_index("sample_id") if len(oracle) else pd.DataFrame()
    cards = []
    for index, sample_id in enumerate(cases["sample_id"].head(20), start=1):
        row = formal_lookup.loc[sample_id].copy()
        # set_index removes the key from the Series returned by .loc, while
        # the renderer needs it to resolve the audited per-sample directory.
        row["sample_id"] = sample_id
        baseline_row = baseline_lookup.loc[sample_id]
        oracle_row = oracle_lookup.loc[sample_id] if len(oracle_lookup) and sample_id in oracle_lookup.index else None
        image_path = assets / f"{index:02d}_{sample_id}.png"
        _render_case(row, baseline_row, oracle_row, image_path)
        cards.append(
            f'<article><h2>{html.escape(sample_id)}</h2><p>{html.escape(str(row["query"]))}</p>'
            f'<img src="assets/{html.escape(name)}/{html.escape(image_path.name)}" alt="{html.escape(sample_id)} comparison"></article>'
        )
    document = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:system-ui,sans-serif;max-width:1800px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#17212b}
article{background:white;padding:1rem;margin:1rem 0;border:1px solid #ddd;border-radius:8px}img{max-width:100%;height:auto}h1{font-size:1.5rem}h2{font-size:1rem}
</style></head><body>""" + f"<h1>{html.escape(name.replace('_', ' ').title())}</h1>" + "".join(cards) + "</body></html>"
    (gallery_root / f"{name}.html").write_text(document, encoding="utf-8")


def main() -> None:
    formal = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/report/formal_per_sample_metrics.parquet")
    baseline = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/baseline_manifest.parquet")
    oracle_candidates = _candidate_oracle(formal, baseline)
    report = ROOT / "outputs/selective_sam3_vg/report"
    oracle_candidates.to_parquet(report / "qualitative_sam_candidate_oracle.parquet", index=False)
    merged = formal.merge(oracle_candidates, on="sample_id", how="left")
    accepted = merged[merged["selected_source"] == "sam3"]
    rejected = merged[merged["sam_invoked"] & (merged["selected_source"] != "sam3")]
    galleries = {
        "best_improvements": accepted.sort_values("delta_iou", ascending=False),
        "harmful_replacements": accepted[accepted["delta_iou"] < 0.0].sort_values("delta_iou"),
        "sam_rejected_correctly": rejected[rejected["best_sam_delta"] <= 0.0].sort_values("best_sam_delta"),
        "sam_rejected_incorrectly": rejected[rejected["best_sam_delta"] >= 0.02].sort_values("best_sam_delta", ascending=False),
        "wrong_target_cases": merged[merged["baseline_iou"] < 0.25].sort_values("baseline_iou"),
        "neighbour_expansion_cases": merged[(merged["maximum_sam_area_ratio"] > 1.5) & (merged["best_sam_delta"] < 0.0)].sort_values("maximum_sam_area_ratio", ascending=False),
    }
    gallery_root = report / "galleries"
    gallery_root.mkdir(parents=True, exist_ok=True)
    for name, cases in galleries.items():
        _write_gallery(name, cases, formal, baseline, oracle_candidates, gallery_root)
    summary = {name: int(len(cases)) for name, cases in galleries.items()}
    (gallery_root / "gallery_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "gallery_case_pools": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
