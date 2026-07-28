"""Fairly compare LAVT and local HiFi-CS masks with one raw-GT evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ocid_vlg.engine import save_json
from ocid_vlg.metrics import aggregate_metrics, compute_sample_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lavt-manifest", type=Path, required=True)
    parser.add_argument("--hifics-export-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) != 0


def _bootstrap_scene_delta(
    rows: list[dict], replicates: int, seed: int
) -> dict[str, float]:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row["iou_delta"])
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        values = [value for scene in sampled for value in by_scene[str(scene)]]
        estimates[index] = float(np.mean(values))
    return {
        "cluster_key": "scene_id",
        "replicates": replicates,
        "seed": seed,
        "mean_delta": float(np.mean(estimates)),
        "ci95_lower": float(np.percentile(estimates, 2.5)),
        "ci95_upper": float(np.percentile(estimates, 97.5)),
    }


def _render_comparison(
    row: dict,
    *,
    lavt_row: dict,
    hifics_row: dict,
    gt_row: dict,
    hifics_root: Path,
    destination: Path,
) -> None:
    with Image.open(gt_row["image_path"]) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    with Image.open(gt_row["mask_path"]) as handle:
        instance_mask = np.asarray(handle)
    gt = instance_mask == int(gt_row["objID"])
    probability = np.load(lavt_row["target_probability"])
    lavt_mask = _mask(Path(lavt_row["target_mask"]))
    hifics_mask = _mask(
        hifics_root
        / hifics_row["directory"]
        / "predicted_mask_original_resolution.png"
    )
    overlay = rgb.astype(np.float32).copy()
    overlay[lavt_mask] = (
        0.45 * overlay[lavt_mask]
        + 0.55 * np.array([255, 64, 64], dtype=np.float32)
    )
    figure, axes = plt.subplots(1, 7, figsize=(29, 4.5))
    entries = (
        (rgb, "RGB", None),
        (gt, "Original GT", "gray"),
        (probability, "LAVT foreground probability", "viridis"),
        (lavt_mask, "LAVT binary mask", "gray"),
        (hifics_mask, "HiFi-CS binary mask", "gray"),
        (overlay.astype(np.uint8), "LAVT overlay", None),
        (
            np.logical_xor(lavt_mask, hifics_mask),
            "LAVT/HiFi disagreement",
            "magma",
        ),
    )
    for axis, (image, title, cmap) in zip(axes, entries):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{row['sentence']}\nLAVT IoU={row['lavt_iou']:.4f} | "
        f"HiFi-CS IoU={row['hifics_iou']:.4f} | delta={row['iou_delta']:+.4f} | "
        f"sent_id={row['sent_id']} | scene={row['scene_id']}",
        fontsize=10,
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _comparison_figures(
    rows: list[dict],
    *,
    lavt_by_id: dict[str, dict],
    hifics_by_id: dict[str, dict],
    gt_by_id: dict[str, dict],
    hifics_root: Path,
    output_dir: Path,
) -> dict[str, int]:
    ranked = sorted(rows, key=lambda row: (row["iou_delta"], row["sent_id"]))
    groups = {
        "hifics_wins": ranked[: min(10, len(ranked))],
        "lavt_wins": list(reversed(ranked[-min(10, len(ranked)) :])),
    }
    for group, selected in groups.items():
        for index, row in enumerate(selected):
            sent_id = row["sent_id"]
            _render_comparison(
                row,
                lavt_row=lavt_by_id[sent_id],
                hifics_row=hifics_by_id[sent_id],
                gt_row=gt_by_id[sent_id],
                hifics_root=hifics_root,
                destination=(
                    output_dir / "figures" / group / f"{index:02d}_{sent_id}.png"
                ),
            )
    return {name: len(values) for name, values in groups.items()}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_inputs = {
        "lavt_prediction_manifest": args.lavt_manifest,
        "hifics_prediction_manifest": args.hifics_export_manifest,
        "test_manifest": args.test_manifest,
    }
    missing_inputs = {
        name: str(path.resolve())
        for name, path in required_inputs.items()
        if not path.is_file()
    }
    if missing_inputs:
        save_json(
            args.output_dir / "comparison_hifics.json",
            {
                "status": "COMPARISON_NOT_FAIR",
                "reason": "required prediction or alignment manifest is missing",
                "missing_inputs": missing_inputs,
            },
        )
        return 0
    lavt_rows = _jsonl(args.lavt_manifest)
    lavt_by_id = {row["sent_id"]: row for row in lavt_rows}
    hifics_document = json.loads(args.hifics_export_manifest.read_text())
    hifics_rows = hifics_document["samples"]
    hifics_by_id = {row["stable_sample_id"]: row for row in hifics_rows}
    gt_rows = _jsonl(args.test_manifest)
    gt_by_id = {row["sent_id"]: row for row in gt_rows}

    conditions = {
        "lavt_unique_sent_ids": len(lavt_by_id) == len(lavt_rows),
        "hifics_unique_sent_ids": len(hifics_by_id) == len(hifics_rows),
        "test_unique_sent_ids": len(gt_by_id) == len(gt_rows),
        "sent_id_sets_equal": set(lavt_by_id) == set(hifics_by_id) == set(gt_by_id),
        "sample_counts_equal": len(lavt_rows) == len(hifics_rows) == len(gt_rows),
        "no_missing_predictions": all(
            Path(row["target_mask"]).is_file() for row in lavt_rows
        )
        and all(
            (
                args.hifics_export_manifest.parent.parent
                / row["directory"]
                / "predicted_mask_original_resolution.png"
            ).is_file()
            for row in hifics_rows
        ),
    }
    alignment_errors: list[str] = []
    for sent_id in sorted(set(lavt_by_id) & set(hifics_by_id) & set(gt_by_id)):
        lavt = lavt_by_id[sent_id]
        hifi = hifics_by_id[sent_id]
        gt = gt_by_id[sent_id]
        if not (
            lavt["scene_id"] == hifi["scene_id"] == gt["scene_id"]
            and lavt["raw_sentence"] == hifi["query"] == gt["sentence"]
        ):
            alignment_errors.append(sent_id)
    conditions["scene_and_expression_equal"] = not alignment_errors
    conditions["gt_target_objid_present"] = all("objID" in row for row in gt_rows)
    conditions["target_identity_inferred_from_frozen_id_alignment"] = (
        not alignment_errors and conditions["sent_id_sets_equal"]
    )
    if not all(conditions.values()):
        save_json(
            args.output_dir / "comparison_hifics.json",
            {
                "status": "COMPARISON_NOT_FAIR",
                "conditions": conditions,
                "alignment_errors": alignment_errors[:100],
            },
        )
        return 0

    lavt_metrics: list[dict] = []
    hifi_metrics: list[dict] = []
    comparison_rows: list[dict] = []
    for sent_id in sorted(gt_by_id):
        gt_row = gt_by_id[sent_id]
        full_instance = np.asarray(Image.open(gt_row["mask_path"]))
        gt = full_instance == int(gt_row["objID"])
        lavt_prediction = _mask(Path(lavt_by_id[sent_id]["target_mask"]))
        hifi_prediction = _mask(
            args.hifics_export_manifest.parent.parent
            / hifics_by_id[sent_id]["directory"]
            / "predicted_mask_original_resolution.png"
        )
        lavt_metric = compute_sample_metrics(lavt_prediction, gt)
        hifi_metric = compute_sample_metrics(hifi_prediction, gt)
        lavt_metrics.append(lavt_metric)
        hifi_metrics.append(hifi_metric)
        delta = lavt_metric["iou"] - hifi_metric["iou"]
        comparison_rows.append(
            {
                "sent_id": sent_id,
                "scene_id": gt_row["scene_id"],
                "sentence": gt_row["sentence"],
                "objID": gt_row["objID"],
                "lavt_iou": lavt_metric["iou"],
                "hifics_iou": hifi_metric["iou"],
                "iou_delta": delta,
                "winner": "lavt"
                if delta > 0
                else "hifics"
                if delta < 0
                else "tie",
            }
        )

    deltas = np.asarray([row["iou_delta"] for row in comparison_rows])
    lavt_aggregate = aggregate_metrics(lavt_metrics)
    hifi_aggregate = aggregate_metrics(hifi_metrics)
    result = {
        "status": "FAIR_IDENTICAL_SPLIT",
        "conditions": conditions,
        "target_identity_evidence": (
            "The stable ID is derived from scene_id and source question_index; "
            "the full stable-ID sets, scene IDs, expressions, and raw-source "
            "objID-bearing rows are aligned. HiFi export does not duplicate objID."
        ),
        "N": len(comparison_rows),
        "lavt": lavt_aggregate,
        "hifics": hifi_aggregate,
        "delta": {
            "mean_iou": (
                lavt_aggregate["mean_iou"] - hifi_aggregate["mean_iou"]
            ),
            "overall_iou": (
                lavt_aggregate["overall_iou"] - hifi_aggregate["overall_iou"]
            ),
            **{
                f"p_at_{threshold}": (
                    lavt_aggregate[f"p_at_{threshold}"]
                    - hifi_aggregate[f"p_at_{threshold}"]
                )
                for threshold in (50, 60, 70, 80, 90)
            },
            "median_sample_iou": float(np.median(deltas)),
            "lavt_wins": sum(row["winner"] == "lavt" for row in comparison_rows),
            "ties": sum(row["winner"] == "tie" for row in comparison_rows),
            "hifics_wins": sum(
                row["winner"] == "hifics" for row in comparison_rows
            ),
        },
        "bootstrap_95_ci": _bootstrap_scene_delta(
            comparison_rows, args.bootstrap_replicates, args.seed
        ),
    }
    result["qualitative_figure_counts"] = _comparison_figures(
        comparison_rows,
        lavt_by_id=lavt_by_id,
        hifics_by_id=hifics_by_id,
        gt_by_id=gt_by_id,
        hifics_root=args.hifics_export_manifest.parent.parent,
        output_dir=args.output_dir,
    )
    save_json(args.output_dir / "comparison_hifics.json", result)
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "comparison_hifics_per_sample.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
