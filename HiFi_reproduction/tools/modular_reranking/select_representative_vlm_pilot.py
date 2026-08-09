#!/usr/bin/env python3
"""Select a deterministic, scene-diverse VLM pilot; keep GT strata offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional offline table with sample_id, hifi_mask_iou, target_area_px",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--scene-cap", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def _table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _stable_order(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def _quartile(values: pd.Series, prefix: str) -> pd.Series:
    ranks = values.rank(method="first", pct=True)
    return pd.cut(
        ranks,
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        labels=[f"{prefix}_q1", f"{prefix}_q2", f"{prefix}_q3", f"{prefix}_q4"],
        include_lowest=True,
    ).astype(str)


def main() -> int:
    args = parse_args()
    samples = _table(args.per_sample.resolve()).copy()
    candidates = _table(args.per_candidate.resolve())
    required_sample = {
        "sample_id",
        "scene_id",
        "split",
        "query_type",
        "candidate_count",
        "q_top1_gap",
        "mask_area_px",
    }
    required_candidate = {
        "sample_id",
        "split",
        "original_gqcnn_rank",
        "candidate_positive",
    }
    if missing := sorted(required_sample - set(samples)):
        raise ValueError(f"per-sample missing columns: {missing}")
    if missing := sorted(required_candidate - set(candidates)):
        raise ValueError(f"per-candidate missing columns: {missing}")
    if (
        set(samples["split"].astype(str)) - {"val", "validation"}
        or set(candidates["split"].astype(str)) - {"val", "validation"}
    ):
        raise ValueError("VLM pilot selection may consume validation rows only")
    grouped = candidates.groupby("sample_id", sort=False)
    outcome = grouped.apply(
        lambda group: pd.Series(
            {
                "q_top1_correct": bool(
                    group.loc[
                        group["original_gqcnn_rank"] == 1, "candidate_positive"
                    ].iloc[0]
                ),
                "top5_positive": bool(
                    group.loc[
                        group["original_gqcnn_rank"] <= 5, "candidate_positive"
                    ].any()
                ),
            }
        ),
        include_groups=False,
    ).reset_index()
    samples = samples.merge(outcome, on="sample_id", how="left", validate="one_to_one")
    valid_empty = samples["candidate_count"].fillna(0).astype(int).eq(0)
    samples["q_top1_correct"] = samples["q_top1_correct"].eq(True)
    samples["top5_positive"] = samples["top5_positive"].eq(True)
    samples["outcome_stratum"] = "top5_all_negative"
    samples.loc[samples["q_top1_correct"], "outcome_stratum"] = "q_top1_correct"
    samples.loc[
        ~samples["q_top1_correct"] & samples["top5_positive"], "outcome_stratum"
    ] = "q_wrong_top5_positive"
    samples.loc[valid_empty, "outcome_stratum"] = "valid_empty_no_candidates"
    samples["q_margin_stratum"] = _quartile(samples["q_top1_gap"], "q_margin")
    samples["target_size_stratum"] = _quartile(
        samples["mask_area_px"], "predicted_target_area"
    )
    samples["clutter_stratum"] = _quartile(
        samples["candidate_count"], "candidate_clutter"
    )
    if args.diagnostics is not None:
        diagnostics = _table(args.diagnostics.resolve())
        allowed = {"sample_id", "hifi_mask_iou", "target_area_px"}
        if set(diagnostics) - allowed or "sample_id" not in diagnostics:
            raise ValueError(
                "diagnostics may contain only sample_id, hifi_mask_iou, target_area_px"
            )
        samples = samples.merge(
            diagnostics, on="sample_id", how="left", validate="one_to_one"
        )
        if "hifi_mask_iou" in samples:
            samples["mask_iou_stratum"] = _quartile(
                samples["hifi_mask_iou"], "mask_iou"
            )
        if "target_area_px" in samples:
            samples["gt_target_size_stratum"] = _quartile(
                samples["target_area_px"], "gt_target_area"
            )
    stratum_columns = [
        "query_type",
        "outcome_stratum",
        "q_margin_stratum",
        "target_size_stratum",
        "clutter_stratum",
        *(["mask_iou_stratum"] if "mask_iou_stratum" in samples else []),
        *(
            ["gt_target_size_stratum"]
            if "gt_target_size_stratum" in samples
            else []
        ),
    ]
    samples["_tie"] = samples["sample_id"].map(
        lambda value: _stable_order(str(value), args.seed)
    )
    strata: dict[str, list[str]] = {}
    for column in stratum_columns:
        for value, group in samples.groupby(column, dropna=False):
            key = f"{column}={value}"
            strata[key] = group.sort_values("_tie")["sample_id"].astype(str).tolist()
    selected: list[str] = []
    selected_set: set[str] = set()
    scene_counts: dict[str, int] = {}
    by_id = samples.set_index("sample_id")
    while len(selected) < args.count:
        progress = False
        for key in sorted(strata):
            for sample_id in strata[key]:
                if sample_id in selected_set:
                    continue
                scene = str(by_id.loc[sample_id, "scene_id"])
                if scene_counts.get(scene, 0) >= args.scene_cap:
                    continue
                selected.append(sample_id)
                selected_set.add(sample_id)
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
                progress = True
                break
            if len(selected) >= args.count:
                break
        if not progress:
            break
    if len(selected) != args.count:
        raise RuntimeError(
            f"could select only {len(selected)} samples under scene cap {args.scene_cap}"
        )
    selected_frame = samples.set_index("sample_id").loc[selected].reset_index()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "selection_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}, sort_keys=True) + "\n"
            for sample_id in selected
        ),
        encoding="utf-8",
    )
    selected_frame.drop(columns=["_tie"]).to_parquet(
        root / "offline_selection_audit.parquet", index=False
    )
    coverage = {
        column: selected_frame[column].value_counts(dropna=False).to_dict()
        for column in stratum_columns
    }
    summary = {
        "schema_version": 1,
        "selected_count": len(selected),
        "scene_count": selected_frame["scene_id"].nunique(),
        "scene_cap": args.scene_cap,
        "seed": args.seed,
        "selection_split": "validation",
        "formal_test_evidence_used": False,
        "selection_manifest_contains_gt": False,
        "offline_audit_contains_gt_derived_selection_strata": True,
        "coverage": coverage,
    }
    temporary = root / f".summary.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
