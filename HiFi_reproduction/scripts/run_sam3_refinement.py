#!/usr/bin/env python3
"""Run official CUDA SAM 3 refinement over a prepared prediction-only manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_model import OfficialSam3Tracker  # noqa: E402
from src.segmentation.sam3_refiner import refine_sample  # noqa: E402
from src.segmentation.sam3_serialization import (  # noqa: E402
    assert_no_ground_truth_leakage,
    save_strict_json,
    sha256_file,
    write_jsonl,
)


SUMMARY_FIELDS = (
    "sample_id",
    "inference_succeeded",
    "selected_mask_source",
    "selected_hypothesis_id",
    "number_of_sam_hypotheses",
    "sam_model_score",
    "refinement_score",
    "fallback",
    "fallback_reason",
    "runtime_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_refinement_inputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_refined_masks",
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "sam3_refinement.yaml"
    )
    parser.add_argument("--model-id-or-path")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"))
    parser.add_argument("--sample-id", action="append")
    return parser.parse_args()


def _input_rows(root: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or len(rows) != len({row["sample_id"] for row in rows}):
        raise ValueError("input manifest must contain unique samples")
    for row in rows:
        assert_no_ground_truth_leakage(row, context="SAM 3 refinement input manifest")
    return rows


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite refinement outputs: {output_root}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    model_id_or_path = args.model_id_or_path or model_config["model_id"]
    revision = args.revision or model_config.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("SAM 3 inference requires an immutable 40-character model revision SHA")
    local_files_only = (
        bool(model_config["local_files_only"])
        if args.local_files_only is None
        else bool(args.local_files_only)
    )
    precision = args.precision or str(model_config["precision"])
    # This constructor performs CUDA preflight before creating any result directory.
    model = OfficialSam3Tracker(
        model_id_or_path,
        revision=revision,
        local_files_only=local_files_only,
        precision=precision,
    )
    rows = _input_rows(input_root)
    if args.sample_id:
        requested = set(args.sample_id)
        known = {row["sample_id"] for row in rows}
        if requested - known:
            raise KeyError(f"unknown sample IDs: {sorted(requested - known)}")
        rows = [row for row in rows if row["sample_id"] in requested]
    temporary = output_root.with_name(output_root.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"remove stale incomplete output first: {temporary}")
    temporary.mkdir(parents=True)
    summaries: list[dict] = []
    try:
        for row in rows:
            sample_id = str(row["sample_id"])
            metadata = refine_sample(
                input_root / sample_id,
                temporary / sample_id,
                model=model,
                config=config,
            )
            summaries.append({field: metadata.get(field) for field in SUMMARY_FIELDS})
        with (temporary / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summaries)
        real_count = sum(bool(row["inference_succeeded"]) for row in summaries)
        fallback_count = sum(bool(row["fallback"]) for row in summaries)
        accepted_sam3_count = sum(row["selected_mask_source"] == "sam3" for row in summaries)
        manifest_rows = []
        for row, summary in zip(rows, summaries, strict=True):
            sample_id = str(row["sample_id"])
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "bundle": str(output_root / sample_id),
                    "inference_succeeded": bool(summary["inference_succeeded"]),
                    "selected_mask_source": summary["selected_mask_source"],
                    "fallback": bool(summary["fallback"]),
                    "metadata_sha256": sha256_file(
                        temporary / sample_id / "refinement_metadata.json"
                    ),
                }
            )
        write_jsonl(temporary / "manifest.jsonl", manifest_rows)
        save_strict_json(
            temporary / "run_metadata.json",
            {
                "schema_version": 1,
                "experiment_status": (
                    "real_sam3_inference_completed" if real_count else "no_real_sam3_inference"
                ),
                "sample_count": len(summaries),
                "real_inference_count": real_count,
                "accepted_sam3_mask_count": accepted_sam3_count,
                "fallback_count": fallback_count,
                "model": model.runtime_metadata,
                "config": config,
                "input_manifest_sha256": sha256_file(input_root / "manifest.jsonl"),
            },
        )
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "DONE",
                "samples": len(summaries),
                "real_inference_count": real_count,
                "accepted_sam3_mask_count": accepted_sam3_count,
                "fallback_count": fallback_count,
                "output": str(output_root),
            }
        )
    )
    return 0 if real_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
