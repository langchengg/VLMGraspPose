#!/usr/bin/env python3
"""Run resumable, batch-size-one Transformers SAM 3 refinement on CPU only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import psutil
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_cpu_model import TransformersSam3Cpu  # noqa: E402
from src.segmentation.sam3_cpu_refiner import refine_cpu_sample  # noqa: E402
from src.segmentation.sam3_cpu_serialization import (  # noqa: E402
    assert_no_ground_truth_leakage,
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)


PROTECTED_OUTPUTS = {
    (REPO_ROOT / "outputs" / "dexnet_candidates_ten_samples").resolve(),
    (REPO_ROOT / "outputs" / "gqcnn_original_ranking_evaluation").resolve(),
}
SUMMARY_FIELDS = (
    "sample_id",
    "inference_succeeded",
    "selected_mask_source",
    "selected_hypothesis_id",
    "hypothesis_count",
    "model_quality",
    "refinement_score",
    "fallback",
    "fallback_reason",
    "inference_time_seconds",
    "peak_rss_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_cpu_inputs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_cpu_refined_masks",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "sam3_cpu_refinement.yaml",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--backend", choices=("tracker", "pcs"))
    parser.add_argument(
        "--prompt-mode",
        choices=(
            "point",
            "box",
            "box_point",
            "box_positive_negative_points",
            "pcs_positive_box",
            "pcs_text_box",
        ),
    )
    parser.add_argument("--processor-size", type=int, choices=(1008, 560))
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--interop-threads", type=int)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--sample-limit", "--max-samples", type=int)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-all-hypotheses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fallback-to-hifics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _rows(input_root: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (input_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or len(rows) != len({row["sample_id"] for row in rows}):
        raise ValueError("input manifest must contain unique samples")
    for row in rows:
        assert_no_ground_truth_leakage(row, context="SAM 3 CPU source manifest")
    return rows


def _select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.sample_id:
        requested = set(args.sample_id)
        unknown = requested - {str(row["sample_id"]) for row in rows}
        if unknown:
            raise KeyError(f"unknown sample IDs: {sorted(unknown)}")
        rows = [row for row in rows if str(row["sample_id"]) in requested]
    if args.sample_limit is not None:
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        rows = rows[: args.sample_limit]
    return rows


def _validate_roots(input_root: Path, output_root: Path) -> None:
    if output_root in PROTECTED_OUTPUTS:
        raise ValueError(f"refusing to use protected baseline output: {output_root}")
    if "sam3_cpu" not in output_root.name:
        raise ValueError("CPU output root name must contain 'sam3_cpu'")
    if output_root == input_root or output_root in input_root.parents or input_root in output_root.parents:
        raise ValueError("input and output roots must not contain one another")


def _existing_metadata(output_root: Path, sample_id: str) -> dict | None:
    path = output_root / sample_id / "refinement_metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_run_tables(output_root: Path, rows: list[dict], config: dict, input_root: Path) -> None:
    summaries = []
    manifest = []
    for row in rows:
        sample_id = str(row["sample_id"])
        metadata = _existing_metadata(output_root, sample_id)
        if metadata is None:
            continue
        summaries.append({field: metadata.get(field) for field in SUMMARY_FIELDS})
        manifest.append(
            {
                "sample_id": sample_id,
                "bundle": str(output_root / sample_id),
                "inference_succeeded": bool(metadata["inference_succeeded"]),
                "selected_mask_source": metadata["selected_mask_source"],
                "fallback": bool(metadata["fallback"]),
                "metadata_sha256": sha256_file(
                    output_root / sample_id / "refinement_metadata.json"
                ),
            }
        )
    temporary_csv = output_root / "summary.csv.incomplete"
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    temporary_csv.replace(output_root / "summary.csv")
    atomic_write_jsonl(output_root / "manifest.jsonl", manifest)
    real_count = sum(bool(row["inference_succeeded"]) for row in summaries)
    atomic_write_json(
        output_root / "run_metadata.json",
        {
            "schema_version": 1,
            "experiment_status": (
                "real_sam3_cpu_inference_completed"
                if real_count
                else "no_real_sam3_cpu_inference"
            ),
            "sample_count": len(summaries),
            "real_inference_count": real_count,
            "accepted_tracker_count": sum(
                row["selected_mask_source"] == "sam3_tracker_cpu" for row in summaries
            ),
            "accepted_pcs_count": sum(
                row["selected_mask_source"] == "sam3_pcs_cpu" for row in summaries
            ),
            "fallback_count": sum(bool(row["fallback"]) for row in summaries),
            "config": config,
            "input_manifest_sha256": sha256_file(input_root / "manifest.jsonl"),
        },
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config, runtime_config = config["model"], config["runtime"]
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    _validate_roots(input_root, output_root)
    rows = _select_rows(_rows(input_root), args)
    backend = args.backend or str(model_config["backend"])
    prompt_mode = args.prompt_mode or str(config["prompt"]["mode"])
    if backend == "tracker" and prompt_mode.startswith("pcs_"):
        raise ValueError("Tracker backend requires a Tracker prompt mode")
    if backend == "pcs" and not prompt_mode.startswith("pcs_"):
        prompt_mode = "pcs_positive_box"
    revision = args.revision or str(model_config["revision"])
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("an immutable 40-character model revision is required")
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path
        else REPO_ROOT / "models" / "huggingface" / "facebook-sam3" / revision
    )
    processor_size = args.processor_size or int(model_config["processor_size"])
    num_threads = args.num_threads or int(runtime_config["num_threads"])
    interop_threads = args.interop_threads or int(runtime_config["interop_threads"])
    plan = {
        "status": "DRY_RUN" if args.dry_run else "READY",
        "device": "cpu",
        "dtype": "float32",
        "backend": backend,
        "prompt_mode": prompt_mode,
        "processor_size": processor_size,
        "model_path": str(model_path),
        "revision": revision,
        "samples": [str(row["sample_id"]) for row in rows],
        "input_root": str(input_root),
        "output_root": str(output_root),
        "local_files_only": bool(args.local_files_only),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.local_files_only:
        raise ValueError("this experiment loads only the pinned local snapshot")
    if args.overwrite:
        if output_root.exists():
            shutil.rmtree(output_root)
    elif output_root.exists() and not (args.resume or args.skip_existing):
        raise FileExistsError("output exists; choose --resume, --skip-existing, or --overwrite")
    model = TransformersSam3Cpu(
        model_path,
        revision=revision,
        backend=backend,
        processor_size=processor_size,
        num_threads=num_threads,
        interop_threads=interop_threads,
        environment_threads={
            "OMP_NUM_THREADS": int(runtime_config["omp_num_threads"]),
            "VECLIB_MAXIMUM_THREADS": int(runtime_config["veclib_maximum_threads"]),
            "OPENBLAS_NUM_THREADS": int(runtime_config["openblas_num_threads"]),
            "MKL_NUM_THREADS": int(runtime_config["mkl_num_threads"]),
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        sample_id = str(row["sample_id"])
        if _existing_metadata(output_root, sample_id) is not None:
            if args.resume or args.skip_existing:
                continue
            raise FileExistsError(f"sample output exists: {sample_id}")
        incomplete = (output_root / sample_id).with_name(sample_id + ".incomplete")
        if incomplete.exists():
            if args.resume:
                shutil.rmtree(incomplete)
            else:
                raise FileExistsError(
                    f"stale incomplete sample requires --resume or manual review: {incomplete}"
                )
        memory = psutil.virtual_memory()
        if memory.available / memory.total < 0.15:
            raise MemoryError("available RAM fell below the 15% safety floor")
        refine_cpu_sample(
            Path(row.get("bundle_path", input_root / sample_id)).expanduser().resolve(),
            output_root / sample_id,
            model=model,
            config=config,
            prompt_mode=prompt_mode,
            save_all_hypotheses=args.save_all_hypotheses,
            fallback_to_hifics=args.fallback_to_hifics,
        )
        _write_run_tables(output_root, rows, config, input_root)
    _write_run_tables(output_root, rows, config, input_root)
    run = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))
    print(json.dumps({**plan, **run}, indent=2, sort_keys=True))
    return 0 if run["real_inference_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
