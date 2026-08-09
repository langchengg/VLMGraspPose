#!/usr/bin/env python3
"""Resumable stage orchestrator for the SAM3 proposal-bank P@90 experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_PYTHON = PROJECT_ROOT / ".venv-sam3-cpu/bin/python"
CLIP_PYTHON = PROJECT_ROOT / "hifics/.venv/bin/python"

STAGES = (
    "audit",
    "baseline",
    "proposal-pilot",
    "proposal-full-development",
    "oracle-stage1",
    "feature-extraction",
    "train-stage1",
    "refinement-stage2",
    "oracle-stage2",
    "train-final-selector",
    "ablations",
    "lock",
    "fresh-holdout",
    "locked-benchmark",
    "output-lock",
    "final-evaluation",
    "report",
    "canonical-export",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/sam3_proposal_bank_p90_v1/experiment.yaml",
    )
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--group-list", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--force-recompute-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-all-candidates", action="store_true")
    parser.add_argument("--max-proposals-per-image", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps({"command": command}), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _proposal_arguments(args: argparse.Namespace, split: str) -> list[str]:
    values = [str(SAM_PYTHON), "scripts/generate_sam3_proposal_bank.py", "--split", split]
    for sample_id in args.sample_id:
        values.extend(["--sample-id", sample_id])
    for name, value in (
        ("--sample-limit", args.sample_limit),
        ("--frame-limit", args.frame_limit),
        ("--num-threads", args.num_threads),
        ("--max-proposals-per-image", args.max_proposals_per_image),
    ):
        if value is not None:
            values.extend([name, str(value)])
    if args.group_list:
        values.extend(["--group-list", str(args.group_list)])
    if args.resume and not args.force_recompute_stage:
        values.append("--resume")
    if args.skip_existing:
        values.append("--skip-existing")
    if args.verify_existing:
        values.append("--verify-existing")
    if args.save_all_candidates:
        values.append("--save-all-candidates")
    if args.force_recompute_stage:
        values.append("--force-recompute-sample")
    values.extend(["--seed", str(args.seed)])
    return values


def _limit(args: argparse.Namespace) -> list[str]:
    return ["--sample-limit", str(args.sample_limit)] if args.sample_limit is not None else []


def commands_for(stage: str, args: argparse.Namespace) -> list[list[str]]:
    py = str(SAM_PYTHON)
    clip = str(CLIP_PYTHON)
    resume = ["--resume"] if args.resume and not args.force_recompute_stage else []
    verify = ["--verify-existing"] if args.verify_existing else []
    if stage == "audit":
        return [
            [py, "scripts/audit_sam3_p90_project.py"],
            [py, "scripts/build_query_semantics_manifest.py"],
            [py, "scripts/run_sam3_p90_leakage_audit.py"],
        ]
    if stage == "baseline":
        return [[py, "scripts/audit_sam3_p90_project.py"]]
    if stage == "proposal-pilot":
        command = _proposal_arguments(args, "val")
        command.extend(
            [
                "--pilot-manifest",
                "outputs/sam3_proposal_bank_p90_v1/pilot/pilot_manifest.parquet",
            ]
        )
        return [[py, "scripts/select_sam3_p90_pilot.py"], command]
    if stage == "proposal-full-development":
        val = _proposal_arguments(args, "val")
        train = _proposal_arguments(args, "train")
        train.extend(["--embedding-cache-max-gib", "4"])
        return [val, train]
    if stage == "oracle-stage1":
        return [
            [py, "scripts/compute_proposal_oracle.py", "--split", "train"],
            [py, "scripts/compute_proposal_oracle.py", "--split", "val"],
        ]
    if stage == "feature-extraction":
        return [
            [clip, "scripts/extract_clip_candidate_features.py", "--split", "train", "--num-threads", str(args.num_threads)],
            [py, "scripts/extract_proposal_features.py", "--split", "train", *resume, *verify],
            [clip, "scripts/extract_clip_candidate_features.py", "--split", "val", "--num-threads", str(args.num_threads)],
            [py, "scripts/extract_proposal_features.py", "--split", "val", *resume, *verify],
        ]
    if stage == "train-stage1":
        return [[py, "scripts/train_p90_selector.py"]]
    if stage == "refinement-stage2":
        return [
            [py, "scripts/generate_stage2_refinements.py", "--split", "train", *resume, *verify, *_limit(args)],
            [py, "scripts/generate_stage2_refinements.py", "--split", "val", *resume, *verify, *_limit(args)],
        ]
    if stage == "oracle-stage2":
        return [
            [py, "scripts/compute_stage2_oracle.py", "--split", "train"],
            [py, "scripts/compute_stage2_oracle.py", "--split", "val"],
        ]
    if stage == "train-final-selector":
        return [
            [clip, "scripts/extract_clip_candidate_features.py", "--split", "train", "--proposal-stage", "stage2", "--num-threads", str(args.num_threads)],
            [py, "scripts/extract_proposal_features.py", "--split", "train", "--proposal-stage", "stage2", *resume, *verify],
            [clip, "scripts/extract_clip_candidate_features.py", "--split", "val", "--proposal-stage", "stage2", "--num-threads", str(args.num_threads)],
            [py, "scripts/extract_proposal_features.py", "--split", "val", "--proposal-stage", "stage2", *resume, *verify],
            [py, "scripts/train_final_mask_selector.py"],
        ]
    if stage == "ablations":
        return [[py, "scripts/run_sam3_p90_ablations.py", "--seed", str(args.seed)]]
    if stage == "lock":
        return [
            [py, "scripts/run_sam3_p90_leakage_audit.py"],
            [py, "scripts/lock_sam3_p90_method.py"],
        ]
    if stage == "fresh-holdout":
        return []
    if stage == "locked-benchmark":
        return [
            _proposal_arguments(args, "test"),
            [clip, "scripts/extract_clip_candidate_features.py", "--split", "test", *_limit(args)],
            [py, "scripts/extract_proposal_features.py", "--split", "test", *resume, *verify, *_limit(args)],
            [py, "scripts/generate_stage2_refinements.py", "--split", "test", *resume, *verify, *_limit(args)],
            [clip, "scripts/extract_clip_candidate_features.py", "--split", "test", "--proposal-stage", "stage2", *_limit(args)],
            [py, "scripts/extract_proposal_features.py", "--split", "test", "--proposal-stage", "stage2", *resume, *verify, *_limit(args)],
            [py, "scripts/finalize_locked_sam3_candidates.py", *resume, *verify, *_limit(args)],
        ]
    if stage == "output-lock":
        return [[py, "scripts/lock_final_masks_before_gt.py"]]
    if stage == "final-evaluation":
        return [
            [py, "scripts/compute_proposal_oracle.py", "--split", "test"],
            [py, "scripts/compute_stage2_oracle.py", "--split", "test"],
            [py, "scripts/evaluate_locked_sam3_p90_masks.py", "--seed", str(args.seed)],
        ]
    if stage == "report":
        return [[py, "scripts/generate_sam3_p90_report.py"]]
    if stage == "canonical-export":
        return [[py, "scripts/create_canonical_sam3_mask_root.py"]]
    raise ValueError(stage)


def main() -> int:
    args = parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    proposal = yaml.safe_load(
        (PROJECT_ROOT / configuration["proposal_config"]).read_text(encoding="utf-8")
    )
    if args.local_files_only and not bool(proposal["model"]["local_files_only"]):
        raise RuntimeError("configuration does not enforce local_files_only")
    if not SAM_PYTHON.is_file() or not CLIP_PYTHON.is_file():
        raise FileNotFoundError("required pinned SAM/CLIP Python environment is missing")
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        commands = commands_for(stage, args)
        if stage == "fresh-holdout" and not commands:
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "status": "NOT_AVAILABLE",
                        "reason": "no genuinely untouched scene-disjoint holdout exists",
                    }
                ),
                flush=True,
            )
            continue
        for command in commands:
            _run(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
