#!/usr/bin/env python3
"""Resume the SAM 3 CPU → Dex-Net → GQ-CNN → paired-evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_PYTHON = REPO_ROOT / ".venv-sam3-cpu" / "bin" / "python"
GQCNN_PYTHON = REPO_ROOT / ".venv-gqcnn" / "bin" / "python"
REFINED = REPO_ROOT / "outputs" / "sam3_cpu_refined_masks"
BUNDLES = REPO_ROOT / "outputs" / "sam3_cpu_dexnet_input_bundles"
CANDIDATES = REPO_ROOT / "outputs" / "dexnet_candidates_sam3_cpu"
MASK_EVALUATION = REPO_ROOT / "outputs" / "sam3_cpu_mask_evaluation"
GQ_EVALUATION = REPO_ROOT / "outputs" / "gqcnn_sam3_cpu_ranking_evaluation"
COMPARISON = REPO_ROOT / "outputs" / "sam3_cpu_grasp_comparison"
ANNOTATIONS = (
    REPO_ROOT.parent
    / "crog_reproduction"
    / "OCID-VLG"
    / "refer"
    / "unique"
    / "test_expressions.json"
)


def _stage_complete(stage: str, marker: Path, sample_limit: int) -> bool:
    if not marker.is_file():
        return False
    try:
        if stage == "refinement":
            return int(json.loads(marker.read_text(encoding="utf-8"))["sample_count"]) == sample_limit
        if stage in {"dexnet", "gqcnn", "geometric"}:
            with marker.open(encoding="utf-8", newline="") as stream:
                return len(list(csv.DictReader(stream))) == sample_limit
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if stage == "mask_evaluation":
            return int(payload["sample_count"]) == sample_limit
        if stage == "grasp_evaluation":
            return int(payload["audit"]["sample_count"]) == sample_limit
        return stage == "paired_comparison" and len(payload.get("pipelines", [])) == 4
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _commands(args: argparse.Namespace) -> list[tuple[str, Path, list[str]]]:
    revision = args.revision
    model_path = (
        args.model_path
        or REPO_ROOT / "models" / "huggingface" / "facebook-sam3" / revision
    )
    parent_mount = f"{REPO_ROOT.parent}:/workspace"
    return [
        (
            "refinement",
            REFINED / "run_metadata.json",
            [
                str(CPU_PYTHON),
                str(REPO_ROOT / "scripts" / "run_sam3_cpu_refinement.py"),
                "--model-path",
                str(model_path),
                "--revision",
                revision,
                "--backend",
                args.backend,
                "--prompt-mode",
                args.prompt_mode,
                "--processor-size",
                str(args.processor_size),
                "--num-threads",
                str(args.num_threads),
                "--interop-threads",
                str(args.interop_threads),
                "--sample-limit",
                str(args.sample_limit),
                "--resume",
            ],
        ),
        (
            "dexnet",
            CANDIDATES / "summary.csv",
            [
                str(CPU_PYTHON),
                str(REPO_ROOT / "scripts" / "regenerate_dexnet_from_sam3_masks.py"),
                "--refined-root",
                str(REFINED),
                "--bundle-root",
                str(BUNDLES),
                "--candidate-output",
                str(CANDIDATES),
                "--resume",
                "--execute",
            ],
        ),
        (
            "gqcnn",
            CANDIDATES / "gqcnn_scoring_summary.csv",
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "-v",
                parent_mount,
                "-v",
                f"{REPO_ROOT / 'models' / 'gqcnn-official'}:/models:ro",
                "-w",
                "/workspace/HiFi_reproduction",
                "vlmgrasp/gqcnn-score:1.3.0",
                "python",
                "scripts/score_existing_dexnet_candidates.py",
                "/workspace/HiFi_reproduction/outputs/dexnet_candidates_sam3_cpu",
                "--model-dir",
                "/models/GQCNN-2.1",
            ],
        ),
        (
            "geometric",
            CANDIDATES / "geometric_ranking_summary.csv",
            [
                str(GQCNN_PYTHON),
                str(REPO_ROOT / "scripts" / "rank_existing_dexnet_candidates.py"),
                str(CANDIDATES),
                "--config",
                str(REPO_ROOT / "configs" / "dexnet_geometric_ranker.yaml"),
                "--annotations",
                str(ANNOTATIONS),
            ],
        ),
        (
            "mask_evaluation",
            MASK_EVALUATION / "summary.json",
            [
                str(CPU_PYTHON),
                str(REPO_ROOT / "scripts" / "evaluate_sam3_mask_refinement.py"),
                "--refined-root",
                str(REFINED),
                "--output-root",
                str(MASK_EVALUATION),
                "--config",
                str(REPO_ROOT / "configs" / "sam3_cpu_refinement.yaml"),
            ],
        ),
        (
            "grasp_evaluation",
            GQ_EVALUATION / "summary.json",
            [
                str(GQCNN_PYTHON),
                str(REPO_ROOT / "scripts" / "evaluate_gqcnn_original_ranking.py"),
                "--candidate-root",
                str(CANDIDATES),
                "--annotation-root",
                str(ANNOTATIONS),
                "--output-dir",
                str(GQ_EVALUATION),
                "--evaluation-config",
                str(REPO_ROOT / "configs" / "dexnet_grasp_consistency.yaml"),
                "--geometric-reference",
            ],
        ),
        (
            "paired_comparison",
            COMPARISON / "summary.json",
            [
                str(CPU_PYTHON),
                str(REPO_ROOT / "scripts" / "compare_sam3_grasp_pipelines.py"),
                "--sam3-summary",
                str(GQ_EVALUATION / "summary.json"),
                "--mask-summary",
                str(MASK_EVALUATION / "summary.json"),
                "--mask-per-sample",
                str(MASK_EVALUATION / "per_sample_metrics.csv"),
                "--sam3-candidates",
                str(CANDIDATES),
                "--output",
                str(COMPARISON),
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--revision",
        default="3c879f39826c281e95690f02c7821c4de09afae7",
    )
    parser.add_argument("--backend", choices=("tracker", "pcs"), default="tracker")
    parser.add_argument("--prompt-mode", default="box_point")
    parser.add_argument("--processor-size", type=int, choices=(1008, 560), default=1008)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commands = _commands(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN",
                    "stages": [
                        {
                            "stage": stage,
                            "complete": _stage_complete(
                                stage, marker, args.sample_limit
                            ),
                            "marker": str(marker),
                            "command": shlex.join(command),
                        }
                        for stage, marker, command in commands
                    ],
                },
                indent=2,
            )
        )
        return 0
    subprocess.run(
        [
            "/opt/anaconda3/bin/python",
            str(REPO_ROOT / "scripts" / "audit_sam3_protected_outputs.py"),
            "verify",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    for stage, marker, command in commands:
        if _stage_complete(stage, marker, args.sample_limit):
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode:
            print(json.dumps({"status": "STOPPED", "stage": stage, "returncode": completed.returncode}))
            return completed.returncode
    subprocess.run(
        [
            "/opt/anaconda3/bin/python",
            str(REPO_ROOT / "scripts" / "audit_sam3_protected_outputs.py"),
            "verify",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    print(json.dumps({"status": "COMPLETE", "stages": len(commands)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
