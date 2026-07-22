#!/usr/bin/env python3
"""Execute the post-refinement Dex-Net, GQ-CNN, ranker, and comparison stages."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFINED = REPO_ROOT / "outputs" / "sam3_refined_masks"
CANDIDATES = REPO_ROOT / "outputs" / "dexnet_candidates_sam3_ten_samples"
ANNOTATIONS = REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG" / "refer" / "unique" / "test_expressions.json"


def _commands() -> list[list[str]]:
    parent_mount = f"{REPO_ROOT.parent}:/workspace"
    return [
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "regenerate_dexnet_from_sam3_masks.py"),
            "--execute",
        ],
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
            "/workspace/HiFi_reproduction/outputs/dexnet_candidates_sam3_ten_samples",
            "--model-dir",
            "/models/GQCNN-2.1",
        ],
        [
            str(REPO_ROOT / ".venv-gqcnn" / "bin" / "python"),
            str(REPO_ROOT / "scripts" / "rank_existing_dexnet_candidates.py"),
            str(CANDIDATES),
            "--config",
            str(REPO_ROOT / "configs" / "dexnet_geometric_ranker.yaml"),
            "--annotations",
            str(ANNOTATIONS),
        ],
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_gqcnn_original_ranking.py"),
            "--candidate-root",
            str(CANDIDATES),
            "--annotation-root",
            str(ANNOTATIONS),
            "--output-dir",
            str(REPO_ROOT / "outputs" / "gqcnn_sam3_ranking_evaluation"),
            "--evaluation-config",
            str(REPO_ROOT / "configs" / "dexnet_grasp_consistency.yaml"),
        ],
        [sys.executable, str(REPO_ROOT / "scripts" / "compare_sam3_grasp_pipelines.py")],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run_metadata = json.loads((REFINED / "run_metadata.json").read_text(encoding="utf-8"))
    if run_metadata.get("experiment_status") != "real_sam3_inference_completed":
        raise RuntimeError("post-refinement pipeline is blocked until real official SAM 3 inference completes")
    commands = _commands()
    if not args.execute:
        print("\n".join(shlex.join(command) for command in commands))
        return 0
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

