#!/usr/bin/env python3
"""Freeze environment, seed, feature, model, and command contracts for a run."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import canonical_sha256, sha256_file  # noqa: E402


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: write_protocol.py RUN_DIR BASE_RUN")
    run = Path(sys.argv[1]).expanduser().resolve()
    base = Path(sys.argv[2]).expanduser().resolve()
    environment: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _version(name)
            for name in (
                "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
                "torch", "torchvision", "lightgbm", "xgboost", "opencv-python",
                "shapely", "matplotlib", "statsmodels", "joblib",
            )
        },
        "mps_built": False,
        "mps_available": False,
        "cuda_available": False,
    }
    try:
        import torch

        environment.update(
            {
                "mps_built": bool(torch.backends.mps.is_built()),
                "mps_available": bool(torch.backends.mps.is_available()),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
    except ImportError:
        pass
    atomic_json(run / "environment.json", environment)
    (run / "environment.txt").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol = {
        "schema_version": 1,
        "experiment": "complete_g1_c1_frozen_candidate_reranking",
        "base_run": str(base),
        "base_finalization_sha256": sha256_file(base / "FINALIZATION_COMPLETE.json"),
        "candidate_contract": {
            "canonical": "All post-NMS candidates in immutable original-image geometry",
            "derived_views": ["Top5", "Top10", "AllNMS"],
            "geometry_mutation": False,
            "source_threshold_or_nms_change": False,
        },
        "splits": {"development": "official Train with 5 scene-grouped folds", "selection": "official Validation", "formal": "official Test opened once after lock"},
        "seeds": {"development": 17, "formal_validation": [17, 29, 43], "bootstrap": 20260806},
        "features": [f"F{index}" for index in range(9)],
        "rungs": [f"R{index}" for index in range(15)],
        "losses": ["query-normalized BCE", "same-query RankNet", "multi-positive listwise softmax-mass"],
        "encoders": ["linear", "residual MLP", "LightGBM LambdaMART", "DeepSets", "Set Transformer", "edge-conditioned GNN"],
        "formal_test_policy": "candidate/features/models/prediction plan hash locked; atomic O_EXCL test-access transaction; no post-test reselection",
        "claims": {"metric": "offline 2D rectangle consistency", "physical_grasp_success": False, "collision_features": "single-view 2.5D proxies"},
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    atomic_json(run / "configs" / "experiment_protocol.json", protocol)
    git = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()),
    }
    atomic_json(run / "00_audit" / "git_state.json", git)
    (run / "git_state.txt").write_text(
        json.dumps(git, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"environment": environment, "protocol": protocol, "git": git}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
