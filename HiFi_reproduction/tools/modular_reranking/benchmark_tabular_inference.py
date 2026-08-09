#!/usr/bin/env python3
"""Benchmark complete tabular reranking inference on the validation universe."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    validate_artifact_identity,
    validate_public_methods,
)
from tools.modular_reranking.apply_rerankers import (  # noqa: E402
    score_candidate_frame,
)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--sample-universe", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.measured_runs < 1:
        parser.error("warmup-runs must be nonnegative and measured-runs positive")

    candidate_path = args.per_candidate.expanduser().resolve()
    universe_path = args.sample_universe.expanduser().resolve()
    training_root = args.training_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")

    frame = pd.read_parquet(candidate_path)
    universe = pd.read_parquet(universe_path)
    if (
        "sample_id" not in universe
        or universe["sample_id"].astype(str).duplicated().any()
        or "split" not in universe
        or set(universe["split"].astype(str)) - {"val", "validation"}
    ):
        raise ValueError("runtime benchmark requires a unique validation universe")
    universe_ids = set(universe["sample_id"].astype(str))
    if not set(frame["sample_id"].astype(str)) <= universe_ids:
        raise ValueError("runtime candidates fall outside validation universe")

    training_manifest_path = training_root / "training_manifest.json"
    inference_bundle_path = training_root / "inference_bundle.json"
    training = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    bundle = json.loads(inference_bundle_path.read_text(encoding="utf-8"))
    validate_artifact_identity(training, context="training manifest")
    validate_artifact_identity(bundle, context="inference bundle")
    if (
        Path(str(training.get("validation_per_candidate", ""))).resolve()
        != candidate_path
        or training.get("validation_sha256") != sha256_file(candidate_path)
        or Path(str(training.get("validation_per_sample", ""))).resolve()
        != universe_path
        or training.get("validation_per_sample_sha256")
        != sha256_file(universe_path)
        or training.get("candidate_pool_modified") is not False
        or bundle.get("candidate_pool_modified") is not False
        or bundle.get("reload_parity_verified") is not True
    ):
        raise ValueError("training artifacts are not bound to this validation input")
    primary_methods = list(map(str, training.get("primary_candidate_methods", [])))
    validate_public_methods(
        primary_methods, context="training primary_candidate_methods"
    )
    if (
        not primary_methods
        or primary_methods
        != list(map(str, bundle.get("primary_candidate_methods", [])))
        or len(primary_methods) != len(set(primary_methods))
    ):
        raise ValueError("training primary candidate method contract is invalid")

    final_audit: dict[str, Any] | None = None
    for _ in range(args.warmup_runs):
        _, _, final_audit = score_candidate_frame(
            frame, training_root=training_root, device=args.device
        )
    elapsed: list[float] = []
    for _ in range(args.measured_runs):
        started = time.perf_counter()
        predictions, decisions, final_audit = score_candidate_frame(
            frame, training_root=training_root, device=args.device
        )
        elapsed.append(time.perf_counter() - started)
        observed = set(predictions["reranker_method"].astype(str))
        if (
            not set(primary_methods) <= observed
            or predictions["candidate_identity_sha256"].isna().any()
            or final_audit.get("candidate_pool_modified") is not False
            or final_audit.get("candidate_identity_invariant") is not True
            or decisions["sample_id"].astype(str).duplicated().any()
        ):
            raise AssertionError("runtime benchmark inference contract failed")

    conservative_seconds = max(elapsed)
    seconds_per_sample = conservative_seconds / len(universe)
    benchmark = {
        "schema_version": 1,
        **identity_payload(),
        "split": "validation",
        "definition": (
            "maximum wall time across measured end-to-end runs of the complete "
            "tabular reranking suite, divided by all validation samples; this "
            "conservative suite-wide upper bound is assigned to every candidate "
            "method"
        ),
        "sample_count": len(universe),
        "nonempty_sample_count": frame["sample_id"].astype(str).nunique(),
        "valid_empty_sample_count": len(universe)
        - frame["sample_id"].astype(str).nunique(),
        "candidate_count": len(frame),
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "elapsed_seconds": elapsed,
        "conservative_elapsed_seconds": conservative_seconds,
        "inference_seconds_per_sample": seconds_per_sample,
        "device_requested": args.device,
        "device_audit": final_audit,
        "methods": {
            method: {
                "protocol": "full_nms",
                "inference_seconds_per_sample": seconds_per_sample,
                "measurement": "conservative_complete_suite_upper_bound",
            }
            for method in primary_methods
        },
        "candidate_pool_modified": False,
        "candidate_identity_invariant": True,
        "inputs": {
            "per_candidate": str(candidate_path),
            "per_candidate_sha256": sha256_file(candidate_path),
            "sample_universe": str(universe_path),
            "sample_universe_sha256": sha256_file(universe_path),
            "training_manifest": str(training_manifest_path.resolve()),
            "training_manifest_sha256": sha256_file(training_manifest_path),
            "inference_bundle": str(inference_bundle_path.resolve()),
            "inference_bundle_sha256": sha256_file(inference_bundle_path),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "runtime_benchmark.json", benchmark)
    (output_root / "run_command.txt").write_text(
        " ".join(map(shlex.quote, [sys.executable, *sys.argv])) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(benchmark, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
