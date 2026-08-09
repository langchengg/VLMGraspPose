#!/usr/bin/env python3
"""Build fail-closed validation eligibility evidence for primary selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    validate_artifact_identity,
    validate_config_identity,
    validate_public_methods,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--feature-allowlist", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--runtime-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "split_audit": args.split_audit.expanduser().resolve(),
        "feature_allowlist": args.feature_allowlist.expanduser().resolve(),
        "training_manifest": args.training_manifest.expanduser().resolve(),
        "runtime_benchmark": args.runtime_benchmark.expanduser().resolve(),
    }
    loaded = {name: load_json(path) for name, path in paths.items()}
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML object")
    validate_config_identity(config)
    maximum_runtime = float(
        config.get("primary_selection", {}).get(
            "maximum_inference_seconds_per_sample", -1.0
        )
    )
    if maximum_runtime <= 0:
        raise ValueError("config must preregister a positive tabular runtime limit")
    inference_device = str(
        config.get("primary_selection", {}).get("inference_device", "")
    )
    if inference_device not in {"auto", "mps", "cpu"}:
        raise ValueError("config must preregister auto, mps, or cpu inference")
    if loaded["split_audit"].get("required_intersections_all_zero") is not True:
        raise ValueError("split audit does not prove zero required intersections")
    allowlist = loaded["feature_allowlist"]
    if (
        allowlist.get("ground_truth_allowed") is not False
        or not allowlist.get("features")
    ):
        raise ValueError("inference feature allowlist is not GT-free")
    training = loaded["training_manifest"]
    benchmark = loaded["runtime_benchmark"]
    validate_artifact_identity(training, context="training manifest")
    validate_artifact_identity(benchmark, context="runtime benchmark")
    benchmark_inputs = benchmark.get("inputs")
    expected_inference_bundle = (
        paths["training_manifest"].parent / "inference_bundle.json"
    )
    expected_benchmark_inputs: dict[str, Path | str] = {
        "per_candidate": Path(
            str(training.get("validation_per_candidate", ""))
        ).resolve(),
        "per_candidate_sha256": str(training.get("validation_sha256", "")),
        "sample_universe": Path(
            str(training.get("validation_per_sample", ""))
        ).resolve(),
        "sample_universe_sha256": str(
            training.get("validation_per_sample_sha256", "")
        ),
        "training_manifest": paths["training_manifest"],
        "training_manifest_sha256": sha256_file(paths["training_manifest"]),
        "inference_bundle": expected_inference_bundle.resolve(),
        "inference_bundle_sha256": (
            sha256_file(expected_inference_bundle)
            if expected_inference_bundle.is_file()
            else ""
        ),
    }
    if not expected_inference_bundle.is_file():
        raise ValueError("training inference bundle is missing")
    inference_bundle = load_json(expected_inference_bundle)
    validate_artifact_identity(inference_bundle, context="inference bundle")
    if not isinstance(benchmark_inputs, dict):
        raise ValueError("runtime benchmark omits bound validation inputs")
    for role in (
        "per_candidate",
        "sample_universe",
        "training_manifest",
        "inference_bundle",
    ):
        actual_path = Path(str(benchmark_inputs.get(role, ""))).resolve()
        actual_hash = benchmark_inputs.get(f"{role}_sha256")
        if (
            actual_path != expected_benchmark_inputs[role]
            or actual_hash != expected_benchmark_inputs[f"{role}_sha256"]
            or not actual_path.is_file()
            or sha256_file(actual_path) != actual_hash
        ):
            raise ValueError(
                f"runtime benchmark input binding is invalid: {role}"
            )
    primary_methods = list(map(str, training.get("primary_candidate_methods", [])))
    validate_public_methods(
        primary_methods, context="training primary_candidate_methods"
    )
    benchmark_methods = benchmark.get("methods")
    if (
        not primary_methods
        or len(primary_methods) != len(set(primary_methods))
        or not isinstance(benchmark_methods, dict)
        or set(primary_methods) != set(map(str, benchmark_methods))
        or training.get("candidate_pool_modified") is not False
        or benchmark.get("candidate_pool_modified") is not False
        or benchmark.get("candidate_identity_invariant") is not True
        or benchmark.get("split") not in {"val", "validation"}
        or benchmark.get("device_requested") != inference_device
        or primary_methods
        != list(map(str, inference_bundle.get("primary_candidate_methods", [])))
    ):
        raise ValueError("training/runtime candidate-method contract is invalid")

    candidate_methods: dict[str, dict[str, Any]] = {}
    for method in primary_methods:
        runtime = float(
            benchmark_methods[method]["inference_seconds_per_sample"]
        )
        if (
            not math.isfinite(runtime)
            or runtime < 0
            or benchmark_methods[method].get("protocol") != "full_nms"
            or benchmark_methods[method].get("measurement")
            != "conservative_complete_suite_upper_bound"
        ):
            raise ValueError(f"invalid runtime evidence for method: {method}")
        candidate_methods[method] = {
            "protocol": str(benchmark_methods[method]["protocol"]),
            "inference_seconds_per_sample": runtime,
        }
    evidence = {
        "schema_version": 1,
        **identity_payload(),
        "selection_split": "validation",
        "maximum_inference_seconds_per_sample": maximum_runtime,
        "formal_inference_device": inference_device,
        "runtime_threshold_preregistered_in_config": True,
        "candidate_methods": candidate_methods,
        "source_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
