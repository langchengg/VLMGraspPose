#!/usr/bin/env python3
"""Select the validation primary only when every preregistered gate passes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
from src.grasping.reranking_v1.method_namespace import (  # noqa: E402
    FULL_NMS_BASELINE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sample-outcomes", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--evaluation-bundle", type=Path, required=True)
    parser.add_argument("--sample-universe", type=Path, required=True)
    parser.add_argument("--eligibility-evidence", type=Path, required=True)
    parser.add_argument("--harmful-rate-limit-all-samples", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.harmful_rate_limit_all_samples <= 1.0:
        parser.error("--harmful-rate-limit-all-samples must be in [0,1]")

    outcomes_path = args.per_sample_outcomes.expanduser().resolve()
    bootstrap_path = args.bootstrap.expanduser().resolve()
    bundle_path = args.evaluation_bundle.expanduser().resolve()
    universe_path = args.sample_universe.expanduser().resolve()
    evidence_path = args.eligibility_evidence.expanduser().resolve()
    outcomes = pd.read_parquet(outcomes_path)
    bootstrap = pd.read_parquet(bootstrap_path)
    universe = pd.read_parquet(universe_path)
    if "sample_id" not in universe or universe["sample_id"].astype(str).duplicated().any():
        raise ValueError("validation sample universe requires unique sample_id")
    if "split" not in universe or set(universe["split"].astype(str)) - {
        "val",
        "validation",
    }:
        raise ValueError("primary selection universe must be the validation split")
    universe_ids = set(universe["sample_id"].astype(str))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_artifact_identity(bundle, context="evaluation bundle")
    if (
        bundle.get("report_recomputation_verified") is not True
        or bundle.get("candidate_pool_modified") is not False
        or bundle.get("sample_count_all") != len(universe)
    ):
        raise ValueError("evaluation bundle is not verified for this universe")
    for file_key, expected_path in (
        ("per_sample_outcomes", outcomes_path),
        ("bootstrap", bootstrap_path),
    ):
        item = bundle.get("files", {}).get(file_key, {})
        if (
            Path(str(item.get("path", ""))).resolve() != expected_path
            or item.get("sha256") != sha256_file(expected_path)
        ):
            raise ValueError(f"evaluation bundle does not bind {file_key}")
    universe_sources = [
        item
        for item in bundle.get("provenance", {}).get("sources", [])
        if item.get("role") == "sample_universe"
    ]
    if (
        len(universe_sources) != 1
        or Path(str(universe_sources[0]["path"])).resolve() != universe_path
        or universe_sources[0]["sha256"] != sha256_file(universe_path)
    ):
        raise ValueError("evaluation bundle does not bind validation universe")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    validate_artifact_identity(evidence, context="eligibility evidence")
    methods = evidence.get("candidate_methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("eligibility evidence requires candidate_methods")
    maximum_runtime = float(evidence["maximum_inference_seconds_per_sample"])
    if maximum_runtime <= 0:
        raise ValueError("maximum inference runtime must be positive")
    source_artifacts = evidence.get("source_artifacts")
    required_roles = {
        "split_audit",
        "feature_allowlist",
        "training_manifest",
        "runtime_benchmark",
    }
    if not isinstance(source_artifacts, dict) or set(source_artifacts) != required_roles:
        raise ValueError(f"eligibility source_artifacts must be exactly {sorted(required_roles)}")
    loaded_sources: dict[str, dict[str, Any]] = {}
    for role, artifact in source_artifacts.items():
        path = Path(str(artifact["path"])).expanduser().resolve()
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"eligibility evidence source changed: {path}")
        loaded_sources[role] = json.loads(path.read_text(encoding="utf-8"))
    if loaded_sources["split_audit"].get("required_intersections_all_zero") is not True:
        raise ValueError("split audit does not prove zero train/validation/test leakage")
    allowlist = loaded_sources["feature_allowlist"]
    if allowlist.get("ground_truth_allowed") is not False or not allowlist.get("features"):
        raise ValueError("inference feature allowlist is invalid")
    training = loaded_sources["training_manifest"]
    validate_artifact_identity(training, context="training manifest")
    if training.get("candidate_pool_modified") is not False:
        raise ValueError("training manifest changed the frozen candidate pool")
    expected_methods = set(map(str, training.get("primary_candidate_methods", [])))
    validate_public_methods(
        sorted(expected_methods), context="training primary_candidate_methods"
    )
    if not expected_methods or set(map(str, methods)) != expected_methods:
        raise ValueError(
            "candidate_methods must exactly match the preregistered training candidates"
        )
    runtime_benchmark = loaded_sources["runtime_benchmark"]
    validate_artifact_identity(
        runtime_benchmark, context="runtime benchmark"
    )
    if int(runtime_benchmark.get("sample_count", -1)) != len(universe):
        raise ValueError("runtime benchmark does not cover validation universe")
    benchmark_inputs = runtime_benchmark.get("inputs")
    training_path = Path(
        str(source_artifacts["training_manifest"]["path"])
    ).expanduser().resolve()
    expected_benchmark_inputs: dict[str, Path | str] = {
        "per_candidate": Path(
            str(training.get("validation_per_candidate", ""))
        ).resolve(),
        "per_candidate_sha256": str(training.get("validation_sha256", "")),
        "sample_universe": universe_path,
        "sample_universe_sha256": sha256_file(universe_path),
        "training_manifest": training_path,
        "training_manifest_sha256": sha256_file(training_path),
        "inference_bundle": (
            training_path.parent / "inference_bundle.json"
        ).resolve(),
    }
    expected_benchmark_inputs["inference_bundle_sha256"] = sha256_file(
        expected_benchmark_inputs["inference_bundle"]
    )
    inference_bundle = json.loads(
        Path(expected_benchmark_inputs["inference_bundle"]).read_text(
            encoding="utf-8"
        )
    )
    validate_artifact_identity(inference_bundle, context="inference bundle")
    if set(
        map(str, inference_bundle.get("primary_candidate_methods", []))
    ) != expected_methods:
        raise ValueError(
            "inference bundle primary methods disagree with training manifest"
        )
    if not isinstance(benchmark_inputs, dict):
        raise ValueError("runtime benchmark omits validation input bindings")
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
            raise ValueError(f"runtime benchmark input changed: {role}")
    registered_device = str(evidence.get("formal_inference_device", ""))
    if (
        registered_device not in {"auto", "mps", "cpu"}
        or runtime_benchmark.get("device_requested") != registered_device
    ):
        raise ValueError(
            "runtime benchmark device disagrees with eligibility evidence"
        )

    audit_rows: list[dict[str, Any]] = []
    for method, method_evidence in sorted(methods.items()):
        protocol = str(method_evidence["protocol"])
        group = outcomes.loc[
            outcomes["method"].astype(str).eq(str(method))
            & outcomes["protocol"].astype(str).eq(protocol)
        ]
        if group.empty or group["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"missing or duplicate validation outcomes: {protocol}/{method}")
        if set(group["sample_id"].astype(str)) != universe_ids:
            raise ValueError(f"validation method has incomplete sample universe: {method}")
        all_samples = len(group)
        recovered = int(group["recovered"].astype(bool).sum())
        harmful = int(group["harmful"].astype(bool).sum())
        net_gain = recovered - harmful
        harmful_rate = harmful / all_samples
        ci = bootstrap.loc[
            bootstrap["method"].astype(str).eq(str(method))
            & bootstrap["protocol"].astype(str).eq(protocol)
            & bootstrap["metric"].astype(str).eq("j_at_1_delta_all")
        ]
        if len(ci) != 1 or int(ci.iloc[0]["replicates"]) < 10_000:
            raise ValueError(f"missing 10k scene bootstrap delta: {protocol}/{method}")
        point = float(ci.iloc[0]["point_estimate"])
        lower = float(ci.iloc[0]["ci_95_lower"])
        upper = float(ci.iloc[0]["ci_95_upper"])
        runtime = float(
            runtime_benchmark.get("methods", {})
            .get(str(method), {})
            .get("inference_seconds_per_sample", float("inf"))
        )
        runtime_item = runtime_benchmark.get("methods", {}).get(str(method), {})
        if (
            not math.isfinite(runtime)
            or runtime < 0
            or runtime_item.get("protocol") != protocol
            or runtime_item.get("measurement")
            != "conservative_complete_suite_upper_bound"
        ):
            raise ValueError(f"invalid runtime benchmark method: {method}")
        gates = {
            "positive_net_gain": net_gain > 0,
            "bootstrap_positive_trend": lower > 0.0,
            "harm_cap_passed": harmful_rate
            <= float(args.harmful_rate_limit_all_samples),
            "leakage_free": True,
            "feature_allowlist_passed": True,
            "candidate_identity_invariant": bundle.get("provenance", {}).get(
                "candidate_identity_invariant_enforced"
            )
            is True,
            "runtime_acceptable": 0.0 <= runtime <= maximum_runtime,
        }
        audit_rows.append(
            {
                "method": str(method),
                "protocol": protocol,
                "all_samples": all_samples,
                "recovered": recovered,
                "harmful": harmful,
                "net_gain": net_gain,
                "harmful_rate_all_samples": harmful_rate,
                "bootstrap_delta_point": point,
                "bootstrap_delta_ci_95_lower": lower,
                "bootstrap_delta_ci_95_upper": upper,
                "inference_seconds_per_sample": runtime,
                **gates,
                "eligible": all(gates.values()),
            }
        )
    audit = pd.DataFrame(audit_rows)
    eligible = audit.loc[audit["eligible"]].sort_values(
        [
            "net_gain",
            "bootstrap_delta_point",
            "bootstrap_delta_ci_95_lower",
            "method",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    if eligible.empty:
        selected = {
            "method": FULL_NMS_BASELINE,
            "protocol": "full_nms",
            "eligible": False,
            "positive_net_gain": False,
            "bootstrap_positive_trend": False,
            "harm_cap_passed": True,
            "leakage_free": True,
            "feature_allowlist_passed": True,
            "candidate_identity_invariant": True,
            "runtime_acceptable": True,
            "net_gain": 0,
            "recovered": 0,
            "harmful": 0,
            "harmful_rate_all_samples": 0.0,
            "bootstrap_delta_point": 0.0,
            "bootstrap_delta_ci_95_lower": 0.0,
            "bootstrap_delta_ci_95_upper": 0.0,
            "inference_seconds_per_sample": 0.0,
            "all_samples": len(universe),
        }
        selection_reason = (
            "no_candidate_reranker_eligible_"
            "repeatedfilm_gqcnn_q_only_fallback"
        )
    else:
        selected = eligible.iloc[0].to_dict()
        selection_reason = "eligible_candidate_max_net_gain"

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    audit_path = root / "primary_selection_audit.csv"
    audit.to_csv(audit_path, index=False)
    selection = {
        "schema_version": 1,
        **identity_payload(),
        "selection_split": "validation",
        "primary_method": selected["method"],
        "protocol": selected["protocol"],
        "selection_reason": selection_reason,
        "baseline_fallback": selected["method"] == FULL_NMS_BASELINE,
        "selection_order": [
            "positive_net_gain",
            "bootstrap_positive_trend",
            "harm_cap_passed",
            "leakage_free",
            "feature_allowlist_passed",
            "candidate_identity_invariant",
            "runtime_acceptable",
            "max_net_gain_then_bootstrap_trend",
        ],
        "bootstrap_positive_trend_definition": (
            "paired scene-bootstrap J@1 delta 95% CI lower bound > 0"
        ),
        "harmful_rate_limit_all_samples": float(
            args.harmful_rate_limit_all_samples
        ),
        "maximum_inference_seconds_per_sample": maximum_runtime,
        "formal_inference_device": registered_device,
        "selected_metrics": selected,
        "inputs": {
            "per_sample_outcomes": str(outcomes_path),
            "per_sample_outcomes_sha256": sha256_file(outcomes_path),
            "bootstrap": str(bootstrap_path),
            "bootstrap_sha256": sha256_file(bootstrap_path),
            "evaluation_bundle": str(bundle_path),
            "evaluation_bundle_sha256": sha256_file(bundle_path),
            "sample_universe": str(universe_path),
            "sample_universe_sha256": sha256_file(universe_path),
            "eligibility_evidence": str(evidence_path),
            "eligibility_evidence_sha256": sha256_file(evidence_path),
            "selection_audit": str(audit_path),
            "selection_audit_sha256": sha256_file(audit_path),
        },
    }
    temporary = root / f".selection.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, root / "selection.json")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
