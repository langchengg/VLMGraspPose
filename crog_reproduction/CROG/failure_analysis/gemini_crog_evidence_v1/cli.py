from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .api import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    BudgetGuard,
    GeminiCache,
    GoogleInteractionsRunner,
    configured_concurrency,
    sha256_json,
)
from .audit import (
    PREEXISTING_V2_TREE_SHA256,
    v2_tree_digest,
    write_phase0_artifacts,
)
from .exporter import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    export_frozen_crog_evidence,
    select_development_smoke_cohort,
)
from .protocol import (
    build_lock_payload,
    candidate_identity_stream_sha256,
    claim_formal_test_once,
    file_identity,
    git_diff_hash,
    lock_experiment,
    verify_experiment_lock,
    verify_lock_payload,
)
from .schema import response_json_schema


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/v2_20260727T174412+0100"
TEST_ROOT = REPO_ROOT / "failure_analysis/reranking_outputs/full_test_17749_v1"


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked_file_identity(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    identity = file_identity(path)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"artifact hash does not match the supplied frozen identity: {Path(path).name}")
    return identity


def _required_positive_environment_float(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} must be set before the formal experiment lock")
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def command_phase0(args: argparse.Namespace) -> dict[str, Any]:
    return write_phase0_artifacts(
        output_dir=args.output,
        features_path=TEST_ROOT / "features.jsonl",
        legacy_labels_path=V2_ROOT / "formal_test_primary_v2/labels/legacy_official/labels.jsonl",
        corrected_labels_path=V2_ROOT / "formal_test_primary_v2/labels/corrected/labels.jsonl",
        split_manifest_path=V2_ROOT / "split_manifest.json",
        v2_root=V2_ROOT,
        verify_v2_tree=args.verify_v2_tree,
    )


def command_select_smoke(args: argparse.Namespace) -> dict[str, Any]:
    selected, audit = select_development_smoke_cohort(
        features_path=V2_ROOT / "base_train/features.jsonl",
        legacy_labels_path=V2_ROOT / "labels_train/legacy_official/labels.jsonl",
        split_manifest_path=V2_ROOT / "split_manifest.json",
        count=args.count,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "smoke_selection_evaluation_only.json", {"selected_local_ids": selected, "audit": audit})
    _json_dump(output / "selected_local_ids.json", selected)
    return {"status": "selected", "count": len(selected), "selected_local_ids": selected}


def command_export(args: argparse.Namespace) -> dict[str, Any]:
    selected = json.loads(Path(args.selected_local_ids).read_text(encoding="utf-8"))
    return export_frozen_crog_evidence(
        split=args.split,
        selected_local_ids=selected,
        frozen_features_path=args.frozen_features,
        output_dir=args.output,
        system_prompt_path=args.system_prompt,
        device=args.device,
        keep_dense_maps=args.keep_dense_maps,
    )


def command_run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    request_rows = []
    with Path(args.request_manifest).open("r", encoding="utf-8") as handle:
        request_rows = [json.loads(line) for line in handle if line.strip()]
    if args.sample_ids:
        requested = set(args.sample_ids)
        request_rows = [row for row in request_rows if row["sample_id"] in requested]
        missing = requested - {row["sample_id"] for row in request_rows}
        if missing:
            raise ValueError(f"requested sample IDs are absent from manifest: {sorted(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer")
        request_rows = request_rows[: args.limit]
    if len(request_rows) > 10:
        raise ValueError("Phase B allows at most 10 smoke samples per model")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    effective_concurrency = configured_concurrency(smoke=True)
    cache = GeminiCache(output / "gemini_cache.sqlite")
    spent, counts = cache.budget_state()
    raw_budget = os.environ.get("GEMINI_MAX_SPEND_USD", "").strip()
    budget = BudgetGuard(
        None if not raw_budget else float(raw_budget),
        already_estimated_usd=spent,
        request_counts=counts,
    )
    runner = GoogleInteractionsRunner(cache=cache, budget=budget)
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    schema_hash = sha256_json(response_json_schema())
    renderer_hash = _hash(Path(__file__).with_name("renderer.py"))
    evidence_schema_hash = _hash(args.evidence_schema)
    decisions = []
    try:
        for model_id in args.models:
            for request in request_rows:
                result = runner.run(
                    sample_id=request["sample_id"],
                    frame_id=request["frame_id"],
                    model_id=model_id,
                    image_path=request["board_path"],
                    system_instruction=system_prompt,
                    metadata_prompt=request["metadata"],
                    mapping=request["mapping"],
                    prompt_hash=prompt_hash,
                    schema_hash=schema_hash,
                    renderer_hash=renderer_hash,
                    evidence_schema_hash=evidence_schema_hash,
                    model_metadata={
                        "requested_model_id": model_id,
                        "sdk": "google-genai==2.16.0",
                        "api_version": "v1beta",
                    },
                    request_upper_bound_usd=args.request_upper_bound_usd,
                    image_resolution=args.image_resolution,
                    thinking_level="medium",
                    max_output_tokens=args.max_output_tokens,
                )
                parsed = result.get("parsed_output") or result.get("parsed_output_json")
                decisions.append(
                    {
                        "sample_id": request["sample_id"],
                        "model_id": model_id,
                        "request_hash": result["request_hash"],
                        "status": result["status"],
                        "valid": bool(result.get("valid", False)),
                        "abstain": bool(result.get("abstain", False)),
                        "fallback_reason": result.get("fallback_reason"),
                        "selected_display_id": None if not parsed else parsed["selected_candidate_id"],
                        "selected_candidate_id": None
                        if not parsed
                        else request["mapping"]["display_to_candidate"][parsed["selected_candidate_id"]],
                        "latency_seconds": result.get("latency_seconds"),
                        "retry_count": result.get("retry_count", 0),
                        "cache_hit": bool(result.get("cache_hit", False)),
                    }
                )
    finally:
        cache.close()
    pq.write_table(pa.Table.from_pylist(decisions), output / "per_model_decisions.parquet", compression="zstd")
    valid_count = sum(item["valid"] for item in decisions)
    missing_key = not bool(os.environ.get("GEMINI_API_KEY"))
    live_api_requests = sum(
        item["status"] not in {"blocked"} and not item["cache_hit"]
        for item in decisions
    )
    status = (
        "blocked_missing_api_key"
        if missing_key
        else ("complete" if valid_count == len(decisions) else "partial_with_fallbacks")
    )
    _json_dump(output / "smoke_status.json", {
        "status": status,
        "models": args.models,
        "prepared_request_count": len(decisions),
        "live_api_request_count": live_api_requests,
        "valid_count": valid_count,
        "fallback_count": sum(not item["valid"] for item in decisions),
        "missing_api_key": missing_key,
        "budget_set": bool(raw_budget),
        "configured_concurrency": os.environ.get("GEMINI_MAX_CONCURRENCY", "2"),
        "effective_concurrency": effective_concurrency,
        "decisions": decisions,
    })
    return {
        "status": status,
        "prepared_request_count": len(decisions),
        "live_api_request_count": live_api_requests,
        "valid_count": valid_count,
        "effective_concurrency": effective_concurrency,
    }


def command_verify_v2(args: argparse.Namespace) -> dict[str, Any]:
    observed = v2_tree_digest(V2_ROOT)
    if observed != PREEXISTING_V2_TREE_SHA256:
        raise AssertionError(f"old V2 tree changed: {observed}")
    return {"status": "unchanged", "v2_tree_sha256": observed}


def command_lock(args: argparse.Namespace) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()
    observed_diff = git_diff_hash(REPO_ROOT)
    if observed_diff != args.git_diff_sha256:
        raise ValueError("supplied git diff hash does not match the current repository")
    checkpoint = _checked_file_identity(args.checkpoint, args.checkpoint_sha256)
    config_identity = _checked_file_identity(args.config)
    candidate_features = _checked_file_identity(args.candidate_features)
    observed_candidate_identity = candidate_identity_stream_sha256(args.candidate_features)
    if observed_candidate_identity != args.candidate_sha256:
        raise ValueError("supplied candidate identity does not match the frozen candidate stream")
    split_identity = _checked_file_identity(args.split_manifest)
    renderer_path = Path(__file__).with_name("renderer.py")
    renderer_identity = _checked_file_identity(renderer_path)
    prompt_identity = _checked_file_identity(args.system_prompt)
    schema_source_identity = _checked_file_identity(Path(__file__).with_name("schema.py"))
    evidence_identity = _checked_file_identity(args.evidence_schema)
    thresholds_values = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds_identity = _checked_file_identity(args.thresholds)
    validation_values = json.loads(Path(args.validation_metrics).read_text(encoding="utf-8"))
    validation_identity = _checked_file_identity(args.validation_metrics)
    primary_values = json.loads(Path(args.primary_selection).read_text(encoding="utf-8"))
    primary_identity = _checked_file_identity(args.primary_selection)
    development_protocol_identity = _checked_file_identity(args.development_protocol_lock)
    package_sources = [
        _checked_file_identity(path)
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    ]
    sdk_version = importlib.metadata.version("google-genai")
    if sdk_version != "2.16.0":
        raise ValueError(f"formal lock requires google-genai==2.16.0; observed {sdk_version}")
    max_spend = _required_positive_environment_float("GEMINI_MAX_SPEND_USD")
    er2_cost_cap = _required_positive_environment_float("GEMINI_ER2_COST_CAP_PER_REQUEST_USD")
    concurrency = configured_concurrency() if args.concurrency is None else int(args.concurrency)
    if concurrency < 1:
        raise ValueError("formal lock concurrency must be positive")
    locked_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = build_lock_payload(
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        locked_at_utc=locked_at,
        source_code=package_sources,
        full_run_plan=_checked_file_identity(args.full_run_plan),
        cohort_manifests={
            Path(path).stem.removesuffix("_manifest"): _checked_file_identity(path)
            for path in args.cohort_manifests
        },
        git_commit=commit,
        git_diff_sha256=observed_diff,
        checkpoint=checkpoint,
        config=config_identity,
        baseline_candidates={
            "features": candidate_features,
            "candidate_identity_stream_sha256": observed_candidate_identity,
        },
        split_manifest=split_identity,
        development_protocol_lock=development_protocol_identity,
        renderer=renderer_identity,
        prompt=prompt_identity,
        response_schema={
            "identity_kind": "canonical_json_sha256",
            "sha256": sha256_json(response_json_schema()),
            "source": schema_source_identity,
        },
        evidence_schema=evidence_identity,
        candidate_permutation={
            **renderer_identity,
            "algorithm": "deterministic_candidate_mapping(seed=47, sha256-derived per-sample permutation)",
            "function": "deterministic_candidate_mapping",
        },
        model_ids=["gemini-robotics-er-2-preview", "gemini-3.6-flash"],
        sdk="google-genai==2.16.0",
        endpoint="https://generativelanguage.googleapis.com/v1beta/interactions",
        store=False,
        background=False,
        stream=False,
        tools_enabled=False,
        previous_interaction=None,
        thinking_level="medium",
        temperature_policy="model_default",
        image_resolution=args.image_resolution,
        max_output_tokens=args.max_output_tokens,
        safe_thresholds={"values": thresholds_values, "source": thresholds_identity},
        calibration_grid=_checked_file_identity(args.calibration_grid),
        harmful_cap=float(args.harmful_cap),
        primary_selection_rule=args.primary_selection_rule,
        primary_method=args.primary_method,
        primary_selection={"values": primary_values, "source": primary_identity},
        secondary_methods=args.secondary_methods,
        request_hash_algorithm="sha256 canonical JSON v1",
        cache_schema=args.cache_schema,
        budget={
            "currency": "USD",
            "max_spend_usd": max_spend,
            "er2_cost_cap_per_request_usd": er2_cost_cap,
            "flash_pricing_source": args.flash_pricing_source,
        },
        retry_policy={"max_retries": 5, "retryable": [429, 500, 502, 503, 504]},
        concurrency=concurrency,
        transport="standard_interactions",
        validation_metrics={"values": validation_values, "source": validation_identity},
        validation_artifacts={
            Path(path).name: _checked_file_identity(path) for path in args.validation_artifacts
        },
        ground_truth_inputs={
            "legacy_labels": _checked_file_identity(args.test_legacy_labels),
            "corrected_labels": _checked_file_identity(args.test_corrected_labels),
            "raw_predictions_with_gt": _checked_file_identity(args.test_raw_predictions),
        },
        formal_test_expected_sample_count=int(args.formal_test_expected_sample_count),
        formal_test_expected_request_count=int(args.formal_test_expected_request_count),
        evaluator={
            "legacy": {
                "definition": "legacy_official_impl_v1",
                "source": _checked_file_identity(REPO_ROOT / "utils/grasp_eval.py"),
            },
            "corrected": {
                "definition": "corrected_geometric_v2",
                "source": _checked_file_identity(REPO_ROOT / "utils/grasp_metrics.py"),
            },
        },
    )
    verification = verify_lock_payload(payload, repo_root=REPO_ROOT)
    result = lock_experiment(args.output, payload, dry_run=bool(args.dry_run))
    return {**result, "verification": verification}


def command_verify_experiment_lock(args: argparse.Namespace) -> dict[str, Any]:
    return verify_experiment_lock(
        args.lock,
        repo_root=REPO_ROOT,
        expected_run_id=args.run_id,
    )


def command_claim_formal_test(args: argparse.Namespace) -> dict[str, Any]:
    return claim_formal_test_once(
        args.claim,
        args.lock,
        run_id=args.run_id,
        repo_root=REPO_ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CROG frozen Top-5 Gemini evidence experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase0 = subparsers.add_parser("phase0")
    phase0.add_argument("--output", required=True)
    phase0.add_argument("--verify-v2-tree", action="store_true")
    phase0.set_defaults(func=command_phase0)
    select = subparsers.add_parser("select-smoke")
    select.add_argument("--output", required=True)
    select.add_argument("--count", type=int, default=10)
    select.set_defaults(func=command_select_smoke)
    export = subparsers.add_parser("export-evidence")
    export.add_argument("--split", choices=("train", "val", "test"), required=True)
    export.add_argument("--selected-local-ids", required=True)
    export.add_argument("--frozen-features", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--system-prompt", default=str(REPO_ROOT / "prompts/crog_gemini_evidence_system_v1.txt"))
    export.add_argument("--device", default="auto")
    export.add_argument("--keep-dense-maps", action="store_true")
    export.set_defaults(func=command_export)
    smoke = subparsers.add_parser("run-smoke")
    smoke.add_argument("--request-manifest", required=True)
    smoke.add_argument("--evidence-schema", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--system-prompt", default=str(REPO_ROOT / "prompts/crog_gemini_evidence_system_v1.txt"))
    smoke.add_argument("--models", nargs="+", default=["gemini-robotics-er-2-preview", "gemini-3.6-flash"])
    smoke.add_argument("--image-resolution", choices=("medium", "high"), default="high")
    smoke.add_argument("--request-upper-bound-usd", type=float, default=0.10)
    smoke.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    smoke.add_argument("--sample-ids", nargs="*")
    smoke.add_argument("--limit", type=int)
    smoke.set_defaults(func=command_run_smoke)
    verify = subparsers.add_parser("verify-v2")
    verify.set_defaults(func=command_verify_v2)
    lock = subparsers.add_parser("lock-experiment")
    lock.add_argument("--output", required=True)
    lock.add_argument("--experiment-id", required=True)
    lock.add_argument("--run-id", required=True)
    lock.add_argument("--git-diff-sha256", required=True)
    lock.add_argument("--full-run-plan", required=True)
    lock.add_argument("--cohort-manifests", nargs=6, required=True)
    lock.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    lock.add_argument("--checkpoint-sha256", required=True)
    lock.add_argument("--config", default=str(DEFAULT_CONFIG))
    lock.add_argument("--candidate-features", default=str(TEST_ROOT / "features.jsonl"))
    lock.add_argument("--candidate-sha256", required=True)
    lock.add_argument("--split-manifest", default=str(V2_ROOT / "split_manifest.json"))
    lock.add_argument("--development-protocol-lock", required=True)
    lock.add_argument("--system-prompt", required=True)
    lock.add_argument("--evidence-schema", required=True)
    lock.add_argument("--thresholds", required=True)
    lock.add_argument("--calibration-grid", required=True)
    lock.add_argument("--validation-metrics", required=True)
    lock.add_argument("--validation-artifacts", nargs="+", required=True)
    lock.add_argument("--test-legacy-labels", required=True)
    lock.add_argument("--test-corrected-labels", required=True)
    lock.add_argument("--test-raw-predictions", required=True)
    lock.add_argument("--primary-selection", required=True)
    lock.add_argument("--primary-method", required=True)
    lock.add_argument("--secondary-methods", nargs="*", default=[])
    lock.add_argument("--image-resolution", default="high")
    lock.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    lock.add_argument("--harmful-cap", type=float, default=0.01)
    lock.add_argument(
        "--primary-selection-rule",
        default="validation Legacy positive net, harmful<=1pp, then outcome precision/corrected net/bootstrap/cost",
    )
    lock.add_argument("--concurrency", type=int)
    lock.add_argument("--cache-schema", default="gemini_cache.sqlite responses v1 + request_state v1")
    lock.add_argument("--flash-pricing-source", default="official Gemini API pricing verified before run")
    lock.add_argument("--formal-test-expected-sample-count", type=int, default=17749)
    lock.add_argument("--formal-test-expected-request-count", type=int, default=35498)
    lock.add_argument("--dry-run", action="store_true")
    lock.set_defaults(func=command_lock)
    verify_lock = subparsers.add_parser("verify-experiment-lock")
    verify_lock.add_argument("--lock", required=True)
    verify_lock.add_argument("--run-id")
    verify_lock.set_defaults(func=command_verify_experiment_lock)
    claim = subparsers.add_parser("claim-formal-test")
    claim.add_argument("--lock", required=True)
    claim.add_argument("--claim", required=True)
    claim.add_argument("--run-id", required=True)
    claim.set_defaults(func=command_claim_formal_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
