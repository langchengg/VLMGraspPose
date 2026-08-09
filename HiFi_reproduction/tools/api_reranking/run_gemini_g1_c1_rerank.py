#!/usr/bin/env python3
"""Pure Gemini G1/C1 frozen Top-5 experiment lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.api_only_gemini_rerank.audit import freeze_candidate_table, run_audit  # noqa: E402
from src.grasping.api_only_gemini_rerank.baseline import run_baseline_recompute  # noqa: E402
from src.grasping.api_only_gemini_rerank.io import atomic_json, atomic_parquet, initialize_run, sha256_file, utc_now  # noqa: E402
from src.grasping.api_only_gemini_rerank.splits import build_scene_split  # noqa: E402
from src.grasping.api_only_gemini_rerank.evidence import evidence_schema, extract_sample_evidence, shared_evidence_context  # noqa: E402
from src.grasping.api_only_gemini_rerank.renderer import render_boards, renderer_hash  # noqa: E402
from src.grasping.api_only_gemini_rerank.cohorts import build_stage_cohorts  # noqa: E402
from src.grasping.api_only_gemini_rerank.preflight import create_preflight, require_paid_preflight  # noqa: E402
from src.grasping.api_only_gemini_rerank.stages import run_provider_stage  # noqa: E402
from src.grasping.api_only_gemini_rerank.locking import create_go_lock, require_formal_authorization  # noqa: E402
from src.grasping.api_only_gemini_rerank.local_finalize import finalize_local  # noqa: E402
from src.grasping.api_only_gemini_rerank.provider_finalize import finalize_provider_run  # noqa: E402
from src.grasping.api_only_gemini_rerank.policy import build_policy_confirmation_manifest, sweep_policy_selection  # noqa: E402
from src.grasping.api_only_gemini_rerank.diagnostic import evaluate_diagnostic  # noqa: E402
from src.grasping.api_only_gemini_rerank.validation import (  # noqa: E402
    build_validation_confirmation_manifest, create_development_policy_lock,
    evaluate_untouched_validation,
)
from src.grasping.api_only_gemini_rerank.formal import (  # noqa: E402
    build_formal_confirmation_manifest, evaluate_formal, initialize_formal_state,
)
from src.grasping.api_only_gemini_rerank.continuation import (  # noqa: E402
    run_continuation_audit,
    run_provider_metadata_preflight,
)
from src.grasping.common.sample_io import read_deployment_manifest  # noqa: E402
from src.grasping.common.sample_io import CompactSampleLoader  # noqa: E402


DEFAULT_SOURCE_RUN = PROJECT_ROOT / "runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"


def _default_run() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "runs" / f"api_only_gemini_rerank_g1_c1_{stamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "audit", "recompute-baseline", "freeze-candidates", "build-evidence", "test",
        "smoke", "diagnostic", "select-policy", "validate", "lock", "formal", "report",
        "continuation-audit", "provider-preflight",
    ))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--backend", choices=("G1", "C1", "both"), default="both")
    parser.add_argument("--model", choices=("er2", "flash", "both"), default="both")
    parser.add_argument("--protocol")
    parser.add_argument("--evidence-variant")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-provider-requests", type=int)
    parser.add_argument("--max-api-cost-usd", type=float)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--allow-formal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-request", action="store_true")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> Path:
    if args.run_dir is None:
        if args.command not in {"audit", "continuation-audit"}:
            raise ValueError("--run-dir is required after audit")
        args.run_dir = _default_run()
    return initialize_run(args.run_dir)


def _freeze(source: Path, run: Path, backends: Sequence[str]) -> dict[str, object]:
    inventory: dict[str, object] = {}
    for backend in backends:
        parts = [freeze_candidate_table(source, split, backend) for split in ("validation", "test")]
        combined = pd.concat(parts, ignore_index=True)
        path = run / f"CANDIDATE_MANIFEST_{backend}.parquet"
        atomic_parquet(path, combined)
        inventory[backend] = {
            "path": str(path), "sha256": sha256_file(path), "rows": len(combined),
            "validation_rows": int((combined["split"] == "validation").sum()),
            "test_rows": int((combined["split"] == "test").sum()),
        }
    atomic_json(run / "audit/candidate_freeze.json", inventory)
    return inventory


def _build_evidence(source: Path, run: Path, backends: Sequence[str]) -> dict[str, object]:
    results: dict[str, object] = {}
    split_table = pd.read_csv(run / "DATA_SPLIT.csv")
    deployment_by_split = {
        split: read_deployment_manifest(source / "manifests" / f"{split}_samples.parquet")
        for split in ("validation", "test")
    }
    manifests = {
        backend: pd.read_parquet(run / f"CANDIDATE_MANIFEST_{backend}.parquet")
        for backend in backends
    }
    rows_by_backend: dict[str, list[dict[str, object]]] = {backend: [] for backend in backends}
    loader = CompactSampleLoader()
    for split in ("validation", "test"):
        groups = {
            backend: {str(sample_id): group for sample_id, group in manifests[backend].loc[manifests[backend]["split"].eq(split)].groupby("sample_id", sort=False)}
            for backend in backends
        }
        for deployment in deployment_by_split[split]:
            sample_id = str(deployment["sample_id"])
            candidate_groups = {backend: groups[backend].get(sample_id) for backend in backends}
            if not any(group is not None and not group.empty for group in candidate_groups.values()):
                continue
            arrays = loader.load(deployment, mask_source="predicted", load_intrinsics=False)
            shared = shared_evidence_context(arrays)
            for backend, group in candidate_groups.items():
                if group is not None and not group.empty:
                    rows_by_backend[backend].extend(extract_sample_evidence(group, deployment, loader, arrays=arrays, shared=shared))
    for backend in backends:
        features = pd.DataFrame(rows_by_backend[backend])
        path = run / f"EVIDENCE_FEATURES_{backend}.parquet"
        atomic_parquet(path, features)
        results[backend] = {"path": str(path), "sha256": sha256_file(path), "rows": len(features)}
    schema = evidence_schema()
    schema["renderer_hash"] = renderer_hash()
    atomic_json(run / "EVIDENCE_SCHEMA.json", schema)
    prompt_source = PROJECT_ROOT / "prompts/api_only_gemini_rerank"
    prompt_hashes = {}
    for name in ("direct_full_list_v1.txt", "baseline_aware_v1.txt", "api_rerank_v1.schema.json"):
        destination = run / "prompts" / name
        shutil.copy2(prompt_source / name, destination)
        prompt_hashes[name] = sha256_file(destination)
    deployment_validation = {str(row["sample_id"]): row for row in deployment_by_split["validation"]}
    inspection: list[dict[str, object]] = []
    for backend in backends:
        manifest = pd.read_parquet(run / f"CANDIDATE_MANIFEST_{backend}.parquet")
        allowed = set(split_table.loc[split_table["cohort"].eq("prompt_dev"), "sample_id"].astype(str))
        groups = [
            (str(sample_id), group)
            for sample_id, group in manifest.loc[manifest["split"].eq("validation")].groupby("sample_id", sort=True)
            if str(sample_id) in allowed and 2 <= len(group) <= 5
        ][:10]
        for index, (sample_id, group) in enumerate(groups):
            destination = run / "boards/smoke" / f"inspection_{backend.lower()}_{index:02d}_{sample_id}"
            _, _, metadata = render_boards(
                group.to_dict(orient="records"),
                deployment_validation[sample_id],
                seed=args_seed(backend, sample_id),
                evidence_variant="E3_RGBD_GEOMETRY_SCORE_AWARE",
                output_dir=destination,
            )
            inspection.append({"backend": backend, "sample_id": sample_id, "directory": str(destination), **metadata})
    atomic_json(run / "audit/board_inspection_manifest.json", {"count": len(inspection), "boards": inspection})
    results["prompt_hashes"] = prompt_hashes
    results["inspection_samples"] = len(inspection)
    return results


def args_seed(backend: str, sample_id: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(f"20260805:{backend}:{sample_id}".encode()).digest()[:8], "big")


def _write_partial_reports(run: Path) -> dict[str, object]:
    baseline = json.loads((run / "BASELINE_RECOMPUTE.json").read_text()) if (run / "BASELINE_RECOMPUTE.json").exists() else {}
    preflight = create_preflight(run) if (run / "STAGE_COHORTS.parquet").exists() else {"ready_for_paid_api": False, "blockers": ["Stage cohorts absent"]}
    ledger = json.loads((run / "API_LEDGER_SUMMARY.json").read_text()) if (run / "API_LEDGER_SUMMARY.json").exists() else {
        "logical_requests": 0, "provider_attempts": 0, "cache_hits": 0,
    }
    result = {
        "schema_version": 1, "experiment_status": "BLOCKED_BEFORE_PAID_API" if not preflight.get("ready_for_paid_api") else "DEVELOPMENT",
        "metric_name": "OCID-VLG offline 2D grasp-rectangle consistency",
        "baseline": baseline, "api_ledger": ledger, "preflight": preflight,
        "formal_test_executed": False,
        "G1_final_primary": "USE_G1_ORIGINAL_SCORE",
        "C1_final_primary": "USE_C1_ORIGINAL_SCORE",
    }
    atomic_json(run / "RESULTS.json", result)
    text = [
        "# Pure Gemini API frozen-candidate reranking", "",
        "Stage 0 baseline/candidate freeze is complete. Paid API stages are not scientific results until untouched validation completes.", "",
        "## Current status", "", f"- Paid API ready: {preflight.get('ready_for_paid_api', False)}",
        f"- API logical requests: {ledger.get('logical_requests', 0)}", f"- Provider attempts: {ledger.get('provider_attempts', 0)}",
        "- Formal test: not executed", "- Current safe primary for both backends: original backend score", "",
        "All J@1 numbers mean OCID-VLG offline 2D grasp-rectangle consistency, not physical grasp success.",
    ]
    (run / "SUMMARY.md").write_text("\n".join(text)+"\n")
    zh = [
        "# 纯 Gemini API 冻结候选重排序", "", "Stage 0 的基线复算和候选冻结已完成。付费 API 阶段完成 untouched validation 前，不构成性能结论。", "",
        f"- 付费 API 就绪：{preflight.get('ready_for_paid_api', False)}", f"- provider attempts：{ledger.get('provider_attempts', 0)}",
        "- 正式测试：未执行", "- 当前 G1/C1 主方法：保留原始 backend score", "",
        "J@1 仅表示 OCID-VLG 离线二维抓取矩形一致性，不代表真实机器人抓取成功率。",
    ]
    (run / "SUMMARY_ZH.md").write_text("\n".join(zh)+"\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = _run(args)
    source = args.source_run.expanduser().resolve()
    backends = ("G1", "C1") if args.backend == "both" else (args.backend,)
    if args.command == "continuation-audit":
        result = run_continuation_audit(PROJECT_ROOT, source, run)
    elif args.command == "provider-preflight":
        result = run_provider_metadata_preflight(PROJECT_ROOT, source, run)
    elif args.command == "audit":
        result = run_audit(PROJECT_ROOT, source, run)
        manifest = {
            "schema_version": 1, "experiment": "api_only_gemini_rerank_g1_c1",
            "created_at_utc": utc_now(), "repo_root": str(PROJECT_ROOT),
            "source_run": str(source), "git_commit": result["git_commit"],
            "git_dirty": result["git_dirty"],
            "exact_model_ids": ["gemini-robotics-er-2-preview", "gemini-3.6-flash"],
            "seed": 20260805, "temperature_policy": "lowest endpoint-supported; omit deprecated fields for Gemini 3.6 Flash",
            "fallback_policy": "original backend Top-1 for every provider/schema/mapping/timeout failure",
            "formal_allow_flags": {"ALLOW_FORMAL_GEMINI_RERANK": False, "ALLOW_PAID_API_RUN": False},
        }
        atomic_json(run / "MANIFEST.json", manifest)
    elif args.command == "recompute-baseline":
        result = run_baseline_recompute(source, run)
    elif args.command == "freeze-candidates":
        result = {"candidates": _freeze(source, run, backends), "split": build_scene_split(source, run)}
    elif args.command == "build-evidence":
        result = _build_evidence(source, run, backends)
        result["cohorts"] = build_stage_cohorts(run, seed=args.seed)
        result["preflight"] = create_preflight(run)
    elif args.command == "test":
        command = [sys.executable, "-m", "pytest", "-q", "tests/test_api_only_gemini_rerank.py"]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        compile_result = subprocess.run([sys.executable, "-m", "compileall", "-q", "src/grasping/api_only_gemini_rerank", "tools/api_reranking"], cwd=PROJECT_ROOT, check=False)
        result = {"pytest_exit_code": completed.returncode, "compileall_exit_code": compile_result.returncode}
        if completed.returncode or compile_result.returncode:
            raise RuntimeError(f"test verification failed: {result}")
    elif args.command in {"smoke", "diagnostic", "select-policy"}:
        stage = {"smoke": "smoke", "diagnostic": "diagnostic", "select-policy": "policy_selection"}[args.command]
        result = run_provider_stage(
            run, source, stage=stage, backends=backends, model_selection=args.model,
            max_samples=args.max_samples, max_provider_requests=args.max_provider_requests,
            max_api_cost_usd=args.max_api_cost_usd, replay_only=args.replay_only,
            dry_run=args.dry_run, concurrency=args.concurrency,
        )
        if args.command == "diagnostic" and not args.dry_run and not args.replay_only:
            result["diagnostic_evaluation"] = evaluate_diagnostic(run)
        if args.command == "select-policy" and not args.dry_run and not args.replay_only:
            result["policy_preselection"] = sweep_policy_selection(run)
            confirmation = build_policy_confirmation_manifest(run)
            result["confirmation_manifest"] = confirmation
            if confirmation["rows"]:
                result["confirmation_run"] = run_provider_stage(
                    run, source, stage="policy_confirmation", backends=backends,
                    model_selection=args.model, max_samples=None,
                    max_provider_requests=args.max_provider_requests,
                    max_api_cost_usd=args.max_api_cost_usd,
                    replay_only=False, dry_run=False, concurrency=args.concurrency,
                )
                result["policy_selection"] = sweep_policy_selection(run)
    elif args.command == "validate":
        if args.dry_run:
            result = {"dry_run": True, "would_require": ["POLICY_PRESELECTION.json", "DIAGNOSTIC_REPORT.json", "DEVELOPMENT_POLICY_LOCK.json"], "writes": False}
            print(json.dumps({"run_dir": str(run), "command": args.command, "result": result}, indent=2, default=str))
            return 0
        create_development_policy_lock(run)
        result = run_provider_stage(
            run, source, stage="untouched_validation", backends=backends,
            model_selection=args.model, max_samples=args.max_samples,
            max_provider_requests=args.max_provider_requests,
            max_api_cost_usd=args.max_api_cost_usd, replay_only=args.replay_only,
            dry_run=args.dry_run, concurrency=args.concurrency,
        )
        if not args.dry_run and not args.replay_only:
            confirmation = build_validation_confirmation_manifest(run)
            result["confirmation_manifest"] = confirmation
            if confirmation["rows"]:
                result["confirmation_run"] = run_provider_stage(
                    run, source, stage="validation_confirmation", backends=backends,
                    model_selection=args.model, max_samples=None,
                    max_provider_requests=args.max_provider_requests,
                    max_api_cost_usd=args.max_api_cost_usd,
                    replay_only=False, dry_run=False, concurrency=args.concurrency,
                )
            result["validation"] = evaluate_untouched_validation(run)
    elif args.command == "lock":
        validation_path = run / "VALIDATION_RESULTS.json"
        if not validation_path.is_file():
            raise RuntimeError("GO lock forbidden: untouched validation results are absent")
        validation = json.loads(validation_path.read_text())
        locked = {}
        development = json.loads((run / "DEVELOPMENT_POLICY_LOCK.json").read_text())
        for backend in backends:
            decision = validation["backend_decisions"][backend]
            if decision["decision"] != "GO":
                locked[backend] = {"status": decision["decision"], "lock_created": False}
                continue
            primary = str(decision["primary"])
            if "CONSENSUS" in primary:
                required_models = ["gemini-robotics-er-2-preview", "gemini-3.6-flash"]
            elif "robotics-er-2" in primary:
                required_models = ["gemini-robotics-er-2-preview"]
            elif "3.6-flash" in primary:
                required_models = ["gemini-3.6-flash"]
            else:
                raise RuntimeError("validation GO primary is not an exact Gemini method")
            model_availability = {
                model: bool(development["per_backend_model"][f"{backend}:{model}"].get("model_available", True))
                for model in ("gemini-robotics-er-2-preview", "gemini-3.6-flash")
            }
            metric = next(
                (row for row in validation["metrics"] if row["method"] == primary), None
            )
            if metric is None or metric.get("validation_decision") != "GO":
                raise RuntimeError("matching untouched-validation GO metric is absent")
            payload = {
                "validation_decision": "GO", "locked_primary": primary,
                "required_models": required_models,
                "model_availability": model_availability,
                "validation_metric": metric,
                "development_policy_lock_sha256": sha256_file(run / "DEVELOPMENT_POLICY_LOCK.json"),
                "candidate_manifest_sha256": sha256_file(run / f"CANDIDATE_MANIFEST_{backend}.parquet"),
                "validation_results_sha256": sha256_file(validation_path),
                "policy": development["per_backend_model"],
                "formal_expected_samples": 7675,
            }
            locked[backend] = create_go_lock(run, backend, payload, dry_run=args.dry_run)
        result = locked
    elif args.command == "formal":
        for backend in backends:
            require_formal_authorization(run, backend, allow_formal_argument=args.allow_formal)
        if args.dry_run:
            result = {"dry_run": True, "verified_backends": list(backends), "writes": False}
            print(json.dumps({"run_dir": str(run), "command": args.command, "result": result}, indent=2, default=str))
            return 0
        required_models = set()
        for backend in backends:
            primary = str(json.loads((run/f"LOCKED_MANIFEST_{backend}.json").read_text())["locked_primary"])
            if "CONSENSUS" in primary or "robotics-er-2" in primary:
                required_models.add("gemini-robotics-er-2-preview")
            if "CONSENSUS" in primary or "3.6-flash" in primary:
                required_models.add("gemini-3.6-flash")
        selected_models = set(("gemini-robotics-er-2-preview", "gemini-3.6-flash") if args.model == "both" else ({"er2": "gemini-robotics-er-2-preview", "flash": "gemini-3.6-flash"}[args.model],))
        if not required_models.issubset(selected_models):
            raise RuntimeError("--model selection omits a model required by the frozen formal primary")
        initialize_formal_state(run, backends)
        result = run_provider_stage(
            run, source, stage="formal", backends=backends, model_selection=args.model,
            max_samples=args.max_samples, max_provider_requests=args.max_provider_requests,
            max_api_cost_usd=args.max_api_cost_usd, replay_only=args.replay_only,
            dry_run=args.dry_run, concurrency=args.concurrency,
        )
        if not args.dry_run and not args.replay_only:
            confirmation=build_formal_confirmation_manifest(run); result["confirmation_manifest"]=confirmation
            if confirmation["rows"]:
                result["confirmation_run"]=run_provider_stage(
                    run, source, stage="formal_confirmation", backends=backends,
                    model_selection=args.model, max_samples=None,
                    max_provider_requests=args.max_provider_requests,
                    max_api_cost_usd=args.max_api_cost_usd,
                    replay_only=False, dry_run=False, concurrency=args.concurrency,
                )
            result["formal_results"]=evaluate_formal(run, backends)
    elif args.command == "report":
        ledger_path = run / "api_ledger.sqlite"
        has_attempts = False
        if ledger_path.is_file():
            import sqlite3
            connection = sqlite3.connect(ledger_path)
            try:
                has_attempts = int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]) > 0
            finally:
                connection.close()
        result = finalize_provider_run(run) if has_attempts else finalize_local(run)
    else:
        raise AssertionError(args.command)
    print(json.dumps({"run_dir": str(run), "command": args.command, "result": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
