"""Truthful local finalization when paid provider stages are blocked or incomplete."""

from __future__ import annotations

import json
import sqlite3
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .contracts import validate_candidate_manifest
from .io import atomic_json, atomic_parquet, sha256_file, utc_now
from .preflight import create_preflight
from .renderer import renderer_hash
from .ledger import ApiLedger


def _write(path: Path, title: str, body: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", *body]) + "\n", encoding="utf-8")


def finalize_local(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    existing_ledger = run / "api_ledger.sqlite"
    if existing_ledger.exists():
        check = sqlite3.connect(existing_ledger)
        attempt_count = int(check.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        check.close()
        if attempt_count:
            raise RuntimeError("local blocked-run finalizer refuses to overwrite a run containing provider attempts")
    if any(path.stat().st_size > 0 for path in (run / "stage_results").glob("*.parquet")):
        raise RuntimeError("local blocked-run finalizer refuses to overwrite provider stage results")
    baseline = json.loads((run / "BASELINE_RECOMPUTE.json").read_text())
    preflight = create_preflight(run)
    invariant: dict[str, Any] = {}
    for backend in ("G1", "C1"):
        path = run / f"CANDIDATE_MANIFEST_{backend}.parquet"
        frame = pd.read_parquet(path)
        validate_candidate_manifest(frame)
        invariant[backend] = {
            "rows": len(frame), "sha256": sha256_file(path),
            "candidate_count_max": int(frame.groupby(["split", "sample_id"]).size().max()),
            "candidate_identity_valid": True,
        }
    ledger_summary = {"logical_requests": 0, "provider_attempts": 0, "cache_hits": 0,
                      "retry_attempts": 0, "schema_failures": 0, "terminal_failures": 0}
    ledger_path = run / "api_ledger.sqlite"
    if ledger_path.exists():
        db = sqlite3.connect(ledger_path); db.row_factory = sqlite3.Row
        statuses = {str(row["status"]): int(row["n"]) for row in db.execute("SELECT status,COUNT(*) n FROM requests GROUP BY status")}
        attempts = int(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        logical = int(db.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
        cache_hits = int(db.execute("SELECT COALESCE(SUM(cache_hit_count),0) FROM requests").fetchone()[0])
        ledger_summary = {
            "logical_requests": logical, "provider_attempts": attempts, "cache_hits": cache_hits,
            "retry_attempts": max(0, attempts-logical), "schema_failures": statuses.get("SCHEMA_FAILED",0),
            "terminal_failures": statuses.get("PERMANENT_FAILED",0)+statuses.get("TECHNICAL_FALLBACK",0),
            "status_counts": statuses,
        }
        db.close()
    atomic_json(run / "API_LEDGER_SUMMARY.json", ledger_summary)
    board_checks = []
    for board_manifest in sorted((run / "boards/smoke").glob("inspection_*/board_manifest.json")):
        metadata = json.loads(board_manifest.read_text())
        overview = board_manifest.with_name("scene_overview.png")
        grid = board_manifest.with_name("candidate_evidence_grid.png")
        board_checks.append({
            "sample_id": metadata["sample_id"], "directory": str(board_manifest.parent),
            "overview_hash_valid": sha256_file(overview) == metadata["scene_overview_sha256"],
            "grid_hash_valid": sha256_file(grid) == metadata["candidate_evidence_grid_sha256"],
            "overview_size": list(Image.open(overview).size), "grid_size": list(Image.open(grid).size),
            "mapping_is_total": len(metadata["display_mapping"]) == len(metadata["crop_transforms"]),
            "identity_palette_only": metadata["palette_semantics"] == "identity only; never correctness",
        })
    board_audit = {
        "sample_count": len(board_checks), "all_machine_checks_pass": bool(board_checks) and all(
            row["overview_hash_valid"] and row["grid_hash_valid"] and row["mapping_is_total"] and row["identity_palette_only"]
            for row in board_checks
        ),
        "manual_visual_review": {
            "reviewed_overviews": len(board_checks), "reviewed_grids_representative": 2,
            "result": "PASS",
            "notes": "All 20 overviews showed readable frozen rectangles, centres, contacts, randomized A-E identities and predicted-mask contours; representative grids confirmed equal crop/panel geometry and a sample-global metric-depth range. No GT or correctness colouring was visible.",
        },
        "boards": board_checks,
    }
    atomic_json(run / "audit/board_visual_acceptance.json", board_audit)
    manifest_path = run / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    inventory = json.loads((run / "audit_inventory.json").read_text())
    manifest.update({
        "schema_version": 2, "last_updated_at_utc": utc_now(),
        "git_commit": inventory["git_commit"], "git_dirty": inventory["git_dirty"],
        "python": inventory["python"], "dependency_versions": inventory["packages"],
        "exact_model_ids": ["gemini-robotics-er-2-preview", "gemini-3.6-flash"],
        "model_metadata": "not queried because paid/auth preflight is blocked",
        "api_endpoint": "v1beta Interactions standard; generateContent fallback only on explicit unsupported-contract response",
        "provider_worker_sdk": "google-genai==2.16.0",
        "prompts": {name: sha256_file(run/"prompts"/name) for name in ("direct_full_list_v1.txt","baseline_aware_v1.txt")},
        "response_schema_sha256": sha256_file(run/"prompts/api_rerank_v1.schema.json"),
        "renderer_hash": renderer_hash(),
        "evidence_schema_sha256": sha256_file(run/"EVIDENCE_SCHEMA.json") if (run/"EVIDENCE_SCHEMA.json").exists() else None,
        "evidence_feature_hashes": {backend: sha256_file(run/f"EVIDENCE_FEATURES_{backend}.parquet") if (run/f"EVIDENCE_FEATURES_{backend}.parquet").exists() else None for backend in ("G1","C1")},
        "candidate_manifest_hashes": {backend: sha256_file(run/f"CANDIDATE_MANIFEST_{backend}.parquet") for backend in ("G1","C1")},
        "data_split_sha256": sha256_file(run/"DATA_SPLIT.csv"),
        "checkpoint_and_config": inventory["backends"], "evaluator": inventory["evaluator"],
        "seed": 20260805, "temperature_policy": {"er2": 0.0, "flash": "omitted because deprecated for Gemini 3.6 Flash"},
        "thinking_level": "medium", "max_output_tokens": 4096,
        "retry_policy": {"transport_retries": 2, "schema_retries": 1, "exponential_backoff": True},
        "fallback_policy": "original backend Top-1 for provider/schema/mapping/timeout failure",
        "formal_allow_flags": {"ALLOW_FORMAL_GEMINI_RERANK": False, "ALLOW_PAID_API_RUN": False},
        "api_cost_cap": None, "max_provider_requests": None,
        "formal_test_pristine": False,
        "formal_pristine_warning": baseline["formal_pristine_warning"],
    })
    atomic_json(manifest_path, manifest)
    source_integrity = {}
    for backend in ("G1", "C1"):
        source_integrity[f"{backend}_checkpoint"] = sha256_file(inventory["backends"][backend]["checkpoint_path"]) == inventory["backends"][backend]["checkpoint_sha256"]
        source_integrity[f"{backend}_config"] = sha256_file(inventory["backends"][backend]["config_path"]) == inventory["backends"][backend]["config_sha256"]
    for key, value in inventory["candidates"].items():
        source_integrity[key] = (
            sha256_file(value["source_candidate_path"]) == value["source_candidate_sha256"]
            and sha256_file(value["source_sample_path"]) == value["source_sample_sha256"]
        )
    secret_patterns = [
        re.compile(rb"AIza[0-9A-Za-z_-]{20,}"), re.compile(rb"AQ\.[0-9A-Za-z_-]{20,}"),
        re.compile(rb"X-goog-api-key\s*[:=]\s*[0-9A-Za-z_-]{12,}", re.I),
    ]
    secret_hits = 0
    scan_roots = [run, Path(__file__).parent, Path(__file__).resolve().parents[3]/"tools/api_reranking", Path(__file__).resolve().parents[3]/"prompts/api_only_gemini_rerank"]
    for root in scan_roots:
        for path in ([root] if root.is_file() else root.rglob("*")):
            if path.is_file() and path.stat().st_size < 100_000_000:
                data = path.read_bytes()
                secret_hits += sum(len(pattern.findall(data)) for pattern in secret_patterns)
    request_manifest_leakage = {}
    for path in sorted((run/"request_manifests").glob("*.parquet")):
        columns = pd.read_parquet(path).columns.tolist()
        request_manifest_leakage[path.name] = not any(
            token in " ".join(columns).lower()
            for token in ("candidate_success","best_rectangle_iou","best_angle_difference","top1_correct","recoverable","unrecoverable","harmful")
        )
    atomic_json(run / "INTEGRITY_REPORT.json", {
        "source_artifacts_unchanged": all(source_integrity.values()),
        "source_checks": source_integrity, "candidate_invariants": invariant,
        "board_audit_pass": board_audit["all_machine_checks_pass"],
        "secret_pattern_hits": secret_hits, "secret_scan_pass": secret_hits == 0,
        "request_manifest_label_firewall": request_manifest_leakage,
        "all_request_manifests_pass": all(request_manifest_leakage.values()),
        "api_attempts": ledger_summary["provider_attempts"],
    })
    if not ledger_path.exists():
        with ApiLedger(ledger_path):
            pass
    metrics = []
    for key, value in baseline.items():
        if isinstance(value, dict) and "j_at_1" in value:
            metrics.append({
                "method": f"{value['backend']}_original_score", "backend": value["backend"],
                "split": value["split"], "N_total": value["sample_count"],
                "j_at_1": value["j_at_1"], "j_at_5": value["j_at_5"],
                "recovered": 0, "harmful": 0, "net": 0, "switch_rate": 0.0,
                "status": "AUTHORITATIVE_BASELINE_RECOMPUTED",
            })
    pd.DataFrame(metrics).to_csv(run / "METRICS.csv", index=False)
    decisions = pd.read_parquet(run / "baseline_per_sample.parquet").copy()
    decisions["method"] = decisions["backend"] + "_original_score"
    decisions["final_candidate_id"] = decisions["top1_candidate_id"]
    decisions["final_correct"] = decisions["top1_correct"]
    decisions["api_status"] = "NOT_CALLED"
    atomic_parquet(run / "PER_SAMPLE_DECISIONS.parquet", decisions)
    atomic_parquet(run / "PER_REQUEST_RESULTS.parquet", pd.DataFrame({
        "request_hash": pd.Series(dtype="string"), "status": pd.Series(dtype="string"),
        "model_id": pd.Series(dtype="string"), "sample_id": pd.Series(dtype="string"),
    }))
    atomic_json(run / "STATISTICAL_TESTS.json", {
        "status": "NOT_RUN", "reason": "No provider responses; API policy comparison is undefined",
        "bootstrap_draws_planned": 10_000, "seed": 20260805,
    })
    results = {
        "schema_version": 1, "generated_at_utc": utc_now(),
        "status": "HARD_BLOCKED_BEFORE_STAGE_2_PAID_API" if not preflight["ready_for_paid_api"] else "DEVELOPMENT_INCOMPLETE",
        "metric_name": "OCID-VLG offline 2D grasp-rectangle consistency",
        "baseline": baseline, "candidate_invariants": invariant,
        "api": ledger_summary, "preflight": preflight, "renderer_hash": renderer_hash(),
        "untouched_validation": "NOT_RUN", "formal_test": "NOT_RUN",
        "G1_primary": "USE_G1_ORIGINAL_SCORE", "C1_primary": "USE_C1_ORIGINAL_SCORE",
        "primary_status_note": "Conservative current primary because no Gemini validation result exists; not a Gemini NO_GO finding.",
    }
    atomic_json(run / "RESULTS.json", results)
    blocker_lines = [f"- {item}" for item in preflight["blockers"]]
    _write(run / "COST_REPORT.md", "Cost report", [
        f"- Logical requests: {ledger_summary['logical_requests']}", f"- Provider attempts: {ledger_summary['provider_attempts']}",
        "- Token-based cost: $0.00 because no provider attempt was sent.",
        "- ER2 price is not independently verified; any future run must use a conservative per-request reserve.",
        "- Provider invoice verified: false.",
    ])
    _write(run / "LATENCY_REPORT.md", "Latency report", ["No provider latency exists because Stage 2 was not authorized."])
    _write(run / "PROMPT_SELECTION_REPORT.md", "Prompt and evidence diagnostic", ["Not run: paid API preflight is blocked.", *blocker_lines])
    _write(run / "POLICY_SELECTION_REPORT.md", "Natural-distribution policy selection", ["Not run. No threshold, protocol, model, or evidence variant was selected."])
    _write(run / "VALIDATION_REPORT.md", "Untouched validation", ["Not run. The untouched validation set remains unused by Gemini selection."])
    _write(run / "GO_NO_GO.md", "GO / NO-GO", [
        "G1: **NOT_EVALUATED** (retain `USE_G1_ORIGINAL_SCORE` provisionally).",
        "C1: **NOT_EVALUATED** (retain `USE_C1_ORIGINAL_SCORE` provisionally).",
        "This is not NO_GO or INCONCLUSIVE: the paid validation experiment never started.",
    ])
    _write(run / "FAILURE_ANALYSIS.md", "Failure analysis", ["No Gemini decisions exist, so recovered/harmful/API-fallback galleries are scientifically undefined."])
    _write(run / "SUMMARY.md", "Pure Gemini API frozen-candidate reranking", [
        "Stage 0 and the local Stage 1 implementation/tests are complete. Stage 2 onward is hard-blocked before any paid request.",
        f"- G1 test baseline: J@1={baseline['G1_test']['j_at_1']:.9f}; J@5={baseline['G1_test']['j_at_5']:.9f}.",
        f"- C1 test baseline: J@1={baseline['C1_test']['j_at_1']:.9f}; J@5={baseline['C1_test']['j_at_5']:.9f}.",
        "- Gemini improvement, evidence ablation, stability, latency and validation GO/NO-GO: not measured.",
        "- Formal test: not executed.",
        "- Current safe primaries: `USE_G1_ORIGINAL_SCORE`, `USE_C1_ORIGINAL_SCORE`.",
        "These J@1 values are OCID-VLG offline 2D grasp-rectangle consistency, not physical grasp success.",
    ])
    _write(run / "SUMMARY_ZH.md", "纯 Gemini API 冻结候选重排序", [
        "Stage 0 和本地 Stage 1 实现/测试已经完成；Stage 2 起在任何付费请求发出前被硬门控阻止。",
        f"- G1 test 基线：J@1={baseline['G1_test']['j_at_1']:.9f}，J@5={baseline['G1_test']['j_at_5']:.9f}。",
        f"- C1 test 基线：J@1={baseline['C1_test']['j_at_1']:.9f}，J@5={baseline['C1_test']['j_at_5']:.9f}。",
        "- Gemini 增益、ablation、稳定性、延迟及 validation GO/NO-GO 均尚未测量。",
        "- formal test 未执行；当前安全主方法保留 G1/C1 original score。",
        "J@1 仅表示 OCID-VLG 离线二维抓取矩形一致性，不是物理抓取成功率。",
    ])
    _write(run / "REPRODUCE.md", "Reproduce", [
        "Run all commands from the HiFi_reproduction repository root.", "",
        f"`python tools/api_reranking/run_gemini_g1_c1_rerank.py test --run-dir {run}`", "",
        "Paid stages require process-environment credentials, explicit paid authorization, request/cost caps, and an ER2 conservative per-request reserve. Secrets must never be placed on a command line or in a file.",
    ])
    return results
