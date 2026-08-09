"""Provider-aware final reports for a completed or honestly partial API run."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .availability import audited_unavailable_models
from .contracts import validate_candidate_manifest
from .gallery import materialize_failure_gallery
from .io import atomic_json, atomic_parquet, sha256_file, utc_now
from .ledger import ApiLedger
from .payload import REQUEST_HASH_VERSION
from .renderer import renderer_hash
from .schema import RESPONSE_PARSER_VERSION
from src.experiments.ocid_annotations import derive_query_type, load_expression_index


PLOT_NAMES = (
    "j1_comparison.png", "recovered_harmful_net.png", "switch_rate.png",
    "recovery_recall.png", "harm_rate.png", "confidence_threshold_sweep.png",
    "evidence_ablation.png", "expression_type_results.png", "rank_transition_matrix.png",
    "model_agreement.png", "perturbation_stability.png", "latency_distribution.png",
    "token_usage.png", "cost_projection.png",
)


def _write(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")


def _placeholder(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(.5, .5, message, ha="center", va="center", wrap=True, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _bar(path: Path, frame: pd.DataFrame, *, x: str, values: list[str], title: str) -> None:
    if frame.empty or x not in frame or not all(value in frame for value in values):
        _placeholder(path, title, "Not evaluated or no applicable data.")
        return
    labels = frame[x].astype(str).tolist()
    positions = np.arange(len(labels))
    width = .8 / max(1, len(values))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * .8), 4.8))
    for index, value in enumerate(values):
        axis.bar(positions + (index - (len(values)-1)/2)*width,
                 pd.to_numeric(frame[value], errors="coerce"), width=width, label=value)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.grid(axis="y", alpha=.25)
    if len(values) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _usage_and_latency(run: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    db = sqlite3.connect(run / "api_ledger.sqlite")
    db.row_factory = sqlite3.Row
    rows = list(db.execute(
        """SELECT r.requested_model,a.status,a.latency_seconds,a.usage_json,a.estimated_cost_usd
           FROM attempts a JOIN requests r USING(request_hash) ORDER BY a.attempt_id"""
    ))
    db.close()
    token_totals: dict[str, dict[str, float]] = {}
    normalized = []
    for row in rows:
        model = str(row["requested_model"])
        usage = json.loads(row["usage_json"] or "{}")
        target = token_totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0})
        target["input_tokens"] += float(usage.get("total_input_tokens", usage.get("input_tokens", 0)) or 0)
        target["output_tokens"] += float(usage.get("total_output_tokens", usage.get("output_tokens", 0)) or 0)
        target["thought_tokens"] += float(usage.get("total_thought_tokens", usage.get("thought_tokens", 0)) or 0)
        normalized.append({
            "model_id": model, "status": str(row["status"]),
            "latency_seconds": row["latency_seconds"],
            "estimated_cost_usd": row["estimated_cost_usd"],
        })
    latency = pd.DataFrame(normalized)
    per_model = {}
    for model, group in latency.dropna(subset=["latency_seconds"]).groupby("model_id"):
        values = pd.to_numeric(group["latency_seconds"], errors="coerce").dropna()
        per_model[str(model)] = {
            "attempts": len(group), "p50_seconds": float(values.quantile(.5)),
            "p95_seconds": float(values.quantile(.95)), "mean_seconds": float(values.mean()),
        }
    db = sqlite3.connect(run / "api_ledger.sqlite")
    db.row_factory = sqlite3.Row
    request_rows = list(db.execute(
        "SELECT requested_model,status,created_at,updated_at FROM requests ORDER BY created_at"
    ))
    attempt_status_counts = {
        str(row["status"]): int(row["n"])
        for row in db.execute("SELECT status,COUNT(*) n FROM attempts GROUP BY status")
    }
    distinct_attempted = int(db.execute("SELECT COUNT(DISTINCT request_hash) FROM attempts").fetchone()[0])
    attempt_count = int(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
    db.close()
    request_wall: dict[str, Any] = {}
    for model in sorted({str(row["requested_model"]) for row in request_rows}):
        values = np.asarray([
            max(0.0, float(row["updated_at"]) - float(row["created_at"]))
            for row in request_rows
            if str(row["requested_model"]) == model and str(row["status"]) not in {"IN_FLIGHT", "RETRYABLE_FAILED"}
        ], dtype=float)
        if len(values):
            request_wall[model] = {
                "requests": int(len(values)), "p50_seconds": float(np.quantile(values, .5)),
                "p95_seconds": float(np.quantile(values, .95)), "mean_seconds": float(values.mean()),
            }
    timestamps = {
        "first_request_at_utc": None,
        "last_request_update_at_utc": None,
    }
    if request_rows:
        timestamps = {
            "first_request_at_utc": datetime.fromtimestamp(min(float(row["created_at"]) for row in request_rows), timezone.utc).isoformat().replace("+00:00", "Z"),
            "last_request_update_at_utc": datetime.fromtimestamp(max(float(row["updated_at"]) for row in request_rows), timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    return {
        "tokens": token_totals, "online_attempt_latency": per_model,
        "retry_inclusive_request_wall_latency": request_wall,
        "batch_latency": {"status": "NOT_APPLICABLE_STANDARD_INTERACTIONS_ONLY"},
        "attempt_status_counts": attempt_status_counts,
        "retry_attempts": max(0, attempt_count - distinct_attempted),
        "distinct_attempted_logical_requests": distinct_attempted,
        "execution_timestamps": timestamps,
    }, latency


def _secret_scan(run: Path) -> dict[str, Any]:
    patterns = (
        re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
        re.compile(rb"AQ\.[0-9A-Za-z_-]{20,}"),
        re.compile(rb"X-goog-api-key\s*[:=]\s*[0-9A-Za-z_-]{12,}", re.I),
    )
    roots = [run, Path(__file__).parent, Path(__file__).resolve().parents[3] / "tools/api_reranking"]
    hits: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.stat().st_size >= 100_000_000:
                continue
            data = path.read_bytes()
            if any(pattern.search(data) for pattern in patterns):
                hits.append(str(path))
    return {"secret_pattern_hits": len(hits), "secret_scan_pass": not hits, "hit_paths": hits}


def _stage_results(run: Path) -> pd.DataFrame:
    parts = []
    for path in sorted((run / "stage_results").glob("*.parquet")):
        frame = pd.read_parquet(path).copy()
        if frame.empty:
            continue
        frame["result_artifact"] = path.name
        parts.append(frame)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _query_type_annotations(run: Path) -> pd.DataFrame:
    """Load the official symbolic-program query types for reporting only."""
    manifest = json.loads((run / "MANIFEST.json").read_text())
    source = Path(manifest["source_run"])
    rows: list[dict[str, str]] = []
    index_cache: dict[str, dict[int, Mapping[str, Any]]] = {}
    verified_paths: set[str] = set()
    for split in ("validation", "test"):
        samples = pd.read_parquet(source / "manifests" / f"{split}_samples.parquet")
        labels = pd.read_parquet(source / "manifests" / f"{split}_labels.parquet")
        aligned = samples[["sample_id", "question_index", "language"]].merge(
            labels[["sample_id", "official_annotations_path", "official_annotations_sha256"]],
            on="sample_id", how="inner", validate="one_to_one",
        )
        for item in aligned.to_dict(orient="records"):
            path = str(Path(str(item["official_annotations_path"])).resolve())
            if path not in verified_paths:
                if sha256_file(path) != str(item["official_annotations_sha256"]):
                    raise RuntimeError(f"official expression annotation hash drift: {path}")
                verified_paths.add(path)
            if path not in index_cache:
                index_cache[path] = load_expression_index(path)
            expressions = index_cache[path]
            question_index = int(item["question_index"])
            expression = expressions.get(question_index)
            if expression is None or str(expression.get("question")) != str(item["language"]):
                raise RuntimeError(f"official expression alignment drift: {item['sample_id']}")
            rows.append({
                "split": split, "sample_id": str(item["sample_id"]),
                "query_type": derive_query_type(expression.get("program")).query_type,
                "query_type_rule_version": "ocid_vlg_symbolic_v1",
            })
    return pd.DataFrame(rows)


def _metrics(run: Path, baseline: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in baseline.items():
        if isinstance(value, dict) and key.endswith("_test") and "j_at_1" in value:
            rows.append({
                "source_stage": "baseline_test", "method": f"{value['backend']}_original_score",
                "backend": value["backend"], "N_total": value["sample_count"],
                "baseline_j_at_1": value["j_at_1"], "final_j_at_1": value["j_at_1"],
                "j_at_5": value["j_at_5"], "recovered": 0, "harmful": 0,
                "net": 0, "switch_rate": 0.0,
            })
    validation_path = run / "validation_metrics.csv"
    if validation_path.is_file():
        frame = pd.read_csv(validation_path)
        frame["source_stage"] = "untouched_validation"
        rows.extend(frame.to_dict(orient="records"))
    for backend in ("G1", "C1"):
        path = run / f"FORMAL_RESULTS_{backend}.json"
        if path.is_file():
            value = json.loads(path.read_text())
            rows.append({"source_stage": "formal_test", "method": value["locked_primary"],
                         "backend": backend, **value["metrics"]})
    return pd.DataFrame(rows)


def _plots(run: Path, metrics: pd.DataFrame, usage: dict[str, Any], latency: pd.DataFrame) -> None:
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    validation = metrics.loc[metrics.get("source_stage", pd.Series(dtype=str)).eq("untouched_validation")].copy()
    _bar(plots/"j1_comparison.png", validation, x="method",
         values=["baseline_j_at_1", "final_j_at_1"], title="Untouched-validation J@1")
    _bar(plots/"recovered_harmful_net.png", validation, x="method",
         values=["recovered", "harmful", "net"], title="Recovered / harmful / net")
    for filename, column, title in (
        ("switch_rate.png", "switch_rate", "Switch coverage"),
        ("recovery_recall.png", "recovery_recall", "Recoverable-error recall"),
        ("harm_rate.png", "harm_rate", "Harm rate"),
    ):
        _bar(plots/filename, validation, x="method", values=[column], title=title)
    threshold_path = run / "policy_threshold_sweep.csv"
    threshold = pd.read_csv(threshold_path) if threshold_path.is_file() else pd.DataFrame()
    if len(threshold) and "threshold" in threshold:
        frame = threshold.dropna(subset=["threshold"])
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for key, group in frame.groupby(["backend", "model_id", "protocol"]):
            axis.plot(group["threshold"], group["delta_j_at_1"], marker="o", label=" / ".join(map(str, key)))
        axis.axhline(0, color="black", linewidth=.8); axis.set_title("Development confidence sweep")
        axis.set_xlabel("Threshold"); axis.set_ylabel("ΔJ@1"); axis.legend(fontsize=7); axis.grid(alpha=.25)
        figure.tight_layout(); figure.savefig(plots/"confidence_threshold_sweep.png", dpi=180); plt.close(figure)
    else:
        _placeholder(plots/"confidence_threshold_sweep.png", "Development confidence sweep", "No threshold sweep available.")
    ablation_path = run / "diagnostic_evidence_ablation.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.is_file() else pd.DataFrame()
    if len(ablation):
        ablation["label"] = ablation["backend"].astype(str)+" / "+ablation["evidence_variant"].astype(str)
    _bar(plots/"evidence_ablation.png", ablation, x="label", values=["net"], title="Balanced diagnostic evidence ablation")
    stability_path = run / "diagnostic_stability.csv"
    stability = pd.read_csv(stability_path) if stability_path.is_file() else pd.DataFrame()
    if len(stability):
        summary = stability.groupby(["backend","model_id"],as_index=False)[["top1_stable","decision_stable"]].mean()
        summary["label"] = summary["backend"].astype(str)+" / "+summary["model_id"].astype(str)
    else:
        summary = pd.DataFrame()
    _bar(plots/"perturbation_stability.png", summary, x="label",
         values=["top1_stable","decision_stable"], title="Permutation stability")
    if len(latency) and latency["latency_seconds"].notna().any():
        figure, axis = plt.subplots(figsize=(8,4.8))
        for model, group in latency.groupby("model_id"):
            axis.hist(pd.to_numeric(group["latency_seconds"],errors="coerce").dropna(), bins=30, alpha=.5, label=model)
        axis.set_title("Provider-attempt latency"); axis.set_xlabel("seconds"); axis.legend(fontsize=8)
        figure.tight_layout(); figure.savefig(plots/"latency_distribution.png",dpi=180); plt.close(figure)
    else:
        _placeholder(plots/"latency_distribution.png", "Latency", "No latency data.")
    tokens = pd.DataFrame([{"model_id":model,**values} for model,values in usage["tokens"].items()])
    _bar(plots/"token_usage.png", tokens, x="model_id",
         values=["input_tokens","output_tokens","thought_tokens"], title="API token usage")
    costs = pd.DataFrame([
        {"category":key,"usd":value} for key,value in usage["cost_accounting"].items()
        if isinstance(value,(int,float)) and not isinstance(value,bool)
    ])
    _bar(plots/"cost_projection.png", costs, x="category", values=["usd"], title="Estimated/reserved cost (not invoice)")
    decisions_path = run / "PER_SAMPLE_DECISIONS.parquet"
    decisions = pd.read_parquet(decisions_path) if decisions_path.is_file() else pd.DataFrame()
    api_decisions = decisions.loc[
        ~decisions.get("method", pd.Series(index=decisions.index, dtype=str)).astype(str).str.endswith("_original_score")
    ].copy() if len(decisions) else pd.DataFrame()
    if len(api_decisions) and "query_type" in api_decisions:
        expression = api_decisions.groupby(["method", "query_type"], as_index=False).agg(
            baseline_j_at_1=("baseline_correct", "mean"), final_j_at_1=("final_correct", "mean")
        )
        expression["label"] = expression["method"].astype(str) + " / " + expression["query_type"].astype(str)
        _bar(plots/"expression_type_results.png", expression, x="label",
             values=["baseline_j_at_1", "final_j_at_1"], title="Official symbolic-program query-type J@1")
    else:
        _placeholder(plots/"expression_type_results.png", "Expression-type results", "No completed API decision cohort.")
    if len(api_decisions):
        rank_parts=[]
        for backend in ("G1","C1"):
            candidate=pd.read_parquet(run/f"CANDIDATE_MANIFEST_{backend}.parquet")[["split","sample_id","candidate_id","original_rank"]]
            part=api_decisions.loc[api_decisions["backend"].eq(backend)].merge(
                candidate, left_on=["split","sample_id","final_candidate_id"],
                right_on=["split","sample_id","candidate_id"], how="left", validate="many_to_one",
            )
            rank_parts.append(part)
        ranked=pd.concat(rank_parts,ignore_index=True)
        matrix=pd.crosstab(pd.Series(1,index=ranked.index,name="original_rank"),
                           pd.to_numeric(ranked["original_rank"],errors="coerce").rename("selected_rank"))
        if len(matrix):
            figure,axis=plt.subplots(figsize=(7,4.8)); image=axis.imshow(matrix.to_numpy(),cmap="Blues")
            axis.set_xticks(range(len(matrix.columns)),[str(int(value)) for value in matrix.columns]); axis.set_yticks([0],["1"])
            axis.set_xlabel("API/final selected original rank"); axis.set_ylabel("Original selected rank")
            axis.set_title("Frozen-candidate rank transition counts")
            for column,value in enumerate(matrix.iloc[0].tolist()): axis.text(column,0,str(int(value)),ha="center",va="center")
            figure.colorbar(image,ax=axis); figure.tight_layout(); figure.savefig(plots/"rank_transition_matrix.png",dpi=180); plt.close(figure)
        else:
            _placeholder(plots/"rank_transition_matrix.png", "Rank transition matrix", "No mapped frozen candidate rank.")
    else:
        _placeholder(plots/"rank_transition_matrix.png", "Rank transition matrix", "No completed API decision cohort.")
    _placeholder(plots/"model_agreement.png", "Model agreement",
                 "Not evaluated: Flash project quota hard-stop prevents a two-model comparison.")
    missing = set(PLOT_NAMES) - {path.name for path in plots.glob("*.png")}
    if missing:
        raise RuntimeError(f"plot generation incomplete: {sorted(missing)}")


def finalize_provider_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    baseline = json.loads((run / "BASELINE_RECOMPUTE.json").read_text())
    unavailable = audited_unavailable_models(run)
    with ApiLedger(run / "api_ledger.sqlite") as ledger:
        ledger_summary = ledger.summary()
    usage, latency = _usage_and_latency(run)
    usage["cost_accounting"] = ledger_summary["cost_accounting"]
    ledger_summary.update({
        "retry_attempts": usage["retry_attempts"],
        "attempt_status_counts": usage["attempt_status_counts"],
        "successful_responses": int(ledger_summary["status_counts"].get("SUCCEEDED", 0)),
        "schema_failures": int(ledger_summary["status_counts"].get("SCHEMA_FAILED", 0)),
        "terminal_failures": int(sum(
            ledger_summary["status_counts"].get(name, 0)
            for name in ("PERMANENT_FAILED", "SCHEMA_FAILED", "TECHNICAL_FALLBACK")
        )),
        "actual_provider_charge_usd": None,
        "provider_invoice_verified": False,
    })
    atomic_json(run / "API_LEDGER_SUMMARY.json", ledger_summary)
    stage_results = _stage_results(run)
    atomic_parquet(run / "PER_REQUEST_RESULTS.parquet", stage_results)
    decisions = []
    base = pd.read_parquet(run / "baseline_per_sample.parquet").copy()
    base["method"] = base["backend"] + "_original_score"
    base["baseline_candidate_id"] = base["top1_candidate_id"]
    base["final_candidate_id"] = base["top1_candidate_id"]
    base["baseline_correct"] = base["top1_correct"]
    base["final_correct"] = base["top1_correct"]
    decisions.append(base)
    for split,path in (("validation",run/"validation_per_sample_decisions.parquet"),("test",run/"FORMAL_PER_SAMPLE_DECISIONS.parquet")):
        if path.is_file():
            frame=pd.read_parquet(path).copy(); frame["split"]=split; decisions.append(frame)
    per_sample=pd.concat(decisions,ignore_index=True,sort=False)
    query_types=_query_type_annotations(run)
    atomic_parquet(run/"QUERY_TYPE_ANNOTATIONS.parquet",query_types)
    per_sample=per_sample.merge(query_types,on=["split","sample_id"],how="left",validate="many_to_one")
    if per_sample["query_type"].isna().any():
        raise RuntimeError("official query-type annotation coverage is incomplete")
    atomic_parquet(run / "PER_SAMPLE_DECISIONS.parquet", per_sample)
    metrics = _metrics(run, baseline)
    metrics.to_csv(run / "METRICS.csv", index=False)
    validation = json.loads((run/"VALIDATION_RESULTS.json").read_text()) if (run/"VALIDATION_RESULTS.json").is_file() else None
    formal = {
        backend: json.loads((run/f"FORMAL_RESULTS_{backend}.json").read_text())
        for backend in ("G1","C1") if (run/f"FORMAL_RESULTS_{backend}.json").is_file()
    }
    status = "FORMAL_COMPLETE" if formal else ("UNTOUCHED_VALIDATION_COMPLETE" if validation else "DEVELOPMENT_INCOMPLETE")
    results = {
        "schema_version": 2, "generated_at_utc": utc_now(), "status": status,
        "metric_name": "OCID-VLG offline 2D grasp-rectangle consistency",
        "baseline": baseline, "unavailable_models": unavailable,
        "api_ledger": ledger_summary, "api_usage": usage,
        "validation": validation, "formal": formal,
        "flash_status": "NOT_EVALUATED_MODEL_UNAVAILABLE" if "gemini-3.6-flash" in unavailable else "AVAILABLE",
        "consensus_status": "NOT_EVALUATED_REQUIRES_BOTH_MODELS" if unavailable else "AVAILABLE",
    }
    atomic_json(run / "RESULTS.json", results)
    manifest_path=run/"MANIFEST.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    source_run=Path(manifest["source_run"])
    manifest.update({
        "schema_version":3,"last_updated_at_utc":utc_now(),"experiment_status":status,
        "exact_model_ids":["gemini-robotics-er-2-preview","gemini-3.6-flash"],
        "unavailable_models":unavailable,
        "model_metadata":json.loads((run/"audit/model_metadata.json").read_text()) if (run/"audit/model_metadata.json").is_file() else None,
        "api_endpoint":json.loads((run/"audit/API_ADAPTER_DECISION.json").read_text()).get("selected_endpoint") if (run/"audit/API_ADAPTER_DECISION.json").is_file() else "interactions",
        "provider_worker_sdk":"google-genai==2.16.0","request_hash_version":REQUEST_HASH_VERSION,
        "response_parser_version":RESPONSE_PARSER_VERSION,"renderer_hash":renderer_hash(),
        "prompt_hashes":{name:sha256_file(run/"prompts"/name) for name in ("direct_full_list_v1.txt","baseline_aware_v1.txt")},
        "response_schema_sha256":sha256_file(run/"prompts/api_rerank_v1.schema.json"),
        "candidate_manifest_hashes":{backend:sha256_file(run/f"CANDIDATE_MANIFEST_{backend}.parquet") for backend in ("G1","C1")},
        "scene_split_sha256":sha256_file(run/"DATA_SPLIT.csv"),"seed":20260805,
        "temperature_policy":{"er2":0.0,"flash":"endpoint minimum / deprecated field omitted"},
        "thinking_level":"medium","max_output_tokens":4096,
        "retry_policy":{"sdk_implicit_retries":0,"outer_transport_retries":2,"schema_retries":1,"rate_limit_retries":6},
        "fallback_policy":"original frozen backend Top-1 for explicit terminal provider/schema failure",
        "formal_allow_flags":{"ALLOW_FORMAL_GEMINI_RERANK":os.environ.get("ALLOW_FORMAL_GEMINI_RERANK")=="1",
                              "ALLOW_PAID_API_RUN":os.environ.get("ALLOW_PAID_API_RUN")=="1"},
        "api_cost_cap_usd":float(os.environ["MAX_API_COST_USD"]) if os.environ.get("MAX_API_COST_USD") else None,
        "max_provider_requests":int(os.environ["MAX_PROVIDER_REQUESTS"]) if os.environ.get("MAX_PROVIDER_REQUESTS") else None,
        "er2_conservative_cost_cap_per_request_usd":float(os.environ["GEMINI_ER2_COST_CAP_PER_REQUEST_USD"]) if os.environ.get("GEMINI_ER2_COST_CAP_PER_REQUEST_USD") else None,
        "validation_results_sha256":sha256_file(run/"VALIDATION_RESULTS.json") if (run/"VALIDATION_RESULTS.json").is_file() else None,
        "formal_locks":{backend:sha256_file(run/f"LOCKED_MANIFEST_{backend}.json") for backend in ("G1","C1") if (run/f"LOCKED_MANIFEST_{backend}.json").is_file()},
        "evidence_schema_sha256":sha256_file(run/"EVIDENCE_SCHEMA.json"),
        "evidence_feature_hashes":{backend:sha256_file(run/f"EVIDENCE_FEATURES_{backend}.parquet") for backend in ("G1","C1")},
        "dataset_manifest_hashes":{
            name:sha256_file(run/name) for name in ("DATA_SPLIT.csv","DATA_SPLIT_SUMMARY.json","STAGE_COHORTS.parquet")
            if (run/name).is_file()
        },
        "source_dataset_manifest_hashes":{
            f"{split}_{kind}":sha256_file(source_run/"manifests"/f"{split}_{kind}.parquet")
            for split in ("validation","test") for kind in ("samples","labels")
        },
        "query_type_annotations_sha256":sha256_file(run/"QUERY_TYPE_ANNOTATIONS.parquet"),
        "query_type_rule_version":"ocid_vlg_symbolic_v1",
        "confidence_and_protocol_lock":json.loads((run/"DEVELOPMENT_POLICY_LOCK.json").read_text()) if (run/"DEVELOPMENT_POLICY_LOCK.json").is_file() else None,
        "stability_requirement":{"metric":"three-permutation exact selected-ID agreement","minimum":0.90},
        "execution_timestamps":usage["execution_timestamps"],
    })
    atomic_json(manifest_path,manifest)
    _plots(run, metrics, usage, latency)
    materialize_failure_gallery(run)
    latency_lines=[]
    for model,value in usage["online_attempt_latency"].items():
        latency_lines.append(f"- Online {model}: attempts={value['attempts']}, p50={value['p50_seconds']:.3f}s, p95={value['p95_seconds']:.3f}s.")
    for model,value in usage["retry_inclusive_request_wall_latency"].items():
        latency_lines.append(f"- Retry-inclusive {model}: requests={value['requests']}, p50={value['p50_seconds']:.3f}s, p95={value['p95_seconds']:.3f}s.")
    latency_lines.append("- Batch latency: not applicable; the frozen run used standard Interactions only.")
    _write(run/"LATENCY_REPORT.md","Latency report",latency_lines or ["No provider latency recorded."])
    costs=ledger_summary["cost_accounting"]
    _write(run/"COST_REPORT.md","Cost report",[
        f"- Ledger attempts: {ledger_summary['provider_attempts']}.",
        f"- Flash paid-tier token-price equivalent: ${costs['flash_token_priced_estimate_usd']:.6f}.",
        f"- ER2 conservative budget reserve: ${costs['er2_conservative_budget_reserve_usd']:.6f}.",
        f"- Failure/unknown conservative reserve: ${costs['failure_unknown_charge_budget_reserve_usd']:.6f}.",
        "- Flash estimate includes preserved audit-only attempts from obsolete request hashes; none contributed a current-contract model decision.",
        "- Exact provider HTTP transmission count is not recoverable for the earliest audit attempts because the installed SDK then had implicit retries; subsequent stages used SDK retries=0 with an explicit ledger attempt per retry.",
        "- Provider invoice verified: false.",
        "- Actual ER2 monetary cost could not be independently verified; the experiment used a conservative configurable per-request budget reserve.",
    ])
    diagnostic=json.loads((run/"DIAGNOSTIC_REPORT.json").read_text()) if (run/"DIAGNOSTIC_REPORT.json").is_file() else {}
    _write(run/"PROMPT_SELECTION_REPORT.md","Prompt and evidence diagnostic",[
        f"- Request coverage: {diagnostic.get('request_coverage')}.",
        f"- Selected evidence: {diagnostic.get('selected_evidence')}.",
        f"- Permutation Top-1 stability: {diagnostic.get('top1_stability')}.",
        "Balanced diagnostic metrics are not natural-distribution performance estimates.",
    ])
    policy=json.loads((run/"POLICY_PRESELECTION.json").read_text()) if (run/"POLICY_PRESELECTION.json").is_file() else {}
    policy_lines=[f"- Locked/preselected policies: {policy.get('selections','NOT_COMPLETED')}."]
    sweep_path=run/"policy_threshold_sweep.csv"
    if sweep_path.is_file():
        sweep=pd.read_csv(sweep_path)
        for (backend,model),group in sweep.groupby(["backend","model_id"]):
            best=group.sort_values(["net","harmful","switch_rate"],ascending=[False,True,True]).iloc[0]
            policy_lines.append(
                f"- Best observed before all hard gates, {backend}/{model}: protocol={best['protocol']}, threshold={best['threshold']}, "
                f"J@1={best['final_j_at_1']:.6f}, R/H/Net={int(best['recovered'])}/{int(best['harmful'])}/{int(best['net'])}, "
                f"stability_pass={bool(best.get('stability_pass',False))}."
            )
    policy_lines.append("Only preregistered deterministic acceptance rules were evaluated; no local scorer was trained.")
    _write(run/"POLICY_SELECTION_REPORT.md","Natural-distribution policy selection",policy_lines)
    validation_lines=[
        f"- Status: {'complete' if validation else 'not complete'}.",
        f"- Backend decisions: {None if validation is None else validation.get('backend_decisions')}.",
        f"- Not evaluated: {None if validation is None else validation.get('not_evaluated_methods')}.",
    ]
    if validation is not None:
        for row in validation.get("metrics",[]):
            validation_lines.append(
                f"- {row['method']}: N={row['N_total']}, J@1={row['final_j_at_1']:.6f}, "
                f"R/H/Net={row['recovered']}/{row['harmful']}/{row['net']}, decision={row['validation_decision']}."
            )
    _write(run/"VALIDATION_REPORT.md","Untouched validation",validation_lines)
    backend_decisions={} if validation is None else validation.get("backend_decisions",{})
    _write(run/"GO_NO_GO.md","GO / NO-GO",[
        f"- G1: {backend_decisions.get('G1','NOT_EVALUATED')}.",
        f"- C1: {backend_decisions.get('C1','NOT_EVALUATED')}.",
        "Flash and ER2+Flash consensus are NOT_EVALUATED under the run-scoped Flash quota hard-stop.",
    ])
    gallery_manifest=json.loads((run/"failure_gallery/MANIFEST.json").read_text())
    failure_lines=[
        "Machine-readable recovered/harmful/neutral outcomes are in PER_SAMPLE_DECISIONS.parquet.",
        "No missing Flash response was treated as a model decision; no consensus was imputed.",
        "Gallery development examples are explicitly labelled and are not untouched-validation estimates.",
    ]
    for category,value in gallery_manifest["categories"].items():
        failure_lines.append(f"- {category}: materialized={value['materialized']}, status={value['status']}.")
    _write(run/"FAILURE_ANALYSIS.md","Failure analysis",failure_lines)
    summary_lines=[
        f"Status: **{status}**.",
        f"- G1 original test J@1/J@5: {baseline['G1_test']['j_at_1']:.9f} / {baseline['G1_test']['j_at_5']:.9f}.",
        f"- C1 original test J@1/J@5: {baseline['C1_test']['j_at_1']:.9f} / {baseline['C1_test']['j_at_5']:.9f}.",
        f"- Untouched-validation decisions: {backend_decisions or 'not completed'}.",
        "- Flash: NOT_EVALUATED_MODEL_UNAVAILABLE (project free-tier quota hard-stop).",
        "- ER2+Flash consensus: NOT_EVALUATED_REQUIRES_BOTH_MODELS.",
        "J@1 is offline 2D grasp-rectangle consistency, not physical grasp success.",
    ]
    _write(run/"SUMMARY.md","Pure Gemini API frozen-candidate reranking",summary_lines)
    _write(run/"SUMMARY_ZH.md","纯 Gemini API 冻结候选重排序",[
        f"状态：**{status}**。", f"- G1 原始 test J@1/J@5：{baseline['G1_test']['j_at_1']:.9f} / {baseline['G1_test']['j_at_5']:.9f}。",
        f"- C1 原始 test J@1/J@5：{baseline['C1_test']['j_at_1']:.9f} / {baseline['C1_test']['j_at_5']:.9f}。",
        f"- untouched validation：{backend_decisions or '尚未完成'}。",
        "- Flash：项目配额硬阻塞，未评价；双模型 consensus 也未评价。",
        "J@1 是离线二维抓取矩形一致性，不是物理抓取成功率。",
    ])
    candidate_integrity={}
    for backend in ("G1","C1"):
        path=run/f"CANDIDATE_MANIFEST_{backend}.parquet"; frame=pd.read_parquet(path); validate_candidate_manifest(frame)
        candidate_integrity[backend]={"sha256":sha256_file(path),"rows":len(frame),"valid":True}
    inventory=json.loads((run/"audit_inventory.json").read_text())
    source_checks={}
    for backend,value in inventory["backends"].items():
        source_checks[f"{backend}_checkpoint"] = sha256_file(value["checkpoint_path"]) == value["checkpoint_sha256"]
        source_checks[f"{backend}_config"] = sha256_file(value["config_path"]) == value["config_sha256"]
    for key,value in inventory["candidates"].items():
        source_checks[key] = (
            sha256_file(value["source_candidate_path"]) == value["source_candidate_sha256"]
            and sha256_file(value["source_sample_path"]) == value["source_sample_sha256"]
        )
    leakage={}
    forbidden=("candidate_success","best_rectangle_iou","best_angle_difference","top1_correct","recoverable","unrecoverable","harmful")
    for path in sorted((run/"request_manifests").glob("*.parquet")):
        columns=" ".join(pd.read_parquet(path).columns).lower()
        leakage[path.name]=not any(token in columns for token in forbidden)
    secret_scan=_secret_scan(run)
    local_repo=Path(__file__).resolve().parents[3]
    local_secret_file=local_repo/".env.gemini_rerank.local"
    gitignore_path=next(
        (root/".gitignore" for root in (local_repo,*local_repo.parents) if (root/".gitignore").is_file()),
        None,
    )
    local_secret_control={
        "path_recorded_without_secret_value":str(local_secret_file),
        "exists":local_secret_file.is_file(),
        "permission_mode":None if not local_secret_file.exists() else oct(local_secret_file.stat().st_mode & 0o777),
        "permission_0600":local_secret_file.is_file() and (local_secret_file.stat().st_mode & 0o777)==0o600,
        "gitignore_path":None if gitignore_path is None else str(gitignore_path),
        "gitignore_env_wildcard_present":False if gitignore_path is None else ".env.*" in gitignore_path.read_text().splitlines(),
        "excluded_from_artifact_secret_scan_because_it_is_the_approved_local_secret_container":True,
    }
    atomic_json(run/"INTEGRITY_REPORT.json",{
        "generated_at_utc":utc_now(),"candidate_manifests":candidate_integrity,
        "source_artifacts_unchanged":all(source_checks.values()),"source_checks":source_checks,
        **secret_scan,
        "local_secret_control":local_secret_control,
        "request_manifest_label_firewall":leakage,"all_request_manifests_pass":all(leakage.values()),
        "duplicate_successful_request_hashes":ledger_summary["duplicate_successful_request_hashes"],
        "no_duplicate_success":ledger_summary["duplicate_successful_request_hashes"]==0,
        "unavailable_models":unavailable,"flash_or_consensus_imputed":False,
    })
    _write(run/"REPRODUCE.md","Reproduce",[
        "Run from the HiFi_reproduction repository root with the same frozen artifacts and process-environment credential.",
        "`python tools/api_reranking/run_gemini_g1_c1_rerank.py test --run-dir " + str(run) + "`",
        "`python tools/api_reranking/run_gemini_g1_c1_rerank.py select-policy --run-dir " + str(run) + " --backend both --model er2 --concurrency 16 --resume`",
        "`python tools/api_reranking/run_gemini_g1_c1_rerank.py validate --run-dir " + str(run) + " --backend both --model er2 --concurrency 16 --resume`",
        "`python tools/api_reranking/run_gemini_g1_c1_rerank.py lock --run-dir " + str(run) + " --backend both`",
        "Formal is permitted only for an exact GO backend with its immutable lock and both explicit allow flags.",
        "`python tools/api_reranking/run_gemini_g1_c1_rerank.py report --run-dir " + str(run) + "`",
        "Successful logical request hashes are cache hits on resume and are never resent.",
    ])
    return results
