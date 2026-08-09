"""Freeze development policy, evaluate untouched validation, and apply GO gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .decisions import baseline_aware, confidence_accept, cross_model_consensus, direct, self_consistent
from .io import atomic_json, atomic_parquet, sha256_file, utc_now
from .metrics import go_no_go, mcnemar_exact, outcome_metrics, scene_bootstrap_delta
from .renderer import renderer_hash
from .contracts import forbidden_payload_hits, validate_candidate_manifest
from .constants import EXACT_MODEL_IDS
from .availability import audited_unavailable_models
from .stages import assert_stage_result_coverage


def create_development_policy_lock(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    destination = run / "DEVELOPMENT_POLICY_LOCK.json"
    if destination.exists():
        return json.loads(destination.read_text())
    selection = json.loads((run / "POLICY_PRESELECTION.json").read_text())["selections"]
    per_model = {}
    for key, value in selection.items():
        if value["status"] in {"NO_BENEFICIAL_DEVELOPMENT_POLICY", "NO_ELIGIBLE_DEVELOPMENT_POLICY"}:
            per_model[key] = {"protocol": "P0_ORIGINAL_BACKEND_SCORE", "threshold": None,
                              "evidence_variant": None, "confirmation_required": False,
                              "selection_status": value["status"],
                              "ineligibility_reasons": value.get("ineligibility_reasons", [])}
        else:
            per_model[key] = {
                "protocol": value["protocol"], "threshold": value.get("threshold"),
                "evidence_variant": value["evidence_variant"],
                "confirmation_required": value["protocol"] == "P4_API_SELF_CONSISTENT",
            }
    unavailable = audited_unavailable_models(run)
    for backend in ("G1", "C1"):
        for model in EXACT_MODEL_IDS:
            key = f"{backend}:{model}"
            if key in per_model:
                continue
            if model not in unavailable:
                raise RuntimeError(
                    f"development policy is missing {key} without an audited model hard stop"
                )
            per_model[key] = {
                "protocol": "P0_ORIGINAL_BACKEND_SCORE",
                "threshold": None,
                "evidence_variant": None,
                "confirmation_required": False,
                "model_available": False,
                "unavailable_reason": unavailable[model]["status"],
                "unavailable_audit": unavailable[model]["path"],
            }
    diagnostic = json.loads((run / "DIAGNOSTIC_REPORT.json").read_text())
    payload = {
        "schema_version": 1, "locked_at_utc": utc_now(), "selection_split": "policy_selection",
        "untouched_validation_used": False, "per_backend_model": per_model,
        "prompt_hashes": {name: sha256_file(run/"prompts"/name) for name in ("direct_full_list_v1.txt","baseline_aware_v1.txt")},
        "schema_hash": sha256_file(run/"prompts/api_rerank_v1.schema.json"),
        "renderer_hash": renderer_hash(), "data_split_hash": sha256_file(run/"DATA_SPLIT.csv"),
        "candidate_hashes": {backend: sha256_file(run/f"CANDIDATE_MANIFEST_{backend}.parquet") for backend in ("G1","C1")},
        "threshold_grid": [50,60,70,80,90,95],
        "stability_requirement": {"metric": "three-permutation exact selected-ID agreement", "minimum": 0.90},
        "per_backend_model_stability": diagnostic["per_backend_model_stability"],
        "unavailable_models": unavailable,
    }
    atomic_json(destination, payload)
    return payload


def _response(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None or row.get("status") != "SUCCEEDED":
        return None
    return {"selected_internal_candidate_id": row.get("selected_candidate_id"),
            "decision": row.get("decision"), "switch_confidence": row.get("switch_confidence"),
            "evidence_reliability": row.get("evidence_reliability")}


def build_validation_confirmation_manifest(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    lock = json.loads((run/"DEVELOPMENT_POLICY_LOCK.json").read_text())
    base = pd.read_parquet(run/"stage_results/untouched_validation.parquet")
    baseline = pd.read_parquet(run/"baseline_per_sample.parquet")[["backend","sample_id","top1_candidate_id"]]
    parts = []
    for key, policy in lock["per_backend_model"].items():
        if not policy["confirmation_required"]:
            continue
        backend, model = key.split(":",1)
        threshold = int(policy["threshold"])
        rows = base.loc[(base["backend"]==backend)&(base["model_id"]==model)&base["status"].eq("SUCCEEDED")
                        &base["decision"].eq("SELECT_CANDIDATE")&base["evidence_reliability"].eq("HIGH")
                        &pd.to_numeric(base["switch_confidence"],errors="coerce").ge(threshold)].merge(
                            baseline,on=["backend","sample_id"],how="left",validate="many_to_one")
        parts.append(rows.loc[rows["selected_candidate_id"].astype(str)!=rows["top1_candidate_id"].astype(str)].copy())
    selected = pd.concat(parts,ignore_index=True) if parts else base.iloc[:0].copy()
    selected["replicate_id"] = 2; selected["perturbation"] = "p4_display_panel_colour_permutation"
    keep = ["stage","backend","sample_id","scene_id","candidate_count","model_id","protocol","evidence_variant","perturbation","replicate_id"]
    path = run/"request_manifests/validation_confirmation.parquet"
    atomic_parquet(path,selected[keep])
    payload={"rows":len(selected),"sha256":sha256_file(path)}
    atomic_json(run/"request_manifests/validation_confirmation.json",payload)
    return payload


def _selected(policy: Mapping[str, Any], first: Mapping[str, Any] | None,
              second: Mapping[str, Any] | None, original: str) -> str:
    protocol=policy["protocol"]
    if protocol=="P0_ORIGINAL_BACKEND_SCORE": return original
    if protocol=="P1_API_DIRECT_FULL_LIST": return direct(first,original)
    if protocol=="P2_API_BASELINE_AWARE": return baseline_aware(first,original)
    if protocol=="P3_API_BASELINE_AWARE_CONFIDENCE": return confidence_accept(first,original,int(policy["threshold"]))
    if protocol=="P4_API_SELF_CONSISTENT": return self_consistent(first,second,original,int(policy["threshold"]))
    raise ValueError(protocol)


def evaluate_untouched_validation(run_dir: str | Path) -> dict[str, Any]:
    run=Path(run_dir); lock=json.loads((run/"DEVELOPMENT_POLICY_LOCK.json").read_text())
    validation_coverage=assert_stage_result_coverage(run,"untouched_validation")
    first=pd.read_parquet(run/"stage_results/untouched_validation.parquet")
    confirmation_path=run/"stage_results/validation_confirmation.parquet"
    second=pd.read_parquet(confirmation_path) if confirmation_path.exists() else first.iloc[:0]
    confirmation_coverage=None
    if (run/"request_manifests/validation_confirmation.parquet").is_file():
        confirmation_coverage=assert_stage_result_coverage(run,"validation_confirmation")
    cohorts=pd.read_parquet(run/"STAGE_COHORTS.parquet")
    baseline_all=pd.read_parquet(run/"baseline_per_sample.parquet")
    metric_rows=[]; tests={}; boots={}; decisions=[]; backend_decisions={}
    model_ids=list(EXACT_MODEL_IDS)
    for backend in ("G1","C1"):
        candidate_manifest = pd.read_parquet(run / f"CANDIDATE_MANIFEST_{backend}.parquet")
        validate_candidate_manifest(candidate_manifest)
        frozen_ids_by_sample = {
            str(sample_id): set(group["candidate_id"].astype(str))
            for sample_id, group in candidate_manifest.loc[candidate_manifest["split"].eq("validation")].groupby("sample_id")
        }
        sample_ids=set(cohorts.loc[(cohorts["stage"]=="untouched_validation")&(cohorts["backend"]==backend),"sample_id"].astype(str))
        baseline=baseline_all.loc[(baseline_all["backend"]==backend)&(baseline_all["split"]=="validation")&baseline_all["sample_id"].astype(str).isin(sample_ids)]
        method_outputs={}
        response_by_model={model:{str(r["sample_id"]):r for r in first.loc[(first["backend"]==backend)&(first["model_id"]==model)].to_dict(orient="records")} for model in model_ids}
        second_by_model={model:{str(r["sample_id"]):r for r in second.loc[(second["backend"]==backend)&(second["model_id"]==model)].to_dict(orient="records")} for model in model_ids}
        for model in model_ids:
            policy=lock["per_backend_model"][f"{backend}:{model}"]
            if not policy.get("model_available", True):
                continue
            rows=[]
            for item in baseline.to_dict(orient="records"):
                sample_id=str(item["sample_id"]); original=item.get("top1_candidate_id")
                final=None if original is None else _selected(policy,_response(response_by_model[model].get(sample_id)),_response(second_by_model[model].get(sample_id)),str(original))
                correctness=json.loads(str(item["candidate_correctness_json"]))
                if final is not None and str(final) not in frozen_ids_by_sample.get(sample_id, set()):
                    raise AssertionError("validation selected a non-frozen candidate")
                rows.append({"backend":backend,"method":f"{backend}_{model}_locked","sample_id":sample_id,"scene_id":item["scene_id"],
                             "baseline_candidate_id":original,"final_candidate_id":final,"baseline_correct":bool(item["top1_correct"]),
                             "final_correct":bool(correctness.get(str(final),False)) if final is not None else False,
                             "recoverable_error":bool(item["recoverable_error"])})
            method_outputs[model]=rows
        # P5 is a real two-model method.  It is absent—not imputed—when either
        # exact model has an audited external hard stop.
        er_policy=lock["per_backend_model"][f"{backend}:{model_ids[0]}"]; fl_policy=lock["per_backend_model"][f"{backend}:{model_ids[1]}"]
        if er_policy.get("model_available", True) and fl_policy.get("model_available", True):
            consensus=[]
            for item in baseline.to_dict(orient="records"):
                sid=str(item["sample_id"]); original=item.get("top1_candidate_id"); correctness=json.loads(str(item["candidate_correctness_json"]))
                er_threshold=int(er_policy.get("threshold") or 101); fl_threshold=int(fl_policy.get("threshold") or 101)
                final=None if original is None else cross_model_consensus(_response(response_by_model[model_ids[0]].get(sid)),_response(response_by_model[model_ids[1]].get(sid)),str(original),er_threshold,fl_threshold)
                if final is not None and str(final) not in frozen_ids_by_sample.get(sid, set()):
                    raise AssertionError("consensus selected a non-frozen candidate")
                consensus.append({"backend":backend,"method":f"{backend}_ER2_FLASH_CONSENSUS","sample_id":sid,"scene_id":item["scene_id"],
                                  "baseline_candidate_id":original,"final_candidate_id":final,"baseline_correct":bool(item["top1_correct"]),
                                  "final_correct":bool(correctness.get(str(final),False)) if final is not None else False,"recoverable_error":bool(item["recoverable_error"])})
            method_outputs["consensus"]=consensus
        gates=[]
        for method, rows in method_outputs.items():
            metrics=outcome_metrics(rows); boot=scene_bootstrap_delta(rows); test=mcnemar_exact(metrics["recovered"],metrics["harmful"])
            if method == "consensus":
                stability = all(
                    bool(lock["per_backend_model_stability"].get(f"{backend}:{model}", {}).get("passes_preregistered_minimum", False))
                    for model in model_ids
                )
            else:
                stability = bool(lock["per_backend_model_stability"].get(f"{backend}:{method}", {}).get("passes_preregistered_minimum", False))
            request_paths = [run/"request_manifests/untouched_validation.parquet", run/"request_manifests/validation_confirmation.parquet"]
            leakage_pass = True
            for request_path in request_paths:
                if request_path.exists():
                    request_manifest = pd.read_parquet(request_path)
                    leakage_pass = leakage_pass and not forbidden_payload_hits({"columns": request_manifest.columns.tolist()})
            invariants_pass = all(
                row["final_candidate_id"] is None or str(row["final_candidate_id"]) in frozen_ids_by_sample.get(str(row["sample_id"]), set())
                for row in rows
            )
            metrics["j_at_5"] = float(baseline["top5_any_correct"].mean())
            metrics["j5_unchanged"] = True
            decision=go_no_go(metrics,boot,stability_pass=stability,invariants_pass=invariants_pass,leakage_pass=leakage_pass)
            name=rows[0]["method"]; metric_rows.append({"method":name,"validation_decision":decision,**metrics}); boots[name]=boot; tests[name]=test
            decisions.extend(rows); gates.append((name,decision,metrics,boot))
        go=[item for item in gates if item[1]=="GO"]
        if not gates:
            backend_decisions[backend]={"decision":"NOT_EVALUATED_MODEL_UNAVAILABLE","primary":f"{backend}_original_score"}
        elif go:
            go.sort(key=lambda item:(-item[2]["net"],item[2]["harmful"],item[2]["switch_rate"]))
            backend_decisions[backend]={"decision":"GO","primary":go[0][0]}
        elif any(item[1]=="INCONCLUSIVE" for item in gates): backend_decisions[backend]={"decision":"INCONCLUSIVE","primary":f"{backend}_original_score"}
        else: backend_decisions[backend]={"decision":"NO_GO","primary":f"{backend}_original_score"}
    pd.DataFrame(metric_rows).to_csv(run/"validation_metrics.csv",index=False)
    atomic_parquet(run/"validation_per_sample_decisions.parquet",pd.DataFrame(decisions))
    atomic_json(run/"STATISTICAL_TESTS.json",tests); atomic_json(run/"BOOTSTRAP_INTERVALS.json",boots)
    not_evaluated=[]
    for backend in ("G1","C1"):
        unavailable_for_backend=[
            model for model in model_ids
            if not lock["per_backend_model"][f"{backend}:{model}"].get("model_available",True)
        ]
        for model in unavailable_for_backend:
            not_evaluated.append({"backend":backend,"model_id":model,
                                  "status":"NOT_EVALUATED_MODEL_UNAVAILABLE"})
        if unavailable_for_backend:
            not_evaluated.append({"backend":backend,"method":f"{backend}_ER2_FLASH_CONSENSUS",
                                  "status":"NOT_EVALUATED_REQUIRES_BOTH_MODELS"})
    payload={"backend_decisions":backend_decisions,"metrics":metric_rows,"selection_used_test":False,"bootstrap_draws":10000,
             "unavailable_models":lock.get("unavailable_models",{}),
             "not_evaluated_methods":not_evaluated,
             "request_coverage":{"untouched_validation":validation_coverage,
                                 "validation_confirmation":confirmation_coverage}}
    atomic_json(run/"VALIDATION_RESULTS.json",payload)
    return payload
