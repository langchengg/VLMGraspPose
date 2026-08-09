"""One-time locked formal-test derivation over frozen candidate IDs only."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from .decisions import cross_model_consensus
from .io import atomic_json, atomic_parquet, sha256_file, utc_now
from .metrics import mcnemar_exact, outcome_metrics, scene_bootstrap_delta
from .validation import _response, _selected
from .stages import assert_stage_result_coverage


MODEL_IDS = ("gemini-robotics-er-2-preview", "gemini-3.6-flash")


def initialize_formal_state(run_dir: str | Path, backends: tuple[str, ...]) -> dict[str, Any]:
    run = Path(run_dir); path = run / "FORMAL_RUN_STATE.json"
    lock_hashes = {backend: sha256_file(run/f"LOCKED_MANIFEST_{backend}.json") for backend in backends}
    if path.exists():
        value=json.loads(path.read_text())
        if value["lock_hashes"] != lock_hashes:
            raise RuntimeError("formal run lock identity changed")
        return value
    value={"formal_run_id":uuid.uuid4().hex,"created_at_utc":utc_now(),"lock_hashes":lock_hashes,
           "status":"IN_PROGRESS","backends":list(backends)}
    atomic_json(path,value); return value


def build_formal_confirmation_manifest(run_dir: str | Path) -> dict[str, Any]:
    run=Path(run_dir); base=pd.read_parquet(run/"stage_results/formal.parquet")
    baseline=pd.read_parquet(run/"baseline_per_sample.parquet")
    parts=[]
    development=json.loads((run/"DEVELOPMENT_POLICY_LOCK.json").read_text())
    for backend in sorted(base["backend"].unique()):
        primary=str(json.loads((run/f"LOCKED_MANIFEST_{backend}.json").read_text())["locked_primary"])
        for model in MODEL_IDS:
            if "CONSENSUS" not in primary and ((model==MODEL_IDS[0] and "robotics-er-2" not in primary) or (model==MODEL_IDS[1] and "3.6-flash" not in primary)):
                continue
            policy=development["per_backend_model"][f"{backend}:{model}"]
            if not policy["confirmation_required"]: continue
            top=baseline.loc[(baseline["backend"]==backend)&baseline["split"].eq("test"),["backend","sample_id","top1_candidate_id"]]
            threshold=int(policy["threshold"])
            rows=base.loc[(base["backend"]==backend)&(base["model_id"]==model)&base["status"].eq("SUCCEEDED")
                          &base["decision"].eq("SELECT_CANDIDATE")&base["evidence_reliability"].eq("HIGH")
                          &pd.to_numeric(base["switch_confidence"],errors="coerce").ge(threshold)].merge(top,on=["backend","sample_id"],how="left",validate="many_to_one")
            parts.append(rows.loc[rows["selected_candidate_id"].astype(str)!=rows["top1_candidate_id"].astype(str)].copy())
    selected=pd.concat(parts,ignore_index=True) if parts else base.iloc[:0].copy()
    selected["replicate_id"]=2; selected["perturbation"]="formal_p4_display_panel_colour_permutation"
    keep=["stage","backend","sample_id","scene_id","candidate_count","model_id","protocol","evidence_variant","perturbation","replicate_id"]
    path=run/"request_manifests/formal_confirmation.parquet"; atomic_parquet(path,selected[keep])
    payload={"rows":len(selected),"sha256":sha256_file(path)}; atomic_json(run/"request_manifests/formal_confirmation.json",payload)
    return payload


def evaluate_formal(run_dir: str | Path, backends: tuple[str, ...]) -> dict[str, Any]:
    run=Path(run_dir); formal_coverage=assert_stage_result_coverage(run,"formal")
    first=pd.read_parquet(run/"stage_results/formal.parquet")
    second_path=run/"stage_results/formal_confirmation.parquet"
    second=pd.read_parquet(second_path) if second_path.exists() else first.iloc[:0]
    confirmation_coverage=None
    if (run/"request_manifests/formal_confirmation.parquet").is_file():
        confirmation_coverage=assert_stage_result_coverage(run,"formal_confirmation")
    baseline_all=pd.read_parquet(run/"baseline_per_sample.parquet")
    development=json.loads((run/"DEVELOPMENT_POLICY_LOCK.json").read_text())
    all_decisions=[]; results={}
    for backend in backends:
        baseline=baseline_all.loc[(baseline_all["backend"]==backend)&baseline_all["split"].eq("test")]
        candidate_manifest=pd.read_parquet(run/f"CANDIDATE_MANIFEST_{backend}.parquet")
        frozen_ids_by_sample={
            str(sample_id):set(group["candidate_id"].astype(str))
            for sample_id,group in candidate_manifest.loc[candidate_manifest["split"].eq("test")].groupby("sample_id")
        }
        lock=json.loads((run/f"LOCKED_MANIFEST_{backend}.json").read_text()); primary=str(lock["locked_primary"])
        response={model:{str(r["sample_id"]):r for r in first.loc[(first["backend"]==backend)&(first["model_id"]==model)].to_dict(orient="records")} for model in MODEL_IDS}
        confirmation={model:{str(r["sample_id"]):r for r in second.loc[(second["backend"]==backend)&(second["model_id"]==model)].to_dict(orient="records")} for model in MODEL_IDS}
        rows=[]
        for item in baseline.to_dict(orient="records"):
            sid=str(item["sample_id"]); original=item.get("top1_candidate_id")
            if original is None: final=None
            elif "CONSENSUS" in primary:
                left=development["per_backend_model"][f"{backend}:{MODEL_IDS[0]}"]; right=development["per_backend_model"][f"{backend}:{MODEL_IDS[1]}"]
                final=cross_model_consensus(_response(response[MODEL_IDS[0]].get(sid)),_response(response[MODEL_IDS[1]].get(sid)),str(original),int(left.get("threshold") or 101),int(right.get("threshold") or 101))
            else:
                model=MODEL_IDS[0] if "robotics-er-2" in primary else MODEL_IDS[1]
                policy=development["per_backend_model"][f"{backend}:{model}"]
                final=_selected(policy,_response(response[model].get(sid)),_response(confirmation[model].get(sid)),str(original))
            if final is not None and str(final) not in frozen_ids_by_sample.get(sid,set()):
                raise AssertionError("formal selected a non-frozen candidate")
            correctness=json.loads(str(item["candidate_correctness_json"]))
            rows.append({"backend":backend,"method":primary,"sample_id":sid,"scene_id":item["scene_id"],
                         "baseline_candidate_id":original,"final_candidate_id":final,"baseline_correct":bool(item["top1_correct"]),
                         "final_correct":bool(correctness.get(str(final),False)) if final is not None else False,
                         "recoverable_error":bool(item["recoverable_error"])})
        metrics=outcome_metrics(rows); metrics["j_at_5"]=float(baseline["top5_any_correct"].mean()); metrics["j5_unchanged"]=True
        results[backend]={"locked_primary":primary,"metrics":metrics,
                          "mcnemar":mcnemar_exact(metrics["recovered"],metrics["harmful"]),
                          "scene_bootstrap":scene_bootstrap_delta(rows)}
        all_decisions.extend(rows)
        atomic_json(run/f"FORMAL_RESULTS_{backend}.json",results[backend])
        (run/f"FORMAL_TEST_REPORT_{backend}.md").write_text(
            f"# {backend} locked formal test\n\nPrimary: `{primary}`\n\n"
            f"J@1={metrics['final_j_at_1']:.9f}; Δ={metrics['delta_j_at_1']:.9f}; R/H/Net={metrics['recovered']}/{metrics['harmful']}/{metrics['net']}.\n\n"
            "Metric: OCID-VLG offline 2D grasp-rectangle consistency, not physical grasp success.\n"
        )
    atomic_parquet(run/"FORMAL_PER_SAMPLE_DECISIONS.parquet",pd.DataFrame(all_decisions))
    state=json.loads((run/"FORMAL_RUN_STATE.json").read_text()); state["status"]="COMPLETED"; state["completed_at_utc"]=utc_now(); atomic_json(run/"FORMAL_RUN_STATE.json",state)
    results["request_coverage"]={"formal":formal_coverage,"formal_confirmation":confirmation_coverage}
    return results
