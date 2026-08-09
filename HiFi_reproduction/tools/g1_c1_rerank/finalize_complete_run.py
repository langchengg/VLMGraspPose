#!/usr/bin/env python3
"""Audit terminal artifacts, write reproduction entry points, and seal COMPLETE."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _verify_inventory(entries: Sequence[dict], *, label: str) -> None:
    if not entries: raise RuntimeError(f"{label} inventory is empty")
    for entry in entries:
        path_value=entry.get("path",entry.get("label_path")); sha_value=entry.get("sha256",entry.get("label_sha256"))
        if path_value is None or sha_value is None: raise RuntimeError(f"{label} inventory entry lacks path/hash")
        path=Path(str(path_value)).expanduser().resolve()
        if not path.is_file() or _sha(path)!=str(sha_value): raise RuntimeError(f"{label} artifact drift: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args=_parse(argv); base=args.base_run.expanduser().resolve(); run=args.run_dir.expanduser().resolve(); project=Path(__file__).resolve().parents[2]; repo=project.parent
    if (run/"COMPLETE").exists():
        raise RuntimeError("refusing to mutate an already sealed run")
    if (run/".RUN_FINISHED").exists():
        if (run/".RUN_ACTIVE").exists():
            raise RuntimeError("both RUN_ACTIVE and RUN_FINISHED exist")
        # Recover a process-level crash that occurred in the final two-file
        # commit window before COMPLETE was atomically installed.
        (run/".RUN_FINISHED").rename(run/".RUN_ACTIVE")
    required=[
        run/"AUDIT_INVENTORY.json",
        run/"MANIFEST.json",
        run/"manifest.json",
        run/"environment.txt",
        run/"git_state.txt",
        run/"commands.log",
        run/"02_features"/"FEATURE_STATUS.json",
        run/"03_splits"/"fold_assignments.parquet",
        run/"04_calibration"/"g1"/"calibrator.pkl",
        run/"04_calibration"/"c1"/"calibrator.pkl",
        run/"07_validation"/"VALIDATION_MATRIX.csv",
        run/"07_validation"/"FEATURE_ABLATION_CUMULATIVE.csv",
        run/"07_validation"/"GATE_SELECTION.json",
        run/"08_lock"/"PRIMARY_METHOD_LOCK.json",
        run/"09_formal_test"/"TEST_ACCESS_TRANSACTION.json",
        run/"09_formal_test"/"test_access.log",
        run/"09_formal_test"/"EVALUATION_COMPLETE.json",
        run/"09_formal_test"/"EVALUATION_TRANSACTION.json",
        run/"tables"/"formal_test_primary.csv",
        run/"tables"/"statistical_tests.csv",
        run/"10_statistics"/"statistical_tests.csv",
        run/"10_statistics"/"primary_deployment_tests.csv",
        run/"14_independent_recompute"/"INDEPENDENT_RECOMPUTE.json",
        run/"12_failure_galleries"/"gallery_manifest.csv",
        run/"13_reports"/"FINAL_SUMMARY_ZH.md",
        run/"13_reports"/"FINAL_REPORT_EN.md",
        run/"13_reports"/"EXPERIMENT_CONCLUSION.json",
    ]
    missing=[str(path) for path in required if not path.is_file() or path.stat().st_size<=0]
    if missing: raise RuntimeError(f"cannot seal incomplete run; missing={missing}")
    audit=json.loads((run/"AUDIT_INVENTORY.json").read_text())
    if audit.get("baseline_regression_match") is not True: raise RuntimeError("source baseline regression audit is not PASS")
    overlap=audit.get("split_audit",{}).get("cross_split_overlap",{})
    if not overlap or any(int(value)!=0 for value in overlap.values()): raise RuntimeError("source split leakage audit has non-zero overlap")
    test_contract=audit.get("test_contract",{})
    if test_contract.get("per_sample_test_labels_loaded") is not False: raise RuntimeError("pre-lock Test-label contract is not sealed")
    split_audit=json.loads((run/"03_splits"/"split_leakage_audit.json").read_text())
    if split_audit.get("status")!="PASS" or int(split_audit.get("group_cross_fold_overlap",-1))!=0: raise RuntimeError("OOF split leakage audit is not PASS")
    feature_status=json.loads((run/"02_features"/"FEATURE_STATUS.json").read_text())
    families={str(row.get("family")):str(row.get("status")) for row in feature_status.get("families",[])}
    allowed={"COMPLETE","COMPLETE_WITH_FALLBACK","FALLBACK","UNAVAILABLE_FALLBACK_SCALAR"}
    if set(families)!={f"F{index}" for index in range(9)} or not set(families.values()).issubset(allowed) or feature_status.get("test_tuning") is not False: raise RuntimeError("F0-F8 feature status/test-tuning contract is invalid")
    smoke=json.loads((run/"00_audit"/"smoke_matrix"/"SMOKE_MATRIX.json").read_text())
    smoke_records=smoke.get("records",[])
    if smoke.get("status")!="PASS" or smoke.get("candidate_count_unchanged") is not True or smoke.get("label_columns_excluded_from_model_features") is not True or not smoke_records or any(row.get("status")!="PASS" or row.get("finite_scores") is not True or int(row.get("candidate_rows",-1))!=int(row.get("output_rows",-2)) for row in smoke_records): raise RuntimeError("no-GT leakage/candidate-preservation smoke matrix is not PASS")
    independent=json.loads((run/"14_independent_recompute"/"INDEPENDENT_RECOMPUTE.json").read_text())
    if independent.get("status")!="PASS": raise RuntimeError("independent recomputation did not pass")
    transaction=json.loads((run/"09_formal_test"/"TEST_ACCESS_TRANSACTION.json").read_text())
    evaluation=json.loads((run/"09_formal_test"/"EVALUATION_COMPLETE.json").read_text())
    evaluation_transaction=json.loads((run/"09_formal_test"/"EVALUATION_TRANSACTION.json").read_text())
    prediction_path=run/"09_formal_test"/"PREDICTIONS_COMPLETE.json"
    lock_path=run/"08_lock"/"PRIMARY_METHOD_LOCK.json"
    lock=json.loads(lock_path.read_text())
    manifest=json.loads((run/"MANIFEST.json").read_text())
    prediction=json.loads(prediction_path.read_text())
    expected_base=base.resolve(); universe_sha=_sha(base/"manifests"/"test_samples.parquet")
    if lock.get("status")!="LOCKED" or Path(str(lock.get("base_run",""))).resolve()!=expected_base: raise RuntimeError("finalizer base differs from primary lock")
    for state,name in ((prediction,"prediction"),(transaction,"label transaction"),(evaluation,"evaluation"),(evaluation_transaction,"evaluation transaction")):
        if Path(str(state.get("base_run",""))).resolve()!=expected_base or state.get("test_universe_sha256")!=universe_sha: raise RuntimeError(f"finalizer {name} base/universe binding mismatch")
    for section in ("locked_models","ranker_calibrators","candidate_artifacts","feature_artifacts","code_artifacts","validation_selection_artifacts","source_manifest_artifacts","source_label_artifacts","audit_artifacts","lock_support_artifacts"):
        _verify_inventory(lock.get(section,[]),label=f"primary lock/{section}")
    _verify_inventory([lock.get("prediction_plan",{})],label="primary lock/prediction plan")
    _verify_inventory([lock.get("evaluator",{})],label="primary lock/evaluator")
    formal_lock=run/"08_lock"/"FORMAL_TEST_LOCK.json"
    if not formal_lock.is_file() or _sha(formal_lock)!=_sha(lock_path): raise RuntimeError("formal Test lock mirror differs from primary lock")
    _verify_inventory(prediction.get("prediction_artifacts",[]),label="formal prediction")
    _verify_inventory(transaction.get("label_artifacts",[]),label="formal label")
    _verify_inventory(evaluation.get("evaluation_artifacts",[]),label="formal evaluation")
    if transaction.get("status")!="CONSUMED" or int(manifest.get("formal_run_count",-1))!=1: raise RuntimeError("formal test one-time contract failed")
    if evaluation.get("status")!="COMPLETE" or int(evaluation.get("formal_run_count",-1))!=1: raise RuntimeError("formal evaluation marker is invalid")
    if evaluation.get("baseline_regression",{}).get("status")!="PASS" or set(evaluation.get("baseline_regression",{}).get("backend",{}))!={"G1","C1"}: raise RuntimeError("formal baseline regression is not PASS for both backends")
    pool_integrity=evaluation.get("pool_integrity",{})
    if pool_integrity.get("status")!="PASS" or set(pool_integrity.get("backend",{}))!={"G1","C1"} or any(record.get("top5_j_at_5_invariant") is not True or record.get("allnms_oracle_all_invariant") is not True for record in pool_integrity["backend"].values()): raise RuntimeError("formal Top5/AllNMS pool invariance is not PASS")
    if evaluation_transaction.get("status")!="CONSUMED" or evaluation_transaction.get("evaluation_complete_sha256")!=_sha(run/"09_formal_test"/"EVALUATION_COMPLETE.json"): raise RuntimeError("formal evaluation transaction is invalid")
    bindings={
        "primary_lock_sha256":_sha(lock_path),
        "predictions_complete_sha256":_sha(prediction_path),
        "test_access_transaction_sha256":_sha(run/"09_formal_test"/"TEST_ACCESS_TRANSACTION.json"),
    }
    for key,value in bindings.items():
        if evaluation.get(key)!=value or evaluation_transaction.get(key)!=value: raise RuntimeError(f"formal binding mismatch: {key}")
    expected_independent={"primary_lock":bindings["primary_lock_sha256"],"predictions_complete":bindings["predictions_complete_sha256"],"test_access_transaction":bindings["test_access_transaction_sha256"],"evaluation_complete":_sha(run/"09_formal_test"/"EVALUATION_COMPLETE.json"),"evaluation_artifact_inventory":hashlib.sha256(json.dumps(evaluation.get("evaluation_artifacts",[]),sort_keys=True).encode()).hexdigest()}
    if independent.get("upstream_sha256")!=expected_independent or independent.get("within_family_holm_recomputed") is not True: raise RuntimeError("independent recompute is not bound to the final formal artifacts")
    required_scope={"G1 primary ungated/gated","Track-C router","validation-locked secondary union"}
    if set(independent.get("independent_scope",[]))!=required_scope: raise RuntimeError("independent recompute scope does not cover core + Track C")
    statistics=pd.read_csv(run/"tables"/"statistical_tests.csv")
    if _sha(run/"tables"/"statistical_tests.csv")!=_sha(run/"10_statistics"/"statistical_tests.csv"): raise RuntimeError("tables/ and 10_statistics/ statistical outputs differ")
    required_statistics={"family","backend","pool","method","pvalue","holm_adjusted_p","bootstrap_lower","bootstrap_upper","bootstrap_iterations","sample_count","bootstrap_sample_count","bootstrap_cluster_count","bootstrap_resampling_unit","bootstrap_point_estimate","comparison_key"}
    if not required_statistics.issubset(statistics.columns) or statistics.empty: raise RuntimeError("statistical test schema is incomplete")
    numeric=statistics[["pvalue","holm_adjusted_p","bootstrap_lower","bootstrap_upper","bootstrap_iterations"]].apply(pd.to_numeric,errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or not numeric["bootstrap_iterations"].eq(10_000).all(): raise RuntimeError("statistical tests contain invalid values or bootstrap counts")
    if not numeric["pvalue"].between(0,1).all() or not numeric["holm_adjusted_p"].between(0,1).all(): raise RuntimeError("statistical p-values are outside [0,1]")
    points=pd.to_numeric(statistics["bootstrap_point_estimate"],errors="coerce")
    if statistics["comparison_key"].astype(str).duplicated().any() or not numeric["bootstrap_lower"].between(-1,1).all() or not numeric["bootstrap_upper"].between(-1,1).all() or not points.between(-1,1).all() or not (numeric["bootstrap_lower"]<=numeric["bootstrap_upper"]).all(): raise RuntimeError("statistical keys/effect intervals are invalid")
    if not {"formal_intra_backend_all_locked_methods","formal_cross_backend_locked_methods"}.issubset(set(statistics["family"].astype(str))): raise RuntimeError("statistical comparison families are incomplete")
    prediction_plan=json.loads(Path(str(lock["prediction_plan"]["path"])).read_text())
    if list(manifest.get("seeds",[]))!=list(prediction_plan.get("seeds",[])): raise RuntimeError("run manifest seeds differ from the locked prediction plan")
    expected_family_sizes={str(key):int(value) for key,value in prediction_plan["statistical_families"].items()}
    if statistics.groupby("family").size().astype(int).to_dict()!=expected_family_sizes: raise RuntimeError("statistical hypothesis inventory differs from the locked plan")
    expected_keys={f"{backend}/{pool}/{method}" for backend in ("g1","c1") for pool in prediction_plan["pools"] for method in [*prediction_plan["manual_methods"],*prediction_plan["matrix_methods"],prediction_plan["pooled_backend_conditioned"]["method"]]}
    expected_keys.update(f"{backend}/{lock['selected_methods']['backend'][backend.upper()]['primary_pool']}/primary_gated" for backend in ("g1","c1"))
    expected_keys.add("g1+c1/top1_pair/backend_top1_router")
    expected_keys.update(f"g1+c1/{route}_{pool}/{method}" for route in ("union_concat","union_nms") for pool in prediction_plan["pools"] for method in ("union_mlp","union_deepsets","union_gnn"))
    if set(statistics["comparison_key"].astype(str))!=expected_keys: raise RuntimeError("statistical comparison-key set differs from the locked plan")
    hypothesis_records=independent.get("hypothesis_records",[])
    hypothesis_keys=[str(record.get("comparison_key","")) for record in hypothesis_records]
    if independent.get("all_hypothesis_contingencies_recomputed") is not True or int(independent.get("hypothesis_count",-1))!=len(expected_keys): raise RuntimeError("independent all-hypothesis recomputation declaration is incomplete")
    if len(hypothesis_keys)!=len(expected_keys) or len(set(hypothesis_keys))!=len(hypothesis_keys) or set(hypothesis_keys)!=expected_keys: raise RuntimeError("independent hypothesis inventory differs from the locked plan")
    for record in hypothesis_records:
        checks=record.get("checks",{})
        path=Path(str(record.get("outcome_path",""))).expanduser().resolve()
        if not checks or not all(value is True for value in checks.values()): raise RuntimeError(f"independent hypothesis checks failed: {record.get('comparison_key')}")
        if not path.is_file() or _sha(path)!=str(record.get("outcome_sha256","")): raise RuntimeError(f"independent hypothesis outcome drift: {record.get('comparison_key')}")
    expected_independent_records=set()
    for backend in ("G1","C1"):
        selected=lock["selected_methods"]["backend"][backend]
        expected_independent_records.add((backend,str(selected["primary_ungated_method"]),str(selected["primary_pool"])))
        expected_independent_records.add((backend,"primary_gated",str(selected["primary_pool"])))
    expected_independent_records.add(("G1+C1","backend_top1_router","top1_pair"))
    union_lock=lock["selected_methods"]["cross_backend"]["union"]
    expected_independent_records.add(("G1+C1",str(union_lock["method"]),f"{union_lock['track']}_{union_lock['pool']}"))
    independent_records=independent.get("records",[])
    independent_record_keys=[(str(record.get("backend","")),str(record.get("method","")),str(record.get("pool",""))) for record in independent_records]
    if len(independent_record_keys)!=6 or len(set(independent_record_keys))!=6 or set(independent_record_keys)!=expected_independent_records: raise RuntimeError("independent core record inventory differs from the lock")
    for record,key in zip(independent_records,independent_record_keys,strict=True):
        checks=record.get("checks",{})
        path=Path(str(record.get("outcome_path",""))).expanduser().resolve()
        if not checks or not all(value is True for value in checks.values()): raise RuntimeError(f"independent core checks failed: {key}")
        if not path.is_file() or _sha(path)!=str(record.get("outcome_sha256","")): raise RuntimeError(f"independent core outcome drift: {key}")
    expected_samples=int(audit["split_audit"]["test"]["sample_count"]); expected_scenes=int(audit["split_audit"]["test"]["scene_count"])
    if not statistics["sample_count"].astype(int).eq(expected_samples).all() or not statistics["bootstrap_sample_count"].astype(int).eq(expected_samples).all() or not statistics["bootstrap_cluster_count"].astype(int).eq(expected_scenes).all() or not statistics["bootstrap_resampling_unit"].astype(str).eq("cluster").all(): raise RuntimeError("statistical denominator/scene-bootstrap audit failed")
    if not np.allclose(statistics["bootstrap_point_estimate"].astype(float),statistics["net_recovered"].astype(float)/expected_samples,rtol=0,atol=1e-15): raise RuntimeError("bootstrap point estimates differ from complete-denominator effects")
    gallery=pd.read_csv(run/"12_failure_galleries"/"gallery_manifest.csv")
    summaries=gallery.loc[gallery["sample_id"].astype(str).eq("__SUMMARY__")]
    local_categories={"recovered","harmful","unchanged_wrong_solvable","no_positive","gate_prevented_harmful"}
    expected_gallery={(backend,category) for backend in ("G1","C1") for category in local_categories}|{("G1+C1","router_success"),("G1+C1","router_failure")}
    observed_gallery=set(zip(summaries["backend"].astype(str),summaries["category"].astype(str)))
    if observed_gallery!=expected_gallery or summaries.duplicated(["backend","category"]).any() or not (summaries["rendered"].astype(int)==summaries[["eligible","requested"]].min(axis=1).astype(int)).all(): raise RuntimeError("gallery summary inventory/quotas are inconsistent")
    details=gallery.loc[~gallery["sample_id"].astype(str).eq("__SUMMARY__")].copy()
    if details.empty or details.duplicated(["backend","category","sample_id"]).any(): raise RuntimeError("gallery details are empty or duplicate samples")
    expected_gallery_samples: dict[tuple[str,str],set[str]]={}
    for backend in ("g1","c1"):
        selected=lock["selected_methods"]["backend"][backend.upper()]
        pool=str(selected["primary_pool"]); method=str(selected["primary_ungated_method"])
        candidates=pd.read_parquet(run/"02_features"/"test"/backend/f"{pool}_features.parquet")
        labels=pd.read_parquet(run/"09_formal_test"/f"{backend}_candidate_labels.parquet")
        positive=candidates[["sample_id","stable_candidate_id"]].merge(labels[["sample_id","stable_candidate_id","candidate_correct"]],on=["sample_id","stable_candidate_id"],validate="one_to_one").groupby("sample_id")["candidate_correct"].any()
        ungated=pd.read_parquet(run/"09_formal_test"/"outcomes"/backend/pool/f"{method}_ensemble.parquet").set_index("sample_id")
        gated=pd.read_parquet(run/"09_formal_test"/"outcomes"/backend/pool/"primary_gated.parquet").set_index("sample_id")
        category_ids={
            "recovered":ungated.index[ungated["recovered"]].tolist(),
            "harmful":ungated.index[ungated["harmful"]].tolist(),
            "unchanged_wrong_solvable":ungated.index[(~ungated["final_correct"].astype(bool))&ungated.index.to_series().map(positive).fillna(False).to_numpy()].tolist(),
            "no_positive":[sample_id for sample_id,value in positive.items() if not bool(value)],
            "gate_prevented_harmful":ungated.index[ungated["harmful"].astype(bool)&~gated["harmful"].astype(bool)].tolist(),
        }
        quota={"recovered":20,"harmful":20,"unchanged_wrong_solvable":20,"no_positive":10,"gate_prevented_harmful":10}
        for category,ids in category_ids.items():
            expected_gallery_samples[(backend.upper(),category)]=set(sorted(map(str,ids))[:quota[category]])
    router_outcome=pd.read_parquet(run/"09_formal_test"/"outcomes"/"cross_backend"/"router.parquet").set_index("sample_id")
    expected_gallery_samples[("G1+C1","router_success")]=set(sorted(map(str,router_outcome.index[router_outcome["recovered"]].tolist()))[:10])
    expected_gallery_samples[("G1+C1","router_failure")]=set(sorted(map(str,router_outcome.index[router_outcome["harmful"]].tolist()))[:10])
    for summary in summaries.itertuples(index=False):
        local=details.loc[(details["backend"].astype(str).eq(str(summary.backend)))&(details["category"].astype(str).eq(str(summary.category)))]
        if len(local)!=int(summary.rendered): raise RuntimeError(f"gallery detail count mismatch: {summary.backend}/{summary.category}")
        key=(str(summary.backend),str(summary.category))
        if set(local["sample_id"].astype(str))!=expected_gallery_samples[key]: raise RuntimeError(f"gallery category predicate/sample selection drift: {summary.backend}/{summary.category}")
        for path_value in local["path"].astype(str):
            path=Path(path_value).expanduser().resolve()
            if not path.is_file() or path.stat().st_size<=0: raise RuntimeError(f"gallery artifact missing: {path}")
            try:
                with Image.open(path) as rendered_image: rendered_image.verify()
            except Exception as error:
                raise RuntimeError(f"gallery image is not decodable: {path}") from error
        folder="cross_backend" if str(summary.backend)=="G1+C1" else str(summary.backend).lower()
        contact=run/"12_failure_galleries"/folder/f"{summary.category}_contact_sheet.png"
        if int(summary.rendered)>0 and (not contact.is_file() or contact.stat().st_size<=0): raise RuntimeError(f"gallery contact sheet missing: {contact}")
        if int(summary.rendered)>0:
            try:
                with Image.open(contact) as contact_image: contact_image.verify()
            except Exception as error:
                raise RuntimeError(f"gallery contact sheet is not decodable: {contact}") from error
    figure_stems={
        "01_candidate_count_distribution","02_oracle_at_k","03_j1_delta_ci","04_recovered_harmful","05_headroom_recovery",
        "06_loss_comparison","07_encoder_comparison","08_feature_cumulative","09_feature_leave_one_out","10_gate_risk_coverage",
        "11_calibration_reliability","12_switch_net_gain","13_result_by_query_type","14_result_by_candidate_count","15_result_by_baseline_margin",
        "16_result_by_mask_quality","17_cross_backend_complementarity","18_runtime_vs_gain","19_parameters_vs_gain","20_feature_importance",
    }
    for stem in figure_stems:
        for suffix in ("pdf","svg","png","caption.txt"):
            path=run/"11_figures"/f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size<=0: raise RuntimeError(f"required publication figure missing: {path}")
    independent_router=[row for row in independent.get("records",[]) if row.get("method")=="backend_top1_router"]
    conclusion=json.loads((run/"13_reports"/"EXPERIMENT_CONCLUSION.json").read_text())
    cross_conclusion=conclusion.get("cross_backend",{})
    router_stat=statistics.loc[statistics["comparison_key"].astype(str).eq("g1+c1/top1_pair/backend_top1_router")]
    if len(independent_router)!=1 or len(router_stat)!=1: raise RuntimeError("Track-C router lacks independent/statistical evidence")
    router_stat=router_stat.iloc[0]
    if cross_conclusion.get("method")!="backend_top1_router" or cross_conclusion.get("decision")!=independent_router[0].get("decision") or not np.isclose(float(cross_conclusion.get("delta_j_at_1",np.nan)),float(independent_router[0]["delta_j_at_1"]),rtol=0,atol=1e-15) or not np.allclose(np.asarray(cross_conclusion.get("ci",[]),dtype=float),np.asarray([router_stat.bootstrap_lower,router_stat.bootstrap_upper],dtype=float),rtol=0,atol=1e-15) or not np.isclose(float(cross_conclusion.get("holm_adjusted_p",np.nan)),float(router_stat.holm_adjusted_p),rtol=0,atol=1e-15): raise RuntimeError("Track-C deployment conclusion is not bound to independent/statistical recomputation")
    if conclusion.get("test_reselection") is not False or conclusion.get("r12_status")!="UNAVAILABLE_WITH_EVIDENCE" or conclusion.get("r13_status")!="COMPLETE": raise RuntimeError("experiment conclusion availability/reselection contract is invalid")
    primary_table=pd.read_csv(run/"tables"/"formal_test_primary.csv")
    gated=primary_table.loc[primary_table["method"].astype(str).eq("primary_gated")]
    if set(conclusion.get("primary",{}))!={"G1","C1"} or set(gated["backend"].astype(str))!={"G1","C1"}: raise RuntimeError("experiment conclusion lacks both gated primary routes")
    for row in gated.itertuples(index=False):
        declared=conclusion["primary"][str(row.backend)]
        if declared.get("method")!="primary_gated" or declared.get("decision")!=str(row.decision) or not np.isclose(float(declared.get("delta_j_at_1",np.nan)),float(row.delta_j_at_1),rtol=0,atol=1e-15) or not np.allclose(np.asarray(declared.get("ci",[]),dtype=float),np.asarray([row.ci_lower,row.ci_upper],dtype=float),rtol=0,atol=1e-15) or not np.isclose(float(declared.get("holm_adjusted_p",np.nan)),float(row.holm_adjusted_p),rtol=0,atol=1e-15): raise RuntimeError(f"primary conclusion drift: {row.backend}")
    # A reproduction script documents the exact lifecycle.  The formal script
    # intentionally reuses sealed predictions and never reopens Test labels.
    python=Path(sys.executable).resolve()
    probe=subprocess.run([str(python),"-c","import numpy,pandas,scipy,pyarrow"],text=True,capture_output=True)
    if probe.returncode!=0: raise RuntimeError(f"reproduction Python dependency probe failed: {probe.stderr}")
    help_probe=subprocess.run([str(python),str(project/"tools"/"g1_c1_rerank"/"independently_recompute.py"),"--help"],env={**os.environ,"PYTHONPATH":str(repo)},text=True,capture_output=True)
    if help_probe.returncode!=0: raise RuntimeError(f"reproduction entry-point probe failed: {help_probe.stderr}")
    packages=["numpy","pandas","scipy","scikit-learn","statsmodels","joblib","torch","lightgbm","pyarrow"]
    versions={}
    for name in packages:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]="not-installed (optional/fallback permitted)"
    environment={"python_executable":str(python),"python_version":sys.version,"packages":versions,"core_recompute_probe":"PASS"}
    (run/"ENVIRONMENT_LOCK.json").write_text(json.dumps(environment,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    common=f"PROJECT=${{PROJECT:-'{project}'}}\nBASE_RUN=${{BASE_RUN:-'{base}'}}\nRUN_DIR=${{RUN_DIR:-'{run}'}}\nPYTHONPATH=${{PYTHONPATH:-'{repo}'}}\nPYTHON=${{PYTHON:-'{python}'}}\ncd \"$PROJECT\"\n"
    reproduce_all="#!/bin/zsh\nset -euo pipefail\n"+common+"""
if [[ -e "$RUN_DIR/COMPLETE" ]]; then echo 'Refusing to mutate a sealed run; set RUN_DIR to a fresh run path.' >&2; exit 2; fi
mkdir -p "$RUN_DIR"
if [[ ! -e "$RUN_DIR/AUDIT_INVENTORY.json" ]]; then PYTHONPATH="$PYTHONPATH" "$PYTHON" -m tools.g1_c1_rerank audit --base-run "$BASE_RUN" --run-dir "$RUN_DIR"; fi
if [[ ! -e "$RUN_DIR/audit/frozen_pool_inventory.json" ]]; then PYTHONPATH="$PYTHONPATH" "$PYTHON" -m tools.g1_c1_rerank freeze-pools --base-run "$BASE_RUN" --run-dir "$RUN_DIR"; fi
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/write_protocol.py "$RUN_DIR" "$BASE_RUN"
for BACKEND in G1 C1; do
  PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/generate_backend_candidates.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --backend "$BACKEND" --resume
done
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/prepare_labels.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --backend all --split validation
for SPLIT in train validation test; do
  for BACKEND in G1 C1; do
    PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/build_features.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --backend "$BACKEND" --split "$SPLIT" --resume
    PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/build_candidate_crops.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --backend "$BACKEND" --split "$SPLIT" --resume
  done
  PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/build_cross_backend_evidence.py --run-dir "$RUN_DIR" --split "$SPLIT"
done
OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/smoke_matrix.py "$RUN_DIR"
if [[ ! -e "$RUN_DIR/00_audit/FINAL_MATRIX_CODE_PROVENANCE.json" ]]; then PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/matrix_provenance.py start --run-dir "$RUN_DIR"; fi
OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/run_local_matrix.py all --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --device mps --resume
OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/run_crop_cnn.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --device mps --resume
OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/run_validation_followups.py all --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --device mps --resume
OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/run_cross_backend.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --device mps --resume
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/matrix_provenance.py finish --run-dir "$RUN_DIR"
if [[ ! -e "$RUN_DIR/08_lock/PRIMARY_METHOD_LOCK.json" ]]; then PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/lock_validation_selection.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR"; fi
if [[ ! -e "$RUN_DIR/09_formal_test/PREDICTIONS_COMPLETE.json" ]]; then PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/run_formal_test_once.py predict --base-run "$BASE_RUN" --run-dir "$RUN_DIR"; fi
if [[ ! -e "$RUN_DIR/09_formal_test/TEST_ACCESS_TRANSACTION.json" ]]; then
  if [[ "${ALLOW_ONE_TIME_FORMAL_TEST:-0}" != 1 ]]; then echo 'Development and label-free Test predictions are complete. Set ALLOW_ONE_TIME_FORMAL_TEST=1 to authorize the one-time local Test-label transaction.' >&2; exit 3; fi
  PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/prepare_labels.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --backend all --split test --unlock-test-once
fi
if [[ ! -e "$RUN_DIR/09_formal_test/EVALUATION_COMPLETE.json" ]]; then PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/run_formal_test_once.py evaluate --base-run "$BASE_RUN" --run-dir "$RUN_DIR"; fi
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/independently_recompute.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR"
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/build_failure_galleries.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR"
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/build_final_reports.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR"
PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/finalize_complete_run.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR"
"""
    reproduce_formal="#!/bin/zsh\nset -euo pipefail\n"+common+"""
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/g1-c1-independent.XXXXXX")}
OMP_NUM_THREADS=1 PYTHONPATH="$PYTHONPATH" "$PYTHON" tools/g1_c1_rerank/independently_recompute.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR" --output-dir "$OUTPUT_DIR"
echo "Read-only independent recomputation written to $OUTPUT_DIR"
"""
    for name,text in (("reproduce_all.sh",reproduce_all),("reproduce_formal_test.sh",reproduce_formal)):
        path=run/name; path.write_text(text,encoding="utf-8"); path.chmod(path.stat().st_mode|stat.S_IXUSR)
    readme=f"""# G1/C1 Complete Reranking Reproduction

- Source baseline: `{base}`
- Experiment run: `{run}`
- Python: `{python}` (exact versions in `ENVIRONMENT_LOCK.json`)
- Formal Test labels were opened exactly once after `08_lock/PRIMARY_METHOD_LOCK.json` and must not be reopened.
- `reproduce_all.sh` covers the complete fresh-run lifecycle. It stops after locked label-free Test predictions unless `ALLOW_ONE_TIME_FORMAL_TEST=1` explicitly authorizes the single local Test-label transaction.
- `reproduce_formal_test.sh` independently recomputes metrics from sealed predictions and labels into a temporary/output directory without mutating the sealed run.
- Claims concern offline target-specific 2D 4-DoF rectangle consistency, not physical grasp success.
"""
    (run/"README_REPRODUCE.md").write_text(readme,encoding="utf-8")
    process=subprocess.run(["ps","-axo","pid=,command="],check=True,text=True,capture_output=True).stdout
    active=[line.strip() for line in process.splitlines() if "g1_c1_rerank" in line and "finalize_complete_run.py" not in line]
    if active: raise RuntimeError(f"reranking processes still active: {active}")
    active_marker=run/".RUN_ACTIVE"; finished_marker=run/".RUN_FINISHED"
    if not active_marker.is_file(): raise RuntimeError("RUN_ACTIVE marker is missing before final seal")
    # Self-referential seal metadata is excluded; every experimental artifact,
    # prediction, report, model, and source snapshot remains covered.
    excluded={"COMPLETE","MANIFEST.json","manifest.json","RUN_SHA256_MANIFEST.txt","RUN_LOCK_SHA256.txt",".RUN_ACTIVE",".RUN_FINISHED"}
    files=sorted(path for path in run.rglob("*") if path.is_file() and path.relative_to(run).as_posix() not in excluded and ".tmp-" not in path.name)
    lines=[f"{_sha(path)}  {path.relative_to(run)}" for path in files]
    sha_manifest=run/"RUN_SHA256_MANIFEST.txt"; sha_manifest.write_text("\n".join(lines)+"\n",encoding="utf-8")
    lock_sha=_sha(sha_manifest); (run/"RUN_LOCK_SHA256.txt").write_text(lock_sha+"\n",encoding="utf-8")
    manifest["status"]="COMPLETE"; manifest["formal_test_status"]="COMPLETE_ONCE"; manifest["run_lock_sha256"]=lock_sha; manifest["artifact_count"]=len(files)
    manifest_text=json.dumps(manifest,indent=2,sort_keys=True)+"\n"
    for manifest_name in ("MANIFEST.json","manifest.json"):
        temporary=run/f".{manifest_name}.tmp-final"
        temporary.write_text(manifest_text,encoding="utf-8")
        os.replace(temporary,run/manifest_name)
    complete={"status":"COMPLETE","run":str(run),"run_lock_sha256":lock_sha,"formal_run_count":1,"independent_recompute":"PASS","residual_processes":0}
    complete_path=run/"COMPLETE"; complete_temporary=run/".COMPLETE.tmp-final"
    active_marker.rename(finished_marker)
    try:
        with complete_temporary.open("w",encoding="utf-8") as stream:
            stream.write(json.dumps(complete,indent=2,sort_keys=True)+"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(complete_temporary,complete_path)
    except BaseException:
        if finished_marker.exists() and not active_marker.exists():
            finished_marker.rename(active_marker)
        raise
    print(json.dumps(complete,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
