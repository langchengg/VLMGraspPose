#!/usr/bin/env python3
"""Independent ID-join recomputation of locked primary formal results.

This module intentionally imports neither model training nor the main evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _verify_entries(entries: list[dict[str,Any]], *, label: str) -> None:
    if not entries: raise PermissionError(f"{label} inventory is empty")
    for entry in entries:
        path_value=entry.get("path",entry.get("label_path")); sha_value=entry.get("sha256",entry.get("label_sha256"))
        if path_value is None or sha_value is None: raise PermissionError(f"{label} entry lacks path/hash")
        path=Path(str(path_value)).expanduser().resolve()
        if not path.is_file() or _sha(path)!=str(sha_value): raise PermissionError(f"{label} artifact drift: {path}")


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _evaluate(
    universe: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    baseline_ids: pd.DataFrame,
    selected_ids: pd.DataFrame,
    *,
    rank_column: str = "original_rank",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    universe=universe.copy(); candidates=candidates.copy(); labels=labels.copy(); baseline_ids=baseline_ids.copy(); selected_ids=selected_ids.copy()
    key = ["sample_id", "stable_candidate_id"]
    if not {"sample_id", "scene_id"}.issubset(universe):
        raise ValueError("independent universe schema is incomplete")
    if universe[["sample_id", "scene_id"]].isna().any().any():
        raise ValueError("independent universe contains null identifiers")
    universe["sample_id"]=universe["sample_id"].astype(str); universe["scene_id"]=universe["scene_id"].astype(str)
    if universe["sample_id"].duplicated().any() or universe[["sample_id", "scene_id"]].eq("").any().any():
        raise ValueError("independent universe identifiers are duplicate or empty")
    for frame in (candidates,labels):
        frame["sample_id"]=frame["sample_id"].astype(str); frame["stable_candidate_id"]=frame["stable_candidate_id"].astype(str)
    for frame,column in ((baseline_ids,"baseline_candidate_id"),(selected_ids,"selected_candidate_id")):
        frame["sample_id"]=frame["sample_id"].astype(str); frame[column]=frame[column].fillna("").astype(str)
    correctness=pd.to_numeric(labels["candidate_correct"],errors="coerce")
    if correctness.isna().any() or not correctness.isin([0,1]).all(): raise ValueError("independent labels are not finite binary values")
    labels["candidate_correct"]=correctness.astype(bool)
    if candidates.duplicated(key).any() or labels.duplicated(key).any():
        raise ValueError("independent evaluator received duplicate candidate keys")
    candidate_keys = set(map(tuple, candidates[key].astype(str).to_numpy()))
    label_keys = set(map(tuple, labels[key].astype(str).to_numpy()))
    if candidate_keys != label_keys:
        raise ValueError("independent candidate/label key sets differ")
    universe_ids = set(universe["sample_id"].astype(str))
    if not set(candidates["sample_id"].astype(str)).issubset(universe_ids):
        raise ValueError("independent candidate lies outside the universe")
    truth = labels.assign(
        sample_id=labels["sample_id"].astype(str),
        stable_candidate_id=labels["stable_candidate_id"].astype(str),
    ).set_index(key)["candidate_correct"]
    baseline = baseline_ids.set_index("sample_id")["baseline_candidate_id"]
    selected = selected_ids.set_index("sample_id")["selected_candidate_id"]
    if baseline_ids["sample_id"].astype(str).duplicated().any() or selected_ids["sample_id"].astype(str).duplicated().any():
        raise ValueError("independent selections contain duplicate sample IDs")
    expected_nonempty = set(candidates["sample_id"].astype(str))
    if set(baseline.index.astype(str)) != expected_nonempty:
        raise ValueError("independent baseline IDs do not exactly cover non-empty samples")
    if set(selected.index.astype(str)) != expected_nonempty:
        raise ValueError("independent selected IDs do not exactly cover non-empty samples")
    required_keys={(str(sample_id),str(candidate_id)) for sample_id,candidate_id in [*baseline.items(),*selected.items()] if str(candidate_id)}
    if not required_keys.issubset(candidate_keys):
        raise ValueError("independent selection references a candidate outside the frozen pool")
    rows = []
    for sample in universe.itertuples(index=False):
        sample_id = str(sample.sample_id)
        old_id = str(baseline.get(sample_id, ""))
        new_id = str(selected.get(sample_id, old_id))
        old = bool(truth.loc[(sample_id, old_id)]) if old_id else False
        new = bool(truth.loc[(sample_id, new_id)]) if new_id else False
        rows.append(
            {
                "sample_id": sample_id,
                "scene_id": str(sample.scene_id),
                "baseline_candidate_id": old_id,
                "selected_candidate_id": new_id,
                "baseline_correct": old,
                "final_correct": new,
                "switch": old_id != new_id,
                "recovered": (not old) and new,
                "harmful": old and (not new),
            }
        )
    outcomes = pd.DataFrame(rows)
    recovered = int(outcomes["recovered"].sum())
    harmful = int(outcomes["harmful"].sum())
    discordant = recovered + harmful
    oracle = labels.groupby("sample_id")["candidate_correct"].any()
    if rank_column not in candidates.columns: raise ValueError(f"independent candidates lack rank column {rank_column}")
    oracle5 = candidates.merge(labels[key + ["candidate_correct"]], on=key, validate="one_to_one").loc[lambda frame: frame[rank_column].le(5)].groupby("sample_id")["candidate_correct"].any()
    total = len(universe)
    result = {
        "sample_count": total,
        "candidate_rows": len(candidates),
        "baseline_j_at_1": float(outcomes["baseline_correct"].sum() / total),
        "j_at_1": float(outcomes["final_correct"].sum() / total),
        "delta_j_at_1": float((recovered - harmful) / total),
        "oracle_at_5": float(oracle5.sum() / total),
        "oracle_at_all": float(oracle.sum() / total),
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switch_count": int(outcomes["switch"].sum()),
        "switch_rate": float(outcomes["switch"].mean()),
        "mcnemar_p": 1.0 if discordant == 0 else float(binomtest(recovered, discordant, p=.5).pvalue),
        "both_wrong": int(((~outcomes["baseline_correct"]) & (~outcomes["final_correct"])).sum()),
        "both_correct": int((outcomes["baseline_correct"] & outcomes["final_correct"]).sum()),
    }
    return outcomes, result


def _stable_seed(base: int, method: str) -> int:
    digest=hashlib.sha256(f"{base}\t{method}".encode()).digest()
    return int.from_bytes(digest[:4],"big")


def _scene_ci(outcomes: pd.DataFrame, *, seed: int, draws: int=10_000) -> tuple[float,float]:
    clusters=outcomes["scene_id"].astype(str).to_numpy(dtype=object)
    values=outcomes["final_correct"].astype(float).to_numpy()-outcomes["baseline_correct"].astype(float).to_numpy()
    unique,inverse=np.unique(clusters,return_inverse=True)
    sums=np.bincount(inverse,weights=values,minlength=len(unique)); counts=np.bincount(inverse,minlength=len(unique))
    rng=np.random.default_rng(seed); distribution=np.empty(draws,dtype=float)
    for start in range(0,draws,1000):
        stop=min(start+1000,draws); sampled=rng.integers(0,len(unique),size=(stop-start,len(unique)))
        distribution[start:stop]=sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)
    low,high=np.quantile(distribution,[.025,.975]); return float(low),float(high)


def _holm(values: np.ndarray) -> np.ndarray:
    order=np.argsort(values,kind="stable"); adjusted_sorted=np.empty(len(values)); running=0.0
    for position,index in enumerate(order):
        running=max(running,min(1.0,float((len(values)-position)*values[index]))); adjusted_sorted[position]=running
    adjusted=np.empty(len(values)); adjusted[order]=adjusted_sorted; return adjusted


def _rank_metrics(candidates: pd.DataFrame, labels: pd.DataFrame, scores: pd.DataFrame, total: int) -> dict[str,float]:
    key=["sample_id","stable_candidate_id"]
    for frame,name in ((candidates,"candidates"),(labels,"labels"),(scores,"scores")):
        if frame.duplicated(key).any(): raise ValueError(f"independent {name} contains duplicate candidate keys")
    candidate_keys=set(map(tuple,candidates[key].astype(str).to_numpy())); score_keys=set(map(tuple,scores[key].astype(str).to_numpy()))
    if candidate_keys!=score_keys or len(scores)!=len(candidates): raise ValueError("independent score keys do not exactly match candidates")
    if not np.isfinite(pd.to_numeric(scores["score"],errors="coerce").to_numpy(dtype=float)).all(): raise ValueError("independent scores are non-finite")
    joined=candidates.merge(labels[["sample_id","stable_candidate_id","candidate_correct"]],on=["sample_id","stable_candidate_id"],validate="one_to_one").merge(scores,on=["sample_id","stable_candidate_id"],validate="one_to_one")
    ordered=joined.sort_values(["sample_id","score","stable_candidate_id"],ascending=[True,False,True],kind="mergesort")
    j5=ordered.groupby("sample_id",sort=False).head(5).groupby("sample_id")["candidate_correct"].any().sum()/total
    return {"j_at_5":float(j5)}


def _formal_outcome_path(
    run: Path,
    key: str,
    plan: dict[str, Any],
) -> Path:
    backend, pool, method = key.split("/", 2)
    if backend in {"g1", "c1"}:
        if method == "primary_gated":
            return run / "09_formal_test" / "outcomes" / backend / pool / "primary_gated.parquet"
        if method == str(plan["pooled_backend_conditioned"]["method"]):
            return run / "09_formal_test" / "outcomes" / "pooled" / pool / f"{backend}_ensemble.parquet"
        suffix = "fixed" if method in set(plan["manual_methods"]) else "ensemble"
        return run / "09_formal_test" / "outcomes" / backend / pool / f"{method}_{suffix}.parquet"
    if pool == "top1_pair" and method == "backend_top1_router":
        return run / "09_formal_test" / "outcomes" / "cross_backend" / "router.parquet"
    route, union_pool = pool.rsplit("_", 1)
    return run / "09_formal_test" / "outcomes" / "cross_backend" / route / union_pool / f"{method}.parquet"


def _independent_contingency(path: Path, universe: pd.DataFrame) -> dict[str, Any]:
    outcomes = pd.read_parquet(path)
    required = {
        "sample_id",
        "scene_id",
        "baseline_correct",
        "final_correct",
        "recovered",
        "harmful",
    }
    if not required.issubset(outcomes):
        raise ValueError(f"independent outcome schema incomplete: {path}")
    if len(outcomes) != len(universe) or outcomes["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"independent outcome denominator mismatch: {path}")
    expected_scene = universe.assign(
        sample_id=universe["sample_id"].astype(str),
        scene_id=universe["scene_id"].astype(str),
    ).set_index("sample_id")["scene_id"]
    observed = outcomes.assign(
        sample_id=outcomes["sample_id"].astype(str),
        scene_id=outcomes["scene_id"].astype(str),
    ).set_index("sample_id")
    if set(observed.index) != set(expected_scene.index) or not observed["scene_id"].eq(expected_scene).all():
        raise ValueError(f"independent outcome universe/scene mismatch: {path}")
    baseline = observed["baseline_correct"].astype(bool)
    final = observed["final_correct"].astype(bool)
    recovered_mask = (~baseline) & final
    harmful_mask = baseline & (~final)
    if not observed["recovered"].astype(bool).eq(recovered_mask).all() or not observed["harmful"].astype(bool).eq(harmful_mask).all():
        raise ValueError(f"independent outcome flags are internally inconsistent: {path}")
    recovered, harmful = int(recovered_mask.sum()), int(harmful_mask.sum())
    discordant = recovered + harmful
    return {
        "recovered": recovered,
        "harmful": harmful,
        "both_wrong": int(((~baseline) & (~final)).sum()),
        "both_correct": int((baseline & final).sum()),
        "pvalue": 1.0 if discordant == 0 else float(binomtest(recovered, discordant, p=0.5).pvalue),
        "outcome_path": str(path.resolve()),
        "outcome_sha256": _sha(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    base = args.base_run.expanduser().resolve()
    run = args.run_dir.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run / "14_independent_recompute"
    )
    if (run / "COMPLETE").exists():
        if args.output_dir is None:
            raise PermissionError("sealed-run recomputation requires an explicit external --output-dir")
        try:
            output_root.relative_to(run)
        except ValueError:
            pass
        else:
            raise PermissionError("sealed-run recomputation output must be outside the run")
    output_root.mkdir(parents=True, exist_ok=True)
    lock = json.loads((run / "08_lock" / "PRIMARY_METHOD_LOCK.json").read_text(encoding="utf-8"))
    lock_path=run/"08_lock"/"PRIMARY_METHOD_LOCK.json"; prediction_path=run/"09_formal_test"/"PREDICTIONS_COMPLETE.json"; access_path=run/"09_formal_test"/"TEST_ACCESS_TRANSACTION.json"; evaluation_path=run/"09_formal_test"/"EVALUATION_COMPLETE.json"
    if lock.get("status")!="LOCKED" or Path(str(lock.get("base_run",""))).resolve()!=base: raise PermissionError("independent recompute base/lock mismatch")
    for section in ("source_manifest_artifacts","source_label_artifacts","audit_artifacts","candidate_artifacts","feature_artifacts","locked_models","ranker_calibrators","code_artifacts","validation_selection_artifacts","lock_support_artifacts"):
        _verify_entries(lock.get(section,[]),label=f"lock/{section}")
    _verify_entries([lock.get("prediction_plan",{})],label="lock/prediction_plan")
    _verify_entries([lock.get("evaluator",{})],label="lock/evaluator")
    mirror_path=run/"08_lock"/"FORMAL_TEST_LOCK.json"
    if not mirror_path.is_file() or _sha(mirror_path)!=_sha(lock_path): raise PermissionError("independent formal-lock mirror mismatch")
    prediction=json.loads(prediction_path.read_text()); access=json.loads(access_path.read_text()); evaluation=json.loads(evaluation_path.read_text())
    universe_sha=_sha(base/"manifests"/"test_samples.parquet")
    if prediction.get("status")!="COMPLETE" or prediction.get("primary_lock_sha256")!=_sha(lock_path) or Path(str(prediction.get("base_run",""))).resolve()!=base or prediction.get("test_universe_sha256")!=universe_sha: raise PermissionError("independent prediction binding mismatch")
    _verify_entries(prediction.get("prediction_artifacts",[]),label="predictions")
    if access.get("status")!="CONSUMED" or access.get("primary_lock_sha256")!=_sha(lock_path) or access.get("predictions_complete_sha256")!=_sha(prediction_path) or Path(str(access.get("base_run",""))).resolve()!=base or access.get("test_universe_sha256")!=universe_sha: raise PermissionError("independent label transaction binding mismatch")
    _verify_entries(access.get("label_artifacts",[]),label="labels")
    if evaluation.get("status")!="COMPLETE" or evaluation.get("primary_lock_sha256")!=_sha(lock_path) or evaluation.get("predictions_complete_sha256")!=_sha(prediction_path) or evaluation.get("test_access_transaction_sha256")!=_sha(access_path) or Path(str(evaluation.get("base_run",""))).resolve()!=base or evaluation.get("test_universe_sha256")!=universe_sha: raise PermissionError("independent evaluation binding mismatch")
    _verify_entries(evaluation.get("evaluation_artifacts",[]),label="evaluation outputs")
    universe = pd.read_parquet(base / "manifests" / "test_samples.parquet", columns=["sample_id", "scene_id"])
    main_table = pd.read_csv(run / "tables" / "formal_test_primary.csv")
    statistical=pd.read_csv(run/"tables"/"statistical_tests.csv")
    plan=json.loads(Path(str(lock["prediction_plan"]["path"])).read_text())
    expected_family_sizes={str(key):int(value) for key,value in plan["statistical_families"].items()}
    observed_family_sizes=statistical.groupby("family").size().astype(int).to_dict()
    if observed_family_sizes!=expected_family_sizes: raise AssertionError(f"independent hypothesis inventory mismatch: {observed_family_sizes} != {expected_family_sizes}")
    expected_keys={f"{backend}/{pool}/{method}" for backend in ("g1","c1") for pool in plan["pools"] for method in [*plan["manual_methods"],*plan["matrix_methods"],plan["pooled_backend_conditioned"]["method"]]}
    expected_keys.update(f"{backend}/{lock['selected_methods']['backend'][backend.upper()]['primary_pool']}/primary_gated" for backend in ("g1","c1"))
    expected_keys.add("g1+c1/top1_pair/backend_top1_router")
    expected_keys.update(f"g1+c1/{route}_{pool}/{method}" for route in ("union_concat","union_nms") for pool in plan["pools"] for method in ("union_mlp","union_deepsets","union_gnn"))
    if statistical["comparison_key"].astype(str).duplicated().any() or set(statistical["comparison_key"].astype(str))!=expected_keys: raise AssertionError("independent statistical comparison-key set differs from the locked plan")
    expected_samples=len(universe); expected_clusters=int(universe["scene_id"].astype(str).nunique())
    if not statistical["sample_count"].astype(int).eq(expected_samples).all(): raise AssertionError("independent statistical sample denominator mismatch")
    if not statistical["bootstrap_sample_count"].astype(int).eq(expected_samples).all(): raise AssertionError("independent bootstrap sample denominator mismatch")
    if not statistical["bootstrap_cluster_count"].astype(int).eq(expected_clusters).all(): raise AssertionError("independent bootstrap scene-cluster count mismatch")
    if not statistical["bootstrap_resampling_unit"].astype(str).eq("cluster").all(): raise AssertionError("independent bootstrap resampling unit is not cluster")
    hypothesis_records: list[dict[str, Any]] = []
    independently_recomputed_p = np.empty(len(statistical), dtype=float)
    for index, row in statistical.reset_index(drop=True).iterrows():
        key = str(row["comparison_key"])
        outcome_path = _formal_outcome_path(run, key, plan)
        contingency = _independent_contingency(outcome_path, universe)
        checks = {
            "recovered": int(row["recovered"]) == contingency["recovered"],
            "harmful": int(row["harmful"]) == contingency["harmful"],
            "both_wrong": int(row["both_wrong"]) == contingency["both_wrong"],
            "both_correct": int(row["both_correct"]) == contingency["both_correct"],
            "pvalue": math.isclose(float(row["pvalue"]), contingency["pvalue"], abs_tol=1e-15),
        }
        if not all(checks.values()):
            raise AssertionError(f"independent hypothesis mismatch {key}: {checks}")
        independently_recomputed_p[index] = contingency["pvalue"]
        hypothesis_records.append(
            {
                "comparison_key": key,
                "family": str(row["family"]),
                **contingency,
                "checks": checks,
            }
        )
    for _,indices in statistical.groupby("family",sort=False).indices.items():
        index=np.asarray(indices,dtype=int); recomputed_holm=_holm(independently_recomputed_p[index])
        if not np.allclose(recomputed_holm,statistical.iloc[index]["holm_adjusted_p"].to_numpy(dtype=float),rtol=0,atol=1e-15): raise AssertionError("independent within-family Holm recomputation failed")
    records = []
    for backend in ("g1", "c1"):
        selected = lock["selected_methods"]["backend"][backend.upper()]
        method, pool = str(selected["primary_ungated_method"]), str(selected["primary_pool"])
        candidates = pd.read_parquet(run / "data" / f"frozen_{backend}_test_candidates.parquet")
        if pool == "top5":
            candidates = candidates.loc[candidates["original_rank"].le(5)].copy()
        labels = pd.read_parquet(run / "09_formal_test" / f"{backend}_candidate_labels.parquet")
        labels = labels.merge(candidates[["sample_id", "stable_candidate_id"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")
        baseline = candidates.loc[candidates["original_rank"].eq(1), ["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "baseline_candidate_id"})
        score = pd.read_parquet(run / "09_formal_test" / "candidate_scores" / backend / pool / f"{method}_ensemble.parquet")
        top = score.sort_values(["sample_id", "reranker_score", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").groupby("sample_id", sort=False).head(1)[["sample_id", "stable_candidate_id"]].rename(columns={"stable_candidate_id": "selected_candidate_id"})
        for name, selections in (
            (method, top),
            ("primary_gated", pd.read_parquet(run / "09_formal_test" / "primary" / backend / "gated_selections.parquet")[["sample_id", "selected_candidate_id"]]),
        ):
            outcomes, metric = _evaluate(universe, candidates, labels, baseline, selections)
            aligned_scores=score[["sample_id","stable_candidate_id","reranker_score"]].rename(columns={"reranker_score":"score"})
            if name=="primary_gated":
                selected_map=selections.set_index("sample_id")["selected_candidate_id"].astype(str)
                maximum=aligned_scores.groupby("sample_id")["score"].transform("max")
                chosen=aligned_scores["stable_candidate_id"].astype(str).eq(aligned_scores["sample_id"].astype(str).map(selected_map))
                aligned_scores.loc[chosen,"score"]=maximum.loc[chosen].to_numpy()+1.0
            metric.update(_rank_metrics(candidates,labels,aligned_scores,len(universe)))
            ci=_scene_ci(outcomes,seed=_stable_seed(20260806,f"formal/{backend}/{pool}/{name}")); metric["ci_lower"],metric["ci_upper"]=ci
            expected = main_table.loc[(main_table["backend"].astype(str).str.lower().eq(backend)) & main_table["method"].astype(str).eq(name)]
            if len(expected) != 1:
                raise AssertionError(f"missing main primary row: {backend}/{name}")
            row = expected.iloc[0]
            checks = {
                "j_at_1": math.isclose(metric["j_at_1"], float(row["j_at_1"]), abs_tol=1e-15),
                "recovered": metric["recovered"] == int(row["recovered"]),
                "harmful": metric["harmful"] == int(row["harmful"]),
                "net": metric["net"] == int(row["net"]),
                "switch_rate": math.isclose(metric["switch_rate"], float(row["switch_rate"]), abs_tol=1e-15),
                "j_at_5": math.isclose(metric["j_at_5"],float(row["j_at_5"]),abs_tol=1e-15),
                "oracle_at_5": math.isclose(metric["oracle_at_5"],float(row["oracle_at_5"]),abs_tol=1e-15),
                "oracle_at_all": math.isclose(metric["oracle_at_all"],float(row["oracle_at_all"]),abs_tol=1e-15),
                "mcnemar_p": math.isclose(metric["mcnemar_p"],float(row["mcnemar_p"]),abs_tol=1e-15),
                "ci_lower": math.isclose(metric["ci_lower"],float(row["ci_lower"]),abs_tol=1e-15),
                "ci_upper": math.isclose(metric["ci_upper"],float(row["ci_upper"]),abs_tol=1e-15),
            }
            stats_row=statistical.loc[(statistical["backend"].astype(str).str.lower().eq(backend))&(statistical["pool"].astype(str).eq(pool))&(statistical["method"].astype(str).eq(name))]
            if len(stats_row)!=1: raise AssertionError(f"missing independent contingency row {backend}/{name}")
            stats_row=stats_row.iloc[0]
            checks.update({
                "both_wrong":metric["both_wrong"]==int(stats_row["both_wrong"]),
                "both_correct":metric["both_correct"]==int(stats_row["both_correct"]),
                "recovered_contingency":metric["recovered"]==int(stats_row["recovered"]),
                "harmful_contingency":metric["harmful"]==int(stats_row["harmful"]),
                "holm":math.isclose(float(row["holm_adjusted_p"]),float(stats_row["holm_adjusted_p"]),abs_tol=1e-15),
                "decision":str(row["decision"])==("GO" if metric["delta_j_at_1"]>0 and metric["ci_lower"]>0 and float(row["holm_adjusted_p"])<.05 else "CAUTION" if metric["delta_j_at_1"]>0 else "NO-GO"),
            })
            if not all(checks.values()):
                raise AssertionError(f"independent recomputation mismatch {backend}/{name}: {checks}")
            metric.update({"backend": backend.upper(), "method": name, "pool": pool, "checks": checks})
            records.append(metric)
            destination = output_root / f"{backend}_{name}_outcomes.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            outcomes.to_parquet(destination, index=False, compression="zstd")
            metric["outcome_path"] = str(destination.resolve())
            metric["outcome_sha256"] = _sha(destination)

    # Independently verify the deployable Track-C router against always-G1.
    # Empty G1 predictions remain explicit incorrect baseline IDs; they are
    # never backfilled from C1.
    backend_candidates={}
    backend_labels={}
    for backend in ("g1","c1"):
        local=pd.read_parquet(run/"data"/f"frozen_{backend}_test_candidates.parquet")
        local=local.loc[local["original_rank"].eq(1)].copy()
        backend_candidates[backend]=local
        local_labels=pd.read_parquet(run/"09_formal_test"/f"{backend}_candidate_labels.parquet")
        backend_labels[backend]=local_labels.merge(local[["sample_id","stable_candidate_id"]],on=["sample_id","stable_candidate_id"],validate="one_to_one")
    router_candidates=pd.concat([backend_candidates["g1"],backend_candidates["c1"]],ignore_index=True)
    router_labels=pd.concat([backend_labels["g1"],backend_labels["c1"]],ignore_index=True)
    router_prediction=pd.read_parquet(run/"09_formal_test"/"cross_backend"/"router_selections.parquet")
    router_nonempty=router_prediction["g1_nonempty"].astype(bool)|router_prediction["c1_nonempty"].astype(bool)
    router_selection=router_prediction.loc[router_nonempty,["sample_id","g1_candidate_id","selected_candidate_id"]].rename(columns={"g1_candidate_id":"baseline_candidate_id"})
    router_selection["baseline_candidate_id"]=router_selection["baseline_candidate_id"].fillna("")
    router_selection["selected_candidate_id"]=router_selection["selected_candidate_id"].fillna(router_selection["baseline_candidate_id"])
    router_baseline=router_selection[["sample_id","baseline_candidate_id"]]
    router_selected=router_selection[["sample_id","selected_candidate_id"]]
    router_outcomes,router_metric=_evaluate(universe,router_candidates,router_labels,router_baseline,router_selected)
    router_key="g1+c1/top1_pair/backend_top1_router"
    router_ci=_scene_ci(router_outcomes,seed=_stable_seed(20260806,f"formal/{router_key}")); router_metric["ci_lower"],router_metric["ci_upper"]=router_ci
    router_expected=pd.read_csv(run/"tables"/"cross_backend.csv").loc[lambda frame:frame["method"].astype(str).eq("backend_top1_router")]
    router_stats=statistical.loc[statistical["comparison_key"].astype(str).eq(router_key)]
    if len(router_expected)!=1 or len(router_stats)!=1: raise AssertionError("independent router result/stat row is missing")
    router_expected=router_expected.iloc[0]; router_stats=router_stats.iloc[0]
    router_checks={
        "j_at_1":math.isclose(router_metric["j_at_1"],float(router_expected["j_at_1"]),abs_tol=1e-15),
        "delta_j_at_1":math.isclose(router_metric["delta_j_at_1"],float(router_expected["delta_j_at_1"]),abs_tol=1e-15),
        "recovered":router_metric["recovered"]==int(router_stats["recovered"]),
        "harmful":router_metric["harmful"]==int(router_stats["harmful"]),
        "net":router_metric["net"]==int(router_expected["net"]),
        "switch_count":router_metric["switch_count"]==int(router_expected["switch_count"]),
        "switch_rate":math.isclose(router_metric["switch_rate"],float(router_expected["switch_rate"]),abs_tol=1e-15),
        "both_wrong":router_metric["both_wrong"]==int(router_stats["both_wrong"]),
        "both_correct":router_metric["both_correct"]==int(router_stats["both_correct"]),
        "mcnemar_p":math.isclose(router_metric["mcnemar_p"],float(router_stats["pvalue"]),abs_tol=1e-15),
        "ci_lower":math.isclose(router_metric["ci_lower"],float(router_stats["bootstrap_lower"]),abs_tol=1e-15),
        "ci_upper":math.isclose(router_metric["ci_upper"],float(router_stats["bootstrap_upper"]),abs_tol=1e-15),
    }
    if not all(router_checks.values()): raise AssertionError(f"independent router mismatch: {router_checks}")
    router_decision="GO" if router_metric["delta_j_at_1"]>0 and router_metric["ci_lower"]>0 and float(router_stats["holm_adjusted_p"])<.05 else "CAUTION" if router_metric["delta_j_at_1"]>0 else "NO-GO"
    router_metric.update({"backend":"G1+C1","method":"backend_top1_router","pool":"top1_pair","decision":router_decision,"checks":router_checks})
    records.append(router_metric)
    router_output = output_root/"cross_backend_router_outcomes.parquet"
    router_outcomes.to_parquet(router_output,index=False,compression="zstd")
    router_metric["outcome_path"] = str(router_output.resolve())
    router_metric["outcome_sha256"] = _sha(router_output)

    # Also recompute the single validation-locked secondary union route.  This
    # does not make it deployable, but prevents an unverified pool-change row
    # from entering the final confirmatory discussion.
    union_lock=lock["selected_methods"]["cross_backend"]["union"]
    union_name=str(union_lock["track"]); union_pool=str(union_lock["pool"]); union_method=str(union_lock["method"])
    union_candidates=pd.read_parquet(run/"09_formal_test"/"cross_backend"/union_name/union_pool/"candidate_pool.parquet")
    all_backend_labels = pd.concat(
        [
            pd.read_parquet(run / "09_formal_test" / f"{backend}_candidate_labels.parquet")
            for backend in ("g1", "c1")
        ],
        ignore_index=True,
    )
    union_labels=all_backend_labels.merge(union_candidates[["sample_id","stable_candidate_id"]],on=["sample_id","stable_candidate_id"],validate="one_to_one")
    union_scores=pd.read_parquet(run/"09_formal_test"/"cross_backend"/union_name/union_pool/union_method/"scores_ensemble.parquet")
    union_selected=union_scores.sort_values(["sample_id","reranker_score","stable_candidate_id"],ascending=[True,False,True],kind="mergesort").groupby("sample_id",sort=False).head(1)[["sample_id","stable_candidate_id"]].rename(columns={"stable_candidate_id":"selected_candidate_id"})
    union_samples=union_candidates[["sample_id"]].drop_duplicates()
    g1_baseline=backend_candidates["g1"].loc[backend_candidates["g1"]["original_rank"].eq(1),["sample_id","stable_candidate_id"]].rename(columns={"stable_candidate_id":"baseline_candidate_id"})
    union_baseline=union_samples.merge(g1_baseline,on="sample_id",how="left",validate="one_to_one"); union_baseline["baseline_candidate_id"]=union_baseline["baseline_candidate_id"].fillna("")
    union_outcomes,union_metric=_evaluate(universe,union_candidates,union_labels,union_baseline,union_selected,rank_column="pool_rank")
    aligned_union_scores=union_scores[["sample_id","stable_candidate_id","reranker_score"]].rename(columns={"reranker_score":"score"})
    union_metric.update(_rank_metrics(union_candidates,union_labels,aligned_union_scores,len(universe)))
    union_key=f"g1+c1/{union_name}_{union_pool}/{union_method}"
    union_ci=_scene_ci(union_outcomes,seed=_stable_seed(20260806,f"formal/{union_key}")); union_metric["ci_lower"],union_metric["ci_upper"]=union_ci
    union_expected=pd.read_csv(run/"tables"/"cross_backend.csv").loc[lambda frame:(frame["track"].astype(str).eq(union_name))&(frame["pool"].astype(str).eq(union_pool))&(frame["method"].astype(str).eq(union_method))]
    union_stats=statistical.loc[statistical["comparison_key"].astype(str).eq(union_key)]
    if len(union_expected)!=1 or len(union_stats)!=1: raise AssertionError("independent locked union result/stat row is missing")
    union_expected=union_expected.iloc[0]; union_stats=union_stats.iloc[0]
    union_checks={
        key:math.isclose(float(union_metric[key]),float(union_expected[key]),abs_tol=1e-15)
        for key in ("j_at_1","j_at_5","oracle_at_5","oracle_at_all","delta_j_at_1")
    }
    union_checks.update({
        "recovered":union_metric["recovered"]==int(union_stats["recovered"]),
        "harmful":union_metric["harmful"]==int(union_stats["harmful"]),
        "net":union_metric["net"]==int(union_expected["net"]),
        "switch_count":union_metric["switch_count"]==int(union_expected["switch_count"]),
        "switch_rate":math.isclose(union_metric["switch_rate"],float(union_expected["switch_rate"]),abs_tol=1e-15),
        "both_wrong":union_metric["both_wrong"]==int(union_stats["both_wrong"]),
        "both_correct":union_metric["both_correct"]==int(union_stats["both_correct"]),
        "mcnemar_p":math.isclose(union_metric["mcnemar_p"],float(union_stats["pvalue"]),abs_tol=1e-15),
        "ci_lower":math.isclose(union_metric["ci_lower"],float(union_stats["bootstrap_lower"]),abs_tol=1e-15),
        "ci_upper":math.isclose(union_metric["ci_upper"],float(union_stats["bootstrap_upper"]),abs_tol=1e-15),
    })
    if not all(union_checks.values()): raise AssertionError(f"independent locked union mismatch: {union_checks}")
    union_metric.update({"backend":"G1+C1","method":union_method,"pool":f"{union_name}_{union_pool}","checks":union_checks})
    records.append(union_metric)
    union_output = output_root/"cross_backend_locked_union_outcomes.parquet"
    union_outcomes.to_parquet(union_output,index=False,compression="zstd")
    union_metric["outcome_path"] = str(union_output.resolve())
    union_metric["outcome_sha256"] = _sha(union_output)
    payload = {
        "status": "PASS",
        "implementation": "independent strict ID join; no training/evaluation module import",
        "upstream_sha256": {
            "primary_lock": _sha(run/"08_lock"/"PRIMARY_METHOD_LOCK.json"),
            "predictions_complete": _sha(run/"09_formal_test"/"PREDICTIONS_COMPLETE.json"),
            "test_access_transaction": _sha(run/"09_formal_test"/"TEST_ACCESS_TRANSACTION.json"),
            "evaluation_complete": _sha(run/"09_formal_test"/"EVALUATION_COMPLETE.json"),
            "evaluation_artifact_inventory": hashlib.sha256(json.dumps(evaluation.get("evaluation_artifacts",[]),sort_keys=True).encode()).hexdigest(),
        },
        "within_family_holm_recomputed": True,
        "all_hypothesis_contingencies_recomputed": True,
        "hypothesis_count": len(hypothesis_records),
        "hypothesis_records": hypothesis_records,
        "independent_scope": ["G1 primary ungated/gated", "Track-C router", "validation-locked secondary union"],
        "records": records,
    }
    output = output_root / "INDEPENDENT_RECOMPUTE.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Independent Recompute", "", "Status: **PASS**", "", "The evaluator uses a separate strict candidate-ID join implementation and imports no ranker training or primary evaluation module.", "", pd.DataFrame(records).drop(columns=["checks"]).to_markdown(index=False), ""]
    (output_root / "INDEPENDENT_RECOMPUTE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "path": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
