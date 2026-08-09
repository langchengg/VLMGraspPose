#!/usr/bin/env python3
"""Build evidence tables, publication figures, and bilingual final reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json  # noqa: E402
from src.grasping.g1_c1_safe_rerank.calibration import ScoreCalibrator  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import FEATURE_FAMILIES  # noqa: E402


COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#8C8C8C"]
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _save(fig: plt.Figure, root: Path, name: str, caption: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(root / f"{name}.{suffix}")
    plt.close(fig)
    (root / f"{name}.caption.txt").write_text(caption + "\n", encoding="utf-8")


def _unavailable(root: Path, name: str, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", weight="bold")
    ax.text(0.5, 0.38, reason, ha="center", va="center", wrap=True)
    _save(fig, root, name, f"{title}. {reason}")


def _bar(frame: pd.DataFrame, *, category: str, value: str, title: str, ylabel: str) -> plt.Figure:
    local = frame.dropna(subset=[category, value]).copy()
    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    positions = np.arange(len(local))
    ax.bar(positions, local[value].astype(float), color=[COLORS[index % len(COLORS)] for index in positions])
    ax.set_xticks(positions)
    ax.set_xticklabels(local[category].astype(str), rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def _formal_failure_taxonomy(run: Path, base: Path) -> pd.DataFrame:
    lock = json.loads((run / "08_lock" / "PRIMARY_METHOD_LOCK.json").read_text(encoding="utf-8"))
    universe = pd.read_parquet(base / "manifests" / "test_samples.parquet", columns=["sample_id"])
    rows = []
    for backend in ("g1", "c1"):
        selected = lock["selected_methods"]["backend"][backend.upper()]
        primary_pool = str(selected["primary_pool"])
        method = str(selected["primary_ungated_method"])
        all_candidates = pd.read_parquet(run / "data" / f"frozen_{backend}_test_candidates.parquet")
        labels = pd.read_parquet(run / "09_formal_test" / f"{backend}_candidate_labels.parquet")
        for pool in ("top5", "allnms"):
            candidates = (
                all_candidates.loc[all_candidates["original_rank"].le(5)].copy()
                if pool == "top5"
                else all_candidates.copy()
            )
            joined = candidates.merge(labels[["sample_id", "stable_candidate_id", "candidate_correct"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")
            any_positive = joined.groupby("sample_id")["candidate_correct"].any()
            all_pool_positive = all_candidates.merge(
                labels[["sample_id", "stable_candidate_id", "candidate_correct"]],
                on=["sample_id", "stable_candidate_id"],
                validate="one_to_one",
            ).groupby("sample_id")["candidate_correct"].any()
            top5_positive = all_candidates.loc[all_candidates["original_rank"].le(5)].merge(
                labels[["sample_id", "stable_candidate_id", "candidate_correct"]],
                on=["sample_id", "stable_candidate_id"], validate="one_to_one",
            ).groupby("sample_id")["candidate_correct"].any()
            baseline = joined.loc[joined["original_rank"].eq(1)].set_index("sample_id")["candidate_correct"]
            ungated = pd.read_parquet(run / "09_formal_test" / "outcomes" / backend / pool / f"{method}_ensemble.parquet").set_index("sample_id")
            all_ids = universe["sample_id"].astype(str)
            nonempty = set(candidates["sample_id"].astype(str))
            indicators = {
                "E0_empty_prediction": ~all_ids.isin(nonempty),
                "E1_no_positive_in_pool": ~all_ids.map(any_positive).fillna(False).astype(bool),
                "E2_positive_only_beyond_rank5": all_ids.map(all_pool_positive).fillna(False).astype(bool) & ~all_ids.map(top5_positive).fillna(False).astype(bool),
                "E3_top5_positive_baseline_wrong": all_ids.map(top5_positive).fillna(False).astype(bool) & ~all_ids.map(baseline).fillna(False).astype(bool),
                "E4_baseline_top1_correct": all_ids.map(baseline).fillna(False).astype(bool),
                "E5_reranker_recovered": all_ids.map(ungated["recovered"]).fillna(False).astype(bool),
                "E6_reranker_harmful": all_ids.map(ungated["harmful"]).fillna(False).astype(bool),
                "E7_wrong_to_wrong_changed": (~all_ids.map(ungated["baseline_correct"]).fillna(False).astype(bool)) & (~all_ids.map(ungated["final_correct"]).fillna(False).astype(bool)) & all_ids.map(ungated["switch"]).fillna(False).astype(bool),
                "E8_correct_to_correct": all_ids.map(ungated["baseline_correct"]).fillna(False).astype(bool) & all_ids.map(ungated["final_correct"]).fillna(False).astype(bool),
            }
            if pool == primary_pool:
                gated = pd.read_parquet(run / "09_formal_test" / "outcomes" / backend / pool / "primary_gated.parquet").set_index("sample_id")
                indicators.update({
                    "E9_gate_abstained_recoverable": all_ids.map(ungated["recovered"]).fillna(False).astype(bool) & ~all_ids.map(gated["recovered"]).fillna(False).astype(bool),
                    "E10_gate_prevented_harmful": all_ids.map(ungated["harmful"]).fillna(False).astype(bool) & ~all_ids.map(gated["harmful"]).fillna(False).astype(bool),
                })
            for category, mask in indicators.items():
                rows.append({"backend": backend.upper(), "pool": pool, "category": category, "count": int(mask.sum()), "rate": float(mask.mean()), "denominator": len(all_ids)})
    return pd.DataFrame(rows)


def _build_tables(run: Path, base: Path) -> dict[str, pd.DataFrame]:
    table_root = run / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    formal = pd.read_csv(table_root / "formal_test_all_methods.csv")
    baseline = formal.loc[(formal["method"].eq("r0_baseline")) & formal["pool"].eq("allnms"), ["backend", "total", "j_at_1", "j_at_5", "oracle_at_5", "oracle_at_all"]].rename(columns={"total": "sample_count"})
    candidate_rows = {
        backend.upper(): len(pd.read_parquet(run / "data" / f"frozen_{backend}_test_candidates.parquet"))
        for backend in ("g1", "c1")
    }
    baseline["candidate_rows"] = baseline["backend"].map(candidate_rows)
    baseline.to_csv(table_root / "baseline_oracle.csv", index=False)
    validation = pd.read_csv(run / "07_validation" / "VALIDATION_MATRIX.csv")
    loss = validation.loc[validation["method"].isin(["r3_mlp_bce", "r4_mlp_ranknet", "r5_mlp_listwise"])]
    loss.to_csv(table_root / "loss_comparison.csv", index=False)
    encoder = pd.read_csv(run / "07_validation" / "ENCODER_COMPARISON.csv")
    encoder.to_csv(table_root / "encoder_comparison.csv", index=False)
    cumulative = pd.read_csv(run / "07_validation" / "FEATURE_ABLATION_CUMULATIVE.csv")
    leave = pd.read_csv(run / "07_validation" / "FEATURE_ABLATION_LEAVE_ONE_OUT.csv")
    feature = pd.concat([cumulative.assign(ablation="cumulative"), leave.assign(ablation="leave_one_out")], ignore_index=True)
    feature.to_csv(table_root / "feature_ablation.csv", index=False)
    gate = pd.read_csv(run / "07_validation" / "GATE_COMPARISON.csv")
    gate.to_csv(table_root / "gate_comparison.csv", index=False)
    cross = pd.read_csv(table_root / "cross_backend.csv")
    cross_statistics=pd.read_csv(table_root/"statistical_tests.csv").loc[lambda frame:frame["backend"].astype(str).eq("G1+C1")].copy()
    cross_statistics.to_csv(table_root/"cross_backend_statistics.csv",index=False)
    cross_for_decision=cross.copy()
    cross_for_decision["stat_pool"]=np.where(
        cross_for_decision["track"].astype(str).eq("cross_backend_router"),
        "top1_pair",
        cross_for_decision["track"].astype(str)+"_"+cross_for_decision["pool"].astype(str),
    )
    cross_decision=cross_for_decision.merge(
        cross_statistics[["pool","method","bootstrap_lower","bootstrap_upper","pvalue","holm_adjusted_p"]].rename(columns={"pool":"stat_pool","pvalue":"mcnemar_p"}),
        on=["stat_pool","method"],how="left",validate="one_to_one",
    )
    cross_decision["decision"]=np.where(
        (cross_decision["delta_j_at_1"]>0)&(cross_decision["bootstrap_lower"]>0)&(cross_decision["holm_adjusted_p"]<.05),
        "GO","CAUTION",
    )
    cross_decision.loc[cross_decision["delta_j_at_1"]<=0,"decision"]="NO-GO"
    cross_decision["deployment_scope"]=np.where(cross_decision["track"].astype(str).str.startswith("union_"),"SECONDARY_POOL-CHANGE_CONFIRMATORY","TRACK_C_DEPLOYABLE")
    cross_decision.to_csv(table_root/"cross_backend_decision.csv",index=False)
    taxonomy = _formal_failure_taxonomy(run, base)
    taxonomy.to_csv(table_root / "failure_taxonomy.csv", index=False)
    prediction_manifest = json.loads((run / "09_formal_test" / "PREDICTIONS_COMPLETE.json").read_text(encoding="utf-8"))
    runtime = pd.DataFrame(
        [
            *prediction_manifest["matrix"]["records"],
            *prediction_manifest.get("manual", []),
        ]
    )
    parameter_rows = []
    for backend in ("g1", "c1"):
        for pool in ("top5", "allnms"):
            for model_path in sorted((run / "05_models" / backend / pool).glob("*_seed17.json")):
                artifact = json.loads(model_path.read_text(encoding="utf-8"))
                parameter_rows.append({"backend": backend.upper(), "pool": pool, "method": model_path.stem.removesuffix("_seed17"), "parameter_count": artifact.get("parameter_count") or artifact.get("model", {}).get("parameter_count")})
    parameters = pd.DataFrame(parameter_rows)
    runtime = runtime.merge(parameters, on=["backend", "pool", "method"], how="left")
    runtime.to_csv(table_root / "runtime_complexity.csv", index=False)
    feature_runtime_rows=[]; missing_rows=[]
    for split in ("train","validation","test"):
        for backend in ("g1","c1"):
            complete=json.loads((run/"02_features"/split/backend/"COMPLETE.json").read_text())
            feature_runtime_rows.append({
                "split":split,"backend":backend.upper(),
                "samples":complete["sample_count"],"candidate_rows":complete["candidate_rows"],
                "elapsed_seconds":complete["elapsed_seconds"],
                "milliseconds_per_sample":1000.0*complete["elapsed_seconds"]/max(complete["sample_count"],1),
            })
            frame=pd.read_parquet(run/"02_features"/split/backend/"allnms_features.parquet")
            for family,columns in FEATURE_FAMILIES.items():
                present=[column for column in columns if column in frame]
                values=frame[present].apply(pd.to_numeric,errors="coerce") if present else pd.DataFrame(index=frame.index)
                missing_rows.append({
                    "split":split,"backend":backend.upper(),"feature_family":family,
                    "declared_columns":len(columns),"present_columns":len(present),
                    "missing_rate":float(values.isna().to_numpy().mean()) if present else math.nan,
                    "fallback_or_unavailable":len(present)!=len(columns),
                })
    feature_runtime=pd.DataFrame(feature_runtime_rows); feature_runtime.to_csv(table_root/"feature_extraction_runtime.csv",index=False)
    missing=pd.DataFrame(missing_rows); missing.to_csv(table_root/"missing_feature_rate.csv",index=False)
    rank_columns=[column for column in formal.columns if column.startswith("selected_")]
    selected_rank=formal[["track","backend","pool","method","seed",*rank_columns]]
    selected_rank.to_csv(table_root/"selected_score_probability_rank_changes.csv",index=False)
    return {"formal": formal, "baseline": baseline, "loss": loss, "encoder": encoder, "cumulative": cumulative, "leave": leave, "gate": gate, "cross": cross, "cross_decision":cross_decision,"taxonomy": taxonomy, "runtime": runtime, "feature_runtime":feature_runtime,"missing":missing,"selected_rank":selected_rank}


def _build_figures(run: Path, tables: dict[str, pd.DataFrame]) -> None:
    root = run / "11_figures"
    # 1 Candidate count distribution.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for index, backend in enumerate(("g1", "c1")):
        candidates = pd.read_parquet(run / "data" / f"frozen_{backend}_test_candidates.parquet", columns=["sample_id"])
        counts = candidates.groupby("sample_id").size()
        total = int(tables["baseline"].loc[tables["baseline"]["backend"].eq(backend.upper()), "sample_count"].iloc[0])
        counts = np.concatenate([counts.to_numpy(dtype=int), np.zeros(total-len(counts),dtype=int)])
        ax.hist(counts, bins=np.arange(-0.5, max(int(counts.max()), 5) + 1.5), alpha=.55, label=backend.upper(), color=COLORS[index])
    ax.set(xlabel="Frozen AllNMS candidates per Test sample (including zero)", ylabel="Samples", title="Candidate-count distribution")
    ax.legend()
    _save(fig, root, "01_candidate_count_distribution", "Distribution of frozen AllNMS candidate counts on the formal Test split.")
    # 2 Oracle curves.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for index, backend in enumerate(("g1", "c1")):
        candidates = pd.read_parquet(run / "data" / f"frozen_{backend}_test_candidates.parquet")
        labels = pd.read_parquet(run / "09_formal_test" / f"{backend}_candidate_labels.parquet")
        joined = candidates.merge(labels[["sample_id", "stable_candidate_id", "candidate_correct"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")
        total = int(tables["baseline"].loc[tables["baseline"]["backend"].eq(backend.upper()), "sample_count"].iloc[0])
        ks = np.arange(1, int(joined["original_rank"].max()) + 1)
        values = [joined.loc[joined["original_rank"].le(k)].groupby("sample_id")["candidate_correct"].any().sum() / total for k in ks]
        ax.plot(ks, values, marker="o", label=backend.upper(), color=COLORS[index])
    ax.set(xlabel="K (original frozen rank)", ylabel="Oracle@K", title="Frozen-pool oracle curve")
    ax.legend()
    _save(fig, root, "02_oracle_at_k", "Oracle@K under the frozen AllNMS candidate pools; no new candidates are added.")
    primary = pd.read_csv(run / "tables" / "formal_test_primary.csv")
    # 3 Delta + CI.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    label = primary["backend"].astype(str) + "\n" + primary["method"].astype(str)
    value = primary["delta_j_at_1"].astype(float) * 100
    # Percentile intervals need not mathematically contain their point estimate.
    # Matplotlib rejects negative asymmetric error lengths, so plot the distance
    # to each endpoint when it exists and clamp only the display length at zero.
    low = np.maximum((primary["delta_j_at_1"] - primary["ci_lower"]).astype(float) * 100, 0.0)
    high = np.maximum((primary["ci_upper"] - primary["delta_j_at_1"]).astype(float) * 100, 0.0)
    ax.errorbar(np.arange(len(primary)), value, yerr=np.vstack([low, high]), fmt="o", capsize=4, color=COLORS[0])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xticks=np.arange(len(primary)), xticklabels=label, ylabel="ΔJ@1 (percentage points)", title="Primary formal-test effect with scene-bootstrap 95% CI")
    _save(fig, root, "03_j1_delta_ci", "Primary ungated and gated formal-test J@1 changes; error bars are 10,000-draw scene-cluster bootstrap intervals.")
    # 4 Recovered/harmful.
    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    x = np.arange(len(primary)); width=.36
    ax.bar(x-width/2, primary["recovered"], width, label="Recovered", color=COLORS[2])
    ax.bar(x+width/2, primary["harmful"], width, label="Harmful", color=COLORS[1])
    ax.set(xticks=x, xticklabels=label, ylabel="Samples", title="Outcome-changing switches")
    ax.legend()
    _save(fig, root, "04_recovered_harmful", "Exact paired counts for recovered and harmful outcomes.")
    # 5 Headroom.
    _save(_bar(primary, category="method", value="headroom_recovery_at_5", title="HeadroomRecovery@5", ylabel="Fraction of Top-5 headroom"), root, "05_headroom_recovery", "Fraction of frozen original Top-5 oracle headroom recovered by each primary route.")
    ensemble_loss = tables["loss"].loc[tables["loss"]["seed"].astype(str).eq("ensemble")].copy()
    ensemble_loss["label"] = ensemble_loss["backend"].astype(str)+"/"+ensemble_loss["pool"].astype(str)+"/"+ensemble_loss["method"].astype(str)
    _save(_bar(ensemble_loss, category="label", value="j_at_1", title="Controlled MLP loss comparison", ylabel="Validation J@1"), root, "06_loss_comparison", "Validation comparison with fixed Residual MLP architecture and F0-F6 features.")
    ensemble_encoder = tables["encoder"].loc[tables["encoder"]["seed"].astype(str).eq("ensemble")].copy()
    ensemble_encoder["label"] = ensemble_encoder["backend"].astype(str)+"/"+ensemble_encoder["pool"].astype(str)+"/"+ensemble_encoder["method"].astype(str)
    _save(_bar(ensemble_encoder, category="label", value="j_at_1", title="Encoder comparison", ylabel="Validation J@1"), root, "07_encoder_comparison", "Validation encoder comparison under the locked scalar feature contract.")
    cumulative = tables["cumulative"].loc[tables["cumulative"]["seed"].astype(str).eq("ensemble")].copy()
    cumulative["label"] = cumulative["backend"].astype(str)+"/"+cumulative["pool"].astype(str)+"/"+cumulative["feature_set"].astype(str)
    _save(_bar(cumulative, category="label", value="j_at_1", title="Cumulative feature ablation", ylabel="Validation J@1"), root, "08_feature_cumulative", "Cumulative F0-F7 validation ablation using the backend-selected core encoder/loss.")
    leave = tables["leave"].loc[tables["leave"]["seed"].astype(str).eq("ensemble")].copy()
    leave["label"] = leave["backend"].astype(str)+"/"+leave["pool"].astype(str)+"/"+leave["feature_set"].astype(str)
    _save(_bar(leave, category="label", value="j_at_1", title="Leave-one-family-out ablation", ylabel="Validation J@1"), root, "09_feature_leave_one_out", "Leave-one-family-out F2-F6 validation ablation.")
    # 10/12 gate curves.
    gate = tables["gate"]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for index,row in enumerate(gate.itertuples(index=False)):
        sweep=pd.read_csv(run/"07_validation"/"gates"/str(row.backend).lower()/str(row.pool)/str(row.method)/str(row.gate_kind)/"risk_coverage_sweep.csv")
        sweep["switch_risk"]=np.where(sweep["switch_count"].astype(float)>0,sweep["harmful"].astype(float)/sweep["switch_count"].astype(float),np.nan)
        ordered=(sweep.dropna(subset=["switch_risk"]).sort_values(["switch_rate","switch_risk"],kind="mergesort").groupby("switch_rate",as_index=False).head(1))
        label_name=f"{row.backend}/{row.method}/{row.gate_kind}"
        ax.plot(ordered["switch_rate"], ordered["switch_risk"] * 100, marker=".", linewidth=.8, alpha=.7, label=label_name, color=COLORS[index % len(COLORS)])
    ax.set(xlabel="Switch rate (coverage)", ylabel="Harmful / switched (%)", title="Gate risk–coverage operating points")
    ax.legend(fontsize=5, ncol=2)
    _save(fig, root, "10_gate_risk_coverage", "Validation switch risk (harmful / switched) versus switch coverage; zero-switch points have undefined risk and are omitted.")
    # 11 Reliability diagrams from validation candidate calibration.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for index, backend in enumerate(("g1", "c1")):
        calibration = pd.read_parquet(run / "04_calibration" / backend / "validation_calibrated.parquet")
        labels = pd.read_parquet(run / "07_validation" / f"{backend}_candidate_labels.parquet")
        joined = calibration.merge(labels[["sample_id", "stable_candidate_id", "candidate_correct"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")
        joined["bin"] = pd.cut(joined["source_score_calibrated"], np.linspace(0,1,11), include_lowest=True)
        reliability = joined.groupby("bin", observed=False).agg(confidence=("source_score_calibrated","mean"), accuracy=("candidate_correct","mean"), count=("candidate_correct","size")).dropna()
        ax.plot(reliability["confidence"], reliability["accuracy"], marker="o", label=backend.upper(), color=COLORS[index])
    ax.plot([0,1],[0,1],"--",color="gray",linewidth=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed candidate consistency", title="Validation reliability diagram")
    ax.legend()
    _save(fig, root, "11_calibration_reliability", "Backend-specific validation reliability after Train OOF calibration.")
    gate_scatter=gate.assign(label=gate["backend"].astype(str)+"/"+gate["method"].astype(str)+"/"+gate["gate_kind"].astype(str))
    fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.scatter(gate_scatter["switch_rate"],gate_scatter["net"],c=[COLORS[index%len(COLORS)] for index in range(len(gate_scatter))])
    for row in gate_scatter.itertuples(index=False): ax.annotate(str(row.label),(float(row.switch_rate),float(row.net)),fontsize=5)
    ax.axhline(0,color="black",linewidth=.8); ax.set(xlabel="Switch rate",ylabel="Recovered − Harmful (samples)",title="Selected gate switch rate versus net recovery")
    _save(fig, root, "12_switch_net_gain", "Validation switch rate versus net recovery at each selected safe-LCB operating point.")
    # Post-hoc strata 13-16 using primary formal outcomes + label-free features.
    for number, column, name, xlabel in (
        (13, "query_type_name", "result_by_query_type", "Query-type indicator"),
        (14, "candidate_count_normalized", "result_by_candidate_count", "Normalized candidate count"),
        (15, "score_margin_to_top1", "result_by_baseline_margin", "Baseline score margin"),
        (16, "grasp_axis_mask_support", "result_by_mask_quality", "Predicted-mask axis support"),
    ):
        strata_rows=[]
        lock=json.loads((run/"08_lock"/"PRIMARY_METHOD_LOCK.json").read_text())
        for backend in ("g1","c1"):
            selected=lock["selected_methods"]["backend"][backend.upper()]; pool=selected["primary_pool"]
            outcome=pd.read_parquet(run/"09_formal_test"/"outcomes"/backend/pool/"primary_gated.parquet")
            feature=pd.read_parquet(run/"02_features"/"test"/backend/f"{pool}_features.parquet")
            top=feature.loc[feature["original_rank"].eq(1),["sample_id",column]]
            local=outcome.merge(top,on="sample_id",how="left")
            if number==13:
                values=pd.to_numeric(local[column],errors="coerce")
                local["stratum"]=np.where(values.isna(),"Missing",np.where(values>.5,"name","other"))
            else:
                values=pd.to_numeric(local[column],errors="coerce")
                local["stratum"]="Missing"
                valid=values.dropna()
                edges=np.unique(valid.quantile([0,.25,.5,.75,1]).to_numpy(dtype=float))
                if len(edges)>=2:
                    labels_q=[f"Q{index+1}" for index in range(len(edges)-1)]
                    local.loc[valid.index,"stratum"]=pd.cut(valid,bins=edges,labels=labels_q,include_lowest=True,duplicates="drop").astype(str)
                elif len(valid):
                    local.loc[valid.index,"stratum"]="AllSame"
                if number==14:
                    local.loc[values.isna(),"stratum"]="E0"
            summary=local.groupby("stratum",observed=False).agg(delta=("final_correct",lambda values: 0.0),baseline=("baseline_correct","mean"),final=("final_correct","mean"),n=("sample_id","size")).reset_index()
            summary["delta"]=(summary["final"]-summary["baseline"])*100; summary["backend"]=backend.upper(); strata_rows.append(summary)
        strata=pd.concat(strata_rows,ignore_index=True); strata["label"]=strata["backend"]+"/"+strata["stratum"].astype(str)
        _save(_bar(strata,category="label",value="delta",title=name.replace("_"," ").title(),ylabel="Gated ΔJ@1 (pp)"),root,f"{number:02d}_{name}",f"Post-hoc gated effect stratified by {xlabel}; this analysis was not used for model selection.")
    # 17 complementarity matrix.
    top1 = pd.read_csv(run / "07_validation" / "cross_backend" / "CROSS_BACKEND_TOP1.csv").iloc[0]
    matrix=np.array([[top1["both_wrong"],top1["g1_wrong_c1_correct"]],[top1["g1_correct_c1_wrong"],top1["g1_correct_c1_correct"]]],dtype=float)
    fig,ax=plt.subplots(figsize=(4,3.5)); im=ax.imshow(matrix,cmap="YlOrRd")
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{int(matrix[i,j])}",ha="center",va="center")
    ax.set(xticks=[0,1],xticklabels=["C1 wrong","C1 correct"],yticks=[0,1],yticklabels=["G1 wrong","G1 correct"],title="Validation backend complementarity")
    fig.colorbar(im,ax=ax,label="Samples")
    _save(fig,root,"17_cross_backend_complementarity","Validation Top-1 correctness complementarity matrix for G1 and C1.")
    # 18/19 complexity scatter.
    formal_seed=tables["formal"]["seed"].astype(str)
    formal_ensemble=tables["formal"].loc[
        formal_seed.eq("ensemble")
        | (formal_seed.eq("fixed") & tables["formal"]["method"].astype(str).str.startswith("r1_")),
        ["backend","pool","method","delta_j_at_1"],
    ].drop_duplicates(["backend","pool","method"])
    runtime=tables["runtime"].drop_duplicates(["backend","pool","method"]).merge(formal_ensemble,on=["backend","pool","method"],how="left",validate="one_to_one")
    for number,xcol,name,xlabel in ((18,"latency_ms_per_nonempty_sample","runtime_vs_gain","Three-seed inference latency (ms/non-empty sample)"),(19,"parameter_count","parameters_vs_gain","Parameter count")):
        local=runtime.dropna(subset=[xcol,"delta_j_at_1"])
        fig,ax=plt.subplots(figsize=(5.5,3.0)); ax.scatter(local[xcol],local["delta_j_at_1"]*100,c=COLORS[0],alpha=.75)
        for row in local.itertuples(): ax.annotate(str(row.method),(getattr(row,xcol),row.delta_j_at_1*100),fontsize=5)
        ax.set(xlabel=xlabel,ylabel="Formal ΔJ@1 (pp)",title=name.replace("_"," ").title())
        _save(fig,root,f"{number:02d}_{name}",f"Observed complexity versus formal-test J@1 gain for locked methods.")
    # 20 feature-family importance from cumulative/leave-one-out validation deltas.
    ensemble_cum=cumulative.copy(); ensemble_leave=leave.copy()
    reference=ensemble_cum.loc[ensemble_cum["feature_set"].astype(str).eq("F0-F7"),["backend","pool","j_at_1"]].rename(columns={"j_at_1":"full_j"})
    importance=ensemble_leave.merge(reference,on=["backend","pool"],how="left")
    importance["importance"]=importance["full_j"]-importance["j_at_1"]
    importance["label"]=importance["backend"].astype(str)+"/"+importance["pool"].astype(str)+"/"+importance["feature_set"].astype(str)
    _save(_bar(importance,category="label",value="importance",title="Feature-family leave-out importance",ylabel="Validation J@1 decrease"),root,"20_feature_importance","Validation leave-one-family-out importance; positive values indicate degradation when the family is removed.")


def _decision_rows(primary: pd.DataFrame) -> str:
    columns=["backend","method","pool","j_at_1","delta_j_at_1","recovered","harmful","net","ci_lower","ci_upper","mcnemar_p","holm_adjusted_p","decision"]
    return primary[columns].to_markdown(index=False)


def _build_reports(run: Path, base: Path, tables: dict[str, pd.DataFrame]) -> None:
    reports=run/"13_reports"; reports.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(run/"tables"/"formal_test_primary.csv")
    baseline=tables["baseline"]
    loss=tables["loss"].loc[tables["loss"]["seed"].astype(str).eq("ensemble")]
    encoder=tables["encoder"].loc[tables["encoder"]["seed"].astype(str).eq("ensemble")]
    feature=pd.read_csv(run/"tables"/"feature_ablation.csv"); gate=tables["gate"]; cross=tables["cross"]; cross_decision=tables["cross_decision"]; taxonomy=tables["taxonomy"]
    best_loss=(loss.sort_values(["backend","pool","j_at_1","harmful"],ascending=[True,True,False,True],kind="mergesort").groupby(["backend","pool"],sort=False).head(1))
    best_encoder=(encoder.sort_values(["backend","pool","j_at_1","harmful"],ascending=[True,True,False,True],kind="mergesort").groupby(["backend","pool"],sort=False).head(1))
    manual=tables["formal"].loc[
        tables["formal"]["method"].astype(str).str.startswith("r1_"),
        ["backend","pool","method","j_at_1","delta_j_at_1","recovered","harmful","net","switch_rate"],
    ]
    candidate_mean={backend.upper():float(pd.read_parquet(run/"data"/f"frozen_{backend}_test_candidates.parquet",columns=["sample_id"]).groupby("sample_id").size().mean()) for backend in ("g1","c1")}
    complement=pd.read_csv(run/"07_validation"/"cross_backend"/"CROSS_BACKEND_TOP1.csv")
    rq=f"""## 逐项研究问题回答

### RQ1–RQ3：冻结候选可恢复空间

{baseline.to_markdown(index=False)}

`Oracle@5 − baseline J@1` 是不改变 Top-5 候选集合时的最大可恢复比例；`Oracle@All − baseline J@1` 是完整 AllNMS 排序 ceiling。它们只描述离线 rectangle-consistency ceiling。

### RQ4：feature family

R1 可解释单项 evidence 与完整 utility 的正式 Test 结果（alpha 仅由 Train 选择）如下：

{manual.to_markdown(index=False)}

{feature.loc[feature['seed'].astype(str).eq('ensemble')].to_markdown(index=False)}

稳定信号以 cumulative 增量和 leave-one-out 降幅共同判断；只在单一 backend/pool 出现的增益报告为 backend-specific，不外推为普适规律。

### RQ5：BCE、RankNet、Listwise

每个 backend/pool 的 Validation 胜者如下：

{best_loss.to_markdown(index=False)}

### RQ6–RQ7：encoder 与候选规模

{best_encoder.to_markdown(index=False)}

正式 Test 非空样本的平均 AllNMS 候选数为 G1={candidate_mean['G1']:.3f}、C1={candidate_mean['C1']:.3f}；若 set-aware 模型未超过 MLP，这个小集合规模是与结果一致的解释，但不是因果证明。

### RQ8–RQ9：Conservative gate

{primary.loc[primary['method'].isin(['primary_gated'])].to_markdown(index=False)}

{gate.to_markdown(index=False)}

Recovered/Harmful 的 ungated→gated 差值分别量化保留下来的可恢复样本与阻止的 harmful；G1/C1 的锁定 λ、switch rate 和 safe-LCB 显示是否需要不同保守度。四类 logistic gate 使用未加权 OOF 经验 posterior；gradient-boosted gate 的输出按决策分数解释，二者均由独立 Validation operating-point sweep 约束，而不宣称额外概率校准。

### RQ10：错误互补

{complement.to_markdown(index=False)}

### RQ11：same-pool cross evidence

F6→F7 cumulative 差值在不改变候选身份时量化跨后端 evidence 的排序价值：

{feature.loc[(feature['ablation'].eq('cumulative')) & feature['feature_set'].astype(str).isin(['F0-F6','F0-F7']) & feature['seed'].astype(str).eq('ensemble')].to_markdown(index=False)}

### RQ12–RQ13：router 与 union

{cross.to_markdown(index=False)}

Router 相对 always-G1；union 结果必须与相应 union oracle 一起解释，不能把 candidate-pool ceiling 增长误称为纯排序增益。

### RQ14：GraRe-4D-lite、latent、crop

R11 GraRe-4D-lite-inspired 与 R13 crop CNN 的正式结果保存在 `formal_test_all_methods.csv`。R12 因锁定接口没有稳定的 backend/HiFi latent hook 而记录为不可用；未用伪造 embedding 替代。

### RQ15：部署选择

{_decision_rows(primary)}

仅 `decision=GO` 的 gated route 可替换相应 baseline；CAUTION/NO-GO 均继续使用原始排序。Cross-backend Track C 独立判定，不取代 core primary 的预声明规则。

### RQ16：bottleneck

{taxonomy.to_markdown(index=False)}

E0/E1/E2 指向 candidate-generation ceiling，E3 指向 ranking headroom，E9/E10 指向 gate conservatism。Grounding 与 label/evaluator mismatch 只能由后验素材提示，本实验不能把它们识别为互斥因果来源。
"""
    rq_en=f"""## Answers to RQ1–RQ16

RQ1–RQ3 are answered by the frozen baseline/oracle table below. Oracle@5 minus baseline J@1 is the order-only Top-5 headroom; Oracle@All minus baseline J@1 is the AllNMS ranking ceiling.

{baseline.to_markdown(index=False)}

RQ4 is answered jointly by the locked Train-selected R1 interpretable screens and cumulative/leave-one-family-out validation ablations; a family is treated as stable only when these signals agree rather than from a single favorable row.

{manual.to_markdown(index=False)}

RQ5 and RQ6 are answered by the controlled winners below.

{best_loss.to_markdown(index=False)}

{best_encoder.to_markdown(index=False)}

For RQ7, the mean non-empty AllNMS set sizes were G1={candidate_mean['G1']:.3f} and C1={candidate_mean['C1']:.3f}; small sets are a result-consistent, non-causal explanation when set encoders fail to beat the MLP. RQ8–RQ9 are answered by the locked gate rows and operating points: changes in Recovered/Harmful quantify missed recoveries and prevented harms, while backend-specific lambda and switch rate quantify conservatism.

{gate.to_markdown(index=False)}

RQ10 is answered by the correctness complementarity matrix. RQ11 is the same-pool F6-to-F7 delta. RQ12–RQ13 are the router and union rows; union gains are interpreted jointly with their higher candidate-pool oracle, not as pure ranking improvements.

{complement.to_markdown(index=False)}

{cross.to_markdown(index=False)}

For RQ14, R11 GraRe-4D-lite-inspired and R13 crop CNN have measured rows; R12 is unavailable because the locked inference interface has no stable pre-freeze backend/HiFi latent hook, and no synthetic latent was substituted. RQ15 follows the formal GO/CAUTION/NO-GO table: only a gated GO route may replace its baseline. RQ16 follows E0/E1/E2 (candidate-generation ceiling), E3 (ranking headroom), and E9/E10 (gate conservatism); grounding and evaluator mismatch remain descriptive inferences rather than identified mutually exclusive causes.
"""
    result_scope="All reported success values are offline OCID-VLG target-specific 2D 4-DoF grasp-rectangle consistency outcomes, not physical robot grasp success. Depth-based clearance values are single-view 2.5D proxies, not full 3D collision checks."
    zh=f"""# G1/C1 冻结候选重排序最终总结

{result_scope}

## 基线与 ceiling

{baseline.to_markdown(index=False)}

## 正式测试 primary

{_decision_rows(primary)}

## Loss 结论

同一 Residual MLP、F0-F6 和候选池下的 Validation 结果如下；选择只使用 Validation：

{loss.to_markdown(index=False)}

## Encoder 结论

{encoder.to_markdown(index=False)}

## Feature 结论

{feature.loc[feature['seed'].astype(str).eq('ensemble')].to_markdown(index=False)}

## Gate 结论

Gate 只使用 scene-grouped OOF Train ranker 预测训练，阈值仅在 Validation 用 10,000 次 scene bootstrap safe-LCB 选择；“永不切换”是合法点。

{gate.to_markdown(index=False)}

## 跨后端结论

{cross.to_markdown(index=False)}

{cross_decision.to_markdown(index=False)}

## Bottleneck

{taxonomy.to_markdown(index=False)}

## Runtime, memory, and missingness

{tables['runtime'].to_markdown(index=False)}

{tables['feature_runtime'].to_markdown(index=False)}

{tables['missing'].to_markdown(index=False)}

E0/E1/E2 量化候选生成 ceiling；E3 量化 Top-5 内可由排序修复的错误；E9/E10 量化 gate 的保守性。本实验没有足够证据把剩余误差全部归因于 grounding 或物理可执行性。

{rq}
"""
    (reports/"FINAL_SUMMARY_ZH.md").write_text(zh,encoding="utf-8")
    methods=f"""# Methods: Frozen G1/C1 Candidate Reranking

{result_scope}

We froze the canonical G1 and C1 post-NMS candidate identities, separately evaluated Top5 and AllNMS order-only pools, and trained with official Train, five scene-grouped folds, official Validation selection, and a one-time locked Test evaluation. Candidate features were inference-available only; identifiers and scene IDs were restricted to joins and grouping. Pointwise BCE used equal total query mass after class weighting; RankNet used normalized positive-negative pairs; the multi-positive listwise objective was reported as a softmax-mass loss rather than ListNet.

The implementation used the mathematical structures of [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html), [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/), and [LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/). LightGBM's official `lambdarank` objective was used. No external implementation code was copied.
"""
    (reports/"METHODS_RERANKING_EN.md").write_text(methods,encoding="utf-8")
    results=f"""# Results: Frozen G1/C1 Candidate Reranking

{result_scope}

## Verified baselines and candidate ceilings

{baseline.to_markdown(index=False)}

## Primary formal Test results

{_decision_rows(primary)}

## Controlled loss comparison

{loss.to_markdown(index=False)}

## Controlled encoder comparison

{encoder.to_markdown(index=False)}

## Cross-backend results

{cross.to_markdown(index=False)}

{cross_decision.to_markdown(index=False)}

{rq_en}
"""
    (reports/"RESULTS_RERANKING_EN.md").write_text(results,encoding="utf-8")
    discussion=f"""# Discussion

Results are interpreted only within the frozen-candidate offline protocol. A positive reranking delta isolates ranking value; a union-pool delta mixes candidate complementarity with ranking and is therefore reported separately. A model is recommended for deployment only when the gated formal delta is positive, its scene-bootstrap lower bound is above zero, Holm-adjusted McNemar is below 0.05, and independent recomputation passes.

## Failure taxonomy

{taxonomy.to_markdown(index=False)}

The relative sizes of no-positive-pool and solvable-but-misranked strata diagnose candidate-generation versus ranking headroom. Grounding remains an inference when based on post-hoc mask agreement and is not identified as a causal failure source by this experiment.
"""
    (reports/"DISCUSSION_RERANKING_EN.md").write_text(discussion,encoding="utf-8")
    limitations=f"""# Limitations

- {result_scope}
- R12 frozen backend/HiFi latent extraction was unavailable because the locked inference interface did not expose a stable pre-freeze spatial latent hook; this negative capability result is not replaced with synthetic features.
- R13 evaluates a 74k-parameter candidate-aligned four-channel crop CNN without geometry refinement.
- Validation was used for the preregistered model, feature, and gate operating-point choices; Test was opened once after artifact locking.
- Post-hoc query/mask strata are descriptive and were not used as model inputs or selection criteria.
- Offline evaluation cannot establish force closure, reachability, or real-robot grasp success.
"""
    (reports/"LIMITATIONS_EN.md").write_text(limitations,encoding="utf-8")
    (reports/"FINAL_REPORT_EN.md").write_text("\n\n".join([methods,results,discussion,limitations]),encoding="utf-8")
    latex=baseline.to_latex(index=False,float_format=lambda value:f"{value:.6f}")+"\n\n"+primary.to_latex(index=False,float_format=lambda value:f"{value:.6f}")
    (reports/"THESIS_READY_TABLES.tex").write_text(latex,encoding="utf-8")
    figure_lines=["# Thesis-ready figures","",*[f"- `{path.name}` — {(path.with_suffix('.caption.txt')).read_text().strip() if path.with_suffix('.caption.txt').is_file() else ''}" for path in sorted((run/"11_figures").glob("*.pdf"))]]
    (reports/"THESIS_READY_FIGURES.md").write_text("\n".join(figure_lines)+"\n",encoding="utf-8")
    decisions={row.backend:{"method":row.method,"decision":row.decision,"delta_j_at_1":row.delta_j_at_1,"ci":[row.ci_lower,row.ci_upper],"holm_adjusted_p":row.holm_adjusted_p} for row in primary.loc[primary["method"].eq("primary_gated")].itertuples()}
    deployable_rows=cross_decision.loc[(cross_decision["deployment_scope"].eq("TRACK_C_DEPLOYABLE"))&(cross_decision["method"].astype(str).eq("backend_top1_router"))]
    if len(deployable_rows)!=1: raise RuntimeError("final report requires exactly one Track-C router decision")
    deployable_cross=deployable_rows.iloc[0]
    cross_conclusion={"method":deployable_cross["method"],"decision":deployable_cross["decision"],"delta_j_at_1":deployable_cross["delta_j_at_1"],"ci":[deployable_cross["bootstrap_lower"],deployable_cross["bootstrap_upper"]],"holm_adjusted_p":deployable_cross["holm_adjusted_p"],"fallback":"always_G1"}
    atomic_json(reports/"EXPERIMENT_CONCLUSION.json",{"scope":result_scope,"primary":decisions,"cross_backend":cross_conclusion,"r12_status":"UNAVAILABLE_WITH_EVIDENCE","r13_status":"COMPLETE","test_reselection":False})
    passport={"run":str(run),"source_base_run":str(base),"candidate_identity":"frozen","labels":"physically separate; Test opened once after lock","claim_boundary":result_scope,"external_code_copied":False}
    (reports/"MATERIAL_PASSPORT.md").write_text("# Material Passport\n\n```json\n"+json.dumps(passport,indent=2)+"\n```\n",encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args=_parse(argv); base=args.base_run.expanduser().resolve(); run=args.run_dir.expanduser().resolve()
    if (run / "COMPLETE").exists(): raise PermissionError("refusing to mutate sealed reports")
    tables=_build_tables(run,base); _build_figures(run,tables); _build_reports(run,base,tables)
    print(json.dumps({"status":"COMPLETE","reports":str(run/"13_reports"),"figures":str(run/"11_figures")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
