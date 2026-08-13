#!/usr/bin/env python3
"""Build the read-only post-formal reranking success/failure analysis run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.case_analysis import (  # noqa: E402
    EXPECTED_FINAL_LOCK_SHA256,
    ROUTES,
    artifact_record,
    build_decision_chain,
    canonical_sha256,
    critical_source_snapshot,
    deterministic_case_selection,
    enrich_samples,
    global_tables,
    pair_contribution_analysis,
    reconcile_formal,
    replay_contributions,
    sha256_file,
    source_index,
)
from unified_reranking.case_visuals import (  # noqa: E402
    build_figures,
    render_case_board,
    write_gallery_html,
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain=v1"),
    }


def _environment() -> str:
    packages = "\n".join(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in sorted(
            importlib.metadata.distributions(),
            key=lambda value: str(value.metadata.get("Name", "")).lower(),
        )
        if distribution.metadata.get("Name")
    )
    return (
        f"python={sys.version}\nexecutable={sys.executable}\nplatform={platform.platform()}\n"
        f"source_git={json.dumps(_git_state(), ensure_ascii=False)}\n\n{packages}"
    )


def _setup(derived: Path) -> None:
    for name in (
        "00_audit",
        "01_source_index",
        "02_joined_tables",
        "03_global_mechanism_analysis",
        "04_feature_contributions",
        "05_case_selection",
        "06_case_boards/crog",
        "06_case_boards/g1",
        "06_case_boards/c1",
        "06_case_boards/cross_route_appendix",
        "07_galleries",
        "08_figures",
        "09_reports",
        "10_slides",
        "11_quality_audit",
        "logs",
        "configs",
        "tables",
    ):
        (derived / name).mkdir(parents=True, exist_ok=True)


def _expected_source_mismatch(path: Path, error: Exception, source: Path) -> None:
    path.write_text(
        "# SOURCE RESULT MISMATCH\n\n"
        "The analysis stopped before interpretation because recomputed formal quantities did not match.\n\n"
        f"- source: `{source}`\n- final lock: `{sha256_file(source / 'FINAL_RUN_LOCK.json')}`\n"
        f"- error: `{type(error).__name__}: {error}`\n"
        "- joins: `(route, sample_id, candidate_id)` with the exact 7,675-sample denominator\n\n"
        "No evaluator, model, threshold, candidate, or sample filter was changed.\n"
    )


def _feature_schema(source: Path) -> pd.DataFrame:
    selected = json.loads((source / "08_lock/selected_features.json").read_text())
    rows = []
    for route in ROUTES:
        for order, feature in enumerate(selected["routes"][route]["feature_columns"]):
            from unified_reranking.ablation import feature_family

            rows.append(
                {
                    "route": route,
                    "feature_order": order,
                    "feature": feature,
                    "family": feature_family(feature),
                }
            )
    return pd.DataFrame(rows)


def _load_visual_ground_truth(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_hashes = json.loads((source / "00_audit/source_run_hashes.json").read_text())
    record = source_hashes["paired_manifest"]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError("transitive visual GT manifest hash mismatch")
    frame = pd.read_parquet(
        path,
        columns=[
            "sample_id",
            "gt_mask_path",
            "gt_grasp_list_json",
            "object_category",
            "scene_family",
        ],
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    if len(frame) != 7675 or frame["sample_id"].nunique() != 7675:
        raise RuntimeError("visual GT manifest does not match the Test denominator")
    return frame, {"path": str(path.resolve()), "sha256": record["sha256"]}


def _case_boards(
    derived: Path,
    selection: pd.DataFrame,
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    gt: pd.DataFrame,
) -> pd.DataFrame:
    sample_lookup = samples.set_index(["route", "sample_id"])
    gt_lookup = gt.set_index("sample_id")
    records = []
    for selected in selection.itertuples(index=False):
        row = sample_lookup.loc[(selected.route, selected.sample_id)].to_dict()
        row.update({"route": selected.route, "sample_id": selected.sample_id})
        grasp_json = gt_lookup.loc[selected.sample_id, "gt_grasp_list_json"]
        rectangles = (
            json.loads(grasp_json) if isinstance(grasp_json, str) and grasp_json else []
        )
        candidate_rows = candidates[
            (candidates["route"] == selected.route)
            & (candidates["sample_id"] == selected.sample_id)
        ]
        filename = f"{selected.category}_{selected.selection_order:02d}_{selected.sample_id}.png"
        path = derived / "06_case_boards" / selected.route / filename
        render_case_board(row, candidate_rows, path, gt_rectangles=rectangles)
        records.append(
            {
                **selected._asdict(),
                "board_path": str(path.resolve()),
                "board_relative_path": os.path.relpath(path, derived / "07_galleries"),
                "board_sha256": sha256_file(path),
                "width_px": 2400,
                "height_px": 1350,
            }
        )
    return pd.DataFrame(records)


def _core_selection(gallery: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    enriched = gallery.merge(
        samples[["route", "sample_id", "mechanism", "dominant_family"]],
        on=["route", "sample_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_sample"),
    )
    rows = []
    for route in ROUTES:
        route_frame = enriched[enriched["route"] == route]
        recovered = route_frame[route_frame["category"] == "recovered"].sort_values(
            "selection_order"
        )
        recovered = (
            pd.concat(
                [recovered.drop_duplicates("mechanism").head(3), recovered.head(3)]
            )
            .drop_duplicates("sample_id")
            .head(3)
        )
        chosen = [recovered]
        for category, count in (
            ("harmful", 2),
            ("gate_prevented_harmful", 1),
            ("gate_missed_recoverable", 1),
            ("no_positive_pool", 1),
        ):
            chosen.append(
                route_frame[route_frame["category"] == category]
                .sort_values("selection_order")
                .head(count)
            )
        picked = pd.concat(chosen).drop_duplicates("sample_id")
        picked = (
            pd.concat(
                [picked, route_frame.sort_values(["category", "selection_order"])]
            )
            .drop_duplicates("sample_id")
            .head(8)
        )
        for order, record in enumerate(picked.to_dict("records"), start=1):
            record["core_order"] = order
            rows.append(record)
    return pd.DataFrame(rows)


def _manual_audit_manifest(gallery: pd.DataFrame) -> pd.DataFrame:
    work = gallery.copy()
    work["audit_sha256"] = [
        canonical_sha256([row.route, row.category, row.sample_id, "visual-audit-30"])
        for row in work.itertuples(index=False)
    ]
    chosen = []
    for (route, category), group in work.groupby(["route", "category"], sort=True):
        chosen.append(group.sort_values("audit_sha256").head(2))
    result = pd.concat(chosen).sort_values("audit_sha256").head(30).copy()
    result["manual_status"] = "PENDING_VISUAL_INSPECTION"
    result["overlay_legible"] = False
    result["text_outside_image"] = False
    result["decision_chain_consistent"] = False
    result["notes"] = ""
    return result


def _write_reports(
    derived: Path,
    source: Path,
    performance: pd.DataFrame,
    samples: pd.DataFrame,
    pairs: pd.DataFrame,
    replay: pd.DataFrame,
    gallery: pd.DataFrame,
) -> None:
    perf = performance.set_index("route")
    mechanism = (
        pairs[pairs["native_candidate_id"] != pairs["challenger_candidate_id"]]
        .groupby(["route", "mechanism"])
        .size()
        .rename("n")
        .reset_index()
    )
    dominant = {
        route: mechanism[mechanism["route"] == route]
        .sort_values("n", ascending=False)
        .head(3)
        .to_dict("records")
        for route in ROUTES
    }
    en = f"""# Reranking success/failure mechanism analysis

## Scope and evidence boundary

This is a strictly read-only post-formal analysis of `{source.name}` at final-lock SHA-256
`{EXPECTED_FINAL_LOCK_SHA256}`. It did not train, select, calibrate, gate, route, or rerun the
formal Test. Formal outcomes, Validation-only ablations, frozen-model contribution
decompositions, and human visual interpretation are reported as distinct evidence layers.
The evaluator is offline 2D Jacquard consistency; it is not a physical robot grasp trial.

## Exact formal reconciliation

| Route | Native J@1 | Ungated J@1 | Gated J@1 | Oracle@5 | Recovered | Harmful | Net | Irreparable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for route in ROUTES:
        row = perf.loc[route]
        en += f"| {route.upper()} | {row.native_j_at_1:.6f} | {row.ungated_j_at_1:.6f} | {row.gated_j_at_1:.6f} | {row.oracle_at_5:.6f} | {int(row.recovered)} | {int(row.harmful)} | {int(row.net)} | {int(row.irreparable)} |\n"
    en += """

## Why native Top-1 was selected

The native decision is the frozen candidate with `native_rank=1`. For G1/C1 this rank
comes from the locked peak decoder (quality threshold 0.2, minimum peak distance 20,
maximum 100 peaks, Gaussian-smoothed angle/width fields and stable quality/row/column
ordering). CROG uses its frozen candidate order. Calibration and the learned selector do
not create this native order.

## Why the learned order changes

The primary selector is **T2 matched-common evidence + three-seed LightGBM LambdaMART
/ lambdarank**. Each model receives the exact 105-column locked, fold-standardized
representation. The seed raw ranking scores are averaged, then candidates are sorted by
score descending, native rank ascending, and candidate ID ascending. The score is a
ranking margin, not a success probability. Averaged native LightGBM contributions add
back to the saved ensemble score and are used only as a post-hoc decomposition.

Top contribution-associated mechanisms by route are:
"""
    for route in ROUTES:
        en += (
            f"\n- **{route.upper()}**: "
            + ", ".join(f"{x['mechanism']} ({x['n']})" for x in dominant[route])
            + "."
        )
    en += """

Recovered decisions occur when evidence associated with the promoted challenger aligns
with a formally positive candidate. Harmful switches occur when the same ranking logic
promotes a formally negative candidate. This contrast demonstrates why contribution
values are not causal guarantees.

## Conservative gate

The gate accepts a challenger only when every locked condition passes: positive recovery-
minus-harm utility, strict ranker margin and reliability thresholds, minimum stability,
at least two agreeing seeds, unchanged candidate/geometry identities, and challenger
existence. It reduces harmful switches (CROG 53 to 37, G1 51 to 29, C1 63 to 52), while also
missing 14, 14, and 12 ungated recoveries respectively.

## Route differences and remaining bottleneck

CROG begins close to its Top-5 oracle, so its absolute order-only opportunity is small;
its final 92.365% result must be named **CROG candidate pool with external matched
RGB-D/HiFi evidence reranking**, not a pure CROG-native reranker. G1 and C1 recover a
larger share of their available Top-5 headroom, but remain well below CROG because 3,206
and 3,163 Test samples respectively are E0/E1/E2 reranking-irreparable. On those samples,
no order-only selector can succeed because the required positive candidate is absent.

## Reproducibility and limitations

All nine frozen seed scores replay exactly. Maximum contribution additivity residuals are
listed in `04_feature_contributions/replay_audit.csv`. The gallery contains
{len(gallery)} deterministically selected boards. The matched HiFi probability map shown
in boards is external T2 evidence, not a CROG-native dense probability map. Contribution
and ablation agreement supports an association claim only; visual inspection is a human
interpretation layer. Union/router results are cross-route appendices and are excluded
from the route-wise order-only mechanism narrative.
"""
    zh = f"""# 重排序成功/失败机制分析

## 范围与证据边界

本报告是对源 run `{source.name}` 的严格只读 post-formal 分析。
源最终锁 SHA-256：`{EXPECTED_FINAL_LOCK_SHA256}`。
没有重新训练、选型、校准、调 gate、路由或再次执行 formal Test。正式结果、仅 Validation 消融、冻结模型贡献分解和人工视觉解释被分层陈述。评价指标是离线二维 Jacquard 一致性，不等同于真实机器人抓取成功。

## 正式结果对账

三条路线均为 7,675 个 Test 样本。CROG 从 {perf.loc["crog"].native_j_at_1:.6f} 提升到 {perf.loc["crog"].gated_j_at_1:.6f}；G1 从 {perf.loc["g1"].native_j_at_1:.6f} 提升到 {perf.loc["g1"].gated_j_at_1:.6f}；C1 从 {perf.loc["c1"].native_j_at_1:.6f} 提升到 {perf.loc["c1"].gated_j_at_1:.6f}。恢复/伤害分别为 CROG 278/37、G1 729/29、C1 1006/52，均与正式 artifact 精确一致。

## 原始 Top-1 与学习式 Top-1

原始 Top-1 是冻结候选池中 `native_rank=1` 的候选；学习式排序不改变候选、几何或评价器，而是在 105 个锁定 T2 matched-common 特征经训练期 fold 预处理后，使用三个 LightGBM LambdaMART 模型的原始 rank score 均值重新排序。该 score 不是成功概率。贡献值严格加和回放到冻结分数，但只属于 post-hoc 模型归因，不能证明因果。

## gate 与剩余瓶颈

conservative gate 只有在 utility、margin、reliability、stability、seed votes、候选与几何不变、challenger 存在等所有锁定条件同时满足时才切换。它减少 harmful switch，但也会错过部分可恢复样本。G1/C1 虽然回收了大部分 Top-5 排序空间，最终仍明显低于 CROG，核心原因是各有 3,206/3,163 个 E0+E1+E2 样本在候选池中根本没有可用正候选；order-only reranker 无法修复。

CROG 的最终结果应表述为“CROG candidate pool with external matched RGB-D/HiFi evidence reranking”，不能写成纯 CROG-native reranker。图库中的 probability map 是 matched HiFi/T2 证据，不是 CROG 原生 dense probability map。

## 产物

本分析包含 {len(gallery)} 张确定性案例板、16 组论文级矢量图、模型贡献/机制/gate/IoU-angle 表、可浏览 HTML 图库、中英文报告和演示材料。Union/router 仅作为 cross-route appendix，不混入单路线 order-only 机制叙述。
"""
    report_dir = derived / "09_reports"
    (report_dir / "ANALYSIS_REPORT_EN.md").write_text(en)
    (report_dir / "ANALYSIS_REPORT_ZH.md").write_text(zh)
    (report_dir / "LIMITATIONS.md").write_text(
        "# Limitations\n\n- Offline 2D Jacquard consistency is not physical grasp success.\n"
        "- Native LightGBM contributions are additive model-score decompositions, not causal effects.\n"
        "- Validation ablations are supportive and were not selected on Test.\n"
        "- Human case-board readings are interpretive.\n"
        "- The matched HiFi probability map is external T2 evidence, not a CROG-native probability map.\n"
        "- E0/E1/E2 pool absence bounds what order-only reranking can repair.\n"
    )
    (report_dir / "METHOD_CARD.md").write_text(
        "# Method card\n\nPrimary: T2 matched-common evidence + LightGBM LambdaMART/lambdarank (seeds 42, 123, 2026) + locked conservative gate.\n\n"
        "Prediction replay uses the exact FoldPreprocessor and stable tie-break. LightGBM `pred_contrib=True` returns one contribution per feature plus an expected-value column.\n\n"
        "Official API reference: https://lightgbm.readthedocs.io/en/v4.7.0/pythonapi/lightgbm.Booster.html\n"
    )
    (report_dir / "THESIS_SECTION.tex").write_text(
        r"""\section{Post-formal reranking mechanism analysis}
We analysed the locked route-wise decision chain without retraining or re-evaluating the formal test. The primary selector combined matched-common RGB--D/HiFi evidence with a three-seed LightGBM LambdaMART model and a conservative transition gate. Native LightGBM contribution vectors were averaged across seeds and verified to sum to the frozen ensemble rank score. These values are interpreted as post-hoc score decompositions rather than causal effects. The largest remaining limitation for G1 and C1 was candidate availability: 3,206 and 3,163 test samples, respectively, were not repairable by order-only reranking. The evaluator measures offline two-dimensional Jacquard consistency and does not constitute a physical robot grasp trial.
"""
    )


def run(source: Path, output: Path | None = None) -> Path:
    source = source.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    derived = (
        output or ROOT / "runs" / f"fair_unified_reranking_case_analysis_{timestamp}"
    ).resolve()
    if derived.exists():
        raise FileExistsError(f"derived analysis run already exists: {derived}")
    _setup(derived)
    command = " ".join(map(str, sys.argv))
    (derived / "commands.log").write_text(
        datetime.now(timezone.utc).isoformat() + "\t" + command + "\n"
    )
    (derived / "environment.txt").write_text(_environment())
    before = critical_source_snapshot(source)
    _json(derived / "00_audit/source_snapshot_before.json", before)
    _json(derived / "00_audit/analysis_git_state.json", _git_state())
    _csv(derived / "01_source_index/source_artifacts.csv", source_index(source))
    _json(
        derived / "01_source_index/lineage.json",
        {
            "source_run": str(source),
            "final_lock": artifact_record(source / "FINAL_RUN_LOCK.json"),
            "status_note": "pipeline_status.json absent; FINAL_RUN_LOCK + manifest + COMPLETE are authoritative; _STATUS.json is stale",
            "independent_recompute": artifact_record(
                source / "15_independent_recompute/INDEPENDENT_RECOMPUTE.json"
            ),
        },
    )
    try:
        samples, candidates, gates = build_decision_chain(source)
        performance = reconcile_formal(source, samples)
    except Exception as error:
        _expected_source_mismatch(derived / "SOURCE_RESULT_MISMATCH.md", error, source)
        raise
    _parquet(derived / "02_joined_tables/per_sample_decision_chain.parquet", samples)
    _parquet(
        derived / "02_joined_tables/per_candidate_decision_chain.parquet", candidates
    )
    _parquet(derived / "02_joined_tables/gate_chain.parquet", gates)
    _csv(derived / "tables/formal_reconciliation.csv", performance)
    schema = _feature_schema(source)
    _csv(derived / "01_source_index/selected_feature_schema.csv", schema)

    contributions, importance, replay = replay_contributions(source, candidates)
    _parquet(
        derived / "04_feature_contributions/per_candidate_contributions.parquet",
        contributions,
    )
    _csv(derived / "04_feature_contributions/gain_split_importance.csv", importance)
    _csv(derived / "04_feature_contributions/replay_audit.csv", replay)
    pairs, contribution_long = pair_contribution_analysis(samples, contributions)
    samples = enrich_samples(samples, candidates, pairs)
    _parquet(
        derived / "02_joined_tables/per_sample_decision_chain_enriched.parquet", samples
    )
    _parquet(
        derived / "04_feature_contributions/pairwise_family_contributions.parquet",
        pairs,
    )
    _parquet(
        derived
        / "04_feature_contributions/pairwise_feature_contributions_long.parquet",
        contribution_long,
    )
    for name, table in global_tables(samples, candidates, pairs).items():
        _csv(derived / f"03_global_mechanism_analysis/{name}.csv", table)

    ablation = pd.read_csv(
        source / "07_validation/ablations/leave_one_family_out_ablation.csv"
    )
    _csv(
        derived / "03_global_mechanism_analysis/validation_leave_one_family_out.csv",
        ablation,
    )
    figures = build_figures(
        derived / "08_figures",
        performance,
        samples,
        candidates,
        pairs,
        importance,
        ablation,
    )

    selection = deterministic_case_selection(samples)
    _csv(derived / "05_case_selection/gallery_selection.csv", selection)
    gt, gt_record = _load_visual_ground_truth(source)
    _json(derived / "01_source_index/visual_ground_truth.json", gt_record)
    gallery = _case_boards(derived, selection, samples, candidates, gt)
    _csv(derived / "05_case_selection/gallery_manifest.csv", gallery)
    core = _core_selection(gallery, samples)
    _csv(derived / "05_case_selection/core_case_selection.csv", core)
    core_dir = derived / "06_case_boards" / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    for row in core.itertuples(index=False):
        destination = (
            core_dir / f"{row.route}_{row.core_order:02d}_{Path(row.board_path).name}"
        )
        shutil.copy2(row.board_path, destination)
    write_gallery_html(
        derived / "07_galleries/index.html",
        gallery,
        title="Frozen reranking success/failure cases",
    )
    audit = _manual_audit_manifest(gallery)
    _csv(derived / "11_quality_audit/manual_visual_audit_30.csv", audit)
    _write_reports(derived, source, performance, samples, pairs, replay, gallery)

    config = {
        "analysis_role": "POST_FORMAL_READ_ONLY_DERIVED",
        "source_run": str(source),
        "source_final_lock_sha256": EXPECTED_FINAL_LOCK_SHA256,
        "routes": list(ROUTES),
        "sample_count": 7675,
        "sample_route_rows": 23025,
        "model_contribution_method": "native LightGBM pred_contrib, averaged over three locked seeds",
        "contribution_interpretation": "post-hoc additive association, not causal",
        "mixed_mechanism_threshold": 0.35,
        "selection": "SHA256(route|category|sample_id), ascending",
        "figure_stems": figures,
        "gallery_board_count": len(gallery),
    }
    config["content_sha256"] = canonical_sha256(config)
    _json(derived / "configs/analysis_contract.json", config)
    (derived / "README_REPRODUCE.md").write_text(
        "# Reproduce the derived analysis\n\nThis directory is a post-formal read-only derivative. It is not a formal experiment.\n\n"
        f"Source run: `{source}`\n\nSource final lock: `{EXPECTED_FINAL_LOCK_SHA256}`\n\n"
        f"Command:\n```bash\nPYTHONPATH=src:. {sys.executable} -m tools.unified_reranking.run_case_analysis --source-run {source}\n```\n\n"
        "The command must use the frozen project Python with LightGBM 4.7.0. It verifies all nine saved scores and contribution additivity before reporting.\n"
    )
    _json(
        derived / "manifest.json",
        {
            "schema_version": 1,
            "status": "ANALYSIS_BUILT_AWAITING_MANUAL_VISUAL_AUDIT_AND_SLIDES",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source),
            "source_final_lock_sha256": EXPECTED_FINAL_LOCK_SHA256,
            "formal_test_reexecuted": False,
            "models_retrained": False,
            "selection_changed": False,
            "candidate_test_labels_used_postformal_diagnostics": True,
            "sample_count": 7675,
            "sample_route_rows": 23025,
            "gallery_board_count": len(gallery),
        },
    )
    after = critical_source_snapshot(source)
    _json(derived / "00_audit/source_snapshot_after.json", after)
    if before != after:
        raise RuntimeError(
            "source run metadata or critical hashes changed during derived analysis"
        )
    _json(
        derived / "00_audit/SOURCE_IMMUTABILITY_PASS.json",
        {
            "status": "PASS",
            "before_equals_after": True,
            "scope": "critical authority content hashes plus exact relative-path/size/mtime inventory; hidden .DS_Store excluded from scientific claim only if outside final-lock inventory",
            "source_final_lock_sha256": EXPECTED_FINAL_LOCK_SHA256,
        },
    )
    return derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        default=ROOT / "runs/fair_unified_reranking_20260809_103012",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(run(args.source_run, args.output))


if __name__ == "__main__":
    main()
