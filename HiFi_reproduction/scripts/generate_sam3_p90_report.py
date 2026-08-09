#!/usr/bin/env python3
"""Generate numerical figures, evaluation galleries, and the final research report."""

from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.selective_sam3_vg.evaluation import (  # noqa: E402
    load_frozen_ground_truth_manifest,
    load_ground_truth_mask,
)
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_binary_mask,
    load_compact_manifest,
    resize_binary_mask,
)
from src.segmentation.proposal_types import load_candidate_masks_npz  # noqa: E402


EXPERIMENT = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1"
REPORT = EXPERIMENT / "report"
FIGURES = REPORT / "figures"
GALLERIES = REPORT / "galleries"
ARTIFACT = PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1"
COLOURS = ["#0F8B8D", "#143642", "#EC9A29", "#A8201A", "#6A4C93", "#3A86FF"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str, data: dict) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300 if suffix == "png" else None)
    (FIGURES / f"{name}.data.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def _method_metrics(payload: dict) -> pd.DataFrame:
    rows = []
    for method, values in payload["metrics"].items():
        if "mean_iou" not in values:
            continue
        rows.append({"method": method, **values})
    return pd.DataFrame(rows)


def _plots(payload: dict) -> None:
    _style()
    methods = _method_metrics(payload)
    focus_names = [
        "Local HiFi-CS baseline",
        "Existing selective SAM3",
        "SAM3 full-query highest score",
        "Maximum HiFi overlap",
        "Deterministic relation-aware",
        "Learned Stage-1 selector",
        "Stage-1 + Stage-2 selector (pre-gate)",
        "Locked conservative method",
        "Stage-1 GT oracle",
        "Stage-2 GT oracle",
    ]
    focus = methods[methods["method"].isin(focus_names)].set_index("method").reindex(focus_names).dropna(how="all")
    thresholds = [50, 60, 70, 80, 90]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for index, (method, row) in enumerate(focus.iterrows()):
        ax.plot(thresholds, [row[f"p_at_{x}"] for x in thresholds], marker="o", linewidth=1.5, label=method, color=COLOURS[index % len(COLOURS)])
    ax.set(xlabel="IoU threshold (%)", ylabel="Precision at threshold", ylim=(0, 1.02))
    ax.legend(fontsize=6.5, ncol=2)
    _save(fig, "01_p50_p90_comparison", {"methods": focus.reset_index().to_dict(orient="records")})

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(np.arange(len(focus)), focus["mean_iou"], color=[COLOURS[i % len(COLOURS)] for i in range(len(focus))])
    ax.set_yticks(np.arange(len(focus)), labels=focus.index, fontsize=7)
    ax.set(xlabel="Mean IoU", xlim=(0, 1.0))
    _save(fig, "02_miou_comparison", {"methods": focus["mean_iou"].to_dict()})

    stage1 = pd.read_csv(EXPERIMENT / "oracle_stage1_test/source_contributions.csv")
    stage2_summary = json.loads((EXPERIMENT / "oracle_stage2_test/summary.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(stage1["stage_index"], stage1["oracle_p90"], marker="o", label="Stage 1 cumulative")
    ax.scatter([len(stage1)], [stage2_summary["stage2_oracle"]["p_at_90"]], s=60, label="Stage 2 oracle", color=COLOURS[2])
    ax.set_xticks(list(stage1["stage_index"]) + [len(stage1)], labels=list(stage1["source_added"]) + ["Stage 2"], rotation=35, ha="right")
    ax.set(ylabel="Oracle P@90", ylim=(0, 1.0))
    ax.legend()
    _save(fig, "03_stage1_stage2_oracle_curves", {"stage1": stage1.to_dict(orient="records"), "stage2": stage2_summary})

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(stage1["mean_candidates_per_sample"], stage1["oracle_p90"], marker="o")
    for row in stage1.itertuples():
        ax.annotate(row.source_added, (row.mean_candidates_per_sample, row.oracle_p90), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Mean cumulative eligible candidates/sample", ylabel="Oracle P@90")
    _save(fig, "04_oracle_p90_vs_bank_size", {"curve": stage1.to_dict(orient="records")})

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(stage1["source_added"], stage1["new_p90_successes"], color=COLOURS[0])
    ax.set(ylabel="New strict P@90 successes")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, "05_source_marginal_contribution", {"sources": stage1.to_dict(orient="records")})

    calibration = pd.read_csv(ARTIFACT / "final_selector/calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    ax.plot(calibration["mean_predicted_probability"], calibration["observed_fraction"], marker="o", color=COLOURS[0])
    ax.set(xlabel="Predicted P(IoU > 0.90)", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
    _save(fig, "06_selector_calibration", {"calibration": calibration.to_dict(orient="records")})

    importance = pd.read_csv(ARTIFACT / "final_selector/feature_importance.csv").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.barh(importance["feature"], importance["logistic_abs_coefficient"], color=COLOURS[1])
    ax.set(xlabel="Absolute regularized-logistic coefficient (diagnostic)")
    _save(fig, "07_feature_importance", {"importance": importance.to_dict(orient="records")})

    transitions = payload["threshold_transitions"]
    keys = [f"p_at_{x}" for x in thresholds]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(keys))
    ax.bar(x - 0.18, [transitions[k]["recovered"] for k in keys], 0.36, label="Recovered", color=COLOURS[0])
    ax.bar(x + 0.18, [transitions[k]["harmed"] for k in keys], 0.36, label="Harmed", color=COLOURS[3])
    ax.set_xticks(x, labels=[f"P@{v}" for v in thresholds])
    ax.set(ylabel="Samples")
    ax.legend()
    _save(fig, "08_recovered_harmed", {"transitions": transitions})

    grouped = pd.read_csv(REPORT / "grouped_results.csv")
    for number, (grouping, filename) in enumerate(
        (("query_type", "09_results_by_query_type"), ("target_size_bin", "10_results_by_target_size"), ("relation_type", "11_results_by_relation_type"))
    ):
        frame = grouped[grouped["grouping"] == grouping].sort_values("value")
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.bar(frame["value"], frame["p_at_90"], color=COLOURS[number])
        ax.set(ylabel="Locked P@90", ylim=(0, 1.0))
        ax.tick_params(axis="x", rotation=35)
        _save(fig, filename, {"groups": frame.to_dict(orient="records")})

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    labels = ["Stage-1 oracle", "Stage-2 oracle", "Locked"]
    values = [
        methods.set_index("method").loc["Stage-1 GT oracle", "p_at_90"],
        methods.set_index("method").loc["Stage-2 GT oracle", "p_at_90"],
        methods.set_index("method").loc["Locked conservative method", "p_at_90"],
    ]
    ax.bar(labels, values, color=COLOURS[:3])
    ax.set(ylabel="P@90", ylim=(0, 1.0))
    _save(fig, "12_stage1_stage2_contribution", {"labels": labels, "values": values})

    efficiency = payload["efficiency"]
    runtime = sum(
        efficiency[key]
        for key in (
            "stage1_total_measured_seconds",
            "stage2_seconds_before_write_sum",
            "final_selection_seconds_sum",
        )
    ) / 7675
    runtime_rows = [
        {"method": "Local HiFi-CS baseline", "seconds": 0.0, "p90": metrics_value}
        for metrics_value in [methods.set_index("method").loc["Local HiFi-CS baseline", "p_at_90"]]
    ] + [{"method": "Locked", "seconds": runtime, "p90": methods.set_index("method").loc["Locked conservative method", "p_at_90"]}]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for index, row in enumerate(runtime_rows):
        ax.scatter(row["seconds"], row["p90"], s=70, color=COLOURS[index])
        ax.annotate(row["method"], (row["seconds"], row["p90"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Measured pipeline seconds/sample", ylabel="P@90", ylim=(0, 1.0))
    _save(fig, "13_runtime_vs_p90", {"runtime": runtime_rows})

    gap = payload["fraction_of_paper_gap_closed"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(list(gap), [100 * gap[key] for key in gap], color=COLOURS[2])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="Paper-reference gap closed (%)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "14_paper_gap_closure", {"gap": gap})


def _overlay(rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> Image.Image:
    value = rgb.copy()
    active = np.asarray(mask, dtype=bool)
    value[active] = (0.55 * value[active] + 0.45 * np.asarray(colour)).astype(np.uint8)
    return Image.fromarray(value).resize((320, 240), Image.Resampling.LANCZOS)


def _gallery_assets(sample_ids: set[str]) -> dict[str, str]:
    compact = load_compact_manifest(
        PROJECT_ROOT / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/test/manifest.jsonl",
        expected_split="test",
        expected_count=7675,
    )
    by_id = {row.sample_id: row for row in compact}
    gt_rows = load_frozen_ground_truth_manifest(
        PROJECT_ROOT / "artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json",
        hifics_root=PROJECT_ROOT / "hifics",
        expected_count=7675,
    )
    gt_by_prefix = {f"q{int(row['question_index']):07d}_": row for row in gt_rows}
    asset_root = GALLERIES / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    result = {}
    for sample_id in sorted(sample_ids):
        row = by_id[sample_id]
        rgb = np.asarray(Image.open(row.rgb_path).convert("RGB"), dtype=np.uint8)
        gt = resize_binary_mask(load_ground_truth_mask(gt_by_prefix[sample_id.split("_", 1)[0] + "_"]), rgb.shape[:2])
        hifi = load_binary_mask(row.native_mask_path, expected_shape=rgb.shape[:2])
        locked_root = EXPERIMENT / "locked_benchmark_masks" / sample_id
        stage1_decision = json.loads(
            (locked_root / "stage1_decision.json").read_text(encoding="utf-8")
        )
        stage1_masks = load_candidate_masks_npz(
            EXPERIMENT / "proposals" / sample_id / "candidate_masks.npz"
        )
        selected_stage1 = stage1_masks[str(stage1_decision["candidate_id"])]
        final = load_binary_mask(EXPERIMENT / "locked_benchmark_masks" / sample_id / "final_mask.png", expected_shape=rgb.shape[:2])
        panels = [
            Image.fromarray(rgb).resize((320, 240)),
            _overlay(rgb, gt, (255, 40, 40)),
            _overlay(rgb, hifi, (255, 180, 0)),
            _overlay(rgb, selected_stage1, (80, 120, 255)),
            _overlay(rgb, final, (0, 220, 160)),
        ]
        canvas = Image.new("RGB", (1600, 280), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (320 * index, 40))
        canvas.save(asset_root / f"{sample_id}.jpg", quality=88)
        result[sample_id] = f"assets/{sample_id}.jpg"
    return result


def _galleries() -> None:
    detail = pd.read_parquet(REPORT / "per_sample_method_metrics.parquet")
    pivot = detail.pivot(index="sample_id", columns="method", values="iou")
    relation_ids = set(detail.loc[detail["relation_type"].notna(), "sample_id"])
    failure = pd.read_csv(REPORT / "failure_taxonomy.csv").set_index("sample_id")
    conditions = {
        "text_only_highest_score_failures.html": (pivot["SAM3 full-query highest score"] <= 0.90) & (pivot["Stage-1 GT oracle"] > 0.90),
        "overlap_selector_successes.html": (pivot["Maximum HiFi overlap"] > 0.90) & (pivot["Local HiFi-CS baseline"] <= 0.90),
        "relation_rule_successes.html": pivot.index.to_series().isin(relation_ids) & (pivot["Deterministic relation-aware"] > 0.90) & (pivot["Local HiFi-CS baseline"] <= 0.90),
        "learned_selector_recoveries.html": (pivot["Learned Stage-1 selector"] > 0.90) & (pivot["Local HiFi-CS baseline"] <= 0.90),
        "learned_selector_harmful_switches.html": (pivot["Learned Stage-1 selector"] <= 0.90) & (pivot["Local HiFi-CS baseline"] > 0.90),
        "stage2_refinement_improvements.html": pivot["Stage-1 + Stage-2 selector (pre-gate)"] > pivot["Learned Stage-1 selector"] + 0.02,
        "stage2_refinement_harmful_cases.html": pivot["Stage-1 + Stage-2 selector (pre-gate)"] + 0.02 < pivot["Learned Stage-1 selector"],
        "oracle_available_but_selector_failed.html": (pivot["Stage-2 GT oracle"] > 0.90) & (pivot["Locked conservative method"] <= 0.90),
        "no_p90_candidate_cases.html": pivot["Stage-2 GT oracle"] <= 0.90,
        "neighbouring_object_leakage.html": pivot.index.to_series().map(failure["category"]).fillna("").str.contains("leakage"),
        "wrong_instance_cases.html": (pivot["Locked conservative method"] <= 0.50) & (pivot["Stage-2 GT oracle"] > 0.90),
    }
    selected = {name: list(pivot.index[mask][:12]) for name, mask in conditions.items()}
    assets = _gallery_assets({sample_id for values in selected.values() for sample_id in values})
    GALLERIES.mkdir(parents=True, exist_ok=True)
    for name, sample_ids in selected.items():
        sections = []
        for sample_id in sample_ids:
            root = EXPERIMENT / "locked_benchmark_masks" / sample_id
            query = json.loads((root / "provenance.json").read_text())["query"]
            parsed = json.loads((root / "query_parse.json").read_text())
            stage1_decision = json.loads(
                (root / "stage1_decision.json").read_text(encoding="utf-8")
            )
            gate = json.loads((root / "gate_decision.json").read_text())
            values = pivot.loc[sample_id]
            all_scores = pd.read_parquet(root / "stage2_scores.parquet")
            hifi_score = all_scores[
                all_scores["source_family"] == "STAGE2_HIFI_FALLBACK"
            ].iloc[0]
            selected_score = all_scores[
                all_scores["candidate_id"].astype(str) == str(gate["candidate_id"])
            ].iloc[0]
            p90_margin = float(
                selected_score["m1_p90_calibrated"]
                - hifi_score["m1_p90_calibrated"]
            )
            iou_margin = float(
                selected_score["m2_predicted_iou"]
                - hifi_score["m2_predicted_iou"]
            )
            scores = all_scores.sort_values(
                "m1_p90_calibrated", ascending=False
            ).head(5)
            score_table = scores[
                [
                    "candidate_id",
                    "source_family",
                    "m1_p90_calibrated",
                    "m2_predicted_iou",
                ]
            ].to_html(index=False, float_format=lambda x: f"{x:.4f}")
            sections.append(
                f"<section><h2>{html.escape(sample_id)}</h2><p>{html.escape(query)}</p>"
                f"<details><summary>Parsed semantics</summary><pre>{html.escape(json.dumps(parsed, indent=2, sort_keys=True))}</pre></details>"
                f"<img class='wide' src='{assets[sample_id]}'><p>Panels: RGB | GT (evaluation only) | HiFi | selected Stage-1 | locked final.</p>"
                f"<p>baseline={values['Local HiFi-CS baseline']:.4f}; final={values['Locked conservative method']:.4f}; "
                f"stage1 oracle={values['Stage-1 GT oracle']:.4f}; stage2 oracle={values['Stage-2 GT oracle']:.4f}; "
                f"Stage-1 ID={html.escape(str(stage1_decision['candidate_id']))}; "
                f"Stage-1 source={html.escape(str(stage1_decision['source_family']))}; "
                f"Stage-1 score={float(stage1_decision['selection_score']):.4f}; "
                f"source={html.escape(str(gate['source_family']))}; selector P@90={float(selected_score['m1_p90_calibrated']):.4f}; "
                f"P@90 margin={p90_margin:+.4f}; predicted-IoU margin={iou_margin:+.4f}; "
                f"accept={bool(gate['gate_accept'])}; reason={html.escape(str(gate['gate_reason']))}</p>"
                f"<img class='grid' src='../../proposals/{sample_id}/proposal_grid.png'>"
                f"<img class='grid' src='../../stage2/{sample_id}/refinement_grid.png'>{score_table}</section>"
            )
        (GALLERIES / name).write_text(
            "<!doctype html><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1400px;margin:auto}.wide{width:100%}.grid{width:48%;vertical-align:top}section{border-bottom:1px solid #bbb;padding:1rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.25rem}</style>"
            + f"<h1>{html.escape(name.removesuffix('.html'))}</h1>"
            + ("".join(sections) if sections else "<p>No matching cases.</p>"),
            encoding="utf-8",
        )


def _report(payload: dict) -> None:
    metrics = payload["metrics"]
    pilot = json.loads((EXPERIMENT / "oracle_stage1_pilot/summary.json").read_text())
    val1 = json.loads((EXPERIMENT / "oracle_stage1/summary.json").read_text())
    val2 = json.loads((EXPERIMENT / "oracle_stage2/summary.json").read_text())
    stage1_training = json.loads((ARTIFACT / "stage1_selector/validation_metrics.json").read_text())
    final_training = json.loads((ARTIFACT / "final_selector/validation_metrics.json").read_text())
    lock = json.loads((ARTIFACT / "formal_lock.json").read_text())
    source = pd.read_csv(EXPERIMENT / "oracle_stage1_test/source_contributions.csv")
    ablations = pd.read_csv(REPORT / "ablation_results.csv")
    source_transitions = pd.read_csv(REPORT / "selected_source_transitions.csv")
    method_detail = pd.read_parquet(REPORT / "per_sample_method_metrics.parquet")
    relation_detail = method_detail[method_detail["relation_type"].notna()]
    relation_metrics = {
        method: summarize
        for method, summarize in (
            (
                method,
                float((frame["iou"].to_numpy(float) > 0.90).mean()),
            )
            for method, frame in relation_detail.groupby("method", sort=False)
        )
    }
    relation_delta = relation_metrics.get("Deterministic relation-aware", np.nan) - relation_metrics.get(
        "Local HiFi-CS baseline", np.nan
    )
    feature_rows = ablations[
        (ablations["ablation_family"] == "feature")
        & (ablations["estimate_type"] == "GROUPED_OOF_GT_FREE_SELECTOR")
    ].set_index("variant")
    depth_delta = (
        float(feature_rows.loc["+ depth", "p_at_90"])
        - float(feature_rows.loc["+ pairwise relations", "p_at_90"])
        if {"+ depth", "+ pairwise relations"}.issubset(feature_rows.index)
        else np.nan
    )
    stage_detail = method_detail[
        method_detail["method"].isin(
            ["Learned Stage-1 selector", "Stage-1 + Stage-2 selector (pre-gate)"]
        )
    ]
    stage_pivot = stage_detail.pivot(index="sample_id", columns="method", values="iou")
    stage2_improved = int(
        (
            stage_pivot["Stage-1 + Stage-2 selector (pre-gate)"]
            > stage_pivot["Learned Stage-1 selector"] + 1e-12
        ).sum()
    )
    stage2_harmed = int(
        (
            stage_pivot["Stage-1 + Stage-2 selector (pre-gate)"] + 1e-12
            < stage_pivot["Learned Stage-1 selector"]
        ).sum()
    )
    stage2_crosses_p90 = (
        stage_pivot["Learned Stage-1 selector"] <= 0.90
    ) & (stage_pivot["Stage-1 + Stage-2 selector (pre-gate)"] > 0.90)
    stage2_identity_recoveries = int(
        (
            stage2_crosses_p90
            & (stage_pivot["Learned Stage-1 selector"] <= 0.50)
        ).sum()
    )
    stage2_strict_boundary_recoveries = int(
        (
            stage2_crosses_p90
            & (stage_pivot["Learned Stage-1 selector"] > 0.50)
        ).sum()
    )
    locked_detail = method_detail[
        method_detail["method"] == "Locked conservative method"
    ][["sample_id", "baseline_iou", "stage1_oracle_source"]]
    oracle_detail = method_detail[
        method_detail["method"] == "Stage-1 GT oracle"
    ][["sample_id", "iou"]].rename(columns={"iou": "stage1_oracle_iou"})
    automatic = locked_detail.merge(
        oracle_detail, on="sample_id", how="inner", validate="one_to_one"
    )
    automatic = automatic[
        (automatic["stage1_oracle_source"] == "AUTOMATIC")
        & (automatic["baseline_iou"] <= 0.90)
        & (automatic["stage1_oracle_iou"] > 0.90)
    ]
    automatic_wrong_instance_proxy = int((automatic["baseline_iou"] <= 0.25).sum())
    automatic_wrong_instance_proxy_fraction = (
        float(automatic_wrong_instance_proxy / len(automatic))
        if len(automatic)
        else None
    )
    added_sources = source[source["source_added"] != "hifi_only"]
    largest_oracle_source = added_sources.sort_values(
        ["new_p90_successes", "marginal_mean_iou"], ascending=False
    ).iloc[0]
    most_harmful_source = source_transitions.sort_values(
        ["harmed_p90", "selected_samples"], ascending=False
    ).iloc[0]
    grouped_findings = {
        "relation_rule_p90_delta_vs_hifi_on_relation_queries": float(relation_delta),
        "validation_depth_feature_p90_delta_after_pairwise_features": float(depth_delta),
        "stage2_selector_iou_improved_samples": stage2_improved,
        "stage2_selector_iou_harmed_samples": stage2_harmed,
        "stage2_new_p90_identity_recoveries_from_iou_le_0_50": (
            stage2_identity_recoveries
        ),
        "stage2_new_p90_strict_boundary_recoveries_from_iou_gt_0_50": (
            stage2_strict_boundary_recoveries
        ),
        "automatic_oracle_new_p90_successes": len(automatic),
        "automatic_oracle_wrong_instance_proxy_iou_le_0_25": (
            automatic_wrong_instance_proxy
        ),
        "automatic_oracle_wrong_instance_proxy_fraction": (
            automatic_wrong_instance_proxy_fraction
        ),
        "largest_stage1_oracle_p90_source": str(largest_oracle_source["source_added"]),
        "largest_stage1_oracle_new_p90_successes": int(
            largest_oracle_source["new_p90_successes"]
        ),
        "most_harmful_locked_selected_source": str(
            most_harmful_source["selected_source"]
        ),
        "most_harmful_locked_selected_source_count": int(
            most_harmful_source["harmed_p90"]
        ),
    }
    (REPORT / "grouped_findings.json").write_text(
        json.dumps(grouped_findings, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    methods = [
        "HiFi-CS paper numeric reference",
        "Local HiFi-CS baseline",
        "Existing selective SAM3",
        "SAM3 full-query highest score",
        "SAM3 target-category highest score",
        "Maximum HiFi overlap",
        "Deterministic relation-aware",
        "Learned Stage-1 selector",
        "Stage-1 + Stage-2 selector (pre-gate)",
        "Locked conservative method",
        "Stage-1 GT oracle",
        "Stage-2 GT oracle",
    ]
    lines = ["| Method | mIoU | P@50 | P@60 | P@70 | P@80 | P@90 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        value = metrics[method]
        suffix = " *" if "GT oracle" in method else ""
        lines.append(f"| {method}{suffix} | {value['mean_iou']:.6f} | {value['p_at_50']:.6f} | {value['p_at_60']:.6f} | {value['p_at_70']:.6f} | {value['p_at_80']:.6f} | {value['p_at_90']:.6f} |")
    transition_lines = ["| Threshold | Baseline success | Locked success | Recovered | Harmed | Net |", "|---|---:|---:|---:|---:|---:|"]
    for threshold in (50, 60, 70, 80, 90):
        value = payload["threshold_transitions"][f"p_at_{threshold}"]
        transition_lines.append(f"| P@{threshold} | {value['baseline_success']}/7675 | {value['method_success']}/7675 | {value['recovered']} | {value['harmed']} | {value['net']} |")
    source_lines = ["| Source added | New unique candidates | New P@90 oracle successes | Marginal P@90 |", "|---|---:|---:|---:|"]
    previous = 0
    for row in source.itertuples():
        count = int(row.cumulative_candidate_count) - previous
        previous = int(row.cumulative_candidate_count)
        source_lines.append(f"| {row.source_added} | {count} | {int(row.new_p90_successes)} | {row.marginal_p90:.6f} |")
    baseline = metrics["Local HiFi-CS baseline"]
    locked = metrics["Locked conservative method"]
    stage2_oracle = metrics["Stage-2 GT oracle"]
    if stage2_oracle["p_at_90"] < metrics["HiFi-CS paper numeric reference"]["p_at_90"]:
        conclusion = "A. Proposal-generation bottleneck: the expanded oracle remains below the paper numeric P@90 reference, so this proposal space cannot close the local-to-paper gap."
    elif locked["p_at_90"] <= baseline["p_at_90"]:
        conclusion = "B. Selector bottleneck: the oracle is high, but the locked GT-free selector does not recover it."
    else:
        conclusion = "C. Successful improvement: the locked selector/refinement improves strict grounding under the predefined safeguards."
    git_diff = subprocess.run(["git", "diff", "--stat"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    document = f"""# SAM 3 Proposal Bank for Strict P@90 — Results

## Objective and protocol

Frozen repeated-FiLM HiFi-CS → official frozen `facebook/sam3` proposal bank → language/relation/depth/CLIP selector → target-specific Stage 2 → conservative HiFi fallback. The 7,675 rows are a **locked post-hoc benchmark**, because prior aggregate failures influenced this design; grouped development evidence is primary. Paper values are numeric references only: 70/30 protocol equivalence was not proven.

Model revision: `{lock['model']['revision']}`; Transformers 5.14.1; CPU float32 on Apple M5 Pro, 24 GiB; `local_files_only=True`. P@X uses strict `IoU > X`.

## Audit, parser, proposals, and cache

All baseline manifests/checkpoints were checksummed; train/validation/test contain 1,104/165/325 unique frames with zero frame/RGB overlap. Query parsing is text-only and covers name, attribute, relation, location, and mixed templates without answer-instance fields. Proposals include H0/H1, full/short/reference PCS, automatic masks, Tracker box/point/mask/component prompts, controlled thresholds, and morphology. H0 remains the canonical fallback after strict 0.995 deduplication. CPU float32 image embeddings are keyed by RGB SHA, revision, processor, resolution, dtype, and backend; validation/test caches are persistent and the 1,104-frame training cache is LRU-bounded to 4 GiB.

## Pilot and oracle-first result

The stratified pilot contains 100 frames/500 queries. Fifty grids were manually reviewed. `expanded_proposal_bank_oracle_P@90 = {pilot['expanded_proposal_bank_oracle_P@90_numerator']}/{pilot['expanded_proposal_bank_oracle_P@90_denominator']} = {pilot['expanded_proposal_bank_oracle_P@90']:.4f}`; oracle mIoU `{pilot['mean_iou']:.6f}`; mean candidates `{pilot['mean_candidate_count']:.2f}`.

Full validation Stage-1 oracle: mIoU `{val1['mean_iou']:.6f}`, P@90 `{val1['p_at_90_numerator']}/{val1['p_at_90_denominator']} = {val1['p_at_90']:.6f}`. Stage-2 oracle: mIoU `{val2['stage2_oracle']['mean_iou']:.6f}`, P@90 `{val2['stage2_oracle']['p_at_90_numerator']}/{val2['stage2_oracle']['p_at_90_denominator']} = {val2['stage2_oracle']['p_at_90']:.6f}`.

## Selector, refinement, gate, and lock

Stage-1 used five-fold exact-frame GroupKFold, inverse-candidate sample weighting, cross-fitted isotonic calibration, and fixed seed 42. Selected Stage-1 model: `{stage1_training['selected_model']}`. Stage 2 used OOF-selected—not GT-best—candidates. Final selected method: `{final_training['selected_method']}`. Gate selection used `{final_training['gate_selection_objective']}`; Boundary F-score was computed only on validation alternatives and never exposed as an inference feature. HiFi is the default; every switch must pass the locked probability-margin, IoU-margin, consensus, probability-mass, depth, expansion, fragmentation, stability, relation, and reliability conditions. `NO_TEST_GT_USED_FOR_SELECTION=true`; output masks were checksummed before GT evaluation.

## Post-hoc benchmark

{chr(10).join(lines)}

The conservative gate triggered on `{payload['gate_statistics']['triggered_samples']}/7675` samples and accepted `{payload['gate_statistics']['accepted_switch_samples']}/7675`. Accepted-switch improvement precision was `{payload['gate_statistics']['accepted_switch_improvement_precision']}`; strict-P@90 accepted-switch precision was `{payload['gate_statistics']['accepted_switch_strict_p90_precision']}`; P@90 outcome-changing precision was `{payload['gate_statistics']['p90_outcome_changing_precision']}` (recovered divided by recovered plus harmed).

*Non-deployable diagnostic upper bound; ground truth used for candidate selection.*

{chr(10).join(source_lines)}

{chr(10).join(transition_lines)}

Scene- and frame-clustered bootstrap CIs use 10,000 replicates (seed 42); threshold comparisons use paired McNemar tests with Holm correction. Full values are in `outputs/sam3_proposal_bank_p90_v1/report/formal_metrics.json`.

## Grouped results, ablations, and failures

Query-type, relation-type, target-size, source, and baseline-IoU tables are in `grouped_results.csv`; failure classes are in `failure_taxonomy.csv`. Proposal and feature/query/selector ablations are in `ablation_results.csv`. Galleries distinguish proposal absence, oracle-available selector failures, relation recoveries, neighbouring leakage, Stage-2 gains, and harmful switches.

- Relation-aware deterministic selection changes relation-query P@90 by `{relation_delta:+.6f}` versus local HiFi on the same rows.
- Adding depth after pairwise-relation features changes grouped-validation OOF P@90 by `{depth_delta:+.6f}`. This is associative ablation evidence, not proof that depth alone caused each rejected leakage case.
- Stage 2 improves IoU for `{stage2_improved}` selected samples and harms `{stage2_harmed}` relative to the learned Stage-1 selection; oracle and deployable changes remain separate.
- Among new deployable pre-gate Stage-2 P@90 crossings, `{stage2_identity_recoveries}` start from Stage-1 IoU <= 0.50 (wrong-identity/severe-failure proxy) and `{stage2_strict_boundary_recoveries}` start above 0.50 (strict-boundary/mask-quality proxy).
- Automatic proposals are the Stage-1 oracle source for `{len(automatic)}` new P@90 opportunities over failed HiFi cases; `{automatic_wrong_instance_proxy}` (`{automatic_wrong_instance_proxy_fraction}`) have baseline IoU <= 0.25. This proxy is reported rather than treating low IoU as proven wrong identity.
- The largest canonical-source Stage-1 oracle contribution is `{largest_oracle_source['source_added']}` with `{int(largest_oracle_source['new_p90_successes'])}` new strict successes. Automatic-proposal effects are reported by source, but the aggregate alone cannot establish that they *mainly* fix wrong-instance cases.
- The locked selected source with the most harmful P@90 replacements is `{most_harmful_source['selected_source']}` with `{int(most_harmful_source['harmed_p90'])}` harms.

## Runtime and storage

Peak RSS `{payload['efficiency']['peak_rss_bytes'] / 1024**3:.2f}` GiB; experiment storage `{payload['efficiency']['experiment_disk_bytes'] / 1024**3:.2f}` GiB. No Dex-Net, GQ-CNN, VGN, or physical grasp evaluation was run.

## Eleven-fallacy scan

1. Pristine-test fallacy: avoided; result is labelled post-hoc.
2. Oracle-as-deployable fallacy: avoided; oracle rows are explicitly GT-only.
3. Candidate/selector conflation: avoided; oracle was computed first.
4. Test-set tuning: blocked by formal config/model hashes before benchmark inference.
5. IID-query fallacy: avoided with frame/scene grouping and clustered bootstrap.
6. Paper-protocol equivalence: not claimed.
7. Mask-to-grasp causality: no physical grasp-success claim.
8. Gallery cherry-picking: galleries are rule-generated and numerical tables are complete.
9. Significance-by-point-estimate: avoided; paired grouped CIs and corrected McNemar tests are reported.
10. Missing-output survivorship: all 7,675 terminal states were verified before evaluation.
11. Canonical-export confirmation bias: export occurs only if prespecified safeguards pass.

## Conclusion

**{conclusion}** Protocol mismatch still limits direct interpretation of the paper-reference gap.

## Files, tests, and leakage

Targeted tests, static/runtime GT-access guards, parser/cache/proposal/oracle/selector/gate/output-lock tests, and repository regression tests are recorded in the final execution log. Key files are under `src/segmentation/`, `scripts/`, `configs/sam3_proposal_bank_p90_v1/`, `artifacts/sam3_proposal_bank_p90_v1/`, and `outputs/sam3_proposal_bank_p90_v1/`.

`git diff --stat` at report generation:

```text
{git_diff.rstrip()}
```

`git status --short` (includes untracked experiment files omitted by diff-stat):

```text
{git_status.rstrip()}
```

Exact next downstream command (documented only; **not executed**):

```bash
.venv-sam3-cpu/bin/python scripts/run_hifics_dexnet_candidates.py \\
  --mask-root runs/hifics_sam3_proposal_selector_LOCKED \\
  --output-dir outputs/dexnet_candidates_sam3_proposal_selector \\
  --split test --mode candidate-only --resume --visualize
```
"""
    (PROJECT_ROOT / "docs/SAM3_PROPOSAL_BANK_P90_RESULTS.md").write_text(document, encoding="utf-8")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GALLERIES.mkdir(parents=True, exist_ok=True)
    payload = json.loads((REPORT / "formal_metrics.json").read_text(encoding="utf-8"))
    _plots(payload)
    _galleries()
    _report(payload)
    figure_files = list(FIGURES.glob("*"))
    if len(list(FIGURES.glob("*.png"))) != 14 or len(list(FIGURES.glob("*.pdf"))) != 14 or len(list(FIGURES.glob("*.svg"))) != 14:
        raise RuntimeError("required 14-figure PNG/PDF/SVG triplets are incomplete")
    print(json.dumps({"status": "COMPLETE", "figure_files": len(figure_files), "galleries": len(list(GALLERIES.glob('*.html')))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
