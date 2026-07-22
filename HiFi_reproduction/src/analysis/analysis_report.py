"""Tables, figures, and a self-contained research report for candidate analysis."""

from __future__ import annotations

import html
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FORBIDDEN_METRIC_PHRASES = ("grasp success", "success rate", "physical success")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def wilson_interval(numerator: int, denominator: int) -> tuple[float | None, float | None]:
    if denominator == 0:
        return None, None
    p = numerator / denominator
    z = NormalDist().inv_cdf(0.975)
    divisor = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / divisor
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / divisor
    return max(0.0, center - radius), min(1.0, center + radius)


def scene_cluster_rate_interval(
    values: Sequence[bool] | np.ndarray,
    scenes: Sequence[str] | np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    """Percentile CI from resampling scene clusters with all contained rows."""

    y = np.asarray(values, dtype=float)
    cluster = np.asarray(scenes, dtype=str)
    unique, inverse = np.unique(cluster, return_inverse=True)
    if not len(unique) or replicates <= 0:
        return None, None
    sums = np.bincount(inverse, weights=y, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    batch = 512
    for start in range(0, replicates, batch):
        stop = min(replicates, start + batch)
        draws = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        estimates[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def rate_record(
    name: str,
    values: pd.Series | np.ndarray,
    scenes: pd.Series | np.ndarray,
    *,
    replicates: int,
    seed: int,
    denominator_label: str = "all_manifest_samples",
) -> dict[str, Any]:
    y = np.asarray(values, dtype=bool)
    numerator, denominator = int(y.sum()), int(len(y))
    wilson = wilson_interval(numerator, denominator)
    bootstrap = scene_cluster_rate_interval(y, scenes, replicates=replicates, seed=seed)
    return {
        "metric": name,
        "numerator": numerator,
        "denominator": denominator,
        "denominator_label": denominator_label,
        "percentage": 100.0 * numerator / denominator if denominator else None,
        "wilson_ci_low": wilson[0],
        "wilson_ci_high": wilson[1],
        "scene_bootstrap_ci_low": bootstrap[0],
        "scene_bootstrap_ci_high": bootstrap[1],
        "bootstrap_replicates": replicates,
        "cluster_key": "scene_id",
    }


def _count_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 9.5, 19.5, 49.5, np.inf],
        labels=["0", "1", "2", "3", "4", "5–9", "10–19", "20–49", "50+"],
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _plot_save(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _group_rate(samples: pd.DataFrame, group: str, metric: str) -> pd.DataFrame:
    rows = []
    for value, data in samples.groupby(group, observed=True, dropna=False):
        rows.append(
            {
                group: value,
                "metric": metric,
                "numerator": int(data[metric].astype(bool).sum()),
                "denominator": len(data),
                "percentage": 100 * data[metric].astype(bool).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_tables(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    output: Path,
    *,
    oracle_table: pd.DataFrame | None,
    sensitivity_table: pd.DataFrame | None,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    s = samples.copy()
    s["candidate_count_bin"] = _count_bin(s["n_official_candidates"])
    s["mask_iou_bin"] = pd.cut(
        s["mask_iou"], [0, .25, .50, .70, .90, 1.0000001], right=False,
        labels=["[0,.25)", "[.25,.50)", "[.50,.70)", "[.70,.90)", "[.90,1]"],
        include_lowest=True,
    )
    overall_metrics = {
        "zero_official": s.n_official_candidates == 0,
        "one_official": s.n_official_candidates == 1,
        "two_or_more_official": s.n_official_candidates >= 2,
        "zero_pred_filter": s.n_pred_filtered_candidates == 0,
        "one_pred_filter": s.n_pred_filtered_candidates == 1,
        "two_or_more_pred_filter": s.n_pred_filtered_candidates >= 2,
        "one_or_more_gt_positive": s.n_gt_positive_primary >= 1,
        "two_or_more_gt_positive": s.n_gt_positive_primary >= 2,
    }
    multiplicity = pd.DataFrame(
        [
            rate_record(key, value, s.scene_id, replicates=bootstrap_replicates, seed=seed)
            for key, value in overall_metrics.items()
        ]
    )
    rankability = pd.DataFrame(
        [
            rate_record("pre_filter_rankable", s.pre_filter_rankable, s.scene_id, replicates=bootstrap_replicates, seed=seed),
            rate_record("post_filter_rankable", s.post_filter_rankable, s.scene_id, replicates=bootstrap_replicates, seed=seed),
            rate_record("gt_rankable", s.gt_rankable, s.scene_id, replicates=bootstrap_replicates, seed=seed),
            rate_record(
                "post_filter_rankable_given_baseline_selection",
                s.loc[s.has_pred_filtered_candidate, "post_filter_rankable"],
                s.loc[s.has_pred_filtered_candidate, "scene_id"],
                replicates=bootstrap_replicates,
                seed=seed,
                denominator_label="samples_with_predicted_target_candidates",
            ),
        ]
    )
    opportunity = pd.DataFrame(
        [
            rate_record(name, s.opportunity_class.eq(name), s.scene_id, replicates=bootstrap_replicates, seed=seed)
            for name in (
                "already_correct",
                "post_filter_ranking_recoverable",
                "filter_recoverable",
                "generation_limited",
                "no_official_candidate",
                "technical_failure",
            )
        ]
    )
    rank_counts = (
        s["rank_first_gt_positive"].dropna().astype(int).value_counts().sort_index()
    )
    first_rank = pd.DataFrame({"rank": rank_counts.index, "sample_count": rank_counts.values})
    recall_rows = []
    for k in (1, 2, 3, 5, 10):
        recall_rows.append(rate_record(f"Recall@{k}", s.rank_first_gt_positive.le(k), s.scene_id, replicates=bootstrap_replicates, seed=seed))
    recall_rows.append(rate_record("Recall@Any", s.has_gt_positive_anywhere, s.scene_id, replicates=bootstrap_replicates, seed=seed))
    recall = pd.DataFrame(recall_rows)
    quality_margin = s.loc[
        s.has_gt_positive_anywhere,
        ["sample_id", "scene_id", "opportunity_class", "best_gt_positive_quality", "baseline_top1_quality", "quality_gap"],
    ].copy()

    primary_rows = []
    top_quality = candidates.groupby("sample_id")["vgn_quality"].max()
    for label, group in s.groupby("primary_failure_class", sort=True):
        rate = rate_record(label, s.primary_failure_class.eq(label), s.scene_id, replicates=bootstrap_replicates, seed=seed)
        rate.update(
            median_mask_iou=float(group.mask_iou.median()),
            median_candidate_count=float(group.n_official_candidates.median()),
            median_top_quality=float(top_quality.reindex(group.sample_id).median()) if len(group) else None,
        )
        primary_rows.append(rate)
    primary = pd.DataFrame(primary_rows)
    flag_columns = sorted(column for column in s if column.startswith("S_"))
    secondary = pd.DataFrame(
        [
            rate_record(column, s[column].fillna(False), s.scene_id, replicates=bootstrap_replicates, seed=seed)
            for column in flag_columns
        ]
    )
    crosswalk = pd.crosstab(s.pred_status, s.primary_failure_class).reset_index()
    query_table = _group_rate(s, "query_type", "has_gt_positive_anywhere")
    iou_table = _group_rate(s, "mask_iou_bin", "has_gt_positive_anywhere")
    taxonomy_query = pd.crosstab(s.query_type, s.primary_failure_class).reset_index()
    taxonomy_iou = pd.crosstab(s.mask_iou_bin, s.primary_failure_class).reset_index()
    taxonomy_count = pd.crosstab(s.candidate_count_bin, s.primary_failure_class).reset_index()
    representative = (
        s.sort_values(["primary_failure_class", "mask_iou", "dataset_index"])
        .groupby("primary_failure_class", as_index=False)
        .head(5)
    )
    outputs = {
        "multiplicity_overall": multiplicity,
        "multiplicity_by_query_type": query_table,
        "multiplicity_by_mask_iou": iou_table,
        "rankability": rankability,
        "reranking_opportunity": opportunity,
        "first_gt_positive_rank": first_rank,
        "recall_at_k": recall,
        "quality_margin": quality_margin,
        "oracle_upper_bounds": oracle_table if oracle_table is not None else pd.DataFrame(),
        "oracle_sensitivity": sensitivity_table if sensitivity_table is not None else pd.DataFrame(),
        "primary_failure_taxonomy": primary,
        "secondary_failure_flags": secondary,
        "taxonomy_by_query_type": taxonomy_query,
        "taxonomy_by_mask_iou": taxonomy_iou,
        "taxonomy_by_candidate_count": taxonomy_count,
        "current_status_vs_taxonomy": crosswalk,
        "representative_samples": representative,
    }
    for name, frame in outputs.items():
        _write_csv(frame, tables / f"{name}.csv")
    return outputs


def _bar_categories(samples: pd.DataFrame, column: str, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    counts = samples[column].value_counts().sort_index()
    plt.figure(figsize=(9, 4.8))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#3977b8")
    plt.title(title)
    plt.xlabel(column.replace("_", " "))
    plt.ylabel("Samples")
    plt.xticks(rotation=30, ha="right")
    _plot_save(path)


def build_figures(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    output: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    root = output / "figures"
    root.mkdir(parents=True, exist_ok=True)
    samples = samples.copy()
    samples["candidate_count_bin"] = _count_bin(samples["n_official_candidates"])
    samples["mask_iou_bin"] = pd.cut(
        samples["mask_iou"], [0, .25, .50, .70, .90, 1.0000001], right=False,
        labels=["[0,.25)", "[.25,.50)", "[.50,.70)", "[.70,.90)", "[.90,1]"],
        include_lowest=True,
    )
    try:
        samples["target_size_bin"] = pd.qcut(
            samples["gt_mask_area_px"], 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"]
        )
    except ValueError:
        samples["target_size_bin"] = "unavailable"
    paths: list[Path] = []

    def save(name: str) -> Path:
        path = root / name
        paths.append(path)
        _plot_save(path)
        return path

    for column, name, title in (
        ("n_official_candidates", "multiplicity_histogram.png", "Official VGN candidate multiplicity"),
        ("n_official_candidates", "multiplicity_hist_all.png", "Official candidate multiplicity — all samples"),
        ("n_pred_filtered_candidates", "multiplicity_hist_pred_filtered.png", "Predicted-filter candidate multiplicity"),
        ("n_gt_positive_primary", "multiplicity_hist_gt_positive.png", "GT target-consistent candidate multiplicity"),
    ):
        plt.figure(figsize=(8, 4.5))
        sns.histplot(samples[column], discrete=True, binrange=(0, min(50, int(samples[column].max()))), color="#3977b8")
        plt.title(title)
        plt.xlabel("Candidates per sample")
        plt.ylabel("Samples")
        save(name)
    plt.figure(figsize=(6, 6))
    plt.scatter(samples.n_official_candidates, samples.n_distinct_pose_modes, s=8, alpha=.25)
    limit = max(samples.n_official_candidates.max(), samples.n_distinct_pose_modes.max())
    plt.plot([0, limit], [0, limit], "k--", linewidth=1)
    plt.title("Raw candidates versus distinct 6-DoF pose modes")
    plt.xlabel("Raw official candidates")
    plt.ylabel("Distinct pose modes")
    save("raw_vs_distinct_modes.png")
    rank = samples.rank_first_gt_positive.dropna().astype(int).sort_values()
    plt.figure(figsize=(7, 4.5))
    if len(rank):
        plt.step(rank, np.arange(1, len(rank) + 1) / len(samples), where="post")
    plt.title("CDF of first GT target-consistent candidate rank")
    plt.xlabel("VGN quality rank")
    plt.ylabel("Fraction of all samples")
    save("first_positive_rank_cdf.png")
    recall = tables["recall_at_k"]
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=recall, x="metric", y="percentage", color="#3a9d5d")
    plt.title("Target-consistent candidate Recall@K")
    plt.ylabel("Samples (%)")
    save("recall_at_k.png")
    plt.figure(figsize=(7, 4.5))
    plot_candidates = candidates.assign(
        target_consistency=np.where(candidates.gt_target_positive_primary, "GT-positive", "GT-negative")
    )
    sns.violinplot(data=plot_candidates, x="target_consistency", y="vgn_quality", cut=0, inner="quartile")
    plt.title("Official VGN quality by target consistency")
    save("quality_positive_vs_negative.png")
    plt.figure(figsize=(7, 4.5))
    gaps = samples.loc[samples.opportunity_class.isin(["filter_recoverable", "post_filter_ranking_recoverable"]), "quality_gap"].dropna()
    sns.histplot(gaps, bins=30)
    plt.title("Quality gap for re-selection-recoverable samples")
    plt.xlabel("Baseline quality − best GT-positive quality")
    save("recoverable_quality_gap.png")
    oracle = tables["oracle_upper_bounds"]
    plt.figure(figsize=(9, 4.8))
    if not oracle.empty:
        sns.barplot(data=oracle, x="metric", y="percentage", color="#7b55a3")
    plt.title("Oracle target-selection upper bounds")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Target-consistent samples (%)")
    save("oracle_upper_bound_bar.png")
    sensitivity = tables["oracle_sensitivity"]
    plt.figure(figsize=(9, 4.8))
    if not sensitivity.empty and {"label_tolerance", "percentage", "metric"}.issubset(sensitivity):
        sns.lineplot(data=sensitivity, x="label_tolerance", y="percentage", hue="metric", marker="o")
    plt.title("Oracle target-selection sensitivity to GT-label tolerance")
    plt.xticks(rotation=30, ha="right")
    save("oracle_sensitivity.png")
    _bar_categories(samples, "primary_failure_class", "Primary target-selection failure taxonomy", root / "failure_taxonomy_bar.png")
    paths.append(root / "failure_taxonomy_bar.png")
    for column, name, title in (
        ("query_type", "multiplicity_by_query_type.png", "Candidate multiplicity by query type"),
        ("mask_iou_bin", "multiplicity_by_mask_iou.png", "Candidate multiplicity by mask IoU"),
        ("target_size_bin", "multiplicity_by_target_size.png", "Candidate multiplicity by target size"),
    ):
        plt.figure(figsize=(9, 4.8))
        data = samples.copy()
        if column not in data:
            data[column] = "unavailable"
        sns.boxplot(data=data, x=column, y="n_official_candidates", showfliers=False)
        plt.title(title)
        plt.xticks(rotation=25, ha="right")
        save(name)
    for column, name, title in (
        ("query_type", "opportunity_by_query_type.png", "Re-selection opportunity by query type"),
        ("candidate_count_bin", "opportunity_by_candidate_count.png", "Re-selection opportunity by candidate multiplicity"),
        ("query_type", "taxonomy_by_query_type.png", "Primary taxonomy by query type"),
        ("mask_iou_bin", "taxonomy_by_mask_iou.png", "Primary taxonomy by mask IoU"),
    ):
        data = samples.copy()
        if column not in data:
            data[column] = "unavailable"
        normalized = pd.crosstab(data[column], data.opportunity_class if name.startswith("opportunity") else data.primary_failure_class, normalize="index") * 100
        normalized.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
        plt.title(title)
        plt.ylabel("Within-group samples (%)")
        plt.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=7)
        save(name)
    funnel_columns = [
        ("Manifest", np.ones(len(samples), dtype=bool)),
        ("Official exists", samples.has_official_candidate),
        ("Any GT-positive", samples.has_gt_positive_anywhere),
        ("GT-positive survives filter", samples.has_gt_positive_after_pred_filter),
        ("Current top-1 GT-positive", samples.hard_filter_top1_is_gt_positive),
    ]
    plt.figure(figsize=(9, 4.8))
    sns.barplot(x=[x[0] for x in funnel_columns], y=[int(np.asarray(x[1]).sum()) for x in funnel_columns], color="#3977b8")
    plt.title("Target-consistency opportunity funnel")
    plt.ylabel("Samples")
    plt.xticks(rotation=20, ha="right")
    save("analysis_funnel.png")
    # Interactive Sankey is intentionally standalone and local-data-only.
    transitions = pd.crosstab(samples.pred_status, samples.primary_failure_class)
    try:
        import plotly.graph_objects as go

        sources = list(transitions.index.astype(str))
        targets = list(transitions.columns.astype(str))
        labels = sources + targets
        source_index, target_index, values = [], [], []
        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                count = int(transitions.loc[source, target])
                if count:
                    source_index.append(i)
                    target_index.append(len(sources) + j)
                    values.append(count)
        figure = go.Figure(go.Sankey(node={"label": labels}, link={"source": source_index, "target": target_index, "value": values}))
        figure.update_layout(title="Existing run status → primary failure taxonomy")
        figure.write_html(root / "failure_taxonomy_sankey.html", include_plotlyjs="cdn")
    except ImportError:
        atomic_text(root / "failure_taxonomy_sankey.html", "<p>Plotly unavailable.</p>\n")
    paths.append(root / "failure_taxonomy_sankey.html")
    return paths


def candidate_quality_auc(candidates: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = candidates.gt_target_positive_primary.astype(int).to_numpy()
    score = candidates.vgn_quality.to_numpy(float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None, "reason": "only one GT label class"}
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "cluster_unit": "sample_id",
        "interpretation": "diagnostic only; VGN quality is not trained for target identity",
    }


def build_report(
    output: Path | str,
    *,
    integrity: Mapping[str, Any] | None = None,
    executive: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    root = Path(output)
    report_root = root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    if executive is None:
        executive_path = report_root / "executive_summary.json"
        executive = json.loads(executive_path.read_text(encoding="utf-8"))
    integrity = integrity or executive.get("integrity", {})
    metrics = executive.get("headline_metrics", {})
    recommendation = executive.get("method_design_recommendation", "not available")
    sample_count = int(metrics.get("sample_count") or 0)
    opportunity_counts = metrics.get("opportunity_counts", {})
    failure_counts = metrics.get("primary_failure_counts", {})
    oracle_rows = metrics.get("oracle_all_sample_rows", {})
    stage_counts = metrics.get("no_official_stage_flag_counts", {})

    def count_text(value: Any, denominator: int = sample_count) -> str:
        count = int(value or 0)
        return (
            f"{count:,} / {denominator:,} = {100.0 * count / denominator:.2f}%"
            if denominator
            else "null (zero denominator)"
        )

    def mapping_lines(values: Mapping[str, Any], *, denominator: int = sample_count) -> str:
        return "\n".join(
            f"- `{key}`: {count_text(value, denominator)}" for key, value in values.items()
        )
    lines = [
        "# Candidate multiplicity, re-selection opportunity, and oracle analysis",
        "",
        "## 1. Scope and definitions",
        "",
        "This report analyzes the frozen official VGN candidate pools. The primary label is whether the projected official grasp-pose origin lies inside the 3 px-dilated GT target mask. It is a target-consistency diagnostic, not physical grasp correctness. VGN quality is `official_vgn_processed_quality`, with no learned/custom re-ranking. Official references: [pinned CoRL 2020 source](https://github.com/ethz-asl/vgn/tree/d7af0622433f52ae88ebe81533f12b46b33e951a) and [VGN paper](https://proceedings.mlr.press/v155/breyer21a.html).",
        "",
        "## 2. Data integrity",
        "",
        f"- Manifest samples: {integrity.get('manifest_count')}",
        f"- Predicted official candidates: {integrity.get('predicted_candidate_count')}",
        f"- GT-regenerated official candidates: {integrity.get('gt_regenerated_candidate_count')}",
        f"- Frozen-run alignment mismatches: {integrity.get('top1_recomputed_mismatch_count', 0)}",
    ]
    section_titles = [
        "3. Candidate multiplicity",
        "4. Distinct pose modes",
        "5. Current baseline selections",
        "6. Re-ranking opportunity",
        "7. Same-pool GT oracle upper bounds",
        "8. GT-regenerated candidate-pool comparison",
        "9. Failure taxonomy",
        "10. Grouped analysis",
        "11. Representative cases",
        "12. Human-audit status",
        "13. Implications for method design",
        "14. Limitations",
        "15. Reproduction commands",
    ]
    body = {
        3: (
            f"Official candidate rows: **{int(metrics.get('candidate_count') or 0):,}**. "
            f"Pre-filter rankable: **{count_text(metrics.get('pre_filter_rankable'))}**. "
            f"Post-filter rankable: **{count_text(metrics.get('post_filter_rankable'))}**. "
            "See `tables/multiplicity_overall.csv` and `tables/multiplicity_distributions.csv`."
        ),
        4: (
            "Pose modes use camera-frame connected components at 10 mm translation, 15° SO(3) "
            "geodesic rotation, and 10 mm width difference; clustering does not alter the baseline. "
            f"Samples with two or more distinct modes: **{count_text((recommendation or {}).get('samples_with_two_or_more_distinct_modes') if isinstance(recommendation, Mapping) else 0)}**."
        ),
        5: (
            "Pure VGN top-1 is the highest official quality candidate. The modular baseline is the "
            "highest-quality candidate surviving the predicted-mask 3 px hard filter. Current "
            f"baseline GT target consistency: **{count_text(metrics.get('baseline_target_consistent'))}**."
        ),
        6: mapping_lines(opportunity_counts) + "\n\nRecall and quality-gap details are in `tables/recall_at_k.csv` and `tables/quality_margin.csv`.",
        7: "Same-pool pre/post-filter oracles select the highest-quality GT-positive candidate without regenerating candidates.\n\n" + mapping_lines({key: value.get("numerator") for key, value in oracle_rows.items() if key.startswith("same_pool") or key == "current_baseline"}),
        8: "The GT-mask run changes task frame/workspace/TSDF and is therefore a candidate-generation oracle, not a same-pool re-ranking bound.\n\n" + mapping_lines({key: value.get("numerator") for key, value in oracle_rows.items() if key in {"gt_regenerated_pool_oracle", "union_diagnostic_ceiling"}}),
        9: "Every sample receives exactly one P0–P6 primary class plus multi-label secondary diagnostic flags.\n\n" + mapping_lines(failure_counts) + "\n\nNo-official post-processing stage flags:\n" + (mapping_lines(stage_counts, denominator=int(metrics.get("no_official_stage_diagnostics_count") or 0)) if stage_counts else "- null: stage diagnostics were not requested or are unavailable"),
        10: "Grouped tables preserve query type, mask IoU, target/candidate size, scene, category, and geometry strata.",
        11: "Representative rows are in `tables/representative_samples.csv`; rendered review cases are in `human_audit/contact_sheets/`.",
        12: "Status: **manual_audit_pending**. No agreement or Cohen's kappa is fabricated before review.",
        13: "```json\n" + json.dumps(recommendation, indent=2, ensure_ascii=False) + "\n```",
        14: "Single-view TSDF adaptation; OCID-VLG has no 6-DoF grasp ground truth; no robot execution validation; the 2-D projected-origin label does not establish collision-free or executable grasps; regenerated and predicted pools differ geometrically.",
        15: """```bash
python -m scripts.analyze_vgn_candidates \\
  --pred-output outputs/hifics_vgn_full \\
  --gt-oracle-output outputs/hifics_vgn_gt_oracle_full \\
  --manifest runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask/manifest.jsonl \\
  --ocid-root ../crog_reproduction/OCID-VLG \\
  --output outputs/hifics_vgn_analysis \\
  --primary-gt-dilation-px 3 --bootstrap-replicates 10000 \\
  --cluster-key scene_id --seed 42 --diagnose-no-official \\
  --build-human-audit --render
python -m scripts.build_reranking_analysis_report \\
  --analysis-output outputs/hifics_vgn_analysis
```""",
    }
    for index, title in enumerate(section_titles, start=3):
        lines.extend(["", f"## {title}", "", str(body[index])])
    lines.extend(["", "## Headline machine-readable metrics", "", "```json", json.dumps(metrics, indent=2, ensure_ascii=False), "```", ""])
    markdown = "\n".join(lines)
    # The limitations may truthfully mention physical execution; metrics and
    # claims must never relabel target consistency as such.
    atomic_text(report_root / "report.md", markdown)
    figure_links = sorted((root / "figures").glob("*"))
    gallery = "\n".join(
        f'<figure><img src="../figures/{html.escape(path.name)}" style="max-width:900px"><figcaption>{html.escape(path.name)}</figcaption></figure>'
        for path in figure_links if path.suffix.lower() == ".png"
    )
    audit_links = sorted((root / "human_audit" / "contact_sheets").glob("*.png"))
    audit_gallery = "\n".join(
        f'<figure><img src="../human_audit/contact_sheets/{html.escape(path.name)}" '
        f'style="max-width:900px"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        for path in audit_links
    )
    html_report = "<!doctype html><meta charset='utf-8'><title>VGN candidate analysis</title><style>body{font-family:sans-serif;max-width:1100px;margin:auto;line-height:1.45}img{max-width:100%}code{background:#eee}</style>" + "<pre>" + html.escape(markdown) + "</pre>" + gallery
    atomic_text(report_root / "report.html", html_report)
    atomic_text(
        report_root / "gallery.html",
        "<!doctype html><meta charset='utf-8'><h1>Analysis figures</h1>"
        + gallery
        + "<h1>Human-audit contact sheets</h1>"
        + audit_gallery,
    )
    return report_root / "report.md", report_root / "report.html"


__all__ = [
    "atomic_json",
    "build_figures",
    "build_report",
    "build_tables",
    "candidate_quality_auc",
    "rate_record",
    "scene_cluster_rate_interval",
    "wilson_interval",
]
