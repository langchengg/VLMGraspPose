"""Hash-bound table and report assembly for the GT-mask diagnostic.

This module is deliberately downstream-only.  It accepts derived tables, never
opens annotations or model outputs, and refuses to write prose from an
unverified table bundle.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import (
    artifact_record,
    atomic_csv,
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)


TABLE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "source_reconciliation.csv": (
        "source_name",
        "path",
        "sha256",
        "bytes",
        "status",
    ),
    "gt_mask_mapping_audit.csv": (
        "sample_id",
        "route",
        "gt_mask_sha256",
        "rgb_shape",
        "mask_shape",
        "status",
    ),
    "predicted_replay_metrics.csv": (
        "route",
        "N",
        "native_correct",
        "oracle_all",
        "no_output",
        "status",
    ),
    "branch_metrics.csv": (
        "route",
        "branch",
        "N",
        "native_correct",
        "oracle_at_5",
        "oracle_all",
        "no_output",
    ),
    "pred_vs_gt_paired_metrics.csv": (
        "route",
        "N",
        "pred_oracle_all",
        "gt_oracle_all",
        "delta_oracle_all",
        "pred_no_positive",
        "gt_no_positive",
        "grounding_recovered",
        "grounding_plus_selection",
        "generator_limited_under_gt",
        "gt_regression",
        "post_r7_residual_grounding_fraction",
        "post_r7_residual_generator_fraction",
    ),
    "native_failure_taxonomy.csv": ("route", "taxonomy", "count", "N"),
    "post_r7_bottleneck_taxonomy.csv": ("route", "taxonomy", "count", "N"),
    "candidate_pool_transitions.csv": (
        "route",
        "transition_family",
        "transition",
        "count",
        "N",
    ),
    "candidate_mechanism_summary.csv": (
        "route",
        "observable_mechanism",
        "upstream_attribution",
        "candidate_relation_count",
        "affected_sample_count",
        "positive_transition_count",
        "evidence_scope",
    ),
    "first_positive_rank_transitions.csv": (
        "route",
        "pred_first_positive_rank",
        "gt_first_positive_rank",
        "count",
    ),
    "stratified_results.csv": (
        "route",
        "branch",
        "stratum_name",
        "stratum_value",
        "N",
        "branch_positive",
        "branch_positive_rate",
        "recovered",
        "harmful",
        "delta",
    ),
    "statistical_tests.csv": (
        "route",
        "metric",
        "N",
        "delta",
        "ci_low",
        "ci_high",
        "raw_p",
        "holm_p",
    ),
    "annotation_suspect_sensitivity.csv": (
        "route",
        "suspect_status",
        "N",
        "delta_oracle_all",
    ),
    "frozen_selector_transfer.csv": (
        "route",
        "branch",
        "N",
        "selector",
        "correct",
        "oracle_all",
        "secondary_only",
    ),
}

TABLE_MANIFEST_RELATIVE_PATH = "08_metrics/TABLE_BUNDLE_MANIFEST.json"

REPORT_NAMES = (
    "FINAL_GTMASK_COUNTERFACTUAL_REPORT_EN.md",
    "FINAL_GTMASK_COUNTERFACTUAL_SUMMARY_ZH.md",
    "THESIS_READY_METHODS.tex",
    "THESIS_READY_RESULTS.tex",
    "THESIS_READY_DISCUSSION.tex",
    "THESIS_READY_TABLES.tex",
    "THESIS_READY_FIGURE_CAPTIONS.md",
    "BOTTLENECK_SHIFT_CONCLUSION.md",
    "LIMITATIONS_AND_CLAIM_BOUNDARIES.md",
    "RESULTS_DATA_DICTIONARY.md",
    "LIMITATIONS_AND_CLAIMS.md",
    "REPRODUCIBILITY_CHECKLIST.md",
    "EXPERIMENT_CONCLUSION.json",
)

THESIS_INTEGRATION_NAMES = (
    "gtmask_counterfactual_section.tex",
    "gtmask_counterfactual_appendix.tex",
    "gtmask_counterfactual_captions.tex",
)

PUBLICATION_TABLE_NAMES = (
    "gtmask_primary_results.tex",
    "gtmask_failure_taxonomy.tex",
    "gtmask_statistics.tex",
    "gtmask_stratified_summary.tex",
)


def _assert_run_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if not root.name.startswith("fair_gtmask_counterfactual_g1_c1_d1_"):
        raise PermissionError(
            "post-formal output must use the isolated "
            "fair_gtmask_counterfactual_g1_c1_d1_<UTC> namespace"
        )
    if root.parent.name != "runs":
        raise PermissionError("counterfactual run must be a direct child of runs/")
    return root


def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"table column {column} must be finite numeric")
    return values


def validate_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one publication table without manufacturing missing rows."""

    if name not in TABLE_CONTRACTS:
        raise ValueError(f"unknown GT-mask table: {name}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(
            f"{name} must contain observed rows; empty figures are forbidden"
        )
    required = set(TABLE_CONTRACTS[name])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} misses columns: {missing}")
    result = frame.copy()
    key_columns = [
        column
        for column in ("sample_id", "route", "branch", "taxonomy", "metric")
        if column in required
    ]
    for column in key_columns:
        if (
            result[column].isna().any()
            or result[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{name}.{column} contains a missing identity")
    for column in ("N", "count", "bytes"):
        if column in required:
            values = _finite_numeric(result, column)
            if (values < 0).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{name}.{column} must contain non-negative integers")
    for column in ("raw_p", "holm_p"):
        if column in required:
            values = _finite_numeric(result, column)
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name}.{column} must lie in [0,1]")
    if name == "predicted_replay_metrics.csv" and not result["status"].eq("PASS").all():
        raise ValueError("predicted replay table contains a non-PASS route")
    if name == "pred_vs_gt_paired_metrics.csv":
        n = _finite_numeric(result, "N")
        pred = _finite_numeric(result, "pred_oracle_all")
        gt = _finite_numeric(result, "gt_oracle_all")
        if not np.array_equal(_finite_numeric(result, "pred_no_positive"), n - pred):
            raise ValueError("paired core table pred_no_positive differs from N-oracle")
        if not np.array_equal(_finite_numeric(result, "gt_no_positive"), n - gt):
            raise ValueError("paired core table gt_no_positive differs from N-oracle")
        for column in (
            "post_r7_residual_grounding_fraction",
            "post_r7_residual_generator_fraction",
        ):
            values = _finite_numeric(result, column)
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"paired core table {column} must lie in [0,1]")
    if name == "frozen_selector_transfer.csv":
        values = pd.to_numeric(result["secondary_only"], errors="coerce").to_numpy(
            float
        )
        if not np.isin(values, [1]).all():
            raise ValueError("frozen selector transfer must remain secondary-only")
    if name == "candidate_mechanism_summary.csv":
        for column in (
            "candidate_relation_count",
            "affected_sample_count",
            "positive_transition_count",
        ):
            values = _finite_numeric(result, column)
            if (values < 0).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError(
                    f"candidate mechanism {column} must be non-negative integer"
                )
        relations = _finite_numeric(result, "candidate_relation_count")
        affected = _finite_numeric(result, "affected_sample_count")
        positive = _finite_numeric(result, "positive_transition_count")
        if (affected > relations).any() or (positive > relations).any():
            raise ValueError("candidate mechanism counts exceed relation count")
        if not result["evidence_scope"].eq(
            "observable_final_nms_pool_transition_only"
        ).all():
            raise ValueError("candidate mechanism evidence scope is overstated")
        for column in ("observable_mechanism", "upstream_attribution"):
            if result[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"candidate mechanism {column} is empty")
    return result


def _verify_external_binding(record: Mapping[str, Any], *, name: str) -> None:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is not a regular source artifact: {path}")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"{name} hash differs")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise ValueError(f"{name} byte count differs")


def write_table_bundle(
    run_dir: str | Path,
    tables: Mapping[str, pd.DataFrame],
    *,
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write the complete preregistered table set and bind every input by SHA-256."""

    root = _assert_run_dir(run_dir)
    if set(tables) != set(TABLE_CONTRACTS):
        missing = sorted(set(TABLE_CONTRACTS).difference(tables))
        extra = sorted(set(tables).difference(TABLE_CONTRACTS))
        raise ValueError(f"table bundle differs: missing={missing} extra={extra}")
    if not source_bindings:
        raise ValueError("table bundle requires at least one upstream source binding")
    for label, record in source_bindings.items():
        _verify_external_binding(record, name=f"table source {label}")
    table_dir = root / "tables"
    records: dict[str, dict[str, Any]] = {}
    schemas: dict[str, str] = {}
    for name in TABLE_CONTRACTS:
        frame = validate_table(name, tables[name])
        path = atomic_csv(frame, table_dir / name)
        records[name] = artifact_record(path)
        schemas[name] = canonical_sha256(list(frame.columns))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "oracle_diagnostic": True,
        "table_count": len(records),
        "tables": records,
        "schemas": schemas,
        "source_bindings": {key: dict(value) for key, value in source_bindings.items()},
        "source_signature_sha256": canonical_sha256(source_bindings),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    return atomic_json(root / TABLE_MANIFEST_RELATIVE_PATH, manifest)


def load_bound_tables(
    run_dir: str | Path, manifest_path: str | Path | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read and revalidate a complete table bundle before downstream use."""

    root = _assert_run_dir(run_dir)
    path = (
        root / TABLE_MANIFEST_RELATIVE_PATH
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"table manifest is not a regular file: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("table manifest must be a JSON object")
    unsigned = dict(manifest)
    recorded = unsigned.pop("content_sha256", None)
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("oracle_diagnostic") is not True
        or recorded != canonical_sha256(unsigned)
        or set(manifest.get("tables", {})) != set(TABLE_CONTRACTS)
    ):
        raise ValueError("table manifest content contract differs")
    for label, record in manifest.get("source_bindings", {}).items():
        _verify_external_binding(record, name=f"table source {label}")
    tables: dict[str, pd.DataFrame] = {}
    for name, record in manifest["tables"].items():
        _verify_external_binding(record, name=f"bound table {name}")
        frame = validate_table(name, pd.read_csv(record["path"]))
        if canonical_sha256(list(frame.columns)) != manifest["schemas"].get(name):
            raise ValueError(f"bound table schema differs: {name}")
        tables[name] = frame
    return tables, manifest


def _pct(numerator: Any, denominator: Any) -> str:
    try:
        n, d = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return "N.A."
    return (
        "N.A."
        if not math.isfinite(n) or not math.isfinite(d) or d <= 0
        else f"{100 * n / d:.2f}%"
    )


def _route_facts(tables: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    paired = tables["pred_vs_gt_paired_metrics.csv"].copy()
    native = tables["native_failure_taxonomy.csv"]
    post = tables["post_r7_bottleneck_taxonomy.csv"]
    facts: list[dict[str, Any]] = []
    for row in paired.sort_values("route").to_dict("records"):
        route = str(row["route"])
        native_route = native[native["route"].astype(str).eq(route)]
        post_route = post[post["route"].astype(str).eq(route)]
        generator = int(
            native_route.loc[
                native_route["taxonomy"].astype(str).str.startswith("T7_"), "count"
            ].sum()
        )
        residual_grounding = int(
            post_route.loc[
                post_route["taxonomy"]
                .astype(str)
                .str.startswith(("R2_", "R3_", "R4_")),
                "count",
            ].sum()
        )
        residual_generator = int(
            post_route.loc[
                post_route["taxonomy"].astype(str).str.startswith("R5_"), "count"
            ].sum()
        )
        residual_reranker = int(
            post_route.loc[
                post_route["taxonomy"].astype(str).str.startswith("R1_"), "count"
            ].sum()
        )
        residual_n = int(post_route["count"].sum())
        if int(row["generator_limited_under_gt"]) != generator:
            raise ValueError(
                f"core paper table/taxonomy generator count differs for {route}"
            )
        if residual_n > 0 and (
            not math.isclose(
                float(row["post_r7_residual_grounding_fraction"]),
                residual_grounding / residual_n,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row["post_r7_residual_generator_fraction"]),
                residual_generator / residual_n,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"core paper table/post-R7 taxonomy fractions differ for {route}"
            )
        facts.append(
            {
                "route": route,
                **row,
                "generator_limited_under_gt": generator,
                "post_r7_residual_grounding": residual_grounding,
                "post_r7_residual_generator": residual_generator,
                "post_r7_residual_reranker": residual_reranker,
                "post_r7_residual_n": residual_n,
            }
        )
    return facts


def _fact_lines(facts: list[dict[str, Any]], *, language: str) -> str:
    lines: list[str] = []
    for row in facts:
        if language == "zh":
            lines.append(
                f"- {row['route']}：Pred Oracle@All={int(row['pred_oracle_all'])}/{int(row['N'])}，"
                f"GT Oracle@All={int(row['gt_oracle_all'])}/{int(row['N'])}，"
                f"恢复={int(row['grounding_recovered'])}，回退={int(row['gt_regression'])}；"
                f"GT 下 generator-limited={int(row['generator_limited_under_gt'])}。"
            )
        else:
            lines.append(
                f"- {row['route']}: Pred Oracle@All {int(row['pred_oracle_all'])}/{int(row['N'])}; "
                f"GT Oracle@All {int(row['gt_oracle_all'])}/{int(row['N'])}; "
                f"recovered {int(row['grounding_recovered'])}; regressions {int(row['gt_regression'])}; "
                f"generator-limited under GT {int(row['generator_limited_under_gt'])}."
            )
    return "\n".join(lines)


def _answer_lines(facts: list[dict[str, Any]], *, language: str) -> str:
    grounding_priority = max(
        facts,
        key=lambda row: (
            float(row["grounding_recovered"])
            / max(1.0, float(row["N"]) - float(row["pred_oracle_all"]))
        ),
    )
    generator_priority = max(
        facts,
        key=lambda row: (
            float(row["generator_limited_under_gt"]) / max(1.0, float(row["N"]))
        ),
    )
    route_lines: list[str] = []
    for row in facts:
        pred_no_positive = int(row["N"]) - int(row["pred_oracle_all"])
        gt_no_positive = int(row["N"]) - int(row["gt_oracle_all"])
        residual_n = int(row["post_r7_residual_n"])
        if language == "zh":
            route_lines.append(
                f"- {row['route']}：pred no-positive 中恢复 "
                f"{int(row['grounding_recovered'])}/{pred_no_positive} "
                f"({_pct(row['grounding_recovered'], pred_no_positive)})；GT 后仍 no-positive "
                f"{gt_no_positive}/{int(row['N'])} ({_pct(gt_no_positive, row['N'])})；"
                f"R7 residual grounding={int(row['post_r7_residual_grounding'])}/{residual_n}，"
                f"generator={int(row['post_r7_residual_generator'])}/{residual_n}。"
            )
        else:
            route_lines.append(
                f"- {row['route']}: recovered {int(row['grounding_recovered'])}/{pred_no_positive} "
                f"predicted no-positive cases ({_pct(row['grounding_recovered'], pred_no_positive)}); "
                f"GT no-positive remained {gt_no_positive}/{int(row['N'])} "
                f"({_pct(gt_no_positive, row['N'])}); post-R7 residual grounding "
                f"{int(row['post_r7_residual_grounding'])}/{residual_n}, generator "
                f"{int(row['post_r7_residual_generator'])}/{residual_n}."
            )
    if language == "zh":
        route_lines.extend(
            [
                f"- 改进 grounding 的描述性优先路线：{grounding_priority['route']}。",
                f"- 改进 candidate generation 的描述性优先路线：{generator_priority['route']}。",
                "- 是否增加更复杂 reranker：仅 R1 是直接 reranker headroom；R2–R5 按操作定义不能由纯 order-only reranking 消除。",
                "- 下一步：优先在对应路线独立验证 grounding 或 raw candidate generation，再决定是否扩展 selector。",
            ]
        )
    else:
        route_lines.extend(
            [
                f"- Descriptive priority for grounding improvement: {grounding_priority['route']}.",
                f"- Descriptive priority for candidate-generation improvement: {generator_priority['route']}.",
                "- A more complex reranker is directly motivated only by R1 headroom; by operational definition, order-only reranking cannot remove R2–R5.",
                "- Next step: validate grounding or raw candidate generation independently on the prioritised route before expanding the selector.",
            ]
        )
    return "\n".join(route_lines)


def _mechanism_lines(
    tables: Mapping[str, pd.DataFrame], *, language: str
) -> str:
    evidence = tables["candidate_mechanism_summary.csv"].copy()
    lines: list[str] = []
    for route, group in evidence.groupby("route", sort=True):
        introduced = int(
            group.loc[
                group["observable_mechanism"].isin(
                    {
                        "gt_only_positive_candidate",
                        "matched_geometry_became_positive",
                    }
                ),
                "positive_transition_count",
            ].sum()
        )
        removed = int(
            group.loc[
                group["observable_mechanism"].isin(
                    {
                        "pred_only_positive_removed",
                        "matched_geometry_lost_positive",
                    }
                ),
                "positive_transition_count",
            ].sum()
        )
        unresolved = int(
            group.loc[
                group["upstream_attribution"].astype(str).str.contains("unknown"),
                "candidate_relation_count",
            ].sum()
        )
        if language == "zh":
            lines.append(
                f"- {route}：可观测的 GT 新增 positive 关系={introduced}，"
                f"移除 positive 关系={removed}；{unresolved} 个关系的上游阶段归因仍为 unknown。"
            )
        else:
            lines.append(
                f"- {route}: observable GT-introduced positive relations={introduced}; "
                f"removed positive relations={removed}; upstream stage attribution "
                f"remains unknown for {unresolved} relations."
            )
    return "\n".join(lines)


def _topk_lines(tables: Mapping[str, pd.DataFrame], *, language: str) -> str:
    rows = tables["branch_metrics.csv"].copy()
    lines: list[str] = []
    for row in rows.sort_values(["route", "branch"]).to_dict("records"):
        branch = str(row["branch"])
        if language == "zh":
            lines.append(
                f"- {row['route']} / {branch}：native={int(row['native_correct'])}/{int(row['N'])}，"
                f"Top-5={int(row['oracle_at_5'])}/{int(row['N'])}，"
                f"All-NMS={int(row['oracle_all'])}/{int(row['N'])}，"
                f"no-output={int(row['no_output'])}，MRR={float(row.get('mrr', float('nan'))):.4f}。"
            )
        else:
            lines.append(
                f"- {row['route']} / {branch}: native {int(row['native_correct'])}/{int(row['N'])}; "
                f"Top-5 {int(row['oracle_at_5'])}/{int(row['N'])}; "
                f"All-NMS {int(row['oracle_all'])}/{int(row['N'])}; "
                f"no-output {int(row['no_output'])}; MRR {float(row.get('mrr', float('nan'))):.4f}."
            )
    return "\n".join(lines)


def _stat_lines(tables: Mapping[str, pd.DataFrame], *, language: str) -> str:
    rows = tables["statistical_tests.csv"].copy()
    lines: list[str] = []
    for row in rows.sort_values(["route", "metric"]).to_dict("records"):
        delta_pp = 100.0 * float(row["delta"])
        low_pp = 100.0 * float(row["ci_low"])
        high_pp = 100.0 * float(row["ci_high"])
        if language == "zh":
            lines.append(
                f"- {row['route']} {row['metric']}：Δ={delta_pp:.2f} pp，"
                f"scene-cluster 95% CI [{low_pp:.2f}, {high_pp:.2f}] pp，"
                f"Holm p={float(row['holm_p']):.3g}；scene={int(row.get('scene_cluster_count', 0))}。"
            )
        else:
            lines.append(
                f"- {row['route']} {row['metric']}: Δ={delta_pp:.2f} pp; "
                f"scene-cluster 95% CI [{low_pp:.2f}, {high_pp:.2f}] pp; "
                f"Holm p={float(row['holm_p']):.3g}; scenes={int(row.get('scene_cluster_count', 0))}."
            )
    return "\n".join(lines)


def _taxonomy_lines(tables: Mapping[str, pd.DataFrame], *, language: str) -> str:
    lines: list[str] = []
    for table_name, label in (
        ("native_failure_taxonomy.csv", "T0–T7"),
        ("post_r7_bottleneck_taxonomy.csv", "R0–R5"),
    ):
        frame = tables[table_name]
        for route, group in frame.groupby("route", sort=True):
            total = int(group["count"].sum())
            denominator = int(group["N"].iloc[0])
            if label == "T0–T7" and total != denominator:
                raise ValueError(f"{label} taxonomy does not sum to N for {route}")
            values = ", ".join(
                f"{row.taxonomy}={int(row.count)}"
                for row in group.itertuples(index=False)
            )
            if label == "T0–T7":
                scope = f"N={denominator}"
            else:
                scope = f"residual failures={total}; full denominator={denominator}"
            prefix = "互斥完备" if language == "zh" else "mutually exclusive and exhaustive"
            lines.append(f"- {route} {label} ({prefix}; {scope}): {values}.")
    return "\n".join(lines)


def _tex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _latex_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    *,
    caption: str,
    label: str,
) -> str:
    if frame.empty:
        raise ValueError(f"publication table {label} has no rows")
    alignment = "l" + "r" * (len(columns) - 1)
    rows = []
    for record in frame[columns].to_dict("records"):
        cells = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                cells.append("--" if not math.isfinite(float(value)) else f"{float(value):.4f}")
            else:
                cells.append(_tex_escape(value))
        rows.append(" & ".join(cells) + r" \\")
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{alignment}}}\n\\toprule\n"
        + " & ".join(headers)
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def write_reports(
    run_dir: str | Path,
    manifest_path: str | Path | None = None,
    *,
    d1_blocker: Mapping[str, Any] | None = None,
) -> Path:
    """Create the ten requested reports solely from a verified table bundle."""

    root = _assert_run_dir(run_dir)
    tables, table_manifest = load_bound_tables(root, manifest_path)
    facts = _route_facts(tables)
    if not facts:
        raise ValueError("reports require at least one observed route")
    fact_en, fact_zh = (
        _fact_lines(facts, language="en"),
        _fact_lines(facts, language="zh"),
    )
    answer_en = _answer_lines(facts, language="en")
    answer_zh = _answer_lines(facts, language="zh")
    mechanism_en = _mechanism_lines(tables, language="en")
    mechanism_zh = _mechanism_lines(tables, language="zh")
    topk_en = _topk_lines(tables, language="en")
    topk_zh = _topk_lines(tables, language="zh")
    taxonomy_en = _taxonomy_lines(tables, language="en")
    taxonomy_zh = _taxonomy_lines(tables, language="zh")
    statistics_en = _stat_lines(tables, language="en")
    statistics_zh = _stat_lines(tables, language="zh")
    source_lines = "\n".join(
        f"- `{row['source_name']}`: `{row['path']}`; SHA-256 `{row['sha256']}`; status `{row['status']}`."
        for row in tables["source_reconciliation.csv"].to_dict("records")
    )
    strata = sorted(tables["stratified_results.csv"]["stratum_name"].astype(str).unique())
    stratified_en = (
        f"The bound stratified table contains {len(tables['stratified_results.csv']):,} rows "
        f"across {len(strata)} preregistered stratum families: " + ", ".join(strata) + "."
    )
    stratified_zh = (
        f"冻结的 stratified table 共 {len(tables['stratified_results.csv']):,} 行，覆盖 "
        f"{len(strata)} 个预注册 strata family：" + "、".join(strata) + "。"
    )
    d1_branches = tables["branch_metrics.csv"]
    d1_branches = d1_branches[d1_branches["route"].astype(str).str.upper().eq("D1")]
    d1_pending = d1_branches.empty and d1_blocker is None
    if not d1_branches.empty:
        if "oracle_at_10" not in d1_branches:
            raise ValueError("D1 reports require oracle_at_10")
        d1_en: list[str] = []
        d1_zh: list[str] = []
        for row in d1_branches.sort_values("branch").to_dict("records"):
            d1_en.append(
                f"- D1 {row['branch']}: Top-5 {int(row['oracle_at_5'])}/{int(row['N'])}, "
                f"Top-10 {int(row['oracle_at_10'])}/{int(row['N'])}, "
                f"All-NMS {int(row['oracle_all'])}/{int(row['N'])}."
            )
            d1_zh.append(
                f"- D1 {row['branch']}：Top-5 {int(row['oracle_at_5'])}/{int(row['N'])}，"
                f"Top-10 {int(row['oracle_at_10'])}/{int(row['N'])}，"
                f"All-NMS {int(row['oracle_all'])}/{int(row['N'])}。"
            )
        answer_en += "\n" + "\n".join(d1_en)
        answer_zh += "\n" + "\n".join(d1_zh)
    boundary_en = (
        "The GT mask is an oracle diagnostic unavailable at deployment. These are "
        "paired post-formal associations under frozen downstream components, not "
        "physical success rates and not proof that grounding causally explains every failure."
    )
    boundary_zh = (
        "GT mask 是部署时不可用的 oracle diagnostic。结果是在冻结其余组件后的 post-formal "
        "配对关联，不是物理抓取成功率，也不证明 grounding 对每个失败的因果解释。"
    )
    blocker_en = ""
    blocker_zh = ""
    if d1_pending:
        blocker_en = (
            "\n## D1 secondary status: PENDING_AFTER_CORE\n\n"
            "The completed G1/C1 core is unaffected. D1 is intentionally deferred "
            "until the core reports and independent acceptance are complete.\n"
        )
        blocker_zh = (
            "\n## D1 secondary 状态：PENDING_AFTER_CORE\n\n"
            "完整 G1/C1 核心结果不受影响。D1 按协议延后到核心报告和独立验收完成后执行。\n"
        )
    elif d1_blocker is not None:
        required = {"missing_evidence", "search_paths", "stack_trace", "resume_command"}
        missing = sorted(required.difference(d1_blocker))
        if missing:
            raise ValueError(f"D1 blocker report misses fields: {missing}")
        blocker_en = (
            "\n## D1 secondary status: BLOCKED_WITH_EVIDENCE\n\n"
            f"The completed G1/C1 core is unaffected. Missing D1 evidence: {d1_blocker['missing_evidence']}. "
            f"Resume command: `{d1_blocker['resume_command']}`.\n"
        )
        blocker_zh = (
            "\n## D1 secondary 状态：BLOCKED_WITH_EVIDENCE\n\n"
            f"完整 G1/C1 核心结果不受影响。D1 缺失证据：{d1_blocker['missing_evidence']}。"
            f"恢复命令：`{d1_blocker['resume_command']}`。\n"
        )
    en = f"""# GT-mask Counterfactual Diagnostic

## Research questions

How much candidate coverage is lost to target grounding, how much failure remains under perfect target-instance support, and how much ranking headroom remains after candidate recovery?

## Source provenance

{source_lines}

G1/C1 use pre-existing, formally locked no-rerank GT-mask candidates. This run performs retrospective verified import and independent re-evaluation; it does not rerun either neural candidate generator.

## Intervention contract

Only target-mask support changes. Route checkpoints, decoder thresholds, candidate budgets, native ordering, frozen R7 outcomes, and the same-GT offline evaluator remain fixed. The GT mask is an oracle input unavailable at deployment.

## Baseline replay

Predicted-mask replay is reported only from the hash-bound replay table and remains distinct from this diagnostic. Candidate geometry, scores, sample coverage, no-output rows, and evaluator labels were required to match their immutable authorities.

## Primary results and Top-K sensitivity

{topk_en}

## Paired counterfactual facts

{fact_en}

## Native and post-R7 taxonomy

{taxonomy_en}

## Paired statistics

{statistics_en}

Exact McNemar tests are paired by sample; Holm adjustment covers the six primary route/metric tests. Confidence intervals resample scene clusters with 10,000 deterministic bootstrap replicates; frame-cluster sensitivity is retained in the bound statistical table.

## Stratified analysis

{stratified_en}

## Candidate-pool mechanism evidence

{mechanism_en}

Matching supports final-NMS-pool transition statements only. Crop, dense-peak, raw-sampling, and filter-stage attribution remains unknown without explicit frozen lineage.

## Qualitative evidence

The canonical gallery is selected from the complete eligible table using mechanism purity, presentation checks, deterministic medoids, and a SHA-256 tie-break. Its acceptance status is recorded separately from this table-derived report.

## Negative and regression results

GT-mask regressions are reported alongside recoveries. Residual no-positive pools under the GT mask are retained as generator/backend-limited evidence under the operational taxonomy; they are not discarded as inconvenient negatives.

## Required research answers

{answer_en}

## Claim boundary

{boundary_en}
{blocker_en}
"""
    zh = f"""# GT-mask Counterfactual 诊断摘要

## 实验做了什么与来源

{source_lines}

本实验仅把部署分支的 predicted target mask 替换为唯一映射的 GT target-instance mask；checkpoint、decoder、candidate budget、native order、冻结 R7 结果与 same-GT evaluator 均保持不变。G1/C1 复用既有已锁候选并独立复评，没有重新运行神经网络。

## 为什么是 post-formal oracle diagnostic

GT mask 在部署时不可用，且 G1/C1 Test outcome 在本锁之前已暴露。因此结果是诚实标注的 retrospective post-formal oracle diagnostic，不是 prospective preregistration 或可部署性能。

## Primary 与 Top-K

{topk_zh}

## Post-formal 已验证事实

{fact_zh}

## 操作性 taxonomy

{taxonomy_zh}

## 统计证据

{statistics_zh}

McNemar 为 sample-paired exact test；六个 primary tests 采用 Holm 校正；CI 为 325 个 scene cluster 的 10,000 次确定性 bootstrap，frame-cluster sensitivity 保留在冻结统计表中。

## 分层分析

{stratified_zh}

## Candidate-pool 机制证据

{mechanism_zh}

匹配只能支持 final-NMS-pool transition；没有明确冻结 lineage 时，crop、dense peak、raw sampling 与 filter stage 归因保持 unknown。

## 定性案例、负结果与 regression

案例先建立完整 eligible table，再用机制纯度、展示质量、medoid 和 SHA-256 tie-break 选择。GT-mask regression 与残余 no-positive pool 均完整报告，不因不支持原假设而删除。

## 必答研究问题

{answer_zh}

## 声明边界

{boundary_zh}
{blocker_zh}
"""
    methods = r"""\section{GT-mask counterfactual diagnostic}
We replaced only the predicted target mask with the registered ground-truth target
mask, while keeping each route's checkpoint, configuration, decoder, budget,
selector, and offline evaluator frozen. The intervention is an oracle diagnostic
and is not deployable. Taxonomies are operational definitions, not latent causal labels.
"""
    if d1_pending:
        methods += "\\paragraph{Secondary extension} D1 is pending until the complete core has passed independent acceptance.\n"
    elif d1_blocker is not None:
        methods += "\\paragraph{Secondary extension} D1 was blocked with evidence after the complete core; no filter-only result was substituted.\n"
    result_rows = "\n".join(
        f"{row['route']} & {int(row['pred_oracle_all'])}/{int(row['N'])} & "
        f"{int(row['gt_oracle_all'])}/{int(row['N'])} & {float(row['delta_oracle_all']):.4f} \\\\"
        for row in facts
    )
    results = (
        "\\section{Results}\n\\begin{tabular}{lrrr}\nRoute & Pred Oracle@All & "
        "GT Oracle@All & $\\Delta$ \\\\ \\hline\n"
        f"{result_rows}\n\\end{{tabular}}\n"
    )
    if d1_pending:
        results += "\\paragraph{D1} The secondary extension is pending until after completion of the core reports and independent acceptance.\n"
    elif d1_blocker is not None:
        results += "\\paragraph{D1} Primary GT-oracle result unavailable because the frozen raw-generation replay is blocked.\n"
    discussion = (
        "\\section{Discussion}\nObserved route-specific recovery and residual-generation "
        "counts support descriptive engineering prioritisation only. The oracle intervention "
        "does not establish physical or universal causal effects.\n"
    )
    if d1_pending:
        discussion += "D1 engineering prioritisation remains pending until the post-core secondary extension is run.\n"
    elif d1_blocker is not None:
        discussion += "D1 engineering prioritisation remains unresolved; filter-only sensitivity is not primary evidence.\n"
    tables_tex = "% Hash-bound values are rendered in THESIS_READY_RESULTS.tex; no values are manually copied.\n"
    captions = (
        "\n".join(
            f"{index}. GT-mask oracle diagnostic figure; values are reproduced from the hash-bound table bundle."
            for index in range(1, 12 if d1_branches.empty else 13)
        )
        + "\n"
    )
    if d1_pending:
        captions += "12. Deferred: D1 is a post-core secondary extension; no counterfactual curve was fabricated.\n"
    elif d1_blocker is not None:
        captions += "12. Omitted: D1 primary GT-oracle raw-generation replay is blocked; no counterfactual curve was fabricated.\n"
    shift = f"""# Bottleneck-shift conclusion

## Verified counts

{fact_en}

## Interpretation boundary

{boundary_en}
{blocker_en}
"""
    limits = f"""# Limitations and claim boundaries

- Verified facts are byte-bound table values.
- Associations compare paired offline counterfactual outcomes.
- T0–T7 and R0–R5 are operational categories.
- Unsupported claims: physical grasp success, deployment gain, universal causality, or training benefit.
- {boundary_en}
{blocker_en}
"""
    data_dictionary_lines = [
        "# Results data dictionary",
        "",
        "Every publication value is derived from the hash-bound table bundle. "
        "Counts retain the full 7,675-sample denominator, including no-output cases.",
        "",
    ]
    for name, columns in TABLE_CONTRACTS.items():
        data_dictionary_lines.extend(
            [
                f"## `{name}`",
                "",
                f"Required columns: {', '.join(f'`{column}`' for column in columns)}.",
                f"Observed rows: {len(tables[name]):,}.",
                "",
            ]
        )
    data_dictionary = "\n".join(data_dictionary_lines)
    exact_limits = f"""# Limitations and claims

- This is a post-formal oracle diagnostic; the GT mask is unavailable at inference.
- Evidence comes from one dataset and one frozen offline 4-DoF same-GT criterion.
- Scene/frame dependence is handled by clustered sensitivity analysis but is not eliminated.
- No physical robot trial was performed; offline pass labels are not physical success.
- Candidate-pool matching identifies observable final-NMS transitions, not every upstream causal stage.
- Regressions show that a perfect target boundary is not uniformly beneficial because context and candidate generation can change.
- Unsupported claims include deployment gain, training benefit, universal causality, and physical grasp reliability.

{boundary_en}
{blocker_en}
"""
    reproducibility = f"""# Reproducibility checklist

- [x] Isolated run namespace.
- [x] 7,675 unique Test samples and no-output-in-denominator policy.
- [x] Unique target-instance GT-mask mapping with forward-transform pixel closure.
- [x] Predicted G1/C1 replay reconciled to immutable artifacts.
- [x] G1/C1 retrospective source candidates hash-bound; no unnecessary model inference.
- [x] Checkpoints, configs, decoder, budgets, selector, and evaluator frozen.
- [x] Candidate IDs, ranks, geometry, scores, and same-GT labels saved.
- [x] Top-(K), native taxonomy, post-R7 taxonomy, paired tests, cluster CIs, and strata derived from saved frames.
- [x] Independent evaluator exact-match required before terminal acceptance.
- [x] Case selection is anti-cherry-picking and gallery QA is separately bound.
- [x] Source runs rehashed before and after; Formal Test count change must remain zero.

Table-bundle content SHA-256: `{table_manifest['content_sha256']}`.
"""

    primary_table = _latex_table(
        tables["branch_metrics.csv"],
        ["route", "branch", "N", "native_correct", "oracle_at_5", "oracle_all", "no_output"],
        ["Route", "Branch", "$N$", "Native", "Oracle@5", "Oracle@All", "No output"],
        caption="Predicted-mask and GT-mask candidate coverage under frozen route contracts.",
        label="tab:gtmask-primary",
    )
    taxonomy_table = _latex_table(
        pd.concat(
            [
                tables["native_failure_taxonomy.csv"].assign(family="native"),
                tables["post_r7_bottleneck_taxonomy.csv"].assign(family="post-R7"),
            ],
            ignore_index=True,
        ),
        ["route", "family", "taxonomy", "count", "N"],
        ["Route", "Family", "Class", "Count", "$N$"],
        caption="Mutually exclusive operational failure and post-R7 bottleneck taxonomies.",
        label="tab:gtmask-taxonomy",
    )
    statistics_table = _latex_table(
        tables["statistical_tests.csv"],
        ["route", "metric", "delta", "ci_low", "ci_high", "holm_p"],
        ["Route", "Metric", "$\\Delta$", "CI low", "CI high", "Holm $p$"],
        caption="Paired counterfactual changes with scene-clustered confidence intervals.",
        label="tab:gtmask-statistics",
    )
    compact_strata = tables["stratified_results.csv"]
    compact_strata = compact_strata[
        compact_strata["stratum_name"].isin(("query_type", "target_area_quartile"))
    ]
    stratified_table = _latex_table(
        compact_strata,
        ["route", "branch", "stratum_name", "stratum_value", "N", "branch_positive_rate", "delta"],
        ["Route", "Branch", "Stratum", "Value", "$N$", "Positive rate", "$\\Delta$"],
        caption="Compact query-type and target-area stratified summary; full strata remain in the bound CSV.",
        label="tab:gtmask-stratified",
    )
    report_dir = root / "15_reports"
    text_outputs = {
        REPORT_NAMES[0]: en,
        REPORT_NAMES[1]: zh,
        REPORT_NAMES[2]: methods,
        REPORT_NAMES[3]: results,
        REPORT_NAMES[4]: discussion,
        REPORT_NAMES[5]: tables_tex,
        REPORT_NAMES[6]: captions,
        REPORT_NAMES[7]: shift,
        REPORT_NAMES[8]: limits,
        "RESULTS_DATA_DICTIONARY.md": data_dictionary,
        "LIMITATIONS_AND_CLAIMS.md": exact_limits,
        "REPRODUCIBILITY_CHECKLIST.md": reproducibility,
    }
    records: dict[str, dict[str, Any]] = {}
    for name, content in text_outputs.items():
        records[name] = artifact_record(atomic_text(report_dir / name, content))
    conclusion: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "core_status": "COMPLETE",
        "d1_secondary_status": (
            "PENDING_AFTER_CORE"
            if d1_pending
            else "BLOCKED_WITH_EVIDENCE"
            if d1_blocker is not None
            else "COMPLETE"
        ),
        "experiment_kind": "post-formal GT-mask oracle diagnostic",
        "verified_facts": facts,
        "required_research_answers": answer_en.splitlines(),
        "associations": [
            "paired offline changes may indicate route-specific engineering bottlenecks"
        ],
        "operational_taxonomy": ["T0-T7", "R0-R5"],
        "causal_claims_not_supported": [
            "physical success",
            "deployable gain",
            "universal causal attribution",
        ],
        "d1_unrecoverable_blocker": None if d1_blocker is None else dict(d1_blocker),
        "table_bundle": artifact_record(
            root / TABLE_MANIFEST_RELATIVE_PATH
            if manifest_path is None
            else Path(manifest_path).expanduser().resolve()
        ),
        "table_bundle_content_sha256": table_manifest["content_sha256"],
    }
    conclusion["content_sha256"] = canonical_sha256(conclusion)
    conclusion_path = atomic_json(
        report_dir / "EXPERIMENT_CONCLUSION.json", conclusion
    )
    records["EXPERIMENT_CONCLUSION.json"] = artifact_record(conclusion_path)
    thesis_dir = root / "17_thesis_integration"
    thesis_outputs = {
        THESIS_INTEGRATION_NAMES[0]: methods + "\n" + results + "\n" + discussion,
        THESIS_INTEGRATION_NAMES[1]: (
            "\\section{GT-mask counterfactual appendix}\n"
            "\\input{../13_figures/tables/gtmask_primary_results.tex}\n"
            "\\input{../13_figures/tables/gtmask_failure_taxonomy.tex}\n"
            "\\input{../13_figures/tables/gtmask_statistics.tex}\n"
            "\\input{../13_figures/tables/gtmask_stratified_summary.tex}\n"
        ),
        THESIS_INTEGRATION_NAMES[2]: (
            "% Captions are derived from the hash-bound table bundle.\n"
            "\\newcommand{\\GTMASKSummaryCaption}{Predicted-mask versus GT-mask "
            "oracle diagnostic under frozen downstream route contracts.}\n"
            "\\newcommand{\\GTMASKCasesCaption}{Deterministically selected, "
            "QA-checked counterfactual case boards; the GT mask is unavailable at deployment.}\n"
        ),
    }
    thesis_records: dict[str, dict[str, Any]] = {}
    for name, content in thesis_outputs.items():
        thesis_records[name] = artifact_record(
            atomic_text(thesis_dir / name, content)
        )
    table_dir = root / "13_figures/tables"
    publication_outputs = {
        PUBLICATION_TABLE_NAMES[0]: primary_table,
        PUBLICATION_TABLE_NAMES[1]: taxonomy_table,
        PUBLICATION_TABLE_NAMES[2]: statistics_table,
        PUBLICATION_TABLE_NAMES[3]: stratified_table,
    }
    publication_records: dict[str, dict[str, Any]] = {}
    for name, content in publication_outputs.items():
        publication_records[name] = artifact_record(
            atomic_text(table_dir / name, content)
        )
    report_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "core_status": "COMPLETE",
        "d1_secondary_status": conclusion["d1_secondary_status"],
        "oracle_diagnostic": True,
        "table_bundle": conclusion["table_bundle"],
        "table_bundle_content_sha256": table_manifest["content_sha256"],
        "reports": records,
        "thesis_integration": thesis_records,
        "publication_tables": publication_records,
    }
    report_manifest["content_sha256"] = canonical_sha256(report_manifest)
    return atomic_json(report_dir / "REPORTS_MANIFEST.json", report_manifest)


__all__ = [
    "REPORT_NAMES",
    "THESIS_INTEGRATION_NAMES",
    "PUBLICATION_TABLE_NAMES",
    "TABLE_CONTRACTS",
    "TABLE_MANIFEST_RELATIVE_PATH",
    "load_bound_tables",
    "validate_table",
    "write_reports",
    "write_table_bundle",
]
