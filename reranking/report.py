"""End-to-end, post-lock reporting stage for the re-ranking matrix."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from reranking.data_contracts import streaming_sha256
from reranking.matrix_reporting import (
    ReportingContractError,
    compute_reporting_statistics,
    load_reporting_inputs,
    multi_seed_summary,
    reliability_bins,
)
from reranking.visualize import FIGURE_NAMES, build_galleries, build_visualizations


CONSISTENCY_EN = (
    "All success values in this report are frozen 2D grasp-rectangle consistency "
    "under the registered evaluator. They are not measurements of physical grasp "
    "success, force closure, collision freedom, or robot execution reliability."
)
CONSISTENCY_ZH = (
    "本报告全部成功率均指注册 evaluator 下的冻结二维抓取矩形一致性；它们不是物理抓取成功率，"
    "也不代表力闭合、无碰撞或真实机器人执行可靠性。"
)


REPORT_NAMES = (
    "FINAL_REPORT_ZH.md",
    "FINAL_REPORT_EN.md",
    "DISSERTATION_TABLES.md",
    "FAILURE_ANALYSIS.md",
    "REPRODUCTION_COMMANDS.md",
    "LIMITATIONS.md",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize reporting value {type(value).__name__}")


def _markdown_table(frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> str:
    if columns is not None:
        available = [column for column in columns if column in frame]
        frame = frame.loc[:, available]
    if frame.empty or len(frame.columns) == 0:
        return "_No machine-readable rows were available for this table._"
    display = frame.copy()
    for column in display:
        display[column] = display[column].map(
            lambda value: ""
            if pd.isna(value)
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value)
        )
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _primary_stat_rows(statistics: Mapping[str, Any], primary: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    def one(name: str) -> dict[str, Any]:
        rows = [row for row in statistics[name] if row["experiment_id"] == primary]
        if len(rows) != 1:
            raise ReportingContractError(f"missing unique primary row in {name}")
        return rows[0]

    return one("mcnemar_rows"), one("bootstrap_rows"), one("holm_rows")


def _render_reports(
    *,
    stage: Path,
    output: Path,
    primary: str,
    baseline: str,
    lock_path: Path,
    validation_registry_path: Path,
    test_registry_path: Path,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed_summary: pd.DataFrame,
    statistics: Mapping[str, Any],
    gallery: Mapping[str, Any],
) -> dict[str, str]:
    reports = output / "reports"
    reports.mkdir()
    summary = statistics["primary_summary"]
    mcnemar, bootstrap, holm = _primary_stat_rows(statistics, primary)
    primary_table = pd.DataFrame(
        [
            {
                "primary": primary,
                "baseline": baseline,
                **summary,
                "scene_ci_lower": bootstrap["ci_lower"],
                "scene_ci_upper": bootstrap["ci_upper"],
                "mcnemar_raw_p": mcnemar["raw_p"],
                "holm_adjusted_p": holm["holm_adjusted_p"],
            }
        ]
    )
    ablation_mask = pd.Series(False, index=validation.index)
    for column in ("category", "experiment_type"):
        if column in validation:
            ablation_mask |= validation[column].astype(str).str.lower().str.contains(
                "ablation|leave|remove|without", regex=True
            )
    ablations = validation.loc[ablation_mask]
    route_table = validation if "route" in validation else validation.iloc[0:0]
    runtime_columns = [
        column for column in validation.columns if any(token in column.lower() for token in ("runtime", "latency", "parameter", "memory"))
    ]
    runtime = validation.loc[:, [column for column in ("experiment_id", "method") if column in validation] + runtime_columns]
    outcome_failure_rows = pd.DataFrame(
        [
            {"outcome_category": name, **{key: counts.get(key) for key in ("requested", "eligible", "selected", "materialized", "shortfall")}}
            for name, counts in gallery["categories"].items()
            if isinstance(counts, Mapping) and "eligible" in counts
        ]
    )
    failure_stage_rows = pd.DataFrame(
        [
            {"failure_stage": name, **{key: counts.get(key) for key in ("requested", "eligible", "selected", "materialized", "evidence_status")}}
            for name, counts in gallery["categories"].get("failure_stages", {}).items()
        ]
    )
    failure_group_rows = pd.DataFrame(
        [
            {"failure_group": name, **{key: counts.get(key) for key in ("requested", "eligible", "selected", "materialized")}}
            for name, counts in gallery["categories"].get("failure_groups", {}).items()
        ]
    )

    zh = f"""# 语言引导 4-DoF 抓取重排序最终报告

## 二维一致性声明

{CONSISTENCY_ZH}

## 实际数据与输入

- Reporting stage：`{stage}`
- Validation registry：`{validation_registry_path}`
- Test registry：`{test_registry_path}`
- Primary lock：`{lock_path}`
- 锁定 primary：`{primary}`
- Baseline：`{baseline}`
- Primary 选择仅来自 lock；test registry 未用于选择。

## 多 seed validation

{_markdown_table(seed_summary)}

## Primary retrospective test

{_markdown_table(primary_table)}

## 全部 validation 证据

{_markdown_table(validation)}

## Post-lock test comparisons

{_markdown_table(test)}

## 特征、模型、loss、encoder、gate、pool 消融

{_markdown_table(ablations)}

未出现在机器 registry 的比较不会在本报告中补造数字或排名。

## CROG 与 Modular 公平比较

{_markdown_table(route_table)}

两条路线候选池不同时，只能比较各自 headroom recovery、harmful switches、上界与运行代价，不能仅凭绝对 J@1 宣称模型优劣。

## 统计显著性

- Exact McNemar recovered/harmful：{mcnemar['recovered']}/{mcnemar['harmful']}
- Exact raw p：{mcnemar['raw_p']}
- Holm adjusted p：{holm['holm_adjusted_p']}
- Scene-cluster bootstrap：[{bootstrap['ci_lower']}, {bootstrap['ci_upper']}]，{bootstrap['iterations']} 次

## 失败阶段与素材

### Outcome categories

{_markdown_table(outcome_failure_rows)}

### F0--F10 stages

{_markdown_table(failure_stage_rows)}

### Grounding / candidate-generation / ranking groups

{_markdown_table(failure_group_rows)}

## 计算开销

{_markdown_table(runtime)}

## 论文级结论与不能支持的主张

锁定 primary 的二维一致性净变化为 {summary['net_recovered']} 个 query；是否具备统计可靠性应以上述 cluster CI 与 Holm 校正 p 值为准。该证据不能支持物理抓取成功、力闭合或无碰撞主张。

## 下一步

对 registry 中缺失的模型/消融保持“无证据”状态；补齐真实训练、预测与素材后重新运行本 reporting stage。
"""

    en = f"""# Final Re-ranking Report

## Metric scope

{CONSISTENCY_EN}

## Registered evidence

The reporting source is `{stage}`. Validation rows came from `{validation_registry_path}` and test rows from `{test_registry_path}`. The primary experiment `{primary}` was read exclusively from `{lock_path}`; the test registry was not inspected to select or replace it. The registered baseline is `{baseline}`.

## Multi-seed validation summary

{_markdown_table(seed_summary)}

## Locked-primary retrospective test

{_markdown_table(primary_table)}

## Validation matrix

{_markdown_table(validation)}

## Post-lock comparisons

{_markdown_table(test)}

## Ablation evidence

{_markdown_table(ablations)}

Unregistered experiments are reported as unavailable rather than assigned synthetic results.

## Route comparison and interpretation

{_markdown_table(route_table)}

Different candidate pools preclude a causal ranking based solely on absolute J@1. Headroom recovery, harmful switches, candidate-pool ceilings, complexity, and runtime must be interpreted jointly.

## Paired inference

The primary comparison produced {mcnemar['recovered']} recovered and {mcnemar['harmful']} harmful queries. The exact McNemar p-value was {mcnemar['raw_p']} (Holm-adjusted {holm['holm_adjusted_p']}); the {bootstrap['iterations']}-replicate scene-cluster interval was [{bootstrap['ci_lower']}, {bootstrap['ci_upper']}].

## Failure-stage decomposition

{_markdown_table(failure_stage_rows)}

Required qualitative groups:

{_markdown_table(failure_group_rows)}

## Limitations

The evaluation is retrospective and limited to registered artifacts. Missing architectures, ablations, seeds, or qualitative assets are not inferred. The benchmark does not measure physical execution.
"""

    tables = f"""# Dissertation Tables

{CONSISTENCY_EN}

## Table 1: Data and candidate-pool statistics

{_markdown_table(primary_table, columns=['sample_count', 'empty_count', 'no_positive_count', 'rank_gt5_count', 'oracle_correct'])}

## Table 2: Top-5 validation comparison

{_markdown_table(validation.loc[validation['pool'].astype(str).str.lower().str.contains('top')]) if 'pool' in validation else '_No pool field was registered._'}

## Table 3: Full-list validation comparison

{_markdown_table(validation.loc[validation['pool'].astype(str).str.lower().str.contains('full')]) if 'pool' in validation else '_No pool field was registered._'}

## Table 4: Primary retrospective test

{_markdown_table(primary_table)}

## Table 5: Feature ablation

{_markdown_table(ablations)}

## Table 6: Loss and encoder ablation

{_markdown_table(validation.loc[validation.apply(lambda row: any(token in str(value).lower() for value in row for token in ('ranknet','listwise','bce','gnn','deepsets','transformer')), axis=1)])}

## Table 7: Gate ablation

{_markdown_table(validation.loc[validation.apply(lambda row: any('gate' in str(value).lower() for value in row), axis=1)])}

## Table 8: CROG versus Modular fair comparison

{_markdown_table(route_table)}

## Table 9: Failure-stage breakdown

{_markdown_table(failure_stage_rows)}

## Table 10: Runtime and model complexity

{_markdown_table(runtime)}
"""

    failures = f"""# Failure Analysis

{CONSISTENCY_EN}

## Registered gallery coverage

### Outcome categories

{_markdown_table(outcome_failure_rows)}

### F0--F10 stages

{_markdown_table(failure_stage_rows)}

### Required failure groups

{_markdown_table(failure_group_rows)}

The gallery index distinguishes eligible, selected, and materialized cases per dataset. A materialized shortfall means a canonical RGB/depth/mask/geometry source was unavailable or failed audited rendering; no placeholder case was fabricated. Stages marked `UNAVAILABLE` have no machine-supported per-case evidence and are not inferred from unrelated proxies.

## Locked-primary outcome counts

{_markdown_table(primary_table, columns=['recovered', 'harmful', 'net_recovered', 'switch_count', 'empty_count', 'no_positive_count', 'rank_gt5_count'])}
"""

    reproduction = f"""# Reproduction Commands

{CONSISTENCY_EN}

```bash
cd {stage.parent}
/opt/anaconda3/bin/python -m reranking.report --stage {stage} --output {output}
```

The command consumes `{lock_path}`, `{validation_registry_path}`, `{test_registry_path}`, and the machine-declared prediction paths. It does not train, select a primary from test data, or write a run-level `_SUCCESS` marker.
"""

    limitations = f"""# Limitations

{CONSISTENCY_EN}

- The analysis is a held-out retrospective benchmark, not an untouched prospective robot trial.
- The locked primary is `{primary}`; post-lock methods remain comparisons and cannot replace it.
- Gallery availability is bounded by real declared assets: {json.dumps(gallery['categories'], sort_keys=True)}.
- Multi-seed claims are limited to the seed counts present in the validation registry.
- Missing registry rows are treated as missing evidence, never as zero, success, or expected performance.
- Candidate-pool differences limit direct CROG-versus-Modular absolute J@1 comparisons.
"""

    values = {
        "FINAL_REPORT_ZH.md": zh,
        "FINAL_REPORT_EN.md": en,
        "DISSERTATION_TABLES.md": tables,
        "FAILURE_ANALYSIS.md": failures,
        "REPRODUCTION_COMMANDS.md": reproduction,
        "LIMITATIONS.md": limitations,
    }
    paths = {}
    for name, text in values.items():
        path = reports / name
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        paths[name] = str(path)
    return paths


def required_artifacts(output: str | os.PathLike[str]) -> list[Path]:
    root = Path(output)
    required = [
        root / "statistics/mcnemar_results.csv",
        root / "statistics/bootstrap_intervals.csv",
        root / "statistics/holm_corrected_results.csv",
        root / "metrics/multi_seed_summary.csv",
        root / "metrics/primary_summary.json",
        root / "galleries/index.html",
        root / "galleries/index.csv",
        root / "galleries/gallery_summary.json",
    ]
    required.extend(root / "reports" / name for name in REPORT_NAMES)
    for name in FIGURE_NAMES:
        required.extend((root / "figures" / f"{name}.png", root / "figures" / f"{name}.pdf"))
    return required


def check_required_artifacts(output: str | os.PathLike[str]) -> dict[str, Any]:
    paths = required_artifacts(output)
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    forbidden = [str(path) for path in Path(output).rglob("_SUCCESS*")]
    if missing or forbidden:
        raise ReportingContractError(
            f"reporting artifact check failed: missing={missing}, forbidden={forbidden}"
        )
    return {"required_count": len(paths), "missing": [], "forbidden_success_markers": []}


def run_reporting_stage(
    stage: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run statistics, visualization and reports without primary reselection."""

    inputs = load_reporting_inputs(stage)
    statistics = compute_reporting_statistics(inputs, bootstrap_iterations=10_000)
    seed_summary = multi_seed_summary(inputs.validation_registry)
    reliability = reliability_bins(statistics["primary_outcomes"])
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"reporting output must be new: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    metrics_dir = destination / "metrics"
    statistics_dir = destination / "statistics"
    metrics_dir.mkdir()
    statistics_dir.mkdir()
    pd.DataFrame(statistics["mcnemar_rows"]).to_csv(
        statistics_dir / "mcnemar_results.csv", index=False
    )
    pd.DataFrame(statistics["bootstrap_rows"]).to_csv(
        statistics_dir / "bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(statistics["holm_rows"]).to_csv(
        statistics_dir / "holm_corrected_results.csv", index=False
    )
    seed_summary.to_csv(metrics_dir / "multi_seed_summary.csv", index=False)
    (metrics_dir / "primary_summary.json").write_text(
        json.dumps(_json_value(statistics["primary_summary"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs.validation_registry.to_csv(metrics_dir / "validation_registry_snapshot.csv", index=False)
    inputs.test_registry.to_csv(metrics_dir / "test_registry_snapshot.csv", index=False)
    figures = build_visualizations(
        destination / "figures",
        validation_registry=inputs.validation_registry,
        primary_outcomes=statistics["primary_outcomes"],
        reliability=reliability,
        bootstrap_rows=statistics["bootstrap_rows"],
        primary_summary=statistics["primary_summary"],
    )
    gallery = build_galleries(destination / "galleries", statistics["primary_outcomes"])
    reports = _render_reports(
        stage=inputs.stage,
        output=destination,
        primary=inputs.primary_experiment_id,
        baseline=inputs.baseline_name,
        lock_path=inputs.lock_path,
        validation_registry_path=inputs.validation_registry_path,
        test_registry_path=inputs.test_registry_path,
        validation=inputs.validation_registry,
        test=inputs.test_registry,
        seed_summary=seed_summary,
        statistics=statistics,
        gallery=gallery,
    )
    checks = check_required_artifacts(destination)
    artifact_paths = required_artifacts(destination)
    manifest = {
        "schema_version": 1,
        "kind": "reranking_reporting_stage",
        "status": "complete",
        "primary_experiment_id": inputs.primary_experiment_id,
        "primary_selection_source": str(inputs.lock_path),
        "test_used_for_primary_selection": False,
        "two_dimensional_consistency_only": True,
        "bootstrap_iterations": 10_000,
        "required_artifact_check": checks,
        "artifacts": {
            path.relative_to(destination).as_posix(): {
                "path": str(path),
                "sha256": streaming_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
        "figures": figures,
        "galleries": gallery,
        "reports": reports,
        "run_success_marker_written": False,
    }
    manifest_path = destination / "reporting_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest | {"manifest_path": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_reporting_stage(args.stage, args.output)
    print(json.dumps(_json_value(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONSISTENCY_EN",
    "CONSISTENCY_ZH",
    "REPORT_NAMES",
    "check_required_artifacts",
    "main",
    "required_artifacts",
    "run_reporting_stage",
]
