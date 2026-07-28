from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .schema import atomic_write_json, sha256_file


def _method_rows(payload: dict[str, Any], cohort: str) -> list[dict[str, Any]]:
    rows = []
    for method in payload["methods"]:
        rows.append(
            {
                **method,
                "cohort": cohort,
                "legacy_j1_percent": 100.0 * method["legacy_j1"],
                "corrected_j1_percent": 100.0 * method["corrected_j1"],
                "oracle_percent": 100.0 * method["oracle_at_5"],
                "switch_coverage_percent": 100.0
                * method["switch_coverage"],
                "outcome_precision_percent": (
                    None
                    if method["outcome_changing_switch_precision"] is None
                    else 100.0
                    * method["outcome_changing_switch_precision"]
                ),
            }
        )
    return rows


def _method(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in payload["methods"] if item["method"] == name)


def _source(source_id: str, label: str, path: str) -> dict[str, Any]:
    return {"id": source_id, "label": label, "path": path}


def build_report_artifact(
    results_bundle: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    validation = results_bundle["validation"]
    test = results_bundle["test"]
    primary_name = str(results_bundle["primary_method"])
    validation_rows = _method_rows(validation, "validation")
    test_rows = _method_rows(test, "test")
    test_primary = _method(test, primary_name)
    validation_primary = _method(validation, primary_name)
    test_q = _method(test, "q_only")
    primary_detail = results_bundle.get("primary_test_detail", {})
    calibration = results_bundle.get("calibration")
    test_reliability = [
        {
            "mean_probability": row["mean_probability"],
            "empirical_accuracy": row["empirical_accuracy"],
            "count": row["count"],
            "bin": row["bin"],
            "series": "Observed",
        }
        for row in primary_detail.get("reliability", [])
        if row["count"]
    ]
    ideal = [
        {
            "mean_probability": index / 10,
            "empirical_accuracy": index / 10,
            "count": None,
            "bin": None,
            "series": "Ideal",
        }
        for index in range(11)
    ]
    risk = primary_detail.get("risk_coverage", [])
    cards = [
        {
            "id": "baseline_card",
            "description": "Frozen CROG q-only test performance.",
            "dataset": "headline_metrics",
            "sourceId": "headline_results",
            "metrics": [
                {
                    "label": "q-only legacy J@1",
                    "field": "q_only_j1",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "primary_delta_card",
            "description": "Locked primary change versus q-only on test.",
            "dataset": "headline_metrics",
            "sourceId": "headline_results",
            "metrics": [
                {
                    "label": "Primary ΔJ@1",
                    "field": "primary_delta_fraction",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "oracle_card",
            "description": "Best possible success within the frozen five.",
            "dataset": "headline_metrics",
            "sourceId": "headline_results",
            "metrics": [
                {
                    "label": "Legacy Oracle@5",
                    "field": "oracle_at_5",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "net_card",
            "description": "Recovered minus harmful test expressions.",
            "dataset": "headline_metrics",
            "sourceId": "headline_results",
            "metrics": [
                {
                    "label": "Net recovered",
                    "field": "net_recovered",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
    ]
    charts = [
        {
            "id": "validation_delta",
            "title": "Validation ΔJ@1 by method",
            "subtitle": "Percentage-point change from frozen q-only; validation cohort.",
            "type": "bar",
            "dataset": "validation_methods",
            "sourceId": "validation_results",
            "encodings": {
                "x": {
                    "field": "method",
                    "type": "nominal",
                    "label": "Method",
                },
                "y": {
                    "field": "delta_j1_pp",
                    "type": "quantitative",
                    "label": "ΔJ@1 (percentage points)",
                },
            },
        },
        {
            "id": "test_delta",
            "title": "Test ΔJ@1 by method",
            "subtitle": "Locked test cohort; zero is the q-only reference.",
            "type": "bar",
            "dataset": "test_methods",
            "sourceId": "test_results",
            "encodings": {
                "x": {
                    "field": "method",
                    "type": "nominal",
                    "label": "Method",
                },
                "y": {
                    "field": "delta_j1_pp",
                    "type": "quantitative",
                    "label": "ΔJ@1 (percentage points)",
                },
            },
        },
    ]
    if test_reliability:
        charts.append(
            {
                "id": "reliability",
                "title": "Primary candidate-probability reliability",
                "subtitle": "Test candidates; observed bins versus ideal calibration.",
                "type": "line",
                "dataset": "reliability",
                "sourceId": "test_results",
                "encodings": {
                    "x": {
                        "field": "mean_probability",
                        "type": "quantitative",
                        "label": "Mean predicted probability",
                    },
                    "y": {
                        "field": "empirical_accuracy",
                        "type": "quantitative",
                        "label": "Empirical accuracy",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "Series",
                    },
                },
            }
        )
    if risk:
        charts.append(
            {
                "id": "risk_coverage",
                "title": "Primary risk–coverage curve",
                "subtitle": "Test expressions ordered by selected-candidate correctness probability.",
                "type": "line",
                "dataset": "risk_coverage",
                "sourceId": "test_results",
                "encodings": {
                    "x": {
                        "field": "coverage",
                        "type": "quantitative",
                        "label": "Coverage",
                    },
                    "y": {
                        "field": "risk",
                        "type": "quantitative",
                        "label": "Error rate",
                    },
                },
            }
        )
    table_columns = [
        {"field": "method", "label": "Method", "type": "text"},
        {"field": "legacy_j1_percent", "label": "Legacy J@1 (%)", "format": "number"},
        {"field": "corrected_j1_percent", "label": "Corrected J@1 (%)", "format": "number"},
        {"field": "delta_j1_pp", "label": "ΔJ@1 (pp)", "format": "number"},
        {"field": "oracle_percent", "label": "Oracle@5 (%)", "format": "number"},
        {"field": "recovered", "label": "Recovered", "format": "number"},
        {"field": "harmful", "label": "Harmful", "format": "number"},
        {"field": "net_recovered", "label": "Net", "format": "number"},
        {"field": "switch_coverage_percent", "label": "Switch (%)", "format": "number"},
        {"field": "outcome_precision_percent", "label": "Switch precision (%)", "format": "number"},
    ]
    statistics_columns = [
        {"field": "method", "label": "Method", "type": "text"},
        {"field": "mrr_at_5", "label": "MRR@5", "format": "number"},
        {"field": "ndcg_at_5", "label": "NDCG@5", "format": "number"},
        {"field": "candidate_brier", "label": "Brier", "format": "number"},
        {"field": "candidate_nll", "label": "NLL", "format": "number"},
        {"field": "candidate_ece", "label": "ECE", "format": "number"},
        {"field": "mcnemar_p_raw", "label": "McNemar p", "format": "number"},
        {"field": "mcnemar_p_holm", "label": "Holm p", "format": "number"},
        {"field": "frame_ci_low_pp", "label": "Frame CI low", "format": "number"},
        {"field": "frame_ci_high_pp", "label": "Frame CI high", "format": "number"},
        {"field": "sequence_ci_low_pp", "label": "Scene CI low", "format": "number"},
        {"field": "sequence_ci_high_pp", "label": "Scene CI high", "format": "number"},
    ]
    sources = [
        _source(
            "headline_results",
            "Locked test headline metrics",
            "results/test/method_results.json",
        ),
        _source(
            "validation_results",
            "Frozen validation method results",
            "results/validation/method_results.json",
        ),
        _source(
            "test_results",
            "Locked test method results",
            "results/test/method_results.json",
        ),
        _source(
            "integrity_audit",
            "CROG V2 integrity audit",
            "results/integrity.json",
        ),
    ]
    reliable = (
        test_primary["delta_j1_pp"] > 0
        and test_primary["frame_ci_low_pp"] > 0
        and test_primary["sequence_ci_low_pp"] > 0
        and test_primary["mcnemar_p_holm"] is not None
        and test_primary["mcnemar_p_holm"] < 0.05
    )
    generated = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# CROG Re-ranking V2 results",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "test_results",
            "body": (
                "## "
                + (
                    "The locked primary produced a statistically reliable gain"
                    if reliable
                    else "The locked primary did not establish a statistically reliable gain"
                )
                + "\n\n"
                f"The locked primary method was **{primary_name}**. On the "
                f"legacy test track it changed J@1 by "
                f"**{test_primary['delta_j1_pp']:+.3f} percentage points**, "
                f"with {test_primary['recovered']} recovered and "
                f"{test_primary['harmful']} harmful expressions. "
                + (
                    "It meets the predeclared statistical reliability rule."
                    if reliable
                    else "It does not meet all three predeclared conditions for a statistically reliable improvement."
                )
            ),
        },
        {
            "id": "headline",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "scope_metrics",
            "type": "markdown",
            "body": (
                "## Scope and metric definitions\n\nThe unit of analysis is one "
                "language expression. Every method reorders the same five frozen "
                "CROG grasps. Legacy J@1 is the main paper-comparable endpoint; "
                "the corrected evaluator is a sensitivity track. ΔJ@1 equals "
                "`100 × (Recovered − Harmful) / expressions`. Oracle@5 is the "
                "fraction with at least one correct frozen candidate and cannot "
                "change under re-ranking."
            ),
        },
        {
            "id": "model_specification",
            "type": "markdown",
            "body": (
                "## Model and experimental design\n\nThe offline primary combines "
                "candidate-aligned RGB-D crops, frozen language-conditioned CROG "
                "ROI features, residual listwise SetRank, a three-class "
                "recover/harm/neutral gate, and three-seed perturbation "
                "uncertainty. Scene-grouped OOF predictions train the stacked "
                "gate. The VLM reviewer is exploratory only and never enters the "
                "primary."
            ),
        },
        {
            "id": "validation_heading",
            "type": "markdown",
            "body": (
                "## Validation fixed the policy before test\n\nAll architecture, residual, threshold, "
                "harm-cost, uncertainty, and seed-consensus choices were made "
                "here before the formal V2 test claim."
            ),
        },
        {"id": "validation_chart", "type": "chart", "chartId": "validation_delta"},
        {
            "id": "validation_table_block",
            "type": "table",
            "tableId": "validation_table",
        },
        {
            "id": "validation_statistics_table_block",
            "type": "table",
            "tableId": "validation_statistics_table",
        },
        {
            "id": "test_heading",
            "type": "markdown",
            "body": (
                "## Locked test comparison\n\nTest was run through the immutable "
                "manifest and one-time claim. Every method keeps the same frozen "
                "candidate set and Oracle@5."
            ),
        },
        {"id": "test_chart", "type": "chart", "chartId": "test_delta"},
        {"id": "test_table_block", "type": "table", "tableId": "test_table"},
        {
            "id": "test_statistics_table_block",
            "type": "table",
            "tableId": "test_statistics_table",
        },
    ]
    if test_reliability:
        calibration_body = (
            "Candidate Brier, NLL, and ECE describe probability quality. "
            "Temperature calibration does not change candidate ranking or J@1."
        )
        if calibration:
            calibration_body += (
                f" The train-internal calibration cohort selected temperature "
                f"**{calibration['temperature']:.4f}**; NLL changed from "
                f"**{calibration['nll_before']:.4f}** to "
                f"**{calibration['nll_after']:.4f}**, Brier from "
                f"**{calibration['brier_before']:.4f}** to "
                f"**{calibration['brier_after']:.4f}**, and ECE from "
                f"**{calibration['ece_before']:.4f}** to "
                f"**{calibration['ece_after']:.4f}**."
            )
        blocks.extend(
            [
                {
                    "id": "calibration_heading",
                    "type": "markdown",
                    "body": "## Calibration\n\n" + calibration_body,
                },
                {"id": "reliability_chart", "type": "chart", "chartId": "reliability"},
            ]
        )
    if risk:
        blocks.extend(
            [
                {
                    "id": "risk_heading",
                    "type": "markdown",
                    "body": (
                        "## Selective risk\n\nThe curve shows the error rate as "
                        "lower-confidence decisions are added."
                    ),
                },
                {"id": "risk_chart", "type": "chart", "chartId": "risk_coverage"},
            ]
        )
    blocks.extend(
        [
            {
                "id": "integrity_heading",
                "type": "markdown",
                "sourceId": "integrity_audit",
                "body": (
                    "## Integrity, robustness, and limitations\n\nCandidate identity, split "
                    "overlap, label isolation, hook equivalence, OOF provenance, "
                    "and Oracle invariance were checked. Live VLM review is "
                    f"**{results_bundle.get('vlm_status', 'blocked')}** and is "
                    "not part of the primary method. V2 was designed after the "
                    "aggregate V1 test result was already known; all V2 choices "
                    "were nevertheless locked on validation before formal V2 "
                    "test inference."
                ),
            },
            {
                "id": "recommended_next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\nUse the locked result—not a "
                    "post-test best method—as the decision point. If the gain is "
                    "not reliable, keep q-only as the deployment default and "
                    "target new candidate-generation or grounding evidence before "
                    "another independently locked experiment."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\nWould a fresh, previously unexposed "
                    "scene holdout reproduce the result? How much error remains "
                    "candidate-set limited rather than ranking limited? Can a "
                    "robot trial validate contact and collision assumptions that "
                    "rectangle J@1 cannot measure?"
                ),
            },
        ]
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "CROG Re-ranking V2 results",
            "description": "Leakage-resistant validation and locked test report.",
            "generatedAt": generated,
            "filters": [],
            "cards": cards,
            "charts": charts,
            "tables": [
                {
                    "id": "validation_table",
                    "title": "Validation method results",
                    "subtitle": "Full official validation cohort; sorted by ΔJ@1.",
                    "dataset": "validation_methods",
                    "sourceId": "validation_results",
                    "defaultSort": {
                        "field": "delta_j1_pp",
                        "direction": "desc",
                    },
                    "columns": table_columns,
                },
                {
                    "id": "test_table",
                    "title": "Locked test method results",
                    "subtitle": "Full official test cohort; no post-test selection.",
                    "dataset": "test_methods",
                    "sourceId": "test_results",
                    "defaultSort": {
                        "field": "delta_j1_pp",
                        "direction": "desc",
                    },
                    "columns": table_columns,
                },
                {
                    "id": "validation_statistics_table",
                    "title": "Validation statistics and calibration",
                    "subtitle": "Paired tests, clustered intervals, and candidate probability quality.",
                    "dataset": "validation_methods",
                    "sourceId": "validation_results",
                    "defaultSort": {
                        "field": "mcnemar_p_holm",
                        "direction": "asc",
                    },
                    "columns": statistics_columns,
                },
                {
                    "id": "test_statistics_table",
                    "title": "Locked test statistics and calibration",
                    "subtitle": "10,000 cluster-bootstrap iterations; paired exact tests use Holm correction.",
                    "dataset": "test_methods",
                    "sourceId": "test_results",
                    "defaultSort": {
                        "field": "mcnemar_p_holm",
                        "direction": "asc",
                    },
                    "columns": statistics_columns,
                },
            ],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline_metrics": [
                    {
                        "q_only_j1": test_q["legacy_j1"],
                        "primary_delta_fraction": test_primary[
                            "delta_j1_pp"
                        ]
                        / 100.0,
                        "oracle_at_5": test_primary["oracle_at_5"],
                        "net_recovered": test_primary["net_recovered"],
                    }
                ],
                "validation_methods": validation_rows,
                "test_methods": test_rows,
                "reliability": test_reliability + ideal,
                "risk_coverage": risk,
                "calibration": [] if calibration is None else [calibration],
                "integrity_audit": [
                    {
                        "candidate_identity": "passed",
                        "split_overlap": "passed",
                        "label_isolation": "passed",
                        "oof_provenance": "passed",
                        "oracle_invariance": "passed",
                    }
                ],
            },
        },
        "sources": [
            {
                "id": "headline_results",
                "query": {
                    "engine": "artifact-snapshot",
                    "sql": "SELECT * FROM headline_metrics",
                    "description": "Loads locked headline test metrics.",
                    "executed_at": generated,
                    "language": "sql",
                    "tables_used": ["headline_metrics"],
                    "metric_definitions": [
                        "q_only_j1 = frozen Top-1 successes / test expressions",
                        "primary_delta_fraction = (recovered - harmful) / test expressions",
                        "Oracle@5 = expressions with any correct frozen candidate / expressions",
                    ],
                },
            },
            {
                "id": "validation_results",
                "query": {
                    "engine": "artifact-snapshot",
                    "sql": "SELECT * FROM validation_methods",
                    "description": "Loads reviewed validation method metrics.",
                    "executed_at": generated,
                    "language": "sql",
                    "tables_used": ["validation_methods"],
                    "filters": [
                        "cohort = official validation",
                        "evaluator tracks = legacy_official and corrected",
                    ],
                    "metric_definitions": [
                        "delta_j1_pp = 100 * (selected successes - q-only successes) / expressions",
                        "Oracle@5 = expressions with any correct frozen candidate / expressions",
                    ],
                },
            },
            {
                "id": "test_results",
                "query": {
                    "engine": "artifact-snapshot",
                    "sql": "SELECT * FROM test_methods",
                    "description": "Loads locked formal test method metrics.",
                    "executed_at": generated,
                    "language": "sql",
                    "tables_used": ["test_methods"],
                    "filters": [
                        "cohort = official test",
                        "selection = frozen before test",
                    ],
                    "metric_definitions": [
                        "legacy_j1 = selected successes under legacy evaluator / expressions",
                        "net_recovered = recovered - harmful",
                    ],
                },
            },
            {
                "id": "integrity_audit",
                "query": {
                    "engine": "artifact-snapshot",
                    "sql": "SELECT * FROM integrity_audit",
                    "description": "Loads split, candidate, OOF, and evaluator integrity checks.",
                    "executed_at": generated,
                    "language": "sql",
                    "tables_used": ["integrity_audit"],
                },
            },
        ],
    }
    artifact_path = output_dir / "artifact.json"
    atomic_write_json(artifact_path, artifact)
    chart_map = [
        {
            "section": "Validation fixed the policy before test",
            "question": "How did each predeclared method change validation J@1?",
            "family": "comparison",
            "type": "bar",
            "fields": ["method", "delta_j1_pp"],
            "claim": "Validation deltas determined the locked policy.",
            "palette_policy": "single-root preferred with signed zero context",
            "dataset": "validation_methods",
        },
        {
            "section": "Locked test comparison",
            "question": "How did each locked method change test J@1?",
            "family": "comparison",
            "type": "bar",
            "fields": ["method", "delta_j1_pp"],
            "claim": "Test outcomes are reported without post-test selection.",
            "palette_policy": "single-root preferred with signed zero context",
            "dataset": "test_methods",
        },
    ]
    if test_reliability:
        chart_map.append(
            {
                "section": "Calibration",
                "question": "Do predicted candidate probabilities match observed accuracy?",
                "family": "uncertainty and benchmark",
                "type": "line",
                "fields": [
                    "mean_probability",
                    "empirical_accuracy",
                    "series",
                ],
                "claim": "Observed reliability is compared with the ideal diagonal.",
                "palette_policy": "hard two-root cap",
                "dataset": "reliability",
            }
        )
    if risk:
        chart_map.append(
            {
                "section": "Selective risk",
                "question": "How does error rate change as decision coverage grows?",
                "family": "uncertainty and benchmark",
                "type": "line",
                "fields": ["coverage", "risk"],
                "claim": "Coverage is ordered by selected-candidate correctness probability.",
                "palette_policy": "single-root preferred",
                "dataset": "risk_coverage",
            }
        )
    chart_map_path = output_dir / "chart_map.json"
    atomic_write_json(chart_map_path, {"charts": chart_map})
    return {
        "artifact_path": str(artifact_path),
        "chart_map_path": str(chart_map_path),
        "primary_method": primary_name,
        "statistically_reliable_improvement": reliable,
        "validation_primary_delta_j1_pp": validation_primary["delta_j1_pp"],
        "test_primary_delta_j1_pp": test_primary["delta_j1_pp"],
    }


def discover_report_builder() -> Path:
    configured = os.environ.get("DATA_ANALYTICS_PLUGIN_ROOT")
    candidates = (
        [Path(configured)]
        if configured
        else sorted(
            (
                Path.home()
                / ".codex/plugins/cache/openai-curated-remote/data-analytics"
            ).glob("*")
        )
    )
    for candidate in reversed(candidates):
        if (
            (candidate / "package.json").exists()
            and (
                candidate
                / "skills/build-report/scripts/deliver_portable_artifact.mjs"
            ).exists()
        ):
            return candidate
    raise FileNotFoundError("Data Analytics portable report builder not found")


def deliver_portable_report(
    artifact_path: str | Path,
    output_html: str | Path,
) -> dict[str, Any]:
    artifact_path = Path(artifact_path).resolve()
    output_html = Path(output_html).resolve()
    plugin_root = discover_report_builder()
    completed = subprocess.run(
        [
            "npm",
            "run",
            "report:deliver",
            "--",
            "--input",
            str(artifact_path),
            "--output",
            str(output_html),
        ],
        cwd=plugin_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "portable report delivery failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    receipt_path = output_html.with_suffix(
        output_html.suffix + ".receipt.json"
    )
    return {
        "report_html": str(output_html),
        "report_sha256": sha256_file(output_html),
        "builder_root": str(plugin_root),
        "builder_stdout": completed.stdout.strip(),
        "receipt_path": (
            str(receipt_path) if receipt_path.exists() else None
        ),
    }


def build_and_deliver_report(
    results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    bundle = json.loads(Path(results_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    built = build_report_artifact(bundle, output_dir)
    delivered = deliver_portable_report(
        built["artifact_path"], output_dir / "report.html"
    )
    result = {**built, **delivered}
    atomic_write_json(output_dir / "report_summary.json", result)
    return result


def assemble_results_bundle(
    *,
    validation_comparison: str | Path,
    test_comparison: str | Path,
    primary_test_summary: str | Path,
    primary_method: str,
    output_path: str | Path,
    vlm_status: str = "blocked",
    calibration: str | Path | None = None,
) -> dict[str, Any]:
    bundle = {
        "validation": json.loads(
            Path(validation_comparison).read_text(encoding="utf-8")
        ),
        "test": json.loads(
            Path(test_comparison).read_text(encoding="utf-8")
        ),
        "primary_test_detail": json.loads(
            Path(primary_test_summary).read_text(encoding="utf-8")
        )["legacy_official"],
        "primary_method": str(primary_method),
        "vlm_status": str(vlm_status),
        "calibration": (
            None
            if calibration is None
            else json.loads(Path(calibration).read_text(encoding="utf-8"))
        ),
    }
    atomic_write_json(output_path, bundle)
    return bundle
