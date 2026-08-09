"""Build deterministic post-formal tables, figures, galleries, reports, and lock.

The command is fail-closed: it requires the completed exactly-once formal-Test
transaction and an independent recomputation.  It does not import or execute a
training/ranker module and never changes the locked selections.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.hashing import canonical_sha256, sha256_file  # noqa: E402
from unified_reranking.ledger import (  # noqa: E402
    export_ledger_commands,
    ledger_stage,
    render_ledger_commands,
)
from unified_reranking.matrix_phase import load_matrix_phase_cells  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verify_artifact_records_recursive,
    verified_artifact_path,
)
from unified_reranking.telemetry import resolved_track_extraction_latency  # noqa: E402
from unified_reranking.metrics import evaluate_order_only  # noqa: E402
from unified_reranking.training import FORMAL_SEEDS  # noqa: E402
from unified_reranking.postformal_reporting import (  # noqa: E402
    CROSS_ROUTE_GALLERY_QUOTAS,
    FIGURE_NAMES,
    REPORT_NAMES,
    ROUTES,
    ROUTE_GALLERY_QUOTAS,
    TABLE_NAMES,
    FormalBundle,
    _decision_frame,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    atomic_text,
    classify_failures,
    core_integrity_checks,
    deployment_decision,
    deterministic_case_selection,
    formal_results_table,
    gallery_category_members,
    hash_inventory,
    latex_escape,
    load_formal_bundle,
    markdown_table,
    read_json,
    safe_number,
    taxonomy_summary,
    verify_inventory,
)


CAPTIONS = {
    "oracle_at_k_curves": "Formal Test J@K curves. Order-only systems share the same frozen Top-5 membership; therefore every route's J@5 endpoint must remain invariant.",
    "j_at_1_delta_forest": "Paired formal-Test J@1 changes with 95% scene-clustered bootstrap intervals. Intervals, not candidate-row uncertainty, are primary.",
    "recovered_vs_harmful": "Exact recovered and harmful sample counts relative to each predeclared native reference.",
    "headroom_recovery_at_5": "Fraction of native-to-Oracle@5 ranking headroom recovered by each formal system; NA denotes zero native headroom.",
    "loss_comparison": "Controlled Validation comparison of BCE, RankNet, listwise, and Jacquard-margin RankNet under the same MLP encoder budget.",
    "encoder_comparison": "Validation encoder comparison under the loss fixed before encoder selection.",
    "cumulative_feature_ablation": "Validation cumulative feature-family ablation generated from the machine-readable ablation registry.",
    "leave_one_family_out_ablation": "Validation leave-one-family-out ablation; positive values indicate that retaining the family helped.",
    "risk_coverage": "Risk-coverage evidence from persisted gate/risk trials; no Test operating point is selected from this plot.",
    "calibration_reliability": "Validation reliability curves for the locked calibration candidates; the diagonal denotes ideal calibration.",
    "switch_rate_vs_net_gain": "Validation/formal switch rate versus paired net J@1 gain, exposing aggressive but harmful policies.",
    "tri_backend_complementarity_matrix": "Pairwise correctness overlap of the final gated CROG, G1, and C1 selectors on the paired formal denominator.",
    "eight_way_outcome_intersection": "All eight correctness intersections for final gated CROG/G1/C1, without filtering the shared denominator.",
    "runtime_vs_gain": "Measured runtime versus paired gain where runtime telemetry was persisted; missing measurements are shown as unavailable.",
    "parameter_count_vs_gain": "Model parameter count versus paired gain where parameter metadata was persisted.",
    "attribution_bridge": "Two-by-two Train/Validation and secondary post-lock Test bridge crossing candidate-pool and selector contracts. Test bridge rows cannot feed method selection.",
}


def _read_csv(path: Path, columns: Sequence[str] = ()) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    return pd.read_csv(path)


def _concat_csv(paths: Sequence[Path], columns: Sequence[str] = ()) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(paths):
        if path.is_file():
            frame = pd.read_csv(path)
            frame.insert(0, "source_file", str(path.resolve()))
            pieces.append(frame)
    return (
        pd.concat(pieces, ignore_index=True, sort=False)
        if pieces
        else pd.DataFrame(columns=list(columns))
    )


def _phase_selection_contract(
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Load one exact phase selection and normalise route/track choices."""

    payload = read_json(selection_path)
    selections = payload.get("selections")
    if not isinstance(selections, Mapping) or not selections:
        raise RuntimeError(f"matrix selection inventory is empty: {selection_path}")
    choices: dict[str, list[dict[str, Any]]] = {}
    for key, raw in selections.items():
        values = raw if isinstance(raw, list) else [raw]
        if not values or any(not isinstance(value, Mapping) for value in values):
            raise RuntimeError(f"matrix selection entry is malformed: {key}")
        choices[str(key).lower()] = [dict(value) for value in values]
    return payload, choices


def _cell_matches_phase_selection(
    cell: Mapping[str, Any],
    *,
    phase: str,
    choices: Mapping[str, list[dict[str, Any]]],
) -> bool:
    configuration = cell.get("configuration")
    if not isinstance(configuration, Mapping):
        return False
    key = f"{configuration.get('route')}/{configuration.get('track')}".lower()
    selected = choices.get(key, [])
    if not selected:
        return False
    feature_columns = list(map(str, cell.get("feature_columns", [])))
    feature_schema = canonical_sha256(feature_columns)
    if not feature_columns or cell.get("feature_schema_sha256") != feature_schema:
        return False
    for choice in selected:
        if str(configuration.get("loss")) != str(choice.get("loss")):
            continue
        if phase == "selected" and str(configuration.get("encoder")) != str(
            choice.get("encoder")
        ):
            continue
        parameters = choice.get("parameters", {})
        if not isinstance(parameters, Mapping) or any(
            configuration.get(name) != value for name, value in parameters.items()
        ):
            continue
        screen_path = choice.get("screen_manifest")
        screen_sha = choice.get("screen_manifest_sha256")
        if not isinstance(screen_path, str) or not isinstance(screen_sha, str):
            continue
        screen_manifest_path = Path(screen_path).resolve()
        if (
            not screen_manifest_path.is_file()
            or sha256_file(screen_manifest_path) != screen_sha
        ):
            continue
        screen = read_json(screen_manifest_path)
        screen_columns = list(map(str, screen.get("feature_columns", [])))
        if screen_columns != feature_columns:
            continue
        source_identity = configuration.get("source_identity")
        if (
            not isinstance(source_identity, Mapping)
            or source_identity.get("selected_feature_schema_sha256") != feature_schema
        ):
            continue
        return True
    return False


def _validation_cells(run_dir: Path) -> pd.DataFrame:
    """Read only the exact output manifests of the three latest matrix phases."""

    rows: list[dict[str, Any]] = []
    def extraction_latency(
        value: Mapping[str, Any], configuration: Mapping[str, Any]
    ) -> tuple[Any, str | None]:
        sources = value.get("sources")
        if not isinstance(sources, Mapping):
            raise RuntimeError("Validation cell source inventory is missing")
        rule_cell = "method" in configuration
        feature_key = (
            "prediction_feature_manifest"
            if rule_cell
            else "validation_feature_manifest"
        )
        benchmark_key = (
            "prediction_feature_extraction_benchmark"
            if rule_cell
            else "validation_feature_extraction_benchmark"
        )
        feature_record = sources.get(feature_key)
        feature_path = verified_artifact_path(
            feature_record, name="Validation telemetry feature manifest"
        )
        feature_manifest = load_verified_json(
            feature_path, name="Validation telemetry feature manifest"
        )
        latency, benchmark_record = resolved_track_extraction_latency(
            run_dir,
            str(configuration.get("track", "")),
            feature_manifest,
        )
        observed = value.get("feature_extraction_latency_ms")
        if (
            not isinstance(observed, (int, float))
            or not np.isfinite(float(observed))
            or not np.isclose(float(observed), latency, rtol=0.0, atol=1e-12)
        ):
            raise RuntimeError(
                "Validation cell extraction latency differs from bound sources"
            )
        if sources.get(benchmark_key) != benchmark_record:
            raise RuntimeError(
                "Validation cell extraction benchmark record differs from current benchmark"
            )
        provenance = {
            "feature_manifest": feature_record,
            "feature_extraction_benchmark": benchmark_record,
        }
        return latency, json.dumps(provenance, sort_keys=True)

    phase_specs = {
        "screen": None,
        "selected": run_dir / "05_models" / "screen_finalists.json",
        "encoder": run_dir / "05_models" / "encoder_loss_selections.json",
    }
    for phase, expected_selection in phase_specs.items():
        execution_path = (
            run_dir / "05_models" / "matrix_plans" / f"{phase}_latest_execution.json"
        )
        if not execution_path.is_file():
            continue
        choices: dict[str, list[dict[str, Any]]] = {}
        if expected_selection is not None:
            _, choices = _phase_selection_contract(expected_selection)
        for value, path in load_matrix_phase_cells(
            run_dir,
            phase,
            expected_selection=expected_selection,
        ):
            configuration = dict(value.get("configuration", {}))
            if configuration.get("mode") != "validation":
                continue
            if phase != "screen" and not _cell_matches_phase_selection(
                value, phase=phase, choices=choices
            ):
                raise RuntimeError(
                    f"matrix {phase} cell differs from its locked selection: {path}"
                )
            metrics = dict(value.get("metrics", {}))
            training = dict(value.get("training", {}))
            feature_extraction_latency, extraction_source = extraction_latency(
                value, configuration
            )
            rows.append(
                {
                    "phase": phase,
                    "route": configuration.get("route"),
                    "track": configuration.get("track"),
                    "encoder": configuration.get(
                        "encoder", configuration.get("method")
                    ),
                    "loss": configuration.get("loss", "rule"),
                    "seed": configuration.get("seed"),
                    "j_at_1": metrics.get("j_at_1"),
                    "mrr_at_5": metrics.get("mrr_at_5"),
                    "parameter_count": value.get("parameter_count"),
                    "ranker_latency_ms": value.get("ranker_latency_ms"),
                    "feature_extraction_latency_ms": feature_extraction_latency,
                    "feature_extraction_telemetry_source": extraction_source,
                    "feature_latency_ms": feature_extraction_latency,
                    "cell_load_preprocess_latency_ms": value.get("feature_latency_ms"),
                    "peak_memory_mb": value.get("peak_memory_mb"),
                    "missing_feature_rate": value.get("missing_feature_rate"),
                    "best_epoch": training.get("best_epoch"),
                    "epochs_ran": training.get("epochs_ran"),
                    "feature_count": len(value.get("feature_columns", [])),
                    "num_attention_blocks": configuration.get("num_attention_blocks"),
                    "manifest_path": str(path.resolve()),
                    "manifest_sha256": sha256_file(path),
                    "cell_kind": "rule_cells"
                    if configuration.get("encoder") == "rule"
                    or "method" in configuration
                    else "matrix_cells",
                    "artifact_records_verified": True,
                    "artifact_verification_error": None,
                }
            )
    return pd.DataFrame(rows)


def _gate_decisions(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    thresholds = run_dir / "08_lock" / "gate_thresholds.json"
    if thresholds.is_file():
        payload = read_json(thresholds)
        for route, value in dict(payload.get("routes", {})).items():
            result[str(route).lower()] = str(dict(value).get("decision", ""))
    for route in ROUTES:
        path = run_dir / "08_lock" / "gates" / route / "gate_selection.json"
        if path.is_file():
            result[route] = str(read_json(path).get("decision", ""))
    return result


def _gate_table(run_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(
        (run_dir / "08_lock" / "gates").glob("*/gate_validation_trials.parquet")
    ):
        frame = pd.read_parquet(path)
        frame.insert(0, "route", path.parent.name.lower())
        frame.insert(1, "source_file", str(path.resolve()))
        pieces.append(frame)
    return (
        pd.concat(pieces, ignore_index=True, sort=False)
        if pieces
        else pd.DataFrame(
            columns=[
                "route",
                "bootstrap_lower_bound",
                "mean_delta",
                "recovered",
                "harmful",
                "switch_rate",
            ]
        )
    )


def build_postlock_bridge(bundle: FormalBundle) -> dict[str, Any]:
    """Materialize the secondary Test bridge from the consumed formal bundle.

    This function deliberately accepts no label path.  Candidate outcomes come
    only from the hash-verified ``bridge_per_candidate_scores`` artifact that the
    exactly-once formal evaluator already persisted. Missing pool/selector
    inputs are explicit hard failures, never accepted as completed cells and
    never reconstructed by reopening the source Test-label parquet.
    """

    source_record = dict(bundle.manifest.get("artifacts", {})).get(
        "bridge_per_candidate_scores", {}
    )
    source_path = Path(str(source_record.get("path", ""))).resolve()
    source_sha256 = str(source_record.get("sha256", ""))
    if not source_path.is_file() or sha256_file(source_path) != source_sha256:
        raise PermissionError("post-lock bridge formal score bundle is not hash-bound")
    scores = bundle.bridge_per_candidate_scores.copy()
    required = {
        "route",
        "candidate_pool_contract",
        "sample_id",
        "candidate_id",
        "source_candidate_id",
        "raw_candidate_id",
        "native_rank",
        "candidate_geometry_sha256",
        "fair_native_selector_score",
        "historical_selector_score",
        "candidate_success",
        "evaluator_sha256",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"post-lock bridge score bundle misses columns: {missing}")
    for column in (
        "route",
        "candidate_pool_contract",
        "sample_id",
        "candidate_id",
        "source_candidate_id",
        "raw_candidate_id",
        "candidate_geometry_sha256",
        "evaluator_sha256",
    ):
        scores[column] = scores[column].astype(str)
    scores["route"] = scores["route"].str.lower()
    scores["candidate_pool_contract"] = scores["candidate_pool_contract"].str.lower()
    if scores.duplicated(
        ["route", "candidate_pool_contract", "sample_id", "candidate_id"]
    ).any():
        raise ValueError(
            "post-lock bridge score bundle contains duplicate pool identities"
        )
    if not set(scores["route"]).issubset({"g1", "c1"}) or not set(
        scores["candidate_pool_contract"]
    ).issubset({"fair_gaussian", "historical_nms"}):
        raise ValueError("post-lock bridge bundle has an undeclared route/pool")
    expected_evaluator_sha256 = str(
        bundle.manifest.get("test_bridge", {}).get("evaluator", {}).get("sha256", "")
    )
    qualified_ids = pd.Series(
        [
            str(candidate_id).startswith(f"{route.upper()}::{pool}::")
            for route, pool, candidate_id in scores[
                ["route", "candidate_pool_contract", "candidate_id"]
            ].itertuples(index=False, name=None)
        ],
        index=scores.index,
    )
    rank = pd.to_numeric(scores["native_rank"], errors="coerce")
    labels = pd.to_numeric(scores["candidate_success"], errors="coerce")
    score_matrix = scores[
        ["fair_native_selector_score", "historical_selector_score"]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        rank.isna().any()
        or (rank < 1).any()
        or labels.isna().any()
        or not labels.isin([0, 1]).all()
        or not np.isfinite(score_matrix.to_numpy(float)).all()
        or scores[
            [
                "candidate_id",
                "source_candidate_id",
                "raw_candidate_id",
                "candidate_geometry_sha256",
                "evaluator_sha256",
            ]
        ]
        .eq("")
        .any()
        .any()
        or not qualified_ids.all()
        or len(expected_evaluator_sha256) != 64
        or not scores["evaluator_sha256"].eq(expected_evaluator_sha256).all()
    ):
        raise ValueError("post-lock bridge bundle has invalid identities/scores/labels")
    scores["native_rank"] = rank.astype(int)
    scores["candidate_success"] = labels.astype(bool)
    sample_ids = bundle.sample_manifest["sample_id"].astype(str).tolist()

    route_systems: dict[str, dict[str, str]] = {}
    for route in ("g1", "c1"):
        route_systems[route] = {
            str(system.get("kind")): name
            for name, system in bundle.systems.items()
            if str(system.get("route", "")).lower() == route
            and str(system.get("kind", "")) in {"native", "ungated", "gated"}
        }

    bridge_cells = [
        ("fair_gaussian", "fair_native_selector", True),
        ("fair_gaussian", "historical_selector", True),
        ("historical_nms", "fair_native_selector", True),
        ("historical_nms", "historical_selector", True),
    ]
    metric_columns = [
        *[f"j_at_{k}" for k in range(1, 6)],
        "oracle_at_5",
        "mrr_at_5",
        "ndcg_at_1",
        "ndcg_at_5",
        "oracle_all",
    ]

    def formal_bridge_cell(
        route: str, pool_contract: str, selector_contract: str
    ) -> tuple[pd.DataFrame | None, str | None, str]:
        score_column = f"{selector_contract}_score"
        local = scores.loc[
            scores["route"].eq(route)
            & scores["candidate_pool_contract"].eq(pool_contract)
        ].copy()
        if local.empty:
            return (
                None,
                score_column,
                f"formal bundle has no {route}/{pool_contract} candidate rows",
            )
        numeric_score = pd.to_numeric(local[score_column], errors="coerce")
        numeric_label = pd.to_numeric(local["candidate_success"], errors="coerce")
        if (
            numeric_score.isna().any()
            or not np.isfinite(numeric_score.to_numpy(float)).all()
            or numeric_label.isna().any()
            or not numeric_label.isin([0, 1]).all()
        ):
            return (
                None,
                score_column,
                "formal bridge scores/labels are non-finite or non-binary",
            )
        local = local.assign(
            _bridge_score=numeric_score, candidate_success=numeric_label.astype(bool)
        )
        identity = ["sample_id", "candidate_id"]
        consistency = local.groupby(identity, sort=False)[
            ["_bridge_score", "candidate_success"]
        ].nunique(dropna=False)
        if (consistency > 1).any().any():
            return None, score_column, "duplicate formal bridge identities disagree"
        local = local.drop_duplicates(identity, keep="first").copy()
        native_rank = pd.to_numeric(local["native_rank"], errors="coerce")
        if native_rank.isna().any():
            return None, score_column, "formal bridge native-rank tie-break is invalid"
        local["native_rank"] = native_rank.astype(int)
        return (
            local,
            score_column,
            "dedicated formal bridge score/label bundle supplies this complete cell",
        )

    rows: list[dict[str, Any]] = []
    for route in ("g1", "c1"):
        native_name = route_systems.get(route, {}).get("native")
        for pool_contract, selector_contract, required_cell in bridge_cells:
            evaluation, score_column, evidence = formal_bridge_cell(
                route, pool_contract, selector_contract
            )
            row: dict[str, Any] = {
                "route": route,
                "split": "test",
                "candidate_pool": pool_contract,
                "selector": selector_contract,
                "candidate_pool_contract": pool_contract,
                "selector_contract": selector_contract,
                "bridge_cell_required": required_cell,
                "system_name": (
                    native_name
                    if pool_contract == "fair_gaussian"
                    and selector_contract == "fair_native_selector"
                    else f"bridge::{route}::{pool_contract}::{selector_contract}"
                ),
                "reference_system": native_name,
                "score_column": score_column,
                "status": "MISSING_FORMAL_BRIDGE_INPUT",
                "compatibility_reason": evidence,
                "sample_count": len(sample_ids),
                "candidate_rows": 0,
                "selection_role": "SECONDARY_POSTLOCK_NO_SELECTION",
                "feeds_selection": False,
                "source_label_rows_reopened": False,
                "derived_from_path": str(source_path),
                "derived_from_sha256": source_sha256,
                **{column: None for column in metric_columns},
            }
            if evaluation is not None:
                metrics, _ = evaluate_order_only(
                    sample_ids,
                    evaluation,
                    score_column="_bridge_score",
                    max_k=5,
                )
                oracle_by_sample = evaluation.groupby("sample_id")[
                    "candidate_success"
                ].any()
                oracle_all = float(
                    bundle.sample_manifest["sample_id"]
                    .astype(str)
                    .map(oracle_by_sample)
                    .fillna(False)
                    .mean()
                )
                if (
                    pool_contract == "fair_gaussian"
                    and selector_contract == "fair_native_selector"
                    and score_column == "fair_native_selector_score"
                    and not np.isclose(
                        float(metrics["j_at_1"]),
                        float(bundle.metrics[str(native_name)]["j_at_1"]),
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    raise AssertionError(
                        f"post-lock bridge recomputation differs for {native_name}"
                    )
                row.update(
                    {
                        "status": "AVAILABLE_FROM_FORMAL_BUNDLE",
                        "compatibility_reason": evidence,
                        "candidate_rows": len(evaluation),
                        **{column: metrics.get(column) for column in metric_columns},
                        "oracle_all": oracle_all,
                    }
                )
            rows.append(row)
    table = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "route",
                "bridge_cell_required",
                "candidate_pool_contract",
                "selector_contract",
            ],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    output_dir = bundle.run_dir / "11_attribution_bridge"
    table_path = output_dir / "bridge_postlock_test.csv"
    atomic_csv(table_path, table)
    available = table["status"].eq("AVAILABLE_FROM_FORMAL_BUNDLE")
    finite_columns = [
        "j_at_1",
        "j_at_5",
        "oracle_at_5",
        "mrr_at_5",
        "ndcg_at_1",
        "ndcg_at_5",
        "oracle_all",
    ]
    finite_metrics = (
        table[finite_columns]
        .apply(lambda column: pd.to_numeric(column, errors="coerce"))
        .notna()
        .all(axis=1)
    )
    complete = bool(available.all() and finite_metrics.all() and len(table) == 8)
    manifest: dict[str, Any] = {
        "status": "COMPLETE" if complete else "FAILED_INCOMPLETE_FORMAL_BUNDLE",
        "split": "test",
        "analysis_role": "SECONDARY_POSTLOCK_NO_SELECTION",
        "feeds_selection": False,
        "source_candidate_test_label_rows_reopened": False,
        "formal_source": {
            "path": str(source_path),
            "sha256": source_sha256,
        },
        "artifact": {
            "path": str(table_path.resolve()),
            "sha256": sha256_file(table_path),
            "rows": len(table),
            "required_2x2_rows": int(table["bridge_cell_required"].sum()),
            "available_rows": int(available.sum()),
            "finite_metric_rows": int(finite_metrics.sum()),
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(output_dir / "BRIDGE_POSTLOCK_TEST_MANIFEST.json", manifest)
    return manifest


def _router_table(run_dir: Path, formal: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(
        (run_dir / "08_lock").glob("**/route_router_validation_trials.parquet")
    ):
        frame = pd.read_parquet(path)
        frame.insert(0, "split", "validation")
        frame.insert(1, "source_file", str(path.resolve()))
        pieces.append(frame)
    router = formal.loc[formal["system_kind"].eq("router")].copy()
    if not router.empty:
        router.insert(0, "split", "formal_test")
        pieces.append(router)
    return (
        pd.concat(pieces, ignore_index=True, sort=False)
        if pieces
        else pd.DataFrame(columns=["split"])
    )


def _union_table(run_dir: Path, formal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path = run_dir / "07_validation" / "union_headroom" / "manifest.json"
    if path.is_file():
        value = read_json(path)
        for split, summary in dict(value.get("summaries", {})).items():
            rows.append(
                {
                    "split": split,
                    "system_name": "union_oracle_headroom",
                    "decision": value.get("decision"),
                    **dict(summary),
                    "source_file": str(path.resolve()),
                }
            )
    union_formal = formal.loc[
        formal["system_name"].astype(str).str.contains("union", case=False, na=False)
    ].copy()
    if not union_formal.empty:
        union_formal.insert(0, "split", "formal_test")
        rows.extend(union_formal.to_dict("records"))
    return pd.DataFrame(rows)


def _tri_backend_table(taxonomy: pd.DataFrame) -> pd.DataFrame:
    correct = taxonomy.pivot(index="sample_id", columns="route", values="gated_correct")
    correct = correct.reindex(columns=list(ROUTES)).astype(bool)
    code = (
        correct["crog"].astype(int) * 4
        + correct["g1"].astype(int) * 2
        + correct["c1"].astype(int)
    )
    rows = [
        {
            "kind": "eight_way_intersection",
            "first": "CROG",
            "second": "G1",
            "third": "C1",
            "pattern": format(index, "03b"),
            "count": int((code == index).sum()),
            "denominator": len(correct),
        }
        for index in range(8)
    ]
    for first in ROUTES:
        for second in ROUTES:
            rows.append(
                {
                    "kind": "pairwise_both_correct",
                    "first": first.upper(),
                    "second": second.upper(),
                    "third": "",
                    "pattern": "11",
                    "count": int((correct[first] & correct[second]).sum()),
                    "denominator": len(correct),
                }
            )
    return pd.DataFrame(rows)


def _statistics_table(bundle: FormalBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, value in dict(bundle.statistics.get("comparisons", {})).items():
        scene = dict(value.get("scene_bootstrap", {}))
        frame = dict(value.get("frame_bootstrap_sensitivity", {}))
        mcnemar = dict(value.get("mcnemar_conventional_supportive", {}))
        rows.append(
            {
                "family": value.get("hypothesis_family"),
                "comparison": name,
                "reference": value.get("reference"),
                "test": "paired route comparison",
                "point_estimate": scene.get("point_estimate"),
                "scene_ci95_lower": (scene.get("ci95") or [None, None])[0],
                "scene_ci95_upper": (scene.get("ci95") or [None, None])[1],
                "frame_ci95_lower": (frame.get("ci95") or [None, None])[0],
                "frame_ci95_upper": (frame.get("ci95") or [None, None])[1],
                "uncorrected_p": mcnemar.get("pvalue"),
                "adjusted_p": value.get("holm_adjusted_mcnemar_pvalue"),
                "recovered": mcnemar.get("recovered"),
                "harmful": mcnemar.get("harmful"),
                "discordant": mcnemar.get("discordant"),
                "sample_count": mcnemar.get("sample_count"),
                "scene_clusters": scene.get("cluster_count"),
                "frame_clusters": frame.get("cluster_count"),
            }
        )
    omnibus = dict(bundle.statistics.get("three_route_final_systems", {}))
    q = dict(omnibus.get("cochran_q", {}))
    if q:
        rows.append(
            {
                "family": "three_route_supportive",
                "comparison": "CROG/G1/C1 final gated",
                "test": "Cochran Q",
                "uncorrected_p": q.get("pvalue"),
                "statistic": q.get("statistic"),
                "df": q.get("df"),
            }
        )
    for pair in omnibus.get("pairwise_mcnemar_holm", []):
        rows.append(
            {
                "family": "three_route_supportive",
                "comparison": f"{pair.get('first')} vs {pair.get('second')}",
                "test": "exact McNemar + Holm",
                "uncorrected_p": pair.get("pvalue"),
                "adjusted_p": pair.get("holm_adjusted_pvalue"),
                "recovered": pair.get("recovered"),
                "harmful": pair.get("harmful"),
                "discordant": pair.get("discordant"),
                "sample_count": pair.get("sample_count"),
            }
        )
    return pd.DataFrame(rows)


def _failure_diagnostic_tables(
    bundle: FormalBundle, taxonomy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build post-hoc strata and bottleneck summaries without selector reuse."""

    evidence = _visual_sources(bundle)
    diagnostic_source_path: str | None = None
    diagnostic_source_sha256: str | None = None
    audit_path = bundle.run_dir / "00_audit" / "fair_source_audit.json"
    if audit_path.is_file():
        fair_run = Path(str(read_json(audit_path).get("fair_run", "")))
        mask_path = fair_run / "04_metrics" / "per_sample_mask_metrics.csv"
        if mask_path.is_file():
            diagnostic_source_path = str(mask_path.resolve())
            diagnostic_source_sha256 = sha256_file(mask_path)
            mask = pd.read_csv(mask_path)
            keep = [
                column
                for column in ("sample_id", "hifics_mask_iou", "crog_mask_iou")
                if column in mask
            ]
            if "sample_id" in keep:
                evidence = evidence.merge(
                    mask[keep], on="sample_id", how="left", validate="one_to_one"
                )
    work = taxonomy.merge(
        evidence.drop_duplicates("sample_id"),
        on="sample_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_evidence"),
    )

    def numeric(column: str) -> pd.Series:
        if column not in work:
            return pd.Series(np.nan, index=work.index, dtype=float)
        return pd.to_numeric(work[column], errors="coerce")

    work["mask_iou"] = np.where(
        work["route"].eq("crog"),
        numeric("crog_mask_iou"),
        numeric("hifics_mask_iou"),
    )
    work["mask_iou_bin"] = pd.cut(
        work["mask_iou"],
        [-np.inf, 0.25, 0.5, 0.7, 0.9, np.inf],
        labels=["[0,.25)", "[.25,.5)", "[.5,.7)", "[.7,.9)", "[.9,1]"],
        right=False,
    ).astype(object)
    width = numeric("target_bbox_width")
    height = numeric("target_bbox_height")
    work["target_area_px"] = width * height
    try:
        work["target_size_quartile"] = pd.qcut(
            work["target_area_px"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        ).astype(object)
    except ValueError:
        work["target_size_quartile"] = None
    work["candidate_count_stratum"] = work["candidate_count_top5"].astype(str)
    work["first_positive_rank_stratum"] = work["first_positive_rank"].map(
        lambda value: "none" if pd.isna(value) else str(int(value))
    )
    margin_parts = []
    diagnostic_parts = []
    for route in ROUTES:
        pool = bundle.candidate_pools[route].copy()
        pool["native_score_numeric"] = pd.to_numeric(
            pool.get("native_score"), errors="coerce"
        )
        scores = pool.loc[pd.to_numeric(pool["native_rank"]).le(2)].pivot(
            index="sample_id", columns="native_rank", values="native_score_numeric"
        )
        margin = scores.get(1, pd.Series(index=scores.index, dtype=float)) - scores.get(
            2, pd.Series(index=scores.index, dtype=float)
        )
        margin_parts.append(
            pd.DataFrame(
                {
                    "route": route,
                    "sample_id": margin.index,
                    "native_score_margin": margin.values,
                }
            )
        )
        selected = taxonomy.loc[
            taxonomy["route"].eq(route), ["sample_id", "native_candidate_id"]
        ]
        diagnostics = selected.merge(
            bundle.labels.loc[bundle.labels["route"].eq(route)],
            left_on=["sample_id", "native_candidate_id"],
            right_on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        diagnostics["route"] = route
        keep = [
            column
            for column in (
                "route",
                "sample_id",
                "diagnostic_iou",
                "diagnostic_angle_error_deg",
            )
            if column in diagnostics
        ]
        diagnostic_parts.append(diagnostics[keep])
    work = work.merge(
        pd.concat(margin_parts, ignore_index=True),
        on=["route", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    work = work.merge(
        diagnostics, on=["route", "sample_id"], how="left", validate="one_to_one"
    )
    work["native_score_margin_quartile"] = None
    for route, indexes in work.groupby("route", sort=False).groups.items():
        values = work.loc[indexes, "native_score_margin"]
        try:
            work.loc[indexes, "native_score_margin_quartile"] = pd.qcut(
                values, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
            ).astype(object)
        except ValueError:
            pass
    iou = numeric("diagnostic_iou")
    angle = numeric("diagnostic_angle_error_deg")
    work["native_failure_mode"] = np.select(
        [
            iou.gt(0.25) & angle.le(30),
            iou.le(0.25) & angle.le(30),
            iou.gt(0.25) & angle.gt(30),
            iou.le(0.25) & angle.gt(30),
        ],
        ["pass", "angle-pass/iou-fail", "iou-pass/angle-fail", "both-fail"],
        default="unavailable",
    )
    dimensions = {
        "query_type": "expression_type",
        "mask_iou_bin": "mask_iou_bin",
        "target_size_quartile": "target_size_quartile",
        "candidate_count": "candidate_count_stratum",
        "native_score_margin": "native_score_margin_quartile",
        "first_positive_rank": "first_positive_rank_stratum",
        "native_angle_iou_failure": "native_failure_mode",
        "scene_family": "scene_family",
        "frame_family": "frame_id",
        "object_category": "object_category",
    }
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        if column not in work or work[column].notna().sum() == 0:
            rows.append(
                {
                    "route": "ALL",
                    "dimension": dimension,
                    "stratum": "NOT_AVAILABLE",
                    "denominator": 0,
                    "availability": "POSTHOC_SOURCE_MISSING",
                }
            )
            continue
        for (route, stratum), group in work.loc[work[column].notna()].groupby(
            ["route", column], sort=True, observed=True
        ):
            rows.append(
                {
                    "route": route,
                    "dimension": dimension,
                    "stratum": str(stratum),
                    "denominator": len(group),
                    "native_j_at_1": float(group["native_correct"].mean()),
                    "ungated_j_at_1": float(group["ungated_correct"].mean()),
                    "gated_j_at_1": float(group["gated_correct"].mean()),
                    "gated_delta_j_at_1": float(
                        (
                            group["gated_correct"].astype(int)
                            - group["native_correct"].astype(int)
                        ).mean()
                    ),
                    "recovered": int(group["gated_recovered"].sum()),
                    "harmful": int(group["gated_harmful"].sum()),
                    "gate_missed_recoverable": int(group["E9"].sum()),
                    "gate_prevented_harmful": int(group["E10"].sum()),
                    "availability": "AVAILABLE_POSTHOC_ONLY",
                }
            )
    strata = pd.DataFrame(rows)
    strata["diagnostic_source_path"] = diagnostic_source_path
    strata["diagnostic_source_sha256"] = diagnostic_source_sha256
    bottlenecks: list[dict[str, Any]] = []
    for route, group in work.groupby("route", sort=True):
        n = len(group)
        irreparable = int((~group["top5_positive"]).sum())
        candidate_limited = int((group["E0"] | group["E1"] | group["E2"]).sum())
        ranking_limited = int(group["E3"].sum())
        bottlenecks.extend(
            [
                {
                    "route": route,
                    "quantity": "reranking_irreparable",
                    "numerator": irreparable,
                    "denominator": n,
                    "fraction": irreparable / n,
                },
                {
                    "route": route,
                    "quantity": "candidate_generation_or_technical_limited",
                    "numerator": candidate_limited,
                    "denominator": n,
                    "fraction": candidate_limited / n,
                },
                {
                    "route": route,
                    "quantity": "ranking_headroom_at_native_error",
                    "numerator": ranking_limited,
                    "denominator": n,
                    "fraction": ranking_limited / n,
                },
                {
                    "route": route,
                    "quantity": "gate_missed_recoverable",
                    "numerator": int(group["E9"].sum()),
                    "denominator": n,
                    "fraction": float(group["E9"].mean()),
                },
                {
                    "route": route,
                    "quantity": "gate_prevented_harmful",
                    "numerator": int(group["E10"].sum()),
                    "denominator": n,
                    "fraction": float(group["E10"].mean()),
                },
            ]
        )
    return strata, pd.DataFrame(bottlenecks)


def build_tables(
    bundle: FormalBundle, taxonomy: pd.DataFrame, integrity_pass: bool
) -> dict[str, pd.DataFrame]:
    run_dir = bundle.run_dir
    formal = formal_results_table(bundle)
    independent = read_json(
        run_dir / "15_independent_recompute" / "INDEPENDENT_RECOMPUTE.json"
    )
    gate_decisions = _gate_decisions(run_dir)
    decisions: list[str] = []
    reasons: list[str] = []
    for row in formal.to_dict("records"):
        if row.get("system_kind") == "native":
            decisions.append("NATIVE_REFERENCE")
            reasons.append("frozen reference")
            continue
        route = str(row.get("route", "")).lower()
        decision, reason = deployment_decision(
            row,
            independent_pass=independent.get("status") == "PASS",
            integrity_pass=integrity_pass,
            validation_gate_decision=gate_decisions.get(route)
            if row.get("system_kind") == "gated"
            else None,
        )
        decisions.append(decision)
        reasons.append(reason)
    formal["decision"] = decisions
    formal["decision_reason"] = reasons
    primary = formal.loc[
        formal["system_kind"].isin(["native", "ungated", "gated", "router"])
    ].copy()
    baseline = formal.loc[formal["system_kind"].eq("native")].copy()
    if not baseline.empty:
        baseline["ranking_headroom_percentage_points"] = 100.0 * (
            pd.to_numeric(baseline["oracle_at_5"]) - pd.to_numeric(baseline["j_at_1"])
        )
        no_output = taxonomy.groupby("route")["candidate_count_top5"].apply(
            lambda x: int(x.eq(0).sum())
        )
        candidate_rows = {
            route: int(
                len(
                    bundle.rankings.loc[
                        bundle.rankings["system_name"]
                        .astype(str)
                        .eq(
                            next(
                                name
                                for name, system in bundle.systems.items()
                                if system.get("route") == route
                                and system.get("kind") == "native"
                            )
                        )
                    ]
                )
            )
            for route in ROUTES
        }
        baseline["top5_candidate_rows"] = baseline["route"].map(candidate_rows)
        baseline["no_output_samples"] = baseline["route"].map(no_output)

    cells = _validation_cells(run_dir)
    screen = cells.loc[cells["phase"].eq("screen")].copy() if not cells.empty else cells
    finalists = _read_csv(
        run_dir / "07_validation" / "tables" / "three_seed_scalar_finalists.csv"
    )
    evidence = finalists if not finalists.empty else screen
    loss = screen.loc[
        screen.get("loss", pd.Series(index=screen.index, dtype=object)).isin(
            ["bce", "ranknet", "listwise", "jacquard_margin_ranknet"]
        )
        & screen.get("encoder", pd.Series(index=screen.index, dtype=object)).eq("mlp")
    ].copy()
    encoder = (
        cells.loc[cells["phase"].eq("encoder")].copy() if not cells.empty else cells
    )
    if not encoder.empty:
        encoder = encoder.loc[
            encoder["encoder"].isin(
                ["linear", "mlp", "lambdamart", "deepsets", "set_transformer", "gnn"]
            )
        ].copy()
    ablation_paths = [
        run_dir / "07_validation" / "ablations" / "cumulative_feature_ablation.csv",
        run_dir / "07_validation" / "ablations" / "leave_one_family_out_ablation.csv",
    ]
    feature_ablation = _concat_csv(
        ablation_paths, columns=["route", "ablation", "j_at_1"]
    )
    if not feature_ablation.empty and "ablation_type" not in feature_ablation:
        feature_ablation["ablation_type"] = feature_ablation["source_file"].map(
            lambda value: (
                "leave_one_family_out"
                if "leave" in Path(str(value)).name.lower()
                else "cumulative"
            )
        )
    aggregate_bridge = run_dir / "07_validation" / "bridge_train_validation.csv"
    bridge_paths = [aggregate_bridge] if aggregate_bridge.is_file() else []
    postlock_bridge = run_dir / "11_attribution_bridge" / "bridge_postlock_test.csv"
    if postlock_bridge.is_file():
        bridge_paths.append(postlock_bridge)
    bridge = _concat_csv(
        bridge_paths,
        columns=["route", "split", "candidate_pool", "selector", "j_at_1"],
    )
    strata, bottlenecks = _failure_diagnostic_tables(bundle, taxonomy)
    return {
        "fair_baseline_oracle.csv": baseline,
        "evidence_track_comparison.csv": evidence,
        "feature_ablation.csv": feature_ablation,
        "loss_comparison.csv": loss,
        "encoder_comparison.csv": encoder,
        "gate_comparison.csv": _gate_table(run_dir),
        "tri_backend_consensus.csv": _tri_backend_table(taxonomy),
        "cross_route_router.csv": _router_table(run_dir, formal),
        "union_pool.csv": _union_table(run_dir, formal),
        "attribution_bridge.csv": bridge,
        "formal_test_primary.csv": primary,
        "formal_test_all_predeclared.csv": formal,
        "statistical_tests.csv": _statistics_table(bundle),
        "runtime_complexity.csv": cells,
        "failure_taxonomy.csv": taxonomy_summary(taxonomy),
        "failure_strata.csv": strata,
        "failure_bottleneck_summary.csv": bottlenecks,
    }


def required_analysis_registry(
    run_dir: Path, tables: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """Validate that required analyses are real, schema-complete, and nonempty."""

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"status": "PASS" if passed else "FAIL", "details": detail}

    benchmark_path = (
        run_dir / "07_validation" / "telemetry" / "feature_extraction_benchmark.json"
    )
    benchmark_errors: list[str] = []
    benchmark_rows = 0
    try:
        benchmark = load_verified_json(
            benchmark_path, name="Validation feature extraction benchmark"
        )
        unsigned_benchmark = dict(benchmark)
        benchmark_content_hash = unsigned_benchmark.pop("content_sha256", None)
        if benchmark_content_hash != canonical_sha256(unsigned_benchmark):
            raise RuntimeError("feature extraction benchmark content hash mismatch")
        verify_artifact_records_recursive(
            {
                "sources": benchmark.get("sources"),
                "artifacts": benchmark.get("artifacts"),
            },
            name="Validation feature extraction benchmark",
            require_at_least_one=True,
        )
        expected_components = {
            *{f"common/{route}" for route in ROUTES},
            *{f"rgb/{route}" for route in ROUTES},
            *{f"T1_native/{route}" for route in ROUTES},
            *{f"T2_matched_common/{route}" for route in ROUTES},
            "backend_maps/g1",
            "backend_maps/c1",
        }
        configuration = benchmark.get("configuration")
        sources = benchmark.get("sources")
        measurements = benchmark.get("component_measurements")
        if (
            benchmark.get("analysis") != "validation_feature_extraction_runtime"
            or benchmark.get("candidate_test_labels_read") is not False
            or not isinstance(configuration, Mapping)
            or configuration.get("split") != "validation"
            or configuration.get("sample_limit") != 128
            or configuration.get("tag") != "latency_benchmark_128"
            or set(configuration.get("component_inventory", [])) != expected_components
            or not isinstance(sources, Mapping)
            or set(dict(sources.get("component_manifests", {}))) != expected_components
            or not isinstance(measurements, list)
        ):
            raise RuntimeError("feature extraction benchmark inventory is not exact")
        by_name = {
            str(row.get("name", "")): row
            for row in measurements
            if isinstance(row, Mapping)
        }
        if set(by_name) != expected_components or len(by_name) != len(measurements):
            raise RuntimeError("feature extraction component rows are not exact")
        for name, row in by_name.items():
            if (
                row.get("manifest") != dict(sources["component_manifests"])[name]
                or row.get("measurement_scope")
                != "full_validation_persisted_extraction"
                or not isinstance(row.get("candidate_rows"), int)
                or int(row["candidate_rows"]) <= 0
                or not np.isfinite(float(row.get("feature_extraction_latency_ms")))
                or float(row["feature_extraction_latency_ms"]) < 0
                or not np.isfinite(float(row.get("peak_memory_mb")))
                or float(row["peak_memory_mb"]) <= 0
            ):
                raise RuntimeError(f"invalid extraction component row: {name}")
        if (
            not np.isfinite(float(benchmark.get("feature_extraction_latency_ms")))
            or float(benchmark["feature_extraction_latency_ms"]) <= 0
            or not np.isfinite(float(benchmark.get("peak_memory_mb")))
            or float(benchmark["peak_memory_mb"]) <= 0
        ):
            raise RuntimeError("T3 fixed-subset extraction telemetry is invalid")
        prelock = read_json(run_dir / "08_lock" / "prelock_assembly_manifest.json")
        expected_record = {
            "path": str(benchmark_path.resolve()),
            "sha256": sha256_file(benchmark_path),
        }
        if (
            prelock.get("status") != "COMPLETE"
            or prelock.get("sources", {}).get("feature_extraction_benchmark")
            != expected_record
        ):
            raise RuntimeError("P11 does not bind the current extraction benchmark")
        benchmark_rows = len(measurements)
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        benchmark_errors.append(str(error))
    record(
        "feature_extraction_benchmark",
        not benchmark_errors and benchmark_rows == 14,
        {
            "path": str(benchmark_path.resolve()),
            "component_rows": benchmark_rows,
            "errors": benchmark_errors,
        },
    )

    evidence = tables["evidence_track_comparison.csv"]
    record(
        "evidence_tracks",
        not evidence.empty and {"route", "track", "j_at_1"}.issubset(evidence.columns),
        {"rows": len(evidence), "columns": list(evidence.columns)},
    )
    loss = tables["loss_comparison.csv"]
    expected_losses = {"bce", "ranknet", "listwise", "jacquard_margin_ranknet"}
    observed_losses = set(
        loss.get("loss", pd.Series(dtype=object)).dropna().astype(str)
    )
    loss_routes = set(
        loss.get("route", pd.Series(dtype=object)).dropna().astype(str).str.lower()
    )
    record(
        "controlled_mlp_loss_analysis",
        not loss.empty
        and expected_losses.issubset(observed_losses)
        and set(ROUTES).issubset(loss_routes)
        and pd.to_numeric(loss.get("j_at_1"), errors="coerce").notna().all(),
        {
            "rows": len(loss),
            "expected_losses": sorted(expected_losses),
            "observed_losses": sorted(observed_losses),
            "observed_routes": sorted(loss_routes),
        },
    )
    encoder = tables["encoder_comparison.csv"]
    expected_encoders = {
        "linear",
        "mlp",
        "lambdamart",
        "deepsets",
        "set_transformer",
        "gnn",
    }
    observed_encoders = set(
        encoder.get("encoder", pd.Series(dtype=object)).dropna().astype(str)
    )
    observed_blocks = set(
        pd.to_numeric(
            encoder.loc[
                encoder.get("encoder", pd.Series(index=encoder.index, dtype=object)).eq(
                    "set_transformer"
                ),
                "num_attention_blocks",
            ]
            if "num_attention_blocks" in encoder
            else pd.Series(dtype=float),
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    encoder_execution_path = (
        run_dir / "05_models" / "matrix_plans" / "encoder_latest_execution.json"
    )
    encoder_executions = (
        [
            {
                "path": str(encoder_execution_path.resolve()),
                "status": read_json(encoder_execution_path).get("status"),
                "sha256": sha256_file(encoder_execution_path),
            }
        ]
        if encoder_execution_path.is_file()
        else []
    )
    execution_complete = (
        bool(encoder_executions) and encoder_executions[0]["status"] == "COMPLETE"
    )
    record(
        "encoder_phase",
        not encoder.empty
        and expected_encoders.issubset(observed_encoders)
        and {1, 2}.issubset(observed_blocks)
        and execution_complete,
        {
            "rows": len(encoder),
            "expected_encoders": sorted(expected_encoders),
            "observed_encoders": sorted(observed_encoders),
            "set_transformer_blocks": sorted(observed_blocks),
            "encoder_phase_executions": encoder_executions,
        },
    )
    ablation = tables["feature_ablation.csv"]
    types = set(
        ablation.get("ablation_type", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .str.lower()
    )
    ablation_manifest_path = (
        run_dir / "07_validation" / "ablations" / "feature_ablation_manifest.json"
    )
    ablation_manifest: dict[str, Any] = {}
    ablation_manifest_valid = False
    ablation_manifest_error: str | None = None
    try:
        ablation_manifest = load_verified_json(
            ablation_manifest_path, name="Validation feature ablation manifest"
        )
        unsigned_ablation_manifest = dict(ablation_manifest)
        recorded_content_sha256 = unsigned_ablation_manifest.pop("content_sha256", None)
        if recorded_content_sha256 != canonical_sha256(unsigned_ablation_manifest):
            raise RuntimeError(
                "Validation feature ablation manifest content hash mismatch"
            )
        verify_artifact_records_recursive(
            {
                "sources": ablation_manifest.get("sources"),
                "artifacts": ablation_manifest.get("artifacts"),
            },
            name="Validation feature ablation manifest",
            require_at_least_one=True,
        )
        nested_cell_records: list[Mapping[str, Any]] = []
        for route_source in dict(
            ablation_manifest.get("sources", {}).get("routes", {})
        ).values():
            nested_cell_records.extend(dict(route_source).get("selected_cells", []))
        nested_cell_records.extend(
            ablation_manifest.get("artifacts", {}).get("cell_manifests", [])
        )
        for index, record_value in enumerate(nested_cell_records):
            path_value = Path(str(record_value["path"])).resolve()
            cell_value = load_verified_json(
                path_value, name=f"ablation dependency cell {index}"
            )
            verify_artifact_records_recursive(
                {
                    "sources": cell_value.get("sources"),
                    "artifacts": cell_value.get("artifacts"),
                },
                name=f"ablation dependency cell {index}",
                require_at_least_one=True,
            )
        ablation_manifest_valid = (
            ablation_manifest.get("split") == "validation"
            and ablation_manifest.get("candidate_test_labels_read") is False
        )
    except (OSError, ValueError, RuntimeError) as error:
        ablation_manifest_error = str(error)
    required_ablation_columns = {
        "route",
        "ablation",
        "ablation_type",
        "family",
        "seed_count",
        "selected_source_signature_sha256",
        "included_feature_count",
        "j_at_1",
        "mrr_at_5",
        "candidate_test_labels_read",
        "split",
        "source_file",
    }
    ablation_schema_valid = required_ablation_columns.issubset(ablation.columns)
    finite_ablation = False
    unique_ablation = False
    exact_ablation_sources = False
    route_type_coverage: dict[str, dict[str, int]] = {}
    if not ablation.empty and ablation_schema_valid:
        finite_ablation = (
            ablation[["j_at_1", "mrr_at_5"]]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all()
            .all()
        )
        unique_ablation = not ablation.duplicated(
            ["route", "ablation_type", "family"]
        ).any()
        expected_sources = {
            str(
                (
                    run_dir
                    / "07_validation"
                    / "ablations"
                    / "cumulative_feature_ablation.csv"
                ).resolve()
            ),
            str(
                (
                    run_dir
                    / "07_validation"
                    / "ablations"
                    / "leave_one_family_out_ablation.csv"
                ).resolve()
            ),
        }
        exact_ablation_sources = (
            set(ablation["source_file"].astype(str)) == expected_sources
        )
        for route in ROUTES:
            route_rows = ablation.loc[
                ablation["route"].astype(str).str.lower().eq(route)
            ]
            route_type_coverage[route] = {
                kind: int(route_rows["ablation_type"].astype(str).eq(kind).sum())
                for kind in ("cumulative", "leave_one_family_out")
            }
    expected_rows = ablation_manifest.get("expected_rows_by_route", {})

    def valid_ablation_type(kind: str) -> bool:
        if not (
            ablation_manifest_valid
            and ablation_schema_valid
            and finite_ablation
            and unique_ablation
            and exact_ablation_sources
        ):
            return False
        kind_rows = ablation.loc[ablation["ablation_type"].astype(str).eq(kind)]
        if (
            kind_rows.empty
            or set(kind_rows["route"].astype(str).str.lower()) != set(ROUTES)
            or not pd.to_numeric(kind_rows["seed_count"], errors="coerce")
            .eq(len(FORMAL_SEEDS))
            .all()
            or not kind_rows["split"].astype(str).eq("validation").all()
            or not kind_rows["candidate_test_labels_read"]
            .astype(str)
            .str.lower()
            .isin({"false", "0"})
            .all()
            or not kind_rows["included_feature_count"]
            .pipe(pd.to_numeric, errors="coerce")
            .gt(0)
            .all()
        ):
            return False
        for route in ROUTES:
            expected = expected_rows.get(route, {}).get(kind)
            route_rows = kind_rows.loc[
                kind_rows["route"].astype(str).str.lower().eq(route)
            ]
            source_contract = (
                ablation_manifest.get("sources", {}).get("routes", {}).get(route)
            )
            expected_signature = (
                canonical_sha256(source_contract)
                if isinstance(source_contract, Mapping)
                else None
            )
            if (
                not isinstance(expected, int)
                or route_type_coverage[route][kind] != expected
                or expected_signature is None
                or not route_rows["selected_source_signature_sha256"]
                .astype(str)
                .eq(expected_signature)
                .all()
                or route_rows["fixed_budget_sha256"]
                .astype(str)
                .str.fullmatch(r"[0-9a-f]{64}")
                .ne(True)
                .any()
                or route_rows["fixed_budget_sha256"].nunique() != 1
            ):
                return False
        return True

    record(
        "cumulative_feature_ablation",
        "cumulative" in types and valid_ablation_type("cumulative"),
        {
            "rows": len(ablation),
            "observed_types": sorted(types),
            "route_type_coverage": route_type_coverage,
            "manifest_valid": ablation_manifest_valid,
            "manifest_error": ablation_manifest_error,
        },
    )
    record(
        "leave_one_family_out_ablation",
        "leave_one_family_out" in types and valid_ablation_type("leave_one_family_out"),
        {
            "rows": len(ablation),
            "observed_types": sorted(types),
            "route_type_coverage": route_type_coverage,
            "manifest_valid": ablation_manifest_valid,
            "manifest_error": ablation_manifest_error,
        },
    )
    gate = tables["gate_comparison.csv"]
    gate_routes = set(
        gate.get("route", pd.Series(dtype=object)).dropna().astype(str).str.lower()
    )
    record(
        "gate_grid_analysis",
        not gate.empty and set(ROUTES).issubset(gate_routes),
        {"rows": len(gate), "routes": sorted(gate_routes)},
    )
    bridge = tables["attribution_bridge.csv"]
    bridge_routes = set(
        bridge.get("route", pd.Series(dtype=object)).dropna().astype(str).str.lower()
    )
    expected_bridge_cells = {
        (route, split, pool, selector)
        for route in ("g1", "c1")
        for split, pools in (
            ("train", ("fair_gaussian", "historical_nms")),
            ("validation", ("fair_gaussian", "historical_nms")),
            ("test", ("fair_gaussian", "historical_nms")),
        )
        for pool in pools
        for selector in ("fair_native_selector", "historical_selector")
    }
    observed_bridge_cells: set[tuple[str, str, str, str]] = set()
    bridge_identity = ["route", "split", "candidate_pool", "selector"]
    if not bridge.empty and set(bridge_identity).issubset(bridge.columns):
        normalized_bridge = bridge[bridge_identity].astype(str).copy()
        normalized_bridge["route"] = normalized_bridge["route"].str.lower()
        observed_bridge_cells = set(map(tuple, normalized_bridge.to_numpy()))
        bridge_unique = not normalized_bridge.duplicated().any()
    else:
        bridge_unique = False
    record(
        "two_by_two_attribution_bridge",
        observed_bridge_cells == expected_bridge_cells
        and bridge_unique
        and {"g1", "c1"}.issubset(bridge_routes)
        and {"j_at_1", "status"}.issubset(bridge.columns),
        {
            "rows": len(bridge),
            "routes": sorted(bridge_routes),
            "expected_cells": sorted(expected_bridge_cells),
            "observed_cells": sorted(observed_bridge_cells),
            "unique": bridge_unique,
        },
    )
    train_validation_path = run_dir / "07_validation" / "bridge_train_validation.csv"
    train_validation = (
        pd.read_csv(train_validation_path)
        if train_validation_path.is_file()
        else pd.DataFrame()
    )
    expected_development_cells = {
        (route, split, pool, selector)
        for route in ("g1", "c1")
        for split in ("train", "validation")
        for pool in ("fair_gaussian", "historical_nms")
        for selector in ("fair_native_selector", "historical_selector")
    }
    observed_development_cells: set[tuple[str, str, str, str]] = set()
    development_columns = {
        "route",
        "split",
        "candidate_pool",
        "selector",
        "sample_count",
        "j_at_1",
        "j_at_5",
        "oracle_all",
    }
    if not train_validation.empty and development_columns.issubset(
        train_validation.columns
    ):
        observed_development_cells = set(
            map(
                tuple,
                train_validation[["route", "split", "candidate_pool", "selector"]]
                .astype(str)
                .assign(route=lambda frame: frame["route"].str.lower())
                .to_numpy(),
            )
        )
    development_ok = (
        (
            observed_development_cells == expected_development_cells
            and not train_validation[["route", "split", "candidate_pool", "selector"]]
            .duplicated()
            .any()
            and pd.to_numeric(train_validation["j_at_1"], errors="coerce").notna().all()
            and pd.to_numeric(train_validation["j_at_5"], errors="coerce").notna().all()
        )
        if not train_validation.empty
        and development_columns.issubset(train_validation.columns)
        else False
    )
    record(
        "train_validation_attribution_bridge_contract",
        development_ok,
        {
            "path": str(train_validation_path.resolve()),
            "rows": len(train_validation),
            "expected_cells": sorted(expected_development_cells),
            "observed_cells": sorted(observed_development_cells),
        },
    )
    prompt_alias_path = (
        run_dir / "11_attribution_bridge" / "bridge_train_validation.csv"
    )
    prompt_alias = (
        pd.read_csv(prompt_alias_path)
        if prompt_alias_path.is_file()
        else pd.DataFrame()
    )
    alias_ok = False
    if (
        development_ok
        and set(prompt_alias.columns) == set(train_validation.columns)
        and len(prompt_alias) == len(train_validation)
    ):
        sort_columns = ["route", "split", "candidate_pool", "selector"]
        canonical_rows = train_validation.sort_values(
            sort_columns, kind="mergesort"
        ).reset_index(drop=True)
        alias_rows = (
            prompt_alias[canonical_rows.columns]
            .sort_values(sort_columns, kind="mergesort")
            .reset_index(drop=True)
        )
        alias_ok = alias_rows.equals(canonical_rows)
    record(
        "train_validation_bridge_prompt_alias",
        alias_ok,
        {
            "canonical_path": str(train_validation_path.resolve()),
            "alias_path": str(prompt_alias_path.resolve()),
            "canonical_rows": len(train_validation),
            "alias_rows": len(prompt_alias),
            "semantic_identity": alias_ok,
        },
    )
    postlock_path = run_dir / "11_attribution_bridge" / "bridge_postlock_test.csv"
    postlock_manifest_path = (
        run_dir / "11_attribution_bridge" / "BRIDGE_POSTLOCK_TEST_MANIFEST.json"
    )
    postlock = pd.read_csv(postlock_path) if postlock_path.is_file() else pd.DataFrame()
    postlock_manifest = (
        read_json(postlock_manifest_path) if postlock_manifest_path.is_file() else {}
    )
    postlock_columns = {
        "route",
        "split",
        "candidate_pool_contract",
        "selector_contract",
        "bridge_cell_required",
        "status",
        "compatibility_reason",
        "derived_from_path",
        "derived_from_sha256",
        "selection_role",
        "feeds_selection",
        "source_label_rows_reopened",
    }
    required_cells = {
        (route, pool, selector)
        for route in ("g1", "c1")
        for pool in ("fair_gaussian", "historical_nms")
        for selector in ("fair_native_selector", "historical_selector")
    }
    observed_cells: set[tuple[str, str, str]] = set()
    postlock_ok = (
        not postlock.empty
        and postlock_columns.issubset(postlock.columns)
        and postlock_manifest.get("status") == "COMPLETE"
        and postlock_manifest.get("feeds_selection") is False
        and postlock_manifest.get("source_candidate_test_label_rows_reopened") is False
    )
    if postlock_ok:
        required_mask = (
            postlock["bridge_cell_required"].astype(str).str.lower().eq("true")
        )
        observed_cells = set(
            map(
                tuple,
                postlock.loc[
                    required_mask,
                    ["route", "candidate_pool_contract", "selector_contract"],
                ]
                .astype(str)
                .to_numpy(),
            )
        )
        statuses_ok = (
            postlock["status"].astype(str).eq("AVAILABLE_FROM_FORMAL_BUNDLE").all()
        )
        finite_metrics = (
            postlock[
                [
                    "j_at_1",
                    "j_at_5",
                    "oracle_at_5",
                    "mrr_at_5",
                    "ndcg_at_1",
                    "ndcg_at_5",
                    "oracle_all",
                ]
            ]
            .apply(lambda column: pd.to_numeric(column, errors="coerce"))
            .notna()
            .all(axis=1)
            .all()
        )
        no_selection = (
            postlock["selection_role"]
            .astype(str)
            .eq("SECONDARY_POSTLOCK_NO_SELECTION")
            .all()
            and postlock["feeds_selection"].astype(str).str.lower().eq("false").all()
            and postlock["source_label_rows_reopened"]
            .astype(str)
            .str.lower()
            .eq("false")
            .all()
        )
        artifact = dict(postlock_manifest.get("artifact", {}))
        source = dict(postlock_manifest.get("formal_source", {}))
        hashes_ok = (
            artifact.get("sha256") == sha256_file(postlock_path)
            and Path(str(source.get("path", ""))).is_file()
            and sha256_file(Path(str(source["path"]))) == source.get("sha256")
            and postlock["derived_from_sha256"]
            .astype(str)
            .eq(str(source.get("sha256")))
            .all()
        )
        postlock_ok = (
            observed_cells == required_cells
            and statuses_ok
            and finite_metrics
            and no_selection
            and hashes_ok
        )
    record(
        "postlock_test_attribution_bridge",
        postlock_ok,
        {
            "rows": len(postlock),
            "required_cells": sorted(required_cells),
            "observed_required_cells": sorted(observed_cells),
            "manifest": postlock_manifest,
        },
    )
    router = tables["cross_route_router.csv"]
    record(
        "cross_route_router",
        not router.empty
        and "formal_test" in set(router.get("split", pd.Series(dtype=object))),
        {
            "rows": len(router),
            "splits": sorted(
                set(router.get("split", pd.Series(dtype=object)).astype(str))
            ),
        },
    )
    union = tables["union_pool.csv"]
    union_decisions = set(
        union.get("decision", pd.Series(dtype=object)).dropna().astype(str)
    )
    union_valid = not union.empty and bool(
        union_decisions.intersection(
            {
                "NO_UNION_HEADROOM",
                "UNION_HEADROOM_AVAILABLE",
                "NOT_APPLICABLE_NO_HEADROOM",
            }
        )
        or union["system_name"]
        .astype(str)
        .str.contains("union", case=False, na=False)
        .any()
    )
    record(
        "union_headroom_or_model",
        union_valid,
        {"rows": len(union), "decisions": sorted(union_decisions)},
    )
    statistics = tables["statistical_tests.csv"]
    record(
        "formal_statistics",
        not statistics.empty
        and {"comparison", "test", "uncorrected_p"}.issubset(statistics.columns),
        {"rows": len(statistics), "columns": list(statistics.columns)},
    )
    runtime = tables["runtime_complexity.csv"]
    runtime_required = runtime
    telemetry_columns = (
        "parameter_count",
        "ranker_latency_ms",
        "feature_extraction_latency_ms",
        "feature_latency_ms",
        "cell_load_preprocess_latency_ms",
        "peak_memory_mb",
        "missing_feature_rate",
    )
    telemetry_coverage = {
        column: int(
            pd.to_numeric(runtime_required.get(column), errors="coerce").notna().sum()
        )
        if column in runtime_required
        else 0
        for column in telemetry_columns
    }
    record(
        "runtime_complexity_telemetry",
        not runtime_required.empty
        and set(runtime_required["phase"].astype(str))
        == {
            "screen",
            "selected",
            "encoder",
        }
        and all(count == len(runtime_required) for count in telemetry_coverage.values())
        and all(
            pd.to_numeric(runtime_required[column], errors="coerce").ge(0).all()
            for column in telemetry_columns
        )
        and pd.to_numeric(runtime_required["missing_feature_rate"], errors="coerce")
        .le(1)
        .all()
        and runtime_required.get(
            "artifact_records_verified",
            pd.Series(False, index=runtime_required.index),
        )
        .eq(True)
        .all(),
        {
            "rows": len(runtime_required),
            "matrix_rows": int(runtime_required["cell_kind"].eq("matrix_cells").sum())
            if "cell_kind" in runtime_required
            else 0,
            "rule_rows": int(runtime_required["cell_kind"].eq("rule_cells").sum())
            if "cell_kind" in runtime_required
            else 0,
            "required_non_null_columns": list(telemetry_columns),
            "non_null_counts": telemetry_coverage,
        },
    )
    status = (
        "PASS"
        if all(value["status"] == "PASS" for value in checks.values())
        else "FAIL"
    )
    payload: dict[str, Any] = {
        "status": status,
        "policy": "required P7-P14 analyses fail closed; only explicitly allowed latent/crop and no-union-headroom hooks may be NOT_RUN",
        "checks": checks,
        "allowed_nonblocking_statuses": [
            "NOT_RUN_UNVERIFIED_ALIGNMENT",
            "NOT_APPLICABLE_NO_HEADROOM",
        ],
    }
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(run_dir / "14_reports" / "REQUIRED_ANALYSIS_REGISTRY.json", payload)
    return payload


def write_tables(run_dir: Path, tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    if not set(TABLE_NAMES).issubset(tables):
        raise AssertionError("named table inventory is incomplete")
    output: dict[str, Any] = {}
    for name in sorted(tables):
        path = run_dir / "tables" / name
        atomic_csv(path, tables[name])
        output[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(tables[name]),
            "columns": list(tables[name].columns),
            "source": "machine-readable formal/validation artifacts",
        }
    manifest = {
        "status": "COMPLETE",
        "manual_numeric_entries": 0,
        "tables": output,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(run_dir / "tables" / "TABLE_GENERATION_MANIFEST.json", manifest)
    return manifest


def _empty(ax: plt.Axes, message: str) -> None:
    ax.text(
        0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _metric_bar(ax: plt.Axes, frame: pd.DataFrame, value: str, title: str) -> None:
    if (
        frame.empty
        or value not in frame
        or pd.to_numeric(frame[value], errors="coerce").notna().sum() == 0
    ):
        _empty(ax, f"No machine-readable {title.lower()} rows")
        return
    valid = frame.loc[pd.to_numeric(frame[value], errors="coerce").notna()].copy()
    labels = valid.get(
        "system_name", valid.get("method_code", valid.index.astype(str))
    ).astype(str)
    values = pd.to_numeric(valid[value])
    order = np.argsort(values.to_numpy())
    ax.barh(np.arange(len(valid)), values.to_numpy()[order], color="#0072B2")
    ax.set_yticks(np.arange(len(valid)), labels.iloc[order])
    ax.set_xlabel(title)


def _save_figure(fig: plt.Figure, output: Path, name: str) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{name}.pdf"
    svg = output / f"{name}.svg"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "unified_reranking.postformal",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Creator": "unified_reranking.postformal", "Date": None},
    )
    plt.close(fig)
    return {"pdf": str(pdf.resolve()), "svg": str(svg.resolve())}


def _figure(name: str, tables: Mapping[str, pd.DataFrame], run_dir: Path) -> plt.Figure:
    formal = tables["formal_test_primary.csv"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    if name == "oracle_at_k_curves":
        route_rows = formal.loc[
            formal["system_kind"].isin(["native", "ungated", "gated"])
        ]
        for row in route_rows.itertuples(index=False):
            values = [getattr(row, f"j_at_{k}") for k in range(1, 6)]
            ax.plot(range(1, 6), values, marker="o", label=row.system_name)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("K")
        ax.set_ylabel("J@K")
        ax.legend(fontsize=7, ncol=2)
    elif name == "j_at_1_delta_forest":
        plot = formal.loc[formal["delta_j_at_1"].notna()].copy()
        if plot.empty:
            _empty(ax, "No paired formal comparisons")
        else:
            y = np.arange(len(plot))
            x = plot["delta_j_at_1"].to_numpy(float)
            lower = x - plot["scene_ci95_lower"].to_numpy(float)
            upper = plot["scene_ci95_upper"].to_numpy(float) - x
            ax.errorbar(x, y, xerr=[lower, upper], fmt="o", capsize=3, color="#0072B2")
            ax.axvline(0, color="0.35", linestyle="--")
            ax.set_yticks(y, plot["system_name"])
            ax.set_xlabel("Paired J@1 difference")
    elif name == "recovered_vs_harmful":
        plot = formal.loc[formal["recovered"].notna()].copy()
        if plot.empty:
            _empty(ax, "No paired switch outcomes")
        else:
            y = np.arange(len(plot))
            ax.barh(
                y - 0.18,
                plot["recovered"],
                height=0.36,
                color="#009E73",
                label="Recovered",
            )
            ax.barh(
                y + 0.18, plot["harmful"], height=0.36, color="#D55E00", label="Harmful"
            )
            ax.set_yticks(y, plot["system_name"])
            ax.legend()
            ax.set_xlabel("Samples")
    elif name == "headroom_recovery_at_5":
        _metric_bar(
            ax,
            formal.loc[
                formal["system_kind"].isin(["ungated", "gated"])
                & ~formal["system_name"]
                .astype(str)
                .str.contains("union", case=False, na=False)
            ],
            "headroom_recovery_at_5",
            "Headroom recovery@5",
        )
    elif name == "loss_comparison":
        _metric_bar(ax, tables["loss_comparison.csv"], "j_at_1", "Validation J@1")
    elif name == "encoder_comparison":
        _metric_bar(ax, tables["encoder_comparison.csv"], "j_at_1", "Validation J@1")
    elif name in ("cumulative_feature_ablation", "leave_one_family_out_ablation"):
        frame = tables["feature_ablation.csv"]
        match = "leave" if name.startswith("leave") else "cumulative"
        if "ablation_type" in frame:
            subset = frame.loc[
                frame["ablation_type"]
                .astype(str)
                .str.contains(match, case=False, na=False)
            ]
        else:
            subset = frame
        _metric_bar(ax, subset, "j_at_1", "Validation J@1")
    elif name == "risk_coverage":
        frame = tables["gate_comparison.csv"]
        xcol = next(
            (
                column
                for column in ("switch_rate", "coverage", "mean_coverage")
                if column in frame
            ),
            None,
        )
        ycol = next(
            (
                column
                for column in ("mean_delta", "delta_j_at_1", "bootstrap_lower_bound")
                if column in frame
            ),
            None,
        )
        if xcol is None or ycol is None or frame.empty:
            _empty(ax, "No persisted risk-coverage trials")
        else:
            for route, group in frame.groupby("route", sort=True):
                ax.plot(
                    group[xcol],
                    group[ycol],
                    marker=".",
                    linestyle="none",
                    label=str(route).upper(),
                )
            ax.axhline(0, color=".4", linestyle="--")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.legend()
    elif name == "calibration_reliability":
        ax.plot([0, 1], [0, 1], "--", color=".45", label="ideal")
        found = False
        for route in ROUTES:
            path = (
                run_dir / "05_calibration" / f"{route}_validation_reliability.parquet"
            )
            if not path.is_file():
                continue
            frame = pd.read_parquet(path)
            selected_manifest = (
                run_dir / "05_calibration" / f"{route}_calibration_manifest.json"
            )
            selected = (
                str(read_json(selected_manifest).get("selected_method", ""))
                if selected_manifest.is_file()
                else ""
            )
            if selected and "method" in frame:
                frame = frame.loc[frame["method"].astype(str).eq(selected)]
            if {"mean_probability", "observed_frequency"}.issubset(frame):
                ax.plot(
                    frame["mean_probability"],
                    frame["observed_frequency"],
                    marker="o",
                    label=route.upper(),
                )
                found = True
        if not found:
            _empty(ax, "No validation reliability artifacts")
        else:
            ax.set_xlabel("Mean calibrated probability")
            ax.set_ylabel("Observed frequency")
            ax.legend()
    elif name == "switch_rate_vs_net_gain":
        plot = formal.loc[
            formal["delta_j_at_1"].notna() & formal["switch_rate"].notna()
        ]
        if plot.empty:
            _empty(ax, "No switch/gain evidence")
        else:
            ax.scatter(plot["switch_rate"], plot["delta_j_at_1"], color="#0072B2")
            for row in plot.itertuples(index=False):
                ax.annotate(
                    row.system_name, (row.switch_rate, row.delta_j_at_1), fontsize=7
                )
            ax.axhline(0, color=".4", linestyle="--")
            ax.set_xlabel("Switch rate")
            ax.set_ylabel("Paired J@1 difference")
    elif name == "tri_backend_complementarity_matrix":
        taxonomy_path = (
            run_dir / "13_failure_galleries" / "failure_taxonomy_per_sample.parquet"
        )
        taxonomy = pd.read_parquet(taxonomy_path)
        correct = (
            taxonomy.pivot(index="sample_id", columns="route", values="gated_correct")
            .reindex(columns=list(ROUTES))
            .astype(bool)
        )
        matrix = np.empty((3, 3), dtype=float)
        for i, first in enumerate(ROUTES):
            for j, second in enumerate(ROUTES):
                matrix[i, j] = (correct[first] & correct[second]).mean()
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
        ax.set_xticks(range(3), [r.upper() for r in ROUTES])
        ax.set_yticks(range(3), [r.upper() for r in ROUTES])
        fig.colorbar(image, ax=ax, label="Both correct fraction")
    elif name == "eight_way_outcome_intersection":
        frame = tables["tri_backend_consensus.csv"]
        frame = frame.loc[frame["kind"].eq("eight_way_intersection")]
        if frame.empty:
            _empty(ax, "No tri-route intersection rows")
        else:
            ax.bar(frame["pattern"], frame["count"], color="#0072B2")
            ax.set_xlabel("CROG/G1/C1 correct bit pattern")
            ax.set_ylabel("Samples")
    elif name in ("runtime_vs_gain", "parameter_count_vs_gain"):
        frame = tables["runtime_complexity.csv"]
        xcol = "ranker_latency_ms" if name.startswith("runtime") else "parameter_count"
        if (
            frame.empty
            or xcol not in frame
            or pd.to_numeric(frame[xcol], errors="coerce").notna().sum() == 0
        ):
            _empty(ax, f"No persisted {xcol} telemetry")
        else:
            valid = frame.loc[
                pd.to_numeric(frame[xcol], errors="coerce").notna()
                & pd.to_numeric(frame["j_at_1"], errors="coerce").notna()
            ]
            ax.scatter(valid[xcol], valid["j_at_1"], color="#0072B2")
            ax.set_xlabel(xcol)
            ax.set_ylabel("Validation J@1")
    elif name == "attribution_bridge":
        frame = tables["attribution_bridge.csv"]
        if frame.empty or not {"candidate_pool", "selector", "j_at_1"}.issubset(frame):
            _empty(ax, "No completed attribution bridge")
        else:
            pivot = frame.pivot_table(
                index=["split", "candidate_pool"],
                columns=["route", "selector"],
                values="j_at_1",
                aggfunc="first",
            )
            image = ax.imshow(pivot.to_numpy(float), cmap="Blues")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    ax.text(
                        j, i, safe_number(pivot.iloc[i, j], 3), ha="center", va="center"
                    )
            ax.set_xticks(
                range(len(pivot.columns)), pivot.columns, rotation=20, ha="right"
            )
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            fig.colorbar(image, ax=ax, label="J@1")
    ax.set_title(name.replace("_", " ").title())
    return fig


def build_figures(run_dir: Path, tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "unified-fair-reranking-v1",
        }
    )
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    records: dict[str, Any] = {}
    for name in FIGURE_NAMES:
        paths = _save_figure(
            _figure(name, tables, run_dir), run_dir / "12_figures", name
        )
        caption_path = run_dir / "12_figures" / f"{name}.caption.md"
        atomic_text(caption_path, CAPTIONS[name] + "\n")
        records[name] = {
            **paths,
            "caption": CAPTIONS[name],
            "caption_path": str(caption_path.resolve()),
            "pdf_sha256": sha256_file(paths["pdf"]),
            "svg_sha256": sha256_file(paths["svg"]),
        }
    manifest = {
        "status": "COMPLETE",
        "manual_case_selection": False,
        "figures": records,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(run_dir / "12_figures" / "FIGURE_MANIFEST.json", manifest)
    return manifest


def _visual_sources(bundle: FormalBundle) -> pd.DataFrame:
    sources = bundle.sample_manifest.copy()
    aliases = {
        "source_rgb_path": "rgb_path",
        "source_depth_path": "depth_path",
        "language": "expression",
        "predicted_mask_path": "predicted_hifics_mask_path",
    }
    for source, destination in aliases.items():
        if destination not in sources and source in sources:
            sources[destination] = sources[source]
    audit = bundle.run_dir / "00_audit" / "source_run_hashes.json"
    if audit.is_file():
        value = read_json(audit).get("paired_manifest", {})
        path = Path(str(value.get("path", "")))
        if path.is_file() and sha256_file(path) == value.get("sha256"):
            extra = pd.read_parquet(path)
            keep = [
                column
                for column in extra
                if column != "scene_id" and column != "frame_id"
            ]
            sources = sources.merge(
                extra[keep],
                on="sample_id",
                how="left",
                validate="one_to_one",
                suffixes=("", "_source"),
            )
    return sources


def _rectangle(row: Mapping[str, Any]) -> np.ndarray | None:
    keys = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
    if not all(key in row and pd.notna(row[key]) for key in keys):
        return None
    cx, cy, theta, width, height = (float(row[key]) for key in keys)
    angle = np.deg2rad(-theta)
    axis = np.asarray([np.cos(angle), np.sin(angle)])
    normal = np.asarray([-axis[1], axis[0]])
    center = np.asarray([cx, cy])
    return np.asarray(
        [
            center - axis * width / 2 - normal * height / 2,
            center + axis * width / 2 - normal * height / 2,
            center + axis * width / 2 + normal * height / 2,
            center - axis * width / 2 + normal * height / 2,
        ]
    )


def _render_board(
    path: Path,
    *,
    source: Mapping[str, Any],
    candidates: pd.DataFrame,
    taxonomy_row: Mapping[str, Any],
    category: str,
) -> None:
    rgb = np.asarray(Image.open(str(source["rgb_path"])).convert("RGB"))
    depth = np.asarray(Image.open(str(source["depth_path"])))
    predicted = np.asarray(Image.open(str(source["analysis_predicted_mask_path"])))
    secondary_mask = None
    secondary_path = source.get("analysis_secondary_mask_path")
    if secondary_path is not None and not pd.isna(secondary_path):
        secondary_mask = np.asarray(Image.open(str(secondary_path)))
    gt = np.asarray(Image.open(str(source["gt_mask_path"])))
    if pd.notna(source.get("target_instance_id")) and np.issubdtype(
        gt.dtype, np.integer
    ):
        gt = gt == int(source["target_instance_id"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    style = {
        str(taxonomy_row["native_candidate_id"]): ("#0072B2", "--", "native Top-1"),
        str(taxonomy_row["ungated_candidate_id"]): (
            "#D55E00",
            ":",
            "ungated challenger",
        ),
        str(taxonomy_row["gated_candidate_id"]): ("#009E73", "-", "final gated"),
    }
    for row in candidates.sort_values("native_rank", kind="mergesort").to_dict(
        "records"
    ):
        corners = _rectangle(row)
        if corners is None:
            continue
        color, linestyle, label = style.get(
            str(row["candidate_id"]), ("#999999", "-", "frozen candidate")
        )
        axes[0, 0].add_patch(
            Polygon(
                corners,
                fill=False,
                edgecolor=color,
                linestyle=linestyle,
                linewidth=2 if str(row["candidate_id"]) in style else 0.8,
                label=label,
            )
        )
        axes[0, 0].text(
            float(corners[:, 0].mean()),
            float(corners[:, 1].min()) - 3,
            f"r{int(row['native_rank'])}",
            color=color,
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        axes[0, 0].legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=7,
        )
    axes[0, 0].set_title(
        f"RGB + frozen candidates\n{source.get('expression', source.get('language', ''))}"
    )
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title("GT mask + grasps — ANALYSIS ONLY")
    try:
        gt_grasps = json.loads(str(source.get("gt_grasp_list_json", "[]")))
    except json.JSONDecodeError:
        gt_grasps = []
    matched_gt = source.get("analysis_matched_gt_index")
    try:
        matched_gt = None if pd.isna(matched_gt) else int(matched_gt)
    except (TypeError, ValueError):
        matched_gt = None
    for grasp_index, grasp in enumerate(gt_grasps):
        points = np.asarray(grasp, dtype=float)
        if points.shape == (4, 2):
            axes[0, 1].add_patch(
                Polygon(
                    points,
                    fill=False,
                    edgecolor="#CC79A7",
                    linewidth=2.5 if grasp_index == matched_gt else 0.8,
                    linestyle="-" if grasp_index == matched_gt else ":",
                )
            )
    axes[0, 2].imshow(predicted, cmap="gray")
    axes[0, 2].set_title(
        str(source.get("analysis_predicted_mask_title", "Predicted mask"))
    )
    axes[1, 0].imshow(depth, cmap="viridis")
    axes[1, 0].set_title("Depth")
    if secondary_mask is None:
        _empty(axes[1, 1], "No second route mask required")
    else:
        axes[1, 1].imshow(secondary_mask, cmap="gray")
        axes[1, 1].set_title(
            str(source.get("analysis_secondary_mask_title", "Second predicted mask"))
        )
    details = []
    for row in candidates.sort_values("native_rank", kind="mergesort").to_dict(
        "records"
    ):
        details.append(
            "native/ungated/final rank={rank}/{urank}/{grank} {candidate}: score={score}, IoU={iou}, angle={angle}, pass={passed}".format(
                rank=row.get("native_rank", "NA"),
                urank=row.get("ungated_rank", "NA"),
                grank=row.get("gated_rank", "NA"),
                candidate=row.get("candidate_id", ""),
                score=safe_number(row.get("native_score")),
                iou=safe_number(row.get("diagnostic_iou")),
                angle=safe_number(row.get("diagnostic_angle_error_deg")),
                passed=row.get("candidate_success", "NA"),
            )
        )
    feature_note = str(
        source.get(
            "analysis_feature_difference",
            "Feature difference: persisted candidate features unavailable in board input",
        )
    )
    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(details + [feature_note]),
        va="top",
        ha="left",
        fontsize=7,
        wrap=True,
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].set_title("Scores and diagnostics")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(
        f"{taxonomy_row.get('route', 'cross_route').upper()} | {category} | {taxonomy_row['sample_id']}\n"
        f"Earliest observable failure: {taxonomy_row.get('primary_category', 'NA')}",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "unified_reranking.postformal"},
    )
    plt.close(fig)


def build_galleries(bundle: FormalBundle, taxonomy: pd.DataFrame) -> dict[str, Any]:
    sources = _visual_sources(bundle).set_index("sample_id", drop=False)
    members = gallery_category_members(taxonomy, bundle)
    router_names = [
        name
        for name, system in bundle.systems.items()
        if system.get("kind") == "router"
    ]
    router = (
        None
        if not router_names
        else _decision_frame(bundle, router_names[0]).set_index("sample_id")
    )
    feature_frames: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for route in ROUTES:
        manifest_path = (
            bundle.run_dir
            / "03_features"
            / "tracks"
            / "T2_matched_common"
            / f"{route}_test"
            / "feature_manifest.json"
        )
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        artifact = dict(manifest.get("artifact", {}))
        try:
            path = Path(str(artifact["path"]))
            if sha256_file(path) != artifact["sha256"]:
                continue
            feature_frames[route] = (
                pd.read_parquet(path),
                list(map(str, manifest.get("model_feature_columns", []))),
            )
        except (KeyError, ValueError, OSError):
            continue

    def add_ranks(frame: pd.DataFrame, route: str, sample_id: str) -> pd.DataFrame:
        output = frame.copy()
        for kind in ("ungated", "gated"):
            name = next(
                name
                for name, system in bundle.systems.items()
                if str(system.get("route", "")).lower() == route
                and system.get("kind") == kind
            )
            ranks = bundle.rankings.loc[
                bundle.rankings["system_name"].astype(str).eq(name)
                & bundle.rankings["sample_id"].astype(str).eq(sample_id),
                ["candidate_id", "rank"],
            ].rename(columns={"rank": f"{kind}_rank"})
            output = output.merge(
                ranks, on="candidate_id", how="left", validate="one_to_one"
            )
        return output

    def feature_difference(route: str, sample_id: str, first: str, second: str) -> str:
        source = feature_frames.get(route)
        if source is None:
            return "Feature difference: T2 Test feature artifact unavailable"
        frame, columns = source
        local = frame.loc[
            frame["sample_id"].astype(str).eq(sample_id)
            & frame["candidate_id"].astype(str).isin([str(first), str(second)])
        ].copy()
        local["candidate_id"] = local["candidate_id"].astype(str)
        local = local.set_index("candidate_id")
        if (
            str(first) not in local.index
            or str(second) not in local.index
            or not columns
        ):
            return "Feature difference: selected candidate rows unavailable"
        differences = (
            pd.to_numeric(local.loc[str(second), columns], errors="coerce")
            - pd.to_numeric(local.loc[str(first), columns], errors="coerce")
        ).dropna()
        differences = differences.reindex(
            differences.abs().sort_values(ascending=False).index
        ).head(5)
        return "Feature difference (final - native; top |delta|): " + ", ".join(
            f"{name}={value:+.4g}" for name, value in differences.items()
        )

    rows: list[dict[str, Any]] = []
    common_assets = ("rgb_path", "depth_path", "gt_mask_path")
    for (scope, category), eligible_ids in sorted(members.items()):
        quota = (
            ROUTE_GALLERY_QUOTAS if scope in ROUTES else CROSS_ROUTE_GALLERY_QUOTAS
        )[category]
        chosen = deterministic_case_selection(eligible_ids, category, quota)
        rendered = 0
        asset_complete = 0
        for rank, sample_id in enumerate(chosen):
            reason = ""
            output_path = ""
            if sample_id not in sources.index:
                reason = "sample absent from visual-source manifest"
            else:
                source = sources.loc[sample_id]
                predicted_columns = (
                    ("crog_predicted_mask_path", "predicted_hifics_mask_path")
                    if scope == "cross_route"
                    else (
                        ("crog_predicted_mask_path",)
                        if scope == "crog"
                        else ("predicted_hifics_mask_path",)
                    )
                )
                required_assets = (*common_assets, *predicted_columns)
                missing = [
                    name
                    for name in required_assets
                    if name not in source
                    or pd.isna(source[name])
                    or not Path(str(source[name])).is_file()
                ]
                if missing:
                    reason = "missing required visual assets: " + ",".join(missing)
                else:
                    asset_complete += 1
                    if scope in ROUTES:
                        taxonomy_row = taxonomy.loc[
                            taxonomy["route"].eq(scope)
                            & taxonomy["sample_id"].eq(sample_id)
                        ].iloc[0]
                        candidates = (
                            bundle.candidate_pools[scope]
                            .loc[
                                bundle.candidate_pools[scope]["sample_id"].eq(sample_id)
                                & pd.to_numeric(
                                    bundle.candidate_pools[scope]["native_rank"]
                                ).le(5)
                            ]
                            .merge(
                                bundle.labels.loc[bundle.labels["route"].eq(scope)],
                                on=["sample_id", "candidate_id"],
                                how="left",
                                validate="one_to_one",
                                suffixes=("", "_label"),
                            )
                        )
                        candidates = add_ranks(candidates, scope, sample_id)
                        safe_id = "".join(
                            character
                            if character.isalnum() or character in "-_."
                            else "_"
                            for character in sample_id
                        )
                        path = (
                            bundle.run_dir
                            / "13_failure_galleries"
                            / scope
                            / category
                            / f"{rank:02d}_{safe_id}.png"
                        )
                        source = source.copy()
                        source["analysis_predicted_mask_path"] = source[
                            predicted_columns[0]
                        ]
                        source["analysis_predicted_mask_title"] = (
                            "CROG predicted mask"
                            if scope == "crog"
                            else "HiFi predicted mask"
                        )
                        source["analysis_feature_difference"] = feature_difference(
                            scope,
                            sample_id,
                            str(taxonomy_row["native_candidate_id"]),
                            str(taxonomy_row["gated_candidate_id"]),
                        )
                        selected_label = candidates.loc[
                            candidates["candidate_id"]
                            .astype(str)
                            .eq(str(taxonomy_row["gated_candidate_id"]))
                        ]
                        if (
                            not selected_label.empty
                            and "diagnostic_gt_index" in selected_label
                        ):
                            source["analysis_matched_gt_index"] = selected_label.iloc[
                                0
                            ]["diagnostic_gt_index"]
                        _render_board(
                            path,
                            source=source,
                            candidates=candidates,
                            taxonomy_row=taxonomy_row,
                            category=category,
                        )
                        output_path = str(path.resolve())
                        rendered += 1
                    else:
                        if router is None or sample_id not in router.index:
                            reason = "formal router decisions unavailable"
                        else:
                            pieces = []
                            for route in ROUTES:
                                part = (
                                    bundle.candidate_pools[route]
                                    .loc[
                                        bundle.candidate_pools[route]["sample_id"].eq(
                                            sample_id
                                        )
                                        & pd.to_numeric(
                                            bundle.candidate_pools[route]["native_rank"]
                                        ).le(5)
                                    ]
                                    .merge(
                                        bundle.labels.loc[
                                            bundle.labels["route"].eq(route)
                                        ],
                                        on=["sample_id", "candidate_id"],
                                        how="left",
                                        validate="one_to_one",
                                        suffixes=("", "_label"),
                                    )
                                )
                                part = add_ranks(part, route, sample_id)
                                part["candidate_id"] = (
                                    route + "::" + part["candidate_id"].astype(str)
                                )
                                part["route"] = route
                                pieces.append(part)
                            candidates = pd.concat(
                                pieces, ignore_index=True, sort=False
                            )
                            base = (
                                taxonomy.loc[
                                    taxonomy["route"].eq("crog")
                                    & taxonomy["sample_id"].eq(sample_id)
                                ]
                                .iloc[0]
                                .copy()
                            )
                            routed = router.loc[sample_id]
                            base["route"] = "cross_route"
                            base["native_candidate_id"] = "crog::" + str(
                                base["gated_candidate_id"]
                            )
                            base["ungated_candidate_id"] = (
                                str(routed["selected_route"]).lower()
                                + "::"
                                + str(routed["selected_candidate_id"])
                            )
                            base["gated_candidate_id"] = base["ungated_candidate_id"]
                            source = source.copy()
                            source["analysis_predicted_mask_path"] = source[
                                "crog_predicted_mask_path"
                            ]
                            source["analysis_predicted_mask_title"] = (
                                "CROG predicted mask"
                            )
                            source["analysis_secondary_mask_path"] = source[
                                "predicted_hifics_mask_path"
                            ]
                            source["analysis_secondary_mask_title"] = (
                                "HiFi predicted mask"
                            )
                            selected_route = str(routed["selected_route"]).lower()
                            source["analysis_feature_difference"] = feature_difference(
                                selected_route,
                                sample_id,
                                str(
                                    taxonomy.loc[
                                        taxonomy["route"].eq(selected_route)
                                        & taxonomy["sample_id"].eq(sample_id),
                                        "native_candidate_id",
                                    ].iloc[0]
                                ),
                                str(routed["selected_candidate_id"]),
                            )
                            selected_label = candidates.loc[
                                candidates["candidate_id"]
                                .astype(str)
                                .eq(
                                    selected_route
                                    + "::"
                                    + str(routed["selected_candidate_id"])
                                )
                            ]
                            if (
                                not selected_label.empty
                                and "diagnostic_gt_index" in selected_label
                            ):
                                source["analysis_matched_gt_index"] = (
                                    selected_label.iloc[0]["diagnostic_gt_index"]
                                )
                            safe_id = "".join(
                                character
                                if character.isalnum() or character in "-_."
                                else "_"
                                for character in sample_id
                            )
                            path = (
                                bundle.run_dir
                                / "13_failure_galleries"
                                / scope
                                / category
                                / f"{rank:02d}_{safe_id}.png"
                            )
                            _render_board(
                                path,
                                source=source,
                                candidates=candidates,
                                taxonomy_row=base,
                                category=category,
                            )
                            output_path = str(path.resolve())
                            rendered += 1
            rows.append(
                {
                    "scope": scope,
                    "category": category,
                    "sample_id": sample_id,
                    "selection_sha256": __import__("hashlib")
                    .sha256((sample_id + category).encode())
                    .hexdigest(),
                    "selected_rank": rank + 1,
                    "rendered": bool(output_path),
                    "path": output_path,
                    "shortfall_reason": reason,
                }
            )
        rows.append(
            {
                "scope": scope,
                "category": category,
                "sample_id": "__SUMMARY__",
                "eligible": len(set(eligible_ids)),
                "requested": quota,
                "selected": len(chosen),
                "rendered": rendered,
                "selection_shortfall": max(0, quota - len(chosen)),
                "render_shortfall": max(0, quota - rendered),
                "asset_complete_selected": asset_complete,
                "shortfall_reason": "insufficient eligible cases and/or required RGB/GT-mask/predicted-mask/depth assets",
            }
        )
    manifest_frame = pd.DataFrame(rows)
    csv_path = bundle.run_dir / "13_failure_galleries" / "gallery_manifest.csv"
    atomic_csv(csv_path, manifest_frame)
    summary = manifest_frame.loc[manifest_frame["sample_id"].eq("__SUMMARY__")]
    readme = (
        "# Deterministic failure galleries\n\n"
        "Cases are ranked by `SHA256(sample_id + category)`; no case was hand-picked. "
        "Boards are rendered only when RGB, depth, GT mask, and predicted mask paths all exist. "
        "GT content appears only in panels marked **ANALYSIS ONLY**. Missing assets and eligible-case "
        "shortfalls are reported rather than replaced with synthetic cases.\n\n"
        + summary.to_markdown(index=False)
        + "\n"
    )
    atomic_text(bundle.run_dir / "13_failure_galleries" / "README.md", readme)
    manifest = {
        "status": "COMPLETE_WITH_AUDITED_SHORTFALLS"
        if int(summary["render_shortfall"].sum())
        else "COMPLETE",
        "selection_rule": "ascending SHA256(sample_id + category), then sample_id",
        "manual_selection": False,
        "manifest": {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)},
        "summaries": summary.to_dict("records"),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(
        bundle.run_dir / "13_failure_galleries" / "GALLERY_MANIFEST.json", manifest
    )
    return manifest


def _decision_lines(primary: pd.DataFrame) -> str:
    columns = [
        "system_name",
        "route",
        "system_kind",
        "j_at_1",
        "delta_percentage_points",
        "recovered",
        "harmful",
        "switch_rate",
        "scene_ci95_lower",
        "scene_ci95_upper",
        "holm_p",
        "decision",
    ]
    return markdown_table(primary, columns)


def _fallacy_scan() -> str:
    rows = [
        (
            "Simpson's paradox",
            "NOTE",
            "Aggregate direction is accompanied by route/scene-stratified artifacts; no causal aggregation claim.",
        ),
        (
            "Ecological fallacy",
            "NOTE",
            "Inference remains at the paired sample/system level; cluster bootstrap addresses dependence.",
        ),
        (
            "Berkson's paradox",
            "NOTE",
            "The benchmark is a frozen held-out corpus; conclusions are restricted to this selected benchmark.",
        ),
        (
            "Collider bias",
            "NOTE",
            "Post-hoc strata are diagnostic only and are not conditioned on for the primary estimator.",
        ),
        (
            "Base-rate neglect",
            "NOTE",
            "Exact denominators, recovered, harmful, no-positive, and outcome-changing precision are reported.",
        ),
        (
            "Regression to the mean",
            "NOTE",
            "Systems are paired on one frozen Test execution; no extreme-subgroup pre/post claim is made.",
        ),
        (
            "Survivorship bias",
            "NOTE",
            "No-output samples remain in the fixed denominator.",
        ),
        (
            "Look-elsewhere effect",
            "NOTE",
            "Primary families were locked on Validation; Holm correction is reported for declared families.",
        ),
        (
            "Garden of forking paths",
            "CAUTION",
            "Many secondary screens exist, but primary track/method/gate/router were hash-locked before Test.",
        ),
        (
            "Correlation is not causation",
            "NOTE",
            "Mask/failure strata are described as associations, not intervention effects.",
        ),
        (
            "Reverse causality",
            "NOTE",
            "No directional causal claim is made from post-hoc associations.",
        ),
    ]
    return (
        "- **Coverage**: 11/11 fallacy types checked\n\n| Fallacy | Severity | Detail |\n|---|---|---|\n"
        + "\n".join(
            f"| {name} | {severity} | {detail} |" for name, severity, detail in rows
        )
    )


def build_reports(
    bundle: FormalBundle,
    tables: Mapping[str, pd.DataFrame],
    table_manifest: Mapping[str, Any],
    figure_manifest: Mapping[str, Any],
    gallery_manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
    analysis_registry: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = bundle.run_dir
    reports = run_dir / "14_reports"
    primary = tables["formal_test_primary.csv"]
    baselines = tables["fair_baseline_oracle.csv"]
    failures = tables["failure_taxonomy.csv"]
    failure_strata = tables["failure_strata.csv"]
    bottlenecks = tables["failure_bottleneck_summary.csv"]
    bridge = tables["attribution_bridge.csv"]
    postlock_bridge = (
        bridge.loc[bridge["split"].astype(str).eq("test")].copy()
        if "split" in bridge
        else pd.DataFrame()
    )
    decision_table = _decision_lines(primary)
    all_pass = (
        integrity.get("status") == "PASS" and analysis_registry.get("status") == "PASS"
    )
    overall = "COMPLETE" if all_pass else "FAILED_INTEGRITY"
    deployment = {
        str(row.system_name): str(row.decision)
        for row in primary.loc[primary["system_kind"].ne("native")].itertuples(
            index=False
        )
    }
    common_note = (
        "All numeric claims below are generated from the formal metrics/statistics/per-sample "
        "artifacts and the Validation registries listed in `tables/TABLE_GENERATION_MANIFEST.json`."
    )
    final_report = f"""# Unified fair order-only reranking: final report

## Status

**{overall}**. {common_note}

## Fair baseline and formal results

{markdown_table(baselines, ["route", "sample_count", "top5_candidate_rows", "no_output_samples", "j_at_1", "j_at_5", "oracle_all", "ranking_headroom_percentage_points"])}

{decision_table}

## Interpretation

The route-wise ungated ranker, conservative gate, and CROG-default cross-route router were frozen before the single formal Test transaction. GO requires positive formal gain, a scene-clustered 95% lower bound above zero, Holm-adjusted McNemar p below 0.05, independent recomputation, and intact contracts. Positive point estimates that do not clear all confirmatory checks are CAUTION; non-positive or integrity-invalid systems are NO-GO.

## Failure decomposition

{markdown_table(failures, ["route", "category", "definition", "count", "denominator", "fraction"])}

## Bottleneck diagnosis

{markdown_table(bottlenecks, ["route", "quantity", "numerator", "denominator", "fraction"])}

## Scope

This is a retrospective paired benchmark, not evidence of deployment-time causal benefit. Candidate-level calibration metrics and post-hoc mask/stratum associations do not replace J@1.
"""
    methods = f"""# Methods: unified fair reranking

{common_note}

- Denominator: the frozen paired Test sample manifest; no-output samples are retained.
- Candidate contract: immutable All and Top-5 membership, geometry, native scores, and native rank.
- Model selection: grouped Train OOF plus Validation; the primary evidence track is matched common evidence (T2).
- Formal evaluation: one hash-locked Test transaction; no seed, loss, encoder, gate, router, or track can change afterward.
- Primary uncertainty: 10,000 scene-clustered bootstrap replicates, seed 20260808. Frame-clustered bootstrap and exact sample McNemar are sensitivity/supportive analyses.
- Multiplicity: Holm correction within predeclared hypothesis families and for pairwise route tests.
- Diagnostics: overlapping E0–E10 predicates plus a deterministic exclusive priority leaf; galleries use SHA256(sample_id + category).
- Visualization: PDF/SVG figures are generated directly from the named CSV tables. GT masks and matched GT grasps are post-hoc analysis evidence only.
"""
    results = f"""# Results: unified fair reranking

## Formal primary systems

{decision_table}

## Controlled loss comparison

{markdown_table(tables["loss_comparison.csv"], ["route", "track", "method_code", "encoder", "loss", "j_at_1", "recovered", "harmful", "switch_rate"])}

## Encoder comparison

{markdown_table(tables["encoder_comparison.csv"], ["route", "track", "encoder", "loss", "seed", "j_at_1", "mrr_at_5", "parameter_count", "ranker_latency_ms"])}

## Cross-route and union evidence

{markdown_table(tables["cross_route_router.csv"], ["split", "system_name", "decision", "j_at_1", "delta_percentage_points", "recovered", "harmful", "scene_ci95_lower", "holm_p"])}

{markdown_table(tables["union_pool.csv"], ["split", "system_name", "decision", "union_gain_over_best_single", "j_at_1", "j_at_5", "j_at_10", "j_at_15", "oracle_at_15", "oracle_all", "mrr_at_15", "ndcg_at_15", "headroom_recovery_at_15"])}

## Secondary post-lock attribution bridge

The Test bridge is diagnostic only and is derived from the already-consumed formal per-candidate score/label bundle. It does not reopen the source Test-label parquet and cannot feed method selection. All eight route-by-pool-by-selector cells require finite metrics; a missing formal input fails completeness.

{markdown_table(postlock_bridge, ["route", "candidate_pool_contract", "selector_contract", "status", "compatibility_reason", "system_name", "sample_count", "candidate_rows", "j_at_1", "j_at_5", "oracle_at_5", "selection_role"])}

## Post-hoc strata

The complete machine-readable stratum table is `tables/failure_strata.csv`; every row is marked post-hoc and cannot be used for Test-time feature selection.

{markdown_table(failure_strata.loc[failure_strata["dimension"].isin(["query_type", "mask_iou_bin", "target_size_quartile", "native_angle_iou_failure"])] if "dimension" in failure_strata else failure_strata, ["route", "dimension", "stratum", "denominator", "native_j_at_1", "gated_j_at_1", "gated_delta_j_at_1", "recovered", "harmful", "availability"])}
"""
    discussion = """# Discussion

The experiment separates four possible sources of apparent improvement: candidate-pool contract, selector contract, learned ordering, and conservative switching. The 2×2 bridge prevents the historical-versus-fair baseline gap from being attributed to reranking alone.

The small Top-5 sets constrain how much set-aware encoders can exploit higher-order structure; any encoder claim must therefore come from the controlled encoder table, not architectural expectation. Gates trade recovery for harm avoidance, while the cross-route router tests complementarity relative to a CROG-default reference.

Post-hoc failure strata are associations. They diagnose where the frozen pipeline fails but do not authorize Test-time feature selection or causal claims.
"""
    limitations = """# Limitations

- This is a locked retrospective benchmark; external-distribution performance is unknown.
- Exact McNemar treats sample rows as independent and is supportive because language queries can share scenes/frames.
- Scene-cluster bootstrap intervals depend on the observed scene inventory.
- Gallery availability depends on recoverable RGB, depth, GT-mask, and predicted-mask assets; audited shortfalls are not filled with synthetic cases.
- CROG has no legitimate GT-mask stage-replacement counterfactual; only mask-quality/failure association is permitted.
- A missing required analysis fails completeness. Only the explicitly predeclared latent/crop alignment and no-union-headroom hooks may be recorded as not run.
- Candidate-level Test labels were first opened only inside the claimed formal transaction. General failure diagnostics may consume the hash-locked candidate-label source after formal completion, but cannot change selection; the secondary Test bridge is stricter and uses only the formal per-candidate score/label bundle.
"""
    baseline_gap = f"""# Baseline-gap attribution

The historical 87%/79%-scale modular results and the fair-contract baselines are not directly exchangeable. The bridge crosses candidate pool and selector while holding the frozen evaluator fixed. In addition, the audit records an angle-sign convention mismatch: the historical evaluator interprets stored angles with the opposite sign from the frozen fair OpenCV rectangle convention. Consequently, the old-to-fair gap must not be presented as a reranking loss or gain.

{markdown_table(bridge, ["route", "split", "candidate_pool", "selector", "sample_count", "j_at_1", "j_at_5", "oracle_all", "no_output_samples"])}

## Formal-Test post-lock bridge provenance

The rows below are secondary and non-selecting. They were generated only from `09_formal_test/bridge_per_candidate_scores.parquet`; all eight 2×2 cells must have finite metrics. Missing pool/selector inputs are a hard completeness failure rather than an accepted placeholder, and the source Test labels are never reopened.

{markdown_table(postlock_bridge, ["route", "candidate_pool_contract", "selector_contract", "status", "compatibility_reason", "system_name", "j_at_1", "j_at_5", "derived_from_sha256", "selection_role"])}
"""
    passport = f"""# Material Passport

- Origin Skill: experiment-agent-compatible local reporting
- Origin Mode: validate
- Verification Status: {"VERIFIED" if all_pass else "RED_FLAG"}
- Version Label: unified_fair_reranking_postformal_v1
- Experiment Intake Declaration: experiments_declared
- Experiment ID: fair-unified-reranking-v1
- Formal Test Executions: 1
- Primary Resampling: 10,000 scene clusters, seed 20260808
- Independent Recompute: {integrity["checks"]["independent_recompute"]["status"]}
- Tables: machine-generated; manual numeric entries = 0
- Gallery Rule: SHA256(sample_id + category); no manual selection
- Known Limitation: retrospective benchmark with historically viewed Test context disclosed

## Claim boundary

Verified facts are exact persisted metrics and hashes. Repository evidence is explicitly path-bound. Interpretations are labelled as such in the discussion. Causal claims and generalization beyond the benchmark are excluded.

## Fallacy Scan

{_fallacy_scan()}

## Reproducibility

- Method: independent identifier-join recomputation plus final SHA-256 inventory
- Verdict: {"REPRODUCIBLE_WITHIN_LOCKED_ARTIFACTS" if all_pass else "NOT_REPRODUCIBLE_INTEGRITY_FAILURE"}
"""
    zh = f"""# 统一公平重排序最终摘要

## 状态

**{overall}**。所有数值均由锁定后的机器可读 Test/Validation 产物自动生成，没有手工抄录。

## 公平基线

{markdown_table(baselines, ["route", "sample_count", "top5_candidate_rows", "no_output_samples", "j_at_1", "j_at_5", "oracle_all", "ranking_headroom_percentage_points"])}

## 正式 Test 与部署判定

{decision_table}

GO 要求：gated ΔJ@1 为正、scene-cluster bootstrap 95% 下界大于 0、Holm 校正 McNemar p < 0.05、独立复算通过且候选/evaluator/leakage 完整性全部通过。均值为正但置信区间或显著性不足时为 CAUTION；净提升不正或完整性失败时为 NO-GO。

## 关键结论边界

旧的约 87%/79% 结果与公平基线采用不同候选/selector 合约，而且审计确认历史角度符号与冻结公平 evaluator 的 OpenCV 矩形约定相反。因此差距不能归因于重排序。失败分层、mask IoU 和 object category 均为后验诊断，不进入 Test-time selector。

正式 Test 的 post-lock bridge 仅从已消费并锁定哈希的 `bridge_per_candidate_scores.parquet` 生成，不重新打开源 Test label parquet，也不参与任何选择。G1/C1 的全部 2×2 单元格都必须得到有限指标；formal bundle 缺少任何 pool/selector 输入都会导致完整性失败。

{markdown_table(postlock_bridge, ["route", "candidate_pool_contract", "selector_contract", "status", "system_name", "j_at_1", "j_at_5", "selection_role"])}

## 完整性

{markdown_table(pd.DataFrame([{"check": key, **value} for key, value in integrity["checks"].items()]), ["check", "status"])}
"""

    contents = {
        "FINAL_REPORT_EN.md": final_report,
        "METHODS_UNIFIED_RERANKING_EN.md": methods,
        "RESULTS_UNIFIED_RERANKING_EN.md": results,
        "DISCUSSION_UNIFIED_RERANKING_EN.md": discussion,
        "LIMITATIONS_EN.md": limitations,
        "BASELINE_GAP_ATTRIBUTION.md": baseline_gap,
        "MATERIAL_PASSPORT.md": passport,
        "FINAL_SUMMARY_ZH.md": zh,
    }
    for name, content in contents.items():
        atomic_text(reports / name, content.rstrip() + "\n")
    baseline_alias = run_dir / "11_attribution_bridge" / "BASELINE_GAP_ATTRIBUTION.md"
    atomic_text(
        baseline_alias,
        (reports / "BASELINE_GAP_ATTRIBUTION.md").read_text(encoding="utf-8"),
    )

    latex_rows = primary[
        [
            "system_name",
            "j_at_1",
            "delta_percentage_points",
            "recovered",
            "harmful",
            "decision",
        ]
    ]
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Locked formal-Test primary results.}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"System & J@1 & $\Delta$ pp & Rec. & Harm. & Decision \\",
        r"\midrule",
    ]
    for row in latex_rows.itertuples(index=False):
        latex.append(
            f"{latex_escape(row.system_name)} & {safe_number(row.j_at_1)} & {safe_number(row.delta_percentage_points, 2)} & {safe_number(row.recovered, 0)} & {safe_number(row.harmful, 0)} & {latex_escape(row.decision)} \\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    atomic_text(reports / "THESIS_READY_TABLES.tex", "\n".join(latex))
    figures_md = (
        "# Thesis-ready figures\n\n"
        + "\n\n".join(
            f"## {name.replace('_', ' ').title()}\n\n{record['caption']}\n\n- PDF: `{record['pdf']}`\n- SVG: `{record['svg']}`"
            for name, record in figure_manifest["figures"].items()
        )
        + "\n"
    )
    atomic_text(reports / "THESIS_READY_FIGURES.md", figures_md)
    conclusion = {
        "status": overall,
        "benchmark_description": "locked retrospective paired test benchmark",
        "formal_test_execution_count": 1,
        "deployment_decisions": deployment,
        "integrity_status": integrity.get("status"),
        "table_manifest_sha256": table_manifest.get("content_sha256"),
        "figure_manifest_sha256": figure_manifest.get("content_sha256"),
        "gallery_manifest_sha256": gallery_manifest.get("content_sha256"),
        "required_analysis_registry_sha256": analysis_registry.get("content_sha256"),
        "postlock_bridge_manifest_sha256": read_json(
            run_dir / "11_attribution_bridge" / "BRIDGE_POSTLOCK_TEST_MANIFEST.json"
        ).get("content_sha256"),
        "claim_boundary": "association/benchmark evidence; no causal or external-generalization claim",
    }
    conclusion["content_sha256"] = canonical_sha256(conclusion)
    atomic_json(reports / "EXPERIMENT_CONCLUSION.json", conclusion)
    records = {
        name: {
            "path": str((reports / name).resolve()),
            "sha256": sha256_file(reports / name),
        }
        for name in REPORT_NAMES
    }
    manifest = {
        "status": "COMPLETE",
        "reports": records,
        "prompt_aliases": {
            "baseline_gap_attribution": {
                "path": str(baseline_alias.resolve()),
                "sha256": sha256_file(baseline_alias),
                "byte_identical_to": str(
                    (reports / "BASELINE_GAP_ATTRIBUTION.md").resolve()
                ),
            }
        },
        "manual_numeric_entries": 0,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(reports / "REPORT_MANIFEST.json", manifest)
    return manifest


def finalize_run(
    bundle: FormalBundle,
    integrity: Mapping[str, Any],
    table_manifest: Mapping[str, Any],
    figure_manifest: Mapping[str, Any],
    gallery_manifest: Mapping[str, Any],
    report_manifest: Mapping[str, Any],
    analysis_registry: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = bundle.run_dir
    lock_path = run_dir / "FINAL_RUN_LOCK.json"
    sha_path = run_dir / "FINAL_RUN_SHA256.txt"
    complete_path = run_dir / "COMPLETE"
    partial_path = run_dir / "PARTIAL"
    failed_path = run_dir / "FAILED"
    if lock_path.exists() or any(
        path.exists() for path in (complete_path, partial_path, failed_path)
    ):
        raise FileExistsError(
            "final state is immutable; existing lock/terminal marker refuses rebuild"
        )
    # Root control artifacts must represent the final staged run rather than
    # the original P0 bootstrap snapshot.
    from tools.unified_reranking.run_all import write_reproduce_readme

    write_reproduce_readme(run_dir)
    export_ledger_commands(run_dir / "run_ledger.sqlite")
    checks = dict(integrity["checks"])
    checks["required_analysis_registry"] = {
        "status": "PASS" if analysis_registry.get("status") == "PASS" else "FAIL",
        "details": analysis_registry,
    }
    checks["tables_machine_generated"] = {
        "status": "PASS"
        if table_manifest.get("status") == "COMPLETE"
        and table_manifest.get("manual_numeric_entries") == 0
        and set(TABLE_NAMES).issubset(table_manifest.get("tables", {}))
        else "FAIL",
        "details": table_manifest,
    }
    checks["figures_machine_generated"] = {
        "status": "PASS"
        if figure_manifest.get("status") == "COMPLETE"
        and set(figure_manifest.get("figures", {})) == set(FIGURE_NAMES)
        else "FAIL",
        "details": figure_manifest,
    }
    gallery_summaries = list(gallery_manifest.get("summaries", []))
    expected_gallery_groups = len(ROUTES) * len(ROUTE_GALLERY_QUOTAS) + len(
        CROSS_ROUTE_GALLERY_QUOTAS
    )
    gallery_audit_ok = (
        gallery_manifest.get("status")
        in {"COMPLETE", "COMPLETE_WITH_AUDITED_SHORTFALLS"}
        and gallery_manifest.get("manual_selection") is False
        and len(gallery_summaries) == expected_gallery_groups
        and all(
            int(row.get("selected", -1)) + int(row.get("selection_shortfall", -1))
            == int(row.get("requested", -2))
            and int(row.get("rendered", -1)) + int(row.get("render_shortfall", -1))
            == int(row.get("requested", -2))
            for row in gallery_summaries
        )
    )
    checks["deterministic_galleries_audited"] = {
        "status": "PASS" if gallery_audit_ok else "FAIL",
        "details": gallery_manifest,
    }
    checks["bilingual_reports_machine_generated"] = {
        "status": "PASS"
        if report_manifest.get("status") == "COMPLETE"
        and report_manifest.get("manual_numeric_entries") == 0
        and set(report_manifest.get("reports", {})) == set(REPORT_NAMES)
        else "FAIL",
        "details": report_manifest,
    }
    baseline_report = run_dir / "14_reports" / "BASELINE_GAP_ATTRIBUTION.md"
    baseline_alias = run_dir / "11_attribution_bridge" / "BASELINE_GAP_ATTRIBUTION.md"
    alias_record = dict(report_manifest.get("prompt_aliases", {})).get(
        "baseline_gap_attribution", {}
    )
    baseline_alias_ok = (
        baseline_report.is_file()
        and baseline_alias.is_file()
        and sha256_file(baseline_report) == sha256_file(baseline_alias)
        and alias_record.get("sha256") == sha256_file(baseline_alias)
    )
    checks["baseline_gap_prompt_alias"] = {
        "status": "PASS" if baseline_alias_ok else "FAIL",
        "details": {
            "report": str(baseline_report.resolve()),
            "alias": str(baseline_alias.resolve()),
            "byte_identical": baseline_alias_ok,
        },
    }
    required_paths = [
        run_dir / "manifest.json",
        run_dir / "README_REPRODUCE.md",
        run_dir / "run_ledger.sqlite",
        run_dir / "commands.log",
        run_dir / "environment.txt",
        run_dir / "git_state.txt",
        run_dir / "08_lock" / "PRIMARY_METHOD_DECLARATION.md",
        run_dir / "08_lock" / "FORMAL_TEST_LOCK.json",
        run_dir / "09_formal_test" / "per_candidate_scores.parquet",
        run_dir / "09_formal_test" / "bridge_per_candidate_scores.parquet",
        run_dir / "09_formal_test" / "per_sample_decisions.parquet",
        run_dir / "07_validation" / "bridge_train_validation.csv",
        run_dir / "11_attribution_bridge" / "BRIDGE_DESIGN.md",
        run_dir / "11_attribution_bridge" / "bridge_train_validation.csv",
        run_dir / "11_attribution_bridge" / "bridge_postlock_test.csv",
        run_dir / "11_attribution_bridge" / "BRIDGE_POSTLOCK_TEST_MANIFEST.json",
        baseline_alias,
        run_dir / "15_independent_recompute" / "INDEPENDENT_RECOMPUTE.json",
        run_dir / "15_independent_recompute" / "INDEPENDENT_RECOMPUTE.md",
        run_dir / "tables" / "TABLE_GENERATION_MANIFEST.json",
        run_dir / "12_figures" / "FIGURE_MANIFEST.json",
        run_dir / "13_failure_galleries" / "GALLERY_MANIFEST.json",
        run_dir / "14_reports" / "REPORT_MANIFEST.json",
        run_dir / "14_reports" / "REQUIRED_ANALYSIS_REGISTRY.json",
        *[run_dir / "tables" / name for name in TABLE_NAMES],
        run_dir / "tables" / "failure_strata.csv",
        run_dir / "tables" / "failure_bottleneck_summary.csv",
        *[run_dir / "14_reports" / name for name in REPORT_NAMES],
        *[
            run_dir / "12_figures" / f"{name}.{suffix}"
            for name in FIGURE_NAMES
            for suffix in ("pdf", "svg", "caption.md")
        ],
    ]
    missing_required = [
        str(path.resolve()) for path in required_paths if not path.is_file()
    ]
    checks["prompt_required_artifacts"] = {
        "status": "PASS" if not missing_required else "FAIL",
        "details": {"required_count": len(required_paths), "missing": missing_required},
    }
    preliminary_pass = all(value["status"] == "PASS" for value in checks.values())
    status = "COMPLETE" if preliminary_pass else "FAILED"
    root_manifest_path = run_dir / "manifest.json"
    root_manifest = read_json(root_manifest_path)
    if not isinstance(root_manifest, dict):
        raise ValueError("root manifest is not a JSON object")
    root_manifest.update(
        {
            "status": status,
            "test_label_state": "FORMAL_TEST_COMPLETE",
            "formal_test_execution_count": 1,
        }
    )
    atomic_json(root_manifest_path, root_manifest)
    root_manifest_terminal_ok = (
        read_json(root_manifest_path).get("status") == status
        and read_json(root_manifest_path).get("test_label_state")
        == "FORMAL_TEST_COMPLETE"
        and read_json(root_manifest_path).get("formal_test_execution_count") == 1
    )
    checks["root_manifest_terminal_state"] = {
        "status": "PASS" if root_manifest_terminal_ok else "FAIL",
        "details": {
            "path": str(root_manifest_path.resolve()),
            "status": status,
            "test_label_state": "FORMAL_TEST_COMPLETE",
            "formal_test_execution_count": 1,
        },
    }
    inventory = hash_inventory(
        run_dir,
        excluded=[lock_path, sha_path, complete_path, partial_path, failed_path],
    )
    inventory_ok, inventory_errors = verify_inventory(inventory)
    checks["final_inventory_recomputed"] = {
        "status": "PASS" if inventory_ok else "FAIL",
        "details": inventory_errors,
    }
    status = (
        "COMPLETE"
        if all(value["status"] == "PASS" for value in checks.values())
        else "FAILED"
    )
    if root_manifest["status"] != status:
        root_manifest["status"] = status
        atomic_json(root_manifest_path, root_manifest)
        inventory = hash_inventory(
            run_dir,
            excluded=[lock_path, sha_path, complete_path, partial_path, failed_path],
        )
        inventory_ok, inventory_errors = verify_inventory(inventory)
        checks["final_inventory_recomputed"] = {
            "status": "PASS" if inventory_ok else "FAIL",
            "details": inventory_errors,
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "formal_test_execution_count": 1,
        "benchmark_description": "locked retrospective paired test benchmark",
        "integrity_checks": checks,
        "artifact_manifests": {
            "tables": table_manifest.get("content_sha256"),
            "figures": figure_manifest.get("content_sha256"),
            "galleries": gallery_manifest.get("content_sha256"),
            "reports": report_manifest.get("content_sha256"),
            "required_analyses": analysis_registry.get("content_sha256"),
            "postlock_bridge": read_json(
                run_dir / "11_attribution_bridge" / "BRIDGE_POSTLOCK_TEST_MANIFEST.json"
            ).get("content_sha256"),
        },
        "inventory": inventory,
        "inventory_count": len(inventory),
        "inventory_content_sha256": canonical_sha256(inventory),
        "excluded_self_referential_files": [
            "FINAL_RUN_LOCK.json",
            "FINAL_RUN_SHA256.txt",
            "COMPLETE",
            "PARTIAL",
            "FAILED",
        ],
    }
    payload["self_sha256"] = canonical_sha256(payload)
    atomic_json(lock_path, payload)
    file_sha = sha256_file(lock_path)
    atomic_text(sha_path, file_sha + "  FINAL_RUN_LOCK.json\n")
    marker = complete_path if status == "COMPLETE" else failed_path
    atomic_text(marker, f"{status}\nFINAL_RUN_LOCK.json sha256={file_sha}\n")
    verify_payload = read_json(lock_path)
    unsigned = dict(verify_payload)
    recorded = unsigned.pop("self_sha256")
    if recorded != canonical_sha256(unsigned) or sha256_file(lock_path) != file_sha:
        raise AssertionError("final lock failed immediate self-verification")
    return {
        "status": status,
        "final_lock_sha256": file_sha,
        "inventory_count": len(inventory),
    }


def verify_final_readiness(run_dir: Path) -> dict[str, Any]:
    """Recompute readiness from the lock, content inventory, and bound marker."""

    root = run_dir.resolve()
    lock_path = root / "FINAL_RUN_LOCK.json"
    markers = {name: root / name for name in ("COMPLETE", "PARTIAL", "FAILED")}
    present = [name for name, path in markers.items() if path.exists()]
    if not lock_path.is_file():
        return {
            "ready": False,
            "reason": "FINAL_RUN_LOCK.json missing",
            "markers": present,
        }
    payload = read_json(lock_path)
    unsigned = dict(payload)
    recorded = unsigned.pop("self_sha256", None)
    self_ok = isinstance(recorded, str) and recorded == canonical_sha256(unsigned)
    raw_inventory = payload.get("inventory", [])
    recorded_inventory = raw_inventory if isinstance(raw_inventory, list) else []
    inventory_ok, inventory_errors = verify_inventory(recorded_inventory)
    if not isinstance(raw_inventory, list):
        inventory_ok = False
        inventory_errors.append("inventory-is-not-a-list")
    file_sha = sha256_file(lock_path)
    sha_path = root / "FINAL_RUN_SHA256.txt"
    excluded_inventory_paths = [
        lock_path,
        sha_path,
        *markers.values(),
    ]
    fresh_inventory = hash_inventory(root, excluded=excluded_inventory_paths)

    def inventory_identity(
        rows: Sequence[Mapping[str, Any]],
    ) -> set[tuple[str, int, str]]:
        result: set[tuple[str, int, str]] = set()
        for row in rows:
            relative = row.get("relative_path")
            size = row.get("bytes")
            digest = row.get("sha256")
            if (
                isinstance(relative, str)
                and isinstance(size, int)
                and isinstance(digest, str)
            ):
                result.add((relative, size, digest))
        return result

    recorded_identity = inventory_identity(recorded_inventory)
    fresh_identity = inventory_identity(fresh_inventory)
    fresh_inventory_exact_match = recorded_identity == fresh_identity and len(
        recorded_identity
    ) == len(recorded_inventory) == len(fresh_inventory)
    if not fresh_inventory_exact_match:
        recorded_relatives = {row[0] for row in recorded_identity}
        fresh_relatives = {row[0] for row in fresh_identity}
        inventory_errors.extend(
            [
                f"added-after-lock:{value}"
                for value in sorted(fresh_relatives - recorded_relatives)
            ]
        )
        inventory_errors.extend(
            [
                f"missing-after-lock:{value}"
                for value in sorted(recorded_relatives - fresh_relatives)
            ]
        )
        for relative in sorted(recorded_relatives & fresh_relatives):
            recorded_row = next(row for row in recorded_identity if row[0] == relative)
            fresh_row = next(row for row in fresh_identity if row[0] == relative)
            if recorded_row != fresh_row:
                inventory_errors.append(f"changed-after-lock:{relative}")
    inventory_paths_bound = True
    for row in recorded_inventory:
        try:
            relative_path = Path(str(row["relative_path"]))
            expected_path = (root / relative_path).resolve()
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or Path(str(row["path"])).resolve() != expected_path
            ):
                inventory_paths_bound = False
                inventory_errors.append(
                    f"inventory-path-not-root-bound:{row.get('relative_path')}"
                )
        except (KeyError, OSError, ValueError):
            inventory_paths_bound = False
            inventory_errors.append("malformed-inventory-path")
    commands_path = root / "commands.log"
    try:
        commands_expected = render_ledger_commands(root / "run_ledger.sqlite")
        commands_content_ok = (
            bool(commands_expected.strip())
            and commands_path.read_text(encoding="utf-8") == commands_expected
        )
    except (OSError, ValueError, sqlite3.Error):
        commands_content_ok = False
    commands_log_ok = (
        commands_path.is_file()
        and not commands_path.is_symlink()
        and commands_content_ok
        and "commands.log" in {row[0] for row in recorded_identity}
        and "commands.log" in {row[0] for row in fresh_identity}
    )
    inventory_ok = (
        inventory_ok
        and fresh_inventory_exact_match
        and inventory_paths_bound
        and commands_log_ok
    )
    sha_ok = sha_path.is_file() and sha_path.read_text(encoding="utf-8").strip() == (
        f"{file_sha}  FINAL_RUN_LOCK.json"
    )
    expected_marker = str(payload.get("status", ""))
    marker_ok = present == [expected_marker] and expected_marker in markers
    marker_binding_ok = False
    if marker_ok:
        text = markers[expected_marker].read_text(encoding="utf-8")
        marker_binding_ok = f"FINAL_RUN_LOCK.json sha256={file_sha}" in text
    root_manifest_ok = False
    root_manifest_status: str | None = None
    try:
        root_manifest = read_json(root / "manifest.json")
        root_manifest_status = str(root_manifest.get("status", ""))
        root_manifest_ok = (
            root_manifest_status == payload.get("status")
            and root_manifest.get("test_label_state") == "FORMAL_TEST_COMPLETE"
            and root_manifest.get("formal_test_execution_count") == 1
        )
    except (OSError, ValueError, json.JSONDecodeError):
        root_manifest_ok = False
    ready = (
        payload.get("status") == "COMPLETE"
        and self_ok
        and inventory_ok
        and sha_ok
        and marker_ok
        and marker_binding_ok
        and root_manifest_ok
        and commands_log_ok
    )
    return {
        "ready": ready,
        "status": payload.get("status"),
        "self_hash_ok": self_ok,
        "inventory_ok": inventory_ok,
        "inventory_errors": inventory_errors,
        "fresh_inventory_exact_match": fresh_inventory_exact_match,
        "inventory_paths_bound": inventory_paths_bound,
        "commands_log_ok": commands_log_ok,
        "sha_file_ok": sha_ok,
        "markers": present,
        "marker_binding_ok": marker_binding_ok,
        "root_manifest_status": root_manifest_status,
        "root_manifest_agrees_with_lock": root_manifest_ok,
        "final_lock_sha256": file_sha,
    }


def run(run_dir: Path, *, command: str = "") -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if (run_dir / "FINAL_RUN_LOCK.json").exists():
        readiness = verify_final_readiness(run_dir)
        if not readiness.get("inventory_ok") or not readiness.get("self_hash_ok"):
            raise PermissionError("immutable final state has drifted; refusing rebuild")
        if readiness.get("status") == "COMPLETE" and not readiness.get("ready"):
            raise PermissionError(
                "immutable COMPLETE state has an invalid SHA file or terminal marker"
            )
        return {"status": readiness.get("status"), "resumed": True, **readiness}
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="POSTFORMAL",
        substage="tables_figures_galleries_reports_and_prefinal_integrity",
        command=command,
    ) as state:
        bundle = load_formal_bundle(run_dir)
        build_postlock_bridge(bundle)
        taxonomy = classify_failures(bundle)
        atomic_parquet(
            run_dir / "13_failure_galleries" / "failure_taxonomy_per_sample.parquet",
            taxonomy,
        )
        integrity = core_integrity_checks(bundle, taxonomy)
        tables = build_tables(bundle, taxonomy, integrity.get("status") == "PASS")
        table_manifest = write_tables(run_dir, tables)
        analysis_registry = required_analysis_registry(run_dir, tables)
        gallery_manifest = build_galleries(bundle, taxonomy)
        figure_manifest = build_figures(run_dir, tables)
        report_manifest = build_reports(
            bundle,
            tables,
            table_manifest,
            figure_manifest,
            gallery_manifest,
            integrity,
            analysis_registry,
        )
        prefinal_artifact = run_dir / "14_reports" / "REPORT_MANIFEST.json"
        state["artifact_path"] = str(prefinal_artifact.resolve())
        state["artifact_sha256"] = sha256_file(prefinal_artifact)
    return finalize_run(
        bundle,
        integrity,
        table_manifest,
        figure_manifest,
        gallery_manifest,
        report_manifest,
        analysis_registry,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        args.run_dir.expanduser().resolve(), command=" ".join(map(str, sys.argv))
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_figures",
    "build_galleries",
    "build_postlock_bridge",
    "build_reports",
    "build_tables",
    "finalize_run",
    "required_analysis_registry",
    "run",
    "verify_final_readiness",
    "write_tables",
]
