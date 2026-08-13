"""Independent semantic replay for all P12 prelock artifacts."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.datasets import FoldPreprocessor
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.metrics import compare_selections, evaluate_order_only
from unified_reranking.route_router import RouterOperatingPoint
from unified_reranking.training import FORMAL_SEEDS

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest
from .four_route import (
    ALTERNATIVE_ROUTES,
    FOUR_ROUTES,
    FourRouteCROGDefaultTransitionRouter,
    four_route_decisions,
    select_four_route_operating_point,
    validate_top20_union_frame,
    validation_router_union_summary,
)
from .four_route_plan import PLAN_RELATIVE_PATH
from .four_route_producer import (
    FORMAL_TOP20_BUDGET,
    UNION_ENCODERS,
    _normalized_artifacts,
    _predict_union_cell,
    _router_evidence,
    _router_trial_frame,
    _router_utilities,
    four_route_router_grid,
    top20_union_plan,
)
from .four_route_t4 import PEER_ROUTES, _read_candidates, build_t4_frame


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"P12 replay source is not regular: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _same_frame(observed: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
            check_dtype=False,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from semantic replay") from error


def _content(path: Path, *, name: str, statuses: tuple[str, ...]) -> dict[str, Any]:
    value = load_content_manifest(path, name=name, statuses=statuses)
    if value.get("candidate_test_labels_read") is not False:
        raise PermissionError(f"{name} lacks Test-label isolation")
    return value


def validate_four_route_source_plan(path: Path) -> dict[str, Any]:
    """Revalidate current bytes and the completed-run inventory closure."""

    plan = _content(path, name="P12 source plan", statuses=("PLANNED",))
    if (
        plan.get("completed_three_route_run_mode") != "STRICT_READ_ONLY"
        or plan.get("selection_used_test_metrics") is not False
        or plan.get("router_contract", {}).get("grid", {}).get("trial_count") != 108
        or plan.get("top20_union_contract", {}).get("cell_count") != 36
        or plan.get("t4_contract", {}).get("track") != "T4_four_route_consensus"
        or plan.get("t4_contract", {}).get("base_track") != "T3_route_rich"
        or plan.get("t4_contract", {}).get("column_order")
        != "exact T3 model columns then stable-sorted t4 consensus"
        or plan.get("t4_contract", {}).get("candidate_join")
        != "exact one-to-one candidate identity and geometry"
    ):
        raise RuntimeError("P12 source plan semantics differ")
    sources = plan.get("sources")
    if not isinstance(sources, Mapping) or plan.get(
        "source_signature_sha256"
    ) != canonical_sha256(sources):
        raise RuntimeError("P12 source-plan signature differs")
    verify_artifact_records_recursive(
        sources, name="P12 plan sources", require_at_least_one=True
    )
    lock_path = verified_artifact_path(
        sources["completed_three_route_final_lock"], name="P12 completed final lock"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    unsigned = dict(lock)
    observed_self = unsigned.pop("self_sha256", None)
    if lock.get("status") != "COMPLETE" or observed_self != canonical_sha256(unsigned):
        raise RuntimeError("P12 completed source lock no longer verifies")
    inventory = lock.get("inventory")
    if not isinstance(inventory, list) or lock.get(
        "inventory_content_sha256"
    ) != canonical_sha256(inventory):
        raise RuntimeError("P12 completed source inventory differs")
    frozen = {str(Path(str(row["path"])).resolve()): row for row in inventory}
    outputs = sources.get("completed_three_route_outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("P12 completed source output records are absent")
    for name, record in outputs.items():
        current = _record(str(record["path"]))
        inventory_record = frozen.get(current["path"])
        if not isinstance(inventory_record, Mapping) or any(
            inventory_record.get(key) != current[key]
            for key in ("path", "sha256", "bytes")
        ):
            raise RuntimeError(
                f"P12 completed source mutated or escaped inventory: {name}"
            )
    return plan


def _replay_four_route_router_selection(
    path: Path,
) -> tuple[dict[str, Any], FourRouteCROGDefaultTransitionRouter]:
    """Refit all transition models and replay every one of the 108 trials."""

    manifest = _content(path, name="P12 router selection", statuses=("COMPLETE",))
    sources = manifest.get("sources")
    artifacts = manifest.get("artifacts")
    if not isinstance(sources, Mapping) or not isinstance(artifacts, Mapping):
        raise RuntimeError("P12 router source/artifact inventory is absent")
    verify_artifact_records_recursive(
        sources, name="P12 router sources", require_at_least_one=True
    )
    verify_artifact_records_recursive(
        artifacts, name="P12 router artifacts", require_at_least_one=True
    )
    config = manifest.get("configuration")
    points = four_route_router_grid()
    if (
        not isinstance(config, Mapping)
        or config.get("default_route") != "CROG"
        or config.get("alternative_routes") != list(ALTERNATIVE_ROUTES)
        or config.get("tie_break") != list(ALTERNATIVE_ROUTES)
        or config.get("operating_points") != [asdict(point) for point in points]
        or int(config.get("bootstrap_iterations", -1)) != 10_000
    ):
        raise RuntimeError("P12 router configuration/grid differs")
    expected_signature = canonical_sha256({"configuration": config, "sources": sources})
    if manifest.get("signature_sha256") != expected_signature:
        raise RuntimeError("P12 router selection signature differs")
    train = pd.read_parquet(
        verified_artifact_path(sources["train_oof"], name="P12 Train OOF")
    )
    validation = pd.read_parquet(
        verified_artifact_path(
            sources["validation"], name="P12 Validation router input"
        )
    )
    columns = {
        route: tuple(map(str, config["feature_columns"][route]))
        for route in ALTERNATIVE_ROUTES
    }

    def transition(route: str) -> Any:
        from unified_reranking.gate import OOFTransitionData

        return OOFTransitionData(
            features=train.loc[:, columns[route]].to_numpy(float),
            feature_names=columns[route],
            native_correct=train["crog_correct"].to_numpy(),
            challenger_correct=train[f"{route.lower()}_correct"].to_numpy(),
            scene_ids=train["scene_id"].to_numpy(),
            oof_fold_ids=train["oof_fold"].to_numpy(),
            prediction_source="train_oof",
        )

    router = FourRouteCROGDefaultTransitionRouter(seed=int(config["model_seed"])).fit(
        {route: transition(route) for route in ALTERNATIVE_ROUTES}
    )
    validation_features = {
        route: validation.loc[:, columns[route]].to_numpy(float)
        for route in ALTERNATIVE_ROUTES
    }
    probabilities = router.predict_probabilities(validation_features)
    evidence = _router_evidence(validation)
    correct = {
        route: validation[f"{route.lower()}_correct"].to_numpy()
        for route in FOUR_ROUTES
    }
    result = select_four_route_operating_point(
        probabilities,
        evidence,
        correct,
        validation["scene_id"].to_numpy(),
        points,
        bootstrap_iterations=10_000,
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    if result.selected_operating_point is None:
        selected_routes = np.full(len(validation), "CROG", dtype=object)
        utilities = {route: np.zeros(len(validation)) for route in ALTERNATIVE_ROUTES}
    else:
        selected_routes = four_route_decisions(
            probabilities, evidence, result.selected_operating_point
        )
        utilities = _router_utilities(
            probabilities, result.selected_operating_point.lambda_router
        )
    route_ids = {
        route: validation[f"{route.lower()}_candidate_id"]
        .fillna("")
        .astype(str)
        .to_numpy()
        for route in FOUR_ROUTES
    }
    route_hashes = {
        route: validation[f"{route.lower()}_candidate_geometry_sha256"]
        .fillna("")
        .astype(str)
        .to_numpy()
        for route in FOUR_ROUTES
    }
    indexes = np.arange(len(validation))
    expected: dict[str, Any] = {
        "sample_id": validation["sample_id"].astype(str),
        "scene_id": validation["scene_id"].astype(str),
        "prediction_source": "validation",
        "selected_route": selected_routes,
        "switched_from_crog": selected_routes != "CROG",
        "selected_candidate_id": [
            route_ids[str(route)][index]
            for index, route in zip(indexes, selected_routes, strict=True)
        ],
        "selected_candidate_geometry_sha256": [
            route_hashes[str(route)][index]
            for index, route in zip(indexes, selected_routes, strict=True)
        ],
        "selected_correct": [
            bool(correct[str(route)][index])
            for index, route in zip(indexes, selected_routes, strict=True)
        ],
    }
    for route in FOUR_ROUTES:
        expected[f"{route.lower()}_candidate_id"] = route_ids[route]
        expected[f"{route.lower()}_correct"] = np.asarray(correct[route], dtype=bool)
    for route in ALTERNATIVE_ROUTES:
        expected[f"{route.lower()}_probability_recover"] = probabilities[route][0]
        expected[f"{route.lower()}_probability_harm"] = probabilities[route][1]
        expected[f"{route.lower()}_utility"] = utilities[route]
    _same_frame(
        pd.read_parquet(
            verified_artifact_path(
                artifacts["validation_decisions"], name="P12 router decisions"
            )
        ),
        pd.DataFrame(expected),
        name="P12 router decisions",
    )
    _same_frame(
        pd.read_parquet(
            verified_artifact_path(
                artifacts["validation_trials"], name="P12 router trials"
            )
        ),
        _router_trial_frame(result),
        name="P12 router trials",
    )
    model_path = verified_artifact_path(
        artifacts["transition_models"], name="P12 transition models"
    )
    expected_model_bytes = pickle.dumps(router, protocol=pickle.HIGHEST_PROTOCOL)
    if (
        model_path.read_bytes() != expected_model_bytes
        or manifest.get("transition_models") != router.artifact()
        or manifest.get("selection") != result.artifact()
        or manifest.get("decision") != result.status
    ):
        raise RuntimeError("P12 router model/selection differs from replay")
    return manifest, router


def validate_four_route_router_selection(path: Path) -> dict[str, Any]:
    """Refit and exactly replay the router without deserializing its model."""

    manifest, _router = _replay_four_route_router_selection(path)
    return manifest


def rebuild_validated_four_route_router(
    path: Path,
) -> tuple[dict[str, Any], FourRouteCROGDefaultTransitionRouter]:
    """Return the refitted router only after the full semantic/byte replay."""

    return _replay_four_route_router_selection(path)


def _validate_union_cell(
    cell_path: Path,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    folds: pd.DataFrame,
    validation_denominator: list[str],
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, Any], pd.DataFrame]:
    cell = _content(cell_path, name="P12 union cell", statuses=("COMPLETE",))
    verify_artifact_records_recursive(
        {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
        name="P12 union cell",
        require_at_least_one=True,
    )
    config = cell["configuration"]
    held_fold = config["held_fold"]
    early_fold = int(config["early_stop_fold"])
    merged = train.merge(folds, on="sample_id", validate="many_to_one")
    fit = merged.loc[
        (merged["fold"] != early_fold)
        & ((merged["fold"] != held_fold) if held_fold is not None else True)
    ]
    expected_preprocessor = FoldPreprocessor.fit(fit, feature_columns)
    if cell.get("preprocessor") != expected_preprocessor.artifact():
        raise RuntimeError("P12 union fold-local preprocessor differs")
    if held_fold is None:
        predict = validation
        denominator = validation_denominator
    else:
        predict = merged.loc[merged["fold"] == held_fold]
        denominator = (
            folds.loc[folds["fold"].eq(held_fold), "sample_id"].astype(str).tolist()
        )
    predictions = _predict_union_cell(cell, predict)
    stored = pd.read_parquet(
        verified_artifact_path(
            cell["artifacts"]["candidate_scores"], name="P12 cell scores"
        )
    )
    _same_frame(stored, predictions, name="P12 union cell scores")
    evaluation = predict[
        ["sample_id", "candidate_id", "native_rank", "candidate_success"]
    ].merge(predictions, on=["sample_id", "candidate_id"], validate="one_to_one")
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score", max_k=20
    )
    _same_frame(
        pd.read_parquet(
            verified_artifact_path(
                cell["artifacts"]["per_sample_decisions"], name="P12 cell decisions"
            )
        ),
        decisions,
        name="P12 union cell decisions",
    )
    if cell.get("metrics") != metrics:
        raise RuntimeError("P12 union cell metrics differ from replay")
    return cell, predictions


def validate_top20_union_selection(
    path: Path, *, require_formal_budget: bool = True
) -> dict[str, Any]:
    """Replay every stored model prediction, seed ensemble, and Validation choice."""

    selection = _content(
        path, name="P12 union selection", statuses=("VALIDATION_LOCKED",)
    )
    plan_path = verified_artifact_path(
        selection["sources"]["plan"], name="P12 union plan"
    )
    plan = _content(plan_path, name="P12 union plan", statuses=("PLANNED",))
    observed_budget = plan.get("budget")
    if not isinstance(observed_budget, Mapping):
        raise RuntimeError("P12 union plan budget is absent")
    from .four_route_producer import Top20UnionBudget

    budget = Top20UnionBudget(**dict(observed_budget))
    expected_budget = asdict(FORMAL_TOP20_BUDGET)
    if (
        plan.get("encoders") != list(UNION_ENCODERS)
        or plan.get("seeds") != list(FORMAL_SEEDS)
        or plan.get("folds") != list(range(5))
        or plan.get("cell_count") != 36
        or plan.get("cells") != list(top20_union_plan(budget))
        or (require_formal_budget and plan.get("budget") != expected_budget)
    ):
        raise RuntimeError("P12 union exact plan differs")
    verify_artifact_records_recursive(
        plan["sources"], name="P12 union plan sources", require_at_least_one=True
    )
    train = pd.read_parquet(
        verified_artifact_path(plan["sources"]["train_top20"], name="P12 Train Top20")
    )
    validation = pd.read_parquet(
        verified_artifact_path(
            plan["sources"]["validation_top20"], name="P12 Validation Top20"
        )
    )
    folds = pd.read_parquet(
        verified_artifact_path(plan["sources"]["folds"], name="P12 folds")
    )
    validation_denominator = (
        pd.read_parquet(
            verified_artifact_path(
                plan["sources"]["validation_denominator"],
                name="P12 Validation denominator",
            ),
            columns=["sample_id"],
        )["sample_id"]
        .astype(str)
        .tolist()
    )
    feature_columns = tuple(map(str, plan["feature_columns"]))
    validate_top20_union_frame(
        train, tuple(dict.fromkeys(train["sample_id"].astype(str)))
    )
    validate_top20_union_frame(validation, validation_denominator)
    cell_root = plan_path.parent / "cells"
    cell_paths = sorted(cell_root.glob("*/manifest.json"))
    if len(cell_paths) != 36:
        raise RuntimeError("P12 union cell artifact count differs")
    cells: list[dict[str, Any]] = []
    predictions_by_cell: dict[tuple[str, int, str, Any], pd.DataFrame] = {}
    for cell_path in cell_paths:
        cell, predictions = _validate_union_cell(
            cell_path,
            train=train,
            validation=validation,
            folds=folds,
            validation_denominator=validation_denominator,
            feature_columns=feature_columns,
        )
        config = cell["configuration"]
        key = (
            config["encoder"],
            int(config["seed"]),
            config["mode"],
            config["held_fold"],
        )
        if key in predictions_by_cell:
            raise RuntimeError("P12 union cell configuration is duplicated")
        predictions_by_cell[key] = predictions
        cells.append(cell)
    if {key for key in predictions_by_cell} != {
        (row["encoder"], int(row["seed"]), row["mode"], row["held_fold"])
        for row in plan["cells"]
    }:
        raise RuntimeError("P12 union cell universe differs from plan")
    trials: list[dict[str, Any]] = []
    for encoder in UNION_ENCODERS:
        for split, frame, denominator in (
            (
                "train_oof",
                train,
                pd.read_parquet(
                    verified_artifact_path(
                        plan["sources"]["train_denominator"],
                        name="P12 Train denominator",
                    ),
                    columns=["sample_id"],
                )["sample_id"]
                .astype(str)
                .tolist(),
            ),
            ("validation", validation, validation_denominator),
        ):
            scores = frame[
                ["sample_id", "candidate_id", "native_rank", "candidate_success"]
            ].copy()
            for seed in FORMAL_SEEDS:
                parts = [
                    value
                    for (
                        cell_encoder,
                        cell_seed,
                        mode,
                        _fold,
                    ), value in predictions_by_cell.items()
                    if cell_encoder == encoder
                    and cell_seed == seed
                    and mode == ("oof" if split == "train_oof" else "validation")
                ]
                combined = pd.concat(parts, ignore_index=True)
                scores = scores.merge(
                    combined.rename(columns={"score": f"score_seed_{seed}"}),
                    on=["sample_id", "candidate_id"],
                    validate="one_to_one",
                )
            scores["score"] = scores[
                [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
            ].mean(axis=1)
            metrics, decisions = evaluate_order_only(
                denominator, scores, score_column="score", max_k=20
            )
            baseline_metrics, baseline_decisions = evaluate_order_only(
                denominator, frame, score_column="base_logit", max_k=20
            )
            comparison = compare_selections(
                baseline_decisions,
                decisions,
                oracle_at_5=float(baseline_metrics["oracle_at_5"]),
            )
            ensemble_path = (
                plan_path.parent / "ensembles" / f"{encoder}_{split}" / "manifest.json"
            )
            ensemble = _content(
                ensemble_path, name="P12 union ensemble", statuses=("COMPLETE",)
            )
            _same_frame(
                pd.read_parquet(
                    verified_artifact_path(
                        ensemble["artifacts"]["candidate_scores"],
                        name="P12 ensemble scores",
                    )
                ),
                scores,
                name="P12 union ensemble scores",
            )
            _same_frame(
                pd.read_parquet(
                    verified_artifact_path(
                        ensemble["artifacts"]["per_sample_decisions"],
                        name="P12 ensemble decisions",
                    )
                ),
                decisions,
                name="P12 union ensemble decisions",
            )
            if (
                ensemble.get("metrics") != metrics
                or ensemble.get("baseline_metrics") != baseline_metrics
                or ensemble.get("comparison_to_native_union_baseline") != comparison
            ):
                raise RuntimeError("P12 union ensemble metrics differ from replay")
            if split == "validation":
                trials.append(
                    {
                        "encoder": encoder,
                        "validation_j_at_1": metrics["j_at_1"],
                        "harmful": comparison["harmful"],
                        "switch_rate": comparison["switch_rate"],
                    }
                )
    tie_order = {encoder: index for index, encoder in enumerate(UNION_ENCODERS)}
    selected = max(
        trials,
        key=lambda row: (
            row["validation_j_at_1"],
            -row["harmful"],
            -row["switch_rate"],
            -tie_order[row["encoder"]],
        ),
    )
    if (
        selection.get("trials") != trials
        or selection.get("selected_encoder") != selected["encoder"]
    ):
        raise RuntimeError("P12 union Validation selection differs from replay")
    return selection


def validate_t4_manifest(path: Path, *, split: str) -> dict[str, Any]:
    manifest = _content(path, name=f"P12 {split} T4", statuses=("COMPLETE",))
    sources = manifest["sources"]
    verify_artifact_records_recursive(
        sources, name="P12 T4 sources", require_at_least_one=True
    )
    configuration = manifest.get("configuration", {})
    if configuration.get(
        "evidence_composition"
    ) != "exact T3_route_rich columns then stable-sorted t4 consensus" or manifest.get(
        "signature_sha256"
    ) != canonical_sha256({"configuration": configuration, "sources": sources}):
        raise RuntimeError("P12 T4 widened source signature differs")
    d1 = _read_candidates(
        verified_artifact_path(sources["d1_top5"], name="P12 T4 D1"),
        route="D1",
        split=split,
    )
    peers = {
        route: _read_candidates(
            verified_artifact_path(sources["peer_top5"][route], name=f"P12 T4 {route}"),
            route=route,
            split=split,
        )
        for route in PEER_ROUTES
    }
    t3_manifest_path = verified_artifact_path(
        sources["t3_manifest"], name="P12 T4 T3 manifest"
    )
    t3_manifest = _content(
        t3_manifest_path, name=f"P12 {split} T3", statuses=("COMPLETE",)
    )
    t3_configuration = t3_manifest.get("configuration", {})
    t3_columns = tuple(map(str, t3_manifest.get("model_feature_columns", ())))
    t3_feature_path = verified_artifact_path(
        t3_manifest.get("artifacts", {}).get("candidate_features", {}),
        name="P12 T4 T3 features",
    )
    if (
        t3_configuration.get("split") != split
        or t3_configuration.get("pool") != "top5"
        or t3_configuration.get("track") != "T3_route_rich"
        or t3_manifest.get("model_feature_schema_sha256")
        != canonical_sha256(t3_columns)
        or sources.get("t3_features") != _record(t3_feature_path)
    ):
        raise RuntimeError("P12 T4 T3 source binding differs")
    if split == "test":
        assert_label_free_parquet_schema(
            t3_feature_path, name="P12 T4 Test T3 features"
        )
    t3 = pd.read_parquet(t3_feature_path)
    expected, columns = build_t4_frame(d1, peers, t3, t3_columns)
    observed = pd.read_parquet(
        verified_artifact_path(
            manifest["artifacts"]["candidate_features"], name="P12 T4 features"
        )
    )
    _same_frame(observed, expected, name=f"P12 {split} T4 features")
    if manifest.get("model_feature_columns") != list(columns):
        raise RuntimeError("P12 T4 model schema differs from replay")
    return manifest


def validate_p12_test_application(
    path: Path,
    *,
    system: str,
    require_formal_budget: bool = True,
) -> dict[str, Any]:
    """Recompute one label-free Test application from its development lock."""

    allowed = {"four_route_crog_default_router", "top20_union"}
    if system not in allowed:
        raise ValueError(f"unknown P12 Test system: {system}")
    manifest = _content(
        path, name=f"P12 {system} Test application", statuses=("COMPLETE",)
    )
    configuration = manifest.get("configuration", {})
    if (
        configuration.get("system") != system
        or configuration.get("routes") != list(FOUR_ROUTES)
        or configuration.get("prediction_source") != "test_label_free"
        or configuration.get("candidate_test_labels_read") is not False
        or manifest.get("candidate_test_labels_read") is not False
        or manifest.get("selection_used_test_metrics") is not False
    ):
        raise PermissionError("P12 Test application provenance differs")
    verify_artifact_records_recursive(
        {"sources": manifest["sources"], "artifacts": manifest["artifacts"]},
        name=f"P12 {system} Test application",
        require_at_least_one=True,
    )
    sources = manifest["sources"]
    test_input_path = verified_artifact_path(
        sources["test_inputs"], name=f"P12 {system} Test inputs"
    )
    denominator_path = verified_artifact_path(
        sources["denominator"], name="P12 Test denominator"
    )
    assert_label_free_parquet_schema(test_input_path, name=f"P12 {system} Test inputs")
    assert_label_free_parquet_schema(denominator_path, name="P12 Test denominator")
    denominator = pd.read_parquet(denominator_path, columns=["sample_id"])[
        "sample_id"
    ].astype(str)
    if denominator.duplicated().any() or denominator.empty:
        raise RuntimeError("P12 Test denominator differs")

    selection_path = verified_artifact_path(
        sources["selection"], name=f"P12 {system} selection"
    )
    if system == "four_route_crog_default_router":
        selection, router = rebuild_validated_four_route_router(selection_path)
        test_inputs = pd.read_parquet(test_input_path)
        if (
            set(test_inputs["prediction_source"].astype(str)) != {"test_label_free"}
            or test_inputs["sample_id"].astype(str).duplicated().any()
            or set(test_inputs["sample_id"].astype(str)) != set(denominator)
        ):
            raise PermissionError("P12 router Test input provenance/universe differs")
        columns = {
            route: tuple(map(str, selection["configuration"]["feature_columns"][route]))
            for route in ALTERNATIVE_ROUTES
        }
        probabilities = router.predict_probabilities(
            {
                route: test_inputs.loc[:, columns[route]].to_numpy(float)
                for route in ALTERNATIVE_ROUTES
            }
        )
        evidence = _router_evidence(test_inputs)
        point_payload = selection["selection"]["selected_operating_point"]
        if point_payload is None:
            selected_routes = np.full(len(test_inputs), "CROG", dtype=object)
            utilities = {
                route: np.zeros(len(test_inputs)) for route in ALTERNATIVE_ROUTES
            }
        else:
            point = RouterOperatingPoint(**point_payload)
            selected_routes = four_route_decisions(probabilities, evidence, point)
            utilities = _router_utilities(probabilities, point.lambda_router)
        universe_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        decision_rows: list[dict[str, str]] = []
        for index, row in enumerate(test_inputs.itertuples(index=False)):
            sample_id = str(getattr(row, "sample_id"))
            selected_route = str(selected_routes[index])
            selected_id = ""
            for route in FOUR_ROUTES:
                prefix = route.lower()
                candidate_id = str(getattr(row, f"{prefix}_candidate_id") or "")
                if not candidate_id:
                    continue
                geometry = str(
                    getattr(row, f"{prefix}_candidate_geometry_sha256") or ""
                )
                native_score = float(getattr(row, f"{prefix}_native_score"))
                native_rank = int(getattr(row, f"{prefix}_native_rank"))
                geometry_values = {
                    field: float(getattr(row, f"{prefix}_{field}"))
                    for field in (
                        "cx_px",
                        "cy_px",
                        "theta_deg",
                        "width_px",
                        "height_px",
                    )
                }
                if (
                    not geometry
                    or not np.isfinite(
                        [native_score, native_rank, *geometry_values.values()]
                    ).all()
                    or native_rank <= 0
                    or geometry_values["width_px"] <= 0
                    or geometry_values["height_px"] <= 0
                ):
                    raise RuntimeError("P12 router Test candidate geometry differs")
                universe_rows.append(
                    {
                        "source_route": route,
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "candidate_geometry_sha256": geometry,
                        "native_rank": native_rank,
                        "native_score": native_score,
                        **geometry_values,
                    }
                )
                score_rows.append(
                    {
                        "source_route": route,
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "score": (
                            0.0 if route == "CROG" else float(utilities[route][index])
                        ),
                        "rank": 1,
                    }
                )
                if route == selected_route:
                    selected_id = candidate_id
            decision_rows.append(
                {
                    "sample_id": sample_id,
                    "selected_source_route": selected_route if selected_id else "",
                    "selected_candidate_id": selected_id,
                }
            )
        expected_universe = pd.DataFrame(universe_rows)
        expected_scores = pd.DataFrame(score_rows)
        expected_decisions = pd.DataFrame(decision_rows)
        expected_models: object = _record(
            verified_artifact_path(
                selection["artifacts"]["transition_models"],
                name="P12 router transition model",
            )
        )
    else:
        selection = validate_top20_union_selection(
            selection_path, require_formal_budget=require_formal_budget
        )
        test_inputs = pd.read_parquet(test_input_path)
        validate_top20_union_frame(test_inputs, denominator.tolist())
        cells: dict[int, Mapping[str, Any]] = {}
        cell_model_records: list[dict[str, Any]] = []
        for record in selection["sources"]["selected_validation_cells"]:
            cell_path = verified_artifact_path(
                record, name="P12 selected union Test cell"
            )
            cell = _content(
                cell_path, name="P12 selected union Test cell", statuses=("COMPLETE",)
            )
            seed = int(cell["configuration"]["seed"])
            if (
                cell["configuration"]["encoder"] != selection["selected_encoder"]
                or cell["configuration"]["mode"] != "validation"
                or seed in cells
            ):
                raise RuntimeError("P12 selected union Test cell differs")
            cells[seed] = cell
            cell_model_records.append(
                _record(
                    verified_artifact_path(
                        cell["artifacts"]["model"], name="P12 selected union model"
                    )
                )
            )
        if set(cells) != set(FORMAL_SEEDS):
            raise RuntimeError("P12 Test union seed inventory differs")
        scored = test_inputs.copy()
        for seed in FORMAL_SEEDS:
            prediction = _predict_union_cell(cells[seed], test_inputs).rename(
                columns={"score": f"score_seed_{seed}"}
            )
            scored = scored.merge(
                prediction, on=["sample_id", "candidate_id"], validate="one_to_one"
            )
        scored["score"] = scored[[f"score_seed_{seed}" for seed in FORMAL_SEEDS]].mean(
            axis=1
        )
        expected_universe, expected_scores, expected_decisions = _normalized_artifacts(
            denominator=denominator.tolist(), universe=scored, scored=scored
        )
        expected_models = cell_model_records
    if sources.get("selected_models") != expected_models:
        raise RuntimeError(f"P12 {system} selected-model binding differs")

    for key in ("candidate_universe", "candidate_scores", "per_sample_decisions"):
        artifact = verified_artifact_path(
            manifest["artifacts"][key], name=f"P12 Test {key}"
        )
        assert_label_free_parquet_schema(artifact, name=f"P12 Test {key}")
    universe = pd.read_parquet(
        verified_artifact_path(
            manifest["artifacts"]["candidate_universe"], name="P12 universe"
        )
    )
    scores = pd.read_parquet(
        verified_artifact_path(
            manifest["artifacts"]["candidate_scores"], name="P12 scores"
        )
    )
    decisions = pd.read_parquet(
        verified_artifact_path(
            manifest["artifacts"]["per_sample_decisions"], name="P12 decisions"
        )
    )
    universe_keys = set(
        map(
            tuple,
            universe[["source_route", "sample_id", "candidate_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    score_keys = set(
        map(
            tuple,
            scores[["source_route", "sample_id", "candidate_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    required_universe = {
        "source_route",
        "sample_id",
        "candidate_id",
        "candidate_geometry_sha256",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    }
    missing_universe = sorted(required_universe.difference(universe.columns))
    numeric_universe = universe[
        [
            column
            for column in (
                "native_rank",
                "native_score",
                "cx_px",
                "cy_px",
                "theta_deg",
                "width_px",
                "height_px",
            )
            if column in universe
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        missing_universe
        or universe_keys != score_keys
        or len(universe) != len(scores)
        or decisions["sample_id"].astype(str).duplicated().any()
        or set(decisions["sample_id"].astype(str)) != set(denominator)
        or not np.isfinite(pd.to_numeric(scores["score"], errors="coerce")).all()
        or not np.isfinite(numeric_universe.to_numpy(float)).all()
        or (
            not numeric_universe.empty
            and (numeric_universe[["width_px", "height_px"]] <= 0).any().any()
        )
    ):
        raise RuntimeError(f"P12 {system} normalized Test contract differs")
    _same_frame(universe, expected_universe, name=f"P12 {system} Test universe")
    _same_frame(scores, expected_scores, name=f"P12 {system} Test scores")
    _same_frame(decisions, expected_decisions, name=f"P12 {system} Test decisions")
    return manifest


def validate_p12_prelock(run_dir: str | Path) -> dict[str, Any]:
    """Validate P12 as a prerequisite of P13 readiness, never trusting its CSV."""

    root = Path(run_dir).expanduser().resolve()
    plan_path = root / PLAN_RELATIVE_PATH
    router_path = root / "13_four_route_extension/router/router_selection_manifest.json"
    union_path = root / "13_four_route_extension/union/selected_union_ranker.json"
    validation_manifest_path = (
        root / "13_four_route_extension/validation_router_union_manifest.json"
    )
    plan = validate_four_route_source_plan(plan_path)
    from .four_route_execution import validate_complete_execution

    execution = validate_complete_execution(root)
    router = validate_four_route_router_selection(router_path)
    union = validate_top20_union_selection(union_path)
    t4 = {
        split: validate_t4_manifest(
            root / f"03_features/{split}/top5/T4_four_route_consensus/manifest.json",
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    test_paths = {
        "four_route_crog_default_router": (
            root / "08_lock/formal_inputs/four_route_crog_default_router/manifest.json"
        ),
        "top20_union": root / "08_lock/formal_inputs/top20_union/manifest.json",
    }
    test = {
        system: validate_p12_test_application(path, system=system)
        for system, path in test_paths.items()
    }
    validation_manifest = _content(
        validation_manifest_path,
        name="P12 Validation summary",
        statuses=("COMPLETE",),
    )
    verify_artifact_records_recursive(
        {
            "sources": validation_manifest["sources"],
            "artifacts": validation_manifest["artifacts"],
        },
        name="P12 Validation summary",
        require_at_least_one=True,
    )
    samples = pd.read_parquet(
        verified_artifact_path(
            validation_manifest["artifacts"]["router_decisions"],
            name="P12 summary router decisions",
        )
    )
    top20 = pd.read_parquet(
        verified_artifact_path(
            validation_manifest["artifacts"]["top20_union"], name="P12 summary Top20"
        )
    )
    expected_table = validation_router_union_summary(samples, top20)
    observed_table = pd.read_csv(
        verified_artifact_path(
            validation_manifest["artifacts"]["prelock_table"], name="P12 summary table"
        )
    )
    _same_frame(observed_table, expected_table, name="P12 prelock table")
    required_sources = {
        "router_selection": _record(router_path),
        "union_selection": _record(union_path),
    }
    for name, record in required_sources.items():
        if validation_manifest["sources"].get(name) != record:
            raise RuntimeError(f"P12 Validation summary does not bind current {name}")
    return {
        "plan": _record(plan_path),
        "validation_execution": execution,
        "router": _record(router_path),
        "union": _record(union_path),
        "t4": {
            split: _record(
                root / f"03_features/{split}/top5/T4_four_route_consensus/manifest.json"
            )
            for split in t4
        },
        "test": {system: _record(test_paths[system]) for system in test},
        "validation": _record(validation_manifest_path),
        "source_plan_content_sha256": plan["content_sha256"],
        "router_content_sha256": router["content_sha256"],
        "union_content_sha256": union["content_sha256"],
    }


__all__ = [
    "rebuild_validated_four_route_router",
    "validate_four_route_router_selection",
    "validate_four_route_source_plan",
    "validate_p12_prelock",
    "validate_p12_test_application",
    "validate_t4_manifest",
    "validate_top20_union_selection",
]
