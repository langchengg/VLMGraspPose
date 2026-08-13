"""P13 semantic replay for K gates, Test applications, and formal inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256
from unified_reranking.metrics import rank_by_score

from .contracts import assert_label_free_parquet_schema
from .execution import artifact_record, load_content_manifest
from .formal import UNIVERSE_COLUMNS
from .gate_inputs import build_label_free_test_gate_input_frame
from .gate_validation import validate_gate_selection_semantics


K_TEST_SCENARIOS = {
    "top10_t2": {"pool": "top10", "name": "d1_top10"},
    "allnms_t2": {"pool": "allnms", "name": "d1_allnms"},
}
NORMALIZED_SYSTEMS = {
    "d1_top5_r0": {"pool": "top5", "source": "r0", "kind": "R0"},
    "d1_top5_r7_ungated": {
        "pool": "top5",
        "source": "top5_ranker",
        "kind": "R7_UNGATED",
    },
    "d1_top5_r7_gated": {
        "pool": "top5",
        "source": "top5_gate",
        "kind": "R7_GATED",
    },
    "d1_top10_locked": {
        "pool": "top10",
        "source": "d1_top10",
        "kind": "K_LOCKED",
    },
    "d1_allnms_locked": {
        "pool": "allnms",
        "source": "d1_allnms",
        "kind": "K_LOCKED",
    },
}


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _assert_frame(observed: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from semantic replay") from error


def _validate_test_access_events(root: Path) -> dict[str, Any]:
    path = root / "09_formal_test/test_access.log"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("D1 K Test access log is absent")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"D1 K Test access log line {line_number} is invalid"
            ) from error
        if isinstance(value, dict):
            events.append(value)
    for scenario_id, definition in K_TEST_SCENARIOS.items():
        for stage, category in (
            ("d1_k_scenario_test_ranker", "label_free_test_rankers"),
            ("d1_k_scenario_test_gate", "label_free_test_gates"),
        ):
            manifest_path = (
                root / f"08_lock/{category}/{definition['name']}/manifest.json"
            ).resolve()
            matching = [
                event
                for event in events
                if event.get("event") == "prelock_label_free_test_stage"
                and event.get("stage") == stage
                and event.get("scenario_id") == scenario_id
                and Path(str(event.get("output_manifest", ""))).resolve()
                == manifest_path
                and event.get("output_manifest_sha256")
                == artifact_record(manifest_path)["sha256"]
                and event.get("candidate_labels_opened_as_table") is False
            ]
            if not matching:
                raise RuntimeError(f"D1 K {scenario_id} {stage} access event is absent")
    normalized_records = {
        name: artifact_record(root / f"08_lock/formal_inputs/{name}/manifest.json")
        for name in sorted(NORMALIZED_SYSTEMS)
    }
    normalized = [
        event
        for event in events
        if event.get("event") == "prelock_label_free_test_stage"
        and event.get("stage") == "d1_formal_input_normalization"
        and event.get("output_manifests") == normalized_records
        and event.get("candidate_labels_opened_as_table") is False
        and event.get("candidate_test_labels_read") is False
    ]
    if not normalized:
        raise RuntimeError("D1 normalized formal-input Test access event is absent")
    return artifact_record(path)


def _candidate_table(root: Path, pool: str) -> tuple[Path, pd.DataFrame]:
    manifest = load_content_manifest(
        root / "02_candidates/test_manifest.json",
        name="D1 Test candidates",
        statuses=("COMPLETE",),
    )
    path = verified_artifact_path(
        _mapping(
            _mapping(manifest.get("artifacts"), name="D1 candidate artifacts").get(
                pool
            ),
            name=f"D1 Test {pool} record",
        ),
        name=f"D1 Test {pool}",
    )
    assert_label_free_parquet_schema(path, name=f"D1 Test {pool} candidates")
    return path, pd.read_parquet(path)


def _validate_k_ranker(
    root: Path, *, scenario_id: str, pool: str, name: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path = root / f"08_lock/label_free_test_rankers/{name}/manifest.json"
    manifest = load_content_manifest(
        path, name=f"D1 {name} ranker", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        manifest, name=f"D1 {name} ranker", require_at_least_one=True
    )
    sources = _mapping(manifest.get("sources"), name=f"D1 {name} ranker sources")
    artifacts = _mapping(manifest.get("artifacts"), name=f"D1 {name} ranker artifacts")
    candidate_path, candidates = _candidate_table(root, pool)
    selection_path = root / "11_k_sensitivity/selection_manifest.json"
    selection = load_content_manifest(
        selection_path, name="D1 K selection", statuses=("COMPLETE",)
    )
    scenario = _mapping(
        _mapping(selection.get("scenarios"), name="D1 K scenarios").get(scenario_id),
        name=f"D1 K {scenario_id}",
    )
    selection_sources = _mapping(
        selection.get("sources"), name="D1 K selection sources"
    )
    cell_records = _mapping(selection_sources.get("cells"), name="D1 K selection cells")
    validation_cells: dict[str, Any] = {}
    validation_job_ids: list[str] = []
    for job_id in map(str, scenario.get("cell_job_ids", ())):
        cell_path = verified_artifact_path(
            _mapping(cell_records.get(job_id), name=f"D1 K cell {job_id}"),
            name=f"D1 K cell {job_id}",
        )
        cell = load_content_manifest(
            cell_path, name=f"D1 K cell {job_id}", statuses=("COMPLETE",)
        )
        configuration = _mapping(
            cell.get("configuration"), name=f"D1 K cell {job_id} configuration"
        )
        if configuration.get("mode") != "validation":
            continue
        seed = str(configuration.get("seed"))
        model_contract = _mapping(
            cell.get("model_contract"), name=f"D1 K cell {job_id} model contract"
        )
        expected_serialization = (
            "lightgbm_native_text"
            if configuration.get("method") == "R5"
            else "torch_state_dict"
        )
        if (
            configuration.get("scenario_id") != scenario_id
            or configuration.get("pool") != pool
            or configuration.get("track") != "T2_matched_common"
            or configuration.get("method") != scenario.get("method")
            or model_contract.get("serialization") != expected_serialization
        ):
            raise RuntimeError(f"D1 K cell {job_id} Test model contract differs")
        validation_cells[seed] = artifact_record(cell_path)
        validation_job_ids.append(job_id)
    feature_manifest_path = (
        root / f"03_features/test/{pool}/T2_matched_common/manifest.json"
    )
    feature_manifest = load_content_manifest(
        feature_manifest_path,
        name=f"D1 Test {pool} feature manifest",
        statuses=("COMPLETE",),
    )
    feature_path = verified_artifact_path(
        _mapping(
            _mapping(
                feature_manifest.get("artifacts"), name="D1 feature artifacts"
            ).get("candidate_features"),
            name="D1 feature table",
        ),
        name=f"D1 Test {pool} feature table",
    )
    repository_root = Path(__file__).resolve().parents[2]
    expected_sources = {
        "k_selection": artifact_record(selection_path),
        "scenario_gate_selection": artifact_record(
            root / "11_k_sensitivity/scenario_gate_selection.json"
        ),
        "candidate_manifest": artifact_record(
            root / "02_candidates/test_manifest.json"
        ),
        "candidates": artifact_record(candidate_path),
        "feature_manifest": artifact_record(feature_manifest_path),
        "candidate_features": artifact_record(feature_path),
        "denominator": artifact_record(
            root / "01_manifests/d1_paired_manifest.parquet"
        ),
        "validation_cells": validation_cells,
        "inference_code": [
            artifact_record(source)
            for source in (
                repository_root / "src/d1_reranking/fold_calibration.py",
                repository_root / "src/d1_reranking/models.py",
                repository_root / "src/d1_reranking/selection.py",
                repository_root / "src/unified_reranking/datasets.py",
                repository_root / "src/unified_reranking/training.py",
                repository_root / "tools/d1_reranking/build_d1_formal_inputs.py",
            )
        ],
    }
    score_path = verified_artifact_path(
        _mapping(artifacts.get("per_candidate_scores"), name=f"D1 {name} scores"),
        name=f"D1 {name} scores",
    )
    decision_path = verified_artifact_path(
        _mapping(artifacts.get("per_sample_decisions"), name=f"D1 {name} decisions"),
        name=f"D1 {name} decisions",
    )
    for source, artifact_name in (
        (score_path, f"D1 {name} scores"),
        (decision_path, f"D1 {name} decisions"),
    ):
        assert_label_free_parquet_schema(source, name=artifact_name)
    scores = pd.read_parquet(score_path)
    decisions = pd.read_parquet(decision_path)
    candidate_keys = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    score_keys = set(
        map(tuple, scores[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    denominator = pd.read_parquet(
        root / "01_manifests/d1_paired_manifest.parquet", columns=["sample_id"]
    )
    candidates = candidates.copy()
    candidates[["sample_id", "candidate_id"]] = candidates[
        ["sample_id", "candidate_id"]
    ].astype(str)
    scores = scores.copy()
    scores[["sample_id", "candidate_id"]] = scores[
        ["sample_id", "candidate_id"]
    ].astype(str)
    decisions = decisions.copy()
    decisions["sample_id"] = decisions["sample_id"].astype(str)
    geometry = candidates.set_index(["sample_id", "candidate_id"])[
        "candidate_geometry_sha256"
    ]
    bearing = decisions["candidate_count"].astype(int).gt(0)
    selected_geometry = [
        str(geometry.loc[(row.sample_id, row.selected_candidate_id)])
        for row in decisions.loc[bearing].itertuples(index=False)
    ]
    expected_counts = (
        denominator["sample_id"]
        .astype(str)
        .map(candidates.groupby("sample_id").size())
        .fillna(0)
        .astype(int)
    )
    no_output = ~bearing
    empty_columns = [
        "native_candidate_id",
        "selected_candidate_id",
        "native_geometry_sha256",
        "selected_geometry_sha256",
    ]
    if (
        manifest.get("scenario_id") != scenario_id
        or manifest.get("pool") != pool
        or manifest.get("candidate_test_labels_read") is not False
        or manifest.get("selected_method") != scenario.get("method")
        or manifest.get("selected_primary_trial_id")
        != scenario.get("selected_primary_trial_id")
        or manifest.get("seed_policy") != "fixed mean score ensemble over 42,123,2026"
        or manifest.get("validation_cell_job_ids") != sorted(validation_job_ids)
        or sources != expected_sources
        or manifest.get("source_signature_sha256") != canonical_sha256(expected_sources)
        or candidate_keys != score_keys
        or len(scores) != len(candidates)
        or not np.isfinite(scores["ensemble_score"].to_numpy(float)).all()
        or not scores[["score_seed_42", "score_seed_123", "score_seed_2026"]]
        .mean(axis=1)
        .equals(scores["ensemble_score"])
        or decisions["sample_id"].astype(str).duplicated().any()
        or set(decisions["sample_id"].astype(str))
        != set(denominator["sample_id"].astype(str))
        or len(decisions) != len(denominator)
        or not decisions["candidate_count"]
        .reset_index(drop=True)
        .equals(expected_counts.reset_index(drop=True))
        or not decisions.loc[no_output, empty_columns].fillna("").eq("").all().all()
        or selected_geometry
        != decisions.loc[bearing, "selected_geometry_sha256"].astype(str).tolist()
    ):
        raise RuntimeError(f"D1 {name} Test ranker semantic replay differs")
    return manifest, scores, decisions


def _validate_k_gate(
    root: Path,
    *,
    scenario_id: str,
    pool: str,
    name: str,
    ranker: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    development_path = (
        root / f"11_k_sensitivity/gates/{scenario_id}/gate_selection.json"
    )
    development = load_content_manifest(
        development_path, name=f"D1 K {scenario_id} gate", statuses=("COMPLETE",)
    )
    validated = validate_gate_selection_semantics(development)
    path = root / f"08_lock/label_free_test_gates/{name}/manifest.json"
    manifest = load_content_manifest(
        path, name=f"D1 {name} Test gate", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        manifest, name=f"D1 {name} Test gate", require_at_least_one=True
    )
    sources = _mapping(manifest.get("sources"), name=f"D1 {name} gate sources")
    artifacts = _mapping(manifest.get("artifacts"), name=f"D1 {name} gate artifacts")
    inputs_path = verified_artifact_path(
        _mapping(artifacts.get("inputs"), name=f"D1 {name} gate inputs"),
        name=f"D1 {name} gate inputs",
    )
    decisions_path = verified_artifact_path(
        _mapping(artifacts.get("decisions"), name=f"D1 {name} gate decisions"),
        name=f"D1 {name} gate decisions",
    )
    candidate_path, candidates = _candidate_table(root, pool)
    feature_path = verified_artifact_path(
        _mapping(sources.get("candidate_features"), name=f"D1 {name} gate features"),
        name=f"D1 {name} gate features",
    )
    ranker_decision_path = verified_artifact_path(
        _mapping(sources.get("ranker_decisions"), name=f"D1 {name} ranker decisions"),
        name=f"D1 {name} ranker decisions",
    )
    paired_path = root / "01_manifests/d1_paired_manifest.parquet"
    feature_manifest_path = (
        root / f"03_features/test/{pool}/T2_matched_common/manifest.json"
    )
    candidate_manifest_path = root / "02_candidates/test_manifest.json"
    ranker_path = root / f"08_lock/label_free_test_rankers/{name}/manifest.json"
    expected_sources = {
        "gate_selection": artifact_record(development_path),
        "transition_model": artifact_record(validated.transition_model_path),
        "ranker_application": artifact_record(ranker_path),
        "ranker_decisions": artifact_record(ranker_decision_path),
        "feature_manifest": artifact_record(feature_manifest_path),
        "candidate_features": artifact_record(feature_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "denominator": artifact_record(paired_path),
        "inference_code": artifact_record(
            Path(__file__).resolve().parents[2]
            / "tools/d1_reranking/build_d1_formal_inputs.py"
        ),
    }
    for source, artifact_name in (
        (inputs_path, f"D1 {name} gate inputs"),
        (decisions_path, f"D1 {name} gate decisions"),
        (feature_path, f"D1 {name} gate features"),
        (ranker_decision_path, f"D1 {name} ranker decisions"),
    ):
        assert_label_free_parquet_schema(source, name=artifact_name)
    expected_inputs = build_label_free_test_gate_input_frame(
        paired=pd.read_parquet(paired_path, columns=["sample_id", "scene_id"]),
        ranker_decisions=pd.read_parquet(ranker_decision_path),
        candidate_features=pd.read_parquet(feature_path),
        candidates=candidates,
    )
    from tools.d1_reranking.apply_locked_test_gate import _apply

    expected_decisions, point = _apply(development, validated.model, expected_inputs)
    _assert_frame(
        pd.read_parquet(inputs_path), expected_inputs, name=f"D1 {name} gate inputs"
    )
    _assert_frame(
        pd.read_parquet(decisions_path),
        expected_decisions,
        name=f"D1 {name} gate decisions",
    )
    declared_point = manifest.get("selected_operating_point")
    expected_point = None if point is None else asdict(point)
    if (
        manifest.get("scenario_id") != scenario_id
        or manifest.get("pool") != pool
        or manifest.get("candidate_test_labels_read") is not False
        or sources != expected_sources
        or manifest.get("source_signature_sha256") != canonical_sha256(expected_sources)
        or declared_point != expected_point
        or manifest.get("sample_count") != len(expected_inputs)
        or manifest.get("switch_count") != int(expected_decisions["switch"].sum())
    ):
        raise RuntimeError(f"D1 {name} Test gate semantic replay differs")
    return manifest, expected_decisions


def _expected_normalized_scores(
    candidates: pd.DataFrame, source_scores: pd.DataFrame | None
) -> pd.DataFrame:
    if source_scores is None:
        return candidates[
            ["route", "sample_id", "candidate_id", "native_score", "native_rank"]
        ].rename(
            columns={
                "route": "source_route",
                "native_score": "score",
                "native_rank": "rank",
            }
        )
    ranked = rank_by_score(source_scores, score_column="ensemble_score")
    return (
        ranked[["sample_id", "candidate_id", "ensemble_score", "rerank_rank"]]
        .assign(source_route="D1")
        .rename(columns={"ensemble_score": "score", "rerank_rank": "rank"})[
            ["source_route", "sample_id", "candidate_id", "score", "rank"]
        ]
    )


def _validate_normalized(
    root: Path,
    *,
    name: str,
    pool: str,
    source_scores: pd.DataFrame | None,
    source_decisions: pd.DataFrame | None,
    selected_column: str,
) -> dict[str, Any]:
    path = root / f"08_lock/formal_inputs/{name}/manifest.json"
    manifest = load_content_manifest(
        path, name=f"D1 formal {name}", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        manifest, name=f"D1 formal {name}", require_at_least_one=True
    )
    artifacts = _mapping(manifest.get("artifacts"), name=f"D1 formal {name} artifacts")
    universe_path = verified_artifact_path(
        _mapping(artifacts.get("candidate_universe"), name=f"D1 {name} universe"),
        name=f"D1 {name} universe",
    )
    scores_path = verified_artifact_path(
        _mapping(artifacts.get("candidate_scores"), name=f"D1 {name} scores"),
        name=f"D1 {name} scores",
    )
    decisions_path = verified_artifact_path(
        _mapping(artifacts.get("per_sample_decisions"), name=f"D1 {name} decisions"),
        name=f"D1 {name} decisions",
    )
    candidate_path, candidates = _candidate_table(root, pool)
    sources = _mapping(manifest.get("sources"), name=f"D1 formal {name} sources")
    expected_sources: dict[str, Any] = {
        "candidate_manifest": artifact_record(
            root / "02_candidates/test_manifest.json"
        ),
        "candidates": artifact_record(candidate_path),
        "denominator": artifact_record(
            root / "01_manifests/d1_paired_manifest.parquet"
        ),
        "normalizer": artifact_record(
            Path(__file__).resolve().parents[2]
            / "tools/d1_reranking/build_d1_formal_inputs.py"
        ),
    }
    if name.startswith("d1_top5_r7"):
        expected_sources.update(
            {
                "feature_manifest": artifact_record(
                    root / "03_features/test/top5/T2_matched_common/manifest.json"
                ),
                "ranker_manifest": artifact_record(
                    root / "08_lock/label_free_test_rankers/d1/manifest.json"
                ),
            }
        )
        if name.endswith("gated") and not name.endswith("ungated"):
            expected_sources["gate_manifest"] = artifact_record(
                root / "08_lock/label_free_test_gates/d1/manifest.json"
            )
    elif name in {"d1_top10_locked", "d1_allnms_locked"}:
        ranker_name = "d1_top10" if pool == "top10" else "d1_allnms"
        expected_sources.update(
            {
                "k_selection": artifact_record(
                    root / "11_k_sensitivity/selection_manifest.json"
                ),
                "scenario_gate_selection": artifact_record(
                    root / "11_k_sensitivity/scenario_gate_selection.json"
                ),
                "feature_manifest": artifact_record(
                    root / f"03_features/test/{pool}/T2_matched_common/manifest.json"
                ),
                "ranker_manifest": artifact_record(
                    root
                    / f"08_lock/label_free_test_rankers/{ranker_name}/manifest.json"
                ),
                "gate_manifest": artifact_record(
                    root / f"08_lock/label_free_test_gates/{ranker_name}/manifest.json"
                ),
            }
        )
    expected_universe = candidates[
        ["route", *[column for column in UNIVERSE_COLUMNS if column != "source_route"]]
    ].rename(columns={"route": "source_route"})
    _assert_frame(
        pd.read_parquet(universe_path), expected_universe, name=f"D1 {name} universe"
    )
    _assert_frame(
        pd.read_parquet(scores_path),
        _expected_normalized_scores(candidates, source_scores),
        name=f"D1 {name} scores",
    )
    decisions = pd.read_parquet(decisions_path)
    denominator = pd.read_parquet(
        root / "01_manifests/d1_paired_manifest.parquet", columns=["sample_id"]
    )
    if (
        manifest.get("name") != name
        or manifest.get("kind") != NORMALIZED_SYSTEMS[name]["kind"]
        or manifest.get("pool") != pool
        or manifest.get("routes") != ["D1"]
        or manifest.get("candidate_test_labels_read") is not False
        or sources != expected_sources
        or decisions["sample_id"].duplicated().any()
        or set(decisions["sample_id"].astype(str))
        != set(denominator["sample_id"].astype(str))
        or len(decisions) != len(denominator)
    ):
        raise RuntimeError(f"D1 formal {name} denominator/contract differs")
    if source_decisions is None:
        expected_ids = candidates.loc[candidates["native_rank"].eq(1)].set_index(
            "sample_id"
        )["candidate_id"]
    else:
        expected_ids = source_decisions.set_index("sample_id")[selected_column].replace(
            "", np.nan
        )
    observed_ids = decisions.set_index("sample_id")["selected_candidate_id"].replace(
        "", np.nan
    )
    if (
        not observed_ids.reindex(denominator["sample_id"].astype(str))
        .fillna("")
        .astype(str)
        .equals(
            expected_ids.reindex(denominator["sample_id"].astype(str))
            .fillna("")
            .astype(str)
        )
    ):
        raise RuntimeError(f"D1 formal {name} decisions differ from source")
    geometry = candidates.copy()
    geometry[["sample_id", "candidate_id"]] = geometry[
        ["sample_id", "candidate_id"]
    ].astype(str)
    geometry_lookup = geometry.set_index(["sample_id", "candidate_id"])[
        "candidate_geometry_sha256"
    ]
    expected_geometry = []
    for row in decisions.itertuples(index=False):
        candidate_id = str(row.selected_candidate_id)
        expected_geometry.append(
            ""
            if not candidate_id
            else str(geometry_lookup.loc[(str(row.sample_id), candidate_id)])
        )
    if decisions["selected_geometry_sha256"].astype(str).tolist() != expected_geometry:
        raise RuntimeError(f"D1 formal {name} selected geometry differs")
    return manifest


def validate_k_formal_replay(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    aggregate_path = root / "11_k_sensitivity/scenario_gate_selection.json"
    aggregate = load_content_manifest(
        aggregate_path, name="D1 K scenario gates", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        aggregate, name="D1 K scenario gate aggregate", require_at_least_one=True
    )
    comparison_path = verified_artifact_path(
        _mapping(
            _mapping(aggregate.get("artifacts"), name="D1 K gate artifacts").get(
                "comparison_table"
            ),
            name="D1 K R0/R7 table",
        ),
        name="D1 K R0/R7 table",
    )
    comparison = pd.read_csv(comparison_path)
    aggregate_sources = _mapping(
        aggregate.get("sources"), name="D1 K scenario gate aggregate sources"
    )
    declared_gates = _mapping(
        aggregate_sources.get("gates"), name="D1 K scenario gate records"
    )
    if (
        len(comparison) != 6
        or set(comparison["scenario_id"]) != set(K_TEST_SCENARIOS)
        or set(comparison["role"])
        != {"native_baseline", "ungated", "expected_gain_gate"}
        or aggregate.get("allnms_t3_sensitivity_only") is not True
        or aggregate.get("candidate_test_labels_read") is not False
        or aggregate_sources.get("k_selection")
        != artifact_record(root / "11_k_sensitivity/selection_manifest.json")
        or set(declared_gates) != set(K_TEST_SCENARIOS)
    ):
        raise RuntimeError("D1 K R0/ungated/R7 comparison differs")
    for scenario_id, definition in K_TEST_SCENARIOS.items():
        gate_path = root / f"11_k_sensitivity/gates/{scenario_id}/gate_selection.json"
        gate = load_content_manifest(
            gate_path, name=f"D1 K {scenario_id} gate", statuses=("COMPLETE",)
        )
        validate_gate_selection_semantics(gate)
        records = _mapping(
            declared_gates.get(scenario_id),
            name=f"D1 K {scenario_id} aggregate records",
        )
        if records.get("selection") != artifact_record(gate_path):
            raise RuntimeError(f"D1 K {scenario_id} aggregate gate binding differs")
        metrics = _mapping(
            gate.get("validation_metrics"), name=f"D1 K {scenario_id} metrics"
        )
        expected = {
            "native_baseline": ("R0", metrics["native_j_at_1"]),
            "ungated": (
                load_content_manifest(
                    root / "11_k_sensitivity/selection_manifest.json",
                    name="D1 K selection",
                    statuses=("COMPLETE",),
                )["scenarios"][scenario_id]["method"],
                metrics["ungated_j_at_1"],
            ),
            "expected_gain_gate": ("R7", metrics["gated_j_at_1"]),
        }
        rows = comparison.loc[comparison["scenario_id"].eq(scenario_id)]
        if len(rows) != 3:
            raise RuntimeError(f"D1 K {scenario_id} comparison coverage differs")
        for role, (method, value) in expected.items():
            row = rows.loc[rows["role"].eq(role)]
            if (
                len(row) != 1
                or str(row.iloc[0]["method"]) != str(method)
                or float(row.iloc[0]["validation_j_at_1"]) != float(value)
                or row.iloc[0]["pool"] != definition["pool"]
                or row.iloc[0]["gate_decision"]
                != (
                    gate["decision"]
                    if role == "expected_gain_gate"
                    else "NOT_APPLICABLE"
                )
            ):
                raise RuntimeError(
                    f"D1 K {scenario_id}/{role} comparison metric differs"
                )

    rankers: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    ranker_scores: dict[str, pd.DataFrame] = {}
    gate_decisions: dict[str, pd.DataFrame] = {}
    for scenario_id, definition in K_TEST_SCENARIOS.items():
        ranker, scores, _ranker_decisions = _validate_k_ranker(
            root,
            scenario_id=scenario_id,
            pool=definition["pool"],
            name=definition["name"],
        )
        gate, decisions = _validate_k_gate(
            root,
            scenario_id=scenario_id,
            pool=definition["pool"],
            name=definition["name"],
            ranker=ranker,
        )
        rankers[definition["name"]] = ranker
        gates[definition["name"]] = gate
        ranker_scores[definition["name"]] = scores
        gate_decisions[definition["name"]] = decisions

    top5_ranker = load_content_manifest(
        root / "08_lock/label_free_test_rankers/d1/manifest.json",
        name="D1 Top5 ranker",
        statuses=("COMPLETE",),
    )
    top5_gate = load_content_manifest(
        root / "08_lock/label_free_test_gates/d1/manifest.json",
        name="D1 Top5 gate",
        statuses=("COMPLETE",),
    )
    for value, name in ((top5_ranker, "ranker"), (top5_gate, "gate")):
        if value.get("candidate_test_labels_read") is not False:
            raise RuntimeError(f"D1 Top5 {name} violates Test-label isolation")
        verify_artifact_records_recursive(
            value, name=f"D1 Top5 {name}", require_at_least_one=True
        )
    top5_ranker_artifacts = _mapping(
        top5_ranker.get("artifacts"), name="D1 Top5 ranker artifacts"
    )
    top5_gate_artifacts = _mapping(
        top5_gate.get("artifacts"), name="D1 Top5 gate artifacts"
    )
    top5_scores = pd.read_parquet(
        verified_artifact_path(
            _mapping(
                top5_ranker_artifacts.get("per_candidate_scores"), name="Top5 scores"
            ),
            name="D1 Top5 scores",
        )
    )
    top5_ranker_decisions = pd.read_parquet(
        verified_artifact_path(
            _mapping(
                top5_ranker_artifacts.get("per_sample_decisions"), name="Top5 decisions"
            ),
            name="D1 Top5 decisions",
        )
    )
    top5_gate_decisions = pd.read_parquet(
        verified_artifact_path(
            _mapping(top5_gate_artifacts.get("decisions"), name="Top5 gate decisions"),
            name="D1 Top5 gate decisions",
        )
    )
    normalized: dict[str, Any] = {}
    normalized["d1_top5_r0"] = _validate_normalized(
        root,
        name="d1_top5_r0",
        pool="top5",
        source_scores=None,
        source_decisions=None,
        selected_column="selected_candidate_id",
    )
    normalized["d1_top5_r7_ungated"] = _validate_normalized(
        root,
        name="d1_top5_r7_ungated",
        pool="top5",
        source_scores=top5_scores,
        source_decisions=top5_ranker_decisions,
        selected_column="selected_candidate_id",
    )
    normalized["d1_top5_r7_gated"] = _validate_normalized(
        root,
        name="d1_top5_r7_gated",
        pool="top5",
        source_scores=top5_scores,
        source_decisions=top5_gate_decisions,
        selected_column="selected_candidate_id",
    )
    for name, pool in (("d1_top10", "top10"), ("d1_allnms", "allnms")):
        normalized[f"{name}_locked"] = _validate_normalized(
            root,
            name=f"{name}_locked",
            pool=pool,
            source_scores=ranker_scores[name],
            source_decisions=gate_decisions[name],
            selected_column="selected_candidate_id",
        )
    access_record = _validate_test_access_events(root)
    records = {
        "scenario_gate_selection": artifact_record(aggregate_path),
        "comparison_table": artifact_record(comparison_path),
        "rankers": {
            name: artifact_record(
                root / f"08_lock/label_free_test_rankers/{name}/manifest.json"
            )
            for name in rankers
        },
        "gates": {
            name: artifact_record(
                root / f"08_lock/label_free_test_gates/{name}/manifest.json"
            )
            for name in gates
        },
        "normalized": {
            name: artifact_record(root / f"08_lock/formal_inputs/{name}/manifest.json")
            for name in normalized
        },
        "test_access_log": access_record,
        "replay_code": artifact_record(Path(__file__)),
    }
    return records, {
        "scenario_gate_count": 2,
        "label_free_test_ranker_count": 2,
        "label_free_test_gate_count": 2,
        "normalized_formal_input_count": 5,
        "allnms_t3_sensitivity_only": True,
        "candidate_test_labels_read": False,
    }


__all__ = ["validate_k_formal_replay"]
