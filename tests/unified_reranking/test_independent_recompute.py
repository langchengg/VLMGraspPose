from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.unified_reranking.create_formal_test_lock import create_unified_formal_lock
from tools.unified_reranking.independent_recompute import (
    _bridge_pool_checks,
    _exact_mcnemar,
    _selection_comparison,
    run,
)
from tools.unified_reranking.run_formal_test_once import execute_formal_test_once
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.postformal_reporting import (
    _independent_recompute_integrity,
    load_formal_bundle,
)


FORMAL_TEST_PATH = Path(__file__).with_name("test_formal_test_orchestration.py")
SPEC = importlib.util.spec_from_file_location(
    "synthetic_formal_fixture", FORMAL_TEST_PATH
)
assert SPEC is not None and SPEC.loader is not None
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def test_exact_mcnemar_is_symmetric_and_handles_no_discordance() -> None:
    assert _exact_mcnemar(0, 0) == 1.0
    assert _exact_mcnemar(1, 4) == _exact_mcnemar(4, 1)


def test_independent_union_same_native_g1_candidate_is_not_a_switch() -> None:
    reference = pd.DataFrame(
        {
            "selected_route": ["g1"],
            "selected_candidate_id": ["same"],
            "independent_correct": [True],
        }
    )
    union = pd.DataFrame(
        {
            "selected_route": ["g1"],
            "selected_candidate_id": ["G1:same"],
            "source_candidate_id": ["same"],
            "independent_correct": [True],
        }
    )
    comparison = _selection_comparison(reference, union, oracle=1.0, union=True)
    assert comparison["switch_count"] == 0
    assert comparison["switch_rate"] == 0.0


def test_standalone_recompute_matches_synthetic_formal_execution(
    tmp_path: Path,
) -> None:
    run_dir, plan, _labels = FIXTURE._build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    execute_formal_test_once(run_dir=run_dir, command="synthetic-independent")
    tiny = {
        "status": "PASS",
        "formal_test_execution_count": 1,
    }
    tiny["content_sha256"] = canonical_sha256(tiny)
    independent_path = run_dir / "15_independent_recompute/INDEPENDENT_RECOMPUTE.json"
    independent_path.parent.mkdir(parents=True, exist_ok=True)
    _json(independent_path, tiny)
    tiny_ok, tiny_details = _independent_recompute_integrity(run_dir)
    assert tiny_ok is False
    assert any("nonpositive_metric_checks" in item for item in tiny_details["errors"])
    result = run(run_dir)
    assert result["status"] == "PASS"
    assert result["sample_count"] == 8
    assert result["system_count"] == 10
    assert result["metric_checks"] > 0
    assert result["comparison_checks"] > 0
    assert Path(result["artifacts"]["metrics"]["path"]).is_file()
    report = Path(result["artifacts"]["report"]["path"])
    assert report.name == "INDEPENDENT_RECOMPUTE.md"
    assert result["artifacts"]["report"]["sha256"] == sha256_file(report)
    assert _independent_recompute_integrity(run_dir)[0] is True


def test_postformal_rejects_execution_manifest_redirect(tmp_path: Path) -> None:
    run_dir, plan, _labels = FIXTURE._build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    execute_formal_test_once(run_dir=run_dir, command="synthetic-redirect")
    manifest_path = run_dir / "09_formal_test/formal_test_manifest.json"
    redirected = run_dir / "09_formal_test/redirected_formal_manifest.json"
    redirected.write_bytes(manifest_path.read_bytes())
    execution_path = run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["artifacts"]["formal_test_manifest"] = {
        "path": str(redirected.resolve()),
        "sha256": sha256_file(redirected),
    }
    _json(execution_path, execution)
    with pytest.raises(PermissionError, match="redirects"):
        load_formal_bundle(run_dir)


def test_postformal_rejects_rebound_manifest_with_invalid_content_hash(
    tmp_path: Path,
) -> None:
    run_dir, plan, _labels = FIXTURE._build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    execute_formal_test_once(run_dir=run_dir, command="synthetic-tamper")
    manifest_path = run_dir / "09_formal_test/formal_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fabricated_field"] = "tampered"
    _json(manifest_path, manifest)
    execution_path = run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["artifacts"]["formal_test_manifest"]["sha256"] = sha256_file(
        manifest_path
    )
    _json(execution_path, execution)
    with pytest.raises(PermissionError, match="content hash mismatch"):
        load_formal_bundle(run_dir)


def _formal_run(root: Path, *, union: bool = False) -> tuple[Path, Path]:
    run_dir, plan, labels = FIXTURE._build_synthetic_run(
        root, positive_union=union
    )
    if union:
        FIXTURE._add_positive_union_system(run_dir, plan)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    execute_formal_test_once(run_dir=run_dir, command="synthetic-independent")
    return run_dir, labels


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rebind_formal_artifact(
    run_dir: Path, manifest_key: str, execution_key: str
) -> None:
    manifest_path = run_dir / "09_formal_test/formal_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = Path(manifest["artifacts"][manifest_key]["path"])
    manifest["artifacts"][manifest_key]["sha256"] = sha256_file(artifact_path)
    _json(manifest_path, manifest)
    execution_path = run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["artifacts"][execution_key] = {
        "path": str(artifact_path.resolve()),
        "sha256": sha256_file(artifact_path),
    }
    execution["artifacts"]["formal_test_manifest"]["sha256"] = sha256_file(
        manifest_path
    )
    _json(execution_path, execution)


def _mutate_union_ranking(run_dir: Path, column: str, value: object) -> None:
    path = run_dir / "09_formal_test/formal_test_realized_rankings.parquet"
    ranking = pd.read_parquet(path)
    index = ranking.index[ranking["system_name"].eq("top15_union_primary")][0]
    ranking.loc[index, column] = value
    ranking.to_parquet(path, index=False)
    _rebind_formal_artifact(
        run_dir, "realized_rankings", "formal_test_realized_rankings"
    )


def _mutate_formal_bridge_output(run_dir: Path, column: str, value: object) -> None:
    path = run_dir / "09_formal_test/bridge_per_candidate_scores.parquet"
    scores = pd.read_parquet(path)
    scores.loc[0, column] = value
    scores.to_parquet(path, index=False)
    _rebind_formal_artifact(
        run_dir, "bridge_per_candidate_scores", "bridge_per_candidate_scores"
    )


def _resign_bridge_formula_mutation(run_dir: Path) -> None:
    plan_path = run_dir / "08_lock/formal_evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    bridge_manifest_path = Path(plan["test_bridge_contract"]["manifest"]["path"])
    bridge_manifest = json.loads(bridge_manifest_path.read_text(encoding="utf-8"))
    bundle_path = Path(bridge_manifest["artifacts"]["candidate_bundle"]["path"])
    bundle = pd.read_parquet(bundle_path)
    index = bundle.index[bundle["candidate_pool_contract"].eq("fair_gaussian")][0]
    bundle.loc[index, "historical_selector_score"] += 0.125
    bundle.to_parquet(bundle_path, index=False)
    denominator = (
        pd.read_parquet(plan["sample_manifest"])["sample_id"].astype(str).tolist()
    )
    bridge_manifest["artifacts"]["candidate_bundle"]["sha256"] = sha256_file(
        bundle_path
    )
    bridge_manifest["pool_checks"] = _bridge_pool_checks(bundle, denominator)
    unsigned_bridge = dict(bridge_manifest)
    unsigned_bridge.pop("content_sha256", None)
    bridge_manifest["content_sha256"] = canonical_sha256(unsigned_bridge)
    _json(bridge_manifest_path, bridge_manifest)

    bridge_record = {
        "path": str(bridge_manifest_path.resolve()),
        "sha256": sha256_file(bridge_manifest_path),
    }
    bundle_record = {
        "path": str(bundle_path.resolve()),
        "sha256": sha256_file(bundle_path),
    }
    plan["test_bridge_contract"]["manifest"] = bridge_record
    plan["test_bridge_contract"]["candidate_bundle"] = bundle_record
    plan["bound_provenance"]["test_bridge_manifest"] = bridge_record
    _json(plan_path, plan)

    lock_path = run_dir / "08_lock/FORMAL_TEST_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["locked_files"]["formal_evaluation_plan"]["sha256"] = sha256_file(plan_path)
    lock["locked_files"]["formal_test_bridge_manifest"]["sha256"] = sha256_file(
        bridge_manifest_path
    )
    lock["locked_files"]["formal_test_bridge_candidate_bundle"]["sha256"] = sha256_file(
        bundle_path
    )
    unsigned_lock = dict(lock)
    unsigned_lock.pop("self_sha256", None)
    lock["self_sha256"] = canonical_sha256(unsigned_lock)
    _json(lock_path, lock)

    formal_bridge_path = run_dir / "09_formal_test/bridge_per_candidate_scores.parquet"
    formal_bridge = pd.read_parquet(formal_bridge_path)
    formal_index = formal_bridge.index[
        formal_bridge["candidate_id"].eq(bundle.loc[index, "candidate_id"])
        & formal_bridge["sample_id"].eq(bundle.loc[index, "sample_id"])
        & formal_bridge["route"].eq(bundle.loc[index, "route"])
    ][0]
    formal_bridge.loc[formal_index, "historical_selector_score"] = bundle.loc[
        index, "historical_selector_score"
    ]
    formal_bridge.to_parquet(formal_bridge_path, index=False)
    manifest_path = run_dir / "09_formal_test/formal_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_lock"]["file_sha256"] = sha256_file(lock_path)
    manifest["formal_lock"]["self_sha256"] = lock["self_sha256"]
    manifest["evaluation_plan"]["sha256"] = sha256_file(plan_path)
    manifest["test_bridge"]["manifest"] = bridge_record
    manifest["test_bridge"]["candidate_bundle"] = bundle_record
    manifest["artifacts"]["bridge_per_candidate_scores"]["sha256"] = sha256_file(
        formal_bridge_path
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("content_sha256", None)
    manifest["content_sha256"] = canonical_sha256(unsigned_manifest)
    _json(manifest_path, manifest)

    execution_path = run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["formal_lock_file_sha256"] = sha256_file(lock_path)
    execution["formal_lock_self_sha256"] = lock["self_sha256"]
    execution["artifacts"]["bridge_per_candidate_scores"]["sha256"] = sha256_file(
        formal_bridge_path
    )
    execution["artifacts"]["formal_test_manifest"]["sha256"] = sha256_file(
        manifest_path
    )
    _json(execution_path, execution)


def test_union_recompute_checks_qualified_duplicate_ids_and_top15(
    tmp_path: Path,
) -> None:
    run_dir, _labels = _formal_run(tmp_path, union=True)
    result = run(run_dir)
    assert result["system_count"] == 11
    metrics = json.loads(
        Path(result["artifacts"]["metrics"]["path"]).read_text(encoding="utf-8")
    )["systems"]["top15_union_primary"]
    assert metrics["oracle_at_15"] == 1.0
    assert metrics["j_at_15"] == 1.0
    per_sample = pd.read_parquet(result["artifacts"]["per_sample"]["path"])
    union = per_sample.loc[per_sample["system_name"].eq("top15_union_primary")]
    assert union["selected_candidate_id"].str.contains(":", regex=False).all()
    assert union["source_candidate_id"].eq("b").all()


def test_union_recompute_rejects_qualifier_drift(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path, union=True)
    _mutate_union_ranking(run_dir, "candidate_id", "WRONG:b")
    with pytest.raises(ValueError, match="qualifier/source identity"):
        run(run_dir)


def test_union_recompute_rejects_geometry_drift(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path, union=True)
    _mutate_union_ranking(run_dir, "candidate_geometry_sha256", "wrong-geometry")
    with pytest.raises(ValueError, match="geometry binding mismatch"):
        run(run_dir)


def test_union_recompute_rejects_invalid_top15_rank(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path, union=True)
    _mutate_union_ranking(run_dir, "rank", 16)
    with pytest.raises(ValueError, match="Top-15"):
        run(run_dir)


def test_recompute_rejects_formal_bridge_membership_tamper(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    _mutate_formal_bridge_output(run_dir, "candidate_id", "G1::fair_gaussian::wrong")
    with pytest.raises(AssertionError, match="membership/geometry"):
        run(run_dir)


def test_recompute_rejects_formal_bridge_label_tamper(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    path = run_dir / "09_formal_test/bridge_per_candidate_scores.parquet"
    scores = pd.read_parquet(path)
    _mutate_formal_bridge_output(
        run_dir, "candidate_success", not bool(scores.loc[0, "candidate_success"])
    )
    with pytest.raises(AssertionError, match="candidate labels mismatch"):
        run(run_dir)


def test_recompute_rejects_resigned_bridge_formula_tamper(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    _resign_bridge_formula_mutation(run_dir)
    with pytest.raises(AssertionError, match="selector formula mismatch"):
        run(run_dir)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "unknown_top_level"])
def test_recompute_rejects_metric_inventory_drift(
    tmp_path: Path, mutation: str
) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    path = run_dir / "09_formal_test/formal_test_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        metrics["systems"]["crog_native"].pop("mrr_at_5")
    elif mutation == "unknown":
        metrics["systems"]["crog_native"]["unlocked_metric"] = 1.0
    else:
        metrics["unlocked_metric_inventory"] = {}
    _json(path, metrics)
    _rebind_formal_artifact(run_dir, "metrics", "formal_test_metrics")
    with pytest.raises(AssertionError, match="metric inventory"):
        run(run_dir)


def test_recompute_rejects_statistical_contingency_drift(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    path = run_dir / "09_formal_test/formal_test_statistics.json"
    statistics = json.loads(path.read_text(encoding="utf-8"))
    statistics["comparisons"]["crog_ungated"]["mcnemar_conventional_supportive"][
        "net_recovered"
    ] += 1
    _json(path, statistics)
    _rebind_formal_artifact(run_dir, "statistics", "formal_test_statistics")
    with pytest.raises(AssertionError, match="net_recovered"):
        run(run_dir)


def test_recompute_rejects_persisted_outcome_drift(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    for filename in ("formal_test_per_sample.parquet", "per_sample_decisions.parquet"):
        path = run_dir / "09_formal_test" / filename
        decisions = pd.read_parquet(path)
        decisions.loc[0, "selected_correct"] = not bool(
            decisions.loc[0, "selected_correct"]
        )
        decisions.to_parquet(path, index=False)
    _rebind_formal_artifact(run_dir, "per_sample", "formal_test_per_sample")
    _rebind_formal_artifact(run_dir, "per_sample_decisions", "per_sample_decisions")
    with pytest.raises(AssertionError, match="selected_correct differs"):
        run(run_dir)


def test_recompute_rejects_three_route_intersection_drift(tmp_path: Path) -> None:
    run_dir, _labels = _formal_run(tmp_path)
    path = run_dir / "09_formal_test/formal_test_statistics.json"
    statistics = json.loads(path.read_text(encoding="utf-8"))
    statistics["three_route_outcome_intersections"]["000"] += 1
    _json(path, statistics)
    _rebind_formal_artifact(run_dir, "statistics", "formal_test_statistics")
    with pytest.raises(AssertionError, match="three_route_outcome_intersections"):
        run(run_dir)


def _resign_label_contract(run_dir: Path, labels_path: Path) -> None:
    plan = json.loads(
        (run_dir / "08_lock/formal_evaluation_plan.json").read_text(encoding="utf-8")
    )
    label_manifest_path = Path(plan["candidate_label_manifest"])
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    label_manifest["candidate_labels_sha256"] = sha256_file(labels_path)
    _json(label_manifest_path, label_manifest)
    lock_path = run_dir / "08_lock/FORMAL_TEST_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["locked_files"]["formal_candidate_label_manifest"]["sha256"] = sha256_file(
        label_manifest_path
    )
    unsigned = dict(lock)
    unsigned.pop("self_sha256", None)
    lock["self_sha256"] = canonical_sha256(unsigned)
    _json(lock_path, lock)
    manifest_path = run_dir / "09_formal_test/formal_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_lock"]["file_sha256"] = sha256_file(lock_path)
    manifest["formal_lock"]["self_sha256"] = lock["self_sha256"]
    manifest["candidate_test_labels"]["sha256"] = sha256_file(labels_path)
    manifest["candidate_test_labels"]["row_count"] = len(pd.read_parquet(labels_path))
    _json(manifest_path, manifest)
    execution_path = run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["formal_lock_file_sha256"] = sha256_file(lock_path)
    execution["formal_lock_self_sha256"] = lock["self_sha256"]
    execution["artifacts"]["formal_test_manifest"]["sha256"] = sha256_file(
        manifest_path
    )
    _json(execution_path, execution)


def test_recompute_rejects_incomplete_normalized_all_label_coverage(
    tmp_path: Path,
) -> None:
    run_dir, labels = _formal_run(tmp_path)
    incomplete = pd.read_parquet(labels).iloc[:-1].copy()
    incomplete.to_parquet(labels, index=False)
    _resign_label_contract(run_dir, labels)
    with pytest.raises(ValueError, match="exactly cover locked All pools"):
        run(run_dir)
