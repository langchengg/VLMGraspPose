from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from reranking.crog_existing_comparison import (
    CrogExistingComparisonError,
    run_crog_existing_comparisons,
    validate_crog_existing_comparison_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": len(payload),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    return _write(
        path,
        b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
            for row in rows
        ),
    )


def _pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for query_index in range(2):
        for candidate_index in range(5):
            rows.append(
                {
                    "query_id": f"multiple:test:{query_index:08d}",
                    "candidate_id": f"candidate_{candidate_index}",
                    "candidate_checksum": hashlib.sha256(
                        f"{query_index}:{candidate_index}".encode()
                    ).hexdigest(),
                    "label": int(
                        (query_index == 0 and candidate_index == 1)
                        or (query_index == 1 and candidate_index == 0)
                    ),
                    "scene_id": f"scene-{query_index}",
                    "frame_id": f"frame-{query_index}",
                }
            )
    pool = pd.DataFrame(rows)
    baseline = pool[["query_id", "candidate_id", "candidate_checksum"]].copy()
    baseline["score"] = baseline["candidate_id"].str[-1].astype(int).rsub(5)
    universe = pool[["query_id", "scene_id", "frame_id"]].drop_duplicates()
    return pool, baseline, universe


def _complete_run(
    root: Path,
    method: str,
    *,
    include_predictions: bool = True,
    include_candidate_checksums: bool = True,
    alter_candidate: bool = False,
    alter_checksum: bool = False,
) -> None:
    checkpoints = [
        _write(root / "models" / f"{method}.fold{fold}.pt", f"fold{fold}".encode())
        for fold in range(2)
    ]
    oof = _write_jsonl(root / "oof" / f"{method}.oof_predictions.jsonl", [{}])
    candidate = _write(root / "data" / "crog_frozen_candidates.jsonl", b"pool\n")
    evaluator = _write(root / "code" / "crog_evaluator.py", b"def evaluate(): pass\n")
    block: dict[str, object] = {
        "status": "complete",
        "checkpoint": _write(root / "models" / f"{method}.final.pt", b"final"),
        "oof": {
            "status": "complete",
            "expected_folds": [0, 1],
            "completed_folds": [0, 1],
            "fold_models": checkpoints,
            "fold_count": 2,
            "oof_predictions": oof,
        },
        "independent_recomputation": {
            "status": "passed",
            "exact_match": True,
            "methods": [method],
        },
    }
    if include_predictions:
        rows: list[dict[str, object]] = []
        for query_index in range(2):
            candidate_order = (
                ["candidate_1", "candidate_0", "candidate_2", "candidate_3", "candidate_4"]
                if query_index == 0
                else ["candidate_0", "candidate_1", "candidate_2", "candidate_3", "candidate_4"]
            )
            if alter_candidate and query_index == 0:
                candidate_order[-1] = "candidate_99"
            checksum_ids = [f"candidate_{index}" for index in range(5)]
            checksums = [
                hashlib.sha256(f"{query_index}:{index}".encode()).hexdigest()
                for index in range(5)
            ]
            if alter_checksum and query_index == 0:
                checksums[0] = "f" * 64
            row: dict[str, object] = {
                "sample_id": f"multiple:test:{query_index:08d}",
                "method": method,
                "candidate_order": candidate_order,
            }
            if include_candidate_checksums:
                row["candidate_checksum_ids"] = checksum_ids
                row["candidate_checksums"] = checksums
            rows.append(row)
        block["test_predictions"] = _write_jsonl(
            root / "frozen" / f"{method}.test_predictions.jsonl", rows
        )
    manifest = {
        "schema_version": 1,
        "kind": "frozen_experiment_manifest",
        "run_id": root.name,
        "status": "locked",
        "primary_method": method,
        "formal_methods": [method],
        "methods": {method: block},
        "candidate_artifact": candidate,
        "evaluator": evaluator,
    }
    path = root / "frozen_experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_complete_method_is_exact_joined_recomputed_and_materialized(
    tmp_path: Path,
) -> None:
    method = "v3_fcer_native"
    run = tmp_path / "existing" / "opaque"
    _complete_run(run, method)
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[tmp_path / "existing"],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=100,
    )

    assert report["status"] == "complete"
    assert report["eligible_count"] == 1
    assert report["comparison_count"] == 1
    comparison = report["comparisons"][0]
    assert comparison["method_id"] == method
    assert comparison["comparison"]["reference_metrics"]["j_at_1"] == 0.5
    assert comparison["comparison"]["challenger_metrics"]["j_at_1"] == 1.0
    evidence = comparison["comparison_evidence"]
    assert evidence["exact_candidate_join"] is True
    assert evidence["independent_recomputed"] is True
    for descriptor in evidence["artifacts"].values():
        path = Path(descriptor["path"])
        assert path.is_file()
        assert _sha256(path) == descriptor["sha256"]
    normalized = pd.read_json(evidence["artifacts"]["predictions"]["path"], lines=True)
    first = normalized[normalized["query_id"].eq("multiple:test:00000000")]
    assert first.sort_values("rank").iloc[0]["candidate_id"] == "candidate_1"
    with Path(evidence["artifacts"]["predictions"]["path"]).open("ab") as stream:
        stream.write(b"tamper\n")
    with pytest.raises(CrogExistingComparisonError, match="hash-invalid"):
        validate_crog_existing_comparison_report(report)


def test_eligible_provenance_without_prediction_is_not_a_comparison(
    tmp_path: Path,
) -> None:
    method = "v2_locked_primary"
    run = tmp_path / "existing" / "run"
    _complete_run(run, method, include_predictions=False)
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[run],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["eligible_count"] == 1
    assert report["comparison_count"] == 0
    assert report["status"] == "incomplete"
    exclusion = next(item for item in report["exclusions"] if item["method_id"] == method)
    assert exclusion["stage"] == "prediction_materialization"
    assert exclusion["exclusion"]["code"] == "frozen_prediction_missing_or_hash_invalid"


def test_changed_candidate_identity_excludes_eligible_method(tmp_path: Path) -> None:
    method = "v3_locked_primary"
    run = tmp_path / "existing" / "run"
    _complete_run(run, method, alter_candidate=True)
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[run],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["comparison_count"] == 0
    exclusion = next(item for item in report["exclusions"] if item["method_id"] == method)
    assert exclusion["stage"] == "independent_recomputation"
    assert exclusion["exclusion"]["code"] == "frozen_prediction_recomputation_failed"
    assert (
        "candidate checksums do not cover its frozen order"
        in exclusion["exclusion"]["message"]
    )


def test_changed_candidate_checksum_excludes_eligible_method(tmp_path: Path) -> None:
    method = "v3_fcer_native"
    run = tmp_path / "existing" / "run"
    _complete_run(run, method, alter_checksum=True)
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[run],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["comparison_count"] == 0
    exclusion = next(item for item in report["exclusions"] if item["method_id"] == method)
    assert exclusion["stage"] == "independent_recomputation"
    assert "changed candidate_identity_sha256" in exclusion["exclusion"]["message"]


def test_generic_candidate_keys_cannot_prove_same_pool_without_checksums(
    tmp_path: Path,
) -> None:
    method = "v3_fcer_native"
    run = tmp_path / "existing" / "run"
    _complete_run(run, method, include_candidate_checksums=False)
    pool, baseline, universe = _pool()
    changed_identity = "e" * 64
    key = pool["query_id"].eq("multiple:test:00000000") & pool[
        "candidate_id"
    ].eq("candidate_0")
    pool.loc[key, "candidate_checksum"] = changed_identity
    baseline.loc[key, "candidate_checksum"] = changed_identity

    report = run_crog_existing_comparisons(
        roots=[run],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["eligible_count"] == 1
    assert report["comparison_count"] == 0
    assert report["status"] == "incomplete"
    exclusion = next(item for item in report["exclusions"] if item["method_id"] == method)
    assert exclusion["stage"] == "independent_recomputation"
    assert "lacks per-candidate checksum identities" in exclusion["exclusion"]["message"]


def test_matching_frozen_checksums_complete_same_pool_comparison(
    tmp_path: Path,
) -> None:
    method = "v3_fcer_native"
    run = tmp_path / "existing" / "run"
    _complete_run(run, method)
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[run],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["status"] == "complete"
    assert report["eligible_count"] == report["comparison_count"] == 1


def test_no_eligible_method_has_verifiable_per_check_exclusions(tmp_path: Path) -> None:
    run = tmp_path / "existing" / "run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"status": "running", "method": "unfinished_fcer"}) + "\n"
    )
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[run],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["status"] == "complete_no_eligible"
    assert report["eligible_count"] == report["comparison_count"] == 0
    exclusion = report["exclusions"][0]
    assert set(exclusion["eligibility_checks"]) == {
        "manifest",
        "complete_oof",
        "checkpoint",
        "candidate_hash",
        "evaluator_hash",
        "independent_recomputation",
    }
    assert all(
        "passed" in result for result in exclusion["eligibility_checks"].values()
    )
    assert Path(report["artifacts"]["exclusions"]["path"]).is_file()


def test_validator_rejects_provenance_only_fake_comparison() -> None:
    fake = {
        "status": "complete",
        "eligible_count": 1,
        "comparison_count": 1,
        "comparisons": [
            {
                "method_id": "provenance_only",
                "comparison_evidence": {
                    "exact_candidate_join": True,
                    "independent_recomputed": False,
                    "artifacts": {"provenance": {"sha256": "0" * 64}},
                },
            }
        ],
    }

    with pytest.raises(CrogExistingComparisonError, match="audit artifacts|evaluator output"):
        validate_crog_existing_comparison_report(fake)


def test_non_top5_pool_is_rejected_before_discovery(tmp_path: Path) -> None:
    pool, baseline, universe = _pool()
    pool = pool[~pool["candidate_id"].eq("candidate_4")]
    baseline = baseline.merge(
        pool[["query_id", "candidate_id"]],
        on=["query_id", "candidate_id"],
        how="inner",
    )

    with pytest.raises(CrogExistingComparisonError, match="not exactly Top-5"):
        run_crog_existing_comparisons(
            roots=[tmp_path],
            candidate_pool=pool,
            reference_predictions=baseline,
            query_universe=universe,
            output_dir=tmp_path / "output",
            bootstrap_iterations=50,
        )


def test_empty_discovery_still_materializes_scan_exclusion(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    pool, baseline, universe = _pool()

    report = run_crog_existing_comparisons(
        roots=[empty_root],
        candidate_pool=pool,
        reference_predictions=baseline,
        query_universe=universe,
        output_dir=tmp_path / "output",
        bootstrap_iterations=50,
    )

    assert report["status"] == "complete_no_eligible"
    assert report["comparison_count"] == 0
    assert report["excluded_count"] == 1
    assert (
        report["exclusions"][0]["exclusion"]["code"]
        == "no_existing_methods_discovered"
    )
    assert report["exclusions"][0]["exclusion"]["scan"]["documents"] == 0
