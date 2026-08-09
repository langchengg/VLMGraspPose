from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from reranking import matrix as matrix_module
from reranking.matrix import (
    DatasetArtifact,
    MatrixError,
    _r10_method_key,
    _r10_post_lock_comparison,
    _validate_r10_discovery_report,
)


_CHECKS = (
    "manifest",
    "complete_oof",
    "checkpoint",
    "candidate_hash",
    "evaluator_hash",
    "independent_recomputation",
)


def _descriptor(path: Path, payload: bytes = b"artifact\n") -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _discovery(run_root: Path, *, eligible: bool) -> dict[str, Any]:
    method_id = "v3_fcer_native"
    checks = {
        name: {"passed": eligible, "evidence": [], "issues": []}
        for name in _CHECKS
    }
    method = {
        "method_id": method_id,
        "run_id": "frozen-run",
        "run_root": str(run_root),
        "eligible": eligible,
        "checks": checks,
        "gaps": [] if eligible else [{"code": "complete_oof_missing"}],
    }
    eligible_rows = (
        [
            {
                "method_id": method_id,
                "run_id": "frozen-run",
                "run_root": str(run_root),
            }
        ]
        if eligible
        else []
    )
    return {
        "schema_version": 1,
        "kind": "existing_reranker_r10_discovery",
        "roots": [str(run_root)],
        "scan": {"documents": 1, "errors": []},
        "methods": [method],
        "eligible_methods": eligible_rows,
        "ineligible_methods": [] if eligible else [method_id],
    }


def _crog_inputs() -> tuple[
    DatasetArtifact,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    rows: list[dict[str, Any]] = []
    for query_index in range(2):
        for candidate_index in range(5):
            rows.append(
                {
                    "query_id": f"multiple:test:{query_index:08d}",
                    "candidate_id": f"candidate_{candidate_index}",
                    "candidate_identity_sha256": hashlib.sha256(
                        f"{query_index}:{candidate_index}".encode()
                    ).hexdigest(),
                    "scene_id": f"scene-{query_index}",
                    "frame_id": f"frame-{query_index}",
                    "q_raw": 1.0 - candidate_index / 10.0,
                    "label": int(candidate_index == query_index),
                }
            )
    frame = pd.DataFrame(rows)
    reference = frame[
        ["query_id", "candidate_id", "scene_id", "frame_id", "label"]
    ].copy()
    universe = reference[["query_id", "scene_id", "frame_id"]].drop_duplicates()
    baseline = frame[["query_id", "candidate_id"]].assign(score=frame["q_raw"])
    artifact = DatasetArtifact(
        "crog_full_post_filter",
        "crog",
        "full_post_filter",
        "test",
        Path("unused-features.parquet"),
        Path("unused-labels.parquet"),
    )
    return artifact, frame, reference, universe, baseline


def _fake_report(
    output_dir: Path,
    *,
    status: str,
    eligible_count: int,
    with_comparison: bool,
    run_root: str = "/frozen/run",
) -> dict[str, Any]:
    discovery = _descriptor(output_dir / "discovery.json")
    exclusions = _descriptor(output_dir / "exclusions.json")
    reference = _descriptor(output_dir / "reference_predictions.jsonl")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "crog_existing_r10_comparison",
        "dataset": "CROG",
        "scope": "test",
        "status": status,
        "eligible_count": eligible_count,
        "comparison_count": int(with_comparison),
        "excluded_count": 0 if with_comparison else 1,
        "comparisons": [],
        "exclusions": (
            []
            if with_comparison
            else [
                {
                    "method_id": None if eligible_count == 0 else "v3_fcer_native",
                    "stage": "eligibility" if eligible_count == 0 else "recomputation",
                    "exclusion": {"code": "no_eligible" if eligible_count == 0 else "failed"},
                }
            ]
        ),
        "candidate_pool_identity_sha256": "a" * 64,
        "artifacts": {
            "discovery": discovery,
            "exclusions": exclusions,
            "reference_predictions": reference,
        },
    }
    if with_comparison:
        method_root = output_dir / "methods" / "v3_fcer_native"
        artifacts = {
            role: _descriptor(method_root / filename)
            for role, filename in {
                "predictions": "predictions.jsonl",
                "per_query": "per_query.jsonl",
                "comparison": "comparison.json",
                "provenance": "provenance.json",
            }.items()
        }
        evaluator = _descriptor(output_dir / "evaluator.py")
        report["comparisons"] = [
            {
                "method_id": "v3_fcer_native",
                "run_id": "frozen-run",
                "run_root": run_root,
                "comparison": {
                    "reference_metrics": {"j_at_1": 0.5, "j_at_1_count": 1},
                    "challenger_metrics": {
                        "j_at_1": 1.0,
                        "j_at_1_count": 2,
                        "query_count": 2,
                    },
                    "switch_metrics": {
                        "query_count": 2,
                        "switch_count": 1,
                        "recovered": 1,
                        "harmful": 0,
                    },
                },
                "comparison_evidence": {
                    "exact_candidate_join": True,
                    "independent_recomputed": True,
                    "candidate_pool_identity_sha256": "a" * 64,
                    "evaluator": evaluator,
                    "artifacts": artifacts,
                },
            }
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _prepare_output(tmp_path: Path, discovery: dict[str, Any]) -> Path:
    output = tmp_path / "matrix"
    for directory in ("audit", "metrics", "manifests/experiments", "predictions"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "audit" / "r10_existing_run_discovery.json").write_text(
        json.dumps(discovery, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def test_post_lock_r10_uses_same_pool_and_materializes_nonprimary_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery = _discovery(tmp_path / "legacy", eligible=True)
    output = _prepare_output(tmp_path, discovery)
    artifact, frame, reference, universe, baseline = _crog_inputs()
    observed: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return _fake_report(
            Path(kwargs["output_dir"]),
            status="complete",
            eligible_count=1,
            with_comparison=True,
            run_root=str(discovery["eligible_methods"][0]["run_root"]),
        )

    monkeypatch.setattr(matrix_module, "run_crog_existing_comparisons", fake_run)
    paths, summary, rows = _r10_post_lock_comparison(
        output,
        {"matrix_profile": "unit_test", "bootstrap_iterations": 17},
        artifact,
        frame,
        reference,
        universe,
        baseline,
    )

    candidate_pool = observed["candidate_pool"]
    assert len(candidate_pool) == len(reference) == 10
    assert "candidate_identity_sha256" in candidate_pool
    assert set(map(tuple, candidate_pool[["query_id", "candidate_id"]].to_numpy())) == set(
        map(tuple, reference[["query_id", "candidate_id"]].to_numpy())
    )
    pd.testing.assert_frame_equal(observed["reference_predictions"], baseline)
    pd.testing.assert_frame_equal(observed["query_universe"], universe)
    assert observed["discovery_report"] == discovery
    assert observed["top_k"] == 5
    assert observed["bootstrap_iterations"] == 17
    assert summary["status"] == "complete"
    assert summary["primary_reselection_permitted"] is False
    assert len(rows) == 1
    assert rows[0]["rung"] == "R10"
    assert rows[0]["eligible_for_primary_reselection"] is False
    assert rows[0]["trained_by_current_matrix"] is False
    assert rows[0]["j_at_1"] == 1.0
    assert rows[0]["method"] == _r10_method_key(
        "v3_fcer_native", str(discovery["eligible_methods"][0]["run_root"])
    )
    assert Path(rows[0]["r10_comparison_report"]) in paths
    manifest = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
        if path.parent.name == "experiments"
    )
    assert manifest["stage"] == "test-post-lock"
    assert manifest["prediction_designation"] == "POST_LOCK_R10_EXISTING_COMPARISON"


def test_post_lock_r10_fails_stage_when_any_eligible_method_is_uncompared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery = _discovery(tmp_path / "legacy", eligible=True)
    output = _prepare_output(tmp_path, discovery)
    artifact, frame, reference, universe, baseline = _crog_inputs()

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        return _fake_report(
            Path(kwargs["output_dir"]),
            status="incomplete",
            eligible_count=1,
            with_comparison=False,
        )

    monkeypatch.setattr(matrix_module, "run_crog_existing_comparisons", fake_run)
    with pytest.raises(MatrixError, match="failed independent comparison"):
        _r10_post_lock_comparison(
            output,
            {"matrix_profile": "unit_test"},
            artifact,
            frame,
            reference,
            universe,
            baseline,
        )
    assert (
        output
        / "metrics/r10_existing/crog_full_post_filter/comparison_report.json"
    ).is_file()


def test_post_lock_r10_accepts_audited_zero_eligible_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery = _discovery(tmp_path / "legacy", eligible=False)
    output = _prepare_output(tmp_path, discovery)
    artifact, frame, reference, universe, baseline = _crog_inputs()

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        return _fake_report(
            Path(kwargs["output_dir"]),
            status="complete_no_eligible",
            eligible_count=0,
            with_comparison=False,
        )

    monkeypatch.setattr(matrix_module, "run_crog_existing_comparisons", fake_run)
    paths, summary, rows = _r10_post_lock_comparison(
        output,
        {"matrix_profile": "unit_test"},
        artifact,
        frame,
        reference,
        universe,
        baseline,
    )

    assert summary["status"] == "complete_no_eligible"
    assert summary["eligible_count"] == summary["comparison_count"] == 0
    assert rows == []
    assert any(path.name == "comparison_report.json" for path in paths)


def test_cached_discovery_cannot_mark_method_eligible_without_all_checks(
    tmp_path: Path,
) -> None:
    discovery = _discovery(tmp_path / "legacy", eligible=True)
    del discovery["methods"][0]["checks"]["complete_oof"]

    with pytest.raises(MatrixError, match="checks are incomplete"):
        _validate_r10_discovery_report(discovery)
