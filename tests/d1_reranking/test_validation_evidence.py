from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from d1_reranking import validation_evidence
from unified_reranking.hashing import canonical_sha256, sha256_file


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _content(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["s0", "s1"]
    common = root / "validation_evidence_common.bin"
    common.write_bytes(b"source")
    decisions = _parquet(
        root / "inputs/decisions.parquet", pd.DataFrame({"sample_id": ids})
    )
    gate_decisions = _parquet(
        root / "inputs/gate.parquet",
        pd.DataFrame(
            {
                "sample_id": ids,
                "native_correct": [False, True],
                "challenger_correct": [True, False],
                "switch": [True, False],
            }
        ),
    )
    operating = _csv(root / "inputs/gate.csv", pd.DataFrame({"gated_j_at_1": [1.0]}))
    comparison = _csv(
        root / "inputs/k.csv",
        pd.DataFrame(
            [["top5", "top5", "T2", 5, "primary", "R3", 0.5, 0.6, 0.7, 0.4, 0.5, 0.6]],
            columns=validation_evidence.TABLE_SCHEMAS["d1_k_sensitivity.csv"],
        ),
    )
    evidence = _csv(
        root / "inputs/evidence.csv",
        pd.DataFrame(
            [["T2", "R3", "GO", 2, 0.5, 0.1, "GO"]],
            columns=validation_evidence.TABLE_SCHEMAS["d1_evidence_tracks.csv"],
        ),
    )
    feature = _csv(
        root / "inputs/feature.csv",
        pd.DataFrame(
            [["full", "family", "all", "GO", 3, "full", "", 0.5, 0.0, 0.0, 0.0, "{}"]],
            columns=validation_evidence.TABLE_SCHEMAS["d1_feature_ablation.csv"],
        ),
    )
    metrics = {"j_at_1": 0.5, "mrr_at_5": 0.6, "ndcg_at_5": 0.7}
    _content(
        root / validation_evidence.FIXED_SOURCES["r0_r1"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "r0_validation_metrics": metrics,
            "r1_validation_metrics": {**metrics, "j_at_1": 0.6},
            "artifacts": {"r0_validation_decisions": _record(decisions)},
        },
    )
    winners = {
        f"R{i}": {
            "validation_j_at_1": 0.5 + i / 100,
            "validation_mrr_at_5": 0.6,
            "validation_ndcg_at_5": 0.7,
        }
        for i in range(2, 7)
    }
    _content(
        root / validation_evidence.FIXED_SOURCES["primary"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "selected_method": "R3",
            "method_winners": winners,
            "artifacts": {"selected_validation_decisions": _record(decisions)},
        },
    )
    _content(
        root / validation_evidence.FIXED_SOURCES["gate"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "validation_metrics": {"gated_j_at_1": 0.65},
            "artifacts": {
                "validation_decisions": _record(gate_decisions),
                "gate_operating_point_table": _record(operating),
            },
        },
    )
    _content(
        root / validation_evidence.FIXED_SOURCES["k"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {"comparison_table": _record(comparison)},
        },
    )
    _content(
        root / validation_evidence.FIXED_SOURCES["ablation"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {
                "evidence_track_table": _record(evidence),
                "feature_ablation_table": _record(feature),
            },
        },
    )
    _content(
        root / validation_evidence.FIXED_SOURCES["ablation_adapter"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "scientific_values_changed": False,
            "artifacts": {"selection": _record(root / validation_evidence.FIXED_SOURCES["ablation"])},
        },
    )
    sample_rows = pd.DataFrame(
        {
            "sample_id": ids,
            "scene_id": ["a", "b"],
            "crog_correct": [True, False],
            "g1_correct": [False, False],
            "c1_correct": [False, True],
            "d1_correct": [True, True],
            "three_route_router_correct": [True, True],
            "four_route_decision": ["CROG", "D1"],
            "existing_top15_oracle": [True, True],
        }
    )
    union_rows = []
    for index, sample_id in enumerate(ids):
        identity, geometry = f"i{index}", f"g{index}"
        member = canonical_sha256(
            {
                "sample_id": sample_id,
                "source_route": "D1",
                "source_candidate_id": f"c{index}",
                "candidate_identity_sha256": identity,
                "candidate_geometry_sha256": geometry,
            }
        )
        union_rows.append(
            {
                "sample_id": sample_id,
                "candidate_id": f"D1:c{index}",
                "native_rank": 1,
                "source_route": "D1",
                "source_candidate_id": f"c{index}",
                "route_native_rank": 1,
                "candidate_identity_sha256": identity,
                "candidate_geometry_sha256": geometry,
                "route_member_sha256": member,
                "union_route_crog": 0,
                "union_route_g1": 0,
                "union_route_c1": 0,
                "union_route_d1": 1,
                "candidate_success": True,
            }
        )
    union = pd.DataFrame(union_rows)
    p12_samples = _parquet(root / "inputs/p12_samples.parquet", sample_rows)
    p12_union = _parquet(root / "inputs/p12_union.parquet", union)
    from d1_reranking.four_route import validation_router_union_summary

    p12_table = _csv(
        root / "inputs/p12.csv", validation_router_union_summary(sample_rows, union)
    )
    _content(
        root / validation_evidence.FIXED_SOURCES["p12"],
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {
                "router_decisions": _record(p12_samples),
                "top20_union": _record(p12_union),
                "prelock_table": _record(p12_table),
            },
        },
    )
    registry = _csv(
        root / "closure/D1_RUN_REGISTRY.csv",
        pd.DataFrame(
            [
                [
                    "A",
                    "run",
                    True,
                    2,
                    2,
                    0,
                    True,
                    False,
                    "canonical",
                    "SELECTED",
                    "VERIFIED",
                    "ok",
                    None,
                ]
            ],
            columns=validation_evidence.TABLE_SCHEMAS["d1_provenance_comparison.csv"],
        ),
    )
    closure_path = _content(
        root / "closure/manifest.json",
        {"status": "PASS", "artifacts": {"run_registry": _record(registry)}},
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        validation_evidence,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )


def test_validation_evidence_is_fixed_and_semantically_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    result = validation_evidence.assemble_validation_evidence_tables(tmp_path)
    assert set(result["artifacts"]["tables"]) == set(validation_evidence.TABLE_NAMES)
    r0 = pd.read_csv(
        tmp_path / validation_evidence.OUTPUT_ROOT / "d1_r0_r7_validation.csv"
    )
    assert list(r0["method"]) == [f"R{i}" for i in range(8)]
    assert set(r0["validation_sample_count"]) == {2}
    transitions = pd.read_csv(
        tmp_path / validation_evidence.OUTPUT_ROOT / "d1_gate_transitions.csv"
    )
    assert set(transitions["transition"]) == {
        "recovered",
        "harmful",
        "missed_recoverable",
        "prevented_harmful",
        "wrong_to_wrong",
        "correct_to_correct",
    }
    assert (
        validation_evidence.assemble_validation_evidence_tables(tmp_path, resume=True)
        == result
    )
    with pytest.raises(FileExistsError):
        validation_evidence.assemble_validation_evidence_tables(tmp_path)
    (tmp_path / "inputs/k.csv").write_text("source drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validation_evidence.load_validation_evidence_tables(tmp_path)
