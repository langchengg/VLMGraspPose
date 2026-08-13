from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from d1_reranking import prelock
from unified_reranking.hashing import canonical_sha256


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _write_content(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return _write_json(path, payload)


def _preformal_run(root: Path) -> None:
    _write_json(
        root / "manifest.json",
        {
            "protocol": "fair-d1-reranking-retrospective-extension-v1",
            "formal_test_execution_count": 0,
        },
    )
    _write_json(
        root / "pipeline_status.json",
        {
            "formal_test_executed": False,
            "test_candidate_labels_read": False,
        },
    )


def _mandatory_tables(root: Path) -> None:
    tables = {
        "r0_r7_full": pd.DataFrame({"method": [f"R{i}" for i in range(8)]}),
        "selected_primary_ungated": pd.DataFrame({"method": ["R3", "R5", "R6"]}),
        "gate_operating_point": pd.DataFrame({"gated_j_at_1": [0.5]}),
        "evidence_track_ablation": pd.DataFrame(
            {"track": ["T1_native_available", "T2_matched_common", "T3_route_rich"]}
        ),
        "k_comparison": pd.DataFrame({"pool": ["top5", "top10", "allnms"]}),
        "feature_ablation": pd.DataFrame({"ablation": ["drop_q"]}),
        "four_route_validation": pd.DataFrame({"route": ["C1", "G1", "CROG", "D1"]}),
    }
    for name, relative in prelock.MANDATORY_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tables[name].to_csv(path, index=False)


def test_artifact_record_accepts_richer_observed_byte_metadata() -> None:
    expected = {"path": "/frozen/a", "sha256": "a" * 64}
    observed = {**expected, "bytes": 123}
    prelock._same_record(observed, expected, name="source closure artifact")

    with pytest.raises(RuntimeError, match="byte count differs"):
        prelock._same_record(
            observed,
            {**expected, "bytes": 124},
            name="exact artifact",
        )


def test_mandatory_tables_are_fail_closed_and_cover_required_axes(
    tmp_path: Path,
) -> None:
    _mandatory_tables(tmp_path)
    records = prelock._validate_mandatory_tables(tmp_path)
    assert set(records) == set(prelock.MANDATORY_TABLES)

    (tmp_path / prelock.MANDATORY_TABLES["k_comparison"]).unlink()
    with pytest.raises(FileNotFoundError, match="k_comparison"):
        prelock._validate_mandatory_tables(tmp_path)


def test_mandatory_tables_reject_prelock_test_metrics(tmp_path: Path) -> None:
    _mandatory_tables(tmp_path)
    path = tmp_path / prelock.MANDATORY_TABLES["feature_ablation"]
    pd.DataFrame({"ablation": ["drop_q"], "test_j_at_1": [0.7]}).to_csv(
        path, index=False
    )
    with pytest.raises(PermissionError, match="Test-derived"):
        prelock._validate_mandatory_tables(tmp_path)


def test_access_log_requires_zero_candidate_label_access(tmp_path: Path) -> None:
    path = tmp_path / "09_formal_test" / "test_access.log"
    path.parent.mkdir(parents=True)
    events = []
    for index, stage in enumerate(sorted(prelock.REQUIRED_LABEL_FREE_TEST_STAGES)):
        output = tmp_path / "outputs" / f"{index}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"stage": stage}), encoding="utf-8")
        events.append(
            {
                "event": "prelock_label_free_test_stage",
                "candidate_labels_opened_as_table": False,
                "stage": stage,
                "output_manifest": prelock._record(output),
            }
        )
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    result = prelock._validate_access_log(tmp_path)
    assert result["candidate_label_access_count"] == 0

    # An append-only rerun may supersede an earlier hash at the same stable
    # output path; the latest event is the active source binding.
    first_output = Path(events[0]["output_manifest"]["path"])
    old_record = dict(events[0]["output_manifest"])
    first_output.write_text(json.dumps({"stage": "rerun"}), encoding="utf-8")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "prelock_label_free_test_stage",
                    "candidate_labels_opened_as_table": False,
                    "stage": events[0]["stage"],
                    "output_manifest": prelock._record(first_output),
                    "supersedes": old_record,
                }
            )
            + "\n"
        )
    prelock._validate_access_log(tmp_path)

    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "candidate_test_label_access_authorized",
                    "candidate_labels_opened_as_table": True,
                }
            )
            + "\n"
        )
    with pytest.raises(PermissionError, match="not zero"):
        prelock._validate_access_log(tmp_path)


def _candidate_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    paired_path = root / "01_manifests" / "d1_paired_train.parquet"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired = pd.DataFrame(
        {"sample_id": ["sample-a", "sample-b"], "scene_id": ["s1", "s2"]}
    )
    paired.to_parquet(paired_path, index=False)
    rows = []
    for rank in range(1, 12):
        rows.append(
            {
                "sample_id": "sample-a",
                "candidate_id": f"candidate-{rank:02d}",
                "native_rank": rank,
                "native_score": float(12 - rank),
                "candidate_identity_sha256": f"identity-{rank}",
                "candidate_geometry_sha256": f"geometry-{rank}",
            }
        )
    allnms = pd.DataFrame(rows)
    pools = {
        "top5": allnms.loc[allnms["native_rank"] <= 5].reset_index(drop=True),
        "top10": allnms.loc[allnms["native_rank"] <= 10].reset_index(drop=True),
        "allnms": allnms,
    }
    output = root / "02_candidates" / "train"
    output.mkdir(parents=True)
    artifacts: dict[str, object] = {}
    summaries: dict[str, object] = {}
    denominator_ids = paired["sample_id"].tolist()
    for pool, frame in pools.items():
        path = output / f"d1_{pool}_candidates.parquet"
        frame.to_parquet(path, index=False)
        artifacts[pool] = prelock._record(path)
        counts = (
            frame.groupby("sample_id").size().reindex(denominator_ids, fill_value=0)
        )
        summaries[pool] = {
            "rows": len(frame),
            "denominator_samples": 2,
            "samples_with_candidates": 1,
            "no_output_samples": 1,
            "maximum_candidates": int(counts.max()),
        }
    hashes_path = output / "candidate_hashes.parquet"
    prelock.pool_hash_rows(pools, paired, split="train").to_parquet(
        hashes_path, index=False
    )
    artifacts["candidate_hashes"] = prelock._record(hashes_path)
    closure_path = root / "00_audit" / "closure.json"
    _write_json(closure_path, {"status": "PASS"})
    configuration = {
        "route": "D1",
        "split": "train",
        "pool_limits": prelock.POOL_LIMITS,
        "paired_manifest": prelock._record(paired_path),
        "source_contract": {
            "snapshot": "A",
            "source_closure": prelock._record(closure_path),
            "source_paths_from_closure_only": True,
            "candidate_test_labels_read": False,
        },
    }
    manifest_path = output / "manifest.json"
    _write_content(
        manifest_path,
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "configuration": configuration,
            "execution_contract": {},
            "artifacts": artifacts,
            "summaries": summaries,
        },
    )
    return closure_path, {"paired": paired, "hashes": hashes_path}


def test_candidate_validator_recomputes_membership_geometry_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closure_path, fixture = _candidate_fixture(tmp_path)
    # Canonical row semantics are already unit-tested by test_candidates.py;
    # this test isolates P13's cross-pool/hash replay.
    monkeypatch.setattr(
        prelock, "verify_canonical_candidate_frame", lambda *_args, **_kwargs: None
    )
    prelock._validate_candidate_split(
        tmp_path,
        split="train",
        paired=fixture["paired"],
        source_closure_record=prelock._record(closure_path),
        source_paired_record=prelock._record(
            tmp_path / "01_manifests" / "d1_paired_train.parquet"
        ),
    )

    hashes = pd.read_parquet(fixture["hashes"])
    hashes.loc[hashes["pool"].eq("top5"), "candidate_geometry_vector_sha256"] = "0" * 64
    hashes.to_parquet(fixture["hashes"], index=False)
    manifest_path = tmp_path / "02_candidates" / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["candidate_hashes"] = prelock._record(fixture["hashes"])
    manifest.pop("content_sha256")
    _write_content(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="membership/geometry/native-score hashes"):
        prelock._validate_candidate_split(
            tmp_path,
            split="train",
            paired=fixture["paired"],
            source_closure_record=prelock._record(closure_path),
            source_paired_record=prelock._record(
                tmp_path / "01_manifests" / "d1_paired_train.parquet"
            ),
        )


def test_assembly_writes_readiness_and_declarations_but_never_formal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _preformal_run(tmp_path)
    source = _write_json(tmp_path / "source.json", {"frozen": True})

    def validated(
        _root: Path, *, code_paths: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        del code_paths
        return {"synthetic_source": prelock._record(source)}, {
            "primary_selected_method": "R5",
            "r7_gate_decision": "GO",
            "candidate_test_label_access_count": 0,
        }

    monkeypatch.setattr(prelock, "_validate_all", validated)
    result = prelock.assemble_prelock_readiness(tmp_path)
    assert result["status"] == "PASS"
    assert (tmp_path / "08_lock" / "PRELOCK_READINESS.json").is_file()
    assert (tmp_path / "08_lock" / "PRIMARY_METHOD_DECLARATION.md").is_file()
    assert (tmp_path / "08_lock" / "SECONDARY_TRACK_DECLARATION.md").is_file()
    assert not (tmp_path / "08_lock" / "FORMAL_TEST_LOCK.json").exists()
    assert prelock.assemble_prelock_readiness(tmp_path, resume=True) == result

    source.write_text('{"frozen": false}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from current"):
        prelock.assemble_prelock_readiness(tmp_path, resume=True)


def test_assembly_failure_publishes_no_readiness_or_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _preformal_run(tmp_path)

    def denied(
        *_args: object, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        raise FileNotFoundError("mandatory K/ablation/four-route evidence missing")

    monkeypatch.setattr(prelock, "_validate_all", denied)
    with pytest.raises(FileNotFoundError, match="mandatory"):
        prelock.assemble_prelock_readiness(tmp_path)
    assert not (tmp_path / "08_lock" / "PRELOCK_READINESS.json").exists()
    assert not (tmp_path / "08_lock" / "PRIMARY_METHOD_DECLARATION.md").exists()
    assert not (tmp_path / "08_lock" / "SECONDARY_TRACK_DECLARATION.md").exists()
    assert not (tmp_path / "08_lock" / "FORMAL_TEST_LOCK.json").exists()
