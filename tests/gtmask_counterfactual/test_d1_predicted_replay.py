from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gtmask_counterfactual.d1_predicted_replay import (
    D1PredictedReplayError,
    build_d1_predicted_replay_manifest,
    validate_d1_predicted_replay_manifest,
)
from gtmask_counterfactual.d1_source_view import build_d1_source_view
from gtmask_counterfactual.io import (
    artifact_record,
    atomic_parquet,
    canonical_sha256,
    sha256_file,
)


def _json(path: Path, value: dict[str, object], *, self_hash: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    if self_hash:
        payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _inventory(path: Path, root: Path) -> dict[str, object]:
    return {
        **artifact_record(path),
        "relative_path": str(path.resolve().relative_to(root.resolve())),
    }


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "candidate_id": "g0001",
                "source_candidate_index": 1,
                "native_rank": 1,
                "native_score": 0.9,
                "cx_px": 20.0,
                "cy_px": 30.0,
                "center_depth_m": 0.8,
                "theta_deg": 10.0,
                "width_px": 40.0,
                "height_px": 20.0,
            },
            {
                "sample_id": "s1",
                "candidate_id": "g0000",
                "source_candidate_index": 0,
                "native_rank": 2,
                "native_score": 0.8,
                "cx_px": 21.0,
                "cy_px": 31.0,
                "center_depth_m": 0.81,
                "theta_deg": 20.0,
                "width_px": 42.0,
                "height_px": 20.0,
            },
        ]
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    run = tmp_path / "counterfactual"
    d1 = tmp_path / "d1"
    paired = tmp_path / "paired.parquet"
    atomic_parquet(pd.DataFrame({"sample_id": ["s1", "s2"]}), paired)
    pools = {}
    rows = _candidate_rows()
    for pool in ("top5", "top10", "allnms"):
        pools[pool] = atomic_parquet(
            rows, d1 / f"02_candidates/d1_{pool}_candidates.parquet"
        )
    candidate_manifest = _json(
        d1 / "02_candidates/test_manifest.json",
        {
            "status": "COMPLETE",
            "artifacts": {
                pool: artifact_record(path) for pool, path in pools.items()
            },
            "configuration": {"paired_manifest": artifact_record(paired)},
        },
        self_hash=True,
    )
    _json(
        d1 / "FINAL_RUN_LOCK.json",
        {
            "status": "COMPLETE",
            "inventory": [
                _inventory(candidate_manifest, d1),
                *[_inventory(path, d1) for path in pools.values()],
            ],
        },
    )
    derived_source = _json(tmp_path / "derived-source.json", {"value": 1})
    derived = _json(
        run / "04_predicted_replay/derived.json",
        {
            "status": "PASS",
            "raw_test_ground_truth_rows_read": 0,
            "routes": {
                "d1": {
                    "oracle_top5": 1,
                    "oracle_top10": 1,
                    "oracle_all": 1,
                }
            },
            "source": artifact_record(derived_source),
        },
        self_hash=True,
    )
    synthetic_source = _json(tmp_path / "source.py", {"source": "synthetic"})
    source_view = build_d1_source_view(
        run / "04_predicted_replay/d1/source_adapter",
        source_files={"scripts/source.py": synthetic_source},
    )

    candidates = run / "04_predicted_replay/d1/raw_candidates"
    frozen_candidates = tmp_path / "frozen/candidates/hierfilm"
    summary_text = (
        "sample_id,query,requested_candidate_count,raw_candidate_count,"
        "mask_validated_count,post_nms_count,failure_reason,status,"
        "question_index,scene_id\n"
        "s1,one,256,2,2,2,,success_nonempty,0,scene-a\n"
        "s2,two,256,0,0,0,empty,success_empty,1,scene-b\n"
    )
    stage_rows = {
        "s1": [
            {"sample_id": "s1", "candidate_id": "g0001", "value": 1.0},
            {"sample_id": "s1", "candidate_id": "g0000", "value": 2.0},
        ],
        "s2": [],
    }

    def write_candidate_root(root: Path) -> None:
        root.mkdir(parents=True)
        root.joinpath("summary.csv").write_text(summary_text, encoding="utf-8")
        _json(root / "run_config.json", {"mode": "candidate-only"})
        for sample_id, records in stage_rows.items():
            sample_root = root / sample_id
            sample_root.mkdir()
            for name in (
                "raw_candidates.json",
                "mask_validated_candidates.json",
                "filtered_candidates.json",
            ):
                sample_root.joinpath(name).write_text(
                    json.dumps(records, sort_keys=True) + "\n", encoding="utf-8"
                )
            _json(
                sample_root / "candidates.json",
                {"metadata": {"sample_id": sample_id}, "candidates": records},
            )
            required = [
                "raw_candidates.json",
                "mask_validated_candidates.json",
                "filtered_candidates.json",
                "candidates.json",
            ]
            _json(
                sample_root / "_SUCCESS.json",
                {
                    "sample_id": sample_id,
                    "configuration_hash": "configuration-sha",
                    "config_file_sha256": "config-file-sha",
                    "seed": 42,
                    "sampler_version": "1.3.0",
                    "sampler_release": "v1.3.0",
                    "sampler_commit": "499a609",
                    "sampler_class": "AntipodalDepthImageGraspSampler",
                    "status": "success_nonempty" if records else "success_empty",
                    "candidate_counts": {
                        "requested": 256,
                        "raw": len(records),
                        "mask_validated": len(records),
                        "post_nms": len(records),
                        "top_k": len(records),
                    },
                    "required_files": required,
                    "required_file_hashes": {
                        name: sha256_file(sample_root / name) for name in required
                    },
                },
            )

    write_candidate_root(candidates)
    write_candidate_root(frozen_candidates)
    _json(tmp_path / "frozen/run_manifest.json", {"status": "COMPLETE"})
    _json(tmp_path / "frozen/final_output_manifest.json", {"status": "COMPLETE"})

    scored = run / "04_predicted_replay/d1/scored"
    scored.mkdir(parents=True)
    sample = scored / "s1"
    sample.mkdir()
    payload = {
        "metadata": {"sample_id": "s1", "scoring_status": "scored_nonempty"},
        "candidates": [
            {
                "sample_id": "s1",
                "candidate_id": "g0001",
                "source_candidate_index": 1,
                "gqcnn_rank": 1,
                "gqcnn_q_value": 0.9,
                "center_u_px": 20.0,
                "center_v_px": 30.0,
                "center_depth_m": 0.8,
                "angle_deg": 10.0,
                "width_px": 40.0,
                "height_px": 20.0,
            },
            {
                "sample_id": "s1",
                "candidate_id": "g0000",
                "source_candidate_index": 0,
                "gqcnn_rank": 2,
                "gqcnn_q_value": 0.8,
                "center_u_px": 21.0,
                "center_v_px": 31.0,
                "center_depth_m": 0.81,
                "angle_deg": 20.0,
                "width_px": 42.0,
                "height_px": 20.0,
            },
        ],
    }
    _json(sample / "gqcnn_scored_candidates.json", payload)
    for name, data in {
        "gqcnn_scored_candidates.npz": b"npz",
        "gqcnn_scored_candidates.csv": b"csv",
        "gqcnn_top1.json": b"{}\n",
        "gqcnn_top5.json": b"{}\n",
        "scoring_metadata.json": b"{}\n",
    }.items():
        sample.joinpath(name).write_bytes(data)
    required = sorted(
        [
            "gqcnn_scored_candidates.json",
            "gqcnn_scored_candidates.npz",
            "gqcnn_scored_candidates.csv",
            "gqcnn_top1.json",
            "gqcnn_top5.json",
            "scoring_metadata.json",
        ]
    )
    common = {
        "scoring_status": "scored_nonempty",
        "source_candidate_count": 2,
        "gqcnn_scored_count": 2,
        "top1_candidate_id": "g0001",
        "source_candidate_sha256": "candidate-sha",
        "model_config_hash": "model-sha",
    }
    _json(
        sample / "_SCORING_COMPLETE.json",
        {
            "sample_id": "s1",
            **common,
            "required_files": required,
            "required_file_hashes": {
                name: sha256_file(sample / name) for name in required
            },
        },
    )
    empty = scored / "s2"
    empty.mkdir()
    empty.joinpath("scoring_metadata.json").write_text("{}\n", encoding="utf-8")
    empty_common = {
        "scoring_status": "skipped_valid_empty",
        "source_candidate_count": 0,
        "gqcnn_scored_count": 0,
        "top1_candidate_id": None,
        "source_candidate_sha256": "empty-sha",
        "model_config_hash": "model-sha",
    }
    _json(
        empty / "_SCORING_COMPLETE.json",
        {
            "sample_id": "s2",
            **empty_common,
            "required_files": ["scoring_metadata.json"],
            "required_file_hashes": {
                "scoring_metadata.json": sha256_file(
                    empty / "scoring_metadata.json"
                )
            },
        },
    )
    _json(scored / "run_config.json", {"runtime": "synthetic"})
    _json(
        scored / "progress.json",
        {
            "total_samples": 2,
            "terminal_samples": 2,
            "completed_nonempty_samples": 1,
            "skipped_empty_samples": 1,
            "failed_samples": 0,
            "scored_candidates": 2,
            "remaining_candidates": 0,
        },
    )
    _json(
        scored / "run_statistics.json",
        {
            "total_samples": 2,
            "terminal_samples": 2,
            "scored_nonempty_samples": 1,
            "skipped_valid_empty_samples": 1,
            "failed_samples": 0,
            "corrupt_committed_samples": 0,
            "expected_candidates": 2,
            "scored_candidates": 2,
            "finite_q_values": 2,
            "invalid_q_values": 0,
        },
    )
    scored.joinpath("summary.csv").write_text(
        "sample_id,scoring_status,scored_candidate_count\n"
        "s1,scored_nonempty,2\n"
        "s2,skipped_valid_empty,0\n",
        encoding="utf-8",
    )
    scored.joinpath("scoring_manifest.jsonl").write_text(
        "\n".join(
            json.dumps({"sample_id": sample_id, **values}, sort_keys=True)
            for sample_id, values in (("s1", common), ("s2", empty_common))
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "run": run,
        "d1": d1,
        "candidate_root": candidates,
        "frozen_candidate_root": frozen_candidates,
        "scored_root": scored,
        "derived": derived,
        "source_view": source_view,
        "payload": sample / "gqcnn_scored_candidates.json",
        "marker": sample / "_SCORING_COMPLETE.json",
        "candidate_stage": candidates / "s1/raw_candidates.json",
    }


def test_d1_predicted_replay_rebuilds_every_candidate_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = build_d1_predicted_replay_manifest(
        run_dir=fixture["run"],
        d1_run=fixture["d1"],
        candidate_root=fixture["candidate_root"],
        frozen_candidate_root=fixture["frozen_candidate_root"],
        scored_root=fixture["scored_root"],
        derived_reconciliation=fixture["derived"],
        source_view_manifest=fixture["source_view"],
        expected_samples=2,
        expected_candidates=2,
        expected_no_output=1,
    )
    value = validate_d1_predicted_replay_manifest(
        manifest,
        expected_samples=2,
        expected_candidates=2,
        expected_no_output=1,
    )
    assert value["status"] == "PASS"
    assert value["oracle_metrics"] == {
        "oracle_top5": 1,
        "oracle_top10": 1,
        "oracle_all": 1,
    }
    candidate_frame = pd.read_parquet(value["candidates"]["path"])
    sample_frame = pd.read_parquet(value["per_sample"]["path"])
    assert candidate_frame[["route", "branch"]].drop_duplicates().to_dict(
        "records"
    ) == [{"route": "D1", "branch": "predicted"}]
    assert sample_frame.set_index("sample_id")["candidate_count"].to_dict() == {
        "s1": 2,
        "s2": 0,
    }

    payload = json.loads(fixture["payload"].read_text(encoding="utf-8"))
    payload["candidates"][0]["gqcnn_q_value"] = 0.85
    _json(fixture["payload"], payload)
    marker = json.loads(fixture["marker"].read_text(encoding="utf-8"))
    marker["required_file_hashes"]["gqcnn_scored_candidates.json"] = sha256_file(
        fixture["payload"]
    )
    _json(fixture["marker"], marker)
    with pytest.raises(D1PredictedReplayError, match="numeric values differ"):
        validate_d1_predicted_replay_manifest(
            manifest,
            expected_samples=2,
            expected_candidates=2,
            expected_no_output=1,
        )


def test_d1_predicted_replay_rejects_rehashed_raw_stage_tamper(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = build_d1_predicted_replay_manifest(
        run_dir=fixture["run"],
        d1_run=fixture["d1"],
        candidate_root=fixture["candidate_root"],
        frozen_candidate_root=fixture["frozen_candidate_root"],
        scored_root=fixture["scored_root"],
        derived_reconciliation=fixture["derived"],
        source_view_manifest=fixture["source_view"],
        expected_samples=2,
        expected_candidates=2,
        expected_no_output=1,
    )
    raw = json.loads(fixture["candidate_stage"].read_text(encoding="utf-8"))
    raw[0]["value"] = 999.0
    fixture["candidate_stage"].write_text(
        json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker_path = fixture["candidate_root"] / "s1/_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["required_file_hashes"]["raw_candidates.json"] = sha256_file(
        fixture["candidate_stage"]
    )
    _json(marker_path, marker)
    with pytest.raises(D1PredictedReplayError, match="raw candidate semantics differ"):
        validate_d1_predicted_replay_manifest(
            manifest,
            expected_samples=2,
            expected_candidates=2,
            expected_no_output=1,
        )
