from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ruamel.yaml import YAML

from src.grasping.dexnet_run_reliability import canonical_json_hash
from src.grasping.grasp_serialization import candidate_to_record
from src.grasping.reranking_v1.identity import sha256_file
from tools.modular_reranking.generate_compact_dexnet_candidates import (
    CANDIDATE_STAGE_SCHEMA,
    SAMPLE_SEED_DERIVATION,
    SAMPLE_SEED_MODE,
    SEED_NAMESPACE,
    STAGE_FILES,
    _candidate_stage_row,
    _candidate_stage_schema_sha256,
    _stage_sidecar_path,
    _validate_tmp_scope,
    _write_candidate_stage_parquets,
    _write_stage_sidecar,
    derive_sample_seed,
)
from tools.modular_reranking.merge_compact_candidate_shards import (
    merge_stage_parquets,
    preflight_shard_artifacts,
    validate_tmp_scope as validate_merge_tmp_scope,
)


RETAINED_FIRST_TEN_SEEDS = {
    "q0000000_b32eb3299dcd3ae9": 2530226156,
    "q0000001_a9a5f9b502546016": 2110806117,
    "q0000002_65b99b4d1aaf2b7b": 1136439867,
    "q0000003_c9f21176e1f0d767": 462243345,
    "q0000004_23c60d130c4f6a9e": 2271826839,
    "q0000006_479d6565d0c9dc86": 1584673221,
    "q0000012_3c71c3e0ca6b64a8": 655264967,
    "q0000014_365bf2150cb2b2fb": 2707346342,
    "q0000020_70edb52adfcfaa63": 601306592,
    "q0000021_ab3537b86eafece1": 2966422137,
}


def _candidate(sample_id: str, candidate_id: str, seed: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "query": "grasp it",
        "center_u_px": 320.0,
        "center_v_px": 240.0,
        "center_uv": [320.0, 240.0],
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.0, 0.0, 1.0],
        "angle_rad": 0.25,
        "angle_deg": float(np.degrees(0.25)),
        "width_m": 0.05,
        "width_px": 28.0,
        "endpoint_1_uv": [306.0, 240.0],
        "endpoint_2_uv": [334.0, 240.0],
        "endpoints_uv": [[306.0, 240.0], [334.0, 240.0]],
        "contact_points_uv": [[307.0, 240.0], [333.0, 240.0]],
        "contact_normals": [[-1.0, 0.0], [1.0, 0.0]],
        "sampler_rank": int(candidate_id[1:]) + 1,
        "seed": seed,
        "rejection_reason": None,
        "rejection_reasons": [],
        "T_camera_grasp_fixed_approach": np.eye(4).tolist(),
    }


def test_stable_sha256_seed_matches_retained_repeatedfilm_first_ten() -> None:
    assert {
        sample_id: derive_sample_seed(sample_id, base_seed=42)
        for sample_id in RETAINED_FIRST_TEN_SEEDS
    } == RETAINED_FIRST_TEN_SEEDS


def test_resolved_protocol_config_matches_retained_configuration_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    config = YAML(typ="safe").load(
        (root / "configs/dexnet_candidates_formal_no_refinement.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["generation"].update(
        {
            "sample_seed_mode": SAMPLE_SEED_MODE,
            "seed_namespace": SEED_NAMESPACE,
            "sample_seed_derivation": SAMPLE_SEED_DERIVATION,
        }
    )
    assert canonical_json_hash(config) == (
        "2308112e3caf486e66118f9a4ee2950c4aaaf562420a2560fb2fde41a094462a"
    )


def test_tmp_scope_is_confined_to_same_active_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "tmp").mkdir(parents=True)
    (run / ".RUN_ACTIVE").write_text("", encoding="utf-8")
    output, temporary = _validate_tmp_scope(
        run / "compact" / "candidates", run / "tmp" / "dex"
    )
    assert output == (run / "compact" / "candidates").resolve()
    assert temporary == (run / "tmp" / "dex").resolve()
    merged, merge_tmp = validate_merge_tmp_scope(
        run / "tmp" / "candidates" / "merged", run / "tmp"
    )
    assert merged == (run / "tmp" / "candidates" / "merged").resolve()
    assert merge_tmp == (run / "tmp").resolve()
    with pytest.raises(ValueError, match="below tmp-root"):
        validate_merge_tmp_scope(
            run / "compact" / "candidates", run / "tmp"
        )
    other = tmp_path / "other"
    (other / "tmp").mkdir(parents=True)
    (other / ".RUN_ACTIVE").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="different active runs"):
        _validate_tmp_scope(run / "compact", other / "tmp")


def test_three_stage_parquets_preserve_identity_geometry_and_no_gt(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    output = run / "compact"
    temporary = run / "tmp"
    sidecars = temporary / "sidecars"
    output.mkdir(parents=True)
    temporary.mkdir(parents=True)
    sample_id = "q0000000_b32eb3299dcd3ae9"
    seed = derive_sample_seed(sample_id, base_seed=42)
    first = _candidate(sample_id, "g0000", seed)
    second = _candidate(sample_id, "g0001", seed)
    rejected = dict(second)
    rejected["rejection_reason"] = "center_outside_target_mask"
    rejected["rejection_reasons"] = ["center_outside_target_mask"]
    result = SimpleNamespace(
        raw_candidates=[first, rejected],
        mask_validated_candidates=[first],
        deduplicated_candidates=[first],
    )
    sample_row = {
        "sample_index": 0,
        "sample_id": sample_id,
        "question_index": 0,
        "scene_id": "scene.png",
    }
    protocol_hash = "a" * 64
    _write_stage_sidecar(
        _stage_sidecar_path(sidecars, sample_id),
        sample_row=sample_row,
        sample_seed=seed,
        protocol_identity_sha256=protocol_hash,
        result=result,
    )
    artifacts = _write_candidate_stage_parquets(
        output_root=output,
        tmp_root=temporary,
        sidecar_root=sidecars,
        selected=[sample_row],
        base_seed=42,
        protocol_identity_sha256=protocol_hash,
    )

    assert _candidate_stage_schema_sha256() == (
        "0bb121785dc47dcfb004dc8f96a108751d7a61328dc06e923c831e89893383c9"
    )
    assert {stage: artifacts[stage]["rows"] for stage in STAGE_FILES} == {
        "raw": 2,
        "mask_validated": 1,
        "nms": 1,
    }
    forbidden = {
        "candidate_success",
        "best_gt_id",
        "rectangle_iou",
        "angle_difference_deg",
        "joint_success",
    }
    assert forbidden.isdisjoint(CANDIDATE_STAGE_SCHEMA.names)
    for stage, filename in STAGE_FILES.items():
        table = pq.read_table(output / filename)
        assert table.schema == CANDIDATE_STAGE_SCHEMA
        frame = table.to_pandas()
        assert not frame.duplicated(["sample_id", "candidate_id"]).any()
        assert set(frame["stage"]) == {stage}
        candidate = json.loads(frame.iloc[0]["candidate_json"])
        assert candidate["contact_points_uv"] == [
            [307.0, 240.0],
            [333.0, 240.0],
        ]
        assert candidate["contact_normals"] == [[-1.0, 0.0], [1.0, 0.0]]
        assert candidate["T_camera_grasp_fixed_approach"] == np.eye(4).tolist()


def test_stage_parquet_shard_merge_preserves_all_rows(tmp_path: Path) -> None:
    shards = [tmp_path / "shard0", tmp_path / "shard1"]
    configs = []
    for index, root in enumerate(shards):
        root.mkdir()
        sample_id = f"q{index:07d}_synthetic"
        seed = derive_sample_seed(sample_id, base_seed=42)
        sample = {
            "sample_index": index,
            "sample_id": sample_id,
            "question_index": index,
            "scene_id": f"scene{index}.png",
        }
        record = candidate_to_record(_candidate(sample_id, "g0000", seed))
        artifacts = {}
        for stage, filename in STAGE_FILES.items():
            table = pa.Table.from_pylist(
                [_candidate_stage_row(stage, sample, record)],
                schema=CANDIDATE_STAGE_SCHEMA,
            )
            path = root / filename
            pq.write_table(table, path, compression="zstd")
            artifacts[stage] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": 1,
            }
        configs.append(
            {
                "status": "COMPLETED",
                "counts": {
                    "execution_failures": 0,
                    "raw_candidates": 1,
                    "mask_validated_candidates": 1,
                    "nms_candidates": 1,
                },
                "candidate_stage_artifacts": artifacts,
            }
        )
    output = tmp_path / "merged"
    temporary = tmp_path / "tmp"
    output.mkdir()
    temporary.mkdir()
    preflight_shard_artifacts(shards, configs)
    artifacts = merge_stage_parquets(
        shards, configs, output, temporary
    )
    assert {stage: item["rows"] for stage, item in artifacts.items()} == {
        "raw": 2,
        "mask_validated": 2,
        "nms": 2,
    }
    for filename in STAGE_FILES.values():
        frame = pq.read_table(output / filename).to_pandas()
        assert frame["sample_id"].tolist() == [
            "q0000000_synthetic",
            "q0000001_synthetic",
        ]

    configs[1]["status"] = "IN_PROGRESS"
    with pytest.raises(ValueError, match="not COMPLETED"):
        preflight_shard_artifacts(shards, configs)
