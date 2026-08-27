from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

import gtmask_counterfactual.sample_covariates as module
from gtmask_counterfactual.io import artifact_record, atomic_json, atomic_parquet


def _png(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)
    return path


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    root = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_synthetic"
    protocol = root / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
    atomic_json(protocol, {"status": "LOCKED"})
    sample = atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": ["s0", "s1"],
                "query_type": ["name", "relation"],
                "scene_id": ["scene0", "scene1"],
                "frame_id": ["frame0", "frame1"],
            }
        ),
        root / "02_sample_manifest/counterfactual_manifest.parquet",
    )
    predicted = _png(
        root / "assets/pred.png", np.array([[255, 0], [255, 0]], dtype=np.uint8)
    )
    target = _png(
        root / "assets/gt.png", np.array([[255, 255], [0, 0]], dtype=np.uint8)
    )
    depth = _png(
        root / "assets/depth.png", np.array([[1, 0], [2, 3]], dtype=np.uint16)
    )
    registry = atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": ["s0", "s1"],
                "original_gt_mask_path": [str(target), str(target)],
                "original_gt_mask_sha256": [
                    artifact_record(target)["sha256"],
                    artifact_record(target)["sha256"],
                ],
                "mapping_status": ["PASS", "unresolved"],
            }
        ),
        root / "03_gt_mask_registry/gt_mask_registry.parquet",
    )
    visual = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "predicted_mask_path": [str(predicted), str(predicted)],
            "predicted_mask_sha256": [
                artifact_record(predicted)["sha256"],
                artifact_record(predicted)["sha256"],
            ],
            "source_depth_path": [str(depth), str(depth)],
            "source_depth_sha256": [
                artifact_record(depth)["sha256"],
                artifact_record(depth)["sha256"],
            ],
        }
    )
    authority = {
        "protocol_lock": artifact_record(protocol),
        "execution_mode": "prospective_locked_execution",
        "gt_candidate_generation_authorized": True,
        "sample_manifest": artifact_record(sample),
        "gt_mask_registry": artifact_record(registry),
    }
    atomic_json(
        root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json",
        {
            "status": "RUNNING",
            "execution_count": 1,
            "protocol_lock_file_sha256": authority["protocol_lock"]["sha256"],
        },
    )
    atomic_json(
        root / "pipeline_status.json",
        {"status": "P3_PROTOCOL_LOCKED", "counterfactual_execution_count": 1},
    )
    monkeypatch.setattr(module, "validate_fresh_gate", lambda gate: None)
    monkeypatch.setattr(module, "load_execution_authority", lambda path: authority)
    monkeypatch.setattr(
        module,
        "validate_visual_asset_registry",
        lambda *args, **kwargs: ({"status": "COMPLETE"}, visual.copy()),
    )
    monkeypatch.setattr(module, "append_gt_access_log", lambda *args, **kwargs: root)
    return root, protocol, {"path": str(root / "visual.json"), "sha256": "0" * 64, "bytes": 0}


def test_sample_covariates_are_pixel_derived_and_resume_is_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, protocol, visual = _fixture(tmp_path, monkeypatch)
    manifest = module.write_sample_covariates(
        root,
        protocol_lock=protocol,
        visual_asset_manifest=visual,
        resource_gate={},
        expected_count=2,
    )
    _, frame = module.validate_sample_covariates(
        root, artifact_record(manifest), expected_count=2
    )
    first = frame.set_index("sample_id").loc["s0"]
    assert first["predicted_mask_iou"] == pytest.approx(1 / 3)
    assert first["target_area_fraction"] == 0.5
    assert first["mask_component_count"] == 1
    assert first["valid_depth_ratio"] == 0.5
    second = frame.set_index("sample_id").loc["s1"]
    assert second["predicted_mask_iou"] == 0.0

    frame_path = root / module.FRAME_RELATIVE_PATH
    frame_path.write_bytes(frame_path.read_bytes() + b"tamper")
    with pytest.raises(module.SampleCovariateError, match="artifact differs"):
        module.write_sample_covariates(
            root,
            protocol_lock=protocol,
            visual_asset_manifest=visual,
            resource_gate={},
            expected_count=2,
            resume=True,
        )


def test_sample_covariates_reject_before_any_pixel_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, protocol, visual = _fixture(tmp_path, monkeypatch)
    atomic_json(
        root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json",
        {"status": "ABSENT", "execution_count": 0},
    )
    monkeypatch.setattr(
        module,
        "_binary",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pixel opened")),
    )
    with pytest.raises(PermissionError, match="exactly-once claim"):
        module.write_sample_covariates(
            root,
            protocol_lock=protocol,
            visual_asset_manifest=visual,
            resource_gate={},
            expected_count=2,
        )


def test_retrospective_covariates_use_protocol_lock_without_execution_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, protocol, visual = _fixture(tmp_path, monkeypatch)
    (root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json").unlink()
    atomic_json(
        root / "pipeline_status.json",
        {
            "status": "P5B_G1_FULL_COMPLETE",
            "counterfactual_execution_count": 0,
        },
    )
    authority = {
        "protocol_lock": artifact_record(protocol),
        "execution_mode": "retrospective_verified_import",
        "gt_candidate_generation_authorized": False,
        "sample_manifest": artifact_record(
            root / "02_sample_manifest/counterfactual_manifest.parquet"
        ),
        "gt_mask_registry": artifact_record(
            root / "03_gt_mask_registry/gt_mask_registry.parquet"
        ),
    }
    monkeypatch.setattr(module, "load_execution_authority", lambda path: authority)
    manifest = module.write_sample_covariates(
        root,
        protocol_lock=protocol,
        visual_asset_manifest=visual,
        resource_gate={},
        expected_count=2,
    )
    value = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    assert value["execution_authority_mode"] == "retrospective_protocol_lock"
    assert value["execution_claim"] == artifact_record(protocol)
