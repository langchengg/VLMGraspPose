from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gtmask_counterfactual.g1_c1_adapter import (
    G1C1AdapterError,
    LABEL_PROJECTION,
    build_g1_c1_source_adapter,
    verify_g1_c1_source_adapter,
)
from gtmask_counterfactual.io import (
    artifact_record,
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "frozen"
    source.joinpath("manifests").mkdir(parents=True)
    source.joinpath("selected_configs").mkdir()
    sample_ids = ["s2", "s0", "s1"]
    atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": sample_ids,
                "scene_id": ["scene-2", "scene-0", "scene-1"],
                "language": ["two", "zero", "one"],
                "source_rgb_path": ["/rgb/2", "/rgb/0", "/rgb/1"],
            }
        ),
        source / "manifests/test_samples.parquet",
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "prepared_gt_mask_path": ["/mask/0", "/mask/1", "/mask/2"],
            "prepared_gt_mask_sha256": ["a" * 64, "b" * 64, "c" * 64],
            "gt_grasp_rectangles": ["forbidden-0", "forbidden-1", "forbidden-2"],
        }
    )
    atomic_parquet(labels, source / "manifests/test_labels.parquet")
    for route in ("G1", "C1"):
        source.joinpath(f"selected_configs/{route}.json").write_text(
            json.dumps({"route": route}) + "\n", encoding="utf-8"
        )
    registry_path = tmp_path / "run/03_gt_mask_registry/gt_mask_registry.parquet"
    registry = labels.loc[:, list(LABEL_PROJECTION)].copy()
    registry["mapping_status"] = ["PASS", "UNRESOLVED", "PASS"]
    registry["pixel_qa_status"] = [
        "P2_MAPPING_QA_PASS",
        "NOT_EVALUABLE",
        "P2_MAPPING_QA_PASS",
    ]
    registry["bulk_gt_pixels_read"] = [True, False, True]
    atomic_parquet(registry, registry_path)
    mapping: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "sample_count": 3,
        "outputs": {"gt_mask_registry": artifact_record(registry_path)},
    }
    mapping["content_sha256"] = canonical_sha256(mapping)
    mapping_path = atomic_json(
        tmp_path / "run/03_gt_mask_registry/GT_MASK_MAPPING_AUDIT.json", mapping
    )
    return tmp_path / "run", source, registry_path, mapping_path


def test_g1_c1_adapter_projects_only_executable_mask_references(tmp_path: Path) -> None:
    run, source, registry, mapping = _fixture(tmp_path)
    manifest_path = build_g1_c1_source_adapter(
        run_dir=run,
        source_run=source,
        registry_path=registry,
        mapping_audit_path=mapping,
        expected_count=3,
    )
    manifest = verify_g1_c1_source_adapter(manifest_path, expected_count=3)
    samples = pd.read_parquet(manifest["test_samples"]["path"])
    labels = pd.read_parquet(manifest["test_labels_projection"]["path"])
    assert samples["sample_id"].tolist() == ["s2", "s0"]
    assert labels["sample_id"].tolist() == ["s2", "s0"]
    assert list(labels.columns) == list(LABEL_PROJECTION)
    assert "gt_grasp_rectangles" not in labels
    assert manifest["evaluable_sample_count"] == 2
    assert manifest["unresolved_sample_count"] == 1
    assert manifest["gt_grasp_rows_read"] == 0
    assert manifest["gt_mask_pixels_read"] == 0
    assert build_g1_c1_source_adapter(
        run_dir=run,
        source_run=source,
        registry_path=registry,
        mapping_audit_path=mapping,
        expected_count=3,
        resume=True,
    ) == manifest_path

    samples.loc[0, "scene_id"] = "tampered"
    samples.to_parquet(manifest["test_samples"]["path"], index=False)
    with pytest.raises(G1C1AdapterError, match="artifact hash differs"):
        verify_g1_c1_source_adapter(manifest_path, expected_count=3)
