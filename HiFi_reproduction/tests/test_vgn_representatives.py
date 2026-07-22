from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.representatives import (
    THREE_D_ARTIFACTS,
    select_representatives,
    sync_representative_3d,
    write_representative_manifest,
)


def _rows(count: int = 90) -> list[dict[str, object]]:
    statuses = ("ok", "no_official_grasp", "no_target_grasp")
    query_types = ("name", "attribute", "relation", "location", "mixed")
    return [
        {
            "sample_id": f"sample_{index:03d}",
            "dataset_index": index,
            "status": statuses[index % len(statuses)],
            "query_type": query_types[index % len(query_types)],
            "target_category": f"category_{index % 13}",
            "mask_iou": index / count,
            "top1_vgn_quality": "" if index % 3 else 0.9 + index / (10 * count),
            "official_candidate_count": index % 17,
            "target_candidate_count": index % 7,
        }
        for index in range(count)
    ]


def test_representative_selection_is_deterministic_and_stratified() -> None:
    first = select_representatives(_rows(), count=60)
    second = select_representatives(_rows(), count=60)
    assert first == second
    assert len(first) == 60
    assert len({row["sample_id"] for row in first}) == 60
    assert {row["status"] for row in first} == {
        "ok",
        "no_official_grasp",
        "no_target_grasp",
    }


def test_representative_manifest_refuses_missing_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"sample_id": "present"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing from manifest"):
        write_representative_manifest(
            source,
            tmp_path / "subset.jsonl",
            [{"sample_id": "missing"}],
        )


def test_sync_requires_all_three_3d_artifacts(tmp_path: Path) -> None:
    sample_id = "sample_000"
    source = tmp_path / "rendered" / "samples" / sample_id
    source.mkdir(parents=True)
    for name in THREE_D_ARTIFACTS:
        (source / name).write_bytes(name.encode())
    result = sync_representative_3d(
        [{"sample_id": sample_id}],
        rendered_output=tmp_path / "rendered",
        experiment_output=tmp_path / "experiment",
    )
    assert result["complete_3d_sample_count"] == 1
    assert result["missing_artifacts"] == []
    for name in THREE_D_ARTIFACTS:
        assert (tmp_path / "experiment" / "samples" / sample_id / name).is_file()
