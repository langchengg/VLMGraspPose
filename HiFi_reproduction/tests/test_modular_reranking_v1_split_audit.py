from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grasping.reranking_v1.split_audit import build_split_audit


def _manifest(root: Path, split: str, scene: str, content: bytes) -> Path:
    data = root / split
    data.mkdir()
    rgb = data / "rgb.bin"
    depth = data / "depth.bin"
    rgb.write_bytes(content + b"-rgb")
    depth.write_bytes(content + b"-depth")
    path = root / f"{split}.json"
    path.write_text(
        json.dumps(
            [
                {
                    "num": 0,
                    "question_index": 0,
                    "scene_id": f"{scene},frame.png",
                    "text": f"pick {split}",
                    "rgb_path": str(rgb),
                    "depth_path": str(depth),
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _provenance() -> dict[str, str]:
    return {
        "hifi_checkpoint_sha256": "a" * 64,
        "dexnet_config_sha256": "b" * 64,
        "gqcnn_model_manifest_sha256": "c" * 64,
    }


def test_split_audit_accepts_disjoint_assets(tmp_path: Path) -> None:
    manifests = {
        split: _manifest(tmp_path, split, f"seq-{split}", split.encode())
        for split in ("train", "val", "test")
    }
    summary, frame = build_split_audit(
        manifest_paths=manifests,
        hifics_root=tmp_path,
        provenance=_provenance(),
        workers=2,
    )
    assert summary["required_intersections_all_zero"] is True
    assert len(frame) == 3


def test_split_audit_rejects_same_rgb_bytes(tmp_path: Path) -> None:
    manifests = {
        split: _manifest(tmp_path, split, f"seq-{split}", b"same")
        for split in ("train", "val", "test")
    }
    with pytest.raises(AssertionError, match="split leakage"):
        build_split_audit(
            manifest_paths=manifests,
            hifics_root=tmp_path,
            provenance=_provenance(),
            workers=2,
        )


def test_split_audit_rejects_depth_hash_leakage_even_when_rgb_differs(
    tmp_path: Path,
) -> None:
    manifests = {
        split: _manifest(tmp_path, split, f"seq-{split}", split.encode())
        for split in ("train", "val", "test")
    }
    for split in manifests:
        (tmp_path / split / "depth.bin").write_bytes(b"identical-depth")
    with pytest.raises(AssertionError, match="split leakage"):
        build_split_audit(
            manifest_paths=manifests,
            hifics_root=tmp_path,
            provenance=_provenance(),
            workers=2,
        )


def test_split_audit_rejects_duplicate_scene_id_with_unique_assets(
    tmp_path: Path,
) -> None:
    manifests = {
        split: _manifest(tmp_path, split, "same-sequence", split.encode())
        for split in ("train", "val", "test")
    }
    with pytest.raises(AssertionError, match="split leakage"):
        build_split_audit(
            manifest_paths=manifests,
            hifics_root=tmp_path,
            provenance=_provenance(),
            workers=2,
        )
