from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.grasping.common.data_contract import (
    DEPLOYMENT_COLUMNS,
    DataContractError,
    assert_deployment_records,
    assert_gt_free_columns,
    build_split_records,
    sha256_file,
    stable_pair_hash,
    stable_sample_id,
    validate_split_isolation,
    write_parquet_bundle,
)
from tools.grasp4dof.build_manifests import main as build_manifests_main


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.resolve()


def _fixture_split(
    root: Path,
    *,
    split: str,
    checkpoint: Path,
    config: Path,
    scene: str,
    question_index: int = 0,
) -> tuple[Path, Path, Path, Path]:
    hifics = root / "hifics"
    prepared_mask = _write(
        hifics / "datasets" / split / "mask" / "0000000.png",
        f"prepared-mask-{split}".encode(),
    )
    prepared_rgb = _write(
        hifics / "datasets" / split / "image" / "0000000.png",
        f"prepared-rgb-{split}".encode(),
    )
    prepared_depth = _write(
        hifics / "datasets" / split / "depth" / "0000000.png",
        f"prepared-depth-{split}".encode(),
    )
    frozen = root / f"ocidvlg_unique_{split}.json"
    language = f"pick {split} object"
    frozen.write_text(
        json.dumps(
            [
                {
                    "num": 0,
                    "question_index": question_index,
                    "scene_id": scene,
                    "text": language,
                    "rgb_path": str(prepared_rgb.relative_to(hifics)),
                    "depth_path": str(prepared_depth.relative_to(hifics)),
                    "mask_path": str(prepared_mask.relative_to(hifics)),
                }
            ]
        ),
        encoding="utf-8",
    )
    source_rgb = _write(root / "source" / split / "rgb.png", f"rgb-{split}".encode())
    source_depth = _write(
        root / "source" / split / "depth.png", f"depth-{split}".encode()
    )
    predicted_mask = _write(
        root / "compact" / split / "predicted-mask.png", f"mask-{split}".encode()
    )
    probability = _write(
        root / "compact" / split / "probability.npz", f"prob-{split}".encode()
    )
    intrinsics = _write(
        root / "source" / split / "intrinsics.json", b'{"fx": 570.0}'
    )
    sample_id = stable_sample_id(scene, question_index)
    compact = root / "compact" / split / "manifest.jsonl"
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "sample_index": 0,
                "scene_id": scene,
                "question_index": question_index,
                "query": language,
                "split": split,
                "ready": True,
                "gt_artifacts_exported": False,
                "manifest_sha256": sha256_file(frozen),
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": sha256_file(config),
                "inference_contract_sha256": "1" * 64,
                "source_rgb_path": str(source_rgb),
                "source_rgb_sha256": sha256_file(source_rgb),
                "source_depth_path": str(source_depth),
                "source_depth_sha256": sha256_file(source_depth),
                "native_mask_path": str(predicted_mask),
                "native_mask_sha256": sha256_file(predicted_mask),
                "probability_path": str(probability),
                "probability_sha256": sha256_file(probability),
                "intrinsics_path": str(intrinsics),
                "intrinsics_sha256": sha256_file(intrinsics),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    annotations = root / "annotations" / f"{split}_expressions.json"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "split": split,
                        "image_filename": scene,
                        "question_index": question_index,
                        "question": language,
                        "answer": 7,
                        "grasps": [
                            [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return frozen, compact, annotations, hifics


def _all_splits(tmp_path: Path):
    checkpoint = _write(tmp_path / "best.pth", b"checkpoint")
    config = _write(tmp_path / "config.yaml", b"model: repeated-film\n")
    deployment = {}
    labels = {}
    for split in ("train", "val", "test"):
        frozen, compact, annotations, hifics = _fixture_split(
            tmp_path,
            split=split,
            checkpoint=checkpoint,
            config=config,
            scene=f"scene/{split},frame.png",
        )
        deployment[split], labels[split] = build_split_records(
            split=split,
            frozen_manifest_path=frozen,
            compact_manifest_path=compact,
            annotations_path=annotations,
            hifics_root=hifics,
            checkpoint_path=checkpoint,
            config_path=config,
        )
    return deployment, labels


def test_builds_gt_free_deployment_and_separate_zstd_labels(tmp_path: Path) -> None:
    deployment, labels = _all_splits(tmp_path)
    audit = validate_split_isolation(deployment)
    assert all(not any(values.values()) for values in audit.values())
    outputs = write_parquet_bundle(
        output_dir=tmp_path / "run" / "manifests",
        deployment_by_split=deployment,
        labels_by_split=labels,
    )

    assert len(outputs) == 6
    samples = pq.read_table(outputs["train_samples"])
    train_labels = pq.read_table(outputs["train_labels"])
    assert tuple(samples.column_names) == DEPLOYMENT_COLUMNS
    assert "gt_grasp_rectangles" not in samples.column_names
    assert "prepared_gt_mask_path" not in samples.column_names
    assert train_labels["gt_grasp_count"].to_pylist() == [1]
    assert train_labels["gt_grasp_rectangles"].to_pylist()[0][0][0] == [1.0, 2.0]
    parquet = pq.ParquetFile(outputs["train_samples"])
    assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"


def test_cli_writes_only_below_run_manifests(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "best.pth", b"checkpoint")
    config = _write(tmp_path / "config.yaml", b"model: repeated-film\n")
    for split in ("train", "val", "test"):
        _fixture_split(
            tmp_path,
            split=split,
            checkpoint=checkpoint,
            config=config,
            scene=f"scene/{split},frame.png",
        )
    run_dir = tmp_path / "new-run"
    assert (
        build_manifests_main(
            [
                "--run-dir",
                str(run_dir),
                "--frozen-manifests-dir",
                str(tmp_path),
                "--compact-root",
                str(tmp_path / "compact"),
                "--annotations-root",
                str(tmp_path / "annotations"),
                "--hifics-root",
                str(tmp_path / "hifics"),
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(config),
            ]
        )
        == 0
    )
    written = sorted(path for path in run_dir.rglob("*") if path.is_file())
    assert len(written) == 6
    assert all(path.parent == run_dir / "manifests" for path in written)
    assert all(path.suffix == ".parquet" for path in written)


def test_rejects_forbidden_deployment_columns() -> None:
    with pytest.raises(DataContractError, match="GT/evaluation columns"):
        assert_gt_free_columns([*DEPLOYMENT_COLUMNS, "gt_mask_path"])


def test_rejects_incomplete_compact_coverage(tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "best.pth", b"checkpoint")
    config = _write(tmp_path / "config.yaml", b"model: repeated-film\n")
    frozen, compact, annotations, hifics = _fixture_split(
        tmp_path,
        split="train",
        checkpoint=checkpoint,
        config=config,
        scene="scene/train,frame.png",
    )
    compact.write_text("", encoding="utf-8")
    with pytest.raises((DataContractError, FileNotFoundError), match="compact"):
        build_split_records(
            split="train",
            frozen_manifest_path=frozen,
            compact_manifest_path=compact,
            annotations_path=annotations,
            hifics_root=hifics,
            checkpoint_path=checkpoint,
            config_path=config,
        )


def test_rejects_row_scene_rgb_depth_and_pair_split_overlap(tmp_path: Path) -> None:
    deployment, _ = _all_splits(tmp_path)
    for key in (
        "sample_id",
        "scene_id",
        "source_rgb_sha256",
        "source_depth_sha256",
        "rgbd_pair_sha256",
    ):
        mutated = {
            split: [dict(row) for row in rows] for split, rows in deployment.items()
        }
        mutated["val"][0][key] = mutated["train"][0][key]
        if key == "sample_id":
            # Preserve row-level validity so the global intersection is the failure.
            mutated["val"][0]["scene_id"] = mutated["train"][0]["scene_id"]
            mutated["val"][0]["question_index"] = mutated["train"][0][
                "question_index"
            ]
            mutated["val"][0]["expression_index"] = mutated["train"][0][
                "expression_index"
            ]
        elif key == "scene_id":
            # The stable ID is a function of scene and expression identity.
            mutated["val"][0]["sample_id"] = stable_sample_id(
                mutated["val"][0]["scene_id"],
                mutated["val"][0]["question_index"],
            )
        elif key == "rgbd_pair_sha256":
            # A pair hash cannot change independently without failing row integrity.
            mutated["val"][0]["source_rgb_sha256"] = mutated["train"][0][
                "source_rgb_sha256"
            ]
            mutated["val"][0]["source_depth_sha256"] = mutated["train"][0][
                "source_depth_sha256"
            ]
        elif key in ("source_rgb_sha256", "source_depth_sha256"):
            mutated["val"][0]["rgbd_pair_sha256"] = stable_pair_hash(
                mutated["val"][0]["source_rgb_sha256"],
                mutated["val"][0]["source_depth_sha256"],
            )
        with pytest.raises(DataContractError, match="split leakage"):
            validate_split_isolation(mutated)


def test_rejects_extra_gt_field_even_before_arrow_serialization(tmp_path: Path) -> None:
    deployment, _ = _all_splits(tmp_path)
    row = dict(deployment["train"][0])
    row["gt_grasps"] = []
    with pytest.raises(DataContractError, match="schema mismatch"):
        assert_deployment_records([row], expected_split="train")
