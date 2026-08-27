"""Grounding-pipeline fixtures are synthetic and never formal evidence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import graspnet6d.grounding_pipeline as pipeline
from graspnet6d.grounding import AdaptationResult
from graspnet6d.io import atomic_jsonl, sha256_file


def _manifests(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, Path]]]:
    targets = []
    languages = []
    sources: dict[str, dict[str, Path]] = {}
    for index, split in enumerate(("train", "validation", "test")):
        scene = f"scene_{index:04d}"
        group_id = f"{scene}_kinect_0000_obj_000"
        folder = tmp_path / scene
        folder.mkdir()
        rgb = np.zeros((5, 6, 3), dtype=np.uint8)
        rgb[1:4, 2:5] = (180, 20, 30)
        depth = np.full((5, 6), 1000, dtype=np.uint16)
        label = np.zeros((5, 6), dtype=np.uint16)
        label[1:4, 2:5] = 1
        rgb_path = folder / "rgb.png"
        depth_path = folder / "depth.png"
        label_path = folder / "label.png"
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(depth).save(depth_path)
        Image.fromarray(label).save(label_path)
        intrinsics_path = folder / "camK.npy"
        poses_path = folder / "poses.npy"
        table_path = folder / "table.npy"
        meta_path = folder / "meta.mat"
        np.save(intrinsics_path, np.eye(3), allow_pickle=False)
        np.save(poses_path, np.eye(4)[None], allow_pickle=False)
        np.save(table_path, np.eye(4), allow_pickle=False)
        meta_path.write_bytes(b"fixture metadata")
        targets.append(
            {
                "group_id": group_id,
                "split": split,
                "scene_id": scene,
                "camera": "kinect",
                "frame_id": 0,
                "target_object_id": 0,
                "target_instance_label": 1,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "instance_label_path": str(label_path),
                "meta_path": str(meta_path),
                "intrinsics_path": str(intrinsics_path),
                "camera_pose_path": str(poses_path),
                "table_transform_path": str(table_path),
            }
        )
        languages.append(
            {
                "group_id": group_id,
                "query": "Pick the object.",
                "template_family": "catalog_name",
                "resolver_result": [0],
                "is_unique": True,
                "provenance": "derived_test_fixture",
            }
        )
        sources[split] = {
            "rgb": rgb_path,
            "depth": depth_path,
            "label": label_path,
        }
    target_path = tmp_path / "targets.jsonl"
    language_path = tmp_path / "language.jsonl"
    atomic_jsonl(target_path, targets)
    atomic_jsonl(language_path, languages)
    return target_path, language_path, sources


def test_adaptation_manifests_exclude_test_and_bind_real_sources(tmp_path: Path) -> None:
    targets, language, _ = _manifests(tmp_path)

    result = pipeline.build_adaptation_manifests(targets, language, tmp_path / "run")

    train = list(
        pipeline.load_jsonl_records(
            result["train_manifest_path"], description="train fixture"
        )
    )
    validation = list(
        pipeline.load_jsonl_records(
            result["validation_manifest_path"], description="validation fixture"
        )
    )
    assert [row["split"] for row in train] == ["train"]
    assert [row["split"] for row in validation] == ["val"]
    assert result["test_rows_consumed"] == 0
    for row in [*train, *validation]:
        assert row["rgb_sha256"] == sha256_file(row["rgb_path"])
        assert row["instance_label_sha256"] == sha256_file(
            row["instance_label_path"]
        )


def test_adaptation_stage_writes_validation_only_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    targets, language, _ = _manifests(tmp_path)
    base = tmp_path / "base.pth"
    clip = tmp_path / "clip.pt"
    base.write_bytes(b"base")
    clip.write_bytes(b"clip")
    monkeypatch.setattr(pipeline, "EXPECTED_HIFI_CHECKPOINT_SHA256", sha256_file(base))
    monkeypatch.setattr(pipeline, "EXPECTED_CLIP_WEIGHT_SHA256", sha256_file(clip))
    validations: list[Path] = []

    def trainer(train_rows, validation_rows, **kwargs):
        output = Path(kwargs["output_checkpoint"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"adapted fixture")
        split_contract = pipeline.validate_adaptation_splits(
            train_rows, validation_rows
        )
        return AdaptationResult(
            model=object(),
            best_epoch=2,
            best_validation_miou=0.75,
            epochs_completed=3,
            stopped_early=False,
            history=(
                {
                    "epoch": 2,
                    "validation_mean_iou": 0.75,
                    "selection_split": "val",
                    "test_rows_consumed": 0,
                },
            ),
            output_checkpoint=output,
            split_contract=split_contract,
        )

    result = pipeline.run_hifi_adaptation_stage(
        targets,
        language,
        tmp_path / "run",
        checkpoint_path=base,
        clip_weight_path=clip,
        trainer=trainer,
        evidence_validator=lambda path: validations.append(Path(path)),
    )

    assert result["selection_split"] == "val"
    assert result["test_rows_consumed"] == 0
    assert result["input_splits"] == ["train", "val"]
    assert result["best_validation_mean_iou"] == 0.75
    assert validations == [Path(result["evidence_path"])]


def _mask_loader(tmp_path: Path, *, perfect: bool):
    def load(_root, condition: str, group_id: str):
        folder = tmp_path / "mask-fixtures" / condition / group_id
        folder.mkdir(parents=True, exist_ok=True)
        sidecar = folder / "mask.json"
        probability = folder / "mask.npz"
        mask_path = folder / "mask.png"
        sidecar.write_text("{}", encoding="utf-8")
        probability.write_bytes(b"fixture")
        predicted = np.zeros((5, 6), dtype=bool)
        if perfect:
            predicted[1:4, 2:5] = True
        else:
            predicted[0, 0] = True
        Image.fromarray(predicted.astype(np.uint8) * 255).save(mask_path)
        return SimpleNamespace(
            binary_mask=predicted,
            sidecar_path=sidecar,
            probability_path=probability,
            mask_path=mask_path,
        )

    return load


def test_validation_grounding_metrics_select_adapted_without_test_access(
    tmp_path: Path,
) -> None:
    targets, language, _ = _manifests(tmp_path)
    run = tmp_path / "run"
    zero = pipeline.run_grounding_metric_stage(
        targets,
        language,
        run,
        run,
        condition="hifics_zero_shot_mask",
        included_splits=("validation",),
        selection_scope=True,
        mask_loader=_mask_loader(tmp_path, perfect=False),
    )
    adapted = pipeline.run_grounding_metric_stage(
        targets,
        language,
        run,
        run,
        condition="hifics_adapted_mask",
        included_splits=("validation",),
        selection_scope=True,
        mask_loader=_mask_loader(tmp_path, perfect=True),
    )
    zero_path = (
        run
        / "grounding_metrics/selection_validation/hifics_zero_shot_mask/summary.json"
    )
    adapted_path = (
        run
        / "grounding_metrics/selection_validation/hifics_adapted_mask/summary.json"
    )
    selected = pipeline.select_predicted_condition(
        zero_path,
        adapted_path,
        run / "predicted_condition_selection.json",
    )

    assert zero["included_splits"] == ["validation"]
    assert adapted["included_splits"] == ["validation"]
    assert zero["test_rows_consumed_for_selection"] == 0
    assert selected["selected_condition"] == "hifics_adapted_mask"
    assert selected["test_rows_consumed"] == 0
