from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.grasping.backends.training import (
    OcidGraspTrainingDataset,
    TrainingConfig,
    TrainingExample,
    _assert_finite_metrics,
    _auto_batch_size,
    _epoch_training_loader,
    _loss,
    _run_training_epoch,
    _validated_data_lineage,
    collate_training_examples,
    train_finetuned_backend,
)
from src.grasping.common.results import (
    aggregate_method_metrics,
    assert_metric_consistency,
    evaluate_prediction_records,
)
from src.grasping.common.types import Grasp4DoF, GraspPrediction


def _grasp(identifier: str, score: float, x: float = 50.0) -> Grasp4DoF:
    return Grasp4DoF(
        center_x=x,
        center_y=50,
        angle_deg=0,
        width_px=60,
        height_px=20,
        score=score,
        candidate_id=identifier,
    )


def test_complete_candidate_pool_drives_oracle_not_only_top5() -> None:
    candidates = tuple(
        [_grasp(f"bad-{index}", 1.0 - index / 10, x=5) for index in range(5)]
        + [_grasp("good-sixth", 0.1)]
    )
    prediction = GraspPrediction(
        sample_id="sample",
        backend="backend",
        conditioning_variant="predicted",
        raw_candidate_count=6,
        nms_candidate_count=6,
        top1=candidates[0],
        top5=candidates[:5],
        candidates=candidates,
    )
    label = {
        "sample_id": "sample",
        "scene_id": "scene",
        "gt_grasp_rectangles": [[[20, 40], [20, 60], [80, 60], [80, 40]]],
    }
    sample, rows = evaluate_prediction_records(
        method="method", prediction=prediction, label=label
    )
    assert len(rows) == 6
    assert sample["j_at_1"] is False
    assert sample["j_at_5"] is False
    assert sample["candidate_pool_oracle"] is True
    assert sample["first_valid_rank"] == 6


def test_aggregate_all_sample_keeps_empty_failure() -> None:
    rows = [
        {
            "method": "m",
            "j_at_1": True,
            "j_at_5": True,
            "candidate_pool_oracle": True,
            "reciprocal_rank": 1.0,
            "first_valid_rank": 1,
            "non_empty": True,
            "raw_candidate_count": 1,
            "nms_candidate_count": 1,
            "latency_seconds": 0.1,
        },
        {
            "method": "m",
            "j_at_1": False,
            "j_at_5": False,
            "candidate_pool_oracle": False,
            "reciprocal_rank": 0.0,
            "first_valid_rank": None,
            "non_empty": False,
            "raw_candidate_count": 0,
            "nms_candidate_count": 0,
            "latency_seconds": 0.2,
        },
    ]
    metrics = aggregate_method_metrics(rows)
    assert metrics["j_at_1"] == 0.5
    assert metrics["non_empty_rate"] == 0.5
    assert metrics["non_empty_j_at_1"] == 1.0
    assert_metric_consistency(metrics)


def _fixture(tmp_path: Path) -> tuple[list[dict], list[dict]]:
    rgb = np.full((100, 100, 3), (40, 80, 120), dtype=np.uint8)
    depth = np.full((100, 100), 750, dtype=np.uint16)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    probability = (mask > 0).astype(np.float32)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(depth).save(tmp_path / "depth.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    np.savez_compressed(tmp_path / "probability.npz", probability=probability)
    paths = {
        name: tmp_path / name
        for name in ("rgb.png", "depth.png", "mask.png", "probability.npz")
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    language = "object"
    deployment = [
        {
            "sample_id": "sample",
            "scene_id": "scene",
            "language": language,
            "language_sha256": hashlib.sha256(language.encode()).hexdigest(),
            "source_rgb_path": str(tmp_path / "rgb.png"),
            "source_rgb_sha256": hashes["rgb.png"],
            "source_depth_path": str(tmp_path / "depth.png"),
            "source_depth_sha256": hashes["depth.png"],
            "predicted_mask_path": str(tmp_path / "mask.png"),
            "predicted_mask_sha256": hashes["mask.png"],
            "predicted_probability_path": str(tmp_path / "probability.npz"),
            "predicted_probability_sha256": hashes["probability.npz"],
            "intrinsics_path": None,
            "intrinsics_provenance": json.dumps({"kind": "unavailable"}),
        }
    ]
    labels = [
        {
            "sample_id": "sample",
            "gt_grasp_rectangles": [
                [[20, 40], [20, 60], [80, 60], [80, 40]]
            ],
        }
    ]
    return deployment, labels


@pytest.mark.parametrize("channels", [1, 4])
def test_training_dataset_uses_official_gr_depth_first_and_ocid_width(
    tmp_path: Path, channels: int
) -> None:
    deployment, labels = _fixture(tmp_path)
    dataset = OcidGraspTrainingDataset(
        deployment,
        labels,
        TrainingConfig(
            backend="grconvnet",
            input_size=100,
            conditioning_variant="hard_mask",
            max_epochs=1,
            patience=1,
            grconvnet_input_channels=channels,
        ),
    )
    example = dataset[0]
    assert example is not None
    assert example.model_input.shape == (channels, 100, 100)
    assert np.array_equal(example.model_input[0], example.conditioned.depth_chw[0])
    if channels == 4:
        assert np.array_equal(example.model_input[1:], example.conditioned.rgb_chw)
    positive = example.targets[0] > 0
    assert np.allclose(example.targets[3][positive], 60.0 / 150.0)
    batch = collate_training_examples([example, None])
    assert batch is not None and batch["input"].shape[0] == 1


def test_official_loss_family_is_backend_specific() -> None:
    output = [torch.tensor([[[[2.0]]]], requires_grad=True) for _ in range(4)]
    target = [torch.zeros_like(value) for value in output]
    gr_loss, _ = _loss("grconvnet", output, target)
    gg_loss, _ = _loss("ggcnn2", output, target)
    assert float(gr_loss.detach()) == pytest.approx(6.0)  # 4 * SmoothL1(2, 0)
    assert float(gg_loss.detach()) == pytest.approx(16.0)  # 4 * MSE(2, 0)


def _synthetic_examples(count: int) -> list[TrainingExample]:
    target = np.zeros((4, 4), dtype=np.float32)
    return [
        TrainingExample(
            sample_id=f"sample-{index}",
            model_input=np.ones((1, 4, 4), dtype=np.float32),
            targets=(target, target, target, target),
            conditioned=None,  # type: ignore[arg-type]
            gt_corners=[],
        )
        for index in range(count)
    ]


class _BatchSensitiveModel(nn.Module):
    def __init__(self, *, non_oom_failure_at_eight: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.non_oom_failure_at_eight = non_oom_failure_at_eight

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.non_oom_failure_at_eight and inputs.shape[0] == 8:
            raise RuntimeError("shape contract broken, not OOM")
        output = inputs[:, :1] * self.scale
        return output, output, output, output


def _tiny_config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "backend": "ggcnn2",
        "input_size": 4,
        "conditioning_variant": "hard_mask",
        "max_epochs": 3,
        "patience": 1,
        "batch_size": 1,
        "num_workers": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def test_training_data_contract_rejects_train_validation_overlap() -> None:
    rows = [{"sample_id": "same-sample"}]
    with pytest.raises(ValueError, match="train/validation sample IDs overlap"):
        _validated_data_lineage(
            train_deployment=rows,
            train_labels=rows,
            validation_deployment=rows,
            validation_labels=rows,
            declared=None,
        )


def test_epoch_loader_recreates_identical_shuffle_after_resume() -> None:
    examples = _synthetic_examples(32)
    config = _tiny_config(batch_size=4, num_workers=2, seed=20260803)

    def order() -> list[str]:
        loader = _epoch_training_loader(
            examples, batch_size=4, config=config, epoch=2
        )
        assert loader.persistent_workers is False
        return [
            example.sample_id
            for batch in loader
            for example in batch["examples"]
        ]

    assert order() == order()


def test_auto_batch_does_not_swallow_non_oom_runtime_error() -> None:
    model = _BatchSensitiveModel(non_oom_failure_at_eight=True)
    with pytest.raises(RuntimeError, match="non-OOM reason"):
        _auto_batch_size(
            model,
            _synthetic_examples(8),  # type: ignore[arg-type]
            _tiny_config(batch_size=0),
            torch.device("cpu"),
        )


def test_non_finite_loss_and_metrics_fail_closed() -> None:
    class NaNModel(_BatchSensitiveModel):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            output = inputs[:, :1] * self.scale * torch.tensor(float("nan"))
            return output, output, output, output

    model = NaNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = [collate_training_examples(_synthetic_examples(1))]
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        _run_training_epoch(
            model,
            loader,  # type: ignore[arg-type]
            optimizer,
            _tiny_config(),
            torch.device("cpu"),
        )
    with pytest.raises(FloatingPointError, match="non-finite training metric"):
        _assert_finite_metrics({"j_at_1": float("nan")})


def test_resume_at_patience_boundary_does_not_run_extra_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_rows = [{"sample_id": "train"}]
    validation_rows = [{"sample_id": "validation"}]
    config = _tiny_config()
    lineage = _validated_data_lineage(
        train_deployment=train_rows,
        train_labels=train_rows,
        validation_deployment=validation_rows,
        validation_labels=validation_rows,
        declared=None,
    )
    model = _BatchSensitiveModel()
    source_sha = "a" * 64
    monkeypatch.setattr(
        "src.grasping.backends.training._model_from_official",
        lambda _config: (model, source_sha, "synthetic"),
    )
    monkeypatch.setattr(
        "src.grasping.backends.training._run_training_epoch",
        lambda *args, **kwargs: pytest.fail("resume crossed early-stop boundary"),
    )
    destination = tmp_path / "job"
    destination.mkdir()
    config_value = json.loads(json.dumps(asdict(config)))
    best_row = {
        "epoch": 1,
        "training_loss": 1.0,
        "training_position_loss": 0.25,
        "training_cos_loss": 0.25,
        "training_sin_loss": 0.25,
        "training_width_loss": 0.25,
        "training_examples": 1,
        "skipped_batches": 0,
        "validation_loss": 1.0,
        "validation_evaluated": 1,
        "validation_valid_inputs": 1,
        "j_at_1": 0.0,
        "j_at_5": 0.0,
        "non_empty_rate": 0.0,
        "validation_elapsed_seconds": 0.1,
        "elapsed_seconds": 0.2,
    }
    stale_row = {
        **best_row,
        "epoch": 2,
        "validation_loss": 1.1,
        "elapsed_seconds": 0.4,
    }
    best_path = destination / "best_state_dict.pt"
    torch.save(
        {
            "schema_version": 2,
            "backend": "ggcnn2",
            "model_state_dict": model.state_dict(),
            "source_checkpoint_sha256": source_sha,
            "initialization": "synthetic",
            "training_config": config_value,
            "data_lineage": lineage,
            "data_lineage_sha256": lineage["content_sha256"],
            "best_epoch": 1,
            "best_validation": best_row,
        },
        best_path,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    torch.save(
        {
            "schema_version": 2,
            "backend": "ggcnn2",
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_row": best_row,
            "stale_epochs": 1,
            "training_config": config_value,
            "data_lineage": lineage,
            "data_lineage_sha256": lineage["content_sha256"],
            "source_checkpoint_sha256": source_sha,
            "batch_size": 1,
            "batch_size_attempts": [{"batch_size": 1, "status": "EXPLICIT"}],
            "best_checkpoint_sha256": hashlib.sha256(
                best_path.read_bytes()
            ).hexdigest(),
            "rng_schedule": "sha256(seed,epoch)-v1",
            "epoch_seed": 1,
            "torch_rng_state": torch.get_rng_state(),
            "history": [best_row, stale_row],
        },
        destination / "resume_state.pt",
    )

    result = train_finetuned_backend(
        train_deployment=train_rows,
        train_labels=train_rows,
        validation_deployment=validation_rows,
        validation_labels=validation_rows,
        output_dir=destination,
        config=config,
    )
    assert result["termination_reason"] == "early_stopping_resume_boundary"
    assert result["last_epoch"] == 2
    assert not (destination / "resume_state.pt").exists()


def test_epoch_oom_rolls_back_before_reducing_auto_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_rows = [{"sample_id": "train"}]
    validation_rows = [{"sample_id": "validation"}]
    config = _tiny_config(max_epochs=1, patience=1, batch_size=0)
    model = _BatchSensitiveModel()
    monkeypatch.setattr(
        "src.grasping.backends.training._model_from_official",
        lambda _config: (model, "a" * 64, "synthetic"),
    )
    monkeypatch.setattr(
        "src.grasping.backends.training._auto_batch_size",
        lambda *args, **kwargs: (8, [{"batch_size": 8, "status": "PASS"}]),
    )
    calls = 0

    def training_epoch(*args: object, **kwargs: object) -> dict[str, float | int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            with torch.no_grad():
                model.scale.add_(10.0)
            raise RuntimeError("MPS backend out of memory")
        assert float(model.scale.detach()) == pytest.approx(1.0)
        loader = args[1]
        assert getattr(loader, "batch_size") == 4
        return {
            "training_loss": 1.0,
            "training_position_loss": 0.25,
            "training_cos_loss": 0.25,
            "training_sin_loss": 0.25,
            "training_width_loss": 0.25,
            "training_examples": 1,
            "skipped_batches": 0,
        }

    monkeypatch.setattr(
        "src.grasping.backends.training._run_training_epoch", training_epoch
    )
    monkeypatch.setattr(
        "src.grasping.backends.training._run_validation",
        lambda *args, **kwargs: {
            "validation_loss": 1.0,
            "validation_evaluated": 1,
            "validation_valid_inputs": 1,
            "j_at_1": 0.0,
            "j_at_5": 0.0,
            "non_empty_rate": 0.0,
        },
    )

    result = train_finetuned_backend(
        train_deployment=train_rows,
        train_labels=train_rows,
        validation_deployment=validation_rows,
        validation_labels=validation_rows,
        output_dir=tmp_path / "job",
        config=config,
    )
    assert calls == 2
    assert result["batch_size"] == 4
    assert any(
        row["status"] == "OOM_DURING_EPOCH_ROLLED_BACK"
        for row in result["batch_size_attempts"]
    )
