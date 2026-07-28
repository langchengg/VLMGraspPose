from pathlib import Path

import torch

import utils.config as config
from train_crog_mac import (
    _limit_dataset,
    _make_mid_epoch_checkpoint_callback,
    _mid_epoch_checkpoint_interval,
    _resolve_device,
    _timing_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_debug_config_is_bounded_and_mps_safe():
    cfg = config.load_cfg_from_cfg_file(str(
        REPO_ROOT / "config" / "OCID-VLG" / "CROG_mac_mps_debug.yaml"
    ))

    assert cfg.batch_size == 1
    assert cfg.batch_size_val == 1
    assert cfg.workers == 0
    assert cfg.workers_val == 0
    assert cfg.epochs == 1
    assert cfg.max_train_samples == 8
    assert cfg.max_val_samples == 8
    assert cfg.sync_bn is False
    assert cfg.device == "auto"
    assert cfg.checkpoint_interval == 0


def test_limit_dataset_uses_first_requested_samples():
    dataset = list(range(10))

    subset = _limit_dataset(dataset, 3)

    assert len(subset) == 3
    assert [subset[index] for index in range(3)] == [0, 1, 2]


def test_resolve_device_accepts_explicit_cpu():
    assert _resolve_device("cpu") == torch.device("cpu")


def test_timing_summary_calculates_iteration_average():
    summary = _timing_summary(
        train_seconds=20.0,
        validation_seconds=5.0,
        checkpoint_seconds=2.0,
        total_seconds=27.0,
        train_iterations=8,
    )

    assert summary["average_seconds_per_iteration"] == 2.5
    assert summary["total_seconds"] == 27.0


def test_full_one_epoch_config_has_unbounded_dataset():
    cfg = config.load_cfg_from_cfg_file(str(
        REPO_ROOT / "config" / "OCID-VLG" / "CROG_mac_mps_full_1epoch.yaml"
    ))

    assert cfg.input_size == 320
    assert cfg.epochs == 1
    assert cfg.accumulation_steps == 8
    assert cfg.checkpoint_interval == 0
    assert cfg.max_train_samples is None
    assert cfg.max_val_samples is None
    assert cfg.freeze_bn1d_stats is True


def test_full_training_config_enables_rolling_mid_epoch_checkpoint():
    cfg = config.load_cfg_from_cfg_file(str(
        REPO_ROOT / "config" / "OCID-VLG" / "CROG_mac_mps.yaml"
    ))

    assert cfg.epochs == 50
    assert cfg.checkpoint_interval == 1000


def test_mid_epoch_checkpoint_interval_defaults_to_disabled():
    class Args:
        pass

    assert _mid_epoch_checkpoint_interval(Args()) == 0


def test_mid_epoch_checkpoint_callback_writes_rolling_recovery_file(tmp_path):
    class Args:
        output_dir = str(tmp_path)
        checkpoint_interval = 2

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = _make_mid_epoch_checkpoint_callback(
        Args(), model, optimizer, None, None, best_iou=0.3, best_j_index=0.4
    )

    callback(epoch=3, iteration=1, total_batches=5)
    assert not (tmp_path / "mid_epoch_model.pth").exists()

    callback(epoch=3, iteration=2, total_batches=5)
    checkpoint = torch.load(tmp_path / "mid_epoch_model.pth", map_location="cpu")
    assert checkpoint["mid_epoch"] is True
    assert checkpoint["epoch"] == 2
    assert checkpoint["epoch_in_progress"] == 3
    assert checkpoint["iteration"] == 2
    assert checkpoint["total_batches"] == 5
    assert checkpoint["best_iou"] == 0.3

    callback(epoch=3, iteration=5, total_batches=5)
    checkpoint = torch.load(tmp_path / "mid_epoch_model.pth", map_location="cpu")
    assert checkpoint["iteration"] == 2
