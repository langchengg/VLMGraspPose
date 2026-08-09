from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/grasp4dof/run_training_grid.py"
SPEC = importlib.util.spec_from_file_location("run_training_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base(backend: str) -> dict:
    value = {
        "input_size": 224 if backend == "grconvnet" else 300,
        "conditioning_variant": "dilated_crop",
        "dilation_fraction": 0.2,
        "minimum_crop_side_px": 64,
        "quality_threshold": 0.05,
        "min_peak_distance_px": 17,
        "max_raw_candidates": 100,
        "fixed_height_px": 20.0,
        "center_gate_exponent": 2.0,
        "jaw_gate_exponent": 0.5,
        "checkpoint_path": "/tmp/official.pt",
        "checkpoint_sha256": "a" * 64,
        "nms": {
            "center_distance_px": 9.0,
            "angle_distance_deg": 14.0,
            "width_distance_px": 11.0,
            "rectangle_iou_threshold": 0.3,
        },
    }
    if backend == "grconvnet":
        value["input_channels"] = 1
    return value


def test_preregistered_grconvnet_grid_is_exact_cartesian_product() -> None:
    assert MODULE.preregistered_grid("grconvnet") == (
        (1e-4, 0.0),
        (1e-4, 1e-5),
        (5e-5, 0.0),
        (5e-5, 1e-5),
    )


def test_preregistered_ggcnn2_grid_has_only_required_learning_rates() -> None:
    assert MODULE.preregistered_grid("ggcnn2") == ((1e-4, 0.0), (5e-5, 0.0))


@pytest.mark.parametrize("backend", ["grconvnet", "ggcnn2"])
def test_training_config_preserves_validation_selected_decoder(backend: str) -> None:
    config = MODULE.build_training_config(
        backend=backend,
        transfer_config=_base(backend),
        learning_rate=5e-5,
        weight_decay=1e-5 if backend == "grconvnet" else 0.0,
        max_epochs=30,
        patience=5,
        seed=20260803,
        batch_size=0,
        device="mps",
    )
    assert config.conditioning_variant == "dilated_crop"
    assert config.dilation_fraction == 0.2
    assert config.quality_threshold == 0.05
    assert config.center_gate_exponent == 2.0
    assert config.jaw_gate_exponent == 0.5
    assert config.nms_iou_threshold == 0.3
    if backend == "grconvnet":
        assert config.grconvnet_input_channels == 1
        assert config.grconvnet_source_checkpoint == "/tmp/official.pt"


def test_training_config_rejects_oracle_or_finetuned_initialization() -> None:
    base = _base("ggcnn2")
    base["allow_oracle"] = True
    with pytest.raises(ValueError, match="predicted masks"):
        MODULE.build_training_config(
            backend="ggcnn2",
            transfer_config=base,
            learning_rate=1e-4,
            weight_decay=0.0,
            max_epochs=30,
            patience=5,
            seed=1,
            batch_size=0,
            device="cpu",
        )


def test_grid_selection_uses_registered_tie_break_order() -> None:
    slow = {
        "best_validation": {
            "j_at_1": 0.5,
            "j_at_5": 0.7,
            "non_empty_rate": 1.0,
            "validation_loss": 0.1,
            "validation_elapsed_seconds": 5.0,
        }
    }
    fast = {"best_validation": {**slow["best_validation"], "validation_elapsed_seconds": 4.0}}
    assert MODULE._selection_tuple(fast) > MODULE._selection_tuple(slow)


def test_completed_job_rejects_rehashed_non_checkpoint_payload(tmp_path: Path) -> None:
    config = MODULE.build_training_config(
        backend="ggcnn2",
        transfer_config=_base("ggcnn2"),
        learning_rate=1e-4,
        weight_decay=0.0,
        max_epochs=30,
        patience=5,
        seed=20260803,
        batch_size=1,
        device="cpu",
    )
    lineage = {"content_sha256": "b" * 64}
    checkpoint = tmp_path / "best_state_dict.pt"
    checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
    curves = tmp_path / "training_curves.csv"
    curves.write_text("epoch\n1\n", encoding="utf-8")
    best_validation = {
        "epoch": 1,
        "j_at_1": 1.0,
        "j_at_5": 1.0,
        "non_empty_rate": 1.0,
        "validation_loss": 0.0,
        "validation_elapsed_seconds": 0.1,
    }
    result = {
        "schema_version": 2,
        "status": "COMPLETE",
        "backend": "ggcnn2",
        "training_config": asdict(config),
        "data_lineage": lineage,
        "data_lineage_sha256": lineage["content_sha256"],
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": MODULE._sha256_file(checkpoint),
        "best_validation": best_validation,
        "source_checkpoint_sha256": "c" * 64,
        "epochs_completed": 1,
        "last_epoch": 1,
        "artifacts": {
            "best_state_dict.pt": MODULE._sha256_file(checkpoint),
            "training_curves.csv": MODULE._sha256_file(curves),
        },
    }
    (tmp_path / "training_complete.json").write_text(
        json.dumps(result), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot be strictly loaded"):
        MODULE._read_completed_job(tmp_path, config, lineage)


def test_completed_job_replays_curve_best_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.build_training_config(
        backend="ggcnn2",
        transfer_config=_base("ggcnn2"),
        learning_rate=1e-4,
        weight_decay=0.0,
        max_epochs=30,
        patience=5,
        seed=20260803,
        batch_size=1,
        device="cpu",
    )
    lineage = {"content_sha256": "b" * 64}
    checkpoint = tmp_path / "best_state_dict.pt"
    checkpoint.write_bytes(b"synthetic strict checkpoint")
    curves = tmp_path / "training_curves.csv"
    fields = (
        "epoch,j_at_1,j_at_5,non_empty_rate,validation_loss,"
        "validation_elapsed_seconds\n"
    )
    curves.write_text(fields + "1,1.0,1.0,1.0,0.1,0.2\n", encoding="utf-8")
    best_validation = {
        "epoch": 1,
        "j_at_1": 1.0,
        "j_at_5": 1.0,
        "non_empty_rate": 1.0,
        "validation_loss": 0.1,
        "validation_elapsed_seconds": 0.2,
    }
    result = {
        "schema_version": 2,
        "status": "COMPLETE",
        "backend": "ggcnn2",
        "training_config": asdict(config),
        "data_lineage": lineage,
        "data_lineage_sha256": lineage["content_sha256"],
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": MODULE._sha256_file(checkpoint),
        "best_validation": best_validation,
        "source_checkpoint_sha256": "c" * 64,
        "epochs_completed": 1,
        "last_epoch": 1,
        "termination_reason": "max_epochs",
        "artifacts": {
            "best_state_dict.pt": MODULE._sha256_file(checkpoint),
            "training_curves.csv": MODULE._sha256_file(curves),
        },
    }
    payload = {
        "schema_version": 2,
        "training_config": asdict(config),
        "data_lineage": lineage,
        "data_lineage_sha256": lineage["content_sha256"],
        "best_validation": best_validation,
        "best_epoch": 1,
        "source_checkpoint_sha256": "c" * 64,
        "training_curves_sha256": MODULE._sha256_file(curves),
        "epochs_completed": 1,
        "last_epoch": 1,
        "termination_reason": "max_epochs",
    }
    monkeypatch.setattr(
        MODULE,
        "load_finetuned_model",
        lambda *args, **kwargs: (
            object(),
            MODULE._sha256_file(checkpoint),
            payload,
        ),
    )
    complete = tmp_path / "training_complete.json"
    complete.write_text(json.dumps(result), encoding="utf-8")
    assert MODULE._read_completed_job(tmp_path, config, lineage) == result

    curves.write_text(fields + "1,0.9,1.0,1.0,0.1,0.2\n", encoding="utf-8")
    curve_sha = MODULE._sha256_file(curves)
    result["artifacts"]["training_curves.csv"] = curve_sha
    payload["training_curves_sha256"] = curve_sha
    complete.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="curve/best mismatch"):
        MODULE._read_completed_job(tmp_path, config, lineage)
