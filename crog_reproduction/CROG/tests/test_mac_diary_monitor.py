import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_monitor_crog_training_diary_generates_markdown_and_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "sample_console.log").write_text(
        "\n".join(
            [
                "2026-07-02 22:33:26 | INFO | engine.crog_engine:371 - "
                "Evaluation: Epoch=[1/50]  IoU=52.93  J_index@1: 38.36  "
                "J_index@5: 46.55  Pr@50: 52.39  Pr@60: 38.24  "
                "Pr@70: 27.97  Pr@80: 18.27  Pr@90: 4.03",
                "2026-07-02 22:51:55 | INFO | __main__:137 - "
                "Saved mid-epoch checkpoint exp/run/mid_epoch_model.pth "
                "at epoch 2 iteration 1000/7903",
                "2026-07-02 22:51:55 | INFO | engine.crog_engine:68 - "
                "MPS memory step=1000/7903 allocated=2746.9MB driver=11031.5MB",
                "2026-07-02 22:51:55 | INFO | utils.misc:111 - "
                "Training: Epoch=[2/50] [1000/7903]  Batch=3.17 (1.11)  "
                "Data=0.00 (0.01)  Lr=0.000100  Loss=0.0417 (0.0319)  "
                "Loss_qua=0.0011 (0.0012)  Loss_sin=0.0009 (0.0011)  "
                "Loss_cos=0.0042 (0.0032)  Loss_wid=0.0006 (0.0006)  "
                "IoU=49.54 (52.71)  Prec@50=37.50 (54.04)",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "timing_epoch_001.json").write_text(
        json.dumps(
            {
                "train_seconds": 7285.5,
                "validation_seconds": 750.6,
                "total_seconds": 8039.6,
                "average_seconds_per_iteration": 0.9219,
                "checkpoint_size_bytes": 1766270621,
                "loss_finite": True,
                "loss_min": 0.011,
                "loss_max": 20.45,
                "memory": {
                    "allocated_peak_bytes": 2886796032,
                    "driver_peak_bytes": 16380280832,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cpu_ps_samples_pid123.txt").write_text(
        "\n".join(
            [
                "timestamp pid ppid cpu_pct mem_pct rss_kb vsz_kb etime command",
                "2026-07-02T23:31:08+01:00 123 1 7.6 1.4 342816 426330800 03:11:43 python train.py",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "last_model.pth").write_bytes(b"checkpoint")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/monitor_crog_training_diary.py"),
            "--run-dir",
            str(run_dir),
            "--pid",
            str(os_getpid_fallback()),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    diary = (run_dir / "TRAINING_DIARY.md").read_text(encoding="utf-8")
    assert "Epoch 2/50, iter 1000/7903" in diary
    assert "| 1 |" in diary
    assert "52.93" in diary
    state = json.loads((run_dir / "TRAINING_DIARY.json").read_text(encoding="utf-8"))
    assert state["latest_train"]["epoch"] == 2
    assert state["completed_epochs"][0]["evaluation"]["j1"] == 38.36


def test_monitor_crog_training_diary_exposes_help():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/monitor_crog_training_diary.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def os_getpid_fallback():
    import os

    return os.getpid()
