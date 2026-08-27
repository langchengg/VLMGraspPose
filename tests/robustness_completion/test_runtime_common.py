from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from robustness_completion import runtime_common
from robustness_completion.runtime_common import (
    build_four_d_subset_manifest,
    run_monitored_command,
    stable_sequence_round_robin,
    timed_call_ns,
)


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "artifacts/robustness_completion/20260822_194405_remaining_robustness"


def test_runtime_subset_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"q{index:03d}" for index in range(30)],
            "sequence_id": [f"s{index % 7}" for index in range(30)],
        }
    )
    first = stable_sequence_round_robin(
        frame, sample_column="sample_id", sequence_column="sequence_id", count=20
    )
    second = stable_sequence_round_robin(
        frame.sample(frac=1, random_state=4),
        sample_column="sample_id",
        sequence_column="sequence_id",
        count=20,
    )
    assert first["sample_id"].tolist() == second["sample_id"].tolist()
    locked = build_four_d_subset_manifest(REPO, RUN)
    assert locked == build_four_d_subset_manifest(REPO, RUN)


def test_runtime_subset_is_outcome_blind() -> None:
    manifest = json.loads(
        (RUN / "runtime_full/profile_subset_manifest_4d.json").read_text(encoding="utf-8")
    )
    assert manifest["selection_uses_outcomes"] is False
    assert manifest["selection_uses_runtime"] is False
    source = Path(manifest["source_manifest"])
    columns = set(pd.read_parquet(source).columns)
    assert "native_correct" not in columns
    assert "reranked_correct" not in columns


def test_whole_pipeline_timer_positive() -> None:
    elapsed, value = timed_call_ns(lambda: sum(range(1000)), device="cpu")
    assert elapsed > 0
    assert value == 499500


def test_device_synchronisation_called(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime_common, "synchronize_device", calls.append)
    elapsed, _ = timed_call_ns(lambda: 1, device="mps")
    assert elapsed >= 0
    assert calls == ["mps", "mps"]


def test_memory_sampler_records_child_process(tmp_path: Path) -> None:
    result = run_monitored_command(
        [
            sys.executable,
            "-c",
            "import time; payload=bytearray(2000000); time.sleep(0.04); print(len(payload))",
        ],
        cwd=REPO,
        output_dir=tmp_path,
        route="synthetic",
        mode="test",
        repetition=0,
    )
    samples = pd.read_parquet(result["memory_samples_path"])
    assert result["return_code"] == 0
    assert result["memory_poll_interval_ms"] == 5.0
    assert len(samples) >= 2
    assert samples["recursive_process_count"].max() >= 1
    assert samples["rss_total_bytes"].max() > 0
