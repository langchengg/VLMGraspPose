from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from reranking import matrix as matrix_module
from reranking.matrix import MatrixError
from reranking.tests.test_matrix import _initialize_output
from reranking import run_experiment_matrix as runner_module


def _worker_command(output: Path, worker_index: int) -> list[str]:
    source = (
        "from reranking.matrix import run_matrix_train_worker; "
        "run_matrix_train_worker("
        f"{str(output)!r}, worker_index={worker_index}, worker_count=2, "
        "torch_thread_count=1, lease_seconds=60)"
    )
    return [sys.executable, "-c", source]


def _unit_worker_output(tmp_path: Path) -> Path:
    output = _initialize_output(tmp_path)
    config_path = output / "configs" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["matrix_methods"] = ["r2_logistic"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return output


def test_stable_group_partition_keeps_every_method_in_one_worker() -> None:
    worker_count = 7
    observed = {
        matrix_module._worker_for_group("dataset", 42, 3, worker_count=worker_count)
        for _method in range(104)
    }
    assert len(observed) == 1
    assert next(iter(observed)) == matrix_module._worker_for_group(
        "dataset", 42, 3, worker_count=worker_count
    )


def test_duplicate_claim_is_rejected_and_dead_owner_is_recovered(
    tmp_path: Path,
) -> None:
    output = _unit_worker_output(tmp_path)
    protocol = matrix_module.configure_train_worker_protocol(
        output, worker_count=2, torch_thread_count=1
    )
    first = matrix_module._acquire_worker_claim(
        output,
        protocol,
        worker_index=0,
        owner="first",
        lease_seconds=60,
    )
    try:
        with pytest.raises(MatrixError, match="active or unexpired claim"):
            matrix_module._acquire_worker_claim(
                output,
                protocol,
                worker_index=0,
                owner="duplicate",
                lease_seconds=60,
            )
    finally:
        first.release()

    claim_path = matrix_module._claim_path(output, 0)
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    matrix_module._atomic_json(
        claim_path,
        {
            "claim_id": "dead-owner",
            "hostname": socket.gethostname(),
            "pid": 999_999_999,
            "process_start_identity": "definitely-not-live",
            "expires_at": future.isoformat(),
        },
    )
    recovered = matrix_module._acquire_worker_claim(
        output,
        protocol,
        worker_index=0,
        owner="recovered",
        lease_seconds=60,
    )
    recovered.release()
    assert not claim_path.exists()


def test_finalizer_lifecycle_lock_blocks_late_worker(tmp_path: Path) -> None:
    output = _unit_worker_output(tmp_path)
    with matrix_module.train_worker_lifecycle_lock(output, exclusive=True):
        process = subprocess.Popen(
            _worker_command(output, 0),
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(0.5)
        assert process.poll() is None
        assert not (output / "configs" / "train_worker_protocol.json").exists()
    stdout, _ = process.communicate(timeout=120)
    assert process.returncode == 0, stdout


def test_workers_are_concurrent_isolated_resumable_and_finalize_once(
    tmp_path: Path,
) -> None:
    output = _unit_worker_output(tmp_path)
    processes = [
        subprocess.Popen(
            _worker_command(output, worker_index),
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for worker_index in range(2)
    ]
    failures: list[str] = []
    for worker_index, process in enumerate(processes):
        stdout, _ = process.communicate(timeout=120)
        if process.returncode != 0:
            failures.append(f"worker {worker_index}: {stdout}")
    assert failures == []

    assert not list((output / "data").glob("split_assignments*.parquet"))
    assert not (output / "metrics" / "experiment_registry.json").exists()
    assert not (output / "logs" / "stages" / "train").exists()
    assert not list((output / "logs" / "train_workers" / "claims").glob("*.json"))

    manifests = sorted(
        (output / "manifests" / "experiments").glob(
            "dataset=toy_full_post_filter__method=r2_logistic__seed=42__fold=*.json"
        )
    )
    assert len(manifests) == 2
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in manifests}

    matrix_module.run_matrix_train_worker(
        output,
        worker_index=0,
        worker_count=2,
        torch_thread_count=1,
        lease_seconds=60,
    )
    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes()) for path in manifests
    } == before

    outputs = matrix_module.run_matrix_train_finalize(output)
    assert str((output / "metrics" / "experiment_registry.json").resolve()) in outputs
    assert (output / "data" / "split_assignments.parquet").is_file()
    assert (output / "configs" / "formal_matrix.yaml").is_file()
    finalization = json.loads(
        (output / "logs" / "train_workers" / "finalization_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert finalization["verified_core_experiment_count"] == 2
    assert finalization["all_artifact_hashes_verified"] is True


def test_neural_protocol_records_batch_worker_and_thread_identity() -> None:
    spec = next(
        spec for spec in matrix_module._specs("formal") if spec.backend == "mlp"
    )
    protocol = matrix_module._neural_training_protocol(
        spec,
        {
            "matrix_profile": "formal",
            "device": "cpu",
            "train_worker_count": 4,
            "train_torch_thread_count": 1,
            "train_torch_interop_thread_count": 1,
            "neural_batching_policy": "length_bucketed_v1",
        },
    )
    assert protocol is not None
    assert protocol["training_batching_policy"] == "length_bucketed_v1"
    assert protocol["training_batching_shuffle"] is True
    assert protocol["early_stopping_batching_policy"] == "length_bucketed_v1"
    assert protocol["early_stopping_batching_shuffle"] is False
    assert protocol["torch_thread_count"] == 1
    assert protocol["torch_interop_thread_count"] == 1
    assert "train_worker_count" not in protocol
    assert "train_worker_protocol_sha256" not in protocol


def test_acceleration_amendment_is_explicit_and_supersedes_only_neural_manifests(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal-run"
    for directory in (
        "audit",
        "configs",
        "logs/stages/train",
        "manifests/experiments",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        routes=["crog", "modular"],
        pools=["top5", "full"],
        folds=5,
        seeds=[42, 123, 2026],
        device="cpu",
        neural_query_batch_size=128,
        neural_batching_policy="length_bucketed_v1",
        train_worker_count=3,
        train_torch_thread_count=5,
        train_torch_interop_thread_count=1,
        bootstrap_iterations=10_000,
        no_hash_audit=False,
        amend_prelock_device=False,
        amend_prelock_training=True,
    )
    new_semantic = runner_module._semantic_run_config(output, args)
    old_semantic = dict(new_semantic)
    for field in (
        "neural_batching_policy",
        "train_torch_thread_count",
        "train_torch_interop_thread_count",
    ):
        old_semantic.pop(field)
    old_identity = runner_module._json_identity_sha256(old_semantic)
    old_config = {
        "schema_version": runner_module.RUN_CONFIG_SCHEMA_VERSION,
        **old_semantic,
        "semantic_config": old_semantic,
        "semantic_config_sha256": old_identity,
    }
    (output / "configs" / "run_config.json").write_text(
        json.dumps(old_config), encoding="utf-8"
    )
    (output / "audit" / "full_list_neural_acceleration_benchmark.json").write_text(
        json.dumps(
            {
                "kind": "full_list_neural_acceleration_benchmark",
                "status": "COMPLETE",
                "formal_training_manifests_written": False,
                "recommendation": {
                    "batching": "deterministic length-aware shuffled bucketing",
                    "cpu_intraop_threads": 5,
                    "cpu_interop_threads": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "logs" / "stages" / "train" / "status.json").write_text(
        json.dumps({"status": "RUNNING"}), encoding="utf-8"
    )
    neural = output / "manifests" / "experiments" / "neural.json"
    failed_neural = output / "manifests" / "experiments" / "failed_neural.json"
    non_neural = output / "manifests" / "experiments" / "logistic.json"
    neural_artifact = output / "checkpoints" / "neural.pt"
    neural_artifact.parent.mkdir(parents=True, exist_ok=True)
    neural_artifact.write_bytes(b"old-neural-checkpoint")
    neural_artifact_sha256 = hashlib.sha256(neural_artifact.read_bytes()).hexdigest()
    neural.write_text(
        json.dumps(
            {
                "experiment_id": "neural",
                "status": "COMPLETE",
                "spec": {"backend": "mlp"},
                "artifacts": [str(neural_artifact.resolve())],
                "artifact_sha256": {
                    str(neural_artifact.resolve()): neural_artifact_sha256
                },
            }
        ),
        encoding="utf-8",
    )
    failed_neural.write_text(
        json.dumps(
            {
                "experiment_id": "failed_neural",
                "status": "FAILED",
                "spec": {"backend": "gnn"},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    non_neural.write_text(
        json.dumps({"experiment_id": "logistic", "spec": {"backend": "logistic"}}),
        encoding="utf-8",
    )

    amended = runner_module._freeze_or_validate_run_config(output, args)

    assert amended["semantic_config"]["neural_batching_policy"] == (
        "length_bucketed_v1"
    )
    assert amended["semantic_config"]["train_torch_thread_count"] == 5
    assert non_neural.is_file()
    # Live files remain until an amended experiment atomically replaces them;
    # the new identity makes the old neural manifest non-resumable meanwhile.
    assert neural.is_file()
    assert (
        output
        / "manifests"
        / f"superseded_pre_acceleration_{old_identity[:12]}"
        / "neural.json"
    ).is_file()
    assert (output / "logs" / "stages" / "train" / "status.json").is_file()
    audit = json.loads(
        (output / "audit" / "prelock_acceleration_amendment.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["retained_non_neural_manifest_count"] == 1
    assert audit["superseded_neural_manifest_count"] == 2
    neural_inventory = next(
        row
        for row in audit["superseded_neural_manifests"]
        if row["experiment_id"] == "neural"
    )
    archived_artifact = Path(neural_inventory["archived_artifacts"][0]["archive_path"])
    assert archived_artifact.read_bytes() == b"old-neural-checkpoint"
    assert neural_artifact.read_bytes() == b"old-neural-checkpoint"
    assert failed_neural.is_file()
    assert (
        output
        / "manifests"
        / f"superseded_pre_acceleration_{old_identity[:12]}"
        / "failed_neural.json"
    ).is_file()
    assert audit["prior_stage_receipts_invalidated"] is True

    audit_bytes = (
        output / "audit" / "prelock_acceleration_amendment.json"
    ).read_bytes()
    archive_bytes = archived_artifact.read_bytes()
    amended_again = runner_module._freeze_or_validate_run_config(output, args)
    assert amended_again["semantic_config_sha256"] == amended["semantic_config_sha256"]
    assert (
        output / "audit" / "prelock_acceleration_amendment.json"
    ).read_bytes() == audit_bytes
    assert archived_artifact.read_bytes() == archive_bytes
