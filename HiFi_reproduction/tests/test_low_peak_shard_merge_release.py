from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.grasping.reranking_v1.identity import sha256_file
from test_streaming_scene_shard_pipeline import (  # noqa: E402
    _candidate_merge_fixture,
    _feature_shard,
    _scene_for_bucket,
    _score_merge_fixture,
)
from tools.modular_reranking import low_peak_transaction as low_peak
from tools.modular_reranking.merge_compact_candidate_shards import (
    STAGE_FILES,
    main as merge_candidate_main,
)
from tools.modular_reranking.merge_compact_gqcnn_score_shards import (
    main as merge_gqcnn_main,
)
from tools.modular_reranking.merge_feature_shards import (
    main as merge_features_main,
)
from tools.modular_reranking.release_compact_shard_parquets import (
    main as release_main,
)


def _tree_identity(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _candidate_argv(
    *,
    tmp_root: Path,
    prediction_manifest: Path,
    shard_roots: list[Path],
    output: Path,
) -> list[str]:
    argv = [
        "merge_compact_candidate_shards.py",
        "--prediction-manifest",
        str(prediction_manifest),
        "--output-root",
        str(output),
        "--tmp-root",
        str(tmp_root),
        "--compact-only",
        "--low-peak-release-inputs",
    ]
    for root in shard_roots:
        argv.extend(["--shard-root", str(root)])
    return argv


def test_candidate_low_peak_dry_run_is_zero_write_and_checks_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, prediction, shards = _candidate_merge_fixture(tmp_path)
    output = tmp_root / "candidates" / "merged"
    argv = _candidate_argv(
        tmp_root=tmp_root,
        prediction_manifest=prediction,
        shard_roots=shards,
        output=output,
    )
    before = _tree_identity(run)
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_candidate_main() == 0
    assert _tree_identity(run) == before
    assert not output.exists()
    assert not list((run / "manifests").glob("low_peak_merge/*"))

    monkeypatch.setattr(sys, "argv", [*argv, "--max-run-bytes", "1"])
    with pytest.raises(ValueError, match="storage preflight exceeds budget"):
        merge_candidate_main()
    assert _tree_identity(run) == before


def test_candidate_low_peak_resume_releases_only_stage_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, prediction, shards = _candidate_merge_fixture(tmp_path)
    output = tmp_root / "candidates" / "merged"
    argv = [
        *_candidate_argv(
            tmp_root=tmp_root,
            prediction_manifest=prediction,
            shard_roots=shards,
            output=output,
        ),
        "--execute",
    ]
    original_small_files = {
        path: sha256_file(path)
        for root in shards
        for path in (
            root / "run_config.json",
            root / "summary.csv",
            root / "funnel_labels.jsonl",
        )
    }
    interrupted = False

    def fail_after_first_unlink(name: str) -> None:
        nonlocal interrupted
        if name == "raw:after_unlink" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(low_peak, "fault_point", fail_after_first_unlink)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        merge_candidate_main()
    assert sum((root / STAGE_FILES["raw"]).exists() for root in shards) == 1
    assert not output.exists()

    monkeypatch.setattr(low_peak, "fault_point", lambda _name: None)
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_candidate_main() == 0
    assert output.is_dir()
    assert all(
        not (root / filename).exists()
        for root in shards
        for filename in STAGE_FILES.values()
    )
    assert all(path.is_file() and sha256_file(path) == digest for path, digest in original_small_files.items())
    config = json.loads((output / "run_config.json").read_text())
    assert (
        config["merge_storage"][
            "source_stage_parquets_released_after_verification"
        ]
        is True
    )
    receipt_path = Path(config["merge_storage"]["transaction_receipt"])
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "COMPLETED"
    assert {state["status"] for state in receipt["stages"].values()} == {
        "RELEASED"
    }
    assert len(receipt["released_inputs"]) == 6


def test_candidate_low_peak_rejects_active_verbose_and_out_of_tmp_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, prediction, shards = _candidate_merge_fixture(tmp_path)
    output = tmp_root / "candidates" / "merged"
    active = shards[0] / "sample-still-active"
    active.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        _candidate_argv(
            tmp_root=tmp_root,
            prediction_manifest=prediction,
            shard_roots=shards,
            output=output,
        ),
    )
    with pytest.raises(ValueError, match="active verbose"):
        merge_candidate_main()
    active.rmdir()

    outside = run / "outside.parquet"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="below the current run tmp"):
        low_peak.validate_release_scope(
            output=output,
            tmp_root=tmp_root,
            input_parquets=[outside],
        )


def test_candidate_low_peak_resume_rejects_parameter_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, tmp_root, prediction, shards = _candidate_merge_fixture(tmp_path)
    output = tmp_root / "candidates" / "merged"
    argv = [
        *_candidate_argv(
            tmp_root=tmp_root,
            prediction_manifest=prediction,
            shard_roots=shards,
            output=output,
        ),
        "--execute",
    ]

    def stop_after_verified(name: str) -> None:
        if name == "candidate_raw:after_verified":
            raise RuntimeError("stop after durable verification")

    monkeypatch.setattr(low_peak, "fault_point", stop_after_verified)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="durable verification"):
        merge_candidate_main()
    monkeypatch.setattr(low_peak, "fault_point", lambda _name: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--max-run-bytes",
            str(low_peak.DEFAULT_MAX_RUN_BYTES - 1),
        ],
    )
    with pytest.raises(ValueError, match="parameters or source identities changed"):
        merge_candidate_main()


def test_candidate_low_peak_recovers_publish_receipt_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, tmp_root, prediction, shards = _candidate_merge_fixture(tmp_path)
    output = tmp_root / "candidates" / "merged"
    argv = [
        *_candidate_argv(
            tmp_root=tmp_root,
            prediction_manifest=prediction,
            shard_roots=shards,
            output=output,
        ),
        "--execute",
    ]

    def stop_after_publish(name: str) -> None:
        if name == "candidate:after_publish":
            raise RuntimeError("stop after atomic publish")

    monkeypatch.setattr(low_peak, "fault_point", stop_after_publish)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="atomic publish"):
        merge_candidate_main()
    assert output.is_dir()

    monkeypatch.setattr(low_peak, "fault_point", lambda _name: None)
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_candidate_main() == 0
    receipt = next(
        (tmp_path / "run" / "manifests" / "low_peak_merge").glob(
            "candidate_*.json"
        )
    )
    assert json.loads(receipt.read_text())["status"] == "COMPLETED"


def test_candidate_low_peak_recovers_interrupted_stage_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, tmp_root, prediction, shards = _candidate_merge_fixture(tmp_path)
    output = tmp_root / "candidates" / "merged"
    argv = [
        *_candidate_argv(
            tmp_root=tmp_root,
            prediction_manifest=prediction,
            shard_roots=shards,
            output=output,
        ),
        "--execute",
    ]
    stopped = False

    def stop_during_build(name: str) -> None:
        nonlocal stopped
        if name == "candidate_raw:after_batch" and not stopped:
            stopped = True
            raise RuntimeError("stop during stage build")

    monkeypatch.setattr(low_peak, "fault_point", stop_during_build)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="during stage build"):
        merge_candidate_main()
    assert all((root / STAGE_FILES["raw"]).is_file() for root in shards)
    assert not output.exists()

    monkeypatch.setattr(low_peak, "fault_point", lambda _name: None)
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_candidate_main() == 0
    assert output.is_dir()


def _merge_gqcnn_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, list[Path]]:
    run, tmp_root, prediction, score_paths, _ = _score_merge_fixture(tmp_path)
    output = run / "compact_inputs" / "train" / "gqcnn_scores.parquet"
    argv = [
        "merge_compact_gqcnn_score_shards.py",
        "--split",
        "train",
        "--prediction-manifest",
        str(prediction),
        "--output-path",
        str(output),
        "--tmp-root",
        str(tmp_root),
    ]
    for path in score_paths:
        argv.extend(["--shard-score", str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_gqcnn_main() == 0
    return run, tmp_root, output, score_paths


def test_gqcnn_release_dry_run_and_interruption_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, output, scores = _merge_gqcnn_fixture(
        tmp_path, monkeypatch
    )
    argv = [
        "release_compact_shard_parquets.py",
        "--family",
        "gqcnn",
        "--output",
        str(output),
        "--tmp-root",
        str(tmp_root),
    ]
    for path in scores:
        argv.extend(["--shard", str(path)])
    before = _tree_identity(run)
    monkeypatch.setattr(sys, "argv", argv)
    assert release_main() == 0
    assert _tree_identity(run) == before

    interrupted = False

    def fail_once(name: str) -> None:
        nonlocal interrupted
        if name == "release:after_unlink" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated release interruption")

    monkeypatch.setattr(low_peak, "fault_point", fail_once)
    monkeypatch.setattr(sys, "argv", [*argv, "--execute"])
    with pytest.raises(RuntimeError, match="simulated release interruption"):
        release_main()
    assert sum(path.exists() for path in scores) == 1
    assert all(path.with_suffix(".manifest.json").is_file() for path in scores)

    monkeypatch.setattr(low_peak, "fault_point", lambda _name: None)
    monkeypatch.setattr(sys, "argv", [*argv, "--execute"])
    assert release_main() == 0
    assert output.is_file()
    assert all(not path.exists() for path in scores)
    assert all(path.with_suffix(".manifest.json").is_file() for path in scores)


def test_gqcnn_release_rejects_tampered_source_before_journaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, output, scores = _merge_gqcnn_fixture(
        tmp_path, monkeypatch
    )
    scores[0].write_bytes(b"tampered")
    argv = [
        "release_compact_shard_parquets.py",
        "--family",
        "gqcnn",
        "--output",
        str(output),
        "--tmp-root",
        str(tmp_root),
    ]
    for path in scores:
        argv.extend(["--shard", str(path)])
    before = _tree_identity(run)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="shard changed"):
        release_main()
    assert _tree_identity(run) == before
    assert not list((run / "manifests").glob("low_peak_merge/*"))


def test_feature_release_preserves_manifests_and_rejects_output_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    tmp_root = run / "tmp"
    tmp_root.mkdir(parents=True)
    (run / ".RUN_ACTIVE").write_text("\n")
    roots = [tmp_root / "features" / f"shard_{index}" for index in range(2)]
    for index, root in enumerate(roots):
        _feature_shard(
            root,
            shard_index=index,
            sample_id=f"sample-{index}",
            scene_id=_scene_for_bucket(index),
        )
    output = run / "features" / "train"
    merge_argv = [
        "merge_feature_shards.py",
        "--output-root",
        str(output),
        "--tmp-root",
        str(tmp_root),
        "--split",
        "train",
    ]
    for root in roots:
        merge_argv.extend(["--shard-root", str(root)])
    monkeypatch.setattr(sys, "argv", merge_argv)
    assert merge_features_main() == 0

    release_argv = [
        "release_compact_shard_parquets.py",
        "--family",
        "features",
        "--output",
        str(output),
        "--tmp-root",
        str(tmp_root),
    ]
    for root in roots:
        release_argv.extend(["--shard", str(root)])
    monkeypatch.setattr(sys, "argv", [*release_argv, "--execute"])
    assert release_main() == 0
    assert all((root / "dataset_manifest.json").is_file() for root in roots)
    assert all(
        not (root / filename).exists()
        for root in roots
        for filename in ("per_candidate.parquet", "per_sample.parquet")
    )

    (output / "per_candidate.parquet").write_bytes(b"tampered")
    monkeypatch.setattr(sys, "argv", release_argv)
    with pytest.raises(Exception):
        release_main()
