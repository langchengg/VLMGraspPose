from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from unified_reranking.hashing import sha256_file
from tools.unified_reranking.benchmark_feature_extraction_latency import run


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _fixture(run_dir: Path) -> None:
    samples = [f"sample-{index:03d}" for index in range(128)]
    paired = run_dir / "01_manifests" / "paired_validation.parquet"
    paired.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": samples}).to_parquet(paired, index=False)
    candidates = run_dir / "02_candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    for route in ("crog", "g1", "c1"):
        pd.DataFrame(
            {
                "sample_id": samples,
                "candidate_id": [f"{route}-{index}" for index in range(128)],
            }
        ).to_parquet(candidates / f"{route}_validation_top5.parquet", index=False)
        component_paths = (
            (
                run_dir
                / "03_features"
                / "common"
                / f"{route}_validation"
                / "feature_manifest.json",
                "candidate_features",
            ),
            (
                run_dir
                / "03_features"
                / "rgb"
                / f"{route}_validation"
                / "feature_manifest.json",
                "candidate_rows",
            ),
            (
                run_dir
                / "03_features"
                / "tracks"
                / "T1_native"
                / f"{route}_validation"
                / "feature_manifest.json",
                "candidate_rows",
            ),
            (
                run_dir
                / "03_features"
                / "tracks"
                / "T2_matched_common"
                / f"{route}_validation"
                / "feature_manifest.json",
                "candidate_rows",
            ),
        )
        if route in {"g1", "c1"}:
            component_paths += (
                (
                    run_dir
                    / "03_features"
                    / "backend_maps"
                    / f"{route}_validation"
                    / "feature_manifest.json",
                    "candidate_rows",
                ),
            )
        for manifest, row_field in component_paths:
            artifact = manifest.with_name("candidate_features.bin")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{route}:{manifest.parent}".encode("utf-8"))
            payload = {
                "status": "COMPLETE",
                row_field: 128,
                "feature_extraction_latency_ms": 0.5,
                "feature_extraction_peak_memory_mb": 12.0,
                "artifact": _record(artifact),
            }
            manifest.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )


def _fake_executor(run_dir: Path, *, peak_memory_mb: float = 64.0):
    def execute(command: Sequence[str], _cwd: Path) -> dict[str, Any]:
        tag = command[command.index("--tag") + 1]
        output = run_dir / "03_features" / "tri_backend_dense" / f"validation_{tag}"
        artifacts = {}
        for route in ("crog", "g1", "c1"):
            path = output / route / "candidate_features.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"sample_id": ["synthetic"], "candidate_id": [route]}
            ).to_parquet(path, index=False)
            artifacts[route] = _record(path)
        manifest = {
            "status": "COMPLETE",
            "split": "validation",
            "tag": tag,
            "candidate_test_labels_read": None,
            "artifacts": artifacts,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "returncode": 0,
            "elapsed_seconds": 1.28,
            "peak_memory_mb": peak_memory_mb,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    return execute


def test_validation_feature_extraction_benchmark_is_source_bound(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    fair = tmp_path / "fair"
    fair.mkdir()
    result = run(tmp_path, fair, execute=_fake_executor(tmp_path))
    assert result["status"] == "COMPLETE"
    assert result["candidate_test_labels_read"] is False
    assert result["configuration"]["split"] == "validation"
    assert result["configuration"]["sample_limit"] == 128
    assert result["feature_extraction_latency_ms"] == pytest.approx(
        1.28 * 1000.0 / (128 * 3)
    )
    assert result["peak_memory_mb"] == pytest.approx(64.0)
    assert len(result["component_measurements"]) == 14

    pool = tmp_path / "02_candidates" / "g1_validation_top5.parquet"
    drifted = pd.read_parquet(pool)
    drifted.loc[0, "candidate_id"] = "drifted"
    drifted.to_parquet(pool, index=False)
    with pytest.raises(RuntimeError, match="source drift"):
        run(tmp_path, fair, execute=_fake_executor(tmp_path))


def test_validation_feature_extraction_benchmark_rejects_null_peak_memory(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    fair = tmp_path / "fair"
    fair.mkdir()
    with pytest.raises(RuntimeError, match="benchmark failed"):
        run(
            tmp_path,
            fair,
            execute=_fake_executor(tmp_path, peak_memory_mb=0.0),
        )
