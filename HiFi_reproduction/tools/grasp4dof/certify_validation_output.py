#!/usr/bin/env python3
"""Issue a content-addressed sidecar for a completed validation output.

The original completion marker is never modified. This is intended for runs
that finished while the stronger schema-v2 lifecycle gate was being added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.results import (  # noqa: E402
    assert_aggregate_matches_sample_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected-count", type=int, default=3778)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.relative_to(run)
    config = args.config.expanduser().resolve()
    config.relative_to(run)
    destination = output / "VALIDATED_COMPLETE.json"
    if destination.exists():
        raise FileExistsError(destination)
    original = output / "COMPLETE.json"
    original_value = _json(original)
    if (
        original_value.get("status") != "COMPLETE"
        or int(original_value.get("sample_count", -1)) != args.expected_count
    ):
        raise ValueError("legacy completion marker is not complete")
    names = (
        "metrics.json",
        "runtime_metrics.json",
        "memory_metrics.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
        "run_config.json",
    )
    if not all((output / name).is_file() for name in names):
        raise FileNotFoundError("validation artifact set is incomplete")
    run_config = _json(output / "run_config.json")
    config_sha256 = _sha256(config)
    sample_manifest = run / "manifests/validation_samples.parquet"
    label_manifest = run / "manifests/validation_labels.parquet"
    if (
        run_config.get("method_id") != args.method
        or run_config.get("split") != "validation"
        or run_config.get("oracle") is not args.oracle
        or int(run_config.get("sample_count", -1)) != args.expected_count
        or run_config.get("config_sha256") != config_sha256
        or run_config.get("samples_manifest_sha256") != _sha256(sample_manifest)
        or run_config.get("labels_manifest_sha256") != _sha256(label_manifest)
        or run_config.get("per_sample_sha256")
        != _sha256(output / "per_sample_predictions.parquet")
        or run_config.get("per_candidate_sha256")
        != _sha256(output / "per_candidate_predictions.parquet")
    ):
        raise ValueError("validation run-config provenance mismatch")
    sample_table = pq.read_table(output / "per_sample_predictions.parquet")
    expected_ids = set(
        pq.read_table(sample_manifest, columns=["sample_id"])
        .column("sample_id")
        .to_pylist()
    )
    observed_ids = set(sample_table.column("sample_id").to_pylist())
    if (
        sample_table.num_rows != args.expected_count
        or len(observed_ids) != args.expected_count
        or observed_ids != expected_ids
    ):
        raise ValueError("validation sample identity/coverage mismatch")
    pq.ParquetFile(output / "per_candidate_predictions.parquet")
    metrics = _json(output / "metrics.json")
    assert_aggregate_matches_sample_rows(sample_table.to_pylist(), metrics)
    certificate = {
        "schema_version": 2,
        "status": "COMPLETE",
        "method_id": args.method,
        "split": "validation",
        "oracle": args.oracle,
        "sample_count": args.expected_count,
        "config_sha256": config_sha256,
        "legacy_complete_sha256": _sha256(original),
        "artifacts": {name: _sha256(output / name) for name in names},
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(certificate, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "COMPLETE", "certificate": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
