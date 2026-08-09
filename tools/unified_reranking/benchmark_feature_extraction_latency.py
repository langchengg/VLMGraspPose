"""Benchmark true T3 feature extraction on a fixed Validation-only subset."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


ROUTES = ("crog", "g1", "c1")
DEFAULT_SAMPLE_LIMIT = 128
DEFAULT_TAG = "latency_benchmark_128"


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _execute(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    """Run one extractor while polling child RSS; return measured wall time."""

    started = time.perf_counter()
    process = subprocess.Popen(
        tuple(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_rss_kib = 0.0
    while process.poll() is None:
        snapshot = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(process.pid)),
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            peak_rss_kib = max(peak_rss_kib, float(snapshot.stdout.strip()))
        except ValueError:
            pass
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    return {
        "returncode": int(process.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_rss_kib / 1024.0,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _validate_previous(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    previous = load_verified_json(path, name="feature extraction benchmark")
    unsigned = dict(previous)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("feature extraction benchmark content hash mismatch")
    if previous.get("source_signature_sha256") != signature:
        raise RuntimeError("immutable feature extraction benchmark source drift")
    verify_artifact_records_recursive(
        {"sources": previous.get("sources"), "artifacts": previous.get("artifacts")},
        name="feature extraction benchmark",
        require_at_least_one=True,
    )
    output_record = dict(previous.get("artifacts", {})).get("tagged_output_manifest")
    if not isinstance(output_record, Mapping):
        raise RuntimeError("feature extraction benchmark output manifest is missing")
    output_path = Path(str(output_record.get("path", ""))).resolve()
    output = load_verified_json(output_path, name="tagged T3 benchmark output")
    verify_artifact_records_recursive(
        output.get("artifacts"),
        name="tagged T3 benchmark outputs",
        require_at_least_one=True,
    )
    return previous


def run(
    run_dir: Path,
    fair_source: Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    tag: str = DEFAULT_TAG,
    device: str = "cpu",
    batch_size: int = 16,
    chunk_size: int = 128,
    execute: Callable[[Sequence[str], Path], Mapping[str, Any]] = _execute,
) -> dict[str, Any]:
    """Produce one immutable, source-bound Validation extraction benchmark."""

    root = run_dir.resolve()
    fair = fair_source.resolve()
    if sample_limit != DEFAULT_SAMPLE_LIMIT or tag != DEFAULT_TAG:
        raise ValueError(
            "formal telemetry uses the fixed 128-sample benchmark contract"
        )
    if device not in {"cpu", "mps"} or batch_size <= 0 or chunk_size <= 0:
        raise ValueError("invalid fixed-subset benchmark execution configuration")
    paired_path = root / "01_manifests" / "paired_validation.parquet"
    paired = pd.read_parquet(paired_path, columns=["sample_id"])
    sample_ids = paired["sample_id"].astype(str).head(sample_limit).tolist()
    if len(sample_ids) != sample_limit or len(set(sample_ids)) != sample_limit:
        raise RuntimeError(
            "Validation denominator cannot supply the fixed benchmark subset"
        )
    candidate_records: dict[str, dict[str, str]] = {}
    candidate_rows: dict[str, int] = {}
    selected_ids = set(sample_ids)
    for route in ROUTES:
        path = root / "02_candidates" / f"{route}_validation_top5.parquet"
        frame = pd.read_parquet(path, columns=["sample_id", "candidate_id"])
        subset = frame.loc[frame["sample_id"].astype(str).isin(selected_ids)]
        candidate_records[route] = _record(path)
        candidate_rows[route] = int(len(subset))
    total_candidates = sum(candidate_rows.values())
    if total_candidates <= 0:
        raise RuntimeError("fixed Validation benchmark subset has no candidates")

    component_specs: list[tuple[str, str, str, Path, str]] = []
    for route in ROUTES:
        component_specs.extend(
            [
                (
                    f"common/{route}",
                    route,
                    "common",
                    root
                    / "03_features"
                    / "common"
                    / f"{route}_validation"
                    / "feature_manifest.json",
                    "candidate_features",
                ),
                (
                    f"rgb/{route}",
                    route,
                    "rgb",
                    root
                    / "03_features"
                    / "rgb"
                    / f"{route}_validation"
                    / "feature_manifest.json",
                    "candidate_rows",
                ),
                (
                    f"T1_native/{route}",
                    route,
                    "T1_native",
                    root
                    / "03_features"
                    / "tracks"
                    / "T1_native"
                    / f"{route}_validation"
                    / "feature_manifest.json",
                    "candidate_rows",
                ),
                (
                    f"T2_matched_common/{route}",
                    route,
                    "T2_matched_common",
                    root
                    / "03_features"
                    / "tracks"
                    / "T2_matched_common"
                    / f"{route}_validation"
                    / "feature_manifest.json",
                    "candidate_rows",
                ),
            ]
        )
    for route in ("g1", "c1"):
        component_specs.append(
            (
                f"backend_maps/{route}",
                route,
                "backend_maps",
                root
                / "03_features"
                / "backend_maps"
                / f"{route}_validation"
                / "feature_manifest.json",
                "candidate_rows",
            )
        )
    component_records: dict[str, dict[str, str]] = {}
    component_measurements: list[dict[str, Any]] = []
    for name, route, component, manifest_path, row_field in component_specs:
        manifest = load_verified_json(
            manifest_path, name=f"Validation extraction component {name}"
        )
        verify_artifact_records_recursive(
            {
                "sources": manifest.get("sources"),
                "artifacts": manifest.get("artifacts"),
                "artifact": manifest.get("artifact"),
            },
            name=f"Validation extraction component {name}",
            require_at_least_one=True,
        )
        latency = manifest.get("feature_extraction_latency_ms")
        peak_memory = manifest.get(
            "feature_extraction_peak_memory_mb", manifest.get("peak_memory_mb")
        )
        rows = manifest.get(row_field)
        if (
            not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0
            or not isinstance(peak_memory, (int, float))
            or not math.isfinite(float(peak_memory))
            or float(peak_memory) <= 0
            or not isinstance(rows, int)
            or rows <= 0
        ):
            raise RuntimeError(
                f"Validation extraction component telemetry is incomplete: {name}"
            )
        component_records[name] = _record(manifest_path)
        component_measurements.append(
            {
                "name": name,
                "route": route,
                "component_or_track": component,
                "measurement_scope": "full_validation_persisted_extraction",
                "candidate_rows": rows,
                "feature_extraction_latency_ms": float(latency),
                "peak_memory_mb": float(peak_memory),
                "manifest": component_records[name],
            }
        )

    extractor = (
        ROOT / "tools" / "unified_reranking" / "extract_tri_backend_dense_features.py"
    )
    sources: dict[str, Any] = {
        "paired_validation": _record(paired_path),
        "candidate_pools": candidate_records,
        "extractor_tool": _record(extractor),
        "benchmark_tool": _record(Path(__file__)),
        "component_manifests": component_records,
    }
    configuration = {
        "split": "validation",
        "sample_limit": sample_limit,
        "tag": tag,
        "device": device,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "sample_identity_sha256": canonical_sha256(sample_ids),
        "candidate_rows_by_route": candidate_rows,
        "total_candidate_rows": total_candidates,
        "candidate_test_labels_read": False,
        "component_inventory": [value[0] for value in component_specs],
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    destination = (
        root / "07_validation" / "telemetry" / "feature_extraction_benchmark.json"
    )
    previous = _validate_previous(destination, signature)
    if previous is not None:
        return previous

    command = (
        sys.executable,
        "-m",
        "tools.unified_reranking.extract_tri_backend_dense_features",
        "--run-dir",
        str(root),
        "--fair-test-source",
        str(fair),
        "--split",
        "validation",
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--chunk-size",
        str(chunk_size),
        "--limit",
        str(sample_limit),
        "--tag",
        tag,
    )
    measured = dict(execute(command, ROOT))
    elapsed = measured.get("elapsed_seconds")
    peak_memory = measured.get("peak_memory_mb")
    if (
        measured.get("returncode") != 0
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0
        or not isinstance(peak_memory, (int, float))
        or not math.isfinite(float(peak_memory))
        or float(peak_memory) <= 0
    ):
        raise RuntimeError(
            f"feature extraction benchmark failed: {measured.get('stderr_tail', '')}"
        )
    output_manifest = (
        root
        / "03_features"
        / "tri_backend_dense"
        / f"validation_{tag}"
        / "manifest.json"
    )
    output = load_verified_json(output_manifest, name="tagged T3 benchmark output")
    if (
        output.get("split") != "validation"
        or output.get("tag") != tag
        or output.get("candidate_test_labels_read") is not None
    ):
        raise RuntimeError(
            "tagged benchmark output violates the Validation-only contract"
        )
    verify_artifact_records_recursive(
        output.get("artifacts"),
        name="tagged T3 benchmark outputs",
        require_at_least_one=True,
    )
    payload: dict[str, Any] = {
        "status": "COMPLETE",
        "schema_version": 1,
        "analysis": "validation_feature_extraction_runtime",
        "source_signature_sha256": signature,
        "configuration": configuration,
        "command": list(command),
        "feature_extraction_elapsed_seconds": float(elapsed),
        "feature_extraction_latency_ms": float(elapsed) * 1000.0 / total_candidates,
        "feature_extraction_latency_ms_per_sample": float(elapsed)
        * 1000.0
        / sample_limit,
        "peak_memory_mb": float(peak_memory),
        "measurement_protocol": (
            "fresh tagged T3 extraction wall time and polled child RSS on the first "
            "128 Validation denominator samples; excludes matrix feature loading/preprocessing"
        ),
        "component_measurements": component_measurements,
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": {"tagged_output_manifest": _record(output_manifest)},
    }
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fair-test-source", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P7",
        substage="validation_feature_extraction_benchmark",
        evidence_track="T3_tri_backend",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(
            run_dir,
            args.fair_test_source.expanduser().resolve(),
            device=args.device,
        )
        manifest = (
            run_dir
            / "07_validation"
            / "telemetry"
            / "feature_extraction_benchmark.json"
        )
        state["artifact_path"] = str(manifest.resolve())
        state["artifact_sha256"] = sha256_file(manifest)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
