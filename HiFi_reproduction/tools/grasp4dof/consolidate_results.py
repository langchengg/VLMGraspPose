#!/usr/bin/env python3
"""Consolidate immutable per-method outputs into dissertation result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd


METHOD_RELATIVES = {
    "R0": "formal_test/R0",
    "G0": "formal_test/G0",
    "G1": "formal_test/G1",
    "C0": "formal_test/C0",
    "C1": "formal_test/C1",
    "A0": "formal_test/A0",
    "G0-O": "oracle/G0-O",
    "G1-O": "oracle/G1-O",
    "C0-O": "oracle/C0-O",
    "C1-O": "oracle/C1-O",
    "A0-O": "oracle/A0-O",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be ID=/absolute/or/relative/path")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("method ID cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method-dir", action="append", type=parse_method, required=True)
    parser.add_argument("--primary-method-id", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    method_dirs = dict(args.method_dir)
    if len(method_dirs) != len(args.method_dir):
        raise ValueError("duplicate method IDs")
    if args.primary_method_id not in method_dirs:
        raise ValueError("primary method ID is not present")
    expected_methods = set(METHOD_RELATIVES)
    if set(method_dirs) != expected_methods:
        raise ValueError("consolidation requires the exact 11-method formal matrix")
    lock = run / "manifests/experiment_lock.json"
    frozen_alias = run / "frozen_4dof_backends_experiment_manifest.json"
    if not lock.is_file():
        raise FileNotFoundError(lock)
    if not frozen_alias.is_file() or frozen_alias.read_bytes() != lock.read_bytes():
        raise ValueError("frozen experiment manifest must pre-exist and match the lock")
    for method_id, relative in METHOD_RELATIVES.items():
        expected_directory = (run / relative).resolve()
        if method_dirs[method_id] != expected_directory:
            raise ValueError(
                f"{method_id}: method directory is not the canonical locked output"
            )
    output_names = (
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
        "formal_test_results.csv",
        "per_method_metrics.csv",
        "oracle_results.csv",
        "common_subset_comparison.csv",
        "training_curves.csv",
        "runtime_metrics.json",
        "memory_metrics.json",
        "results_bundle.json",
    )
    existing_outputs = [name for name in output_names if (run / name).exists()]
    if existing_outputs:
        raise FileExistsError(f"consolidated outputs already exist: {existing_outputs}")

    sample_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    runtime: dict[str, object] = {}
    memory: dict[str, object] = {}
    sources: dict[str, object] = {}
    reference_ids: set[str] | None = None
    for method_id, directory in method_dirs.items():
        sample_path = directory / "per_sample_predictions.parquet"
        candidate_path = directory / "per_candidate_predictions.parquet"
        metrics_path = directory / "metrics.json"
        run_config_path = directory / "run_config.json"
        complete_path = directory / "COMPLETE.json"
        if not all(
            path.is_file()
            for path in (
                sample_path,
                candidate_path,
                metrics_path,
                run_config_path,
                complete_path,
            )
        ):
            raise FileNotFoundError(f"incomplete method directory: {directory}")
        samples = pd.read_parquet(sample_path)
        candidates = pd.read_parquet(candidate_path)
        if len(samples) != args.expected_count or samples.sample_id.duplicated().any():
            raise ValueError(f"{method_id}: sample count/identity failure")
        ids = set(samples.sample_id.astype(str))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{method_id}: sample IDs do not align")
        if "method" in samples:
            samples.rename(columns={"method": "method_name"}, inplace=True)
        if "method" in candidates:
            candidates.rename(columns={"method": "method_name"}, inplace=True)
        samples.insert(0, "method_id", method_id)
        samples.insert(1, "method", method_id)
        candidates.insert(0, "method_id", method_id)
        candidates.insert(1, "method", method_id)
        sample_frames.append(samples)
        candidate_frames.append(candidates)
        row = json.loads(metrics_path.read_text())
        row["method_id"] = method_id
        metrics_rows.append(row)
        runtime_path, memory_path = directory / "runtime_metrics.json", directory / "memory_metrics.json"
        runtime[method_id] = (
            json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
        )
        memory[method_id] = (
            json.loads(memory_path.read_text()) if memory_path.is_file() else None
        )
        sources[method_id] = {
            "directory": str(directory),
            "per_sample_sha256": sha256_file(sample_path),
            "per_candidate_sha256": sha256_file(candidate_path),
            "metrics_sha256": sha256_file(metrics_path),
            "run_config_sha256": sha256_file(run_config_path),
            "complete_sha256": sha256_file(complete_path),
            "complete_filename": complete_path.name,
        }
    samples_all = pd.concat(sample_frames, ignore_index=True)
    candidates_all = pd.concat(candidate_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    primary = metrics.loc[metrics.method_id == args.primary_method_id].iloc[0].to_dict()
    primary["method_id"] = "locked_primary_4dof_backend"
    primary["source_method_id"] = args.primary_method_id
    result_table = pd.concat([metrics, pd.DataFrame([primary])], ignore_index=True)

    curve_frames: list[pd.DataFrame] = []
    for method_id, backend_dir in (("G1", "grconvnet"), ("C1", "ggcnn2")):
        for curve_path in sorted(
            (run / backend_dir / "finetuned/grid").glob("*/training_curves.csv")
        ):
            curve = pd.read_csv(curve_path)
            curve.insert(0, "training_job", curve_path.parent.name)
            curve.insert(0, "method_id", method_id)
            curve_frames.append(curve)
    if not curve_frames:
        raise FileNotFoundError("no fine-tuning curves were found")
    training_curves = pd.concat(curve_frames, ignore_index=True)

    oracle_rows = []
    for method_id in ("G0", "G1", "C0", "C1", "A0"):
        oracle_id = f"{method_id}-O"
        if method_id not in method_dirs or oracle_id not in method_dirs:
            continue
        predicted = metrics.loc[metrics.method_id == method_id].iloc[0]
        oracle = metrics.loc[metrics.method_id == oracle_id].iloc[0]
        oracle_rows.append(
            {
                "method_id": method_id,
                "pred_mask_j_at_1": predicted.j_at_1,
                "gt_mask_j_at_1": oracle.j_at_1,
                "j_at_1_gap": oracle.j_at_1 - predicted.j_at_1,
                "pred_mask_oracle": predicted.candidate_pool_oracle,
                "gt_mask_oracle": oracle.candidate_pool_oracle,
                "candidate_oracle_gap": oracle.candidate_pool_oracle
                - predicted.candidate_pool_oracle,
            }
        )

    outputs = {
        "per_sample_predictions.parquet": samples_all,
        "per_candidate_predictions.parquet": candidates_all,
        "formal_test_results.csv": result_table,
        "per_method_metrics.csv": result_table,
        "oracle_results.csv": pd.DataFrame(oracle_rows),
        "common_subset_comparison.csv": metrics,
        "training_curves.csv": training_curves,
    }
    stage_parent = run / "tmp"
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="consolidate-", dir=stage_parent))
    committed: list[Path] = []
    try:
        for name, frame in outputs.items():
            path = stage / name
            if path.suffix == ".parquet":
                frame.to_parquet(path, compression="zstd", index=False)
            else:
                frame.to_csv(path, index=False)
        write_json(stage / "runtime_metrics.json", runtime)
        write_json(stage / "memory_metrics.json", memory)
        bundle = {
            "status": "COMPLETE",
            "expected_sample_count_per_method": args.expected_count,
            "method_count": len(method_dirs),
            "locked_primary_method_id": args.primary_method_id,
            "sources": sources,
            "outputs": {
                name: {"path": str(run / name), "sha256": sha256_file(stage / name)}
                for name in outputs
            },
        }
        write_json(stage / "results_bundle.json", bundle)
        # Recheck the two immutable boundaries immediately before committing.
        if not lock.is_file() or not frozen_alias.is_file() or frozen_alias.read_bytes() != lock.read_bytes():
            raise RuntimeError("experiment lock changed during consolidation")
        if any((run / name).exists() for name in output_names):
            raise FileExistsError("a consolidated output appeared during staging")
        for name in output_names:
            destination = run / name
            os.replace(stage / name, destination)
            committed.append(destination)
    except BaseException:
        # Every destination was proven absent before this transaction. Roll
        # back only files created by this invocation, leaving inputs untouched.
        for destination in reversed(committed):
            destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps(bundle, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
