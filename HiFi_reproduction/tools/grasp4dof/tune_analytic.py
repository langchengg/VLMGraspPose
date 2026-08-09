#!/usr/bin/env python3
"""Validation-only staged tuning for the A0 analytic grasp backend.

The protocol deliberately performs coordinate search instead of the Cartesian
product.  Every evaluated configuration is executed through ``run_method.py``
so the ordinary per-sample and per-candidate Parquet artifacts remain the
source of truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends import AnalyticGraspConfig  # noqa: E402


SCHEMA_VERSION = "grasp4dof_analytic_coordinate_search_v1"
RESULT_COLUMNS = (
    "stage",
    "axis",
    "axis_value",
    "config_id",
    "config_sha256",
    "runner_config_sha256",
    "sample_count",
    "corrected_j_at_1",
    "corrected_j_at_5",
    "recall_at_5",
    "no_grasp_rate",
    "p50_latency_seconds",
    "p95_latency_seconds",
    "total_wall_seconds",
    "complexity",
    "output_dir",
    "resumed",
    "selected",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Return the stable semantic hash used to identify an A0 configuration."""

    return hashlib.sha256(_canonical_bytes(dict(config))).hexdigest()


def _id_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in RESULT_COLUMNS})
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_within(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.relative_to(root.expanduser().resolve())
    return resolved


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if existing != dict(value):
            raise RuntimeError(f"refusing to change preregistered artifact: {path}")
        return
    _atomic_json(path, value)


def _manifest_sample_ids(path: Path) -> tuple[list[str], list[str]]:
    table = pq.read_table(path, columns=["sample_id", "split"])
    sample_ids = [str(value) for value in table.column("sample_id").to_pylist()]
    splits = [str(value) for value in table.column("split").to_pylist()]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("validation manifest sample IDs must be non-empty and unique")
    if any(value not in {"val", "validation"} for value in splits):
        raise ValueError("tuner accepts only a validation deployment manifest")
    return sample_ids, splits


def _pilot_sample_ids(path: Path) -> list[str]:
    payload = _load_json(path)
    rows = payload.get("samples")
    if not isinstance(rows, list):
        raise ValueError("pilot manifest must contain a samples list")
    sample_ids = [str(row["sample_id"]) for row in rows if isinstance(row, dict)]
    if len(sample_ids) != len(rows) or not sample_ids:
        raise ValueError("each pilot row must contain a sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("pilot sample IDs must be unique")
    declared_count = payload.get("sample_count")
    if declared_count is not None and int(declared_count) != len(sample_ids):
        raise ValueError("pilot sample_count does not match the samples list")
    return sample_ids


def _deduplicated_axis_values(
    base_config: Mapping[str, Any], search_space: Mapping[str, Iterable[Any]]
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for axis, declared in search_space.items():
        if axis not in base_config:
            raise ValueError(f"search axis is absent from AnalyticGraspConfig: {axis}")
        values: list[Any] = []
        for value in (base_config[axis], *tuple(declared)):
            if value not in values:
                values.append(value)
        if len(values) < 2:
            raise ValueError(f"search axis has no alternative value: {axis}")
        result[axis] = values
    return result


def coordinate_trial_upper_bound(
    base_config: Mapping[str, Any], search_space: Mapping[str, Iterable[Any]]
) -> int:
    """Maximum pilot executions for one sequential coordinate-search pass."""

    axes = _deduplicated_axis_values(base_config, search_space)
    return 1 + sum(len(values) - 1 for values in axes.values())


def _complexity(config: Mapping[str, Any], base: Mapping[str, Any], axes: Sequence[str]) -> int:
    return sum(config[name] != base[name] for name in axes)


def _finite_metric(record: Mapping[str, Any], name: str, *, high_is_good: bool) -> float:
    value = float(record.get(name, math.nan))
    if math.isfinite(value):
        return value
    return -math.inf if high_is_good else math.inf


def selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Protocol ordering: J@1, J@5, no-grasp, runtime, simplicity."""

    return (
        -_finite_metric(record, "corrected_j_at_1", high_is_good=True),
        -_finite_metric(record, "corrected_j_at_5", high_is_good=True),
        _finite_metric(record, "no_grasp_rate", high_is_good=False),
        _finite_metric(record, "p95_latency_seconds", high_is_good=False),
        int(record["complexity"]),
        str(record["config_sha256"]),
    )


class AnalyticTuner:
    def __init__(
        self,
        *,
        run_dir: Path,
        runner: Path,
        python: Path,
        pilot_manifest: Path,
        max_full_configs: int,
        full_workers: int,
    ) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.project_root = PROJECT_ROOT.resolve()
        self.runner = runner.expanduser().resolve()
        # Preserve a virtual-environment interpreter symlink.  Resolving it to
        # the Homebrew/system target silently drops the venv's site-packages.
        self.python = python.expanduser().absolute()
        self.manifest_dir = self.run_dir / "manifests"
        self.validation_samples = self.manifest_dir / "validation_samples.parquet"
        self.validation_labels = self.manifest_dir / "validation_labels.parquet"
        self.pilot_manifest = pilot_manifest.expanduser().resolve()
        self.analytic_root = self.run_dir / "analytic"
        self.search_root = self.run_dir / "validation" / "analytic_search"
        self.config_root = self.search_root / "configs"
        self.log_root = self.search_root / "logs"
        self.results_path = self.analytic_root / "validation_search_results.csv"
        self.max_full_configs = int(max_full_configs)
        self.full_workers = int(full_workers)
        if not 1 <= self.max_full_configs <= 3:
            raise ValueError("max_full_configs must be between 1 and 3")
        if not 1 <= self.full_workers <= 3:
            raise ValueError("full_workers must be between 1 and 3")
        for required in (
            self.runner,
            self.python,
            self.validation_samples,
            self.validation_labels,
            self.pilot_manifest,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        _assert_within(self.pilot_manifest, self.manifest_dir)
        if "test" in self.pilot_manifest.name.lower():
            raise ValueError("test manifests are forbidden during analytic tuning")
        self.validation_ids, _ = _manifest_sample_ids(self.validation_samples)
        self.pilot_ids = _pilot_sample_ids(self.pilot_manifest)
        missing = sorted(set(self.pilot_ids) - set(self.validation_ids))
        if missing:
            raise ValueError(f"pilot IDs are outside validation: {missing[:3]}")
        self.base_config = asdict(AnalyticGraspConfig())
        self.axes = _deduplicated_axis_values(
            self.base_config, AnalyticGraspConfig.validation_search_space()
        )
        self.records: list[dict[str, Any]] = []
        self.pilot_cache: dict[str, dict[str, Any]] = {}
        self.configs: dict[str, dict[str, Any]] = {}

    def preregister(self) -> None:
        self.analytic_root.mkdir(parents=True, exist_ok=True)
        self.search_root.mkdir(parents=True, exist_ok=True)
        for path in (self.analytic_root, self.search_root):
            _assert_within(path, self.run_dir)
            if path.is_symlink():
                raise RuntimeError(f"artifact root must not be a symlink: {path}")
        protocol = {
            "schema_version": SCHEMA_VERSION,
            "split": "validation",
            "method": "A0",
            "device": "cpu",
            "selection_order": [
                "corrected_j_at_1_desc",
                "corrected_j_at_5_desc",
                "no_grasp_rate_asc",
                "p95_latency_seconds_asc",
                "changed_axis_count_asc",
                "config_sha256_asc",
            ],
            "search_strategy": "single_pass_sequential_coordinate_search_on_pilot_100",
            "cartesian_product_forbidden": True,
            "pilot_trial_upper_bound": coordinate_trial_upper_bound(
                self.base_config, self.axes
            ),
            "full_validation_config_limit": self.max_full_configs,
            "full_validation_workers": self.full_workers,
            "base_config": self.base_config,
            "base_config_sha256": canonical_config_hash(self.base_config),
            "axes_in_order": self.axes,
            "runner": str(self.runner),
            "runner_sha256": _sha256(self.runner),
            "validation_samples_manifest": str(self.validation_samples),
            "validation_samples_manifest_sha256": _sha256(self.validation_samples),
            "validation_labels_manifest": str(self.validation_labels),
            "validation_labels_manifest_sha256": _sha256(self.validation_labels),
            "validation_sample_count": len(self.validation_ids),
            "validation_sample_ids_sha256": _id_hash(self.validation_ids),
            "pilot_manifest": str(self.pilot_manifest),
            "pilot_manifest_sha256": _sha256(self.pilot_manifest),
            "pilot_sample_count": len(self.pilot_ids),
            "pilot_sample_ids_sha256": _id_hash(self.pilot_ids),
            "test_data_access": "forbidden",
        }
        _write_immutable_json(
            self.analytic_root / "validation_search_space.json", protocol
        )

    def _config_file(self, config: Mapping[str, Any]) -> tuple[str, str, Path]:
        digest = canonical_config_hash(config)
        config_id = f"a0_{digest[:12]}"
        path = self.config_root / f"{config_id}.json"
        _write_immutable_json(path, dict(config))
        return config_id, digest, path

    def _validate_output(
        self,
        *,
        output_dir: Path,
        config_path: Path,
        expected_ids: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        complete = _load_json(output_dir / "COMPLETE.json")
        if complete.get("status") != "COMPLETE":
            raise RuntimeError(f"runner did not complete: {output_dir}")
        provenance = _load_json(output_dir / "run_config.json")
        if provenance.get("split") != "validation":
            raise RuntimeError(f"non-validation runner output: {output_dir}")
        if provenance.get("method") != "repeatedfilm_mask_depth_analytic":
            raise RuntimeError(f"unexpected method in runner output: {output_dir}")
        actual_config_sha = _sha256(config_path)
        if provenance.get("config_sha256") != actual_config_sha:
            raise RuntimeError(f"runner config hash mismatch: {output_dir}")
        if provenance.get("samples_manifest_sha256") != _sha256(self.validation_samples):
            raise RuntimeError(f"validation manifest hash mismatch: {output_dir}")
        if provenance.get("labels_manifest_sha256") != _sha256(self.validation_labels):
            raise RuntimeError(f"validation label hash mismatch: {output_dir}")
        table = pq.read_table(output_dir / "per_sample_predictions.parquet", columns=["sample_id"])
        output_ids = [str(value) for value in table.column("sample_id").to_pylist()]
        if len(output_ids) != len(set(output_ids)):
            raise RuntimeError(f"duplicate output sample IDs: {output_dir}")
        if set(output_ids) != set(expected_ids) or len(output_ids) != len(expected_ids):
            raise RuntimeError(f"output sample IDs do not match requested validation IDs: {output_dir}")
        metrics = _load_json(output_dir / "metrics.json")
        runtime = _load_json(output_dir / "runtime_metrics.json")
        if int(provenance.get("sample_count", -1)) != len(expected_ids):
            raise RuntimeError(f"runner sample count mismatch: {output_dir}")
        return metrics, runtime, provenance

    def _run_one(
        self,
        *,
        config: Mapping[str, Any],
        stage: str,
        axis: str,
        axis_value: Any,
    ) -> dict[str, Any]:
        if stage not in {"pilot", "full"}:
            raise ValueError(stage)
        config = dict(config)
        config_id, semantic_sha, config_path = self._config_file(config)
        output_dir = self.search_root / stage / config_id
        _assert_within(output_dir, self.search_root)
        expected_ids = self.pilot_ids if stage == "pilot" else self.validation_ids
        command = [
            str(self.python),
            str(self.runner),
            "--run-dir",
            str(self.run_dir),
            "--split",
            "validation",
            "--method",
            "A0",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if stage == "pilot":
            command.extend(("--sample-list", str(self.pilot_manifest)))
        command_record = {
            "command": command,
            "config_sha256": semantic_sha,
            "expected_sample_count": len(expected_ids),
            "split": "validation",
            "stage": stage,
        }
        _write_immutable_json(
            self.config_root / f"{stage}_{config_id}_command.json", command_record
        )
        resumed = (output_dir / "COMPLETE.json").is_file()
        if not resumed:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise RuntimeError(
                    f"incomplete non-empty runner output requires manual inspection: {output_dir}"
                )
            self.log_root.mkdir(parents=True, exist_ok=True)
            log_path = self.log_root / f"{stage}_{config_id}.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                }
            )
            with log_path.open("w", encoding="utf-8") as log_stream:
                process = subprocess.run(
                    command,
                    cwd=self.project_root,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if process.returncode:
                raise RuntimeError(
                    f"A0 runner failed with exit {process.returncode}; see {log_path}"
                )
        metrics, runtime, provenance = self._validate_output(
            output_dir=output_dir,
            config_path=config_path,
            expected_ids=expected_ids,
        )
        record = {
            "stage": stage,
            "axis": axis,
            "axis_value": json.dumps(axis_value, sort_keys=True),
            "config_id": config_id,
            "config_sha256": semantic_sha,
            "runner_config_sha256": provenance["config_sha256"],
            "sample_count": len(expected_ids),
            "corrected_j_at_1": float(metrics["j_at_1"]),
            "corrected_j_at_5": float(metrics["j_at_5"]),
            "recall_at_5": float(metrics.get("recall_at_5", math.nan)),
            "no_grasp_rate": float(metrics["no_grasp_rate"]),
            "p50_latency_seconds": float(metrics["p50_latency_seconds"]),
            "p95_latency_seconds": float(metrics["p95_latency_seconds"]),
            "total_wall_seconds": float(
                runtime["total_wall_seconds_including_io_and_evaluation"]
            ),
            "complexity": _complexity(config, self.base_config, tuple(self.axes)),
            "output_dir": str(output_dir),
            "resumed": bool(resumed),
            "selected": False,
        }
        self.configs[semantic_sha] = config
        return record

    def _save_results(self) -> None:
        ordered = sorted(
            self.records,
            key=lambda row: (
                0 if row["stage"] == "pilot" else 1,
                str(row["axis"]),
                str(row["config_sha256"]),
            ),
        )
        _atomic_csv(self.results_path, ordered)

    def run(self) -> dict[str, Any]:
        self.preregister()
        current = dict(self.base_config)
        base_record = self._run_one(
            config=current, stage="pilot", axis="base", axis_value=None
        )
        self.records.append(base_record)
        self.pilot_cache[base_record["config_sha256"]] = base_record
        lineage_hashes = [base_record["config_sha256"]]
        self._save_results()

        for axis, values in self.axes.items():
            candidates: list[dict[str, Any]] = []
            for value in values:
                candidate = dict(current)
                candidate[axis] = value
                digest = canonical_config_hash(candidate)
                record = self.pilot_cache.get(digest)
                if record is None:
                    record = self._run_one(
                        config=candidate,
                        stage="pilot",
                        axis=axis,
                        axis_value=value,
                    )
                    self.records.append(record)
                    self.pilot_cache[digest] = record
                    self._save_results()
                self.configs[digest] = candidate
                candidates.append(record)
            winner = min(candidates, key=selection_key)
            winner_hash = str(winner["config_sha256"])
            current = dict(self.configs[winner_hash])
            lineage_hashes.append(winner_hash)

        unique_lineage = list(dict.fromkeys(lineage_hashes))
        shortlist_hashes = sorted(
            unique_lineage, key=lambda digest: selection_key(self.pilot_cache[digest])
        )[: self.max_full_configs]
        if len(shortlist_hashes) > 3:
            raise AssertionError("full-validation shortlist exceeded protocol limit")

        full_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(self.full_workers, len(shortlist_hashes))) as pool:
            futures = {
                pool.submit(
                    self._run_one,
                    config=self.configs[digest],
                    stage="full",
                    axis="shortlist",
                    axis_value=index,
                ): digest
                for index, digest in enumerate(shortlist_hashes, start=1)
            }
            for future in as_completed(futures):
                full_records.append(future.result())
        self.records.extend(full_records)
        selected = min(full_records, key=selection_key)
        selected_hash = str(selected["config_sha256"])
        for record in self.records:
            record["selected"] = (
                record["stage"] == "full"
                and record["config_sha256"] == selected_hash
            )
        self._save_results()
        _atomic_json(
            self.analytic_root / "selected_config.json", self.configs[selected_hash]
        )
        return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--runner", type=Path, default=PROJECT_ROOT / "tools/grasp4dof/run_method.py"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--max-full-configs", type=int, default=3)
    parser.add_argument("--full-workers", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    pilot_manifest = args.pilot_manifest or run_dir / "manifests" / "pilot_100.json"
    tuner = AnalyticTuner(
        run_dir=run_dir,
        runner=args.runner,
        python=args.python,
        pilot_manifest=pilot_manifest,
        max_full_configs=args.max_full_configs,
        full_workers=args.full_workers,
    )
    selected = tuner.run()
    print(json.dumps({"status": "COMPLETE", "selected": selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
