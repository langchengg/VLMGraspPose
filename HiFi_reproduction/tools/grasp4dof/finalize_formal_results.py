#!/usr/bin/env python3
"""Recompute, consolidate, analyse, visualize, and report locked formal outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.experiment_lock import verify_lock  # noqa: E402
from tools.grasp4dof.generate_final_reports import (  # noqa: E402
    SCIENTIFIC_SCOPE_SENTENCE,
    render_final_reports,
)
from tools.grasp4dof.run_formal_inference import _complete as _formal_complete  # noqa: E402


METHOD_DIRS = {
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
REPORTS = (
    "REPEATEDFILM_4DOF_IMPLEMENTATION.md",
    "REPEATEDFILM_4DOF_AUDIT.md",
    "REPEATEDFILM_4DOF_VALIDATION.md",
    "REPEATEDFILM_4DOF_RESULTS.md",
    "REPEATEDFILM_4DOF_FAILURE_ANALYSIS.md",
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


def _run(command: list[str], *, run: Path, label: str) -> None:
    rendered = shlex.join(command)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with (run / "commands.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp}\t{rendered}\n")
    log = run / "logs" / f"finalize_{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _validated_report_hashes(run: Path) -> dict[str, str]:
    _, expected_reports = render_final_reports(run)
    report_dir = run / "reports"
    result: dict[str, str] = {}
    for name in REPORTS:
        path = report_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing final report: {path}")
        observed = path.read_text(encoding="utf-8")
        if SCIENTIFIC_SCOPE_SENTENCE not in observed:
            raise RuntimeError(f"invalid final report: {path}")
        if observed != expected_reports[name]:
            raise RuntimeError(f"final report content does not match evidence: {path}")
        result[name] = _sha256(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError("formal finalization requires .venv-grasp4dof")
    lock = verify_lock(run)
    selected_configs = _json(run / "selected_configs.json")
    for method_id, relative in METHOD_DIRS.items():
        directory = run / relative
        if method_id == "R0":
            complete = _formal_complete(
                directory,
                method_id="R0",
                oracle=False,
                lock=lock,
            )
        else:
            base_method = method_id.removesuffix("-O")
            config = Path(str(selected_configs[base_method]["path"])).resolve()
            complete = _formal_complete(
                directory,
                method_id=base_method,
                oracle=method_id.endswith("-O"),
                lock=lock,
                config=config,
            )
        if not complete:
            raise FileNotFoundError(f"{method_id} formal output is incomplete")
    python = sys.executable
    primary = str(lock["protocol"]["primary_method"])
    if primary not in {"G1", "C1", "A0"}:
        raise ValueError("locked primary ID is invalid")

    if not (run / "results_bundle.json").exists():
        command = [
            python,
            str(PROJECT_ROOT / "tools/grasp4dof/consolidate_results.py"),
            "--run-dir",
            str(run),
            "--primary-method-id",
            primary,
            "--expected-count",
            "7675",
        ]
        for method_id, relative in METHOD_DIRS.items():
            command.extend(["--method-dir", f"{method_id}={run / relative}"])
        _run(command, run=run, label="consolidate")

    statistics_dir = run / "statistics"
    if not (statistics_dir / "statistical_tests.json").exists():
        if statistics_dir.exists() and any(statistics_dir.iterdir()):
            raise RuntimeError("partial statistics directory requires audit")
        _run(
            [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/run_statistics.py"),
                "--predictions",
                str(run / "per_sample_predictions.parquet"),
                "--output-dir",
                str(statistics_dir),
                "--bootstrap-draws",
                "10000",
                "--seed",
                "20260803",
            ],
            run=run,
            label="statistics",
        )
    for name in ("statistical_tests.json", "bootstrap_intervals.json"):
        root_output = run / name
        if not root_output.exists():
            root_output.write_bytes((statistics_dir / name).read_bytes())
        elif root_output.read_bytes() != (statistics_dir / name).read_bytes():
            raise RuntimeError(f"root/statistics copy drift: {name}")

    subgroup_outputs = (
        "subgroup_results.csv",
        "subgroup_sample_features.parquet",
        "subgroup_analysis_manifest.json",
    )
    if not all((statistics_dir / name).is_file() for name in subgroup_outputs):
        existing = [name for name in subgroup_outputs if (statistics_dir / name).exists()]
        if existing:
            raise RuntimeError(f"partial subgroup analysis requires audit: {existing}")
        _run(
            [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/analyze_subgroups.py"),
                "--run-dir",
                str(run),
                "--predictions",
                str(run / "per_sample_predictions.parquet"),
                "--output-dir",
                str(statistics_dir),
            ],
            run=run,
            label="subgroups",
        )
    for name in subgroup_outputs:
        root_output = run / name
        if not root_output.exists():
            root_output.write_bytes((statistics_dir / name).read_bytes())
        elif root_output.read_bytes() != (statistics_dir / name).read_bytes():
            raise RuntimeError(f"root/subgroup copy drift: {name}")

    if not (run / "independent_recompute_results.json").exists():
        command = [
            python,
            str(PROJECT_ROOT / "tools/grasp4dof/independent_recompute.py"),
            "--run-dir",
            str(run),
        ]
        for method_id, relative in METHOD_DIRS.items():
            command.extend(["--method-dir", f"{method_id}={run / relative}"])
        _run(command, run=run, label="independent_recompute")

    if not (run / "gallery/index.html").exists():
        if (run / "gallery").exists() or (run / "per_sample_failure_stage.parquet").exists():
            raise RuntimeError("partial gallery/failure output requires audit")
        selected = _json(run / "selected_configs.json")
        command = [
            python,
            str(PROJECT_ROOT / "tools/grasp4dof/build_failure_gallery.py"),
            "--run-dir",
            str(run),
        ]
        for method_id in ("R0", "G0", "G1", "C0", "C1", "A0"):
            command.extend(
                ["--method-dir", f"{method_id}={run / METHOD_DIRS[method_id]}"]
            )
        for method_id in ("G0", "G1", "C0", "C1"):
            command.extend(
                ["--dense-map-config", f"{method_id}={selected[method_id]['path']}"]
            )
        _run(command, run=run, label="failure_gallery")

    protected_verification = run / "audit/protected_source_final_verification.json"
    if not protected_verification.exists():
        _run(
            [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/verify_protected_sources.py"),
                "--run-dir",
                str(run),
            ],
            run=run,
            label="protected_sources",
        )

    storage_log = run / "storage_usage_by_stage.csv"
    recorded_stages: set[str] = set()
    if storage_log.is_file():
        with storage_log.open(newline="", encoding="utf-8") as stream:
            recorded_stages = {str(row["stage"]) for row in csv.DictReader(stream)}
    if "formal_complete_before_reports" not in recorded_stages:
        _run(
            [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/record_storage.py"),
                "--run-dir",
                str(run),
                "--stage",
                "formal_complete_before_reports",
            ],
            run=run,
            label="storage_final",
        )

    report_dir = run / "reports"
    if not all((report_dir / name).is_file() for name in REPORTS):
        existing = [name for name in REPORTS if (report_dir / name).exists()]
        if existing:
            raise RuntimeError(f"partial final report set requires audit: {existing}")
        _run(
            [
                python,
                str(PROJECT_ROOT / "tools/grasp4dof/generate_final_reports.py"),
                "--run-dir",
                str(run),
            ],
            run=run,
            label="reports",
        )
    # Re-run both the evidence gate and deterministic renderer even when a
    # complete-looking report set already existed from an interrupted run.
    report_hashes = _validated_report_hashes(run)
    completion_path = run / "FINALIZATION_COMPLETE.json"
    completion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment_lock_sha256": lock["manifest_content_sha256"],
        "primary_method_id": primary,
        "reports": report_hashes,
        "results_bundle_sha256": _sha256(run / "results_bundle.json"),
        "independent_recompute_sha256": _sha256(
            run / "independent_recompute_results.json"
        ),
        "statistical_tests_sha256": _sha256(run / "statistical_tests.json"),
        "bootstrap_intervals_sha256": _sha256(run / "bootstrap_intervals.json"),
        "failure_stage_sha256": _sha256(run / "per_sample_failure_stage.parquet"),
        "gallery_manifest_sha256": _sha256(run / "gallery/selection_manifest.json"),
        "protected_verification_sha256": _sha256(protected_verification),
    }
    encoded = json.dumps(completion, indent=2, sort_keys=True) + "\n"
    if completion_path.exists():
        if completion_path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("finalization completion manifest drift")
    else:
        descriptor = os.open(
            completion_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "lock_sha256": lock["manifest_content_sha256"],
                "primary_method_id": primary,
                "method_count": len(METHOD_DIRS),
                "independent_recompute": str(run / "independent_recompute_results.json"),
                "gallery": str(run / "gallery/index.html"),
                "reports": [str(report_dir / name) for name in REPORTS],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
