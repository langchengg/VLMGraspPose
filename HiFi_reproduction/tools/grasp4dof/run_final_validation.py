#!/usr/bin/env python3
"""Run final full-validation predictions/oracles and freeze primary selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.results import (  # noqa: E402
    assert_aggregate_matches_sample_rows,
)
from tools.grasp4dof.select_validation_primary import (  # noqa: E402
    validate_existing_selection,
)


METHODS = ("G0", "G1", "C0", "C1", "A0")
EXPECTED_VALIDATION_COUNT = 3778


def _completion_marker(path: Path) -> Path:
    certified = path / "VALIDATED_COMPLETE.json"
    return certified if certified.is_file() else path / "COMPLETE.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete(
    path: Path, *, method_id: str, config_path: Path, oracle: bool
) -> bool:
    marker = _completion_marker(path)
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    config_sha256 = _sha256(config_path)
    if (
        value.get("schema_version") != 2
        or value.get("status") != "COMPLETE"
        or value.get("method_id") != method_id
        or value.get("split") != "validation"
        or value.get("oracle") is not oracle
        or value.get("config_sha256") != config_sha256
        or int(value.get("sample_count", -1)) != EXPECTED_VALIDATION_COUNT
    ):
        raise ValueError(f"invalid full-validation completion marker: {marker}")
    names = (
        "metrics.json",
        "runtime_metrics.json",
        "memory_metrics.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
        "run_config.json",
    )
    if not all((path / name).is_file() for name in names):
        raise ValueError(f"validation completion marker has missing outputs: {path}")
    declared = value.get("artifacts")
    if not isinstance(declared, dict) or any(
        declared.get(name) != _sha256(path / name) for name in names
    ):
        raise ValueError(f"validation completion artifact digest mismatch: {path}")
    run_config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    if (
        run_config.get("method_id") != method_id
        or run_config.get("split") != "validation"
        or run_config.get("oracle") is not oracle
        or int(run_config.get("sample_count", -1)) != EXPECTED_VALIDATION_COUNT
        or run_config.get("config_sha256") != config_sha256
        or run_config.get("per_sample_sha256")
        != _sha256(path / "per_sample_predictions.parquet")
        or run_config.get("per_candidate_sha256")
        != _sha256(path / "per_candidate_predictions.parquet")
    ):
        raise ValueError(f"validation completion provenance mismatch: {path}")
    sample_table = pq.read_table(path / "per_sample_predictions.parquet")
    if sample_table.num_rows != EXPECTED_VALIDATION_COUNT:
        raise ValueError(f"validation per-sample row count mismatch: {path}")
    sample_ids = sample_table.column("sample_id").to_pylist()
    if len(set(sample_ids)) != EXPECTED_VALIDATION_COUNT:
        raise ValueError(f"validation per-sample identity mismatch: {path}")
    # Opening both Parquet files rejects placeholder/non-Parquet files even if
    # their hashes were copied into a locally fabricated completion marker.
    pq.ParquetFile(path / "per_candidate_predictions.parquet")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    assert_aggregate_matches_sample_rows(sample_table.to_pylist(), metrics)
    return True


def _run(command: list[str], *, run: Path, label: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with (run / "commands.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp}\t{shlex.join(command)}\n")
    log = run / "logs" / f"validation_{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _validate_existing_selection(
    run: Path,
    *,
    configs: dict[str, Path],
    predicted: dict[str, Path],
    oracles: dict[str, Path],
) -> dict:
    # Keep the runner's resolved-path checks at the call site, but make the
    # content validator single-sourced. It rebuilds every row from the actual
    # prediction/oracle Parquet outputs instead of trusting a self-consistent
    # validation CSV and selection trace.
    if set(configs) != set(METHODS) or set(predicted) != set(METHODS) or set(oracles) != set(
        METHODS
    ):
        raise ValueError("validation selection inputs have incomplete method coverage")
    return validate_existing_selection(run)


def _selected_existing_output(
    table_path: Path, *, run: Path, method_id: str, config_path: Path
) -> Path:
    table = pd.read_csv(table_path)
    config_sha = _sha256(config_path)
    method_mask = (
        table["method_id"].astype(str) == method_id
        if "method_id" in table
        else pd.Series(True, index=table.index)
    )
    config_match = table["config_sha256"].astype(str) == config_sha
    if "runner_config_sha256" in table:
        config_match |= table["runner_config_sha256"].astype(str) == config_sha
    subset = table.loc[
        method_mask
        & config_match
        & (table["sample_count"].astype(int) == 3778)
    ].copy()
    if "stage" in subset:
        full = subset.loc[subset["stage"].astype(str) == "full"]
        if not full.empty:
            subset = full
    if subset.empty:
        raise ValueError(f"no complete selected validation output for {method_id}")
    subset = subset.sort_values(
        [column for column in ("selected", "selected_in_stage", "stage", "candidate_id") if column in subset],
        ascending=False,
        kind="mergesort",
    )
    outputs = [Path(str(value)).resolve() for value in subset["output_dir"].tolist()]
    for output in outputs:
        output.relative_to(run)
        if _complete(
            output, method_id=method_id, config_path=config_path, oracle=False
        ):
            return output
    raise ValueError(f"selected {method_id} result rows have no complete output")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError("full validation requires .venv-grasp4dof")
    if (run / "manifests/experiment_lock.json").exists():
        raise RuntimeError("full validation/selection is forbidden after formal lock")
    configs = {
        "G0": run / "grconvnet/pretrained_transfer/selected_config.json",
        "G1": run / "selected_configs/G1.json",
        "C0": run / "ggcnn2/pretrained_transfer/selected_config.json",
        "C1": run / "selected_configs/C1.json",
        "A0": run / "analytic/selected_config.json",
    }
    for method_id, path in configs.items():
        if not path.is_file():
            raise FileNotFoundError(f"selected {method_id} config missing: {path}")
    predicted = {
        "G0": _selected_existing_output(
            run / "validation/transfer_search_results.csv",
            run=run,
            method_id="G0",
            config_path=configs["G0"],
        ),
        "C0": _selected_existing_output(
            run / "validation/transfer_search_results.csv",
            run=run,
            method_id="C0",
            config_path=configs["C0"],
        ),
        "A0": _selected_existing_output(
            run / "analytic/validation_search_results.csv",
            run=run,
            method_id="A0",
            config_path=configs["A0"],
        ),
    }
    python = sys.executable
    for method_id in ("G1", "C1"):
        output = run / f"validation/final/{method_id}"
        if not _complete(
            output, method_id=method_id, config_path=configs[method_id], oracle=False
        ):
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"partial validation output: {output}")
            _run(
                [
                    python,
                    str(PROJECT_ROOT / "tools/grasp4dof/run_method.py"),
                    "--run-dir",
                    str(run),
                    "--split",
                    "validation",
                    "--method",
                    method_id,
                    "--config",
                    str(configs[method_id]),
                    "--output-dir",
                    str(output),
                ],
                run=run,
                label=method_id,
            )
            if not _complete(
                output,
                method_id=method_id,
                config_path=configs[method_id],
                oracle=False,
            ):
                raise RuntimeError(f"{method_id} validation did not complete")
        predicted[method_id] = output

    oracles: dict[str, Path] = {}
    # The validation matrix mirrors the formal matrix: every backend receives
    # a predicted-mask run and a separately labelled GT-mask oracle run. Only
    # G1/C1/A0 participate in primary selection, but omitting the G0/C0 oracle
    # would leave their grounding upper bounds unmeasured.
    for method_id in METHODS:
        output = run / f"validation/oracle/{method_id}-O"
        if not _complete(
            output, method_id=method_id, config_path=configs[method_id], oracle=True
        ):
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"partial validation oracle output: {output}")
            _run(
                [
                    python,
                    str(PROJECT_ROOT / "tools/grasp4dof/run_method.py"),
                    "--run-dir",
                    str(run),
                    "--split",
                    "validation",
                    "--method",
                    method_id,
                    "--config",
                    str(configs[method_id]),
                    "--oracle",
                    "--output-dir",
                    str(output),
                ],
                run=run,
                label=f"{method_id}-O",
            )
            if not _complete(
                output,
                method_id=method_id,
                config_path=configs[method_id],
                oracle=True,
            ):
                raise RuntimeError(f"{method_id} validation oracle did not complete")
        oracles[method_id] = output

    if not (run / "primary_validation_selection.json").exists():
        command = [
            python,
            str(PROJECT_ROOT / "tools/grasp4dof/select_validation_primary.py"),
            "--run-dir",
            str(run),
            "--expected-count",
            "3778",
            "--rate-tolerance",
            "0.005",
        ]
        for method_id in METHODS:
            command.extend(["--method-dir", f"{method_id}={predicted[method_id]}"])
            command.extend(["--config", f"{method_id}={configs[method_id]}"])
        for method_id, output in oracles.items():
            command.extend(["--oracle-dir", f"{method_id}={output}"])
        _run(command, run=run, label="primary_selection")
    selection = _validate_existing_selection(
        run,
        configs=configs,
        predicted=predicted,
        oracles=oracles,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "primary_method_id": selection["primary_method_id"],
                "predicted_outputs": {key: str(value) for key, value in predicted.items()},
                "oracle_outputs": {key: str(value) for key, value in oracles.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
