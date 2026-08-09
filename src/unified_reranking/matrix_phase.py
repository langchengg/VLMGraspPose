"""Exact, hash-bound matrix phase execution inventories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import verify_artifact_records_recursive, verified_artifact_path
from .hashing import canonical_sha256


def _assert_cell_matches_command(
    cell: dict[str, Any], command: Any, *, run_dir: Path, name: str
) -> None:
    if not isinstance(command, list):
        raise RuntimeError(f"{name} command is not a token list")
    try:
        module_index = command.index("-m") + 1
        module = str(command[module_index])
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"{name} command has no Python module") from error
    if module not in {
        "tools.unified_reranking.train_matrix_cell",
        "tools.unified_reranking.run_interpretable_rule",
    }:
        raise RuntimeError(f"{name} command has an unsupported worker module")
    options: dict[str, str] = {}
    index = module_index + 1
    while index < len(command):
        flag = str(command[index])
        if not flag.startswith("--") or index + 1 >= len(command):
            raise RuntimeError(f"{name} command options are malformed")
        key = flag[2:].replace("-", "_")
        if key in options:
            raise RuntimeError(f"{name} command repeats --{key.replace('_', '-')}")
        options[key] = str(command[index + 1])
        index += 2
    required = {"run_dir", "route", "track", "seed", "mode"}
    required.add(
        "method"
        if module == "tools.unified_reranking.run_interpretable_rule"
        else "encoder"
    )
    if module == "tools.unified_reranking.train_matrix_cell":
        required.add("loss")
    if not required.issubset(options):
        raise RuntimeError(f"{name} command misses required worker options")
    if Path(options.pop("run_dir")).resolve() != run_dir:
        raise RuntimeError(f"{name} command uses the wrong run directory")
    configuration = cell.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError(f"{name} has no configuration")
    held_fold = configuration.get("held_fold")
    if "fold" not in options and held_fold is not None:
        raise RuntimeError(f"{name} fold differs from its command")
    for key, expected in options.items():
        configuration_key = "held_fold" if key == "fold" else key
        observed = configuration.get(configuration_key)
        if observed is None or str(observed) != expected:
            raise RuntimeError(
                f"{name} configuration differs from command option --{key.replace('_', '-')}"
            )


def load_matrix_phase_cells(
    run_dir: str | Path,
    phase: str,
    *,
    expected_selection: Path | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    """Return exactly the cells declared by one completed matrix phase."""

    root = Path(run_dir).resolve()
    execution_path = (
        root / "05_models" / "matrix_plans" / f"{phase}_latest_execution.json"
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "COMPLETE" or execution.get("phase") != phase:
        raise RuntimeError(f"matrix {phase} execution is not COMPLETE")
    plan_path = verified_artifact_path(
        execution.get("plan"), name=f"matrix {phase} plan"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "PLANNED"
        or plan.get("phase") != phase
        or plan.get("job_count") != len(plan.get("jobs", []))
        or execution.get("job_count") != plan.get("job_count")
    ):
        raise RuntimeError(f"matrix {phase} plan/execution contract mismatch")
    planner = plan.get("planner_tool")
    verified_artifact_path(planner, name=f"matrix {phase} planner tool")
    if phase == "screen":
        if plan.get("selection") is not None or execution.get("selection") is not None:
            raise RuntimeError("matrix screen must not have a selection input")
    else:
        plan_selection = plan.get("selection")
        execution_selection = execution.get("selection")
        if plan_selection != execution_selection:
            raise RuntimeError(f"matrix {phase} selection record mismatch")
        selection_path = verified_artifact_path(
            plan_selection, name=f"matrix {phase} selection"
        )
        if expected_selection is not None and selection_path != expected_selection.resolve():
            raise RuntimeError(f"matrix {phase} used the wrong selection artifact")
    jobs = plan["jobs"]
    identifiers = [str(job.get("identifier", "")) for job in jobs]
    if (
        any(not identifier for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or any(
            identifier
            != canonical_sha256(tuple(map(str, job.get("command", ()))))[:16]
            for identifier, job in zip(identifiers, jobs, strict=True)
        )
    ):
        raise RuntimeError(f"matrix {phase} job inventory is malformed")
    results = execution.get("results")
    outputs = execution.get("output_manifests")
    if (
        not isinstance(results, list)
        or not isinstance(outputs, list)
        or not (len(results) == len(outputs) == len(jobs))
    ):
        raise RuntimeError(f"matrix {phase} output inventory is incomplete")
    by_identifier = {str(result.get("identifier", "")): result for result in results}
    if set(by_identifier) != set(identifiers) or any(
        int(result.get("returncode", -1)) != 0 for result in results
    ):
        raise RuntimeError(f"matrix {phase} result inventory differs from the plan")
    result_records = [
        result.get("output_manifest")
        for result in sorted(results, key=lambda item: str(item.get("identifier", "")))
    ]
    if result_records != outputs:
        raise RuntimeError(f"matrix {phase} output ordering/binding mismatch")
    cells: list[tuple[dict[str, Any], Path]] = []
    seen: set[Path] = set()
    sorted_jobs = sorted(jobs, key=lambda item: str(item.get("identifier", "")))
    for index, (job, record) in enumerate(zip(sorted_jobs, outputs, strict=True)):
        path = verified_artifact_path(record, name=f"matrix {phase} cell {index}")
        if path in seen:
            raise RuntimeError(f"matrix {phase} contains a duplicate cell manifest")
        seen.add(path)
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("status") != "COMPLETE":
            raise RuntimeError(f"matrix {phase} cell is not COMPLETE: {path}")
        configuration = cell.get("configuration")
        cell_key = str(cell.get("cell_key", ""))
        if (
            not isinstance(configuration, dict)
            or cell_key != canonical_sha256(configuration)[:16]
            or path.parent.name != cell_key
        ):
            raise RuntimeError(f"matrix {phase} cell identity is invalid: {path}")
        _assert_cell_matches_command(
            cell,
            job.get("command"),
            run_dir=root,
            name=f"matrix {phase} cell {index}",
        )
        verify_artifact_records_recursive(
            {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
            name=f"matrix {phase} cell {index}",
            require_at_least_one=True,
        )
        cells.append((cell, path))
    return cells


__all__ = ["load_matrix_phase_cells"]
