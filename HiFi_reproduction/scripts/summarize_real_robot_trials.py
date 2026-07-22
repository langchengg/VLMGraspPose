#!/usr/bin/env python3
"""Summarize manually recorded real-robot trials with Wilson 95% intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.grasping.vgn_pipeline import atomic_write_csv, atomic_write_json


SUMMARY_TRIAL_FIELDS = (
    "trial_id",
    "sample_id",
    "instruction",
    "physical_execution_attempted",
    "grounding_correct",
    "target_contact",
    "object_lifted",
    "held_for_3s",
    "placed_in_bin",
    "collision",
    "wrong_object",
    "slip",
    "planning_failure",
    "execution_failure",
    "physical_success",
    "end_to_end_success",
    "annotator",
    "label_source",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise ValueError("success count must lie in [0, total]")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "numerator": successes,
        "denominator": total,
        "rate": successes / total if total else None,
        "wilson_95_ci": wilson_interval(successes, total),
    }


def summarize_real_robot_records(
    records: Iterable[Mapping[str, Any]],
    *,
    planned_trial_count: int = 0,
) -> dict[str, Any]:
    """Compute only metrics supported by genuine physical execution logs."""

    rows = [dict(record) for record in records]
    language_trials = [
        row
        for row in rows
        if row.get("execution_status")
        in {
            "physically_executed_and_manually_recorded",
            "planning_failed_before_physical_motion",
        }
    ]
    attempted = [
        row for row in language_trials if row.get("physical_execution_attempted") is True
    ]
    base: dict[str, Any] = {
        "metric_scope": "real_robot_physical_execution",
        "planned_trial_count": int(planned_trial_count),
        "recorded_trial_count": len(rows),
        "physical_attempt_count": len(attempted),
        "simulation_substitution": False,
        "offline_metric_substitution": False,
    }
    if not language_trials:
        return {
            **base,
            "status": "not_available",
            "real_robot_grasp_success_rate": None,
            "end_to_end_real_success_rate": None,
            "conditional_grasp_success_given_correct_grounding": None,
            "reason": "no physical robot execution logs",
        }

    for index, row in enumerate(attempted):
        if row.get("simulation_result_used_as_label") is True:
            raise ValueError(f"trial {index} attempts to use simulation as a real label")
        if row.get("offline_vgn_quality_used_as_label") is True:
            raise ValueError(f"trial {index} attempts to use VGN quality as a real label")
        if not isinstance(row.get("physical_success"), bool):
            raise ValueError(f"physical trial {index} lacks a boolean physical_success label")
        if not isinstance(row.get("grounding_correct"), bool):
            raise ValueError(f"physical trial {index} lacks grounding_correct")

    physical_successes = sum(bool(row["physical_success"]) for row in attempted)
    end_to_end_successes = sum(
        row.get("physical_success") is True
        and bool(row["grounding_correct"])
        and not bool(row.get("wrong_object", False))
        for row in language_trials
    )
    grounding_correct = [
        row for row in language_trials if row.get("grounding_correct") is True
    ]
    conditional_successes = sum(
        row.get("physical_success") is True for row in grounding_correct
    )
    physical_metric = _metric(physical_successes, len(attempted))
    end_to_end_metric = _metric(end_to_end_successes, len(language_trials))
    conditional_metric = _metric(conditional_successes, len(grounding_correct))
    return {
        **base,
        "status": (
            "completed_from_physical_logs"
            if attempted
            else "language_trials_recorded_without_physical_attempts"
        ),
        "real_robot_grasp_success_rate": physical_metric["rate"],
        "real_robot_grasp_success": physical_metric if attempted else None,
        "end_to_end_real_success_rate": end_to_end_metric["rate"],
        "end_to_end_real_success": end_to_end_metric,
        "conditional_grasp_success_given_correct_grounding": conditional_metric["rate"],
        "conditional_grasp_success": conditional_metric,
        "reason": None if attempted else "no physical robot execution logs",
    }


def _read_planned_count(manifest: Path | None) -> int:
    if manifest is None:
        return 0
    return sum(
        1
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _load_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("trials/*/trial.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"trial record is not an object: {path}")
        row = dict(payload)
        row["record_path"] = str(path)
        rows.append(row)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--trials-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.trials_root.expanduser().resolve()
    manifest = (
        args.trials_manifest.expanduser().resolve()
        if args.trials_manifest is not None
        else None
    )
    records = _load_records(root)
    summary = summarize_real_robot_records(
        records, planned_trial_count=_read_planned_count(manifest)
    )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "trials.csv", records, SUMMARY_TRIAL_FIELDS)
    atomic_write_json(output / "aggregate.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
