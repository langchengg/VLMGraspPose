#!/usr/bin/env python3
"""Record one manually verified physical robot trial without commanding hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.grasping.vgn_pipeline import atomic_write_json


CONFIRMATION = "I_CONFIRM_PHYSICAL_EXECUTION_OCCURRED"
REQUIRED_ARTIFACTS = (
    "before_rgb.png",
    "predicted_mask.png",
    "top1_overlay.png",
    "pregrasp.png",
    "closure.png",
    "lift.png",
    "after.png",
    "trial.mp4",
)
LABEL_FIELDS = (
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
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("trial manifest contains a non-object row")
    return [dict(row) for row in rows]


def success_definition_met(
    success_definition: str, labels: Mapping[str, bool]
) -> bool:
    """Evaluate the preregistered target-object success definition."""

    correct_target = not bool(labels["wrong_object"])
    if success_definition == "lift_10cm_hold_3s":
        return bool(
            correct_target
            and labels["object_lifted"]
            and labels["held_for_3s"]
        )
    if success_definition == "placed_in_bin":
        return bool(correct_target and labels["placed_in_bin"])
    raise ValueError(f"unsupported success definition: {success_definition}")


def build_trial_record(
    plan: Mapping[str, Any],
    *,
    annotator: str,
    label_source: str,
    labels: Mapping[str, bool],
    reported_success: bool,
    artifacts: Mapping[str, Mapping[str, Any]],
    physical_execution_attempted: bool = True,
) -> dict[str, Any]:
    if not annotator.strip():
        raise ValueError("annotator must be non-empty")
    missing_labels = [name for name in LABEL_FIELDS if name not in labels]
    if missing_labels:
        raise ValueError(f"missing physical labels: {missing_labels}")
    normalized = {name: bool(labels[name]) for name in LABEL_FIELDS}
    definition = str(plan["success_definition"])
    if physical_execution_attempted:
        if normalized["planning_failure"]:
            raise ValueError("a planning failure is not a physical execution attempt")
        physical_success: bool | None = success_definition_met(definition, normalized)
        if bool(reported_success) != physical_success:
            raise ValueError(
                "reported success disagrees with the preregistered success definition"
            )
        execution_status = "physically_executed_and_manually_recorded"
    else:
        if not normalized["planning_failure"]:
            raise ValueError(
                "a non-executed language trial must be labelled planning_failure"
            )
        if reported_success:
            raise ValueError("a non-executed trial cannot be reported as successful")
        physical_success = None
        execution_status = "planning_failed_before_physical_motion"
    return {
        "trial_id": str(plan["trial_id"]),
        "sample_id": str(plan["sample_id"]),
        "instruction": str(plan["instruction"]),
        "preregistered_order": int(plan["preregistered_order"]),
        "success_definition": definition,
        "execution_status": execution_status,
        "physical_execution_attempted": bool(physical_execution_attempted),
        "executor_mode": "externally_reviewed_hardware_interface",
        "annotator": annotator,
        "label_source": label_source,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **normalized,
        "physical_success": physical_success,
        "end_to_end_success": bool(
            normalized["grounding_correct"]
            and physical_success is True
            and not normalized["wrong_object"]
        ),
        "artifacts": {name: dict(value) for name, value in artifacts.items()},
        "simulation_result_used_as_label": False,
        "offline_vgn_quality_used_as_label": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-manifest", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument(
        "--label-source", choices=("human", "sensor", "human_and_sensor"), required=True
    )
    parser.add_argument(
        "--execution-status",
        choices=("physical_attempt", "planning_failure"),
        default="physical_attempt",
    )
    parser.add_argument("--confirm-physical-execution")
    parser.add_argument(
        "--success", action=argparse.BooleanOptionalAction, default=None, required=True
    )
    for field in LABEL_FIELDS:
        parser.add_argument(
            "--" + field.replace("_", "-"),
            dest=field,
            action=argparse.BooleanOptionalAction,
            default=None,
            required=True,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    physical_attempt = args.execution_status == "physical_attempt"
    if physical_attempt and args.confirm_physical_execution != CONFIRMATION:
        raise SystemExit(
            "physical execution was not explicitly confirmed; no real trial was recorded"
        )
    if not physical_attempt and args.confirm_physical_execution is not None:
        raise SystemExit("planning failures must not claim physical execution confirmation")
    manifest = _read_manifest(args.trials_manifest.expanduser().resolve())
    matches = [row for row in manifest if str(row.get("trial_id")) == args.trial_id]
    if len(matches) != 1:
        raise SystemExit(f"trial_id must match exactly one preregistered trial: {args.trial_id}")

    artifact_root = args.artifacts_dir.expanduser().resolve()
    required_artifacts = (
        REQUIRED_ARTIFACTS
        if physical_attempt
        else ("before_rgb.png", "predicted_mask.png", "top1_overlay.png")
    )
    missing = [name for name in required_artifacts if not (artifact_root / name).is_file()]
    if missing:
        raise SystemExit(f"required physical trial artifacts are missing: {missing}")
    artifacts = {
        name: {
            "path": str((artifact_root / name).resolve()),
            "sha256": _sha256(artifact_root / name),
        }
        for name in required_artifacts
    }
    labels = {name: bool(getattr(args, name)) for name in LABEL_FIELDS}
    record = build_trial_record(
        matches[0],
        annotator=args.annotator,
        label_source=args.label_source,
        labels=labels,
        reported_success=bool(args.success),
        artifacts=artifacts,
        physical_execution_attempted=physical_attempt,
    )
    destination = args.output.expanduser().resolve() / "trials" / args.trial_id / "trial.json"
    if destination.exists():
        raise SystemExit(
            f"trial record already exists and will not be overwritten: {destination}"
        )
    atomic_write_json(destination, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
