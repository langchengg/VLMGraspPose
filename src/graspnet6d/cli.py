"""Unified, resumable CLI for the GraspNet + frozen VGN experiment route."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .audit import repository_root, run_audit
from .compact_download import deterministic_frame_ids
from .dataset import (
    DatasetValidationError,
    discover_scene_ids,
    validate_graspnet_structure,
)
from .io import atomic_json, atomic_jsonl, atomic_text, canonical_sha256, sha256_file
from .manifest import (
    SelectionConfig,
    build_language_audit,
    build_target_and_language_manifests,
)
from .provenance import (
    create_run,
    new_run_id,
    record_stage,
    resolve_profile_config,
    update_manifest,
)
from .smoke import run_device_benchmark, run_smoke
from .splits import SceneSplit, deterministic_scene_split, uniform_frame_indices
from .workflow import (
    WorkflowBlocked,
    download,
    ensure_compact_training_frames,
    prepare,
    record_failure,
    retire_compact_training_archives,
    write_blocked_status,
)


STAGES = (
    "masks",
    "candidates",
    "labels",
    "features",
    "train-ranker",
    "evaluate",
    "ablate",
    "report",
)


def _profile_name(value: str) -> str:
    name = value.strip().lower().replace("_", "-")
    if name not in {
        "smoke",
        "paper-lite",
        "paper-lite-train3",
        "paper-extended",
    }:
        raise argparse.ArgumentTypeError(
            "profile must be smoke, paper-lite, paper-lite-train3, or paper-extended"
        )
    return name


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config(profile: str) -> tuple[Path, dict[str, Any]]:
    filename = {
        "smoke": "smoke.yaml",
        "paper-lite": "paper_lite.yaml",
        "paper-lite-train3": "paper_lite_train3.yaml",
        "paper-extended": "paper_extended.yaml",
    }[profile]
    path = repository_root() / "configs" / "graspnet6d" / filename
    resolved, _ = resolve_profile_config(profile)
    return path, resolved


def _active_run_id() -> str | None:
    path = repository_root() / "artifacts" / "graspnet6d" / "ACTIVE_RUN_ID"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _run_dir(args: argparse.Namespace, *, command: str | None = None) -> Path:
    # A fresh download/all command starts a run.  Every downstream command
    # defaults to the immutable active run so the documented command sequence
    # does not silently create unrelated empty runs.
    selected_command = command or getattr(args, "command", None)
    starts_fresh = selected_command in {"download", "all"} and not bool(
        getattr(args, "resume", False)
    )
    active = None if starts_fresh else _active_run_id()
    if active is not None and args.run_id is None:
        active_manifest = (
            repository_root()
            / "artifacts"
            / "graspnet6d"
            / active
            / "run_manifest.json"
        )
        try:
            active_payload = json.loads(active_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            active_payload = {}
        if active_payload.get("profile") != args.profile:
            # A changed profile is a new immutable experiment identity.  The
            # configured lineage, not ACTIVE_RUN_ID, links it to its parent.
            active = None
    run_id = args.run_id or active or new_run_id()
    run_dir = create_run(run_id, profile=args.profile, command=sys.argv)
    config_path, config = _config(args.profile)
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        temporary.replace(resolved)
    if config.get("formal_results") is True:
        # Publish the immutable environment evidence before any data-backed
        # stage, so even an honest download/geometry block leaves the required
        # per-run snapshot.  Existing commits are always hash-revalidated.
        from .run_outputs import publish_environment

        publish_environment(
            run_dir,
            resume=(run_dir / "environment_provenance.json").is_file(),
        )
    return run_dir


def _write_data_validation(report: Any, *, run_dir: Path | None = None) -> None:
    root = repository_root()
    payload = report.to_record()
    atomic_json(root / "artifacts/graspnet6d/data_validation.json", payload)
    lines = [
        "# GraspNet data validation",
        "",
        f"Valid: **{payload['valid']}**",
        f"Scenes: {len(payload['scene_ids'])}",
        f"Frames requested per scene: {len(payload['frame_ids'])}",
        f"Files checked: {payload['checked_file_count']}",
        f"Missing: {len(payload['missing'])}",
        f"Invalid: {len(payload['invalid'])}",
        "",
    ]
    if payload["missing"]:
        lines.extend(
            ["## Missing", "", *[f"- `{item}`" for item in payload["missing"]]]
        )
    if payload["invalid"]:
        lines.extend(
            ["", "## Invalid", "", *[f"- {item}" for item in payload["invalid"]]]
        )
    from .provenance import atomic_text

    atomic_text(
        root / "artifacts/graspnet6d/data_validation.md", "\n".join(lines) + "\n"
    )
    if run_dir is not None:
        atomic_json(run_dir / "data_validation.json", payload)
        atomic_text(run_dir / "data_validation.md", "\n".join(lines) + "\n")


def prepare_experiment(profile: str, *, run_dir: Path, resume: bool) -> dict[str, Any]:
    root = repository_root()
    prepare(profile, run_dir=run_dir, resume=resume)
    dataset_root = root / "data_external" / "graspnet"
    scenes = discover_scene_ids(dataset_root)
    if not scenes:
        raise WorkflowBlocked("no extracted GraspNet scenes were discovered")
    _, config = _config(profile)
    dataset_config = config["dataset"]
    split_config = config["split"]
    compact_split_path = (
        root
        / "configs"
        / "graspnet6d"
        / "splits"
        / str(
            split_config.get(
                "manifest_filename",
                "graspnet_scene_split_smoke_v2.json"
                if profile == "smoke"
                else "graspnet_scene_split_v2.json",
            )
        )
    )
    compact_split: dict[str, Any] | None = None
    if compact_split_path.is_file():
        compact_split = json.loads(compact_split_path.read_text(encoding="utf-8"))
        if compact_split.get("schema_version") != "graspnet6d_scene_split_v2":
            raise WorkflowBlocked(f"unsupported compact split schema: {compact_split_path}")
        selected_names = [
            str(scene_id)
            for key in ("train", "validation", "test")
            for scene_id in compact_split.get(key, [])
        ]
        selected_scenes = sorted(int(value.split("_")[1]) for value in selected_names)
        if set(selected_scenes) != set(scenes):
            raise WorkflowBlocked(
                "compact extracted scenes differ from the locked v2 scene split"
            )
        scenes = selected_scenes
    else:
        scene_limit = dataset_config.get("scenes")
        if scene_limit is not None:
            scenes = scenes[: int(scene_limit)]
    selected_frames = (
        deterministic_frame_ids(int(dataset_config["frames_per_scene"]))
        if profile == "paper-lite-train3"
        else uniform_frame_indices(
            total_frames=int(dataset_config.get("total_views_per_scene", 256)),
            frames_per_scene=int(dataset_config["frames_per_scene"]),
        )
    )
    report = validate_graspnet_structure(
        dataset_root,
        camera=str(config["camera"]),
        scene_ids=scenes,
        frame_ids=selected_frames,
        strict=True,
    )
    _write_data_validation(report, run_dir=run_dir)
    if not report.valid:
        raise DatasetValidationError(report)
    if compact_split is not None:
        split = SceneSplit(
            train=tuple(map(str, compact_split["train"])),
            validation=tuple(map(str, compact_split["validation"])),
            test=tuple(map(str, compact_split["test"])),
            seed=int(compact_split["seed"]),
            source_profile=str(compact_split["source_profile"]),
        )
        split_document = compact_split
        split_path = compact_split_path
    else:
        ratios = split_config["ratios"]
        split = deterministic_scene_split(
            [f"scene_{scene:04d}" for scene in scenes],
            seed=int(config["seed"]),
            ratios=(
                float(ratios["train"]),
                float(ratios["validation"]),
                float(ratios["test"]),
            ),
            min_validation_scenes=int(split_config["min_validation_scenes"]),
            min_test_scenes=int(split_config["min_test_scenes"]),
            source_profile=str(
                split_config.get(
                    "description",
                    "smoke-only scene split"
                    if profile == "smoke"
                    else "held-out GraspNet training-scene split",
                )
            ),
        )
        split_document = split.to_dict()
        split_filename = f"graspnet_scene_split_{profile.replace('-', '_')}.json"
        split_path = root / "configs/graspnet6d/splits" / split_filename
        atomic_json(split_path, split_document)
    atomic_json(run_dir / "split_audit.json", split_document)
    atomic_json(run_dir / "split_manifest.json", split_document)
    target_config = config["target_selection"]
    selection = SelectionConfig(
        frames_per_scene=int(dataset_config["frames_per_scene"]),
        max_targets_per_frame=int(dataset_config["max_targets_per_frame"]),
        min_mask_pixels=int(target_config["min_mask_pixels"]),
        min_valid_depth_fraction=float(target_config["min_valid_depth_fraction"]),
        seed=int(config["seed"]),
        max_groups=(
            None
            if dataset_config.get("max_groups") is None
            else int(dataset_config["max_groups"])
        ),
        frame_ids=(selected_frames if profile == "paper-lite-train3" else None),
    )
    manifest = build_target_and_language_manifests(
        dataset_root,
        split,
        camera=str(config["camera"]),
        catalog_path=root / "configs/graspnet6d/graspnet_object_catalog_v1.json",
        output_root=run_dir / "manifests",
        config=selection,
    )
    if profile == "paper-lite-train3":
        sampling = config.get("sampling", {})
        levels = tuple(int(value) for value in sampling["adaptive_frames_per_scene"])
        minimums = {
            key: int(value) for key, value in sampling["minimum_groups"].items()
        }

        def group_minimums_met(current: dict[str, Any]) -> bool:
            observed = current.get("split_counts", {})
            return all(
                int(observed.get(partition, -1)) >= minimums[partition]
                for partition in ("train", "validation", "test")
            )

        for frame_count in levels:
            if frame_count <= len(selected_frames) or group_minimums_met(manifest):
                continue
            ensure_compact_training_frames(
                profile, frame_count, run_dir=run_dir
            )
            selected_frames = deterministic_frame_ids(frame_count)
            report = validate_graspnet_structure(
                dataset_root,
                camera=str(config["camera"]),
                scene_ids=scenes,
                frame_ids=selected_frames,
                strict=True,
            )
            _write_data_validation(report, run_dir=run_dir)
            selection = SelectionConfig(
                frames_per_scene=frame_count,
                max_targets_per_frame=int(dataset_config["max_targets_per_frame"]),
                min_mask_pixels=int(target_config["min_mask_pixels"]),
                min_valid_depth_fraction=float(
                    target_config["min_valid_depth_fraction"]
                ),
                seed=int(config["seed"]),
                max_groups=(
                    None
                    if dataset_config.get("max_groups") is None
                    else int(dataset_config["max_groups"])
                ),
                frame_ids=selected_frames,
            )
            manifest = build_target_and_language_manifests(
                dataset_root,
                split,
                camera=str(config["camera"]),
                catalog_path=root
                / "configs/graspnet6d/graspnet_object_catalog_v1.json",
                output_root=run_dir / "manifests",
                config=selection,
            )
        if not group_minimums_met(manifest):
            atomic_json(
                run_dir / "train2_fallback_eligibility.json",
                {
                    "eligible": True,
                    "reason": "target-group minimums unmet after 32 frames/scene",
                    "observed": manifest.get("split_counts", {}),
                    "required": minimums,
                    "train3_only_run_preserved": True,
                },
            )
            raise WorkflowBlocked(
                "train3 target-group minimums remain unmet after deterministic "
                "16/24/32-frame expansion; train_2 fallback is now eligible but "
                "must use a separate child run"
            )
        retire_compact_training_archives(profile, run_dir=run_dir)
    target_manifest_path = run_dir / "manifests" / "target_groups.jsonl"
    language_manifest_path = run_dir / "manifests" / "language_queries.jsonl"
    language_audit = build_language_audit(
        target_manifest_path,
        language_manifest_path,
        run_dir,
        seed=int(config["seed"]),
        sample_count=100,
        required_minimum=1 if profile == "smoke" else 100,
    )
    # Keep the canonical stage inputs under manifests/ while also publishing
    # the run-root language file required by the paper artifact contract.
    atomic_text(
        run_dir / "language_manifest.jsonl",
        language_manifest_path.read_text(encoding="utf-8"),
    )
    from .provenance import update_manifest

    update_manifest(
        run_dir,
        dataset_manifest_hash=manifest["target_manifest_sha256"],
        split_hash=manifest.get("split_hash", None)
        or hashlib.sha256(
            json.dumps(split_document, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        sample_counts={
            "scenes": len(scenes),
            "frames": len(scenes) * len(selected_frames),
            "target_groups": int(manifest["target_group_count"]),
            "excluded_groups": int(manifest["excluded_count"]),
        },
    )
    record_stage(
        run_dir,
        "prepare_experiment",
        "COMPLETE",
        data_validation=report.to_record(),
        manifest=manifest,
        language_audit=language_audit,
    )
    return {**manifest, "language_audit": language_audit}


def _manifest_paths(run_dir: Path) -> tuple[Path, Path]:
    target = run_dir / "manifests" / "target_groups.jsonl"
    language = run_dir / "manifests" / "language_queries.jsonl"
    missing = [str(path) for path in (target, language) if not path.is_file()]
    if missing:
        raise WorkflowBlocked(
            f"prepared target/language manifests are missing: {missing}; run prepare first"
        )
    return target, language


def _record_value(value: Any) -> Any:
    if is_dataclass(value):
        return _record_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _record_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_record_value(item) for item in value]
    return value


def _atomic_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise WorkflowBlocked(f"refusing to publish an empty required CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _merge_csv_files(sources: list[Path], destination: Path) -> None:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not source.is_file():
            raise WorkflowBlocked(f"required stage CSV is absent: {source}")
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    _atomic_csv_rows(destination, rows)


def _record_runtime(
    run_dir: Path, stage: str, *, started: float, status: str, resumed: bool
) -> None:
    destination = run_dir / "runtime.csv"
    existing: list[dict[str, Any]] = []
    if destination.is_file():
        with destination.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        # runtime.csv is formal first-success evidence, not an attempt log.
        # A strict resume may exercise the same stage again, but it must not
        # replace the measured wall time that produced the published inputs.
        if any(
            row.get("stage") == stage and row.get("status") == "COMPLETE"
            for row in rows
        ):
            return
        existing = [row for row in rows if row.get("stage") != stage]
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = -1
    existing.append(
        {
            "stage": stage,
            "status": status,
            "wall_time_s": f"{time.perf_counter() - started:.9f}",
            "process_rss_bytes_at_end": rss,
            "resume_requested": bool(resumed),
        }
    )
    _atomic_csv_rows(destination, existing)


def _mark_run_failed(run_dir: Path, stage: str, error: BaseException) -> None:
    """Close an unexpected failed attempt without deleting published evidence."""

    update_manifest(
        run_dir,
        status="FAILED",
        failed_stage=stage,
        failure_type=type(error).__name__,
        failure_message=str(error),
        ended_at_utc=datetime.now(timezone.utc).isoformat(),
        formal_results_emitted=False,
    )


def _selected_predicted_condition(run_dir: Path) -> str:
    selection_path = run_dir / "predicted_condition_selection.json"
    if not selection_path.is_file():
        raise WorkflowBlocked(
            "validation-only predicted-mask selection is absent; run masks --condition all"
        )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = str(payload.get("selected_condition", ""))
    if selected not in {"hifics_zero_shot_mask", "hifics_adapted_mask"}:
        raise WorkflowBlocked(f"invalid predicted-mask selection: {selection_path}")
    sources = payload.get("source_paths", {})
    hashes = payload.get("source_sha256", {})
    for condition in ("hifics_zero_shot_mask", "hifics_adapted_mask"):
        path = Path(str(sources.get(condition, ""))).expanduser().resolve()
        if not path.is_file() or hashes.get(condition) != sha256_file(path):
            raise WorkflowBlocked(
                f"predicted-mask selection source is missing or stale for {condition}"
            )
    return selected


def _formal_conditions(run_dir: Path) -> tuple[str, str, str]:
    # The validation selector decides the primary predicted result, not which
    # artifacts exist.  All three arms are required for the pre-registered A8
    # grounding ablation and are generated without consulting test metrics.
    _selected_predicted_condition(run_dir)
    return (
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    )


def _run_masks_stage(
    args: argparse.Namespace, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    from .formal_inputs import run_predicted_mask_stage
    from .grounding_pipeline import (
        run_grounding_metric_stage,
        run_hifi_adaptation_stage,
        select_predicted_condition,
    )
    from .stages import run_oracle_mask_stage

    target, language = _manifest_paths(run_dir)
    requested = str(getattr(args, "condition", "all"))
    summaries: dict[str, Any] = {}
    if requested in {"all", "oracle_gt_mask"}:
        summaries["oracle_gt_mask"] = _record_value(
            run_oracle_mask_stage(target, language, run_dir, resume=args.resume)
        )
    device = str(config.get("vgn", {}).get("device", "cpu"))
    if requested in {"all", "hifics_zero_shot_mask"}:
        summaries["hifics_zero_shot_mask"] = _record_value(
            run_predicted_mask_stage(
                target,
                language,
                run_dir,
                condition="hifics_zero_shot_mask",
                device=device,
                resume=args.resume,
            )
        )
    adaptation: dict[str, Any] | None = None
    if requested in {"all", "hifics_adapted_mask"}:
        adaptation = run_hifi_adaptation_stage(
            target, language, run_dir, device=device, resume=args.resume
        )
        summaries["hifics_adapted_mask"] = _record_value(
            run_predicted_mask_stage(
                target,
                language,
                run_dir,
                condition="hifics_adapted_mask",
                device=device,
                resume=args.resume,
                adaptation_evidence_path=adaptation["evidence_path"],
                adapted_checkpoint_path=adaptation["adaptation_checkpoint_path"],
            )
        )
    metrics: dict[str, Any] = {}
    predicted_requested = [
        condition
        for condition in ("hifics_zero_shot_mask", "hifics_adapted_mask")
        if requested in {"all", condition}
    ]
    for condition in predicted_requested:
        metrics[condition] = run_grounding_metric_stage(
            target,
            language,
            run_dir,
            run_dir,
            condition=condition,
            resume=args.resume,
        )
    selection: dict[str, Any] | None = None
    if requested == "all":
        validation: dict[str, Any] = {}
        for condition in predicted_requested:
            validation[condition] = run_grounding_metric_stage(
                target,
                language,
                run_dir,
                run_dir,
                condition=condition,
                included_splits=("validation",),
                selection_scope=True,
                resume=args.resume,
            )
        selection = select_predicted_condition(
            Path(validation["hifics_zero_shot_mask"]["raw_metrics_path"]).parent
            / "summary.json",
            Path(validation["hifics_adapted_mask"]["raw_metrics_path"]).parent
            / "summary.json",
            run_dir / "predicted_condition_selection.json",
            resume=args.resume,
        )
    if metrics:
        _merge_csv_files(
            [Path(item["raw_metrics_path"]) for item in metrics.values()],
            run_dir / "grounding_metrics.csv",
        )
    result = {
        "requested_condition": requested,
        "mask_summaries": summaries,
        "adaptation": adaptation,
        "grounding_metrics": metrics,
        "predicted_condition_selection": selection,
    }
    record_stage(run_dir, "masks", "COMPLETE", **_record_value(result))
    return result


def _candidate_bundle_path(run_dir: Path, condition: str, group_id: str) -> Path:
    from .formal_inputs import group_artifact_slug

    return (
        run_dir / "vgn_candidates" / condition / f"{group_artifact_slug(group_id)}.json"
    )


def _label_bundle_path(run_dir: Path, condition: str, group_id: str) -> Path:
    from .formal_inputs import group_artifact_slug

    return (
        run_dir
        / "official_labels"
        / condition
        / f"{group_artifact_slug(group_id)}.json"
    )


def _indexed_candidate_bundles(
    run_dir: Path, condition: str, *, non_test_only: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    from .stages import load_frozen_candidate_bundle, load_target_language_jsonl

    target, language = _manifest_paths(run_dir)
    groups = load_target_language_jsonl(target, language)
    rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for group in groups:
        split = str(group.target.get("split", ""))
        if non_test_only and split == "test":
            continue
        path = _candidate_bundle_path(run_dir, condition, group.group_id)
        if not path.is_file():
            raise WorkflowBlocked(f"candidate bundle is missing: {path}")
        payload = load_frozen_candidate_bundle(
            path,
            group_id=group.group_id,
            grounding_condition=condition,
        )
        count = int(payload["candidate_count"])
        generation_status = str(payload["generation_status"])
        empty_reason = (
            str(payload["grounding_failure_reason"])
            if generation_status == "skipped_grounding_failure"
            else ("vgn_no_candidates" if count == 0 else None)
        )
        rows.append(
            {
                "group_id": group.group_id,
                "scene_id": str(group.target["scene_id"]),
                "split": split,
                "grounding_condition": condition,
                "candidate_count": count,
                "empty_pool": count == 0,
                "generation_status": generation_status,
                "empty_pool_reason": empty_reason,
                "inference_calls_for_group": int(payload["inference_calls_for_group"]),
                "candidate_pool_fingerprint": payload.get("candidate_pool_fingerprint"),
                "bundle_path": str(path),
                "bundle_sha256": sha256_file(path),
            }
        )
        paths[group.group_id] = path
    return rows, paths


def _write_candidate_indexes(
    run_dir: Path, conditions: tuple[str, ...]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_rows, _ = _indexed_candidate_bundles(run_dir, condition)
        rows.extend(condition_rows)
    atomic_jsonl(run_dir / "candidate_manifest.jsonl", rows)
    _atomic_csv_rows(run_dir / "candidate_pool_summary.csv", rows)
    fingerprint = canonical_sha256(rows)
    from .provenance import update_manifest

    update_manifest(run_dir, candidate_cache_hash=fingerprint)
    return {
        "candidate_manifest_sha256": sha256_file(run_dir / "candidate_manifest.jsonl"),
        "candidate_pool_summary_sha256": sha256_file(
            run_dir / "candidate_pool_summary.csv"
        ),
        "candidate_cache_hash": fingerprint,
        "groups": len(rows),
        "candidates": sum(int(row["candidate_count"]) for row in rows),
    }


def _extraction_config(config: dict[str, Any]) -> Any:
    from .vgn import ExtractionConfig

    vgn = config["vgn"]
    nms = config["nms"]
    return ExtractionConfig(
        pre_nms_max_candidates=int(vgn["pre_nms_max_candidates"]),
        frozen_top_k=int(vgn["frozen_top_k"]),
        translation_threshold_m=float(nms["translation_threshold_m"]),
        rotation_threshold_deg=float(nms["rotation_threshold_deg"]),
        width_threshold_m=float(nms["width_threshold_m"]),
    )


def _geometry_probe_manifests(
    run_dir: Path, *, maximum_groups: int = 100
) -> tuple[Path, Path, list[str]]:
    from .stages import load_jsonl_records

    target, language = _manifest_paths(run_dir)
    targets = load_jsonl_records(target, description="target manifest")
    languages = load_jsonl_records(language, description="language manifest")
    selected_targets = [
        row for row in targets if str(row.get("split")) in {"train", "validation"}
    ][:maximum_groups]
    selected_ids = [str(row["group_id"]) for row in selected_targets]
    language_by_id = {str(row["group_id"]): row for row in languages}
    if len(selected_targets) < 20 or any(
        group_id not in language_by_id for group_id in selected_ids
    ):
        raise WorkflowBlocked(
            "geometry validation requires at least twenty paired train/validation target groups"
        )
    root = run_dir / "geometry_validation" / "probe_manifests"
    target_path = root / "target_groups.jsonl"
    language_path = root / "language_queries.jsonl"
    atomic_jsonl(target_path, selected_targets)
    atomic_jsonl(language_path, [language_by_id[group_id] for group_id in selected_ids])
    return target_path, language_path, selected_ids


def _run_candidates_stage(
    args: argparse.Namespace, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    from .formal_validation import (
        AI_ASSISTED_GROUP_CHECKS,
        AI_ASSISTED_REVIEW_DISCLAIMER,
        AI_ASSISTED_REVIEW_KIND,
        FORMAL_GEOMETRY_MINIMUM_GROUPS,
        VISUAL_REVIEW_SCHEMA,
        FormalValidationError,
        run_geometry_validation,
    )
    from .geometry_probe import (
        discover_geometry_probe_group_ids,
        promote_geometry_probe_bundles,
        run_geometry_probe_stage,
        select_nonempty_geometry_probe_paths,
    )
    from .stages import StageBatchError, run_tsdf_stage, run_vgn_candidate_stage

    target, language = _manifest_paths(run_dir)
    conditions = _formal_conditions(run_dir)
    extraction = _extraction_config(config)
    device = str(config["vgn"].get("device", "cpu"))
    tsdf_summaries: dict[str, Any] = {}
    for condition in conditions:
        tsdf_summaries[condition] = _record_value(
            run_tsdf_stage(
                target,
                language,
                run_dir,
                resume=args.resume,
                grounding_condition=condition,
                mask_output_root=run_dir,
            )
        )

    geometry_root = run_dir / "geometry_validation"
    geometry_contract = geometry_root / "evaluator_geometry_contract.json"
    geometry_result: Any = None
    probe_summary: Any = None
    promoted_ids: list[str] = []
    if not geometry_contract.is_file():
        probe_target, probe_language, probe_ids = _geometry_probe_manifests(run_dir)
        try:
            probe_summary = run_geometry_probe_stage(
                probe_target,
                probe_language,
                run_dir,
                checkpoint=repository_root() / config["vgn"]["checkpoint"],
                device=device,
                grounding_condition="oracle_gt_mask",
                config=extraction,
                resume=args.resume,
            )
        except StageBatchError as error:
            # Empty local-maxima pools are expected observations.  They remain
            # fully recorded, while any different probe error still aborts.
            if not error.failures or any(
                "empty candidate pool" not in failure.message
                for failure in error.failures
            ):
                raise
            probe_summary = error.to_record()
        from .formal_inputs import group_artifact_slug

        available = [
            group_id
            for group_id in probe_ids
            if (
                run_dir
                / "geometry_probe"
                / "raw_vgn_candidates"
                / "oracle_gt_mask"
                / f"{group_artifact_slug(group_id)}.json"
            ).is_file()
        ][:FORMAL_GEOMETRY_MINIMUM_GROUPS]
        if len(available) < FORMAL_GEOMETRY_MINIMUM_GROUPS:
            raise WorkflowBlocked(
                "geometry probe produced only "
                f"{len(available)} non-empty real groups; "
                f"{FORMAL_GEOMETRY_MINIMUM_GROUPS} are required"
            )
        raw_paths = select_nonempty_geometry_probe_paths(
            run_dir,
            available,
            grounding_condition="oracle_gt_mask",
            minimum_groups=FORMAL_GEOMETRY_MINIMUM_GROUPS,
        )
        tsdf_paths = {
            group_id: run_dir
            / "target_tsdf"
            / "oracle_gt_mask"
            / f"{group_artifact_slug(group_id)}.npz"
            for group_id in available
        }
        geometry = config["candidate_geometry"]
        review = geometry_root / "ai_assisted_visual_review.json"
        binding_paths = {
            "data_manifest_sha256": run_dir / "data_validation.json",
            "group_manifest_sha256": target,
            "tsdf_config_sha256": run_dir / "resolved_config.yaml",
            "extraction_config_sha256": run_dir / "resolved_config.yaml",
            "upstream_versions_sha256": repository_root()
            / "configs"
            / "graspnet6d"
            / "upstream_versions.lock",
        }
        try:
            geometry_result = run_geometry_validation(
                target,
                available,
                tsdf_paths,
                raw_paths,
                geometry_root,
                R_vgn_gripper_to_graspnet_gripper=np.asarray(
                    geometry["R_vgn_gripper_to_graspnet_gripper"], dtype=float
                ),
                height_m=float(geometry["graspnet_height_m"]),
                depth_m=float(geometry["graspnet_depth_m"]),
                binding_paths=binding_paths,
                visual_review_path=review if review.is_file() else None,
            )
        except FormalValidationError as error:
            if "no AI-assisted visual review" in str(error):
                render_manifest = json.loads(
                    (geometry_root / "geometry_render_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                template = geometry_root / "ai_assisted_visual_review.TEMPLATE.json"
                atomic_json(
                    template,
                    {
                        "schema_version": VISUAL_REVIEW_SCHEMA,
                        "scope": "formal_real_data",
                        "fixture_only": False,
                        "status": "PENDING_AI_ASSISTED_REVIEW",
                        "review_kind": AI_ASSISTED_REVIEW_KIND,
                        "reviewer_system": "",
                        "independent_human_review": False,
                        "disclaimer": AI_ASSISTED_REVIEW_DISCLAIMER,
                        "reviewed_group_ids": render_manifest["group_ids"],
                        "checks_by_group": {
                            group_id: {
                                name: False for name in AI_ASSISTED_GROUP_CHECKS
                            }
                            for group_id in render_manifest["group_ids"]
                        },
                        "figure_sha256": render_manifest["figure_sha256"],
                    },
                )
                raise WorkflowBlocked(
                    "geometry audit figures are ready but formal conversion requires a "
                    "hashed AI-assisted visual review. Inspect geometry_validation/figures "
                    "with visual capability, create "
                    f"{review} from {template}, set every visual check explicitly, record "
                    "the reviewer_system, keep independent_human_review=false, and resume"
                ) from error
            raise
        geometry_contract = Path(geometry_result.contract_path)

    # Promote the exact raw geometry-probe pools after the contract passes.
    # This makes the subsequent full candidate stage resume those groups and
    # guarantees one and only one VGN forward call per group.
    _, _, possible_probe_ids = _geometry_probe_manifests(run_dir)
    observed_probe_ids = set(
        discover_geometry_probe_group_ids(run_dir, grounding_condition="oracle_gt_mask")
    )
    promoted_ids = [
        group_id for group_id in possible_probe_ids if group_id in observed_probe_ids
    ]
    promotion_summary: Any = None
    if promoted_ids:
        promotion_summary = promote_geometry_probe_bundles(
            target,
            language,
            run_dir,
            geometry_contract_path=geometry_contract,
            selected_group_ids=promoted_ids,
            checkpoint=repository_root() / config["vgn"]["checkpoint"],
            device=device,
            grounding_condition="oracle_gt_mask",
            config=extraction,
        )

    candidate_summaries: dict[str, Any] = {}
    for condition in conditions:
        candidate_summaries[condition] = _record_value(
            run_vgn_candidate_stage(
                target,
                language,
                run_dir,
                geometry_contract_path=geometry_contract,
                checkpoint=repository_root() / config["vgn"]["checkpoint"],
                device=device,
                grounding_condition=condition,
                config=extraction,
                resume=args.resume
                or (condition == "oracle_gt_mask" and bool(promoted_ids)),
            )
        )
    index = _write_candidate_indexes(run_dir, conditions)
    result = {
        "conditions": list(conditions),
        "tsdf_summaries": tsdf_summaries,
        "geometry_probe": _record_value(probe_summary),
        "geometry_validation": _record_value(geometry_result),
        "geometry_probe_promotion": _record_value(promotion_summary),
        "geometry_contract_path": str(geometry_contract),
        "geometry_contract_sha256": sha256_file(geometry_contract),
        "candidate_summaries": candidate_summaries,
        "candidate_index": index,
    }
    record_stage(run_dir, "candidates", "COMPLETE", **result)
    return result


def _run_labels_stage(
    args: argparse.Namespace, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    from .formal_validation import run_evaluator_parity_validation
    from .stages import run_official_label_stage

    target, language = _manifest_paths(run_dir)
    conditions = _formal_conditions(run_dir)
    parity_root = run_dir / "evaluator_parity"
    parity_gate = parity_root / "evaluator_parity_gate.json"
    parity_result: Any = None
    if not parity_gate.is_file():
        oracle_rows, oracle_paths = _indexed_candidate_bundles(
            run_dir, "oracle_gt_mask", non_test_only=True
        )
        selected = [
            str(row["group_id"])
            for row in oracle_rows
            if int(row["candidate_count"]) > 0
        ][:10]
        if not selected:
            raise WorkflowBlocked(
                "no non-empty train/validation frozen pool is available for evaluator parity"
            )
        parity_result = run_evaluator_parity_validation(
            target,
            selected,
            {group_id: oracle_paths[group_id] for group_id in selected},
            parity_root,
            dataset_root=repository_root() / "data_external" / "graspnet",
        )
        parity_gate = Path(parity_result.gate_path)
    summaries: dict[str, Any] = {}
    for condition in conditions:
        summaries[condition] = _record_value(
            run_official_label_stage(
                target,
                language,
                run_dir,
                dataset_root=repository_root() / "data_external" / "graspnet",
                parity_gate_path=parity_gate,
                grounding_condition=condition,
                resume=args.resume,
            )
        )
    result = {
        "conditions": list(conditions),
        "parity_gate_path": str(parity_gate),
        "parity_gate_sha256": sha256_file(parity_gate),
        "parity_validation": _record_value(parity_result),
        "label_summaries": summaries,
    }
    record_stage(run_dir, "labels", "COMPLETE", **result)
    return result


def _run_features_stage(
    args: argparse.Namespace, run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    del config
    from .formal_inputs import run_formal_feature_stage

    target, language = _manifest_paths(run_dir)
    conditions = _formal_conditions(run_dir)
    summaries: dict[str, Any] = {}
    for condition in conditions:
        summaries[condition] = _record_value(
            run_formal_feature_stage(
                target,
                language,
                run_dir,
                condition=condition,
                mask_output_root=run_dir,
                candidate_output_root=run_dir,
                resume=args.resume,
            )
        )
    result = {"conditions": list(conditions), "feature_summaries": summaries}
    record_stage(run_dir, "features", "COMPLETE", **result)
    return result


def _assemble_analysis_inputs(run_dir: Path) -> dict[str, Any]:
    from .analysis_inputs import (
        assemble_analysis_inputs,
        assemble_combined_analysis_inputs,
    )

    target, language = _manifest_paths(run_dir)
    assemblies: dict[str, Any] = {}
    for condition in _formal_conditions(run_dir):
        assemblies[condition] = assemble_analysis_inputs(
            target,
            language,
            run_dir,
            run_dir,
            condition=condition,
            run_id=run_dir.name,
        )
    assemblies["combined_a8"] = assemble_combined_analysis_inputs(
        target,
        language,
        run_dir,
        run_dir,
        run_id=run_dir.name,
    )
    return assemblies


def _require_formal_profile(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    profile = str(manifest.get("profile", ""))
    _, config = _config(profile)
    if config.get("formal_results") is not True:
        raise WorkflowBlocked(
            f"profile {profile!r} is diagnostic-only and cannot emit formal analysis/results"
        )


def _require_smoke_go(run_dir: Path) -> None:
    checks = run_smoke(run_dir / "smoke", run_dir=run_dir)
    failed = [check.number for check in checks if not check.passed]
    if failed:
        raise WorkflowBlocked(
            f"formal analysis is barred until real-data smoke checks pass; failed={failed}"
        )


def _run_analysis_stage(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from .experiment_analysis import AnalysisConfig, run_post_feature_analysis

    _require_formal_profile(run_dir)
    _require_smoke_go(run_dir)
    assemblies = _assemble_analysis_inputs(run_dir)
    analyses: dict[str, Any] = {}
    for name, assembly in assemblies.items():
        analyses[name] = run_post_feature_analysis(
            assembly.manifest_path,
            run_dir / "analysis" / name,
            config=AnalysisConfig(),
            resume=args.resume,
        )
    result = {
        "selected_predicted_condition": _selected_predicted_condition(run_dir),
        "assemblies": {
            name: _record_value(value) for name, value in assemblies.items()
        },
        "analyses": {
            name: {
                "output_dir": str(value.output_dir),
                "analysis_fingerprint": value.analysis_fingerprint,
                "resumed": value.resumed,
                "manifest_sha256": sha256_file(
                    value.output_dir / "analysis_manifest.json"
                ),
            }
            for name, value in analyses.items()
        },
    }
    record_stage(run_dir, "train-ranker", "COMPLETE", **result)
    return result


def _run_real_ranker_smoke_stage(
    args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    from .analysis_inputs import assemble_analysis_inputs
    from .smoke import run_real_ranker_smoke

    target, language = _manifest_paths(run_dir)
    assembly = assemble_analysis_inputs(
        target,
        language,
        run_dir,
        run_dir,
        condition="oracle_gt_mask",
        run_id=run_dir.name,
    )
    evidence = run_real_ranker_smoke(
        assembly.partition_rows["train"],
        assembly.partition_rows["validation"],
        assembly.partition_rows["test"],
        (
            assembly.partition_group_universes["train"],
            assembly.partition_group_universes["validation"],
            assembly.partition_group_universes["test"],
        ),
        repository_root() / "configs" / "graspnet6d" / "feature_schema_6d_v1.json",
        run_dir / "smoke",
        minimum_real_groups=20,
        resume=args.resume,
    )
    record_stage(run_dir, "real-ranker-smoke", "COMPLETE", evidence=evidence)
    return evidence


def _resume_analyses(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from .experiment_analysis import AnalysisConfig, run_post_feature_analysis

    del args
    _require_formal_profile(run_dir)
    _require_smoke_go(run_dir)
    assemblies = _assemble_analysis_inputs(run_dir)
    return {
        name: run_post_feature_analysis(
            assembly.manifest_path,
            run_dir / "analysis" / name,
            config=AnalysisConfig(),
            resume=True,
        )
        for name, assembly in assemblies.items()
    }


def _run_evaluate_stage(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    analyses = _resume_analyses(args, run_dir)
    outputs: dict[str, Any] = {}
    for name, result in analyses.items():
        required = (
            "metrics.csv",
            "metrics.json",
            "paired_outcomes.csv",
            "bootstrap_results.csv",
            "significance_tests.json",
            "frozen_pool_audit.json",
        )
        hashes = {
            filename: sha256_file(result.output_dir / filename) for filename in required
        }
        outputs[name] = hashes
    record_stage(run_dir, "evaluate", "COMPLETE", outputs=outputs)
    return outputs


def _run_ablate_stage(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    analyses = _resume_analyses(args, run_dir)
    outputs: dict[str, Any] = {}
    for name, result in analyses.items():
        required = (
            "ablation_results.csv",
            "failure_taxonomy.csv",
            "failure_summary.json",
        )
        outputs[name] = {
            filename: sha256_file(result.output_dir / filename) for filename in required
        }
    record_stage(run_dir, "ablate", "COMPLETE", outputs=outputs)
    return outputs


def _run_report_stage(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from .paper_artifacts import generate_formal_paper_artifacts
    from .run_outputs import publish_formal_run_outputs

    _require_formal_profile(run_dir)
    _require_smoke_go(run_dir)
    # Re-open and hash-verify every analysis before the publication preflight.
    # The paper finalizer owns all top-level prose/figures and is the only code
    # allowed to mark a formal run COMPLETE.
    _resume_analyses(args, run_dir)
    aggregate = publish_formal_run_outputs(run_dir, resume=args.resume)
    result = generate_formal_paper_artifacts(run_dir, resume=args.resume)
    record_stage(
        run_dir,
        "report",
        "COMPLETE",
        paper_artifact_manifest=str(result.manifest_path),
        paper_artifact_manifest_sha256=result.manifest_sha256,
        input_fingerprint=result.input_fingerprint,
        selected_predicted_condition=result.selected_predicted_condition,
        executed_figures=list(result.executed_figures),
        unexecuted_figures=list(result.unexecuted_figures),
        run_outputs_manifest=str(aggregate.manifest_path),
        run_outputs_manifest_sha256=aggregate.manifest_sha256,
        resumed=result.resumed,
    )
    return {
        "paper_artifacts": _record_value(result),
        "formal_run_outputs": _record_value(aggregate),
    }


def _execute_one(args: argparse.Namespace, command: str) -> int:
    if command == "audit":
        result = run_audit()
        print(json.dumps(result, indent=2))
        return 0
    if command == "benchmark-device":
        print(json.dumps(run_device_benchmark(), indent=2))
        return 0
    if command == "smoke":
        active = getattr(args, "run_id", None) or _active_run_id()
        run_dir = (
            repository_root() / "artifacts" / "graspnet6d" / active if active else None
        )
        checks = run_smoke(
            None if run_dir is None else run_dir / "smoke",
            run_dir=run_dir,
        )
        passed = sum(check.passed for check in checks)
        print(
            f"smoke checks passed: {passed}/{len(checks)}; see artifacts/graspnet6d/smoke/GO_NO_GO.md"
        )
        return 0 if passed == len(checks) else 2

    run_dir = _run_dir(args, command=command)
    _, config = _config(args.profile)
    started = time.perf_counter()
    try:
        if command == "download":
            evidence = download(args.profile, run_dir=run_dir)
            print(json.dumps(evidence, indent=2))
        elif command == "prepare":
            print(
                json.dumps(
                    prepare_experiment(
                        args.profile, run_dir=run_dir, resume=args.resume
                    ),
                    indent=2,
                )
            )
        elif command == "masks":
            print(
                json.dumps(
                    _record_value(_run_masks_stage(args, run_dir, config)), indent=2
                )
            )
        elif command == "candidates":
            print(
                json.dumps(
                    _record_value(_run_candidates_stage(args, run_dir, config)),
                    indent=2,
                )
            )
        elif command == "labels":
            print(
                json.dumps(
                    _record_value(_run_labels_stage(args, run_dir, config)), indent=2
                )
            )
        elif command == "features":
            print(
                json.dumps(
                    _record_value(_run_features_stage(args, run_dir, config)), indent=2
                )
            )
        elif command == "train-ranker":
            print(
                json.dumps(_record_value(_run_analysis_stage(args, run_dir)), indent=2)
            )
        elif command == "evaluate":
            print(
                json.dumps(_record_value(_run_evaluate_stage(args, run_dir)), indent=2)
            )
        elif command == "ablate":
            print(json.dumps(_record_value(_run_ablate_stage(args, run_dir)), indent=2))
        elif command == "report":
            print(json.dumps(_record_value(_run_report_stage(args, run_dir)), indent=2))
        else:
            raise AssertionError(command)
        # The paper finalizer binds a canonical projection of the eleven
        # pre-report rows, so the report duration itself can be recorded after
        # publication without making the figure input self-referential.
        _record_runtime(
            run_dir,
            command,
            started=started,
            status="COMPLETE",
            resumed=args.resume,
        )
        return 0
    except Exception as error:
        record_failure(run_dir, command, error)
        expected_block = isinstance(error, (WorkflowBlocked, DatasetValidationError))
        if not expected_block:
            _mark_run_failed(run_dir, command, error)
        _record_runtime(
            run_dir,
            command,
            started=started,
            status="BLOCKED" if isinstance(error, WorkflowBlocked) else "FAILED",
            resumed=args.resume,
        )
        if expected_block:
            write_blocked_status(run_dir, command, str(error))
            print(f"BLOCKED: {error}", file=sys.stderr)
            return 2
        raise


def _run_all(args: argparse.Namespace) -> int:
    # Refresh host/software evidence before freezing a new run identity.  A
    # resume deliberately reuses that immutable snapshot; rewriting global
    # audit timestamps mid-run would make its hash-bound environment artifact
    # stale even though the machine and software had not changed.
    active = args.run_id or _active_run_id()
    resume_existing = False
    if args.resume and active:
        manifest_path = (
            repository_root()
            / "artifacts"
            / "graspnet6d"
            / active
            / "run_manifest.json"
        )
        try:
            resume_existing = (
                json.loads(manifest_path.read_text(encoding="utf-8")).get("profile")
                == args.profile
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            resume_existing = False
    if not resume_existing:
        run_audit()
        run_device_benchmark()
    run_dir = _run_dir(args, command="all")
    _, config = _config(args.profile)
    stage_functions = {
        "masks": lambda: _run_masks_stage(args, run_dir, config),
        "candidates": lambda: _run_candidates_stage(args, run_dir, config),
        "labels": lambda: _run_labels_stage(args, run_dir, config),
        "features": lambda: _run_features_stage(args, run_dir, config),
        "train-ranker": lambda: _run_analysis_stage(args, run_dir),
        "evaluate": lambda: _run_evaluate_stage(args, run_dir),
        "ablate": lambda: _run_ablate_stage(args, run_dir),
        "report": lambda: _run_report_stage(args, run_dir),
    }
    # Data-backed smoke evidence must exist before the formal analysis/report.
    # Candidate/label/feature construction below is the measurable system
    # under test; the small real-ranker probe is explicitly non-reportable.
    smoke_sequence = (
        "download",
        "prepare",
        "masks",
        "candidates",
        "labels",
        "features",
        "real-ranker-smoke",
        "smoke",
    )
    formal_sequence = (
        "train-ranker",
        "evaluate",
        "ablate",
        "report",
    )
    sequence = smoke_sequence + (
        formal_sequence if config.get("formal_results") is True else ()
    )
    for command in sequence:
        started = time.perf_counter()
        try:
            if command == "download":
                download(args.profile, run_dir=run_dir)
            elif command == "prepare":
                prepare_experiment(args.profile, run_dir=run_dir, resume=args.resume)
            elif command == "real-ranker-smoke":
                _run_real_ranker_smoke_stage(args, run_dir)
            elif command == "smoke":
                checks = run_smoke(run_dir / "smoke", run_dir=run_dir)
                if not all(check.passed for check in checks):
                    raise WorkflowBlocked(
                        "smoke GO/NO-GO has not passed all twelve real-data checks"
                    )
            else:
                stage_functions[command]()
            _record_runtime(
                run_dir,
                command,
                started=started,
                status="COMPLETE",
                resumed=args.resume,
            )
        except Exception as error:
            record_failure(run_dir, command, error)
            expected_block = isinstance(
                error, (WorkflowBlocked, DatasetValidationError)
            )
            if not expected_block:
                _mark_run_failed(run_dir, command, error)
            _record_runtime(
                run_dir,
                command,
                started=started,
                status="BLOCKED" if expected_block else "FAILED",
                resumed=args.resume,
            )
            if expected_block:
                write_blocked_status(run_dir, command, str(error))
                print(f"BLOCKED: {error}", file=sys.stderr)
                return 2
            raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "benchmark-device"):
        subparsers.add_parser(name)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--run-id")
    for name in ("download", "prepare", *STAGES, "all"):
        child = subparsers.add_parser(name)
        child.add_argument("--profile", type=_profile_name, default="paper-lite")
        child.add_argument("--run-id")
        child.add_argument("--resume", action="store_true")
        if name == "masks":
            child.add_argument(
                "--condition",
                choices=(
                    "all",
                    "oracle_gt_mask",
                    "hifics_zero_shot_mask",
                    "hifics_adapted_mask",
                ),
                default="all",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        return _run_all(args)
    return _execute_one(args, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
