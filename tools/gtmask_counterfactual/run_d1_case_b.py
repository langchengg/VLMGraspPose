#!/usr/bin/env python3
"""Prepare, plan, or explicitly execute the locked D1 Case-B counterfactual."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.d1_adapter import (  # noqa: E402
    D1AdapterError,
    _safe_id,
    assemble_d1_scored_outputs,
    build_frozen_scorer_command,
    build_gt_candidate_command,
    build_isolated_gt_bundle_root,
    build_predicted_replay_command,
    candidate_inventory_from_summary,
    machine_blocker_payload,
    verify_frozen_d1_sources,
    verify_isolated_bundle_root,
)
from gtmask_counterfactual.d1_source_view import (  # noqa: E402
    build_d1_source_view,
    verify_d1_source_view,
)
from gtmask_counterfactual.execution import (  # noqa: E402
    FROZEN_D1_SOURCE,
    append_gt_access_log,
    artifact_record,
    authorize_gt_candidate_generation,
    counterfactual_job,
    probe_frozen_docker_image,
)
from gtmask_counterfactual.io import atomic_json, sha256_file  # noqa: E402
from gtmask_counterfactual.resource import (  # noqa: E402
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)
from gtmask_counterfactual.audit import (  # noqa: E402
    RunState,
    require_verified_source,
    transition_pipeline_status,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


EXPECTED_SAMPLE_COUNT = 7_675
CANDIDATE_PYTHON = ROOT / "HiFi_reproduction/.venv-gqcnn/bin/python"
BLOCKER_RELATIVE_PATH = Path("00_audit/machine_blockers/D1_CASE_B_MACHINE_BLOCKER.json")
REGISTRY_COLUMNS = (
    "sample_id",
    "mapping_status",
    "pixel_qa_status",
    "bulk_gt_pixels_read",
    "original_gt_mask_path",
    "original_gt_mask_sha256",
)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=False)
    if path.expanduser().is_symlink() or not source.is_file():
        raise D1AdapterError(f"{label} must be a regular non-symlink file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1AdapterError(f"cannot parse {label}: {source}") from error
    if not isinstance(value, dict):
        raise D1AdapterError(f"{label} must contain one JSON object")
    return value


def _load_registry_after_authorization(
    registry: Path, authority: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Read registry rows only after the P4 exactly-once guard has succeeded."""

    locked = authority.get("gt_mask_registry")
    source = registry.expanduser().resolve(strict=False)
    if not isinstance(locked, Mapping) or (
        Path(str(locked.get("path", ""))).resolve(strict=False) != source
        or locked.get("sha256") != sha256_file(source)
    ):
        raise PermissionError("registry read attempted without matching P4 authority")
    import pyarrow.parquet as pq

    return pq.read_table(source, columns=list(REGISTRY_COLUMNS)).to_pylist()


def _authorise(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    return authorize_gt_candidate_generation(
        protocol_lock_path=args.protocol_lock,
        registry_path=args.registry,
        route="d1",
        branch="gt_oracle",
    )


def _fresh_gate(path: Path) -> dict[str, Any]:
    value = _json_object(path, label="D1 resource gate")
    validate_fresh_gate(value)
    return value


def _resume_command() -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]


def _write_machine_blocker(
    *, run_dir: Path, probe: Mapping[str, Any], resume_command: list[str]
) -> Path:
    blocker = machine_blocker_payload(
        blocker_code=str(probe.get("blocker_code", "D1_DOCKER_UNAVAILABLE")),
        detail=str(
            probe.get("diagnostic", "frozen Docker runtime/image is unavailable")
        ),
        resume_command=resume_command,
        docker_probe=probe,
    )
    destination = run_dir.expanduser().resolve() / BLOCKER_RELATIVE_PATH
    atomic_json(destination, blocker)
    return destination


def _docker_or_block(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    probe = probe_frozen_docker_image()
    if probe.get("status") == "PASS":
        return probe, None
    blocker = _write_machine_blocker(
        run_dir=args.run_dir,
        probe=probe,
        resume_command=_resume_command(),
    )
    return probe, blocker


def _bundle_sample_ids(bundle: Mapping[str, Any]) -> set[str]:
    records = bundle.get("samples")
    if not isinstance(records, list):
        raise D1AdapterError("D1 bundle lacks its executable sample inventory")
    identities = {
        str(record.get("sample_id", ""))
        for record in records
        if isinstance(record, Mapping)
    }
    if len(identities) != len(records) or "" in identities:
        raise D1AdapterError("D1 bundle sample identities differ")
    if (
        int(bundle.get("denominator_sample_count", -1)) != EXPECTED_SAMPLE_COUNT
        or int(bundle.get("evaluable_sample_count", -1)) != len(identities)
        or int(bundle.get("unresolved_sample_count", -1))
        != EXPECTED_SAMPLE_COUNT - len(identities)
    ):
        raise D1AdapterError("D1 bundle denominator partition differs")
    return identities


def _candidate_arguments(
    args: argparse.Namespace, *, source_view: Path | None = None
) -> list[str]:
    config = (
        verify_frozen_d1_sources()["config"]["path"]
        if source_view is None
        else source_view / "configs/dexnet_candidates_formal_no_refinement.yaml"
    )
    values = [
        "--dataset-root",
        str(args.dataset_root.expanduser().resolve()),
        "--mask-root",
        str(args.bundle_root.expanduser().resolve()),
        "--output-dir",
        str(args.candidate_root.expanduser().resolve()),
        "--config",
        str(config),
        "--mode",
        "candidate-only",
        "--num-candidates",
        "256",
        "--top-k",
        "30",
        "--seed",
        "42",
        "--sample-seed-mode",
        "stable-sha256",
        "--seed-namespace",
        "hierfilm-modular-formal-v1",
        "--visualize-policy",
        "none",
        "--status-every",
        "25",
        "--checkpoint-every",
        "25",
        "--max-failures",
        "1",
    ]
    if args.resume:
        values.extend(["--resume", "--verify-existing", "--retry-failures"])
    return values


def _source_view(run_dir: Path) -> tuple[Path, Path]:
    manifest = build_d1_source_view(
        run_dir.expanduser().resolve()
        / "04_predicted_replay/d1/source_adapter"
    )
    verify_d1_source_view(manifest)
    return manifest.parent, manifest


def _predicted_authority(args: argparse.Namespace) -> set[str]:
    """Verify source locks and return the label-free D1 Test denominator."""

    require_verified_source(args.run_dir, source_name="d1", source_run=args.d1_run)
    source = FROZEN_D1_SOURCE.resolve()
    run_config_path = source / "candidates/hierfilm/run_config.json"
    run_config = _json_object(run_config_path, label="frozen D1 candidate run config")
    identity = run_config.get("identity")
    if not isinstance(identity, Mapping):
        raise D1AdapterError("frozen D1 candidate identity is malformed")
    predicted_root = args.predicted_root.expanduser().resolve()
    predicted_manifest = predicted_root / "manifest.jsonl"
    if (
        Path(str(identity.get("mask_root", ""))).resolve(strict=False)
        != predicted_root
        or Path(str(identity.get("manifest", ""))).resolve(strict=False)
        != predicted_manifest
        or identity.get("manifest_sha256") != sha256_file(predicted_manifest)
    ):
        raise D1AdapterError("predicted bundle authority differs from frozen D1")
    d1_root = args.d1_run.expanduser().resolve()
    candidate_manifest = _json_object(
        d1_root / "02_candidates/test_manifest.json",
        label="D1 candidate manifest",
    )
    configuration = candidate_manifest.get("configuration")
    paired = (
        configuration.get("paired_manifest")
        if isinstance(configuration, Mapping)
        else None
    )
    if not isinstance(paired, Mapping):
        raise D1AdapterError("D1 candidate manifest lacks its denominator")
    paired_path = Path(str(paired.get("path", ""))).expanduser().resolve()
    if paired.get("sha256") != sha256_file(paired_path):
        raise D1AdapterError("D1 paired denominator hash differs")
    import pyarrow.parquet as pq

    rows = pq.read_table(paired_path, columns=["sample_id"]).to_pylist()
    identifiers = {_safe_id(row.get("sample_id")) for row in rows}
    if len(rows) != EXPECTED_SAMPLE_COUNT or len(identifiers) != EXPECTED_SAMPLE_COUNT:
        raise D1AdapterError("D1 predicted denominator differs")
    return identifiers


def prepare(args: argparse.Namespace) -> int:
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B bundle materialisation"):
        _fresh_gate(args.resource_gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_bundle_materialisation_launch",
        )
        authority, _ = _authorise(args)
        rows = _load_registry_after_authorization(args.registry, authority)
        append_gt_access_log(
            args.run_dir,
            {
                "stage": "D1_CASE_B_BUNDLE_MATERIALISATION",
                "route": "d1",
                "branch": "gt_oracle",
                "purpose": "locked_case_b_input_materialisation",
                "protocol_lock": artifact_record(args.protocol_lock),
                "gt_registry": artifact_record(args.registry),
                "resource_gate": artifact_record(args.resource_gate),
                "gt_rows_read": len(rows),
            },
        )
        root = build_isolated_gt_bundle_root(
            predicted_root=args.predicted_root,
            output_root=args.bundle_root,
            registry_path=args.registry,
            registry_rows=rows,
            authority=authority,
            expected_count=args.expected_count,
        )
    print(json.dumps({"status": "COMPLETE", "bundle_root": str(root)}, sort_keys=True))
    return 0


def print_predicted_replay(args: argparse.Namespace) -> int:
    sample_ids = _predicted_authority(args)
    _fresh_gate(args.resource_gate)
    source_view, source_manifest = _source_view(args.run_dir)
    command = build_predicted_replay_command(
        python=CANDIDATE_PYTHON,
        dataset_root=args.dataset_root,
        predicted_root=args.predicted_root,
        output_root=args.candidate_root,
        source_view_root=source_view,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "gt_pixels_read": 0,
                "sample_count": len(sample_ids),
                "command": command,
                "source_view": artifact_record(source_manifest),
                "execution_available": True,
                "next": "execute-predicted-candidates, score-predicted, close-predicted",
            },
            sort_keys=True,
        )
    )
    return 0


def execute_predicted_candidates(args: argparse.Namespace) -> int:
    """Run the frozen predicted-mask Case-B candidate generator."""

    expected_ids = _predicted_authority(args)
    candidate_root = args.candidate_root.expanduser().resolve(strict=False)
    run = args.run_dir.expanduser().resolve()
    if run not in candidate_root.parents:
        raise D1AdapterError("predicted candidate root must be inside the new run")
    source_view, source_manifest = _source_view(run)
    with exclusive_d1_flock(run, purpose="D1 predicted candidate replay"):
        _fresh_gate(args.resource_gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_predicted_candidates_launch",
        )
        arguments = [
            "--dataset-root",
            str(args.dataset_root.expanduser().resolve()),
            "--mask-root",
            str(args.predicted_root.expanduser().resolve()),
            "--output-dir",
            str(candidate_root),
            "--config",
            str(source_view / "configs/dexnet_candidates_formal_no_refinement.yaml"),
            "--mode",
            "candidate-only",
            "--num-candidates",
            "256",
            "--top-k",
            "30",
            "--seed",
            "42",
            "--sample-seed-mode",
            "stable-sha256",
            "--seed-namespace",
            "hierfilm-modular-formal-v1",
            "--visualize-policy",
            "none",
            "--status-every",
            "25",
            "--checkpoint-every",
            "25",
            "--max-failures",
            "1",
        ]
        if args.resume:
            arguments.extend(["--resume", "--verify-existing", "--retry-failures"])
        candidate_script = source_view / "scripts/run_hifics_dexnet_candidates.py"
        command_values = [str(CANDIDATE_PYTHON.resolve()), str(candidate_script), *arguments]
        command = shlex.join(command_values)
        with ledger_stage(
            run / "run_ledger.sqlite",
            stage="P3_PREDICTED_REPLAY",
            substage="d1_candidates",
            route="d1",
            pool="allnms",
            command=command,
        ) as ledger:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(source_view)
            completed = subprocess.run(
                command_values,
                check=False,
                cwd=source_view,
                env=environment,
            )
            if completed.returncode != 0:
                raise D1AdapterError(
                    "frozen predicted candidate runner returned "
                    f"{completed.returncode}"
                )
            candidate_inventory_from_summary(
                candidate_root, expected_sample_ids=expected_ids
            )
            run_config = candidate_root / "run_config.json"
            ledger["artifact_path"] = str(run_config.resolve())
            ledger["artifact_sha256"] = sha256_file(run_config)
        append_gt_access_log(
            run,
            {
                "stage": "D1_PREDICTED_CANDIDATE_REPLAY",
                "route": "d1",
                "branch": "predicted",
                "purpose": "label_free_baseline_replay",
                "resource_gate": artifact_record(args.resource_gate),
                "source_view": artifact_record(source_manifest),
                "gt_rows_read": 0,
            },
        )
    return 0


def score_predicted(args: argparse.Namespace) -> int:
    """Score the regenerated predicted-mask candidate pool with frozen GQ-CNN."""

    expected_ids = _predicted_authority(args)
    run = args.run_dir.expanduser().resolve()
    candidate_root = args.candidate_root.expanduser().resolve(strict=False)
    scored_root = args.scored_root.expanduser().resolve(strict=False)
    if run not in candidate_root.parents or run not in scored_root.parents:
        raise D1AdapterError("predicted replay outputs must be inside the new run")
    source_view, source_manifest = _source_view(run)
    with exclusive_d1_flock(run, purpose="D1 predicted GQ-CNN replay"):
        _fresh_gate(args.resource_gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_predicted_scoring_launch",
        )
        inventory = candidate_inventory_from_summary(
            candidate_root, expected_sample_ids=expected_ids
        )
        probe, blocker = _docker_or_block(args)
        if blocker is not None:
            print(json.dumps({"status": "MACHINE_BLOCKED", "artifact": str(blocker)}))
            return 75
        docker = shutil.which("docker")
        if docker is None:
            raise D1AdapterError("Docker disappeared after a successful probe")
        command = build_frozen_scorer_command(
            docker=Path(docker),
            candidate_root=candidate_root,
            output_root=scored_root,
            inventory=inventory,
            source_view_root=source_view,
            resume=args.resume,
        )
        if not args.execute:
            print(json.dumps({"status": "READY", "command": command}, sort_keys=True))
            return 0
        with ledger_stage(
            run / "run_ledger.sqlite",
            stage="P3_PREDICTED_REPLAY",
            substage="d1_scoring",
            route="d1",
            pool="allnms",
            command=shlex.join(command),
        ) as ledger:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise D1AdapterError(
                    f"frozen predicted scorer returned {completed.returncode}"
                )
            run_config = scored_root / "run_config.json"
            ledger["artifact_path"] = str(run_config.resolve())
            ledger["artifact_sha256"] = sha256_file(run_config)
        append_gt_access_log(
            run,
            {
                "stage": "D1_PREDICTED_GQCNN_REPLAY",
                "route": "d1",
                "branch": "predicted",
                "purpose": "label_free_baseline_replay",
                "resource_gate": artifact_record(args.resource_gate),
                "source_view": artifact_record(source_manifest),
                "gt_rows_read": 0,
            },
        )
    return 0


def close_predicted(args: argparse.Namespace) -> int:
    """Independently compare regenerated D1 candidates/scores to frozen pools."""

    _predicted_authority(args)
    from gtmask_counterfactual.d1_predicted_replay import (
        build_d1_predicted_replay_manifest,
    )
    run = args.run_dir.expanduser().resolve()
    _, source_manifest = _source_view(run)
    with exclusive_d1_flock(run, purpose="D1 predicted replay semantic closure"):
        _fresh_gate(args.resource_gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_predicted_closure_launch",
        )
        with ledger_stage(
            run / "run_ledger.sqlite",
            stage="P3_PREDICTED_REPLAY",
            substage="d1_semantic_closure",
            route="d1",
            pool="allnms",
            command=shlex.join(_resume_command()),
        ) as ledger:
            manifest = build_d1_predicted_replay_manifest(
                run_dir=run,
                d1_run=args.d1_run,
                candidate_root=args.candidate_root,
                frozen_candidate_root=FROZEN_D1_SOURCE / "candidates/hierfilm",
                scored_root=args.scored_root,
                derived_reconciliation=args.derived_reconciliation,
                source_view_manifest=source_manifest,
            )
            ledger["artifact_path"] = str(manifest.resolve())
            ledger["artifact_sha256"] = sha256_file(manifest)
    print(json.dumps({"status": "PASS", "manifest": str(manifest)}, sort_keys=True))
    return 0


def print_plan(args: argparse.Namespace) -> int:
    _authorise(args)
    _fresh_gate(args.resource_gate)
    bundle = verify_isolated_bundle_root(args.bundle_root)
    source_view, source_manifest = _source_view(args.run_dir)
    registry = bundle.get("gt_mask_registry")
    if not isinstance(registry, Mapping) or (
        Path(str(registry.get("path", ""))).resolve(strict=False)
        != args.registry.expanduser().resolve()
        or registry.get("sha256") != sha256_file(args.registry)
    ):
        raise D1AdapterError("D1 bundle root differs from the locked registry")
    candidate = build_gt_candidate_command(
        python=Path(sys.executable),
        launcher=Path(__file__),
        run_dir=args.run_dir,
        protocol_lock=args.protocol_lock,
        registry_path=args.registry,
        resource_gate=args.resource_gate,
        dataset_root=args.dataset_root,
        bundle_root=args.bundle_root,
        output_root=args.candidate_root,
        resume=args.resume,
    )
    probe, blocker = _docker_or_block(args)
    result: dict[str, Any] = {
        "status": (
            "READY_TO_EXECUTE_GT_ORACLE_CASE_B"
            if blocker is None
            else "MACHINE_BLOCKED"
        ),
        "pipeline_ready": blocker is None,
        "d1_case": "B",
        "raw_candidate_regeneration_required": True,
        "filter_only_primary_allowed": False,
        "candidate_command": candidate,
        "source_view": artifact_record(source_manifest),
        "docker": probe,
    }
    if blocker is not None:
        result["machine_blocker"] = artifact_record(blocker)
    elif args.candidate_root.joinpath("summary.csv").is_file():
        inventory = candidate_inventory_from_summary(
            args.candidate_root, expected_sample_ids=_bundle_sample_ids(bundle)
        )
        docker = shutil.which("docker")
        if docker is None:
            raise D1AdapterError("Docker disappeared after a successful probe")
        result["scorer_command"] = build_frozen_scorer_command(
            docker=Path(docker),
            candidate_root=args.candidate_root,
            output_root=args.scored_root,
            inventory=inventory,
            source_view_root=source_view,
            resume=args.resume,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if blocker is None else 75


def execute_candidates(args: argparse.Namespace) -> int:
    authority, _ = _authorise(args)
    del authority
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B GT candidate generation"):
        _fresh_gate(args.resource_gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_oracle_candidate_launch",
        )
        bundle = verify_isolated_bundle_root(args.bundle_root)
        registry = bundle.get("gt_mask_registry")
        if not isinstance(registry, Mapping) or registry.get("sha256") != sha256_file(
            args.registry
        ):
            raise D1AdapterError("D1 bundle root differs from the locked registry")
        _, blocker = _docker_or_block(args)
        if blocker is not None:
            print(json.dumps({"status": "MACHINE_BLOCKED", "artifact": str(blocker)}))
            return 75
        append_gt_access_log(
            args.run_dir,
            {
                "stage": "D1_CASE_B_RAW_CANDIDATE_GENERATION",
                "route": "d1",
                "branch": "gt_oracle",
                "purpose": "locked_stage_replacement_counterfactual",
                "protocol_lock": artifact_record(args.protocol_lock),
                "gt_registry": artifact_record(args.registry),
                "bundle_manifest": artifact_record(
                    args.bundle_root / "D1_CASE_B_BUNDLE_MANIFEST.json"
                ),
                "filter_only": False,
            },
        )
        source_view, source_manifest = _source_view(args.run_dir)
        frozen_arguments = _candidate_arguments(args, source_view=source_view)
        bootstrap = source_view / "scripts/gtmask_oracle_candidate_bootstrap.py"
        bundle_manifest = args.bundle_root / "D1_CASE_B_BUNDLE_MANIFEST.json"
        command = [
            str(CANDIDATE_PYTHON),
            str(bootstrap),
            "--source-view",
            str(source_view),
            "--source-view-manifest-sha256",
            sha256_file(source_manifest),
            "--bundle-root",
            str(args.bundle_root.expanduser().resolve()),
            "--bundle-manifest-sha256",
            sha256_file(bundle_manifest),
            "--",
            *frozen_arguments,
        ]
        command_text = shlex.join(command)
        with counterfactual_job(
            args.run_dir,
            stage="P6_D1_RAW_CANDIDATES",
            route="d1",
            branch="gt_oracle",
            command=command_text,
        ) as job:
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONPATH": str(source_view),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                }
            )
            completed = subprocess.run(
                command, check=False, cwd=source_view, env=environment
            )
            if completed.returncode != 0:
                raise D1AdapterError(
                    f"frozen candidate runner returned {completed.returncode}"
                )
            candidate_inventory_from_summary(
                args.candidate_root, expected_sample_ids=_bundle_sample_ids(bundle)
            )
            run_config = args.candidate_root / "run_config.json"
            job["artifact_path"] = str(run_config.resolve())
            job["artifact_sha256"] = sha256_file(run_config)
            job["source_view_sha256"] = sha256_file(source_manifest)
    return 0


def scorer(args: argparse.Namespace) -> int:
    _authorise(args)
    if not args.execute:
        _fresh_gate(args.resource_gate)
        bundle = verify_isolated_bundle_root(args.bundle_root)
        inventory = candidate_inventory_from_summary(
            args.candidate_root, expected_sample_ids=_bundle_sample_ids(bundle)
        )
        probe, blocker = _docker_or_block(args)
        if blocker is not None:
            print(json.dumps({"status": "MACHINE_BLOCKED", "artifact": str(blocker)}))
            return 75
        docker = shutil.which("docker")
        if docker is None:
            raise D1AdapterError("Docker disappeared after a successful probe")
        source_view, _ = _source_view(args.run_dir)
        command = build_frozen_scorer_command(
            docker=Path(docker),
            candidate_root=args.candidate_root,
            output_root=args.scored_root,
            inventory=inventory,
            source_view_root=source_view,
            resume=args.resume,
        )
        print(json.dumps({"status": "READY", "command": command}, sort_keys=True))
        return 0
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B frozen GQ-CNN scoring"):
        _fresh_gate(args.resource_gate)
        bundle = verify_isolated_bundle_root(args.bundle_root)
        inventory = candidate_inventory_from_summary(
            args.candidate_root, expected_sample_ids=_bundle_sample_ids(bundle)
        )
        probe, blocker = _docker_or_block(args)
        if blocker is not None:
            print(json.dumps({"status": "MACHINE_BLOCKED", "artifact": str(blocker)}))
            return 75
        docker = shutil.which("docker")
        if docker is None:
            raise D1AdapterError("Docker disappeared after a successful probe")
        source_view, source_manifest = _source_view(args.run_dir)
        command = build_frozen_scorer_command(
            docker=Path(docker),
            candidate_root=args.candidate_root,
            output_root=args.scored_root,
            inventory=inventory,
            source_view_root=source_view,
            resume=args.resume,
        )
        with counterfactual_job(
            args.run_dir,
            stage="P6_D1_GQCNN_SCORING",
            route="d1",
            branch="gt_oracle",
            command=shlex.join(command),
        ) as job:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise D1AdapterError(f"frozen scorer returned {completed.returncode}")
            run_config = args.scored_root / "run_config.json"
            job["artifact_path"] = str(run_config.resolve())
            job["artifact_sha256"] = sha256_file(run_config)
            job["source_view_sha256"] = sha256_file(source_manifest)
    return 0


def assemble(args: argparse.Namespace) -> int:
    """Publish the only canonical D1 route output after frozen scoring."""

    _authorise(args)
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B canonical assembly"):
        _fresh_gate(args.resource_gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_canonical_assembly_launch",
        )
        bundle = verify_isolated_bundle_root(args.bundle_root)
        if int(bundle.get("denominator_sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
            raise D1AdapterError("D1 bundle denominator differs before assembly")
        append_gt_access_log(
            args.run_dir,
            {
                "stage": "D1_CASE_B_CANONICAL_ASSEMBLY",
                "route": "d1",
                "branch": "gt_oracle",
                "purpose": "label_free_scored_output_normalisation",
                "protocol_lock": artifact_record(args.protocol_lock),
                "gt_registry": artifact_record(args.registry),
                "bundle_manifest": artifact_record(
                    args.bundle_root / "D1_CASE_B_BUNDLE_MANIFEST.json"
                ),
                "gt_rows_read": 0,
            },
        )
        with counterfactual_job(
            args.run_dir,
            stage="P6_D1_CANONICAL_OUTPUT",
            route="d1",
            branch="gt_oracle",
            command=shlex.join(_resume_command()),
        ) as job:
            manifest = assemble_d1_scored_outputs(
                run_dir=args.run_dir,
                bundle_root=args.bundle_root,
                candidate_root=args.candidate_root,
                scored_root=args.scored_root,
                protocol_lock_path=args.protocol_lock,
                execution_claim_path=(
                    args.run_dir / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
                ),
            )
            job["artifact_path"] = str(manifest)
            job["artifact_sha256"] = sha256_file(manifest)
        transition_pipeline_status(
            args.run_dir,
            RunState.P6_D1_COUNTERFACTUAL_COMPLETE,
            first_incomplete_stage=RunState.P7_TAXONOMY_COMPLETE.value,
        )
    print(json.dumps({"status": "COMPLETE", "manifest": str(manifest)}, sort_keys=True))
    return 0


def _add_authority(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)


def _add_heavy_authority(parser: argparse.ArgumentParser) -> None:
    _add_authority(parser)
    parser.add_argument("--resource-gate", type=Path, required=True)
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )


def _add_execution_paths(parser: argparse.ArgumentParser) -> None:
    _add_heavy_authority(parser)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path)
    parser.add_argument("--resume", action="store_true")


def _add_assembly_paths(parser: argparse.ArgumentParser) -> None:
    _add_heavy_authority(parser)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")


def _add_predicted_paths(
    parser: argparse.ArgumentParser, *, require_scored: bool = False
) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--d1-run", type=Path, required=True)
    parser.add_argument("--resource-gate", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predicted-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    if require_scored:
        parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    _add_heavy_authority(prepare_parser)
    prepare_parser.add_argument("--predicted-root", type=Path, required=True)
    prepare_parser.add_argument("--bundle-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--expected-count", type=int, default=EXPECTED_SAMPLE_COUNT
    )
    prepare_parser.set_defaults(handler=prepare)

    predicted = commands.add_parser("print-predicted-replay")
    _add_predicted_paths(predicted)
    predicted.set_defaults(handler=print_predicted_replay)

    predicted_candidates = commands.add_parser("execute-predicted-candidates")
    _add_predicted_paths(predicted_candidates)
    predicted_candidates.set_defaults(handler=execute_predicted_candidates)

    predicted_score = commands.add_parser("score-predicted")
    _add_predicted_paths(predicted_score, require_scored=True)
    predicted_score.add_argument("--execute", action="store_true")
    predicted_score.set_defaults(handler=score_predicted)

    predicted_close = commands.add_parser("close-predicted")
    _add_predicted_paths(predicted_close, require_scored=True)
    predicted_close.add_argument(
        "--derived-reconciliation", type=Path, required=True
    )
    predicted_close.set_defaults(handler=close_predicted)

    plan = commands.add_parser("plan")
    _add_execution_paths(plan)
    plan.set_defaults(handler=print_plan)

    candidates = commands.add_parser("execute-candidates")
    _add_execution_paths(candidates)
    candidates.set_defaults(handler=execute_candidates)

    score = commands.add_parser("scorer")
    _add_execution_paths(score)
    score.add_argument("--execute", action="store_true")
    score.set_defaults(handler=scorer)
    close = commands.add_parser("assemble")
    _add_assembly_paths(close)
    close.set_defaults(handler=assemble)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command in {"plan", "scorer"} and args.scored_root is None:
        raise D1AdapterError("--scored-root is required for planning/scoring")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
