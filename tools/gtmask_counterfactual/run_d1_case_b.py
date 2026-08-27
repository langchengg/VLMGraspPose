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
    verify_d1_execution_binding,
    verify_d1_source_view,
    write_d1_execution_binding,
)
from gtmask_counterfactual.execution import (  # noqa: E402
    FROZEN_D1_SOURCE,
    append_gt_access_log,
    artifact_record,
    authorize_gt_candidate_generation,
    counterfactual_job,
    probe_frozen_docker_image,
)
from gtmask_counterfactual.io import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from gtmask_counterfactual.protocol import (  # noqa: E402
    D1_EXECUTION_COMPLETION_RELATIVE_PATH,
)
from gtmask_counterfactual.resource import (  # noqa: E402
    DOCKER_SCORING_RESOURCE_SCOPE,
    STANDARD_RESOURCE_SCOPE,
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)
from gtmask_counterfactual.audit import (  # noqa: E402
    RunState,
    require_verified_source,
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
SOURCE_VIEW_EXECUTION_MANIFEST = "SOURCE_VIEW_EXECUTION_MANIFEST.json"


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


def _assert_command_stage(args: argparse.Namespace) -> None:
    """Reject out-of-order D1 work before any gate, GT read, or subprocess."""

    run = args.run_dir.expanduser().resolve()
    pipeline = _json_object(run / "pipeline_status.json", label="pipeline status")
    observed = str(pipeline.get("status", ""))
    execution_count = int(pipeline.get("counterfactual_execution_count", -1))
    predicted_commands = {
        "print-predicted-replay",
        "execute-predicted-candidates",
        "score-predicted",
        "close-predicted",
    }
    if args.command in predicted_commands:
        if observed != RunState.P0_AUDIT.value or execution_count != 0:
            raise PermissionError(
                "D1 predicted replay is pre-protocol P0 work and requires count zero"
            )
        return
    completion = run / D1_EXECUTION_COMPLETION_RELATIVE_PATH
    if completion.is_file():
        if args.command != "assemble" or not getattr(args, "resume", False):
            raise PermissionError(
                "completed D1 output may only be revalidated by assemble --resume"
            )
        if execution_count != 1:
            raise PermissionError("completed D1 output requires one execution claim")
        return
    if observed != RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value:
        raise PermissionError(
            "D1 secondary work requires P10_INDEPENDENT_RECOMPUTE_PASS"
        )
    required_core = (
        run / "08_metrics/POSTPROCESS_MANIFEST.json",
        run / "13_figures/FIGURES_MANIFEST.json",
        run / "14_galleries/GALLERY_MANIFEST.json",
        run / "15_reports/REPORTS_MANIFEST.json",
        run / "16_independent_recompute/INDEPENDENT_VALIDATION.json",
    )
    missing = [str(path) for path in required_core if not path.is_file()]
    if missing:
        raise PermissionError(
            f"D1 secondary requires completed core artifacts: {missing}"
        )
    if execution_count not in {0, 1}:
        raise PermissionError("D1 GT-oracle work has an invalid execution count")


def _fresh_gate(
    path: Path, *, resource_scope: str = STANDARD_RESOURCE_SCOPE
) -> dict[str, Any]:
    value = _json_object(path, label="D1 resource gate")
    validate_fresh_gate(value, resource_scope=resource_scope)
    return value


def _prepare_gate(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    """Collect or load a fresh gate while the global heavy lease is held."""

    resource_scope = (
        DOCKER_SCORING_RESOURCE_SCOPE
        if stage in {"predicted_scoring", "oracle_scoring"}
        else STANDARD_RESOURCE_SCOPE
    )
    if getattr(args, "collect_resource_gate", False):
        value = collect_fresh_three_by_five_gate(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir.expanduser().resolve(),
            resource_scope=resource_scope,
        )
        path = (
            args.run_dir.expanduser().resolve()
            / "00_audit/resource_gates"
            / f"d1_{stage}_{value['content_sha256'][:20]}.json"
        )
        if path.exists():
            raise FileExistsError(f"resource gate already exists: {path}")
        atomic_json(path, value)
        args.resource_gate = path
    if args.resource_gate is None:
        raise D1AdapterError("D1 heavy execution requires a resource gate")
    return _fresh_gate(args.resource_gate, resource_scope=resource_scope)


def _resume_command() -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]


def _ledger_identity(command: list[str] | None = None) -> str:
    """Canonicalize a D1 scientific command without recovery-only flags."""

    values = list(_resume_command() if command is None else command)
    ignored_flags = {
        "--resume",
        "--verify-existing",
        "--retry-failed",
        "--retry-failures",
    }
    result: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in ignored_flags:
            index += 1
            continue
        if value == "--resource-gate":
            index += 2
            continue
        result.append(value)
        index += 1
    return shlex.join(result)


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


def _local_source_verified_command(
    command: list[str], *, source_view: Path, source_manifest: Path
) -> list[str]:
    """Route a frozen local script through the hash-verifying bootstrap."""

    if len(command) < 2:
        raise D1AdapterError("D1 source command is incomplete")
    script = Path(command[1]).expanduser().resolve(strict=False)
    try:
        relative = script.relative_to(source_view.expanduser().resolve())
    except ValueError as error:
        raise D1AdapterError("D1 executable is outside its source view") from error
    bootstrap = source_view / "scripts/gtmask_oracle_candidate_bootstrap.py"
    return [
        command[0],
        str(bootstrap),
        "--source-view",
        str(source_view),
        "--source-view-manifest-sha256",
        sha256_file(source_manifest),
        "--exec-source-member",
        relative.as_posix(),
        "--",
        *command[2:],
    ]


def _container_source_verified_command(
    command: list[str], *, source_manifest: Path
) -> list[str]:
    """Make the container rehash its read-only source mount before scoring."""

    scorer = "scripts/run_full_gqcnn_scoring.py"
    try:
        script_index = command.index(scorer)
    except ValueError as error:
        raise D1AdapterError("D1 container command lacks the frozen scorer") from error
    if script_index == 0 or command[script_index - 1] != "python":
        raise D1AdapterError("D1 container scorer launch shape differs")
    return [
        *command[: script_index - 1],
        "python",
        "scripts/gtmask_oracle_candidate_bootstrap.py",
        "--source-view",
        "/workspace",
        "--source-view-manifest-sha256",
        sha256_file(source_manifest),
        "--allow-relocated-source-view",
        "--exec-source-member",
        scorer,
        "--",
        *command[script_index + 1 :],
    ]


def _run_source_bound(
    command: list[str], *, source_manifest: Path, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Leave no unchecked host work between a full source rehash and launch."""

    verify_d1_source_view(source_manifest)
    return subprocess.run(command, check=False, **kwargs)


def _execution_binding(
    output_root: Path,
    *,
    stage: str,
    source_manifest: Path,
    artifacts: Mapping[str, Path],
) -> Path:
    destination = (
        output_root.expanduser().resolve(strict=False)
        / SOURCE_VIEW_EXECUTION_MANIFEST
    )
    return write_d1_execution_binding(
        destination,
        stage=stage,
        source_view_manifest=source_manifest,
        artifacts=artifacts,
    )


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
    authority, _ = _authorise(args)
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B bundle materialisation"):
        _prepare_gate(args, stage="oracle_bundle")
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_bundle_materialisation_launch",
        )
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
            resume=args.resume,
        )
    print(json.dumps({"status": "COMPLETE", "bundle_root": str(root)}, sort_keys=True))
    return 0


def print_predicted_replay(args: argparse.Namespace) -> int:
    sample_ids = _predicted_authority(args)
    if args.collect_resource_gate:
        raise D1AdapterError("planning cannot consume a 15-minute resource gate")
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
    command = _local_source_verified_command(
        command, source_view=source_view, source_manifest=source_manifest
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
        _prepare_gate(args, stage="predicted_candidates")
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
        command_values = _local_source_verified_command(
            [str(CANDIDATE_PYTHON), str(candidate_script), *arguments],
            source_view=source_view,
            source_manifest=source_manifest,
        )
        with ledger_stage(
            run / "run_ledger.sqlite",
            stage="P3_PREDICTED_REPLAY",
            substage="d1_candidates",
            route="d1",
            pool="allnms",
            command=_ledger_identity(command_values),
        ) as ledger:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(source_view)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = _run_source_bound(
                command_values,
                source_manifest=source_manifest,
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
            binding = _execution_binding(
                candidate_root,
                stage="P3_PREDICTED_REPLAY/d1_candidates",
                source_manifest=source_manifest,
                artifacts={"candidate_run_config": run_config},
            )
            ledger["artifact_path"] = str(binding.resolve())
            ledger["artifact_sha256"] = sha256_file(binding)
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
        _prepare_gate(args, stage="predicted_scoring")
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_predicted_scoring_launch",
            resource_scope=DOCKER_SCORING_RESOURCE_SCOPE,
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
        command = _container_source_verified_command(
            command, source_manifest=source_manifest
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
            command=_ledger_identity(command),
        ) as ledger:
            completed = _run_source_bound(
                command, source_manifest=source_manifest
            )
            if completed.returncode != 0:
                raise D1AdapterError(
                    f"frozen predicted scorer returned {completed.returncode}"
                )
            binding = _execution_binding(
                scored_root,
                stage="P3_PREDICTED_REPLAY/d1_scoring",
                source_manifest=source_manifest,
                artifacts={
                    "scorer_manifest": scored_root / "scoring_manifest.jsonl",
                    "scorer_summary": scored_root / "summary.csv",
                },
            )
            ledger["artifact_path"] = str(binding.resolve())
            ledger["artifact_sha256"] = sha256_file(binding)
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
        _prepare_gate(args, stage="predicted_closure")
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
            command=_ledger_identity(),
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
            candidate_binding = verify_d1_execution_binding(
                args.candidate_root / SOURCE_VIEW_EXECUTION_MANIFEST
            )
            scorer_binding = verify_d1_execution_binding(
                args.scored_root / SOURCE_VIEW_EXECUTION_MANIFEST
            )
            del candidate_binding, scorer_binding
            binding = _execution_binding(
                manifest.parent,
                stage="P3_PREDICTED_REPLAY/d1_semantic_closure",
                source_manifest=source_manifest,
                artifacts={
                    "candidate_execution": (
                        args.candidate_root / SOURCE_VIEW_EXECUTION_MANIFEST
                    ),
                    "predicted_replay_manifest": manifest,
                    "scorer_execution": (
                        args.scored_root / SOURCE_VIEW_EXECUTION_MANIFEST
                    ),
                },
            )
            ledger["artifact_path"] = str(binding.resolve())
            ledger["artifact_sha256"] = sha256_file(binding)
    print(json.dumps({"status": "PASS", "manifest": str(manifest)}, sort_keys=True))
    return 0


def print_plan(args: argparse.Namespace) -> int:
    _authorise(args)
    if args.collect_resource_gate:
        raise D1AdapterError("planning cannot consume a 15-minute resource gate")
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
        scorer_command = build_frozen_scorer_command(
            docker=Path(docker),
            candidate_root=args.candidate_root,
            output_root=args.scored_root,
            inventory=inventory,
            source_view_root=source_view,
            resume=args.resume,
        )
        result["scorer_command"] = _container_source_verified_command(
            scorer_command, source_manifest=source_manifest
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if blocker is None else 75


def execute_candidates(args: argparse.Namespace) -> int:
    authority, _ = _authorise(args)
    del authority
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B GT candidate generation"):
        _prepare_gate(args, stage="oracle_candidates")
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
        with counterfactual_job(
            args.run_dir,
            stage="P6_D1_RAW_CANDIDATES",
            route="d1",
            branch="gt_oracle",
            command=_ledger_identity(command),
        ) as job:
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONPATH": str(source_view),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = _run_source_bound(
                command,
                source_manifest=source_manifest,
                cwd=source_view,
                env=environment,
            )
            if completed.returncode != 0:
                raise D1AdapterError(
                    f"frozen candidate runner returned {completed.returncode}"
                )
            candidate_inventory_from_summary(
                args.candidate_root, expected_sample_ids=_bundle_sample_ids(bundle)
            )
            run_config = args.candidate_root / "run_config.json"
            binding = _execution_binding(
                args.candidate_root,
                stage="P6_D1_RAW_CANDIDATES",
                source_manifest=source_manifest,
                artifacts={"candidate_run_config": run_config},
            )
            job["artifact_path"] = str(binding.resolve())
            job["artifact_sha256"] = sha256_file(binding)
    return 0


def scorer(args: argparse.Namespace) -> int:
    _authorise(args)
    if not args.execute:
        if args.collect_resource_gate:
            raise D1AdapterError("planning cannot consume a 15-minute resource gate")
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
        command = _container_source_verified_command(
            command, source_manifest=source_manifest
        )
        print(json.dumps({"status": "READY", "command": command}, sort_keys=True))
        return 0
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B frozen GQ-CNN scoring"):
        _prepare_gate(args, stage="oracle_scoring")
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_gqcnn_scoring_launch",
            resource_scope=DOCKER_SCORING_RESOURCE_SCOPE,
        )
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
        command = _container_source_verified_command(
            command, source_manifest=source_manifest
        )
        with counterfactual_job(
            args.run_dir,
            stage="P6_D1_GQCNN_SCORING",
            route="d1",
            branch="gt_oracle",
            command=_ledger_identity(command),
        ) as job:
            completed = _run_source_bound(
                command, source_manifest=source_manifest
            )
            if completed.returncode != 0:
                raise D1AdapterError(f"frozen scorer returned {completed.returncode}")
            binding = _execution_binding(
                args.scored_root,
                stage="P6_D1_GQCNN_SCORING",
                source_manifest=source_manifest,
                artifacts={
                    "scorer_manifest": (
                        args.scored_root / "scoring_manifest.jsonl"
                    ),
                    "scorer_summary": args.scored_root / "summary.csv",
                },
            )
            job["artifact_path"] = str(binding.resolve())
            job["artifact_sha256"] = sha256_file(binding)
    return 0


def assemble(args: argparse.Namespace) -> int:
    """Publish the only canonical D1 route output after frozen scoring."""

    _authorise(args)
    with exclusive_d1_flock(args.run_dir, purpose="D1 Case-B canonical assembly"):
        _prepare_gate(args, stage="oracle_assembly")
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_d1_canonical_assembly_launch",
        )
        bundle = verify_isolated_bundle_root(args.bundle_root)
        if int(bundle.get("denominator_sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
            raise D1AdapterError("D1 bundle denominator differs before assembly")
        _, source_manifest = _source_view(args.run_dir)
        verify_d1_execution_binding(
            args.candidate_root / SOURCE_VIEW_EXECUTION_MANIFEST
        )
        verify_d1_execution_binding(
            args.scored_root / SOURCE_VIEW_EXECUTION_MANIFEST
        )
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
            command=_ledger_identity(),
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
            binding = _execution_binding(
                manifest.parent,
                stage="P6_D1_CANONICAL_OUTPUT",
                source_manifest=source_manifest,
                artifacts={
                    "candidate_execution": (
                        args.candidate_root / SOURCE_VIEW_EXECUTION_MANIFEST
                    ),
                    "canonical_manifest": manifest,
                    "scorer_execution": (
                        args.scored_root / SOURCE_VIEW_EXECUTION_MANIFEST
                    ),
                },
            )
            job["artifact_path"] = str(binding.resolve())
            job["artifact_sha256"] = sha256_file(binding)
        claim = args.run_dir / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
        completion: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "scope": "d1_secondary",
            "execution_count": 1,
            "protocol_lock": artifact_record(args.protocol_lock),
            "execution_claim": artifact_record(claim),
            "canonical_manifest": artifact_record(manifest),
            "source_view_execution_manifest": artifact_record(binding),
            "core_pipeline_status_preserved": RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value,
        }
        completion["content_sha256"] = canonical_sha256(completion)
        completion_path = args.run_dir / D1_EXECUTION_COMPLETION_RELATIVE_PATH
        if completion_path.exists():
            if _json_object(completion_path, label="D1 completion") != completion:
                raise D1AdapterError("existing D1 secondary completion differs")
        else:
            atomic_json(completion_path, completion)
        for status_path in (
            args.run_dir / "pipeline_status.json",
            args.run_dir / "manifest.json",
        ):
            status_value = _json_object(status_path, label=status_path.name)
            status_value["d1_secondary_status"] = "COMPLETE"
            status_value["d1_secondary_execution_count"] = 1
            atomic_json(status_path, status_value)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "manifest": str(manifest),
                "source_view_execution_manifest": str(binding),
                "d1_secondary_completion": str(completion_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _add_authority(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)


def _add_heavy_authority(parser: argparse.ArgumentParser) -> None:
    _add_authority(parser)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
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
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
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
    prepare_parser.add_argument("--resume", action="store_true")
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
    _assert_command_stage(args)
    if args.command in {"plan", "scorer"} and args.scored_root is None:
        raise D1AdapterError("--scored-root is required for planning/scoring")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
