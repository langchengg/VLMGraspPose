"""Fail-closed terminal lock assembly for the GT-mask counterfactual run."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .acceptance import (
    GALLERY_ACCEPTANCE_RELATIVE_PATH,
    INDEPENDENT_ACCEPTANCE_RELATIVE_PATH,
)
from .gallery_pipeline import verify_complete_gallery
from .audit import export_counterfactual_commands, transition_pipeline_status
from .contracts import RunState
from .io import (
    artifact_record,
    canonical_sha256,
    exclusive_json,
    exclusive_text,
    sha256_file,
)
from .reporting import (
    PUBLICATION_TABLE_NAMES,
    REPORT_NAMES,
    TABLE_CONTRACTS,
    THESIS_INTEGRATION_NAMES,
)
from .protocol import (
    D1_EXECUTION_COMPLETION_RELATIVE_PATH,
    EXECUTION_COMPLETION_RELATIVE_PATH,
    EXECUTION_RELATIVE_PATH,
    LOCK_RELATIVE_PATH,
    verify_protocol_lock,
)


FINAL_LOCK_NAME = "FINAL_COUNTERFACTUAL_RUN_LOCK.json"
FINAL_LOCK_DIGEST_NAME = "FINAL_COUNTERFACTUAL_RUN_LOCK.sha256"

REQUIRED_FIGURE_STEMS = tuple(f"{index:02d}_" for index in range(1, 13))


def assert_counterfactual_run(run_dir: str | Path) -> Path:
    """Reject source formal runs and any ambiguous output directory."""

    root = Path(run_dir).expanduser().resolve()
    if root.parent.name != "runs" or not root.name.startswith(
        "fair_gtmask_counterfactual_g1_c1_d1_"
    ):
        raise PermissionError(
            "finalization is restricted to runs/fair_gtmask_counterfactual_g1_c1_d1_<UTC>"
        )
    return root


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, name: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise ValueError(f"{name} self hash differs")


def _verify_record(record: Mapping[str, Any], *, name: str) -> None:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is not a regular file: {path}")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"{name} hash differs")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise ValueError(f"{name} byte count differs")


def _source_immutability(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _read_json(
        root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json",
        name="source immutability BEFORE",
    )
    after = _read_json(
        root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_AFTER.json",
        name="source immutability AFTER",
    )
    for label, value in (("BEFORE", before), ("AFTER", after)):
        _verify_self_hash(value, name=f"source immutability {label}")
        if value.get("status") != "PASS" or not isinstance(
            value.get("sources"), Mapping
        ) or not isinstance(value.get("inventory_rehashes"), Mapping):
            raise ValueError(f"source immutability {label} did not PASS")
        for name, record in value["sources"].items():
            if not isinstance(record, Mapping):
                raise ValueError(f"source immutability {label}.{name} is invalid")
            _verify_record(record, name=f"source immutability {label}.{name}")
    if before["sources"] != after["sources"]:
        raise ValueError("source before/after exact records differ")
    if before.get("formal_test_counts") != after.get("formal_test_counts"):
        raise ValueError("source formal test count changed")
    if before.get("inventory_rehashes") != after.get("inventory_rehashes"):
        raise ValueError("source full-inventory rehash evidence changed")
    return before, after


def _check_table_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(
        root / "08_metrics" / "TABLE_BUNDLE_MANIFEST.json",
        name="table bundle manifest",
    )
    _verify_self_hash(manifest, name="table bundle manifest")
    if (
        manifest.get("status") != "COMPLETE"
        or set(manifest.get("tables", {})) != set(TABLE_CONTRACTS)
        or int(manifest.get("table_count", -1)) != len(TABLE_CONTRACTS)
    ):
        raise ValueError("table bundle is incomplete")
    for name, record in manifest["tables"].items():
        _verify_record(record, name=f"bound table {name}")
    return manifest


def _check_partial_table_scope(manifest: Mapping[str, Any]) -> None:
    """Ensure a blocked D1 primary was not silently replaced in derived tables."""

    tables = manifest["tables"]
    branch = pd.read_csv(tables["branch_metrics.csv"]["path"])
    paired = pd.read_csv(tables["pred_vs_gt_paired_metrics.csv"]["path"])
    if (
        branch["route"].astype(str).str.upper().eq("D1")
        & branch["branch"].astype(str).str.lower().eq("gt_oracle")
    ).any() or paired["route"].astype(str).str.upper().eq("D1").any():
        raise ValueError(
            "PARTIAL D1 blocker is incompatible with a reported D1 primary counterfactual"
        )


def _check_figures(root: Path, *, partial: bool) -> dict[str, Any]:
    manifest = _read_json(
        root / "13_figures" / "FIGURES_MANIFEST.json", name="figures manifest"
    )
    _verify_self_hash(manifest, name="figures manifest")
    figures = manifest.get("figures")
    d1_secondary = manifest.get("d1_secondary_status")
    expected_names = 12 if d1_secondary == "COMPLETE" else 11
    expected_prefixes = REQUIRED_FIGURE_STEMS[:expected_names]
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("palette") != "Okabe-Ito"
        or not isinstance(figures, Mapping)
        or len(figures) != expected_names
        or not all(
            any(str(name).startswith(prefix) for name in figures)
            for prefix in expected_prefixes
        )
        or (expected_names == 11 and any(str(name).startswith("12_") for name in figures))
    ):
        raise ValueError("figure bundle is incomplete")
    for name, formats in figures.items():
        if not isinstance(formats, Mapping) or set(formats) != {"pdf", "svg", "png"}:
            raise ValueError(f"figure formats differ: {name}")
        for suffix, record in formats.items():
            _verify_record(record, name=f"figure {name}.{suffix}")
    _verify_record(
        manifest["core_summary_figure"], name="core summary figure"
    )
    return manifest


def _check_reports(root: Path, *, partial: bool) -> dict[str, Any]:
    manifest = _read_json(
        root / "15_reports" / "REPORTS_MANIFEST.json", name="reports manifest"
    )
    _verify_self_hash(manifest, name="reports manifest")
    reports = manifest.get("reports")
    if (
        manifest.get("status") != "COMPLETE"
        or not isinstance(reports, Mapping)
        or set(reports) != set(REPORT_NAMES)
    ):
        raise ValueError("report bundle is incomplete")
    for name, record in reports.items():
        _verify_record(record, name=f"report {name}")
    thesis = manifest.get("thesis_integration")
    publication = manifest.get("publication_tables")
    if (
        not isinstance(thesis, Mapping)
        or set(thesis) != set(THESIS_INTEGRATION_NAMES)
        or not isinstance(publication, Mapping)
        or set(publication) != set(PUBLICATION_TABLE_NAMES)
    ):
        raise ValueError("thesis/publication report bundle is incomplete")
    for name, record in thesis.items():
        _verify_record(record, name=f"thesis integration {name}")
    for name, record in publication.items():
        _verify_record(record, name=f"publication table {name}")
    return manifest


def _check_independent(root: Path) -> dict[str, Any]:
    recompute = _read_json(
        root / "16_independent_recompute" / "recomputed_metrics.json",
        name="independent recompute",
    )
    _verify_self_hash(recompute, name="independent recompute")
    if (
        recompute.get("status") != "PASS"
        or recompute.get("per_sample_exact_match") is not True
        or recompute.get("metrics_exact_match") is not True
        or recompute.get("taxonomy_exact_match") is not True
        or recompute.get("paired_inputs_exact_match") is not True
        or not isinstance(recompute.get("source_candidate_geometry"), Mapping)
    ):
        raise ValueError("independent recompute did not exact-match")
    _verify_record(
        recompute["source_candidate_geometry"],
        name="independent candidate geometry",
    )
    mismatches = root / "16_independent_recompute/mismatch_samples.csv"
    if not mismatches.is_file() or not pd.read_csv(mismatches).empty:
        raise ValueError("independent mismatch ledger is absent or non-empty")
    return recompute


def _check_protocol_execution(
    root: Path, *, retrospective_partial: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable pre-execution lock and its unique later claim."""

    lock_path = root / LOCK_RELATIVE_PATH
    protocol = verify_protocol_lock(root)
    if protocol.get("execution_mode") == "retrospective_verified_import":
        pipeline = _read_json(root / "pipeline_status.json", name="pipeline status")
        manifest = _read_json(root / "manifest.json", name="run manifest")
        claim_path = root / EXECUTION_RELATIVE_PATH
        if claim_path.exists():
            claim = _read_json(claim_path, name="D1 secondary execution claim")
            completion = _read_json(
                root / D1_EXECUTION_COMPLETION_RELATIVE_PATH,
                name="D1 secondary execution completion",
            )
            _verify_self_hash(completion, name="D1 secondary execution completion")
            if (
                claim.get("scope") != "d1_secondary"
                or claim.get("status") != "RUNNING"
                or int(claim.get("execution_count", -1)) != 1
                or claim.get("protocol_lock_file_sha256") != sha256_file(lock_path)
                or int(pipeline.get("counterfactual_execution_count", -1)) != 1
                or int(manifest.get("counterfactual_execution_count", -1)) != 1
                or completion.get("status") != "COMPLETE"
                or completion.get("scope") != "d1_secondary"
                or completion.get("execution_claim") != artifact_record(claim_path)
                or completion.get("protocol_lock") != artifact_record(lock_path)
            ):
                raise ValueError("retrospective D1 execution authority differs")
            return protocol, {
                "status": "RETROSPECTIVE_IMPORT_WITH_D1_SECONDARY",
                "execution_count": 1,
                "authority": artifact_record(claim_path),
                "d1_secondary_completion": artifact_record(
                    root / D1_EXECUTION_COMPLETION_RELATIVE_PATH
                ),
            }
        if (
            protocol.get("execution_mode") != "retrospective_verified_import"
            or int(pipeline.get("counterfactual_execution_count", -1)) != 0
            or int(manifest.get("counterfactual_execution_count", -1)) != 0
            or (root / EXECUTION_COMPLETION_RELATIVE_PATH).exists()
        ):
            raise ValueError("retrospective import execution authority differs")
        return protocol, {
            "status": "RETROSPECTIVE_VERIFIED_IMPORT",
            "execution_count": 0,
            "authority": artifact_record(lock_path),
        }
    execution = _read_json(
        root / EXECUTION_RELATIVE_PATH,
        name="counterfactual execution claim",
    )
    if (
        execution.get("status") != "RUNNING"
        or int(execution.get("execution_count", -1)) != 1
        or execution.get("protocol_lock_file_sha256") != sha256_file(lock_path)
    ):
        raise ValueError("unique counterfactual execution claim differs")
    completion = _read_json(
        root / EXECUTION_COMPLETION_RELATIVE_PATH,
        name="counterfactual execution completion",
    )
    _verify_self_hash(completion, name="counterfactual execution completion")
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("execution_count") != 1
        or completion.get("execution_claim")
        != artifact_record(root / EXECUTION_RELATIVE_PATH)
        or completion.get("protocol_lock") != artifact_record(lock_path)
        or completion.get("route_status")
        != artifact_record(root / "08_metrics/ROUTE_STATUS.json")
    ):
        raise ValueError("execution completion authority differs")
    pipeline = _read_json(root / "pipeline_status.json", name="pipeline status")
    manifest = _read_json(root / "manifest.json", name="run manifest")
    if (
        int(pipeline.get("counterfactual_execution_count", -1)) != 1
        or int(manifest.get("counterfactual_execution_count", -1)) != 1
        or int(manifest.get("formal_test_execution_count", -1)) != 0
        or manifest.get("source_formal_test_execution_modified") is not False
    ):
        raise ValueError("counterfactual/source execution counts differ")
    return protocol, completion


def _check_gallery(root: Path) -> dict[str, Any]:
    verified = verify_complete_gallery(root)
    manifest = _read_json(
        root / "14_galleries" / "GALLERY_MANIFEST.json", name="gallery manifest"
    )
    _verify_self_hash(manifest, name="gallery manifest")
    boards = manifest.get("boards")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("manual_qa_status") != "PASS"
        or manifest.get("manual_qa_coverage_pass") is not True
        or not isinstance(boards, list)
        or not boards
        or any(row.get("status") != "AUTO_QA_PASS" for row in boards)
    ):
        raise ValueError("gallery auto/manual QA is incomplete")
    for record_name in (
        "eligible",
        "selected",
        "selected_csv",
        "selection_rules",
        "selection_audit",
        "case_selection_audit",
        "core_cases_figure",
    ):
        _verify_record(manifest[record_name], name=f"gallery {record_name}")
    for index, row in enumerate(boards):
        for suffix in ("png", "svg"):
            _verify_record(row[suffix], name=f"gallery board {index}.{suffix}")
    if manifest != verified:
        raise ValueError("gallery canonical replay differs")
    return manifest


def _check_acceptances(
    root: Path, *, partial: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    gallery = _read_json(
        root / GALLERY_ACCEPTANCE_RELATIVE_PATH, name="P9 gallery acceptance"
    )
    _verify_self_hash(gallery, name="P9 gallery acceptance")
    if (
        gallery.get("status") != "PASS"
        or gallery.get("gallery_manifest")
        != artifact_record(root / "14_galleries/GALLERY_MANIFEST.json")
        or gallery.get("postprocess_manifest")
        != artifact_record(root / "08_metrics/POSTPROCESS_MANIFEST.json")
    ):
        raise ValueError("P9 gallery acceptance differs")
    independent = _read_json(
        root / INDEPENDENT_ACCEPTANCE_RELATIVE_PATH,
        name="P10 independent acceptance",
    )
    _verify_self_hash(independent, name="P10 independent acceptance")
    import_audit = independent.get("forbidden_import_audit")
    if (
        independent.get("schema_version") != 2
        or independent.get("status") != "PASS"
        or independent.get("process_role")
        != "standalone saved-frame independent recompute"
        or independent.get("forbidden_modules_imported") is not False
        or not isinstance(import_audit, Mapping)
        or import_audit.get("status") != "PASS"
        or import_audit.get("observed_forbidden_modules") != []
        or independent.get("postprocess_manifest")
        != artifact_record(root / "08_metrics/POSTPROCESS_MANIFEST.json")
        or independent.get("gallery_acceptance")
        != artifact_record(root / GALLERY_ACCEPTANCE_RELATIVE_PATH)
        or any(
            independent.get(name) is not True
            for name in (
                "per_sample_exact_match",
                "metrics_exact_match",
                "taxonomy_exact_match",
                "paired_inputs_exact_match",
            )
        )
        or not isinstance(independent.get("source_candidate_geometry"), Mapping)
    ):
        raise ValueError("P10 independent acceptance differs")
    _verify_record(
        independent["source_candidate_geometry"],
        name="P10 independent candidate geometry",
    )
    pipeline = _read_json(root / "pipeline_status.json", name="pipeline status")
    expected = (
        RunState.P5B_G1_FULL_COMPLETE.value
        if partial
        else RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value
    )
    terminal = RunState.PARTIAL.value if partial else RunState.COMPLETE.value
    terminal_recovery = (
        pipeline.get("status") == terminal
        and pipeline.get("previous_status") == expected
        and not (root / FINAL_LOCK_NAME).exists()
    )
    if pipeline.get("status") != expected and not terminal_recovery:
        raise ValueError(f"terminal pipeline must be at {expected}")
    return gallery, independent


def _check_routes(
    root: Path, *, d1_blocker: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    route_status = _read_json(
        root / "08_metrics" / "ROUTE_STATUS.json", name="route status"
    )
    _verify_self_hash(route_status, name="route status")
    postprocess_record = route_status.get("artifacts", {}).get(
        "postprocess_manifest"
    )
    if not isinstance(postprocess_record, Mapping):
        raise ValueError("route status lacks postprocess authority")
    _verify_record(postprocess_record, name="route status postprocess manifest")
    canonical_postprocess = artifact_record(
        root / "08_metrics" / "POSTPROCESS_MANIFEST.json"
    )
    if dict(postprocess_record) != canonical_postprocess:
        raise ValueError("route status uses a noncanonical postprocess manifest")
    if route_status.get("protocol_lock") != artifact_record(
        root / LOCK_RELATIVE_PATH
    ):
        raise ValueError("route status protocol authority differs")
    routes = route_status.get("routes")
    if not isinstance(routes, Mapping) or not {"G1", "C1"}.issubset(routes):
        raise ValueError("route status must cover the G1/C1 core")
    if routes.get("G1") != "COMPLETE" or routes.get("C1") != "COMPLETE":
        raise ValueError("G1 and C1 must be COMPLETE before terminal finalization")
    d1_status = route_status.get("d1_secondary_status", "PENDING_AFTER_CORE")
    if d1_blocker is None:
        if d1_status not in {"PENDING_AFTER_CORE", "SKIPPED_AFTER_CORE", "COMPLETE"}:
            raise ValueError("D1 secondary status differs")
        return "COMPLETE", route_status
    bound_blocker = dict(d1_blocker)
    if (
        bound_blocker.get("status")
        not in {"UNRECOVERABLE_BLOCKER", "BLOCKED_WITH_EVIDENCE"}
        or bound_blocker.get("blocker_class")
        != "IRRECOVERABLE_FROZEN_SOURCE_EVIDENCE"
        or bound_blocker.get("raw_candidate_regeneration_required") is not True
    ):
        raise ValueError("D1 blocker is not irrecoverable frozen-source evidence")
    required = {"missing_evidence", "search_paths", "stack_trace", "resume_command"}
    missing = sorted(required.difference(bound_blocker))
    if missing or any(
        not value
        for value in (
            bound_blocker.get("missing_evidence"),
            bound_blocker.get("search_paths"),
            bound_blocker.get("stack_trace"),
            bound_blocker.get("resume_command"),
        )
    ):
        raise ValueError(f"D1 blocker evidence is incomplete: {missing}")
    if bound_blocker.get("filter_only_primary_allowed") is not False:
        raise ValueError("filter-only sensitivity cannot substitute for D1 primary")
    return "COMPLETE", route_status


def load_bound_d1_blocker(root: Path) -> dict[str, Any] | None:
    """Resolve the sole permitted D1 blocker from the canonical input graph."""

    postprocess_path = root / "08_metrics" / "POSTPROCESS_MANIFEST.json"
    postprocess = _read_json(postprocess_path, name="postprocess manifest")
    _verify_self_hash(postprocess, name="postprocess manifest")
    input_path = root / "07_candidate_tables" / "POSTPROCESS_INPUTS.json"
    input_record = postprocess.get("postprocess_inputs")
    if not isinstance(input_record, Mapping):
        if postprocess.get("d1_blocker") is not None:
            raise ValueError("postprocess blocker lacks canonical input authority")
        return None
    if dict(input_record) != artifact_record(input_path):
        raise ValueError("postprocess input authority differs from canonical path")
    _verify_record(input_record, name="postprocess input manifest")
    inputs = _read_json(input_path, name="postprocess input manifest")
    _verify_self_hash(inputs, name="postprocess input manifest")
    artifacts = inputs.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("postprocess input artifacts are absent")
    record = artifacts.get("d1_blocker")
    manifest_record = postprocess.get("d1_blocker")
    if record is None:
        if manifest_record is not None:
            raise ValueError("postprocess blocker is absent from its input authority")
        return None
    if not isinstance(record, Mapping) or dict(record) != manifest_record:
        raise ValueError("postprocess blocker record differs from its input authority")
    blocker_path = Path(str(record.get("path", ""))).expanduser().resolve()
    blocker_root = (root / "00_audit" / "machine_blockers").resolve()
    if blocker_path.parent != blocker_root:
        raise ValueError("D1 blocker is outside the canonical blocker namespace")
    _verify_record(record, name="D1 blocker")
    blocker = _read_json(blocker_path, name="D1 blocker")
    if "content_sha256" in blocker:
        _verify_self_hash(blocker, name="D1 blocker")
    return blocker


def _inventory(root: Path, *, excluded: Iterable[Path]) -> list[dict[str, Any]]:
    excluded_set = {path.resolve() for path in excluded}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() in excluded_set
            or path.name in {".DS_Store", ".commands.log.lock"}
            or path.name.endswith(("-wal", "-shm"))
            or (path.name.startswith(".") and path.name.endswith(".tmp"))
        ):
            continue
        rows.append(
            {"relative_path": str(path.relative_to(root)), **artifact_record(path)}
        )
    return rows


def _repair_existing_terminal(root: Path) -> dict[str, Any] | None:
    """Finish detached sidecars after a crash without rebuilding the lock."""

    lock_path = root / FINAL_LOCK_NAME
    digest_path = root / FINAL_LOCK_DIGEST_NAME
    complete_path, partial_path = root / "COMPLETE", root / "PARTIAL"
    if not lock_path.exists():
        orphaned = [
            path for path in (digest_path, complete_path, partial_path) if path.exists()
        ]
        if orphaned:
            raise PermissionError(f"terminal sidecar exists without final lock: {orphaned}")
        return None
    lock = _read_json(lock_path, name="final counterfactual lock")
    unsigned = dict(lock)
    recorded = unsigned.pop("self_sha256", None)
    inventory = lock.get("inventory")
    if (
        lock.get("status") not in {"COMPLETE", "PARTIAL"}
        or recorded != canonical_sha256(unsigned)
        or not isinstance(inventory, list)
        or lock.get("inventory_count") != len(inventory)
        or lock.get("inventory_sha256") != canonical_sha256(inventory)
    ):
        raise ValueError("existing final lock is not repairable")
    fresh = _inventory(
        root, excluded=[lock_path, digest_path, complete_path, partial_path]
    )
    if fresh != inventory:
        raise ValueError("existing final lock inventory changed before repair")
    lock_sha = sha256_file(lock_path)
    expected_digest = lock_sha + "\n"
    if digest_path.exists():
        if digest_path.read_text(encoding="ascii") != expected_digest:
            raise ValueError("existing terminal digest differs")
    else:
        exclusive_text(digest_path, expected_digest)
    status = str(lock["status"])
    marker = complete_path if status == "COMPLETE" else partial_path
    other = partial_path if status == "COMPLETE" else complete_path
    if other.exists():
        raise ValueError("opposite terminal marker already exists")
    expected_marker = f"{status}\n{FINAL_LOCK_NAME} sha256={lock_sha}\n"
    if marker.exists():
        if marker.read_text(encoding="utf-8") != expected_marker:
            raise ValueError("existing terminal marker differs")
    else:
        exclusive_text(marker, expected_marker)
    verify_final_lock(root)
    return {
        "status": status,
        "final_lock_sha256": lock_sha,
        "inventory_count": len(inventory),
        "repaired": True,
    }


def finalize_run(
    run_dir: str | Path, *, d1_blocker: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Finalize the complete G1/C1 core; D1 remains separate secondary evidence."""

    root = assert_counterfactual_run(run_dir)
    repaired = _repair_existing_terminal(root)
    if repaired is not None:
        return repaired
    before, after = _source_immutability(root)
    baseline = _read_json(
        root / "04_predicted_replay" / "BASELINE_REPLAY_MANIFEST.json",
        name="baseline replay manifest",
    )
    _verify_self_hash(baseline, name="baseline replay manifest")
    if baseline.get("status") != "PASS" or baseline.get("sample_count") != 7675:
        raise ValueError("predicted replay baseline did not PASS at N=7,675")
    mapping = _read_json(
        root / "03_gt_mask_registry" / "GT_MASK_MAPPING_AUDIT.json",
        name="GT mask mapping audit",
    )
    _verify_self_hash(mapping, name="GT mask mapping audit")
    if mapping.get("status") != "PASS" or mapping.get("sample_count") != 7675:
        raise ValueError("GT mask mapping audit did not PASS at N=7,675")
    status, route_status = _check_routes(root, d1_blocker=d1_blocker)
    bound_d1_blocker = None if d1_blocker is None else dict(d1_blocker)
    partial = status == "PARTIAL"
    _protocol, execution = _check_protocol_execution(
        root, retrospective_partial=partial
    )
    table_manifest = _check_table_manifest(root)
    if partial:
        _check_partial_table_scope(table_manifest)
    figure_manifest = _check_figures(root, partial=partial)
    report_manifest = _check_reports(root, partial=partial)
    gallery_manifest = _check_gallery(root)
    recompute = _check_independent(root)
    gallery_acceptance, independent_acceptance = _check_acceptances(
        root, partial=partial
    )
    commands_path = export_counterfactual_commands(root)
    pipeline = _read_json(root / "pipeline_status.json", name="pipeline status")
    terminal_state = RunState.PARTIAL if partial else RunState.COMPLETE
    if pipeline.get("status") != terminal_state.value:
        transition_pipeline_status(
            root,
            terminal_state,
            first_incomplete_stage=None,
        )
    lock_path = root / FINAL_LOCK_NAME
    digest_path = root / FINAL_LOCK_DIGEST_NAME
    complete_path, partial_path = root / "COMPLETE", root / "PARTIAL"
    excluded = [lock_path, digest_path, complete_path, partial_path]
    inventory = _inventory(root, excluded=excluded)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "experiment_kind": "post-formal GT-mask oracle diagnostic",
        "source_before": artifact_record(
            root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json"
        ),
        "source_after": artifact_record(
            root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_AFTER.json"
        ),
        "source_records_exact_match": before["sources"] == after["sources"],
        "source_formal_test_counts_unchanged": before.get("formal_test_counts")
        == after.get("formal_test_counts"),
        "commands": artifact_record(commands_path),
        "baseline_replay": artifact_record(
            root / "04_predicted_replay" / "BASELINE_REPLAY_MANIFEST.json"
        ),
        "gt_mapping": artifact_record(
            root / "03_gt_mask_registry" / "GT_MASK_MAPPING_AUDIT.json"
        ),
        "protocol_lock": artifact_record(
            root / "01_protocol_lock" / "COUNTERFACTUAL_PROTOCOL_LOCK.json"
        ),
        "counterfactual_execution": (
            artifact_record(root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json")
            if execution["execution_count"] == 1
            else dict(execution["authority"])
        ),
        "counterfactual_execution_count": execution["execution_count"],
        "d1_secondary_execution_completion": (
            execution.get("d1_secondary_completion")
        ),
        "route_status": route_status,
        "table_manifest": artifact_record(
            root / "08_metrics" / "TABLE_BUNDLE_MANIFEST.json"
        ),
        "table_manifest_content_sha256": table_manifest["content_sha256"],
        "figure_manifest": artifact_record(
            root / "13_figures" / "FIGURES_MANIFEST.json"
        ),
        "figure_manifest_content_sha256": figure_manifest["content_sha256"],
        "gallery_manifest": artifact_record(
            root / "14_galleries" / "GALLERY_MANIFEST.json"
        ),
        "gallery_manifest_content_sha256": gallery_manifest["content_sha256"],
        "gallery_acceptance": artifact_record(root / GALLERY_ACCEPTANCE_RELATIVE_PATH),
        "gallery_acceptance_content_sha256": gallery_acceptance["content_sha256"],
        "report_manifest": artifact_record(
            root / "15_reports" / "REPORTS_MANIFEST.json"
        ),
        "report_manifest_content_sha256": report_manifest["content_sha256"],
        "independent_recompute": artifact_record(
            root / "16_independent_recompute" / "recomputed_metrics.json"
        ),
        "independent_recompute_status": recompute["status"],
        "independent_acceptance": artifact_record(
            root / INDEPENDENT_ACCEPTANCE_RELATIVE_PATH
        ),
        "independent_acceptance_content_sha256": independent_acceptance[
            "content_sha256"
        ],
        "d1_unrecoverable_blocker": bound_d1_blocker,
        "d1_secondary_status": (
            "BLOCKED_WITH_EVIDENCE"
            if bound_d1_blocker is not None
            else (
                "COMPLETE"
                if execution.get("d1_secondary_completion") is not None
                else route_status.get(
                    "d1_secondary_status", "SKIPPED_AFTER_CORE"
                )
            )
        ),
        "inventory": inventory,
        "inventory_count": len(inventory),
        "inventory_sha256": canonical_sha256(inventory),
        "excluded_self_referential_files": [
            FINAL_LOCK_NAME,
            FINAL_LOCK_DIGEST_NAME,
            "COMPLETE",
            "PARTIAL",
        ],
    }
    payload["self_sha256"] = canonical_sha256(payload)
    exclusive_json(lock_path, payload)
    lock_sha = sha256_file(lock_path)
    exclusive_text(digest_path, lock_sha + "\n")
    marker_path = complete_path if status == "COMPLETE" else partial_path
    exclusive_text(marker_path, f"{status}\n{FINAL_LOCK_NAME} sha256={lock_sha}\n")
    verify_final_lock(root)
    return {
        "status": status,
        "final_lock_sha256": lock_sha,
        "inventory_count": len(inventory),
    }


def verify_final_lock(run_dir: str | Path) -> dict[str, Any]:
    root = assert_counterfactual_run(run_dir)
    lock_path = root / FINAL_LOCK_NAME
    lock = _read_json(lock_path, name="final counterfactual lock")
    unsigned = dict(lock)
    recorded = unsigned.pop("self_sha256", None)
    inventory = lock.get("inventory")
    if (
        lock.get("status") not in {"COMPLETE", "PARTIAL"}
        or recorded != canonical_sha256(unsigned)
        or not isinstance(inventory, list)
        or lock.get("inventory_count") != len(inventory)
        or lock.get("inventory_sha256") != canonical_sha256(inventory)
    ):
        raise ValueError("final counterfactual lock self/inventory contract differs")
    lock_sha = sha256_file(lock_path)
    digest_path = root / FINAL_LOCK_DIGEST_NAME
    if digest_path.read_text(encoding="ascii").strip() != lock_sha:
        raise ValueError("final counterfactual lock detached digest differs")
    status = str(lock["status"])
    marker = root / status
    other = root / ("PARTIAL" if status == "COMPLETE" else "COMPLETE")
    expected_marker = f"{status}\n{FINAL_LOCK_NAME} sha256={lock_sha}\n"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != expected_marker:
        raise ValueError("terminal marker binding differs")
    if other.exists():
        raise ValueError("both COMPLETE and PARTIAL markers exist")
    if status == "PARTIAL" and lock.get("d1_unrecoverable_blocker") is None:
        raise ValueError("PARTIAL lock lacks D1 blocker evidence")
    if status == "COMPLETE" and lock.get("d1_unrecoverable_blocker") is not None:
        if lock.get("d1_secondary_status") != "BLOCKED_WITH_EVIDENCE":
            raise ValueError("COMPLETE lock D1 evidence lacks secondary status")
    excluded = [
        lock_path,
        digest_path,
        root / "COMPLETE",
        root / "PARTIAL",
    ]
    fresh = _inventory(root, excluded=excluded)
    if fresh != inventory:
        raise ValueError("fresh exact final inventory differs")
    return lock


__all__ = [
    "FINAL_LOCK_DIGEST_NAME",
    "FINAL_LOCK_NAME",
    "assert_counterfactual_run",
    "finalize_run",
    "verify_final_lock",
]
