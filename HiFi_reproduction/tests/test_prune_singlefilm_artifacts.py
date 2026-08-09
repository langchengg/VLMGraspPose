from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prune_singlefilm_artifacts as prune


def _clear_lsof(command, **kwargs):
    return subprocess.CompletedProcess(command, 1, "", "")


def _busy_lsof(command, **kwargs):
    return subprocess.CompletedProcess(command, 0, "p123\0cpython\0", "")


def _uncertain_lsof(command, **kwargs):
    return subprocess.CompletedProcess(command, 1, "", "permission denied")


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _plan(
    root: Path,
    targets: list[Path],
    *,
    protected_roots: tuple[Path, ...] = (),
    runner=_clear_lsof,
):
    return prune.build_deletion_plan(
        targets,
        protected_roots=protected_roots,
        project_root=root,
        prohibited_parent_targets=(root, root / "runs", root / "outputs"),
        lsof_runner=runner,
    )


def test_default_dry_run_plan_is_complete_and_performs_zero_writes(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "outputs" / "old_run"
    existing.mkdir(parents=True)
    (existing / "payload.bin").write_bytes(b"preserve")
    missing = tmp_path / "outputs" / "missing_run"
    before = _tree_digest(tmp_path)

    plan = _plan(tmp_path, [existing, missing])

    assert [row["decision"] for row in plan] == [
        "WOULD_DELETE",
        "SKIP_MISSING",
    ]
    assert [row["relative_path"] for row in plan] == [
        "outputs/old_run",
        "outputs/missing_run",
    ]
    assert _tree_digest(tmp_path) == before
    assert (existing / "payload.bin").read_bytes() == b"preserve"


@pytest.mark.parametrize("location", ["target", "ancestor", "descendant"])
def test_marker_at_target_ancestor_or_descendant_refuses(
    tmp_path: Path, location: str
) -> None:
    target = tmp_path / "outputs" / "old_run"
    target.mkdir(parents=True)
    if location == "target":
        marker = target / ".RUN_ACTIVE"
    elif location == "ancestor":
        marker = target.parent / ".DO_NOT_PRUNE"
    else:
        marker = target / "nested" / "frozen_experiment_manifest.json"
        marker.parent.mkdir()
    marker.write_text("protected\n")

    row = _plan(tmp_path, [target])[0]

    assert row["decision"] == "REFUSE_PROTECTED"
    assert str(marker) in row["markers_or_locks"]
    assert target.exists()


def test_hard_protected_source_descendant_and_parent_are_refused(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runs" / "repeatedfilm_source"
    child = source / "legacy-looking-child"
    child.mkdir(parents=True)

    child_row = _plan(
        tmp_path, [child], protected_roots=(source,)
    )[0]
    parent_row = _plan(
        tmp_path, [source.parent], protected_roots=(source,)
    )[0]

    assert child_row["decision"] == "REFUSE_PROTECTED"
    assert parent_row["decision"] == "REFUSE_PROTECTED"
    assert child.exists()


def test_broad_runs_and_outputs_parents_are_refused(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    outputs = tmp_path / "outputs"
    runs.mkdir()
    outputs.mkdir()

    decisions = [
        row["decision"] for row in _plan(tmp_path, [runs, outputs])
    ]

    assert decisions == ["REFUSE_PROTECTED", "REFUSE_PROTECTED"]


def test_outside_project_and_symlink_escape_are_refused(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    link = project / "escape"
    link.symlink_to(outside, target_is_directory=True)

    direct = prune.build_deletion_plan(
        [outside],
        protected_roots=(),
        project_root=project,
        prohibited_parent_targets=(project,),
        lsof_runner=_clear_lsof,
    )[0]
    escaped = prune.build_deletion_plan(
        [link],
        protected_roots=(),
        project_root=project,
        prohibited_parent_targets=(project,),
        lsof_runner=_clear_lsof,
    )[0]

    assert direct["decision"] == "REFUSE_OUTSIDE_ROOT"
    assert escaped["decision"] == "REFUSE_OUTSIDE_ROOT"
    assert link.is_symlink()
    assert outside.exists()


def test_lsof_busy_refuses_and_empty_rc1_is_clear(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "old"
    target.mkdir(parents=True)

    busy = _plan(tmp_path, [target], runner=_busy_lsof)[0]
    clear = _plan(tmp_path, [target], runner=_clear_lsof)[0]
    uncertain = _plan(tmp_path, [target], runner=_uncertain_lsof)[0]

    assert busy["decision"] == "REFUSE_IN_USE"
    assert busy["lsof_detail"] == "lsof_busy"
    assert clear["decision"] == "WOULD_DELETE"
    assert uncertain["decision"] == "REFUSE_IN_USE"
    assert "lsof_uncertain" in uncertain["lsof_detail"]


def test_execute_requires_both_keys_and_exact_run_id(tmp_path: Path) -> None:
    missing_record = tmp_path / "not-created.json"
    prune.assert_execution_authorized(
        execute=False,
        confirm_run_id=None,
        cleanup_record=missing_record,
    )
    with pytest.raises(ValueError):
        prune.assert_execution_authorized(
            execute=False,
            confirm_run_id=prune.CONFIRM_RUN_ID,
            cleanup_record=missing_record,
        )
    with pytest.raises(ValueError):
        prune.assert_execution_authorized(
            execute=True,
            confirm_run_id=None,
            cleanup_record=missing_record,
        )
    with pytest.raises(ValueError):
        prune.assert_execution_authorized(
            execute=True,
            confirm_run_id="wrong-run",
            cleanup_record=missing_record,
        )
    prune.assert_execution_authorized(
        execute=True,
        confirm_run_id=prune.CONFIRM_RUN_ID,
        cleanup_record=missing_record,
    )


def test_legacy_apply_flag_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prune_singlefilm_artifacts.py", "--apply"])
    with pytest.raises(SystemExit) as error:
        prune.parse_args()
    assert error.value.code == 2


def test_completed_cleanup_record_is_immutable_and_cannot_replay(
    tmp_path: Path,
) -> None:
    record = tmp_path / "cleanup.json"
    record.write_text(
        json.dumps({"status": "COMPLETED", "removed": 122}) + "\n"
    )
    before = record.read_bytes()

    with pytest.raises(RuntimeError, match="replay is forbidden"):
        prune.assert_execution_authorized(
            execute=True,
            confirm_run_id=prune.CONFIRM_RUN_ID,
            cleanup_record=record,
        )

    assert record.read_bytes() == before


def test_only_explicit_target_is_deleted_and_glob_like_sibling_survives(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outputs" / "old_singlefilm.result.json"
    sibling = tmp_path / "outputs" / "future_old_singlefilm.result.json"
    target.parent.mkdir(parents=True)
    target.write_text("old\n")
    sibling.write_text("future\n")
    plan = _plan(tmp_path, [target])

    removed = prune.execute_deletion_plan(
        plan,
        protected_roots=(),
        project_root=tmp_path,
        prohibited_parent_targets=(tmp_path, tmp_path / "outputs"),
        lsof_runner=_clear_lsof,
    )

    assert removed == [str(target)]
    assert not target.exists()
    assert sibling.read_text() == "future\n"


def test_execute_rechecks_marker_immediately_before_delete(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outputs" / "old_run"
    target.mkdir(parents=True)
    (target / "payload").write_text("keep\n")
    plan = _plan(tmp_path, [target])
    (target / ".RUN_ACTIVE").write_text("became active\n")

    with pytest.raises(RuntimeError, match="changed after planning"):
        prune.execute_deletion_plan(
            plan,
            protected_roots=(),
            project_root=tmp_path,
            prohibited_parent_targets=(tmp_path, tmp_path / "outputs"),
            lsof_runner=_clear_lsof,
        )

    assert (target / "payload").read_text() == "keep\n"


def test_allowlist_is_static_unique_and_contains_historical_122_paths() -> None:
    mixed_source = inspect.getsource(prune.mixed_legacy_targets)
    targets = prune.legacy_targets() + prune.mixed_legacy_targets()

    assert ".glob(" not in mixed_source
    assert len(prune.mixed_legacy_targets()) == 43
    assert len(targets) == 122
    assert len({str(path) for path in targets}) == 122
