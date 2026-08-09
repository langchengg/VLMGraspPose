#!/usr/bin/env python3
"""Conservatively remove audited SAM3 P@90 intermediate payloads.

This post-experiment utility never runs inference or evaluation. It preserves
every formal-lock path, the canonical root, reports, compact audit evidence,
and the per-sample JSON chain under the pre-GT output lock.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / "outputs/sam3_proposal_bank_p90_v1"
REPORT = EXPERIMENT / "report"
PLAN = REPORT / "CLEANUP_PLAN.json"
LOG = REPORT / "cleanup_deleted_paths.jsonl"
EXPECTED_LOCK = "985d6fc2218183fc473dd10692c00d9ab4d1c20a09f1ab384822871814c8aaec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Perform deletion after validation.")
    return parser.parse_args()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _measure(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    size = files = 0
    for item in path.rglob("*"):
        if item.is_file():
            size += item.stat().st_size
            files += 1
    return size, files


def _append_log(path: str, size: int, files: int, reason: str, status: str) -> None:
    record = {
        "bytes": size,
        "deletion_status": status,
        "files": files,
        "path": path,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _validate_plan() -> dict:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    if payload["protected_final_lock"] != EXPECTED_LOCK:
        raise RuntimeError("cleanup plan formal-lock mismatch")
    if payload["canonical_manifest_rows"] != 7675:
        raise RuntimeError("cleanup plan canonical-row mismatch")
    lock = json.loads(
        (PROJECT / "artifacts/sam3_proposal_bank_p90_v1/formal_lock.json").read_text()
    )
    if lock["formal_lock_sha256"] != EXPECTED_LOCK:
        raise RuntimeError("active formal lock changed")
    equivalence = json.loads((REPORT / "LOCKED_MASK_CANONICAL_EQUIVALENCE.json").read_text())
    if equivalence["status"] != "PROVEN_CONTENT_EQUIVALENT":
        raise RuntimeError("locked/canonical equivalence is not proven")
    canonical_rows = sum(
        1
        for _ in (PROJECT / "runs/hifics_sam3_proposal_selector_LOCKED/manifest.jsonl").open()
    )
    if canonical_rows != 7675:
        raise RuntimeError(f"canonical manifest has {canonical_rows} rows")
    return payload


def _prepare_gallery_assets(execute: bool) -> dict[str, int]:
    pattern = re.compile(r"\.\./\.\./(proposals|stage2)/([^/'\"]+)/(proposal_grid|refinement_grid)\.png")
    assets = REPORT / "galleries/assets/raw_grids"
    refs: dict[str, tuple[Path, Path]] = {}
    html_files = sorted((REPORT / "galleries").glob("*.html"))
    for html in html_files:
        text = html.read_text(encoding="utf-8")
        for bank, sample_id, stem in pattern.findall(text):
            old = f"../../{bank}/{sample_id}/{stem}.png"
            source = EXPERIMENT / bank / sample_id / f"{stem}.png"
            destination = assets / f"{bank}__{sample_id}__{stem}.png"
            refs[old] = (source, destination)
    missing = [str(source) for source, _ in refs.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"gallery sources missing before cleanup: {missing[:3]}")
    total_bytes = sum(source.stat().st_size for source, _ in refs.values())
    if execute:
        assets.mkdir(parents=True, exist_ok=True)
        for source, destination in refs.values():
            if not destination.exists():
                shutil.copy2(source, destination)
        for html in html_files:
            text = html.read_text(encoding="utf-8")
            for old, (_, destination) in refs.items():
                text = text.replace(old, f"assets/raw_grids/{destination.name}")
            html.write_text(text, encoding="utf-8")
        for _, destination in refs.values():
            if not destination.is_file():
                raise RuntimeError(f"failed to preserve gallery asset: {destination}")
        remaining = sum(
            len(pattern.findall(html.read_text(encoding="utf-8"))) for html in html_files
        )
        if remaining:
            raise RuntimeError(f"{remaining} raw-bank gallery links remain")
    return {"references": len(refs), "bytes": total_bytes}


def _resolve_entries(plan: dict) -> list[tuple[str, list[Path], str]]:
    groups: list[tuple[str, list[Path], str]] = []
    for group in plan["delete"]:
        paths = [EXPERIMENT / relative for relative in group.get("paths", [])]
        for expression in group.get("globs", []):
            paths.extend(EXPERIMENT.glob(expression))
        project_paths = [
            PROJECT / relative for relative in group.get("project_paths", [])
        ]
        for path in project_paths:
            if not _within(path, PROJECT):
                raise RuntimeError(f"unsafe project cleanup path: {path}")
            if ".venv" in path.parts or path.name not in {
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
            }:
                raise RuntimeError(f"project path is not an allowed cache: {path}")
        paths.extend(project_paths)
        unique = sorted(set(paths))
        for path in unique:
            if path not in project_paths and not _within(path, EXPERIMENT):
                raise RuntimeError(f"unsafe cleanup path: {path}")
            if _within(path, REPORT):
                raise RuntimeError(f"report path cannot be deleted: {path}")
        groups.append((group["id"], unique, group["reason"]))
    return groups


def _validate_no_formal_path_deleted(groups: list[tuple[str, list[Path], str]]) -> None:
    formal = json.loads(
        (PROJECT / "artifacts/sam3_proposal_bank_p90_v1/formal_lock.json").read_text()
    )
    protected: set[Path] = set()
    for key in (
        "code_hashes",
        "config_hashes",
        "selector_artifact_hashes",
        "validation_evidence_hashes",
        "training_group_hashes",
    ):
        protected.update((PROJECT / item).resolve() for item in formal[key])
    for _, paths, _ in groups:
        for delete_path in paths:
            resolved = delete_path.resolve()
            for protected_path in protected:
                if resolved == protected_path or resolved in protected_path.parents:
                    raise RuntimeError(f"delete target contains formal-lock path: {delete_path}")


def main() -> int:
    args = parse_args()
    plan = _validate_plan()
    gallery = _prepare_gallery_assets(args.execute)
    groups = _resolve_entries(plan)
    _validate_no_formal_path_deleted(groups)
    totals = {"bytes": 0, "files": 0, "paths": 0}
    inventory: list[tuple[str, Path, int, int, str]] = []
    for group_id, paths, reason in groups:
        for path in paths:
            size, files = _measure(path)
            if not path.exists():
                continue
            totals["bytes"] += size
            totals["files"] += files
            totals["paths"] += 1
            inventory.append((group_id, path, size, files, reason))
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "delete_bytes": totals["bytes"],
                "delete_files": totals["files"],
                "delete_paths": totals["paths"],
                "gallery_assets": gallery,
            },
            sort_keys=True,
        )
    )
    if not args.execute:
        return 0
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    for group_id, path, size, files, reason in inventory:
        label = str(path.relative_to(PROJECT))
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            _append_log(label, size, files, f"{group_id}: {reason}", "DELETED")
        except Exception as error:
            _append_log(label, size, files, f"{group_id}: {reason}; {error}", "FAILED")
            raise
    print(json.dumps({"status": "COMPLETE", **totals}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
