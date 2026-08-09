"""Read-only provenance audits for canonical re-ranking artifacts.

Audit functions only read their source paths.  The sole writer,
``write_audit_bundle``, requires a new caller-selected directory, rejects any
overlap with canonical/source roots, and creates only JSON, Markdown and TSV
files inside that directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from reranking.data_contracts import (
    CANONICAL_ARTIFACTS,
    CANONICAL_PATHS,
    CROG_CANONICAL_RUN,
    MODULAR_CANONICAL_RUN,
    CanonicalArtifact,
    file_inventory,
    streaming_sha256,
)


class ProvenanceAuditError(ValueError):
    """Raised when provenance or output-isolation requirements are violated."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceAuditError(f"cannot read JSON manifest {path}: {error}") from error


def _manifest_descriptors(value: Any, trail: str = "$") -> Iterable[dict[str, Any]]:
    """Yield nested manifest objects that carry a path plus SHA-256."""

    if isinstance(value, Mapping):
        path = value.get("path")
        sha256 = value.get("sha256")
        if isinstance(path, str) and path and isinstance(sha256, str) and sha256:
            yield {
                "manifest_location": trail,
                "declared_path": path,
                "expected_sha256": sha256.lower(),
                "expected_size_bytes": value.get(
                    "size_bytes", value.get("bytes")
                ),
            }
        for key, child in value.items():
            yield from _manifest_descriptors(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _manifest_descriptors(child, f"{trail}[{index}]")


def manifest_file_inventory(
    manifest_path: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Audit every nested ``{path, sha256}`` descriptor in a JSON manifest."""

    manifest_input = Path(os.path.abspath(manifest_path))
    if manifest_input.is_symlink() or not manifest_input.is_file():
        raise ProvenanceAuditError(
            f"manifest is not a regular file: {manifest_input}"
        )
    manifest = manifest_input.resolve()
    payload = _load_json(manifest)
    descriptors = list(_manifest_descriptors(payload))
    by_path: dict[str, dict[str, Any]] = {}
    declaration_conflicts: list[dict[str, Any]] = []
    for descriptor in descriptors:
        declared = descriptor["declared_path"]
        absolute = Path(declared)
        if not absolute.is_absolute():
            absolute = manifest.parent / absolute
        # Normalize dot segments without resolving the final symlink: symlink
        # provenance must remain observable and is rejected below.
        absolute = Path(os.path.abspath(absolute))
        key = str(absolute)
        previous = by_path.get(key)
        if previous is not None:
            if (
                previous["expected_sha256"] != descriptor["expected_sha256"]
                or previous["expected_size_bytes"]
                not in (None, descriptor["expected_size_bytes"])
                and descriptor["expected_size_bytes"] is not None
            ):
                declaration_conflicts.append(
                    {"path": key, "first": previous, "second": descriptor}
                )
            continue
        expected_size = descriptor["expected_size_bytes"]
        if expected_size is not None:
            try:
                expected_size = int(expected_size)
            except (TypeError, ValueError):
                declaration_conflicts.append(
                    {"path": key, "error": "invalid expected size"}
                )
                expected_size = None
        exists = absolute.exists()
        symlink = absolute.is_symlink()
        regular_file = absolute.is_file() and not symlink
        actual_size = int(absolute.stat().st_size) if regular_file else None
        actual_sha = (
            streaming_sha256(absolute)
            if verify_hashes and regular_file
            else None
        )
        by_path[key] = {
            **descriptor,
            "path": key,
            "exists": exists,
            "regular_file": regular_file,
            "symlink": symlink,
            "actual_size_bytes": actual_size,
            "size_matches": (
                None if expected_size is None or actual_size is None else actual_size == expected_size
            ),
            "actual_sha256": actual_sha,
            "sha256_matches": (
                None
                if actual_sha is None
                else actual_sha == descriptor["expected_sha256"]
            ),
        }
    inventory = sorted(by_path.values(), key=lambda item: item["path"])
    all_present = all(item["regular_file"] for item in inventory)
    sizes_match = all(item["size_matches"] is not False for item in inventory)
    hashes_match = all(item["sha256_matches"] is not False for item in inventory)
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": streaming_sha256(manifest) if verify_hashes else None,
        "descriptor_count": len(descriptors),
        "unique_file_count": len(inventory),
        "declaration_conflicts": declaration_conflicts,
        "inventory": inventory,
        "all_present": all_present,
        "sizes_match": sizes_match,
        "hashes_match": hashes_match,
        "passed": (
            not declaration_conflicts
            and all_present
            and sizes_match
            and hashes_match
        ),
    }


def audit_known_artifacts(
    artifacts: Sequence[CanonicalArtifact] = CANONICAL_ARTIFACTS,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Check existence, regular-file status, size and known digest contracts."""

    inventory: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = artifact.path
        symlink = path.is_symlink()
        regular_file = path.is_file() and not symlink
        actual_sha = (
            streaming_sha256(path) if verify_hashes and regular_file else None
        )
        inventory.append(
            {
                **artifact.to_dict(),
                "exists": path.exists(),
                "regular_file": regular_file,
                "symlink": symlink,
                "size_bytes": int(path.stat().st_size) if regular_file else None,
                "actual_sha256": actual_sha,
                "sha256_matches": (
                    None if actual_sha is None else actual_sha == artifact.sha256
                ),
            }
        )
    passed = all(
        item["regular_file"] and item["sha256_matches"] is not False
        for item in inventory
    )
    return {
        "artifact_count": len(inventory),
        "verify_hashes": bool(verify_hashes),
        "inventory": inventory,
        "passed": passed,
    }


def _top_level_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        records.append(
            {
                "relative_path": path.name,
                "type": (
                    "symlink"
                    if path.is_symlink()
                    else "file"
                    if path.is_file()
                    else "directory"
                    if path.is_dir()
                    else "other"
                ),
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
                "sha256": None,
            }
        )
    return records


def audit_run(
    run_root: str | os.PathLike[str],
    *,
    manifest_names: Sequence[str] = (
        "final_output_manifest.json",
        "frozen_experiment_manifest.json",
    ),
    completion_markers: Sequence[str] = (
        "COMPLETED",
        "formal/test/TEST_RUN_COMPLETE.json",
    ),
    verify_hashes: bool = True,
    recursive_file_inventory: bool = False,
) -> dict[str, Any]:
    """Read-only audit of one run root and its declared artifact manifests."""

    root_input = Path(os.path.abspath(run_root))
    if root_input.is_symlink() or not root_input.is_dir():
        raise ProvenanceAuditError(
            f"run root is not a regular directory: {root_input}"
        )
    root = root_input.resolve()
    manifest_results = []
    for name in manifest_names:
        candidate = root / name
        if candidate.is_file() and not candidate.is_symlink():
            manifest_results.append(
                manifest_file_inventory(candidate, verify_hashes=verify_hashes)
            )
    markers = [
        {
            "relative_path": name,
            "exists": (root / name).is_file() and not (root / name).is_symlink(),
        }
        for name in completion_markers
    ]
    completion_present = any(item["exists"] for item in markers)
    inventory = (
        file_inventory(root, hash_files=False)
        if recursive_file_inventory
        else _top_level_inventory(root)
    )
    symlink_count = sum(item["type"] == "symlink" for item in inventory)
    return {
        "run_root": str(root),
        "read_only_source": True,
        "manifest_count": len(manifest_results),
        "manifests": manifest_results,
        "completion_markers": markers,
        "completion_marker_present": completion_present,
        "file_inventory_scope": (
            "recursive" if recursive_file_inventory else "top_level"
        ),
        "file_inventory": inventory,
        "symlink_count": symlink_count,
        "passed": (
            completion_present
            and bool(manifest_results)
            and all(item["passed"] for item in manifest_results)
            and symlink_count == 0
        ),
    }


def audit_canonical_runs(
    *,
    verify_hashes: bool = True,
    verify_all_known_artifacts: bool = False,
) -> dict[str, Any]:
    """Audit the retained Modular and CROG canonical run roots."""

    modular = audit_run(
        MODULAR_CANONICAL_RUN,
        manifest_names=("final_output_manifest.json",),
        completion_markers=("COMPLETED",),
        verify_hashes=verify_hashes,
    )
    crog = audit_run(
        CROG_CANONICAL_RUN,
        manifest_names=("frozen_experiment_manifest.json",),
        completion_markers=("formal/test/TEST_RUN_COMPLETE.json",),
        verify_hashes=verify_hashes,
    )
    known = (
        audit_known_artifacts(verify_hashes=verify_hashes)
        if verify_all_known_artifacts
        else None
    )
    return {
        "schema_version": 1,
        "kind": "canonical_reranking_provenance_audit",
        "source_mutated": False,
        "modular": modular,
        "crog": crog,
        "known_artifacts": known,
        "passed": modular["passed"] and crog["passed"] and (
            known is None or known["passed"]
        ),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_safe_new_output(
    output_dir: Path,
    source_roots: Sequence[str | os.PathLike[str]],
) -> None:
    output = output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"audit output directory must be new: {output}")
    protected_names = {
        "modular_run",
        "modular_incomplete_rerank_run",
        "crog_run",
        "crog_candidates",
        "hifi_label_evaluator",
    }
    protected: set[Path] = {
        Path(CANONICAL_PATHS[name]).resolve()
        for name in protected_names
        if Path(CANONICAL_PATHS[name]).exists()
    }
    protected.update(Path(path).resolve() for path in source_roots)
    for source in protected:
        if _is_relative_to(output, source) or _is_relative_to(source, output):
            raise ProvenanceAuditError(
                f"audit output overlaps protected source path: {source}"
            )


def _inventory_rows(value: Any, section: str = "report") -> Iterable[dict[str, Any]]:
    if isinstance(value, Mapping):
        if "inventory" in value and isinstance(value["inventory"], list):
            for item in value["inventory"]:
                if isinstance(item, Mapping):
                    yield {
                        "section": section,
                        "path": item.get("path", item.get("relative_path", "")),
                        "exists": item.get("exists", ""),
                        "size_bytes": item.get(
                            "actual_size_bytes", item.get("size_bytes", "")
                        ),
                        "expected_sha256": item.get("expected_sha256", item.get("sha256", "")),
                        "actual_sha256": item.get("actual_sha256", ""),
                        "sha256_matches": item.get("sha256_matches", ""),
                    }
        for key, child in value.items():
            if key != "inventory":
                yield from _inventory_rows(child, f"{section}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _inventory_rows(child, f"{section}[{index}]")


def write_audit_bundle(
    report: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    source_roots: Sequence[str | os.PathLike[str]] = (),
) -> dict[str, str]:
    """Write JSON/Markdown/TSV into one new, source-disjoint directory."""

    output = Path(output_dir)
    _assert_safe_new_output(output, source_roots)
    json_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    passed = bool(report.get("passed", False))
    markdown = "\n".join(
        [
            "# Reranking provenance audit",
            "",
            f"- Status: {'PASS' if passed else 'FAIL'}",
            "- Canonical sources mutated: no",
            "- Machine-readable details: `audit.json`",
            "- File inventory: `inventory.tsv`",
            "",
        ]
    )
    rows = list(_inventory_rows(report))
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"audit output parent directory does not exist: {output.parent}"
        )
    output.mkdir(exist_ok=False)
    json_path = output / "audit.json"
    markdown_path = output / "audit.md"
    tsv_path = output / "inventory.tsv"
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "section",
                "path",
                "exists",
                "size_bytes",
                "expected_sha256",
                "actual_sha256",
                "sha256_matches",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output_dir": str(output.resolve()),
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "tsv": str(tsv_path.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--verify-all-known-artifacts", action="store_true")
    args = parser.parse_args(argv)
    report = audit_canonical_runs(
        verify_hashes=not args.no_hash,
        verify_all_known_artifacts=args.verify_all_known_artifacts,
    )
    outputs = write_audit_bundle(
        report,
        args.output_dir,
        source_roots=(MODULAR_CANONICAL_RUN, CROG_CANONICAL_RUN),
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProvenanceAuditError",
    "audit_canonical_runs",
    "audit_known_artifacts",
    "audit_run",
    "main",
    "manifest_file_inventory",
    "write_audit_bundle",
]
