"""Read-only discovery and eligibility audit for existing reranker runs.

R10 reuse is intentionally conservative.  A checkpoint on disk is not a
completed experiment: a method is eligible only when repository content binds
it to a completed manifest, complete OOF evidence, verified checkpoints,
candidate and evaluator hashes, and a passing independent recomputation.

Method identifiers are read from manifest/config content.  This module never
uses method-name allowlists and never follows symlinks while scanning or while
validating referenced artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml"})
_CHECKPOINT_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".safetensors", ".joblib", ".pkl"}
)
_COMPLETE_STATUSES = frozenset(
    {"complete", "completed", "locked", "pass", "passed", "success", "verified"}
)
_PASS_STATUSES = frozenset({"pass", "passed", "success", "verified", "exact"})
_SINGULAR_METHOD_KEYS = frozenset(
    {
        "method",
        "method_id",
        "method_name",
        "model_id",
        "model_key",
        "model_name",
        "primary_method",
        "locked_method",
        "selected_method",
    }
)
_PLURAL_METHOD_KEYS = frozenset(
    {"methods", "formal_methods", "exploratory_methods", "method_names"}
)
_REQUIRED_CHECKS = (
    "manifest",
    "complete_oof",
    "checkpoint",
    "candidate_hash",
    "evaluator_hash",
    "independent_recomputation",
)


@dataclass(frozen=True)
class _Document:
    path: Path
    scan_root: Path
    payload: Any
    sha256: str
    kind: str


@dataclass(frozen=True)
class _Declaration:
    method_id: str
    document: _Document
    trail: str
    block: Mapping[str, Any]
    role: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_status(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _valid_method_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 256 or any(char in candidate for char in "\n\r\0"):
        return None
    return candidate


def _normal_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-z0-9]+", value.lower()) if token)


def _mentions_method(value: str, method_id: str) -> bool:
    haystack = "_".join(_normal_tokens(value))
    needle = "_".join(_normal_tokens(method_id))
    if needle and needle in haystack:
        return True
    # Composite method names commonly add protocol words around the component
    # recorded in an artifact label.  Requiring two informative shared tokens
    # avoids treating generic words such as "model" or "oof" as identity.
    ignored = {
        "model",
        "method",
        "oof",
        "ensemble",
        "final",
        "primary",
        "locked",
        "residual",
        "full",
        "feature",
    }
    informative = {token for token in _normal_tokens(method_id) if token not in ignored}
    present = informative & set(_normal_tokens(value))
    return len(present) >= min(2, len(informative)) if informative else False


def _walk(value: Any, trail: str = "$") -> Iterable[tuple[str, Any]]:
    yield trail, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{trail}[{index}]")


def _document_kind(path: Path, payload: Any) -> str:
    name = path.name.lower()
    declared_kind = ""
    if isinstance(payload, Mapping):
        declared_kind = str(payload.get("kind", "")).lower()
    if "manifest" in name or "manifest" in declared_kind or "lock" in name:
        return "manifest"
    if "config" in name or "config" in declared_kind or path.suffix.lower() in {".yaml", ".yml"}:
        return "config"
    return "evidence"


def _load_document(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - local project has PyYAML
        raise ValueError("PyYAML is required to inspect YAML configs") from error
    return yaml.safe_load(text)


def _contains_symlink(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            return True
    return False


def _scan(
    roots: Sequence[str | os.PathLike[str]],
) -> tuple[list[_Document], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[_Document] = []
    checkpoints: list[dict[str, Any]] = []
    skipped_symlinks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_roots: set[Path] = set()
    for raw_root in roots:
        root = Path(os.path.abspath(os.fspath(raw_root)))
        if root in seen_roots:
            continue
        seen_roots.add(root)
        if root.is_symlink() or not root.is_dir():
            errors.append(
                {
                    "code": "invalid_scan_root",
                    "path": str(root),
                    "symlink": root.is_symlink(),
                    "message": "scan root must be a non-symlink directory",
                }
            )
            continue
        for directory, names, files in os.walk(root, followlinks=False):
            current = Path(directory)
            retained_names: list[str] = []
            for name in sorted(names):
                child = current / name
                if child.is_symlink():
                    skipped_symlinks.append(
                        {"path": str(child), "type": "directory"}
                    )
                else:
                    retained_names.append(name)
            names[:] = retained_names
            for filename in sorted(files):
                path = current / filename
                if path.is_symlink():
                    skipped_symlinks.append({"path": str(path), "type": "file"})
                    continue
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix in _CHECKPOINT_SUFFIXES:
                    checkpoints.append(
                        {
                            "path": str(path),
                            "size_bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                            "attributed_method": None,
                        }
                    )
                if suffix not in _DOCUMENT_SUFFIXES:
                    continue
                if suffix == ".jsonl" and not any(
                    token in filename.lower()
                    for token in ("manifest", "config", "summary", "evidence", "recomput")
                ):
                    # Prediction/provenance JSONL files can be hundreds of MB.
                    # Their integrity is checked through manifest descriptors;
                    # they are not themselves method declarations.
                    continue
                try:
                    payload = _load_document(path)
                except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                    errors.append(
                        {
                            "code": "document_parse_failed",
                            "path": str(path),
                            "message": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                documents.append(
                    _Document(
                        path=path,
                        scan_root=root,
                        payload=payload,
                        sha256=_sha256(path),
                        kind=_document_kind(path, payload),
                    )
                )
    return documents, checkpoints, skipped_symlinks, errors


def _method_declarations(document: _Document) -> list[_Declaration]:
    declarations: list[_Declaration] = []
    seen: set[tuple[str, str]] = set()

    def add(method: Any, trail: str, block: Any, role: str) -> None:
        method_id = _valid_method_id(method)
        if method_id is None:
            return
        key = (method_id, trail)
        if key in seen:
            return
        seen.add(key)
        declarations.append(
            _Declaration(
                method_id=method_id,
                document=document,
                trail=trail,
                block=block if isinstance(block, Mapping) else {},
                role=role,
            )
        )

    def visit(value: Any, trail: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).lower()
                child_trail = f"{trail}.{raw_key}"
                if key in _SINGULAR_METHOD_KEYS:
                    add(child, child_trail, value, key)
                elif key in _PLURAL_METHOD_KEYS:
                    if isinstance(child, Mapping):
                        for method, block in child.items():
                            add(method, f"{child_trail}.{method}", block, key)
                            visit(block, f"{child_trail}.{method}")
                        continue
                    if isinstance(child, list):
                        for index, item in enumerate(child):
                            item_trail = f"{child_trail}[{index}]"
                            if isinstance(item, Mapping):
                                method = next(
                                    (
                                        item.get(candidate)
                                        for candidate in _SINGULAR_METHOD_KEYS
                                        if candidate in item
                                    ),
                                    None,
                                )
                                add(method, item_trail, item, key)
                            else:
                                add(item, item_trail, value, key)
                            visit(item, item_trail)
                        continue
                elif key == "spec" and isinstance(child, Mapping):
                    add(child.get("key"), child_trail, value, "spec.key")
                visit(child, child_trail)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{trail}[{index}]")

    visit(document.payload, "$")
    return declarations


def _scope_roots(
    documents: Sequence[_Document], declarations: Sequence[_Declaration]
) -> list[Path]:
    candidates = {
        declaration.document.path.parent
        for declaration in declarations
        if declaration.document.kind in {"manifest", "config"}
    }
    # A shallow run manifest owns its referenced/nested evidence; nested
    # artifact manifests must not split the same run into unrelated scopes.
    roots: list[Path] = []
    for candidate in sorted(candidates, key=lambda path: (len(path.parts), str(path))):
        if not any(root == candidate or root in candidate.parents for root in roots):
            roots.append(candidate)
    if roots:
        return roots
    return sorted({document.scan_root for document in documents}, key=str)


def _scope_for(document: _Document, scopes: Sequence[Path]) -> Path:
    owners = [
        scope
        for scope in scopes
        if scope == document.path.parent or scope in document.path.parents
    ]
    if owners:
        return max(owners, key=lambda path: len(path.parts))
    relative = document.path.relative_to(document.scan_root)
    return (
        document.scan_root / relative.parts[0]
        if len(relative.parts) > 1
        else document.scan_root
    )


def _resolve_descriptor_path(document: _Document, declared: str) -> Path:
    candidate = Path(declared).expanduser()
    if not candidate.is_absolute():
        candidate = document.path.parent / candidate
    return Path(os.path.abspath(candidate))


def _audit_descriptor(
    document: _Document,
    trail: str,
    value: Mapping[str, Any],
    artifact_cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    declared = value.get("path")
    expected = str(value.get("sha256", "")).lower()
    path = (
        _resolve_descriptor_path(document, declared)
        if isinstance(declared, str) and declared
        else None
    )
    cached = artifact_cache.get(path) if artifact_cache is not None and path is not None else None
    if cached is None:
        symlink = bool(path is not None and _contains_symlink(path))
        regular = bool(path is not None and path.is_file() and not symlink)
        actual = _sha256(path) if regular and path is not None else None
        cached = {
            "symlink": symlink,
            "regular_file": regular,
            "actual_sha256": actual,
            "size_bytes": path.stat().st_size if regular and path is not None else None,
        }
        if artifact_cache is not None and path is not None:
            artifact_cache[path] = cached
    symlink = bool(cached["symlink"])
    regular = bool(cached["regular_file"])
    actual = cached["actual_sha256"]
    expected_valid = bool(_SHA256.fullmatch(expected))
    issue = None
    if path is None:
        issue = "artifact_path_missing"
    elif symlink:
        issue = "artifact_symlink"
    elif not regular:
        issue = "artifact_missing"
    elif not expected_valid:
        issue = "artifact_sha256_invalid"
    elif actual != expected:
        issue = "artifact_sha256_mismatch"
    return {
        "source_document": str(document.path),
        "manifest_location": trail,
        "declared_path": declared,
        "path": str(path) if path is not None else None,
        "expected_sha256": expected or None,
        "actual_sha256": actual,
        "size_bytes": cached["size_bytes"],
        "regular_file": regular,
        "symlink": symlink,
        "sha256_matches": bool(expected_valid and actual == expected),
        "passed": issue is None,
        "issue": issue,
    }


def _descriptors(
    document: _Document,
    value: Any | None = None,
    *,
    artifact_cache: dict[Path, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source = document.payload if value is None else value
    records: list[dict[str, Any]] = []
    for trail, child in _walk(source):
        if (
            isinstance(child, Mapping)
            and isinstance(child.get("path"), str)
            and "sha256" in child
        ):
            records.append(
                _audit_descriptor(
                    document, trail, child, artifact_cache=artifact_cache
                )
            )
    return records


def _descriptor_class(record: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        str(record.get(key, ""))
        for key in ("manifest_location", "declared_path", "path")
    ).lower()
    suffix = Path(str(record.get("path") or "")).suffix.lower()
    classes: set[str] = set()
    if suffix in _CHECKPOINT_SUFFIXES or any(
        token in text for token in ("checkpoint", "model", "weights")
    ):
        classes.add("checkpoint")
    if any(token in text for token in ("candidate", "frozen_features", "frozen_predictions")):
        classes.add("candidate")
    if any(token in text for token in ("evaluator", "evaluation_code", "metric_code")):
        classes.add("evaluator")
    if "oof" in text:
        classes.add("oof")
    if any(token in text for token in ("independent", "recomput", "replication")):
        classes.add("independent")
    if "config" in text or "run_spec" in text:
        classes.add("config")
    return classes


def _method_is_primary(method_id: str, documents: Sequence[_Document]) -> bool:
    for document in documents:
        if not isinstance(document.payload, Mapping):
            continue
        for key in ("primary_method", "locked_method", "selected_method"):
            if document.payload.get(key) == method_id:
                return True
    return False


def _record_associated(
    record: Mapping[str, Any],
    method_id: str,
    *,
    primary: bool,
    method_count: int,
) -> bool:
    text = " ".join(
        str(record.get(key, ""))
        for key in ("manifest_location", "declared_path", "path")
    )
    if _mentions_method(text, method_id):
        return True
    return primary or method_count == 1


def _hash_evidence(
    documents: Sequence[_Document],
    descriptors: Sequence[dict[str, Any]],
    *,
    category: str,
) -> list[dict[str, Any]]:
    evidence = [record for record in descriptors if category in _descriptor_class(record)]
    keyword = "candidate" if category == "candidate" else "evaluator"
    for document in documents:
        for trail, value in _walk(document.payload):
            if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
                continue
            lowered = trail.lower()
            if keyword not in lowered or not any(token in lowered for token in ("sha", "hash")):
                continue
            evidence.append(
                {
                    "source_document": str(document.path),
                    "manifest_location": trail,
                    "path": None,
                    "expected_sha256": value.lower(),
                    "actual_sha256": None,
                    "sha256_matches": None,
                    "passed": True,
                    "issue": None,
                    "hash_only": True,
                }
            )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in evidence:
        key = (
            item.get("source_document"),
            item.get("manifest_location"),
            item.get("path"),
            item.get("expected_sha256"),
        )
        unique[key] = item
    return sorted(unique.values(), key=lambda item: str(item.get("manifest_location")))


def _fold_values(value: Any) -> set[int] | None:
    if not isinstance(value, list):
        return None
    output: set[int] = set()
    for item in value:
        try:
            output.add(int(item))
        except (TypeError, ValueError):
            return None
    return output


def _oof_mapping_passes(
    value: Mapping[str, Any], descriptor_records: Sequence[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    status = _normalized_status(value.get("status"))
    explicit_complete = any(
        value.get(key) is True
        for key in (
            "complete",
            "completed",
            "oof_complete",
            "coverage_complete",
            "all_folds_complete",
        )
    )
    status_complete = status in _COMPLETE_STATUSES
    expected = _fold_values(value.get("expected_folds"))
    completed = _fold_values(value.get("completed_folds"))
    coverage = bool(expected is not None and expected and completed == expected)
    fold_count = value.get("fold_count")
    fold_models = value.get("fold_models")
    if isinstance(fold_count, int) and fold_count > 0 and isinstance(fold_models, list):
        coverage = coverage or len(fold_models) == fold_count
    folds = _fold_values(value.get("folds"))
    if folds and isinstance(fold_models, list):
        coverage = coverage or len(fold_models) == len(folds)
    oof_descriptors = [
        record
        for record in descriptor_records
        if "oof" in _descriptor_class(record)
        or any(
            token in str(record.get("manifest_location", "")).lower()
            for token in ("prediction", "provenance")
        )
    ]
    artifacts_valid = bool(oof_descriptors) and all(
        record["passed"] for record in oof_descriptors
    )
    passed = bool((status_complete or explicit_complete) and coverage and artifacts_valid)
    return passed, {
        "status": status or None,
        "explicit_complete": explicit_complete,
        "coverage_complete": coverage,
        "artifacts_valid": artifacts_valid,
        "artifacts": oof_descriptors,
    }


def _oof_evidence(
    method_id: str,
    declarations: Sequence[_Declaration],
    documents: Sequence[_Document],
    descriptor_records: Sequence[dict[str, Any]],
    reference_sources: Mapping[Path, list[dict[str, Any]]],
    *,
    primary: bool,
    artifact_cache: dict[Path, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[_Document, str, Mapping[str, Any]]] = []
    for declaration in declarations:
        if declaration.method_id != method_id:
            continue
        for trail, child in _walk(declaration.block, declaration.trail):
            if isinstance(child, Mapping) and "oof" in trail.lower():
                candidates.append((declaration.document, trail, child))
    for document in documents:
        path_text = f"{document.path.parent.name}/{document.path.name}".lower()
        payload_kind = (
            str(document.payload.get("kind", "")).lower()
            if isinstance(document.payload, Mapping)
            else ""
        )
        if not isinstance(document.payload, Mapping):
            continue
        sources = reference_sources.get(document.path, [])
        associated_reference = any(
            _mentions_method(str(item.get("manifest_location", "")), method_id)
            or primary
            for item in sources
            if "oof" in _descriptor_class(item)
        )
        if (
            ("oof" in path_text or "oof" in payload_kind or associated_reference)
            and (_mentions_method(path_text, method_id) or primary or associated_reference)
        ):
            candidates.append((document, "$", document.payload))
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document, trail, mapping in candidates:
        key = (str(document.path), trail)
        if key in seen:
            continue
        seen.add(key)
        local_descriptors = _descriptors(
            document, mapping, artifact_cache=artifact_cache
        )
        passed, detail = _oof_mapping_passes(mapping, local_descriptors)
        output.append(
            {
                "source_document": str(document.path),
                "source_document_sha256": document.sha256,
                "manifest_location": trail,
                "passed": passed,
                **detail,
            }
        )
    return output


def _declared_methods(value: Any) -> set[str]:
    synthetic = _Document(Path("/content"), Path("/"), value, "", "evidence")
    return {item.method_id for item in _method_declarations(synthetic)}


def _independent_payload_passes(value: Mapping[str, Any]) -> bool:
    status = _normalized_status(value.get("status"))
    explicit = any(
        value.get(key) is True
        for key in (
            "passed",
            "pass",
            "exact_match",
            "matches",
            "agrees",
            "clean",
            "all_checks_passed",
            "all_correct_counts_exact",
            "all_oracle_counts_exact",
        )
    )
    return status in _PASS_STATUSES or explicit


def _independent_evidence(
    method_id: str,
    declarations: Sequence[_Declaration],
    documents: Sequence[_Document],
    reference_sources: Mapping[Path, list[dict[str, Any]]],
    *,
    primary: bool,
    artifact_cache: dict[Path, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        for trail, child in _walk(document.payload):
            if not isinstance(child, Mapping):
                continue
            local_path = document.path.name
            semantic_text = f"{trail} {child.get('kind', '')} {local_path}".lower()
            if not any(token in semantic_text for token in ("independent", "recomput", "replication")):
                continue
            declared = _declared_methods(child)
            associated = method_id in declared or _mentions_method(semantic_text, method_id)
            if not associated and primary and not declared:
                associated = True
            references = reference_sources.get(document.path, [])
            bound = document.kind == "manifest" or bool(references)
            if trail != "$":
                bound = True
            if not associated or not bound:
                continue
            key = (str(document.path), trail)
            if key in seen:
                continue
            seen.add(key)
            descriptors = _descriptors(
                document, child, artifact_cache=artifact_cache
            )
            passed = _independent_payload_passes(child) and all(
                record["passed"] for record in descriptors
            )
            output.append(
                {
                    "source_document": str(document.path),
                    "source_document_sha256": document.sha256,
                    "manifest_location": trail,
                    "status": child.get("status"),
                    "passed": passed,
                    "referenced_by": references,
                    "artifacts": descriptors,
                }
            )
    return output


def _check(
    passed: bool,
    evidence: Sequence[dict[str, Any]],
    *,
    gap_code: str,
    message: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    issues = [item for item in evidence if item.get("passed") is False]
    result = {"passed": bool(passed), "evidence": list(evidence), "issues": issues}
    if passed:
        return result, None
    return result, {
        "code": gap_code,
        "message": message,
        "evidence_issues": issues,
        "paths": sorted(
            {
                str(item.get("path") or item.get("source_document"))
                for item in evidence
                if item.get("path") or item.get("source_document")
            }
        ),
    }


def _scope_run_id(scope: Path, documents: Sequence[_Document]) -> str:
    identifiers = {
        str(document.payload["run_id"])
        for document in documents
        if isinstance(document.payload, Mapping)
        and isinstance(document.payload.get("run_id"), (str, int))
    }
    return sorted(identifiers)[0] if len(identifiers) == 1 else scope.name


def _audit_method(
    method_id: str,
    scope: Path,
    documents: Sequence[_Document],
    declarations: Sequence[_Declaration],
    method_count: int,
    artifact_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    own_declarations = [item for item in declarations if item.method_id == method_id]
    primary = _method_is_primary(method_id, documents)
    all_descriptors = [
        record
        for document in documents
        for record in _descriptors(document, artifact_cache=artifact_cache)
    ]
    reference_sources: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in all_descriptors:
        if record.get("passed") and record.get("path"):
            reference_sources[Path(str(record["path"]))].append(record)

    manifest_evidence: list[dict[str, Any]] = []
    for declaration in own_declarations:
        if declaration.document.kind != "manifest":
            continue
        payload_status = (
            declaration.document.payload.get("status")
            if isinstance(declaration.document.payload, Mapping)
            else None
        )
        block_status = declaration.block.get("status")
        status = block_status if block_status is not None else payload_status
        passed = _normalized_status(status) in _COMPLETE_STATUSES
        manifest_evidence.append(
            {
                "path": str(declaration.document.path),
                "sha256": declaration.document.sha256,
                "manifest_location": declaration.trail,
                "role": declaration.role,
                "status": status,
                "passed": passed,
            }
        )

    block_descriptors: list[dict[str, Any]] = []
    for declaration in own_declarations:
        block_descriptors.extend(
            _descriptors(
                declaration.document,
                declaration.block,
                artifact_cache=artifact_cache,
            )
        )
    checkpoint_candidates = [
        record
        for record in [*block_descriptors, *all_descriptors]
        if "checkpoint" in _descriptor_class(record)
        and _record_associated(
            record, method_id, primary=primary, method_count=method_count
        )
    ]
    checkpoint_unique = {
        (item.get("source_document"), item.get("manifest_location"), item.get("path")): item
        for item in checkpoint_candidates
    }
    checkpoint_evidence = sorted(
        checkpoint_unique.values(), key=lambda item: str(item.get("path"))
    )
    oof_evidence = _oof_evidence(
        method_id,
        own_declarations,
        documents,
        all_descriptors,
        reference_sources,
        primary=primary,
        artifact_cache=artifact_cache,
    )
    candidate_evidence = _hash_evidence(
        documents, all_descriptors, category="candidate"
    )
    evaluator_evidence = _hash_evidence(
        documents, all_descriptors, category="evaluator"
    )
    independent_evidence = _independent_evidence(
        method_id,
        own_declarations,
        documents,
        reference_sources,
        primary=primary,
        artifact_cache=artifact_cache,
    )

    checks: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, Any]] = []
    specifications = (
        (
            "manifest",
            bool(manifest_evidence) and any(item["passed"] for item in manifest_evidence),
            manifest_evidence,
            "manifest_missing_or_incomplete",
            "no completed/locked manifest declares this method",
        ),
        (
            "complete_oof",
            bool(oof_evidence) and any(item["passed"] for item in oof_evidence),
            oof_evidence,
            "complete_oof_missing",
            "complete, hashed OOF coverage evidence is absent or invalid",
        ),
        (
            "checkpoint",
            bool(checkpoint_evidence)
            and all(item["passed"] for item in checkpoint_evidence),
            checkpoint_evidence,
            "checkpoint_missing_or_invalid",
            "method checkpoints are absent, unbound, symlinked, or hash-invalid",
        ),
        (
            "candidate_hash",
            bool(candidate_evidence)
            and any(item["passed"] for item in candidate_evidence),
            candidate_evidence,
            "candidate_hash_missing_or_invalid",
            "candidate artifact/hash provenance is absent or invalid",
        ),
        (
            "evaluator_hash",
            bool(evaluator_evidence)
            and any(item["passed"] for item in evaluator_evidence),
            evaluator_evidence,
            "evaluator_hash_missing_or_invalid",
            "evaluator artifact/hash provenance is absent or invalid",
        ),
        (
            "independent_recomputation",
            bool(independent_evidence)
            and any(item["passed"] for item in independent_evidence),
            independent_evidence,
            "independent_recomputation_missing_or_failed",
            "bound independent recomputation PASS evidence is absent or failed",
        ),
    )
    for name, passed, evidence, code, message in specifications:
        checks[name], gap = _check(
            passed, evidence, gap_code=code, message=message
        )
        if gap is not None:
            gaps.append(gap)

    config_sources = sorted(
        {
            str(item.document.path)
            for item in own_declarations
            if item.document.kind == "config"
        }
        | {
            str(record["path"])
            for record in all_descriptors
            if "config" in _descriptor_class(record) and record.get("passed")
        }
    )
    eligible = all(checks[name]["passed"] for name in _REQUIRED_CHECKS)
    return {
        "method_id": method_id,
        "run_id": _scope_run_id(scope, documents),
        "run_root": str(scope),
        "primary_or_locked": primary,
        "eligible": eligible,
        "declarations": [
            {
                "path": str(item.document.path),
                "document_sha256": item.document.sha256,
                "document_kind": item.document.kind,
                "manifest_location": item.trail,
                "role": item.role,
            }
            for item in own_declarations
        ],
        "configuration_sources": config_sources,
        "checks": checks,
        "gaps": gaps,
    }


def discover_existing_runs(
    roots: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Discover methods and audit whether each is safe to reuse for R10.

    Args:
        roots: Directories containing one or more existing runs.  Traversal is
            recursive and never follows symlinked directories or files.

    Returns:
        A JSON-serializable report.  Callers should consume ``methods`` and
        select only entries whose ``eligible`` field is true.  ``gaps`` is the
        structured reason an otherwise discovered method must not be reused.
    """

    documents, checkpoint_inventory, skipped_symlinks, scan_errors = _scan(roots)
    declarations = [
        declaration
        for document in documents
        for declaration in _method_declarations(document)
    ]
    scopes = _scope_roots(documents, declarations)
    documents_by_scope: dict[Path, list[_Document]] = defaultdict(list)
    declarations_by_scope: dict[Path, list[_Declaration]] = defaultdict(list)
    for document in documents:
        documents_by_scope[_scope_for(document, scopes)].append(document)
    for declaration in declarations:
        declarations_by_scope[_scope_for(declaration.document, scopes)].append(
            declaration
        )

    methods: list[dict[str, Any]] = []
    artifact_cache: dict[Path, dict[str, Any]] = {}
    for scope in sorted(declarations_by_scope, key=str):
        local_declarations = declarations_by_scope[scope]
        method_ids = sorted({item.method_id for item in local_declarations})
        local_documents = documents_by_scope.get(scope, [])
        for method_id in method_ids:
            methods.append(
                _audit_method(
                    method_id,
                    scope,
                    local_documents,
                    local_declarations,
                    len(method_ids),
                    artifact_cache,
                )
            )
    methods.sort(key=lambda item: (item["run_root"], item["method_id"]))
    return {
        "schema_version": 1,
        "kind": "existing_reranker_r10_discovery",
        "read_only": True,
        "roots": [str(Path(os.path.abspath(os.fspath(root)))) for root in roots],
        "scan": {
            "documents": len(documents),
            "checkpoint_files": len(checkpoint_inventory),
            "skipped_symlinks": skipped_symlinks,
            "errors": scan_errors,
        },
        "checkpoint_inventory": checkpoint_inventory,
        "methods": methods,
        "eligible_methods": [
            {
                "run_id": item["run_id"],
                "run_root": item["run_root"],
                "method_id": item["method_id"],
            }
            for item in methods
            if item["eligible"]
        ],
        "ineligible_methods": [
            {
                "run_id": item["run_id"],
                "run_root": item["run_root"],
                "method_id": item["method_id"],
                "gap_codes": [gap["code"] for gap in item["gaps"]],
            }
            for item in methods
            if not item["eligible"]
        ],
    }


__all__ = ["discover_existing_runs"]
