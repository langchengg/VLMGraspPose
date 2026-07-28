from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION


FORBIDDEN_MODEL_FIELD_TOKENS = (
    "gt",
    "label",
    "success",
    "j1",
    "jany",
    "iou",
    "angle_error",
    "oracle",
    "correctness",
)

SCALAR_FEATURES = (
    "q",
    "q_patch_mean",
    "q_prominence",
    "mask_consistency",
    "soft_coverage",
    "binary_coverage",
    "center_prob",
    "center_margin",
    "image_support",
    "width_compatibility",
    "width_ratio",
    "width_symmetry",
    "angle_consistency",
    "depth_mad_m",
    "contact_depth_difference_m",
    "safety",
    "collision_proxy",
    "clearance",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_id(split: str, local_sample_id: int | str) -> str:
    """Make OCID-VLG's split-local question index globally unique."""
    split = str(split).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    return f"multiple:{split}:{int(local_sample_id):08d}"


def stable_candidate_id(sample_id: str, candidate_id: str) -> str:
    return f"{sample_id}/{candidate_id}"


def assert_model_feature_names(names: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    for name in result:
        lowered = name.lower()
        token = next(
            (item for item in FORBIDDEN_MODEL_FIELD_TOKENS if item in lowered),
            None,
        )
        if token is not None:
            raise ValueError(
                f"forbidden model-input field {name!r} contains token {token!r}"
            )
    return result


def assert_inference_record_has_no_evaluation_fields(
    value: Any, *, path: str = "record"
) -> None:
    """Reject evaluation-derived fields anywhere in an inference artifact."""
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            lowered = name.lower()
            token = next(
                (
                    item
                    for item in FORBIDDEN_MODEL_FIELD_TOKENS
                    if item in lowered
                ),
                None,
            )
            if token is not None:
                raise ValueError(
                    f"forbidden inference field {path}.{name} "
                    f"contains token {token!r}"
                )
            assert_inference_record_has_no_evaluation_fields(
                child, path=f"{path}.{name}"
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_inference_record_has_no_evaluation_fields(
                child, path=f"{path}[{index}]"
            )


def atomic_write_text(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{hashlib.sha256(content.encode()).hexdigest()[:8]}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def append_jsonl_record(handle, value: Any) -> None:
    handle.write(canonical_json(value) + "\n")
    handle.flush()


def atomic_write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def code_fingerprint(repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    files = []
    for relative in (
        "failure_analysis/reranking",
        "failure_analysis/reranking_v2",
        "model",
        "utils",
    ):
        files.extend((root / relative).rglob("*.py"))
        files.extend((root / relative).rglob("*.json"))
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in files if item.is_file()}):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def schema_header(kind: str) -> dict[str, str]:
    return {"schema_version": SCHEMA_VERSION, "kind": str(kind)}
