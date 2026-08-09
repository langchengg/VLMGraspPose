from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import SCHEMA_VERSION


# The list is deliberately stricter than V2.  Matching is case-insensitive and
# applies to both a field name and its complete lineage.
FORBIDDEN_INFERENCE_TOKENS = (
    "ground_truth",
    "target_idx",
    "angle_error",
    "positive_label",
    "evaluation_result",
    "matched_gt",
    "correctness",
    "success",
    "answer",
    "objid",
    "label",
    "jany",
    "oracle",
    "iou",
    "j1",
    "gt",
)

ALLOWED_INFERENCE_LINEAGES = (
    "rgb_derived",
    "text_derived",
    "depth_derived",
    "predicted_maps",
    "frozen_latent",
    "candidate_geometry",
    "candidate_relations",
    "oof_model_outputs",
    "missingness",
    "uncertainty",
)

FEATURE_GROUPS = {
    "G0": "frozen candidate and q-list evidence",
    "G1": "Q-map local shape and peak evidence",
    "G2": "predicted-mask support",
    "G3": "axial-angle confidence and consistency",
    "G4": "width-head confidence and geometry consistency",
    "G5": "five-head joint and cross-head consistency",
    "G6": "multiscale visual and multimodal latent ROI",
    "G7": "candidate-conditioned token interaction",
    "G8": "candidate-aligned RGB and predicted-map crop",
    "G9": "optional RGB-D geometry",
    "G10": "permutation-equivariant candidate-set context",
    "G11": "provenance-valid V2 OOF prior evidence",
}


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


def atomic_write_text(path: str | Path, content: str, *, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"immutable output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{sha256_bytes(content.encode())[:8]}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        overwrite=overwrite,
    )


def atomic_write_jsonl(path: str | Path, records: Iterable[Any], *, overwrite: bool = False) -> Path:
    return atomic_write_text(
        path,
        "".join(canonical_json(record) + "\n" for record in records),
        overwrite=overwrite,
    )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def artifact_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def find_forbidden_token(name_or_lineage: str) -> str | None:
    lowered = str(name_or_lineage).lower()
    return next((token for token in FORBIDDEN_INFERENCE_TOKENS if token in lowered), None)


def assert_inference_field(name: str, lineage: str) -> None:
    for value in (name, lineage):
        token = find_forbidden_token(value)
        if token is not None:
            raise ValueError(f"forbidden inference field {name!r}: {value!r} contains {token!r}")
    root = lineage.split(".", 1)[0]
    if root not in ALLOWED_INFERENCE_LINEAGES:
        raise ValueError(f"inference lineage {lineage!r} is not allowlisted")


def scan_inference_record(value: Any, *, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            token = find_forbidden_token(str(key))
            if token is not None:
                raise ValueError(f"forbidden inference field {path}.{key} contains {token!r}")
            scan_inference_record(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            scan_inference_record(child, path=f"{path}[{index}]")


def stable_sample_id(split: str, source_id: int | str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"unsupported official split: {split}")
    return f"multiple:{normalized}:{int(source_id):08d}"


def schema_hash(rows: Iterable[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(list(rows)).encode("utf-8"))

