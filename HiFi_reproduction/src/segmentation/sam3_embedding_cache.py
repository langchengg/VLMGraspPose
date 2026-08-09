"""Portable safetensors cache for official SAM 3 image representations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddingCacheKey:
    rgb_sha256: str
    model_revision: str
    processor_sha256: str
    input_resolution: int
    dtype: str
    backend_class: str

    def __post_init__(self) -> None:
        for name in ("rgb_sha256", "processor_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if len(self.model_revision) != 40:
            raise ValueError("model_revision must be an immutable commit SHA")
        if int(self.input_resolution) <= 0:
            raise ValueError("input_resolution must be positive")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json_files(paths: list[Path]) -> str:
    records = [(str(path.name), sha256_file(path)) for path in sorted(paths)]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class Sam3EmbeddingCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: EmbeddingCacheKey) -> tuple[Path, Path]:
        backend = key.backend_class.replace(".", "_").lower()
        directory = self.root / backend / key.digest[:2]
        return directory / f"{key.digest}.safetensors", directory / f"{key.digest}.json"

    def exists(self, key: EmbeddingCacheKey) -> bool:
        tensor_path, metadata_path = self._paths(key)
        return tensor_path.is_file() and metadata_path.is_file()

    def _write(self, key: EmbeddingCacheKey, tensors: dict[str, Any], kind: str) -> dict[str, Any]:
        import torch
        from safetensors.torch import save_file

        tensor_path, metadata_path = self._paths(key)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, value in tensors.items()
        }
        descriptor, raw = tempfile.mkstemp(prefix=f".{tensor_path.name}.", dir=tensor_path.parent)
        os.close(descriptor)
        temporary_tensor = Path(raw)
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
        try:
            save_file(serializable, temporary_tensor)
            payload = {
                "schema_version": 1,
                "kind": kind,
                "cache_key": asdict(key),
                "cache_digest": key.digest,
                "tensor_sha256": sha256_file(temporary_tensor),
                "tensor_shapes": {name: list(value.shape) for name, value in serializable.items()},
                "tensor_dtypes": {name: str(value.dtype) for name, value in serializable.items()},
            }
            temporary_metadata.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary_tensor.replace(tensor_path)
            temporary_metadata.replace(metadata_path)
        finally:
            temporary_tensor.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
        return payload

    def _read(self, key: EmbeddingCacheKey, expected_kind: str) -> dict[str, Any]:
        from safetensors.torch import load_file

        tensor_path, metadata_path = self._paths(key)
        if not tensor_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"SAM 3 embedding cache miss: {key.digest}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("kind") != expected_kind
            or metadata.get("cache_key") != asdict(key)
            or metadata.get("cache_digest") != key.digest
            or metadata.get("tensor_sha256") != sha256_file(tensor_path)
        ):
            raise ValueError(f"SAM 3 embedding cache integrity failure: {key.digest}")
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != set(metadata["tensor_shapes"]):
            raise ValueError("cached tensor schema mismatch")
        for name, value in tensors.items():
            if list(value.shape) != metadata["tensor_shapes"][name]:
                raise ValueError(f"cached tensor shape mismatch: {name}")
        return tensors

    def save_pcs(self, key: EmbeddingCacheKey, vision_output: Any) -> dict[str, Any]:
        tensors: dict[str, Any] = {}
        for index, value in enumerate(vision_output.fpn_hidden_states):
            tensors[f"fpn_hidden_{index}"] = value
        for index, value in enumerate(vision_output.fpn_position_encoding):
            tensors[f"fpn_position_{index}"] = value
        return self._write(key, tensors, "Sam3VisionEncoderOutput")

    def load_pcs(self, key: EmbeddingCacheKey) -> Any:
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        tensors = self._read(key, "Sam3VisionEncoderOutput")
        hidden = tuple(tensors[name] for name in sorted(tensors) if name.startswith("fpn_hidden_"))
        positions = tuple(
            tensors[name] for name in sorted(tensors) if name.startswith("fpn_position_")
        )
        return Sam3VisionEncoderOutput(
            fpn_hidden_states=hidden,
            fpn_position_encoding=positions,
        )

    def save_tracker(self, key: EmbeddingCacheKey, embeddings: list[Any]) -> dict[str, Any]:
        return self._write(
            key,
            {f"image_embedding_{index}": value for index, value in enumerate(embeddings)},
            "Sam3TrackerImageEmbeddings",
        )

    def load_tracker(self, key: EmbeddingCacheKey) -> list[Any]:
        tensors = self._read(key, "Sam3TrackerImageEmbeddings")
        return [tensors[name] for name in sorted(tensors)]

    def manifest_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("**/*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                records.append({"path": str(path), "status": "INVALID"})
        return records

    def prune_to_max_bytes(
        self,
        max_bytes: int,
        *,
        protected_rgb_sha256: str | None = None,
    ) -> dict[str, int]:
        """Bound cache storage after a frame while retaining its active entries."""

        entries: list[tuple[float, Path, Path, int, str | None]] = []
        total = 0
        for metadata_path in self.root.glob("**/*.json"):
            tensor_path = metadata_path.with_suffix(".safetensors")
            if not tensor_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                rgb_sha = metadata.get("cache_key", {}).get("rgb_sha256")
            except (OSError, json.JSONDecodeError):
                rgb_sha = None
            size = tensor_path.stat().st_size + metadata_path.stat().st_size
            total += size
            entries.append(
                (
                    min(tensor_path.stat().st_mtime, metadata_path.stat().st_mtime),
                    tensor_path,
                    metadata_path,
                    size,
                    rgb_sha,
                )
            )
        removed_entries = 0
        removed_bytes = 0
        for _, tensor_path, metadata_path, size, rgb_sha in sorted(entries):
            if total <= int(max_bytes):
                break
            if protected_rgb_sha256 and rgb_sha == protected_rgb_sha256:
                continue
            tensor_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            total -= size
            removed_entries += 1
            removed_bytes += size
        return {
            "remaining_bytes": int(total),
            "removed_entries": removed_entries,
            "removed_bytes": int(removed_bytes),
        }


__all__ = [
    "EmbeddingCacheKey",
    "Sam3EmbeddingCache",
    "sha256_file",
    "sha256_json_files",
]
