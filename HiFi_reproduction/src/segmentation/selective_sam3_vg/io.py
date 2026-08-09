"""Prediction-only I/O shared by validation and formal selective inference.

This module deliberately has no ground-truth path resolver. Ground truth is
confined to :mod:`evaluation` and audit/reporting entry points.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CompactPrediction:
    split: str
    sample_index: int
    sample_id: str
    question_index: int
    scene_id: str
    query: str
    rgb_path: Path
    depth_path: Path
    probability_path: Path
    native_mask_path: Path
    probability_sha256: str
    native_mask_sha256: str
    checkpoint_sha256: str
    manifest_sha256: str
    foreground_threshold: float
    raw: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path(row: dict[str, Any], *names: str) -> Path:
    for name in names:
        raw = row.get(name)
        if raw:
            path = Path(str(raw)).expanduser().resolve()
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing or unsafe prediction input: {path}")
            return path
    raise ValueError(f"prediction row has none of the required paths: {names}")


def load_compact_manifest(
    path: str | Path,
    *,
    expected_split: str,
    expected_count: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> list[CompactPrediction]:
    manifest_path = Path(path).expanduser().resolve()
    rows: list[CompactPrediction] = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if str(raw.get("split")) != expected_split:
                raise ValueError(
                    f"compact manifest split mismatch at line {line_number}: "
                    f"{raw.get('split')!r}"
                )
            checkpoint_sha = str(raw.get("checkpoint_sha256", ""))
            manifest_sha = str(raw.get("manifest_sha256", ""))
            if expected_checkpoint_sha256 and checkpoint_sha != expected_checkpoint_sha256:
                raise ValueError(f"checkpoint identity mismatch at line {line_number}")
            if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
                raise ValueError(f"split-manifest identity mismatch at line {line_number}")
            threshold = float(raw.get("foreground_threshold", -1.0))
            if threshold != 0.5 or raw.get("threshold_comparison") != ">=":
                raise ValueError(f"unexpected coarse-mask threshold contract at line {line_number}")
            rows.append(
                CompactPrediction(
                    split=expected_split,
                    sample_index=int(raw["sample_index"]),
                    sample_id=str(raw["sample_id"]),
                    question_index=int(raw["question_index"]),
                    scene_id=str(raw["scene_id"]),
                    query=str(raw["query"]),
                    rgb_path=_path(raw, "source_rgb_path", "processed_rgb_path"),
                    depth_path=_path(raw, "source_depth_path"),
                    probability_path=_path(raw, "probability_path"),
                    native_mask_path=_path(raw, "native_mask_path"),
                    probability_sha256=str(raw["probability_sha256"]),
                    native_mask_sha256=str(raw["native_mask_sha256"]),
                    checkpoint_sha256=checkpoint_sha,
                    manifest_sha256=manifest_sha,
                    foreground_threshold=threshold,
                    raw=raw,
                )
            )
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"expected {expected_count} predictions, found {len(rows)}")
    indices = [row.sample_index for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("compact prediction sample_index values are not ordered and contiguous")
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("compact prediction sample_id values are not unique")
    return rows


def load_probability(path: str | Path) -> np.ndarray:
    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if loaded.files != ["probability"]:
                raise ValueError(f"unexpected probability archive keys: {loaded.files}")
            array = loaded["probability"]
        finally:
            loaded.close()
    else:
        array = loaded
    probability = np.asarray(array, dtype=np.float32)
    if probability.shape != (352, 352):
        raise ValueError(f"probability map must be 352x352, got {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("probability map contains non-finite values")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("probability map is outside [0,1]")
    return probability


def load_binary_mask(path: str | Path, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(f"mask shape {mask.shape} != {expected_shape}")
    return np.asarray(mask, dtype=bool)


def resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = (int(shape[0]), int(shape[1]))
    source = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    resized = source.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= 128


def resize_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map probabilities exactly as the repository's inverse-resize exporter.

    PyTorch bilinear interpolation with ``align_corners=False`` is the frozen
    local source of truth. The import stays lazy so basic manifest audits do
    not initialize PyTorch.
    """

    height, width = (int(shape[0]), int(shape[1]))
    array = np.asarray(probability, dtype=np.float32)
    if array.shape == (height, width):
        return array.copy()
    import torch
    import torch.nn.functional as torch_functional

    tensor = torch.from_numpy(array)[None, None]
    result = (
        torch_functional.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .numpy()
        .astype(np.float32, copy=False)
    )
    if result.shape != (height, width) or not np.isfinite(result).all():
        raise ValueError("probability resize produced an invalid map")
    return np.clip(result, 0.0, 1.0)


def aggregate_identity_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
