"""Read the GT-free deployment contract and separately gated label contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .data_contract import assert_gt_free_columns


MaskSource = Literal["predicted", "gt_mask_oracle"]


@dataclass(frozen=True, slots=True)
class SampleArrays:
    sample_id: str
    scene_id: str
    language: str
    rgb: np.ndarray
    depth_mm: np.ndarray
    depth_m: np.ndarray
    binary_mask: np.ndarray
    probability: np.ndarray
    intrinsics: Mapping[str, Any] | None
    mask_source: MaskSource


def read_deployment_manifest(path: str | Path) -> list[dict[str, Any]]:
    table = pq.read_table(Path(path).expanduser().resolve())
    assert_gt_free_columns(table.column_names)
    rows = table.to_pylist()
    assert_gt_free_columns(rows[0].keys() if rows else ())
    return rows


def read_label_manifest(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(Path(path).expanduser().resolve()).to_pylist()


def aligned_labels(
    deployment: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in labels}
    if len(by_id) != len(labels):
        raise ValueError("duplicate sample IDs in labels")
    if set(by_id) != {str(row["sample_id"]) for row in deployment}:
        raise ValueError("deployment/label sample coverage mismatch")
    return [by_id[str(row["sample_id"])] for row in deployment]


class CompactSampleLoader:
    """Load content-addressed source arrays and reject manifest/file drift."""

    def __init__(self) -> None:
        self._intrinsics_cache: dict[str, Mapping[str, Any]] = {}
        self._hash_cache: dict[str, tuple[int, int, str]] = {}

    def _verify_file(self, path: str | Path, expected: Any, *, label: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise FileNotFoundError(f"missing or empty {label}: {resolved}")
        declared = str(expected).lower()
        if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
            raise ValueError(f"invalid declared SHA-256 for {label}")
        stat = resolved.stat()
        cache_key = str(resolved)
        cached = self._hash_cache.get(cache_key)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            observed = cached[2]
        else:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed = digest.hexdigest()
            self._hash_cache[cache_key] = (stat.st_size, stat.st_mtime_ns, observed)
        if observed != declared:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {declared}, observed {observed}"
            )
        return resolved

    @staticmethod
    def _probability(path: str | Path) -> np.ndarray:
        loaded = np.load(Path(path), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if loaded.files != ["probability"]:
                    raise ValueError(f"unexpected probability archive keys: {loaded.files}")
                value = np.asarray(loaded["probability"], dtype=np.float32)
            finally:
                loaded.close()
        else:
            # The retained reference test run predates compact NPZ export and
            # stores the same audited float32 probability arrays as .npy.
            value = np.asarray(loaded, dtype=np.float32)
        if value.ndim != 2 or not np.all(np.isfinite(value)):
            raise ValueError("predicted probability must be finite and 2D")
        if value.min(initial=0.0) < 0.0 or value.max(initial=0.0) > 1.0:
            raise ValueError("predicted probability must be in [0, 1]")
        return value

    def _intrinsics(
        self, deployment: Mapping[str, Any], depth_mm: np.ndarray
    ) -> Mapping[str, Any] | None:
        explicit = deployment.get("intrinsics_path")
        if explicit:
            path = self._verify_file(
                str(explicit), deployment.get("intrinsics_sha256"), label="intrinsics"
            )
            return json.loads(path.read_text(encoding="utf-8"))
        provenance = json.loads(str(deployment["intrinsics_provenance"]))
        if provenance.get("kind") != "derived_from_organized_pcd":
            return None
        pcd_path = str(Path(str(provenance["source_pcd_path"])).resolve())
        if pcd_path not in self._intrinsics_cache:
            self._verify_file(
                pcd_path,
                provenance.get("source_pcd_sha256"),
                label="organized PCD intrinsics provenance",
            )
            # This is the repository's already audited, read-only derivation from
            # the organized point cloud; it does not use GT grasp or mask data.
            from tools.export_anygrasp_inputs import derive_intrinsics_from_pcd

            self._intrinsics_cache[pcd_path] = derive_intrinsics_from_pcd(
                Path(pcd_path), depth_mm
            )
        return self._intrinsics_cache[pcd_path]

    def load(
        self,
        deployment: Mapping[str, Any],
        *,
        mask_source: MaskSource = "predicted",
        labels: Mapping[str, Any] | None = None,
        load_intrinsics: bool = False,
    ) -> SampleArrays:
        """Load one sample; GT mask requires both an oracle name and labels."""

        rgb_path = self._verify_file(
            deployment["source_rgb_path"],
            deployment.get("source_rgb_sha256"),
            label="source RGB",
        )
        depth_path = self._verify_file(
            deployment["source_depth_path"],
            deployment.get("source_depth_sha256"),
            label="source depth",
        )
        language_digest = hashlib.sha256(
            str(deployment["language"]).encode("utf-8")
        ).hexdigest()
        if language_digest != str(deployment.get("language_sha256")):
            raise ValueError("language SHA-256 mismatch")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth_mm = np.asarray(Image.open(depth_path))
        if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            raise ValueError(f"depth must be uint16 millimetres, got {depth_mm.dtype}")
        if rgb.shape[:2] != depth_mm.shape:
            raise ValueError("RGB/depth shape mismatch")
        if mask_source == "predicted":
            probability_path = self._verify_file(
                deployment["predicted_probability_path"],
                deployment.get("predicted_probability_sha256"),
                label="predicted probability",
            )
            mask_path = self._verify_file(
                deployment["predicted_mask_path"],
                deployment.get("predicted_mask_sha256"),
                label="predicted mask",
            )
            probability = self._probability(probability_path)
            mask = np.asarray(Image.open(mask_path)) > 0
        elif mask_source == "gt_mask_oracle":
            if labels is None:
                raise PermissionError("GT-mask oracle requires the separate label record")
            if str(labels.get("sample_id")) != str(deployment["sample_id"]):
                raise ValueError("deployment/label sample ID mismatch")
            gt_mask_path = self._verify_file(
                labels["prepared_gt_mask_path"],
                labels.get("prepared_gt_mask_sha256"),
                label="prepared GT oracle mask",
            )
            mask_352 = np.asarray(Image.open(gt_mask_path)) > 0
            mask = np.asarray(
                Image.fromarray(mask_352).resize(
                    (rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.NEAREST
                )
            ).astype(bool)
            # Oracle conditioning is explicitly binary; no predicted soft mask
            # information may leak into its output gate.
            probability = mask.astype(np.float32)
        else:
            raise ValueError(f"unsupported mask_source: {mask_source}")
        if mask.shape != depth_mm.shape:
            raise ValueError("native mask/depth shape mismatch")
        intrinsics = self._intrinsics(deployment, depth_mm) if load_intrinsics else None
        return SampleArrays(
            sample_id=str(deployment["sample_id"]),
            scene_id=str(deployment["scene_id"]),
            language=str(deployment["language"]),
            rgb=rgb,
            depth_mm=depth_mm,
            depth_m=depth_mm.astype(np.float32) / np.float32(1000.0),
            binary_mask=mask,
            probability=probability,
            intrinsics=intrinsics,
            mask_source=mask_source,
        )
