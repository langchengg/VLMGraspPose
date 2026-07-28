"""Resumable full-manifest runner for the validated HiFi-CS -> VGN pipeline.

The module deliberately composes :mod:`scripts.run_vgn_on_hifics` instead of
forking its geometry or candidate definitions.  SQLite is the authoritative
scheduler state; per-sample files are atomic, human-inspectable products.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import socket
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.run_vgn_on_hifics import process_sample
from src.grasping.vgn_adapter import VGNAdapterError
from src.grasping.vgn_geometry import GeometryError
from src.grasping.vgn_pipeline import (
    LIMITATIONS,
    SCORE_SOURCE,
    TSDF_MODE,
    ManifestSample,
    PipelineError,
    atomic_write_csv,
    atomic_write_json,
    empty_summary,
    sha256_file,
)

from .experiment_store import ExperimentStore
from .failure_taxonomy import is_retryable, is_terminal


LOGGER = logging.getLogger(__name__)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class FileHashCache:
    """Memoize SHA256 by resolved path and immutable stat identity."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, int, int], str] = {}

    def get(self, path: Path | str) -> str:
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
        value = self._values.get(key)
        if value is None:
            value = sha256_file(resolved)
            self._values[key] = value
        return value


@dataclass(frozen=True)
class SceneCacheEntry:
    key: str
    rgb: np.ndarray
    depth: np.ndarray


class SceneArrayCache:
    """Bounded scene cache for decoded RGB and raw metric-source depth."""

    def __init__(
        self,
        *,
        geometry_config_hash: str,
        maximum_scenes: int = 4,
        hash_cache: FileHashCache | None = None,
    ) -> None:
        if maximum_scenes < 1:
            raise ValueError("maximum_scenes must be positive")
        self.geometry_config_hash = str(geometry_config_hash)
        self.maximum_scenes = int(maximum_scenes)
        self.hash_cache = hash_cache or FileHashCache()
        self._entries: OrderedDict[str, SceneCacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key_for(self, sample: ManifestSample) -> str:
        pcd = sample.metadata.get("source_pcd")
        pcd_hash = sample.metadata.get("source_pcd_sha256")
        if not pcd_hash and pcd and Path(str(pcd)).is_file():
            pcd_hash = self.hash_cache.get(Path(str(pcd)))
        intrinsics_hash = (
            self.hash_cache.get(sample.intrinsics_path)
            if sample.intrinsics_path is not None
            else _canonical_hash(
                {
                    key: sample.metadata.get(key)
                    for key in ("width", "height", "fx", "fy", "cx", "cy")
                }
            )
        )
        return _canonical_hash(
            {
                "scene_id": sample.scene_id,
                "rgb_sha256": sample.metadata.get("source_rgb_sha256")
                or self.hash_cache.get(sample.rgb_path),
                "depth_sha256": sample.metadata.get("source_depth_sha256")
                or self.hash_cache.get(sample.depth_path),
                "pcd_sha256": pcd_hash,
                "intrinsics_sha256": intrinsics_hash,
                "geometry_config_sha256": self.geometry_config_hash,
            }
        )

    def _verify_declared(self, path: Path, expected: Any, label: str) -> None:
        if expected not in (None, ""):
            actual = self.hash_cache.get(path)
            if actual != str(expected):
                raise PipelineError(
                    "source_checksum_mismatch", f"{label} SHA256 mismatch: {path}"
                )

    def arrays_for(self, sample: ManifestSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = self.key_for(sample)
        entry = self._entries.get(key)
        if entry is None:
            self._verify_declared(
                sample.rgb_path, sample.metadata.get("source_rgb_sha256"), "RGB"
            )
            self._verify_declared(
                sample.depth_path, sample.metadata.get("source_depth_sha256"), "depth"
            )
            rgb = np.asarray(Image.open(sample.rgb_path).convert("RGB"), dtype=np.uint8)
            depth = np.asarray(Image.open(sample.depth_path))
            if depth.ndim != 2:
                raise PipelineError("depth_shape_error", f"Depth must be HxW, got {depth.shape}")
            if rgb.shape[:2] != depth.shape:
                raise PipelineError(
                    "rgb_depth_shape_error", f"RGB/depth mismatch {rgb.shape[:2]} != {depth.shape}"
                )
            entry = SceneCacheEntry(key=key, rgb=rgb, depth=depth)
            self._entries[key] = entry
            self.misses += 1
            while len(self._entries) > self.maximum_scenes:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
            self.hits += 1

        self._verify_declared(
            sample.mask_path, sample.metadata.get("prediction_mask_sha256"), "predicted mask"
        )
        mask = np.asarray(Image.open(sample.mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.ndim != 2:
            raise PipelineError("mask_depth_shape_error", f"Mask must be HxW, got {mask.shape}")
        return entry.rgb, entry.depth, mask

    def diagnostics(self) -> dict[str, int]:
        return {
            "scene_cache_hits": self.hits,
            "scene_cache_misses": self.misses,
            "resident_scenes": len(self._entries),
        }


class ContentAddressedSampleCache:
    """Persistent depth/mask/intrinsics/config cache index.

    Each manifest row is still registered and receives its own output.  A hit
    only avoids recomputing an identical sample payload.
    """

    def __init__(self, path: Path | str, *, hash_cache: FileHashCache | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.hash_cache = hash_cache or FileHashCache()
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = dict(payload.get("entries", {}))
        else:
            self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def key_for(
        self,
        sample: ManifestSample,
        *,
        vgn_config_hash: str,
        checkpoint_hash: str,
    ) -> str:
        intrinsics_hash = (
            self.hash_cache.get(sample.intrinsics_path)
            if sample.intrinsics_path is not None
            else _canonical_hash(
                {
                    key: sample.metadata.get(key)
                    for key in ("width", "height", "fx", "fy", "cx", "cy")
                }
            )
        )
        return _canonical_hash(
            {
                "depth_sha256": sample.metadata.get("source_depth_sha256")
                or self.hash_cache.get(sample.depth_path),
                "mask_sha256": self.hash_cache.get(sample.mask_path),
                "intrinsics_sha256": intrinsics_hash,
                "vgn_config_sha256": vgn_config_hash,
                "checkpoint_sha256": checkpoint_hash,
            }
        )

    def lookup(self, key: str) -> dict[str, Any] | None:
        value = self._entries.get(str(key))
        if value is None or not Path(str(value.get("sample_dir", ""))).is_dir():
            self.misses += 1
            return None
        self.hits += 1
        return dict(value)

    def register(self, key: str, *, sample_id: str, sample_dir: Path, result: Mapping[str, Any]) -> None:
        self._entries[str(key)] = {
            "sample_id": str(sample_id),
            "sample_dir": str(sample_dir.resolve()),
            "status": result.get("status"),
        }
        atomic_write_json(self.path, {"version": 1, "entries": self._entries})

    def diagnostics(self) -> dict[str, int]:
        return {
            "sample_cache_hits": self.hits,
            "sample_cache_misses": self.misses,
            "indexed_payloads": len(self._entries),
        }


def resolve_gt_oracle_mapping(sample: ManifestSample) -> Path:
    """Resolve a unique GT mask through the dedicated annotation adapter."""

    from .ocid_annotations import resolve_gt_oracle_mapping as resolve

    return resolve(sample)


def build_gt_oracle_sample(sample: ManifestSample) -> ManifestSample:
    """Return an explicit GT-oracle variant without mutating manifest truth."""

    from .ocid_annotations import build_gt_oracle_sample as build

    return build(sample)


def _copy_cached_artifacts(source: Path, destination: Path, sample: ManifestSample) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "candidates_all.npz",
        "candidates_target.npz",
        "workspace_frame.json",
        "support_plane.json",
        "tsdf_grid.npz",
        "local_scene_point_cloud.ply",
        "target_point_cloud.ply",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    candidates_path = source / "candidates.json"
    if candidates_path.is_file():
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidates.update(
            sample_id=sample.sample_id,
            scene_id=sample.scene_id,
            instruction=sample.instruction,
        )
        atomic_write_json(destination / "candidates.json", candidates)
    top1 = json.loads((source / "top1.json").read_text(encoding="utf-8"))
    top1.update(
        sample_id=sample.sample_id,
        scene_id=sample.scene_id,
        instruction=sample.instruction,
        cache_reused_from=source.name,
    )
    atomic_write_json(destination / "top1.json", top1)
    return top1


def _flatten_top1(sample: ManifestSample, sample_dir: Path) -> dict[str, Any]:
    top1_path = sample_dir / "top1.json"
    if not top1_path.is_file():
        raise PipelineError("write_error", f"processor did not write {top1_path}")
    payload = json.loads(top1_path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate")
    flat = dict(payload)
    flat.update(
        sample_id=sample.sample_id,
        scene_id=sample.scene_id,
        instruction=sample.instruction,
        score_source=SCORE_SOURCE,
        custom_reranking=False,
        tsdf_mode=TSDF_MODE,
    )
    if isinstance(candidate, Mapping):
        for key in (
            "vgn_quality",
            "width_m",
            "position_task_m",
            "position_camera_m",
            "quaternion_task_xyzw",
            "quaternion_camera_xyzw",
            "projected_uv",
            "official_selection_index",
            "score_rank",
        ):
            flat[key] = candidate.get(key)
        camera_transform = candidate.get("T_camera_grasp")
        flat["rotation_camera_3x3"] = (
            [row[:3] for row in camera_transform[:3]]
            if isinstance(camera_transform, list) and len(camera_transform) >= 3
            else None
        )
        flat["inside_predicted_mask"] = candidate.get("inside_raw_target_mask")
        flat["inside_dilated_predicted_mask"] = candidate.get("inside_dilated_target_mask")
        # The scalar is derived from the selected official candidate rather
        # than copied from an independent summary field.
        flat["vgn_quality"] = candidate.get("vgn_quality")
    else:
        for key in (
            "vgn_quality",
            "width_m",
            "position_task_m",
            "position_camera_m",
            "quaternion_task_xyzw",
            "quaternion_camera_xyzw",
            "rotation_camera_3x3",
            "projected_uv",
            "inside_predicted_mask",
        ):
            flat.setdefault(key, None)
    workspace = sample_dir / "workspace_frame.json"
    if workspace.is_file():
        geometry = json.loads(workspace.read_text(encoding="utf-8"))
        flat["intrinsics_source"] = geometry.get("intrinsics", {}).get("source")
        flat["depth_unit_resolved"] = geometry.get("depth", {}).get("resolved_unit")
    else:
        flat.setdefault("intrinsics_source", None)
        flat.setdefault("depth_unit_resolved", None)
    atomic_write_json(top1_path, flat)
    return flat


def _result_row(
    sample: ManifestSample,
    summary: Mapping[str, Any],
    top1: Mapping[str, Any],
    *,
    processing_time_total: float,
    processing_time_render: float,
    sample_cache_key: str,
    sample_cache_hit: bool,
) -> dict[str, Any]:
    result = {
        **dict(summary),
        "sample_id": sample.sample_id,
        "dataset_index": sample.dataset_index,
        "scene_id": sample.scene_id,
        "instruction": sample.instruction,
        "view": sample.view,
        "question_index": sample.row.get("question_index"),
        "fit_rmse_px": sample.row.get("fit_rmse_px"),
        "fit_p95_px": sample.row.get("fit_p95_px"),
        "mask_source": sample.metadata.get("mask_source"),
        "official_candidate_count": top1.get(
            "candidate_count_before_target_filter", summary.get("official_candidate_count", 0)
        ),
        "target_candidate_count": top1.get(
            "candidate_count_after_target_filter", summary.get("target_candidate_count", 0)
        ),
        "top1_vgn_quality": top1.get("vgn_quality"),
        "top1_width_m": top1.get("width_m"),
        "selection_policy": top1.get("selection_policy", "highest_vgn_quality"),
        "score_source": SCORE_SOURCE,
        "custom_reranking": False,
        "tsdf_mode": TSDF_MODE,
        "processing_time_render": float(processing_time_render),
        "processing_time_total": float(processing_time_total),
        "sample_cache_key": sample_cache_key,
        "sample_cache_hit": bool(sample_cache_hit),
        "limitations": LIMITATIONS,
    }
    return result


Processor = Callable[..., Mapping[str, Any]]


class FullVGNRunner:
    """Execute selected rows while preserving a full immutable manifest store."""

    def __init__(
        self,
        *,
        samples: Sequence[ManifestSample],
        store: ExperimentStore,
        args: Any,
        net: Any,
        device: str,
        run_metadata: Mapping[str, Any],
        processor: Processor | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.samples = list(samples)
        self.by_id = {sample.sample_id: sample for sample in self.samples}
        if len(self.by_id) != len(self.samples):
            raise ValueError("samples contain duplicate sample_id values")
        self.store = store
        self.args = args
        self.net = net
        self.device = str(device)
        self.processor = processor or process_sample
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.run_metadata = dict(run_metadata)
        geometry_hash = _canonical_hash(
            {
                key: getattr(args, key, None)
                for key in (
                    "depth_unit",
                    "depth_scale",
                    "depth_min_m",
                    "depth_max_m",
                    "workspace_size_m",
                    "resolution",
                    "table_height_m",
                    "allow_camera_aligned_fallback",
                )
            }
        )
        self.hash_cache = FileHashCache()
        self.scene_cache = SceneArrayCache(
            geometry_config_hash=geometry_hash,
            maximum_scenes=int(getattr(args, "scene_cache_size", 4)),
            hash_cache=self.hash_cache,
        )
        self.vgn_config_hash = _canonical_hash(
            {
                "geometry": geometry_hash,
                "selection_policy": getattr(args, "selection_policy", None),
                "target_mask_dilation_px": getattr(args, "target_mask_dilation_px", None),
                "mask_cleanup": getattr(args, "mask_cleanup", None),
                "score_source": SCORE_SOURCE,
            }
        )
        self.sample_cache = ContentAddressedSampleCache(
            Path(args.output) / "sample_cache_index.json", hash_cache=self.hash_cache
        )
        self.expression_index: Mapping[int, Mapping[str, Any]] | None = None
        expression_path = next(
            (
                sample.metadata.get("source_expression_file")
                for sample in self.samples
                if sample.metadata.get("source_expression_file") not in (None, "")
            ),
            None,
        )
        if expression_path is not None:
            from .ocid_annotations import load_expression_index

            self.expression_index = load_expression_index(Path(str(expression_path)))

    def _call_processor(
        self, sample: ManifestSample, arrays: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> dict[str, Any]:
        try:
            result = self.processor(
                sample,
                args=self.args,
                net=self.net,
                device=self.device,
                arrays=arrays,
            )
        except TypeError as error:
            # Synthetic processors used by tests may intentionally expose the
            # original four-argument surface.
            if "arrays" not in str(error):
                raise
            result = self.processor(
                sample, args=self.args, net=self.net, device=self.device
            )
        return dict(result)

    def _failure(
        self, sample: ManifestSample, status: str, reason: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        sample_dir = Path(self.args.output) / "samples" / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        top1 = {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "instruction": sample.instruction,
            "status": status,
            "failure_reason": reason,
            "selection_policy": getattr(
                self.args, "selection_policy", "highest_vgn_quality"
            ),
            "score_source": SCORE_SOURCE,
            "custom_reranking": False,
            "tsdf_mode": TSDF_MODE,
            "candidate_count_before_target_filter": 0,
            "candidate_count_after_target_filter": 0,
            "candidate": None,
            "limitations": LIMITATIONS,
        }
        atomic_write_json(sample_dir / "top1.json", top1)
        return empty_summary(sample, status=status, reason=reason), top1

    def _render(self, sample: ManifestSample, sample_dir: Path) -> float:
        if not bool(getattr(self.args, "render_all_2d", False)):
            return 0.0
        start = time.perf_counter()
        from .render_gallery import render_sample_webp

        gt_path: Path | None = None
        try:
            gt_path = resolve_gt_oracle_mapping(sample)
        except Exception:
            gt_path = None
        render_sample_webp(
            sample_dir,
            sample,
            top_k=int(getattr(self.args, "top_k", 50)),
            gt_mask_path=gt_path,
        )
        return time.perf_counter() - start

    def _enrich_gt(self, sample: ManifestSample, top1: dict[str, Any]) -> dict[str, Any]:
        if str(getattr(self.args, "mask_source", "predicted")) != "predicted":
            return {}
        result: dict[str, Any] = {}
        try:
            from .ocid_annotations import (
                annotate_expression,
                annotate_top1_with_gt,
                evaluate_predicted_mask,
                expression_for_sample,
            )

            result.update(evaluate_predicted_mask(sample))
            if self.expression_index is not None:
                expression = expression_for_sample(sample, self.expression_index)
                result.update(annotate_expression(expression))
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            result.update(gt_oracle_available=False, gt_oracle_reason=str(error))
        if isinstance(top1.get("candidate"), Mapping):
            try:
                from .ocid_annotations import annotate_top1_with_gt

                diagnostics = annotate_top1_with_gt(sample, top1)
                result.update(diagnostics)
                top1.update(diagnostics)
                top1["inside_gt_mask"] = diagnostics["top1_inside_gt_target_mask"]
                atomic_write_json(
                    Path(self.args.output) / "samples" / sample.sample_id / "top1.json",
                    top1,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                result["target_consistency_unavailable_reason"] = str(error)
                top1.setdefault("inside_gt_mask", None)
        else:
            top1.setdefault("inside_gt_mask", None)
        atomic_write_json(
            Path(self.args.output) / "samples" / sample.sample_id / "top1.json", top1
        )
        return result

    def _process_one(self, sample: ManifestSample) -> dict[str, Any]:
        sample_dir = Path(self.args.output) / "samples" / sample.sample_id
        started = time.perf_counter()
        checkpoint_hash = str(self.run_metadata["checkpoint_sha256"])
        content_key = self.sample_cache.key_for(
            sample,
            vgn_config_hash=self.vgn_config_hash,
            checkpoint_hash=checkpoint_hash,
        )
        cache_entry = self.sample_cache.lookup(content_key)
        cache_hit = cache_entry is not None
        if cache_entry is not None:
            top1 = _copy_cached_artifacts(
                Path(str(cache_entry["sample_dir"])), sample_dir, sample
            )
            top1 = _flatten_top1(sample, sample_dir)
            summary = empty_summary(
                sample,
                status=str(top1["status"]),
                reason=str(top1.get("failure_reason", "")),
            )
            summary.update(
                official_candidate_count=top1.get("candidate_count_before_target_filter", 0),
                target_candidate_count=top1.get("candidate_count_after_target_filter", 0),
                top1_vgn_quality=top1.get("vgn_quality"),
                top1_width_m=top1.get("width_m"),
            )
        else:
            arrays = self.scene_cache.arrays_for(sample)
            summary = self._call_processor(sample, arrays)
            top1 = _flatten_top1(sample, sample_dir)
        gt = self._enrich_gt(sample, top1)
        try:
            render_time = self._render(sample, sample_dir)
        except Exception as error:
            raise PipelineError("render_error", f"{type(error).__name__}: {error}") from error
        total = time.perf_counter() - started
        result = _result_row(
            sample,
            summary,
            top1,
            processing_time_total=total,
            processing_time_render=render_time,
            sample_cache_key=content_key,
            sample_cache_hit=cache_hit,
        )
        result.update(gt)
        atomic_write_json(sample_dir / "result.json", result)
        if cache_entry is None and is_terminal(str(result["status"])):
            self.sample_cache.register(
                content_key,
                sample_id=sample.sample_id,
                sample_dir=sample_dir,
                result=result,
            )
        return result

    def run_pending(self, selected: Iterable[ManifestSample] | None = None) -> dict[str, Any]:
        chosen = list(self.samples if selected is None else selected)
        processed = 0
        skipped = 0
        for original_sample in chosen:
            claim = self.store.claim_sample(
                original_sample.sample_id,
                self.worker_id,
                lease_seconds=float(getattr(self.args, "lease_seconds", 3600.0)),
            )
            if claim is None:
                skipped += 1
                continue
            sample = original_sample
            if str(getattr(self.args, "mask_source", "predicted")) == "gt-oracle":
                try:
                    sample = build_gt_oracle_sample(original_sample)
                except Exception as error:
                    status = getattr(error, "status", "oracle_mask_forbidden")
                    summary, top1 = self._failure(original_sample, status, str(error))
                    result = _result_row(
                        original_sample,
                        summary,
                        top1,
                        processing_time_total=0.0,
                        processing_time_render=0.0,
                        sample_cache_key="",
                        sample_cache_hit=False,
                    )
                    atomic_write_json(
                        Path(self.args.output)
                        / "samples"
                        / original_sample.sample_id
                        / "result.json",
                        result,
                    )
                    self.store.complete_sample(
                        original_sample.sample_id,
                        status,
                        result=result,
                        failure_reason=str(error),
                        worker_id=self.worker_id,
                    )
                    processed += 1
                    continue
            try:
                result = self._process_one(sample)
                status = str(result["status"])
                reason = str(result.get("failure_reason", ""))
            except (GeometryError, PipelineError, VGNAdapterError) as error:
                status = str(getattr(error, "status", "vgn_inference_error"))
                reason = str(error)
                summary, top1 = self._failure(original_sample, status, reason)
                result = _result_row(
                    original_sample,
                    summary,
                    top1,
                    processing_time_total=0.0,
                    processing_time_render=0.0,
                    sample_cache_key="",
                    sample_cache_hit=False,
                )
                atomic_write_json(
                    Path(self.args.output) / "samples" / original_sample.sample_id / "result.json",
                    result,
                )
            except Exception as error:  # one corrupt sample never aborts the manifest
                LOGGER.exception("[%s] unexpected full-run failure", original_sample.sample_id)
                status = "processing_error"
                reason = f"{type(error).__name__}: {error}"
                summary, top1 = self._failure(original_sample, status, reason)
                result = _result_row(
                    original_sample,
                    summary,
                    top1,
                    processing_time_total=0.0,
                    processing_time_render=0.0,
                    sample_cache_key="",
                    sample_cache_hit=False,
                )
                atomic_write_json(
                    Path(self.args.output) / "samples" / original_sample.sample_id / "result.json",
                    result,
                )

            if is_retryable(status):
                self.store.fail_sample(
                    original_sample.sample_id,
                    status,
                    result=result,
                    failure_reason=reason,
                    worker_id=self.worker_id,
                )
            else:
                self.store.complete_sample(
                    original_sample.sample_id,
                    status,
                    result=result,
                    failure_reason=reason,
                    worker_id=self.worker_id,
                )
            processed += 1

        rows = self.store.sample_rows()
        summary_fields = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in {"manifest_row", "result", "manifest_row_json", "result_json"}
                and not isinstance(row.get(key), (dict, list, tuple))
            }
        )
        if rows:
            atomic_write_csv(Path(self.args.output) / "summary.csv", rows, summary_fields)
            failures = [
                row
                for row in rows
                if row.get("status") not in {"ok", "pending", "running"}
            ]
            atomic_write_csv(Path(self.args.output) / "failures.csv", failures, summary_fields)
        terminal_count = sum(is_terminal(str(row["status"])) for row in rows)
        return {
            "selected_count": len(chosen),
            "processed_count": processed,
            "resume_skipped_count": skipped,
            "registered_count": len(rows),
            "terminal_count": terminal_count,
            "pending_count": sum(row["state"] == "pending" for row in rows),
            "status_counts": self.store.status_counts(),
            "scene_cache": self.scene_cache.diagnostics(),
            "sample_cache": self.sample_cache.diagnostics(),
        }


def benchmark_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_count: int,
    wall_time_s: float,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Return measured throughput and a linear, explicitly empirical ETA."""

    completed = [row for row in rows if is_terminal(str(row.get("status", "")))]

    def median(key: str) -> float | None:
        values = [
            float(row[key])
            for row in completed
            if row.get(key) not in (None, "") and np.isfinite(float(row[key]))
        ]
        return float(np.median(values)) if values else None

    count = len(completed)
    rate = count / wall_time_s if wall_time_s > 0 else None
    directory = Path(output_dir)
    disk_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    estimated_disk = disk_bytes / count * manifest_count if count else None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    peak_rss_bytes = int(rss * 1024 if platform.system() == "Linux" else rss)
    return {
        "measured_sample_count": count,
        "wall_time_s": float(wall_time_s),
        "samples_per_second": rate,
        "median_preprocessing_time_s": median("processing_time_depth"),
        "median_tsdf_time_s": median("processing_time_tsdf"),
        "median_inference_time_s": median("processing_time_vgn"),
        "median_rendering_time_s": median("processing_time_render"),
        "eta_for_manifest_s": manifest_count / rate if rate else None,
        "measured_disk_bytes": disk_bytes,
        "estimated_manifest_disk_bytes": estimated_disk,
        "peak_rss_bytes": peak_rss_bytes,
        "estimation_method": "linear extrapolation from this measured shard",
    }


def manifest_registration_rows(samples: Sequence[ManifestSample]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "dataset_index": sample.dataset_index,
            "scene_id": sample.scene_id,
            "instruction": sample.instruction,
            "view": sample.view,
            "cluster_id": sample.scene_id,
            "question_index": sample.row.get("question_index"),
        }
        for sample in samples
    ]


__all__ = [
    "ContentAddressedSampleCache",
    "FileHashCache",
    "FullVGNRunner",
    "SceneArrayCache",
    "benchmark_summary",
    "build_gt_oracle_sample",
    "manifest_registration_rows",
    "resolve_gt_oracle_mapping",
]
