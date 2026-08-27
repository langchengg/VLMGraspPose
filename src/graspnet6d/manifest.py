"""Deterministic, scene-disjoint target/query manifest construction.

This module consumes only real GraspNet files.  Tiny arrays created by its unit
tests are explicitly test fixtures and are never accepted as experiment data.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .geometry import CameraIntrinsics, backproject_pixels, compose_transforms, transform_points
from .io import atomic_json, atomic_jsonl, atomic_text, sha256_file
from .language import VisibleObject, generate_unique_queries_for_target
from .splits import SceneSplit, uniform_frame_indices


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    frames_per_scene: int = 16
    max_targets_per_frame: int = 3
    min_mask_pixels: int = 500
    min_valid_depth_fraction: float = 0.70
    seed: int = 20260815
    max_groups: int | None = None
    frame_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.frames_per_scene <= 0 or self.frames_per_scene > 256:
            raise ValueError("frames_per_scene must be in [1, 256]")
        if self.max_targets_per_frame <= 0:
            raise ValueError("max_targets_per_frame must be positive")
        if self.min_mask_pixels <= 0:
            raise ValueError("min_mask_pixels must be positive")
        if not 0.0 <= self.min_valid_depth_fraction <= 1.0:
            raise ValueError("min_valid_depth_fraction must be in [0, 1]")
        if self.max_groups is not None and self.max_groups <= 0:
            raise ValueError("max_groups must be positive when provided")
        if self.frame_ids is not None:
            if (
                len(self.frame_ids) != self.frames_per_scene
                or len(set(self.frame_ids)) != len(self.frame_ids)
                or tuple(sorted(self.frame_ids)) != self.frame_ids
                or any(value < 0 or value >= 256 for value in self.frame_ids)
            ):
                raise ValueError(
                    "frame_ids must be sorted, unique, in [0,255], and match "
                    "frames_per_scene"
                )


@dataclass(frozen=True, slots=True)
class TargetGroup:
    group_id: str
    split: str
    scene_id: str
    camera: str
    frame_id: int
    target_object_id: int
    target_instance_label: int
    rgb_path: str
    depth_path: str
    instance_label_path: str
    meta_path: str
    intrinsics_path: str
    camera_pose_path: str
    table_transform_path: str
    mask_area: int
    valid_depth_fraction: float
    selection_reason: str
    target_center_camera_m: tuple[float, float, float]
    target_center_table_m: tuple[float, float, float]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExcludedTarget:
    scene_id: str
    camera: str
    frame_id: int
    instance_label: int
    object_id: int | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def load_object_catalog(path: Path | str) -> dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    objects = {int(key): str(value) for key, value in payload["objects"].items()}
    if set(objects) != set(range(88)):
        raise ValueError("GraspNet object catalog must contain exactly IDs 0..87")
    return objects


def _scalar(value: np.ndarray, name: str) -> float:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size != 1 or not np.isfinite(flat[0]) or flat[0] <= 0:
        raise ValueError(f"{name} must be one finite positive scalar")
    return float(flat[0])


def _frame_paths(root: Path, scene_id: str, camera: str, frame_id: int) -> dict[str, Path]:
    base = root / "scenes" / scene_id / camera
    stem = f"{frame_id:04d}"
    return {
        "rgb": base / "rgb" / f"{stem}.png",
        "depth": base / "depth" / f"{stem}.png",
        "label": base / "label" / f"{stem}.png",
        "meta": base / "meta" / f"{stem}.mat",
        "intrinsics": base / "camK.npy",
        "camera_poses": base / "camera_poses.npy",
        "table": base / "cam0_wrt_table.npy",
    }


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _stable_target_key(seed: int, scene: str, frame: int, object_id: int) -> str:
    return hashlib.sha256(f"{seed}\0{scene}\0{frame}\0{object_id}".encode()).hexdigest()


def _visible_frame_objects(
    label: np.ndarray,
    depth: np.ndarray,
    *,
    factor_depth: float,
    intrinsics: CameraIntrinsics,
    T_table_camera: np.ndarray,
    catalog: Mapping[int, str],
) -> dict[int, VisibleObject]:
    visible: dict[int, VisibleObject] = {}
    for instance_label in sorted(int(value) for value in np.unique(label) if int(value) > 0):
        object_id = instance_label - 1
        if object_id not in catalog:
            raise ValueError(f"instance label {instance_label} maps outside official object catalog")
        mask = (label == instance_label) & (depth > 0)
        pixels_vu = np.argwhere(mask)
        if not len(pixels_vu):
            continue
        depths = depth[mask].astype(np.float64) / factor_depth
        pixels_uv = pixels_vu[:, ::-1].astype(np.float64)
        points_camera = backproject_pixels(pixels_uv, depths, intrinsics)
        center_camera = np.mean(points_camera, axis=0)
        center_table = transform_points(T_table_camera, center_camera)
        visible[instance_label] = VisibleObject(
            object_id=object_id,
            catalog_name=catalog[object_id],
            center_camera_m=center_camera,
            center_table_m=center_table,
        )
    return visible


def inspect_frame_targets(
    dataset_root: Path | str,
    *,
    scene_id: str,
    camera: str,
    frame_id: int,
    split: str,
    catalog: Mapping[int, str],
    config: SelectionConfig,
) -> tuple[list[TargetGroup], list[ExcludedTarget], dict[int, VisibleObject]]:
    root = Path(dataset_root).expanduser().resolve()
    paths = _frame_paths(root, scene_id, camera, frame_id)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frame is structurally incomplete: {missing}")
    label = np.asarray(Image.open(paths["label"]))
    depth = np.asarray(Image.open(paths["depth"]))
    if label.ndim != 2 or depth.shape != label.shape:
        raise ValueError(f"depth/label shape mismatch: {depth.shape} vs {label.shape}")
    meta = loadmat(paths["meta"])
    required_meta = {"cls_indexes", "poses", "intrinsic_matrix", "factor_depth"}
    missing_meta = sorted(required_meta - set(meta))
    if missing_meta:
        raise ValueError(f"metadata is missing fields: {missing_meta}")
    instance_ids = {int(value) for value in np.asarray(meta["cls_indexes"]).reshape(-1)}
    poses = np.asarray(meta["poses"])
    if poses.ndim != 3 or poses.shape[:2] != (3, 4) or poses.shape[2] != len(instance_ids):
        raise ValueError("metadata poses must have shape (3,4,N) matching cls_indexes")
    factor_depth = _scalar(meta["factor_depth"], "factor_depth")
    matrix = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
    stored = np.asarray(np.load(paths["intrinsics"], allow_pickle=False), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(matrix, stored, atol=1e-6, rtol=0):
        raise ValueError("meta intrinsic_matrix disagrees with camK.npy")
    intrinsics = CameraIntrinsics(
        fx=float(matrix[0, 0]), fy=float(matrix[1, 1]), cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
        width=int(label.shape[1]), height=int(label.shape[0]),
    )
    camera_poses = np.load(paths["camera_poses"], allow_pickle=False)
    if camera_poses.shape[1:] != (4, 4) or frame_id >= len(camera_poses):
        raise ValueError("camera_poses.npy does not contain the requested frame")
    T_table_camera = compose_transforms(
        np.load(paths["table"], allow_pickle=False), camera_poses[frame_id]
    )
    visible = _visible_frame_objects(
        label, depth, factor_depth=factor_depth, intrinsics=intrinsics,
        T_table_camera=T_table_camera, catalog=catalog,
    )
    accepted: list[TargetGroup] = []
    excluded: list[ExcludedTarget] = []
    for instance_label in sorted(int(value) for value in np.unique(label) if int(value) > 0):
        object_id = instance_label - 1
        reasons: list[str] = []
        mask = label == instance_label
        area = int(mask.sum())
        valid_fraction = float(np.mean(depth[mask] > 0)) if area else 0.0
        if instance_label not in instance_ids:
            reasons.append("instance_missing_from_meta_cls_indexes")
        if area < config.min_mask_pixels:
            reasons.append("mask_below_min_pixels")
        if valid_fraction < config.min_valid_depth_fraction:
            reasons.append("valid_depth_fraction_below_threshold")
        if _touches_border(mask):
            reasons.append("mask_touches_image_border")
        if instance_label not in visible:
            reasons.append("target_point_cloud_empty")
        if reasons:
            excluded.append(ExcludedTarget(scene_id, camera, frame_id, instance_label, object_id, ";".join(reasons)))
            continue
        item = visible[instance_label]
        group_id = f"{scene_id}_{camera}_{frame_id:04d}_obj_{object_id:03d}"
        accepted.append(
            TargetGroup(
                group_id=group_id, split=split, scene_id=scene_id, camera=camera,
                frame_id=frame_id, target_object_id=object_id,
                target_instance_label=instance_label, rgb_path=str(paths["rgb"]),
                depth_path=str(paths["depth"]), instance_label_path=str(paths["label"]),
                meta_path=str(paths["meta"]), intrinsics_path=str(paths["intrinsics"]),
                camera_pose_path=str(paths["camera_poses"]), table_transform_path=str(paths["table"]),
                mask_area=area, valid_depth_fraction=valid_fraction,
                selection_reason="passed_preregistered_visibility_checks_then_seeded_target_cap",
                target_center_camera_m=tuple(item.center_camera_m),
                target_center_table_m=tuple(item.center_table_m),
            )
        )
    accepted.sort(key=lambda row: _stable_target_key(config.seed, scene_id, frame_id, row.target_object_id))
    for omitted in accepted[config.max_targets_per_frame :]:
        excluded.append(
            ExcludedTarget(scene_id, camera, frame_id, omitted.target_instance_label, omitted.target_object_id, "deterministic_max_targets_per_frame_cap")
        )
    return accepted[: config.max_targets_per_frame], excluded, visible


def build_target_and_language_manifests(
    dataset_root: Path | str,
    split: SceneSplit,
    *,
    camera: str,
    catalog_path: Path | str,
    output_root: Path | str,
    config: SelectionConfig = SelectionConfig(),
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    catalog = load_object_catalog(catalog_path)
    targets: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    frame_ids = (
        uniform_frame_indices(
            total_frames=256, frames_per_scene=config.frames_per_scene
        )
        if config.frame_ids is None
        else config.frame_ids
    )
    for split_name in ("train", "validation", "test"):
        for scene_id in getattr(split, split_name):
            for frame_id in frame_ids:
                accepted, rejected, visible_by_label = inspect_frame_targets(
                    root, scene_id=str(scene_id), camera=camera, frame_id=frame_id,
                    split=split_name, catalog=catalog, config=config,
                )
                exclusions.extend(item.to_record() for item in rejected)
                visible = tuple(visible_by_label.values())
                for target in accepted:
                    generated = generate_unique_queries_for_target(visible, target.target_object_id)
                    if not generated:
                        exclusions.append(
                            ExcludedTarget(str(scene_id), camera, frame_id, target.target_instance_label, target.target_object_id, "no_unique_derived_language_query").to_record()
                        )
                        continue
                    chosen = sorted(
                        generated,
                        key=lambda query: hashlib.sha256(
                            f"{config.seed}\0{target.group_id}\0{query.query}".encode()
                        ).hexdigest(),
                    )[0]
                    targets.append(target.to_record())
                    queries.append(
                        {
                            "group_id": target.group_id,
                            "query": chosen.query,
                            "template_family": chosen.template_family,
                            "attributes": dict(chosen.attributes),
                            "resolver_result": list(chosen.resolver_result),
                            "is_unique": chosen.is_unique,
                            "provenance": "derived",
                        }
                    )
    if config.max_groups is not None and len(targets) > config.max_groups:
        areas = np.asarray([row["mask_area"] for row in targets], dtype=np.float64)
        depth_fractions = np.asarray(
            [row["valid_depth_fraction"] for row in targets], dtype=np.float64
        )
        area_edges = np.unique(np.quantile(areas, (1 / 3, 2 / 3)))
        depth_edges = np.unique(np.quantile(depth_fractions, (1 / 3, 2 / 3)))
        strata: dict[tuple[str, int, int, int], list[int]] = {}
        for index, row in enumerate(targets):
            key = (
                str(row["scene_id"]),
                int(row["target_object_id"]),
                int(np.searchsorted(area_edges, float(row["mask_area"]), side="right")),
                int(
                    np.searchsorted(
                        depth_edges, float(row["valid_depth_fraction"]), side="right"
                    )
                ),
            )
            strata.setdefault(key, []).append(index)
        for key, indices in strata.items():
            indices.sort(
                key=lambda index: hashlib.sha256(
                    f"{config.seed}\0{key}\0{targets[index]['group_id']}".encode()
                ).hexdigest()
            )
        selected: list[int] = []
        ordered_strata = sorted(
            strata,
            key=lambda key: hashlib.sha256(f"{config.seed}\0{key}".encode()).hexdigest(),
        )
        offset = 0
        while len(selected) < config.max_groups:
            progressed = False
            for key in ordered_strata:
                values = strata[key]
                if offset < len(values):
                    selected.append(values[offset])
                    progressed = True
                    if len(selected) == config.max_groups:
                        break
            if not progressed:
                break
            offset += 1
        selected_set = set(selected)
        query_by_group = {row["group_id"]: row for row in queries}
        for index, row in enumerate(targets):
            if index not in selected_set:
                exclusions.append(
                    ExcludedTarget(
                        str(row["scene_id"]),
                        str(row["camera"]),
                        int(row["frame_id"]),
                        int(row["target_instance_label"]),
                        int(row["target_object_id"]),
                        "deterministic_scene_object_visibility_stratified_group_cap",
                    ).to_record()
                )
        targets = [targets[index] for index in selected]
        queries = [query_by_group[row["group_id"]] for row in targets]
    target_path = output / "target_groups.jsonl"
    query_path = output / "language_queries.jsonl"
    excluded_path = output / "excluded_target_groups.jsonl"
    atomic_jsonl(target_path, targets)
    atomic_jsonl(query_path, queries)
    atomic_jsonl(excluded_path, exclusions)
    report = {
        "schema_version": 1,
        "dataset_root": str(root),
        "split_counts": {
            name: sum(row["split"] == name for row in targets)
            for name in ("train", "validation", "test")
        },
        "target_group_count": len(targets),
        "language_query_count": len(queries),
        "excluded_count": len(exclusions),
        "frame_ids_per_scene": list(frame_ids),
        "target_manifest_sha256": sha256_file(target_path),
        "language_manifest_sha256": sha256_file(query_path),
        "catalog_sha256": sha256_file(Path(catalog_path)),
    }
    atomic_json(output / "manifest_report.json", report)
    return report


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing {description}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{description} line {line_number} is not a JSON object"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"{description} is empty: {path}")
    return rows


def _atomic_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_language_audit(
    target_manifest_path: Path | str,
    language_manifest_path: Path | str,
    output_root: Path | str,
    *,
    seed: int = 20260815,
    sample_count: int = 100,
    required_minimum: int = 100,
) -> dict[str, Any]:
    """Render deterministic real RGB/query/GT-target audit panels and HTML.

    This is an audit of the explicitly derived language layer, not a grounding
    result.  Formal profiles fail closed when fewer than ``required_minimum``
    real, uniquely resolved groups are available; smoke fixtures may request a
    smaller minimum explicitly.
    """

    target_path = Path(target_manifest_path).expanduser().resolve()
    language_path = Path(language_manifest_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("language-audit seed must be an integer")
    if sample_count <= 0 or required_minimum <= 0:
        raise ValueError("language-audit sample counts must be positive")
    targets = _read_jsonl(target_path, "target manifest")
    queries = _read_jsonl(language_path, "language manifest")
    target_by_group = {str(row.get("group_id", "")): row for row in targets}
    query_by_group = {str(row.get("group_id", "")): row for row in queries}
    if (
        "" in target_by_group
        or "" in query_by_group
        or len(target_by_group) != len(targets)
        or len(query_by_group) != len(queries)
    ):
        raise ValueError("language-audit manifests require unique non-empty group IDs")
    if set(target_by_group) != set(query_by_group):
        raise ValueError("target and language manifests have different group universes")
    for group_id, query in query_by_group.items():
        target = target_by_group[group_id]
        if (
            query.get("provenance") != "derived"
            or query.get("is_unique") is not True
            or list(query.get("resolver_result", []))
            != [int(target["target_object_id"])]
        ):
            raise ValueError(
                f"language query is not a uniquely resolved derived annotation: {group_id}"
            )
    if len(targets) < required_minimum:
        raise ValueError(
            f"language audit requires at least {required_minimum} real groups; "
            f"found {len(targets)}"
        )
    selected_count = min(int(sample_count), len(targets))
    ordered_groups = sorted(
        target_by_group,
        key=lambda group_id: (
            hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).hexdigest(),
            group_id,
        ),
    )[:selected_count]
    figure_root = output / "figures" / "language_audit"
    cards: list[str] = []
    records: list[dict[str, Any]] = []
    for index, group_id in enumerate(ordered_groups, start=1):
        target = target_by_group[group_id]
        query = query_by_group[group_id]
        rgb_path = Path(str(target["rgb_path"])).expanduser().resolve()
        label_path = Path(str(target["instance_label_path"])).expanduser().resolve()
        if not rgb_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"language audit source files missing for {group_id}: "
                f"rgb={rgb_path}, label={label_path}"
            )
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(label_path) as image:
            label = np.asarray(image)
        if label.ndim != 2 or label.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/label shape mismatch for {group_id}")
        mask = label == int(target["target_instance_label"])
        if not bool(mask.any()):
            raise ValueError(f"audited target mask is empty for {group_id}")
        highlighted = rgb.copy()
        tint = np.array([235.0, 55.0, 75.0], dtype=np.float64)
        highlighted[mask] = np.rint(
            0.55 * highlighted[mask].astype(np.float64) + 0.45 * tint
        ).astype(np.uint8)
        interior = mask.copy()
        interior[1:-1, 1:-1] &= (
            mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
        highlighted[mask & ~interior] = np.array([255, 230, 40], dtype=np.uint8)
        safe_group = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in group_id
        )
        image_path = figure_root / f"{index:03d}_{safe_group}.png"
        _atomic_png(image_path, highlighted)
        relative_image = image_path.relative_to(output).as_posix()
        records.append(
            {
                "group_id": group_id,
                "query": str(query["query"]),
                "target_object_id": int(target["target_object_id"]),
                "target_instance_label": int(target["target_instance_label"]),
                "template_family": str(query["template_family"]),
                "rgb_sha256": sha256_file(rgb_path),
                "instance_label_sha256": sha256_file(label_path),
                "audit_image": relative_image,
                "audit_image_sha256": sha256_file(image_path),
            }
        )
        cards.append(
            "<article class='card'>"
            f"<img src='{html.escape(relative_image)}' "
            f"alt='{html.escape(group_id)} highlighted target'>"
            f"<h2>{html.escape(str(query['query']))}</h2>"
            f"<p><code>{html.escape(group_id)}</code><br>"
            f"target object {int(target['target_object_id'])}; "
            f"template {html.escape(str(query['template_family']))}</p>"
            "</article>"
        )
    audit_manifest = output / "language_audit_manifest.json"
    atomic_json(
        audit_manifest,
        {
            "schema_version": "graspnet6d_language_audit_v1",
            "scope": "real_manifest_inputs",
            "seed": seed,
            "target_manifest_sha256": sha256_file(target_path),
            "language_manifest_sha256": sha256_file(language_path),
            "available_group_count": len(targets),
            "audited_group_count": len(records),
            "records": records,
        },
    )
    html_document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Derived GraspNet language audit</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#f6f7f9;color:#17202a}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}"
        ".card{background:white;border:1px solid #ccd2d9;border-radius:8px;padding:12px}"
        "img{width:100%;height:auto}h2{font-size:17px}code{font-size:12px}</style>"
        "</head><body><h1>Derived GraspNet language audit</h1>"
        "<p>Yellow/red overlay is the GraspNet GT instance used only for annotation audit. "
        "Queries are deterministic derived annotations, not official GraspNet language.</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>"
    )
    html_path = atomic_text(output / "language_audit.html", html_document)
    family_counts: dict[str, int] = {}
    for query in queries:
        family = str(query["template_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    report_lines = [
        "# Derived language generation report",
        "",
        "These queries are deterministic project-derived annotations; GraspNet does not "
        "supply referring expressions.",
        "",
        f"- Target/query groups: {len(targets)}",
        f"- Uniquely resolved queries: {len(queries)}",
        f"- Human-audit panels: {len(records)}",
        f"- Seed: {seed}",
        f"- Target manifest SHA-256: `{sha256_file(target_path)}`",
        f"- Language manifest SHA-256: `{sha256_file(language_path)}`",
        f"- Audit manifest SHA-256: `{sha256_file(audit_manifest)}`",
        "",
        "## Template-family counts",
        "",
        *[f"- `{family}`: {count}" for family, count in sorted(family_counts.items())],
    ]
    report_path = atomic_text(
        output / "language_generation_report.md", "\n".join(report_lines) + "\n"
    )
    return {
        "schema_version": "graspnet6d_language_audit_summary_v1",
        "available_group_count": len(targets),
        "audited_group_count": len(records),
        "audit_manifest_path": str(audit_manifest),
        "audit_manifest_sha256": sha256_file(audit_manifest),
        "html_path": str(html_path),
        "html_sha256": sha256_file(html_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


__all__ = [
    "ExcludedTarget",
    "SelectionConfig",
    "TargetGroup",
    "build_language_audit",
    "build_target_and_language_manifests",
    "inspect_frame_targets",
    "load_object_catalog",
]
