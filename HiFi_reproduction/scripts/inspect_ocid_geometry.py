#!/usr/bin/env python3
"""Inspect OCID-VLG calibration, PCD presence, depth units, and resolutions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.vgn_geometry import (  # noqa: E402
    GeometryError,
    load_intrinsics_config,
    resolve_depth_m,
)

DEPTH_FIELDS = ("depth_path", "depth_dataset_rel", "depth_dataset_path", "source_depth")
VIEW_FIELDS = ("view", "camera_view", "camera_view_from_sequence_path", "sensor")
BUNDLE_FIELDS = ("output_dir", "bundle_path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument(
        "--intrinsics", type=Path, help="JSON/YAML direct/default/per-view calibration"
    )
    parser.add_argument(
        "--manifest", type=Path, help="Optional JSON/JSONL/CSV manifest"
    )
    parser.add_argument("--depth-unit", choices=("auto", "m", "mm"), default="auto")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-min-m", type=float, default=0.0)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def _read_manifest(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = loaded
        elif isinstance(loaded, Mapping):
            rows = loaded.get("samples", loaded.get("data", []))
        else:
            rows = []
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        raise ValueError("manifest must be JSON, JSONL, or CSV")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("manifest must contain an array/stream of object rows")
    return rows


def _field(row: Mapping[str, Any], candidates: Iterable[str]) -> tuple[str | None, Any]:
    present = [
        (name, row[name]) for name in candidates if row.get(name) not in (None, "")
    ]
    if len(present) > 1 and len({str(value) for _, value in present}) > 1:
        raise ValueError(f"conflicting manifest fields: {present}")
    return present[0] if present else (None, None)


def _resolve_path(value: Any, root: Path, manifest_parent: Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root_candidate = (root / path).resolve()
    if root_candidate.exists() or manifest_parent is None:
        return root_candidate
    return (manifest_parent / path).resolve()


def _view_from_path(path: Path) -> str:
    for part in path.parts:
        lowered = part.lower()
        if lowered in {"top", "bottom"}:
            return lowered
    return "unknown"


def _corresponding_pcd(depth_path: Path) -> Path:
    parts = list(depth_path.parts)
    indices = [index for index, part in enumerate(parts) if part == "depth"]
    if not indices:
        return depth_path.with_suffix(".pcd")
    parts[indices[-1]] = "pcd"
    return Path(*parts).with_suffix(".pcd")


def _pcd_header(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    header: dict[str, Any] = {"exists": True}
    with path.open("rb") as stream:
        for _ in range(100):
            raw = stream.readline()
            if not raw:
                break
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError:
                break
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            key = key.lower()
            if key in {"width", "height", "points"} and values:
                header[key] = int(values[0])
            elif key in {"fields", "type", "size", "count"}:
                header[key] = values
            elif key == "data":
                header["data"] = values[0] if values else None
                break
    return header


def _depth_paths(
    args: argparse.Namespace,
) -> tuple[list[tuple[Path, str, Mapping[str, Any]]], dict[str, Any]]:
    root = args.ocid_root.expanduser().resolve()
    mapping_report: dict[str, Any] = {"source": "recursive_scan", "detected_fields": {}}
    if args.manifest is None:
        paths = sorted(root.glob("**/depth/*.png"))
        return [(path, _view_from_path(path), {}) for path in paths], mapping_report
    manifest_path = args.manifest.expanduser().resolve()
    rows = _read_manifest(manifest_path)
    resolved: list[tuple[Path, str, Mapping[str, Any]]] = []
    depth_fields: Counter[str] = Counter()
    view_fields: Counter[str] = Counter()
    for index, row in enumerate(rows):
        depth_field, depth_value = _field(row, DEPTH_FIELDS)
        effective_metadata = dict(row)
        if depth_field is None:
            bundle_field, bundle_value = _field(row, BUNDLE_FIELDS)
            if bundle_field is None:
                raise ValueError(
                    f"manifest row {index} has no supported depth or bundle field; "
                    f"keys={sorted(row)}"
                )
            bundle = _resolve_path(bundle_value, root, manifest_path.parent)
            metadata_path = bundle / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"manifest row {index} bundle metadata missing: {metadata_path}"
                )
            bundle_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(bundle_metadata, Mapping) or not bundle_metadata.get(
                "source_depth"
            ):
                raise ValueError(
                    f"manifest row {index} bundle metadata has no explicit source_depth"
                )
            effective_metadata = {**bundle_metadata, **row}
            depth_field = f"{bundle_field}->metadata.source_depth"
            depth_value = bundle_metadata["source_depth"]
            intrinsics_path = bundle / "intrinsics.json"
            if intrinsics_path.is_file():
                effective_metadata["__intrinsics_path"] = str(intrinsics_path.resolve())
        view_field, view_value = _field(effective_metadata, VIEW_FIELDS)
        depth_fields[depth_field] += 1
        if view_field:
            view_fields[view_field] += 1
        path = _resolve_path(depth_value, root, manifest_path.parent)
        view = (
            str(view_value).lower() if view_value is not None else _view_from_path(path)
        )
        resolved.append((path, view, effective_metadata))
    mapping_report = {
        "source": str(manifest_path),
        "detected_fields": {
            "depth": dict(depth_fields),
            "view": dict(view_fields),
        },
    }
    return resolved, mapping_report


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    root = args.ocid_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OCID-VLG root not found: {root}")
    records, mapping_report = _depth_paths(args)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        records = records[: args.max_samples]
    discovered_intrinsics = sorted(
        {
            str(metadata["__intrinsics_path"])
            for _, _, metadata in records
            if metadata.get("__intrinsics_path")
        }
    )
    calibration_report: dict[str, Any]
    if args.intrinsics is None and not discovered_intrinsics:
        calibration_report = {
            "status": "missing_camera_intrinsics",
            "path": None,
            "message": "No --intrinsics JSON/YAML was supplied; VGN must not run without calibration.",
        }
    elif args.intrinsics is not None:
        calibration_report = {
            "status": "provided",
            "path": str(args.intrinsics.expanduser().resolve()),
        }
    else:
        calibration_report = {
            "status": "per_sample_bundle_intrinsics",
            "path": None,
            "file_count": len(discovered_intrinsics),
            "message": "Using each bundle's explicit intrinsics.json; values are not guessed.",
        }

    samples: list[dict[str, Any]] = []
    view_accumulator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for depth_path, view, metadata in records:
        row: dict[str, Any] = {
            "depth_path": str(depth_path),
            "view": view,
            "status": "ok",
        }
        try:
            if not depth_path.is_file():
                raise FileNotFoundError(f"depth file missing: {depth_path}")
            with Image.open(depth_path) as image:
                depth = np.asarray(image)
            if depth.ndim != 2:
                raise ValueError(f"depth is not a 2-D image: {depth.shape}")
            row["raw_depth_shape"] = list(depth.shape)
            row["raw_depth_dtype"] = str(depth.dtype)
            conversion = resolve_depth_m(
                depth,
                unit=args.depth_unit,
                depth_scale=args.depth_scale,
                min_depth_m=args.depth_min_m,
                max_depth_m=args.depth_max_m,
                metadata=metadata,
            )
            row["depth"] = conversion.log_dict()
            pcd_path = (
                Path(str(metadata["source_pcd"])).expanduser().resolve()
                if metadata.get("source_pcd")
                else _corresponding_pcd(depth_path)
            )
            pcd = _pcd_header(pcd_path)
            row["pcd_path"] = str(pcd_path)
            row["pcd"] = pcd
            if pcd.get("exists") and (pcd.get("height"), pcd.get("width")) != tuple(
                depth.shape
            ):
                row["pcd_resolution_matches_depth"] = False
                row["status"] = "pcd_depth_resolution_mismatch"
            else:
                row["pcd_resolution_matches_depth"] = bool(pcd.get("exists"))
            intrinsics_source = args.intrinsics or metadata.get("__intrinsics_path")
            if intrinsics_source is not None:
                intrinsics = load_intrinsics_config(
                    intrinsics_source, view=view, image_shape=depth.shape
                )
                row["intrinsics"] = intrinsics.to_dict()
            view_accumulator[view].append(row)
        except (GeometryError, FileNotFoundError, ValueError) as error:
            row["status"] = getattr(error, "status", "inspection_failed")
            row["error"] = str(error)
            if isinstance(error, GeometryError) and error.details:
                row["error_details"] = error.details
            view_accumulator[view].append(row)
        samples.append(row)

    view_summary: dict[str, Any] = {}
    for view, view_rows in sorted(view_accumulator.items()):
        ok_depth = [item["depth"] for item in view_rows if "depth" in item]
        view_summary[view] = {
            "sample_count": len(view_rows),
            "status_counts": dict(Counter(str(item["status"]) for item in view_rows)),
            "depth_units": dict(
                Counter(
                    str(item["depth"]["depth_unit"])
                    for item in view_rows
                    if "depth" in item
                )
            ),
            "resolutions": dict(
                Counter(
                    "x".join(map(str, item["raw_depth_shape"]))
                    for item in view_rows
                    if "raw_depth_shape" in item
                )
            ),
            "pcd_present_count": sum(
                bool(item.get("pcd", {}).get("exists")) for item in view_rows
            ),
            "metric_p1_m_range": _range(item["metric_p1_m"] for item in ok_depth),
            "metric_p50_m_range": _range(item["metric_p50_m"] for item in ok_depth),
            "metric_p99_m_range": _range(item["metric_p99_m"] for item in ok_depth),
        }
    return {
        "ocid_root": str(root),
        "calibration": calibration_report,
        "manifest_mapping": mapping_report,
        "requested_depth_unit": args.depth_unit,
        "depth_scale": args.depth_scale,
        "inspected_sample_count": len(samples),
        "views": view_summary,
        "samples": samples,
    }


def _range(values: Iterable[float | None]) -> list[float] | None:
    finite = [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]
    return [min(finite), max(finite)] if finite else None


def main() -> int:
    args = _parser().parse_args()
    try:
        report = inspect(args)
    except Exception as error:
        print(
            json.dumps({"status": "inspection_failed", "error": str(error)}, indent=2),
            file=sys.stderr,
        )
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(destination)
    failures = sum(
        count
        for summary in report["views"].values()
        for status, count in summary["status_counts"].items()
        if status != "ok"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
