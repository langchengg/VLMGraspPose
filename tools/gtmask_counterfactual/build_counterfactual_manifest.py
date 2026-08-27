#!/usr/bin/env python3
"""Publish the exact 7,675-row P1 denominator using identity joins only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import transition_pipeline_status  # noqa: E402
from gtmask_counterfactual.contracts import RunState  # noqa: E402
from gtmask_counterfactual.io import (  # noqa: E402
    artifact_record,
    canonical_sha256,
    sha256_file,
)
from gtmask_counterfactual.manifest import (  # noqa: E402
    EXPECTED_SAMPLE_COUNT,
    build_counterfactual_manifest,
)
from gtmask_counterfactual.mapping_pipeline import (  # noqa: E402
    publish_manifest_artifacts,
)
from d1_reranking.provenance import load_source_closure  # noqa: E402


DENOMINATOR_COLUMNS = (
    "sample_id",
    "scene_id",
    "frame_id",
    "query_id",
    "query_type",
    "target_instance_id",
    "gt_grasp_target_instance_id",
    "language_prompt",
    "rgb_path",
    "rgb_sha256",
    "rgb_height",
    "rgb_width",
    "depth_path",
    "depth_sha256",
    "depth_height",
    "depth_width",
    "intrinsics_path",
    "intrinsics_sha256",
    "prepared_gt_mask_path",
    "prepared_gt_mask_sha256",
    "source_instance_mask_path",
    "source_instance_mask_sha256",
    "gt_grasp_set_path",
    "gt_grasp_set_sha256",
)
ROUTE_OPTIONAL_COLUMNS = (
    "sample_id",
    "scene_id",
    "frame_id",
    "query_id",
    "target_instance_id",
    "rgb_sha256",
    "depth_sha256",
    "language_prompt",
    "source_identity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--denominator", required=True, type=Path)
    parser.add_argument("--g1-source", required=True, type=Path)
    parser.add_argument("--c1-source", required=True, type=Path)
    parser.add_argument("--d1-source", required=True, type=Path)
    parser.add_argument("--unified-final-lock", required=True, type=Path)
    parser.add_argument("--d1-final-lock", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    return parser.parse_args()


def _regular(path: Path) -> Path:
    value = path.expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {value}")
    return value


def _index(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in result:
            raise ValueError(f"{label} has an empty or duplicate sample_id")
        result[sample_id] = row
    return result


def _canonical_denominator(
    visual_path: Path, *, labels_path: Path, paired_path: Path
) -> list[dict[str, Any]]:
    """Join the three frozen Test authorities without opening GT geometry/pixels."""

    visual = _regular(visual_path)
    labels = _regular(labels_path)
    paired = _regular(paired_path)
    visual_schema = pq.read_schema(visual)
    if "gt_grasp_list_json" not in visual_schema.names:
        raise ValueError("visual GT authority lacks frozen grasp geometry")
    visual_columns = [
        "sample_id",
        "scene_id",
        "frame_id",
        "query_id",
        "target_instance_id",
        "expression",
        "expression_type",
        "rgb_path",
        "depth_path",
        "gt_mask_path",
        "image_width",
        "image_height",
        "image_sha256",
        "depth_sha256",
        "rgbd_pair_sha256",
    ]
    label_columns = [
        "sample_id",
        "scene_id",
        "question_index",
        "target_object_id",
        "prepared_gt_mask_path",
        "prepared_gt_mask_sha256",
        "official_annotations_path",
        "official_annotations_sha256",
    ]
    paired_columns = [
        "sample_id",
        "scene_id",
        "frame_id",
        "question_index",
        "source_rgb_path",
        "source_rgb_sha256",
        "source_depth_path",
        "source_depth_sha256",
        "rgbd_pair_sha256",
        "language",
        "intrinsics_path",
        "intrinsics_sha256",
    ]
    for path, columns, label in (
        (visual, visual_columns, "visual GT authority"),
        (labels, label_columns, "opaque label authority"),
        (paired, paired_columns, "paired Test authority"),
    ):
        missing = sorted(set(columns).difference(pq.read_schema(path).names))
        if missing:
            raise ValueError(f"{label} misses columns: {missing}")
    visual_rows = _index(
        pq.read_table(visual, columns=visual_columns).to_pylist(), label="visual GT"
    )
    label_rows = _index(
        pq.read_table(labels, columns=label_columns).to_pylist(), label="opaque labels"
    )
    paired_rows = _index(
        pq.read_table(paired, columns=paired_columns).to_pylist(), label="paired Test"
    )
    if set(visual_rows) != set(label_rows) or set(visual_rows) != set(paired_rows):
        raise ValueError("frozen denominator authority sample universes differ")
    output: list[dict[str, Any]] = []
    for sample_id in sorted(visual_rows):
        visual_row = visual_rows[sample_id]
        label_row = label_rows[sample_id]
        paired_row = paired_rows[sample_id]
        compared = (
            ("scene_id", visual_row["scene_id"], label_row["scene_id"]),
            ("scene_id", visual_row["scene_id"], paired_row["scene_id"]),
            (
                "visual frame path",
                str(Path(str(visual_row["frame_id"])).resolve()),
                str(Path(str(paired_row["source_rgb_path"])).resolve()),
            ),
            (
                "RGBD frame identity",
                visual_row["rgbd_pair_sha256"],
                paired_row["frame_id"],
            ),
            (
                "RGBD frame identity",
                visual_row["rgbd_pair_sha256"],
                paired_row["rgbd_pair_sha256"],
            ),
            ("query_id", visual_row["query_id"], label_row["question_index"]),
            ("query_id", visual_row["query_id"], paired_row["question_index"]),
            (
                "target_instance_id",
                visual_row["target_instance_id"],
                label_row["target_object_id"],
            ),
            ("RGB SHA256", visual_row["image_sha256"], paired_row["source_rgb_sha256"]),
            (
                "depth SHA256",
                visual_row["depth_sha256"],
                paired_row["source_depth_sha256"],
            ),
            ("language", visual_row["expression"], paired_row["language"]),
        )
        for name, left, right in compared:
            if str(left) != str(right):
                raise ValueError(f"{sample_id}: frozen {name} authorities differ")
        output.append(
            {
                "sample_id": sample_id,
                "scene_id": str(visual_row["scene_id"]),
                # The old visual authority used the RGB path as ``frame_id``.
                # Unified reranking canonically defines it as the RGBD-pair
                # digest; the comparisons above prove both records identify
                # the same frozen frame before publishing that canonical ID.
                "frame_id": str(paired_row["frame_id"]),
                "query_id": str(visual_row["query_id"]),
                "query_type": str(visual_row["expression_type"]),
                "target_instance_id": int(visual_row["target_instance_id"]),
                "gt_grasp_target_instance_id": int(label_row["target_object_id"]),
                "language_prompt": str(visual_row["expression"]),
                "rgb_path": str(Path(str(visual_row["rgb_path"])).resolve()),
                "rgb_sha256": str(visual_row["image_sha256"]),
                "rgb_height": int(visual_row["image_height"]),
                "rgb_width": int(visual_row["image_width"]),
                "depth_path": str(Path(str(visual_row["depth_path"])).resolve()),
                "depth_sha256": str(visual_row["depth_sha256"]),
                "depth_height": int(visual_row["image_height"]),
                "depth_width": int(visual_row["image_width"]),
                "intrinsics_path": str(
                    Path(str(paired_row["intrinsics_path"])).resolve()
                ),
                "intrinsics_sha256": str(paired_row["intrinsics_sha256"]),
                "prepared_gt_mask_path": str(
                    Path(str(label_row["prepared_gt_mask_path"])).resolve()
                ),
                "prepared_gt_mask_sha256": str(label_row["prepared_gt_mask_sha256"]),
                "source_instance_mask_path": str(
                    Path(str(visual_row["gt_mask_path"])).resolve()
                ),
                "source_instance_mask_sha256": None,
                "source_instance_mask_hash_status": "PENDING_P2_AUTHORIZED_HASH",
                "gt_grasp_set_path": str(
                    Path(str(label_row["official_annotations_path"])).resolve()
                ),
                "gt_grasp_set_sha256": str(label_row["official_annotations_sha256"]),
            }
        )
    return output


def _route(path: Path, *, route: str) -> list[dict[str, Any]]:
    source = _regular(path)
    schema = pq.read_schema(source)
    if "system_name" in schema.names:
        if route in {"g1", "c1"}:
            columns = [
                "sample_id",
                "scene_id",
                "frame_id",
                "system_name",
                "selected_route",
                "selected_candidate_id",
                "source_route",
                "source_candidate_id",
                "candidate_geometry_sha256",
            ]
            rows = pq.read_table(source, columns=columns).to_pylist()
            rows = [row for row in rows if row["system_name"] == f"{route}_native"]
        else:
            columns = [
                "sample_id",
                "system_name",
                "row_kind",
                "source_route",
                "candidate_id",
                "rank",
                "selected_source_route",
                "selected_candidate_id",
                "is_selected",
                "no_output",
            ]
            rows = pq.read_table(source, columns=columns).to_pylist()
            rows = [
                row
                for row in rows
                if row["system_name"] == "d1_top5_r0"
                and (
                    (row["row_kind"] == "candidate" and bool(row["is_selected"]))
                    or row["row_kind"] == "decision_no_output"
                )
            ]
        result = []
        for row in rows:
            identity_payload = {
                key: row.get(key)
                for key in columns
                if key not in {"scene_id", "frame_id"}
            }
            result.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "scene_id": str(row.get("scene_id") or ""),
                    "frame_id": str(row.get("frame_id") or ""),
                    "source_identity": canonical_sha256(
                        {"route": route, "identity": identity_payload}
                    ),
                }
            )
        if len(result) != 7_675 or len({row["sample_id"] for row in result}) != 7_675:
            raise ValueError(f"{route} formal route identity coverage differs")
        return result
    columns = [name for name in ROUTE_OPTIONAL_COLUMNS if name in schema.names]
    if "source_identity" not in columns:
        raise ValueError(f"route source misses source_identity: {source}")
    if "sample_id" not in columns and not {
        "scene_id",
        "frame_id",
        "query_id",
        "target_instance_id",
    }.issubset(columns):
        raise ValueError(f"route source lacks an exact identity key: {source}")
    return pq.read_table(source, columns=columns).to_pylist()


def _closure_test_inputs(
    d1_lock: Path, *, denominator: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    d1_root = _regular(d1_lock).parent
    closure_path, closure = load_source_closure(d1_root)
    canonical_inputs = closure.get("canonical_inputs")
    test = canonical_inputs.get("test") if isinstance(canonical_inputs, dict) else None
    if not isinstance(test, dict):
        raise PermissionError("D1 source closure lacks canonical Test inputs")
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name in (
        "opaque_visual_ground_truth",
        "opaque_ground_truth",
        "paired_manifest",
    ):
        record = test.get(name)
        if not isinstance(record, dict):
            raise PermissionError(f"D1 source closure lacks {name}")
        path = _regular(Path(str(record.get("path", ""))))
        if artifact_record(path) != record:
            raise PermissionError(f"D1 source closure {name} record differs")
        records[name] = dict(record)
        paths[name] = path
    if paths["opaque_visual_ground_truth"] != _regular(denominator):
        raise PermissionError(
            "denominator is not the active opaque visual GT authority"
        )
    records["source_closure"] = artifact_record(closure_path)
    records["source_closure_pointer"] = artifact_record(
        d1_root / "00_audit/D1_SOURCE_RECONCILIATION_ACTIVE.json"
    )
    return paths, records


def _inventory_records(value: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            records.append(value)
        for child in value.values():
            records.extend(_inventory_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_inventory_records(child))
    return records


def _require_locked_source(path: Path, lock_path: Path) -> None:
    source = _regular(path)
    final_lock = _regular(lock_path)
    value = json.loads(final_lock.read_text(encoding="utf-8"))
    inventory = value.get("inventory") if isinstance(value, dict) else None
    records = _inventory_records(inventory)
    observed_path = str(source)
    observed_hash = sha256_file(source)
    for record in records:
        record_path = record.get("path")
        if record_path is None and record.get("relative_path") is not None:
            record_path = str(
                (final_lock.parent / str(record["relative_path"])).resolve()
            )
        if (
            str(Path(str(record_path)).expanduser().resolve()) == observed_path
            and str(record.get("sha256")) == observed_hash
        ):
            return
    raise PermissionError(f"source is not byte-bound by final lock: {source}")


def _require_bootstrap_lock_authority(
    status: dict[str, Any], *, unified_lock: Path, d1_lock: Path
) -> None:
    """Tie P1 sources to the exact locks fully rehashed at bootstrap."""

    record = status.get("source_lock_verification")
    if not isinstance(record, dict):
        raise PermissionError("pipeline lacks bootstrap source-lock authority")
    verification_path = _regular(Path(str(record.get("path", ""))))
    if artifact_record(verification_path) != record:
        raise PermissionError("bootstrap source-lock verification artifact differs")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    sources = verification.get("sources") if isinstance(verification, dict) else None
    if (
        verification.get("status") != "PASS"
        or verification.get("full_inventory_byte_rehash") is not True
        or not isinstance(sources, dict)
    ):
        raise PermissionError("bootstrap source-lock verification is not a full PASS")
    for name, requested in (("unified", unified_lock), ("d1", d1_lock)):
        source = sources.get(name)
        expected = source.get("final_lock") if isinstance(source, dict) else None
        if not isinstance(expected, dict) or artifact_record(requested) != expected:
            raise PermissionError(
                f"P1 {name} final lock differs from the bootstrap authority"
            )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    status = json.loads((root / "pipeline_status.json").read_text(encoding="utf-8"))
    if status.get("status") != RunState.P1_BASELINE_REPLAY_PASS.value:
        raise PermissionError("P1 manifest requires baseline replay PASS first")
    sources = {
        "denominator": _regular(args.denominator),
        "g1": _regular(args.g1_source),
        "c1": _regular(args.c1_source),
        "d1": _regular(args.d1_source),
        "unified_final_lock": _regular(args.unified_final_lock),
        "d1_final_lock": _regular(args.d1_final_lock),
    }
    _require_bootstrap_lock_authority(
        status,
        unified_lock=sources["unified_final_lock"],
        d1_lock=sources["d1_final_lock"],
    )
    _require_locked_source(sources["g1"], sources["unified_final_lock"])
    _require_locked_source(sources["c1"], sources["unified_final_lock"])
    _require_locked_source(sources["d1"], sources["d1_final_lock"])
    closure_paths, closure_records = _closure_test_inputs(
        sources["d1_final_lock"], denominator=sources["denominator"]
    )
    result = build_counterfactual_manifest(
        _canonical_denominator(
            closure_paths["opaque_visual_ground_truth"],
            labels_path=closure_paths["opaque_ground_truth"],
            paired_path=closure_paths["paired_manifest"],
        ),
        {route: _route(sources[route], route=route) for route in ("g1", "c1", "d1")},
        expected_count=args.expected_count,
    )
    publish_manifest_artifacts(
        root,
        result,
        source_records={
            **{name: artifact_record(path) for name, path in sources.items()},
            **closure_records,
        },
    )
    transition_pipeline_status(
        root,
        RunState.P1_BASELINE_REPLAY_PASS,
        first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
