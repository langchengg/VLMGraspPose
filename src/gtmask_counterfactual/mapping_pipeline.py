"""Durable P1/P2 artifact publication and deterministic annotation QA."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import json
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from .io import (
    artifact_record,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from .manifest import ManifestBuildResult
from .mapping import GTMappingError, mapping_pixel_qa


P1_DIRECTORY = Path("02_sample_manifest")
P2_DIRECTORY = Path("03_gt_mask_registry")
P2_ACCESS = {
    "status": "AUTHORIZED",
    "stage": "P2_GT_MAPPING_PASS",
    "purpose": "mapping_and_annotation_pixel_qa_only",
    "candidate_generation_allowed": False,
}


def _record(path: Path) -> dict[str, Any]:
    return artifact_record(path.expanduser().resolve())


def publish_manifest_artifacts(
    run_dir: str | Path,
    result: ManifestBuildResult,
    *,
    source_records: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the exact denominator, its complement, and a hashed join audit."""

    root = Path(run_dir).expanduser().resolve()
    output = root / P1_DIRECTORY
    manifest_frame = pd.DataFrame(result.rows)
    unresolved_frame = pd.DataFrame(
        result.unresolved,
        columns=("sample_id", "counterfactual_status", "mapping_reason"),
    )
    parquet = atomic_parquet(manifest_frame, output / "counterfactual_manifest.parquet")
    csv = atomic_csv(manifest_frame, output / "counterfactual_manifest.csv")
    unresolved = atomic_csv(unresolved_frame, output / "unresolved_samples.csv")
    route_lines = []
    for route, counts in sorted(result.audit["route_join_counts"].items()):
        route_lines.append(
            f"| {route} | {counts.get('sample_id', 0)} | "
            f"{counts.get('composite', 0)} | "
            f"{counts.get('asset_prompt_annotation', 0)} | "
            f"{counts.get('unresolved', 0)} |"
        )
    markdown = "\n".join(
        [
            "# Counterfactual sample join audit",
            "",
            "No row-number join, GT-mask pixel read, or GT-grasp-row read was used.",
            "",
            f"- N_total: {result.audit['sample_count']}",
            "- N_counterfactual_evaluable: "
            f"{result.audit['counterfactual_evaluable_count']}",
            f"- N_unresolved: {result.audit['unresolved_count']}",
            "",
            "| route | sample_id | composite | asset/prompt/annotation | unresolved |",
            "|---|---:|---:|---:|---:|",
            *route_lines,
            "",
            "Machine audit: `JOIN_AUDIT.json`.",
            "",
        ]
    )
    join_markdown = atomic_text(output / "join_audit.md", markdown)
    audit = {
        **result.audit,
        "sources": dict(source_records),
        "outputs": {
            "counterfactual_manifest_parquet": _record(parquet),
            "counterfactual_manifest_csv": _record(csv),
            "unresolved_samples": _record(unresolved),
            "join_audit_markdown": _record(join_markdown),
        },
    }
    audit["content_sha256"] = canonical_sha256(audit)
    atomic_json(output / "JOIN_AUDIT.json", audit)
    return audit


def _hash_order(sample_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return digest, sample_id


def select_mapping_qa_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_total: int = 150,
    minimum_per_query_type: int = 20,
    minimum_per_quartile: int = 20,
    seed: int = 20260813,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one deterministic shared case set satisfying every frozen stratum."""

    eligible = [
        dict(row)
        for row in rows
        if row.get("mapping_status") == "PASS"
        and row.get("pixel_qa_status") == "P2_MAPPING_QA_PASS"
    ]
    ordered_fraction = sorted(
        eligible,
        key=lambda row: (
            float(row["foreground_fraction"]),
            _hash_order(str(row["sample_id"]), seed),
        ),
    )
    size = len(ordered_fraction)
    for index, row in enumerate(ordered_fraction):
        row["target_size_quartile"] = min(4, (index * 4) // max(size, 1) + 1)
    by_id = {str(row["sample_id"]): row for row in ordered_fraction}
    query_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quartile_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered_fraction:
        query_groups[str(row.get("query_type", ""))].append(row)
        quartile_groups[int(row["target_size_quartile"])].append(row)
    failures: list[str] = []
    if "" in query_groups:
        failures.append("eligible rows contain an empty query_type")
    for query_type, group in sorted(query_groups.items()):
        if query_type and len(group) < minimum_per_query_type:
            failures.append(
                f"query_type={query_type} has {len(group)} < {minimum_per_query_type}"
            )
    for quartile in range(1, 5):
        count = len(quartile_groups.get(quartile, []))
        if count < minimum_per_quartile:
            failures.append(
                f"target_size_quartile={quartile} has {count} < {minimum_per_quartile}"
            )
    if len(eligible) < minimum_total:
        failures.append(f"eligible case count {len(eligible)} < {minimum_total}")

    selected: set[str] = set()

    def add(group: Sequence[Mapping[str, Any]], count: int) -> None:
        ranked = sorted(
            group, key=lambda row: _hash_order(str(row["sample_id"]), seed)
        )
        selected.update(str(row["sample_id"]) for row in ranked[:count])

    if not failures:
        for query_type, group in sorted(query_groups.items()):
            if query_type:
                add(group, minimum_per_query_type)
        for quartile in range(1, 5):
            add(quartile_groups[quartile], minimum_per_quartile)
        remaining = sorted(
            (row for row in eligible if str(row["sample_id"]) not in selected),
            key=lambda row: _hash_order(str(row["sample_id"]), seed),
        )
        selected.update(
            str(row["sample_id"])
            for row in remaining[: max(0, minimum_total - len(selected))]
        )
    selected_rows = [by_id[sample_id] for sample_id in sorted(selected)]
    query_counts = Counter(str(row["query_type"]) for row in selected_rows)
    quartile_counts = Counter(int(row["target_size_quartile"]) for row in selected_rows)
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "seed": seed,
        "selection_key": "sha256(seed:sample_id)",
        "eligible_count": len(eligible),
        "selected_case_count": len(selected_rows),
        "minimum_total": minimum_total,
        "minimum_per_query_type": minimum_per_query_type,
        "minimum_per_target_size_quartile": minimum_per_quartile,
        "query_type_counts": dict(sorted(query_counts.items())),
        "target_size_quartile_counts": {
            str(key): value for key, value in sorted(quartile_counts.items())
        },
        "routes": ["g1", "c1", "d1"],
        "same_cases_shared_across_routes": True,
        "selection_is_outcome_blind": True,
        "selected_sample_ids_sha256": canonical_sha256(
            sorted(str(row["sample_id"]) for row in selected_rows)
        ),
    }
    return selected_rows, audit


def _atomic_image(image: Image.Image, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format="PNG")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _case_panel(row: Mapping[str, Any], *, width: int = 320, height: int = 282) -> Image.Image:
    rgb_path = Path(str(row["rgb_path"]))
    mask_path = Path(str(row["gt_mask_path"]))
    with Image.open(rgb_path) as image:
        rgb = image.convert("RGB")
    with Image.open(mask_path) as image:
        mask = image.convert("L")
    if rgb.size != mask.size:
        raise GTMappingError(f"{row['sample_id']}: contact-sheet RGB/mask size differs")
    overlay = Image.new("RGBA", rgb.size, (255, 0, 0, 0))
    alpha = mask.point(lambda value: 100 if value else 0)
    overlay.putalpha(alpha)
    composed = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
    composed.thumbnail((width, height - 42), Image.Resampling.BILINEAR)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(composed, ((width - composed.width) // 2, 42))
    draw = ImageDraw.Draw(panel)
    title = (
        f"{row['sample_id']} | {row['query_type']} | Q{row['target_size_quartile']}"
    )
    draw.text((5, 4), title[:54], fill="black")
    draw.text((5, 21), str(row.get("language_prompt", ""))[:54], fill="black")
    return panel


def render_mapping_contact_sheets(
    rows: Sequence[Mapping[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    per_sheet = 25
    records: list[dict[str, Any]] = []
    for page, offset in enumerate(range(0, len(rows), per_sheet), start=1):
        batch = rows[offset : offset + per_sheet]
        sheet = Image.new("RGB", (5 * 320, 5 * 282), (235, 235, 235))
        for index, row in enumerate(batch):
            sheet.paste(_case_panel(row), ((index % 5) * 320, (index // 5) * 282))
        path = _atomic_image(
            sheet, destination / f"mapping_pixel_qa_contact_sheet_{page:03d}.png"
        )
        records.append(
            {
                **_record(path),
                "sample_ids": [str(row["sample_id"]) for row in batch],
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "RENDERED_PENDING_MANUAL_QA",
        "case_count": len(rows),
        "sheet_count": len(records),
        "routes": ["g1", "c1", "d1"],
        "same_cases_shared_across_routes": True,
        "selected_sample_ids_sha256": canonical_sha256(
            sorted(str(row["sample_id"]) for row in rows)
        ),
        "sheets": records,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    path = atomic_json(destination / "MAPPING_PIXEL_QA_CONTACT_SHEETS.json", manifest)
    return {**manifest, "artifact": _record(path)}


def validate_manual_mapping_qa(
    selected_rows: Sequence[Mapping[str, Any]],
    manual_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Require a one-to-one signed visual review of the frozen case set."""

    selected = {str(row["sample_id"]) for row in selected_rows}
    if manual_rows is None:
        return {
            "status": "PENDING_MANUAL_QA",
            "expected_case_count": len(selected),
            "reviewed_case_count": 0,
            "annotation_suspect_sample_ids": [],
            "failure_reasons": ["manual mapping QA CSV has not been supplied"],
        }
    indexed: dict[str, Mapping[str, Any]] = {}
    failures: list[str] = []
    suspects: list[str] = []
    for row in manual_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            failures.append("manual QA contains an empty sample_id")
            continue
        if sample_id in indexed:
            failures.append(f"manual QA duplicates sample_id={sample_id}")
            continue
        indexed[sample_id] = row
        decision = str(row.get("review_status", "")).strip().upper()
        reviewer = str(row.get("reviewer", "")).strip()
        reviewed_at = str(row.get("reviewed_at_utc", "")).strip()
        signature = str(row.get("review_signature", "")).strip()
        if decision not in {"PASS", "ANNOTATION_SUSPECT"}:
            failures.append(f"{sample_id}: invalid review_status={decision!r}")
        signed_payload = {
            "sample_id": sample_id,
            "review_status": decision,
            "reviewer": reviewer,
            "reviewed_at_utc": reviewed_at,
            "review_notes": str(row.get("review_notes", "")).strip(),
        }
        if not reviewer or not reviewed_at or signature != canonical_sha256(signed_payload):
            failures.append(
                f"{sample_id}: reviewer/timestamp/signature contract differs"
            )
        if decision == "ANNOTATION_SUSPECT":
            suspects.append(sample_id)
    missing = sorted(selected - set(indexed))
    extras = sorted(set(indexed) - selected)
    if missing:
        failures.append(f"manual QA misses selected samples: {missing[:5]}")
    if extras:
        failures.append(f"manual QA includes non-selected samples: {extras[:5]}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "expected_case_count": len(selected),
        "reviewed_case_count": len(indexed),
        "annotation_suspect_sample_ids": sorted(suspects),
        "reviewed_sample_ids_sha256": canonical_sha256(sorted(indexed)),
        "selected_sample_ids_sha256": canonical_sha256(sorted(selected)),
        "failure_reasons": failures,
    }


def _asset_shape_and_hash_qa(row: Mapping[str, Any]) -> None:
    sample_id = str(row["sample_id"])
    expected = (int(row["rgb_width"]), int(row["rgb_height"]))
    for prefix in ("rgb", "depth"):
        path = Path(str(row[f"{prefix}_path"]))
        if sha256_file(path) != str(row[f"{prefix}_sha256"]):
            raise GTMappingError(f"{sample_id}: {prefix} hash mismatch")
        with Image.open(path) as image:
            if image.size != expected:
                raise GTMappingError(
                    f"{sample_id}: {prefix} size {image.size} differs from {expected}"
                )
    intrinsics = Path(str(row["intrinsics_path"]))
    if sha256_file(intrinsics) != str(row["intrinsics_sha256"]):
        raise GTMappingError(f"{sample_id}: intrinsics hash mismatch")


def run_mapping_pixel_qa_bulk(
    manifest_rows: Sequence[Mapping[str, Any]],
    prelock_rows: Sequence[Mapping[str, Any]],
    *,
    run_dir: str | Path,
    expected_count: int = 7_675,
    minimum_contact_cases: int = 150,
    minimum_per_query_type: int = 20,
    minimum_per_quartile: int = 20,
    input_records: Mapping[str, Any] | None = None,
    manual_qa_rows: Sequence[Mapping[str, Any]] | None = None,
    manual_qa_record: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the P2-only bulk read and publish its exact denominator partition."""

    if len(manifest_rows) != expected_count or len(prelock_rows) != expected_count:
        raise GTMappingError("P2 inputs do not preserve the expected denominator")
    manifest = {str(row["sample_id"]): dict(row) for row in manifest_rows}
    prelock = {str(row["sample_id"]): dict(row) for row in prelock_rows}
    if len(manifest) != expected_count or len(prelock) != expected_count:
        raise GTMappingError("P2 inputs contain duplicate sample IDs")
    if set(manifest) != set(prelock):
        raise GTMappingError("P1 manifest and pre-lock registry sample IDs differ")

    root = Path(run_dir).expanduser().resolve()
    output = root / P2_DIRECTORY
    derived = output / "derived_original_gt_masks"
    evidence_dir = output / "per_sample_pixel_qa"
    rows: list[dict[str, Any]] = []
    pixels_read = 0
    for sample_id in sorted(manifest):
        source = {**manifest[sample_id], **prelock[sample_id]}
        source_fingerprint = canonical_sha256(source)
        shard_path = evidence_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"
        if resume and shard_path.is_file() and not shard_path.is_symlink():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            unsigned = dict(shard)
            recorded = unsigned.pop("content_sha256", None)
            if (
                recorded != canonical_sha256(unsigned)
                or shard.get("sample_id") != sample_id
                or shard.get("source_fingerprint") != source_fingerprint
                or not isinstance(shard.get("row"), dict)
            ):
                raise GTMappingError(f"{sample_id}: resume evidence shard differs")
            legacy_round_trip_rejection = (
                shard["row"].get("mapping_status")
                == "technical_or_annotation_mapping_failure"
                and "PIL-nearest resize/inverse round-trip IoU"
                in str(shard["row"].get("mapping_reason", ""))
            )
            if not legacy_round_trip_rejection:
                rows.append(dict(shard["row"]))
                pixels_read += int(bool(shard["row"].get("bulk_gt_pixels_read")))
                continue
        mapping_status = source.get("mapping_status")
        if mapping_status not in {
            "PATH_HASH_INSTANCE_AUTHORITY_MAPPED",
            "PATH_INSTANCE_AUTHORITY_MAPPED_PENDING_P2_HASH",
        }:
            result = {
                **source,
                "mapping_status": "technical_or_annotation_mapping_failure",
                "pixel_qa_status": "NOT_READ_UNRESOLVED",
                "mapping_reason": source.get("mapping_reason")
                or "pre-lock GT authority unresolved",
                "bulk_gt_pixels_read": False,
            }
            rows.append(result)
            shard = {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_fingerprint": source_fingerprint,
                "row": result,
            }
            shard["content_sha256"] = canonical_sha256(shard)
            atomic_json(shard_path, shard)
            continue
        try:
            pixels_read += 1
            if mapping_status == "PATH_INSTANCE_AUTHORITY_MAPPED_PENDING_P2_HASH":
                instance_path = Path(str(source["source_instance_mask_path"]))
                source["source_instance_mask_sha256"] = sha256_file(instance_path)
                source["source_instance_mask_hash_status"] = (
                    "COMPUTED_UNDER_P2_AUTHORITY"
                )
                source["mapping_status"] = "PATH_HASH_INSTANCE_AUTHORITY_MAPPED"
            _asset_shape_and_hash_qa(source)
            result = mapping_pixel_qa(
                source,
                access_authority=P2_ACCESS,
                derived_output_dir=derived,
            )
            result.update(
                {
                    "gt_mask_path": result["original_gt_mask_path"],
                    "gt_mask_sha256": result["original_gt_mask_sha256"],
                    "height": int(source["rgb_height"]),
                    "width": int(source["rgb_width"]),
                    "annotation_suspect": False,
                    "query_type": source.get("query_type", ""),
                    "language_prompt": source.get("language_prompt", ""),
                    "rgb_path": source.get("rgb_path", ""),
                }
            )
            if source.get("counterfactual_evaluable") is False:
                result["mapping_status"] = "technical_or_annotation_mapping_failure"
                result["mapping_reason"] = str(source.get("mapping_reason", "")) or (
                    "P1 route/source identity unresolved"
                )
            rows.append(result)
        except (GTMappingError, ValueError, OSError) as error:
            result = {
                **source,
                "mapping_status": "technical_or_annotation_mapping_failure",
                "pixel_qa_status": "FAIL",
                "mapping_reason": str(error),
                "bulk_gt_pixels_read": True,
                "annotation_suspect": True,
            }
            rows.append(result)
        shard = {
            "schema_version": 1,
            "sample_id": sample_id,
            "source_fingerprint": source_fingerprint,
            "row": result,
        }
        shard["content_sha256"] = canonical_sha256(shard)
        atomic_json(shard_path, shard)

    # Multiple language prompts for one physical target must resolve identically.
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("pixel_qa_status") == "P2_MAPPING_QA_PASS":
            groups[
                (
                    str(row.get("scene_id", "")),
                    str(row.get("frame_id", "")),
                    str(row.get("target_instance_id", "")),
                )
            ].append(index)
    for indices in groups.values():
        hashes = {str(rows[index].get("gt_mask_sha256", "")) for index in indices}
        if len(hashes) > 1:
            for index in indices:
                rows[index]["mapping_status"] = "technical_or_annotation_mapping_failure"
                rows[index]["pixel_qa_status"] = "FAIL"
                rows[index]["annotation_suspect"] = True
                rows[index]["mapping_reason"] = (
                    "multi-language prompts for one target have different GT mask hashes"
                )

    selected, selection = select_mapping_qa_cases(
        rows,
        minimum_total=minimum_contact_cases,
        minimum_per_query_type=minimum_per_query_type,
        minimum_per_quartile=minimum_per_quartile,
    )
    contact: dict[str, Any] = {
        "status": "NOT_RENDERED",
        "case_count": 0,
        "artifact": None,
    }
    if selection["status"] == "PASS":
        try:
            contact = render_mapping_contact_sheets(selected, output / "mapping_pixel_qa")
        except (GTMappingError, ValueError, OSError) as error:
            selection["status"] = "FAIL"
            selection["failure_reasons"].append(f"contact-sheet render failed: {error}")

    manual = validate_manual_mapping_qa(selected, manual_qa_rows)
    suspect_ids = set(manual["annotation_suspect_sample_ids"])
    for row in rows:
        if str(row["sample_id"]) in suspect_ids:
            row["annotation_suspect"] = True
    registry_frame = pd.DataFrame(rows)
    registry_path = atomic_parquet(registry_frame, output / "gt_mask_registry.parquet")
    registry_csv = atomic_csv(registry_frame, output / "gt_mask_registry.csv")
    unresolved_ids = sorted(
        str(row["sample_id"]) for row in rows if row.get("mapping_status") != "PASS"
    )
    mapping_unresolved = atomic_csv(
        pd.DataFrame(
            [
                {
                    "sample_id": row["sample_id"],
                    "counterfactual_status": "technical_or_annotation_mapping_failure",
                    "mapping_reason": row.get("mapping_reason", ""),
                }
                for row in rows
                if row.get("mapping_status") != "PASS"
            ],
            columns=("sample_id", "counterfactual_status", "mapping_reason"),
        ),
        output / "mapping_unresolved_samples.csv",
    )
    evidence_records = [_record(path) for path in sorted(evidence_dir.glob("*.json"))]
    if len(evidence_records) != expected_count:
        raise GTMappingError("P2 per-sample evidence shard count differs from denominator")
    evidence_manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "sample_count": expected_count,
        "records": evidence_records,
    }
    evidence_manifest["content_sha256"] = canonical_sha256(evidence_manifest)
    evidence_manifest_path = atomic_json(
        output / "P2_PER_SAMPLE_EVIDENCE_MANIFEST.json", evidence_manifest
    )
    qa_pass = (
        selection["status"] == "PASS"
        and contact.get("status") == "RENDERED_PENDING_MANUAL_QA"
        and manual["status"] == "PASS"
    )
    round_trip_values = [
        float(row["resize_inverse_round_trip_iou"])
        for row in rows
        if row.get("mapping_status") == "PASS"
    ]
    round_trip_below_reference_count = sum(
        bool(row.get("resize_inverse_round_trip_below_reference"))
        for row in rows
        if row.get("mapping_status") == "PASS"
    )
    transpose_evidence_count = sum(
        isinstance(row.get("xy_orientation_evidence"), Mapping)
        and row["xy_orientation_evidence"].get("status") == "PASS"
        for row in rows
        if row.get("mapping_status") == "PASS"
    )
    identity_evidence_count = sum(
        isinstance(row.get("gt_mask_grasp_target_identity_evidence"), Mapping)
        and row["gt_mask_grasp_target_identity_evidence"].get("status") == "PASS"
        for row in rows
        if row.get("mapping_status") == "PASS"
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if qa_pass else manual["status"],
        "stage": "P2_GT_MAPPING_PASS",
        "pixel_qa_status": (
            "P2_MAPPING_QA_PASS"
            if qa_pass
            else "P2_MAPPING_QA_PENDING_MANUAL"
            if manual["status"] == "PENDING_MANUAL_QA"
            else "P2_MAPPING_QA_FAIL"
        ),
        "sample_count": len(rows),
        "counterfactual_evaluable_count": len(rows) - len(unresolved_ids),
        "unresolved_count": len(unresolved_ids),
        "partition_total": len(rows),
        "unresolved_sample_ids_sha256": canonical_sha256(unresolved_ids),
        "mapping_qa_gt_mask_rows_read": pixels_read,
        "candidate_generation_gt_mask_rows_read": 0,
        "candidate_generation_allowed": False,
        "annotation_suspect_count": sum(
            bool(row.get("annotation_suspect")) for row in rows
        ),
        "inputs": dict(input_records or {}),
        "qa_evidence": {
            "successful_pixel_qa_count": len(round_trip_values),
            "gt_mask_grasp_authority_identity_evidence_count": identity_evidence_count,
            "xy_orientation_evidence_count": transpose_evidence_count,
            "observed_resize_inverse_round_trip_min_iou": min(
                round_trip_values, default=None
            ),
            "required_resize_inverse_round_trip_min_iou": None,
            "resize_alignment_acceptance_rule": (
                "exact_forward_PIL_nearest_match; inverse IoU is diagnostic only"
            ),
            "resize_inverse_round_trip_below_0_95_count": (
                round_trip_below_reference_count
            ),
            "gt_grasp_rows_read": 0,
            "identity_evidence_scope": "hash-bound authority metadata only",
        },
        "asset_qa": selection,
        "contact_sheets": contact,
        "manual_asset_qa": {
            **manual,
            "artifact": dict(manual_qa_record) if manual_qa_record else None,
        },
        "outputs": {
            "gt_mask_registry": _record(registry_path),
            "gt_mask_registry_csv": _record(registry_csv),
            "mapping_unresolved_samples": _record(mapping_unresolved),
            "per_sample_evidence_manifest": _record(evidence_manifest_path),
        },
    }
    audit["content_sha256"] = canonical_sha256(audit)
    audit_path = atomic_json(output / "GT_MASK_MAPPING_AUDIT.json", audit)
    final_path = atomic_json(output / "FINAL_P2_MAPPING_PIXEL_QA.json", audit)
    return {
        **audit,
        "audit_artifact": _record(audit_path),
        "final_artifact": _record(final_path),
    }


__all__ = [
    "P1_DIRECTORY",
    "P2_ACCESS",
    "P2_DIRECTORY",
    "publish_manifest_artifacts",
    "render_mapping_contact_sheets",
    "run_mapping_pixel_qa_bulk",
    "select_mapping_qa_cases",
    "validate_manual_mapping_qa",
]
