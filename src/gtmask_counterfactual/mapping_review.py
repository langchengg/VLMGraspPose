"""Deterministic handoff and signing for the P2 visual mapping review."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import artifact_record, atomic_csv, atomic_json, canonical_sha256


P2_DIRECTORY = Path("03_gt_mask_registry")
TEMPLATE_NAME = "MAPPING_MANUAL_QA_TEMPLATE.csv"
SIGNED_NAME = "MAPPING_MANUAL_QA_SIGNED.csv"


def _verified_json(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"required mapping-review artifact is absent: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"mapping-review JSON is not an object: {source}")
    claimed = value.get("content_sha256")
    payload = dict(value)
    payload.pop("content_sha256", None)
    if not isinstance(claimed, str) or claimed != canonical_sha256(payload):
        raise ValueError(f"mapping-review JSON self-hash differs: {source}")
    return value


def _verified_record(record: object, *, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} artifact record is absent")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    observed = artifact_record(path)
    if any(observed[key] != record.get(key) for key in observed):
        raise ValueError(f"{name} artifact record differs")
    return path


def _template_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = run_dir / P2_DIRECTORY
    audit_path = output / "GT_MASK_MAPPING_AUDIT.json"
    audit = _verified_json(audit_path)
    if (
        audit.get("status") != "PENDING_MANUAL_QA"
        or audit.get("pixel_qa_status") != "P2_MAPPING_QA_PENDING_MANUAL"
    ):
        raise PermissionError("mapping review template requires pending P2 manual QA")
    contact_payload = audit.get("contact_sheets")
    if not isinstance(contact_payload, Mapping):
        raise ValueError("P2 audit lacks contact-sheet evidence")
    contact_path = _verified_record(
        contact_payload.get("artifact"), name="mapping contact-sheet manifest"
    )
    contact = _verified_json(contact_path)
    if contact.get("status") != "RENDERED_PENDING_MANUAL_QA":
        raise ValueError("mapping contact sheets are not pending manual QA")
    registry_record = (
        audit.get("outputs", {}).get("gt_mask_registry")
        if isinstance(audit.get("outputs"), Mapping)
        else None
    )
    registry_path = _verified_record(registry_record, name="P2 GT-mask registry")
    registry = pd.read_parquet(
        registry_path,
        columns=(
            "sample_id",
            "query_type",
            "language_prompt",
            "mapping_status",
            "pixel_qa_status",
        ),
    )
    if registry["sample_id"].duplicated().any():
        raise ValueError("P2 registry duplicates sample_id")
    indexed = registry.set_index("sample_id", drop=False)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    sheets = contact.get("sheets")
    if not isinstance(sheets, list):
        raise ValueError("contact-sheet manifest lacks sheets")
    for sheet_index, sheet in enumerate(sheets, start=1):
        sheet_path = _verified_record(sheet, name=f"contact sheet {sheet_index}")
        sample_ids = sheet.get("sample_ids")
        if not isinstance(sample_ids, list):
            raise ValueError(f"contact sheet {sheet_index} lacks sample IDs")
        for case_position, raw_sample_id in enumerate(sample_ids, start=1):
            sample_id = str(raw_sample_id)
            if sample_id in seen or sample_id not in indexed.index:
                raise ValueError(
                    "contact-sheet sample coverage differs from P2 registry"
                )
            seen.add(sample_id)
            source = indexed.loc[sample_id]
            if (
                source["mapping_status"] != "PASS"
                or source["pixel_qa_status"] != "P2_MAPPING_QA_PASS"
            ):
                raise ValueError(
                    f"manual-review case is not P2 pixel-QA PASS: {sample_id}"
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "query_type": str(source["query_type"]),
                    "language_prompt": str(source["language_prompt"]),
                    "contact_sheet_path": str(sheet_path),
                    "contact_sheet_index": sheet_index,
                    "case_position": case_position,
                    "review_status": "",
                    "reviewer": "",
                    "reviewed_at_utc": "",
                    "review_notes": "",
                    "review_signature": "",
                }
            )
    if len(rows) != int(contact.get("case_count", -1)) or len(rows) < 150:
        raise ValueError("mapping contact-sheet case count differs")
    if canonical_sha256(sorted(seen)) != contact.get("selected_sample_ids_sha256"):
        raise ValueError("mapping contact-sheet sample identity hash differs")
    return rows, {
        "mapping_audit": artifact_record(audit_path),
        "contact_sheets": artifact_record(contact_path),
        "gt_mask_registry": artifact_record(registry_path),
    }


def prepare_mapping_review_template(
    run_dir: str | Path, *, resume: bool = False
) -> Path:
    """Create the exact 150-case review sheet without making review decisions."""

    root = Path(run_dir).expanduser().resolve()
    rows, sources = _template_rows(root)
    output = root / P2_DIRECTORY
    destination = output / TEMPLATE_NAME
    expected = pd.DataFrame(rows)
    if destination.exists():
        observed = pd.read_csv(destination, keep_default_na=False)
        if not resume:
            raise FileExistsError("existing mapping manual-QA template differs")
        if not observed.equals(expected):
            if (output / SIGNED_NAME).exists():
                raise FileExistsError(
                    "cannot replace a stale mapping template after review signing"
                )
            atomic_csv(expected, destination)
    else:
        atomic_csv(expected, destination)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PENDING_MANUAL_QA",
        "case_count": len(rows),
        "selected_sample_ids_sha256": canonical_sha256(
            sorted(row["sample_id"] for row in rows)
        ),
        "sources": sources,
        "template": artifact_record(destination),
        "review_decisions_populated": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path = output / "MAPPING_MANUAL_QA_TEMPLATE.json"
    if manifest_path.exists():
        observed_manifest = _verified_json(manifest_path)
        if not resume:
            raise FileExistsError("existing mapping review-template manifest differs")
        if observed_manifest != manifest:
            if (output / SIGNED_NAME).exists():
                raise FileExistsError(
                    "cannot replace a stale mapping manifest after review signing"
                )
            atomic_json(manifest_path, manifest)
    else:
        atomic_json(manifest_path, manifest)
    return destination


def sign_mapping_review(
    run_dir: str | Path, review_csv: str | Path, *, resume: bool = False
) -> Path:
    """Sign fully populated visual decisions; never infer or fill a decision."""

    root = Path(run_dir).expanduser().resolve()
    template_path = prepare_mapping_review_template(root, resume=True)
    template = pd.read_csv(template_path, keep_default_na=False)
    review_path = Path(review_csv).expanduser().resolve()
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError(f"manual mapping review CSV is absent: {review_path}")
    review = pd.read_csv(review_path, keep_default_na=False)
    required = {
        "sample_id",
        "review_status",
        "reviewer",
        "reviewed_at_utc",
        "review_notes",
    }
    missing = sorted(required.difference(review.columns))
    if missing:
        raise ValueError(f"manual mapping review misses columns: {missing}")
    if review["sample_id"].duplicated().any():
        raise ValueError("manual mapping review duplicates sample_id")
    if set(review["sample_id"].astype(str)) != set(template["sample_id"].astype(str)):
        raise ValueError("manual mapping review sample coverage differs from template")
    review = review.set_index("sample_id").loc[template["sample_id"]].reset_index()
    signed: list[dict[str, str]] = []
    for row in review.to_dict("records"):
        payload = {
            "sample_id": str(row["sample_id"]).strip(),
            "review_status": str(row["review_status"]).strip().upper(),
            "reviewer": str(row["reviewer"]).strip(),
            "reviewed_at_utc": str(row["reviewed_at_utc"]).strip(),
            "review_notes": str(row["review_notes"]).strip(),
        }
        if payload["review_status"] not in {"PASS", "ANNOTATION_SUSPECT"}:
            raise ValueError(f"{payload['sample_id']}: manual decision is incomplete")
        if not payload["reviewer"]:
            raise ValueError(f"{payload['sample_id']}: reviewer is empty")
        try:
            reviewed_at = datetime.fromisoformat(
                payload["reviewed_at_utc"].replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError(
                f"{payload['sample_id']}: reviewed_at_utc is invalid"
            ) from error
        if reviewed_at.tzinfo is None:
            raise ValueError(f"{payload['sample_id']}: reviewed_at_utc lacks timezone")
        signed.append({**payload, "review_signature": canonical_sha256(payload)})
    destination = root / P2_DIRECTORY / SIGNED_NAME
    expected = pd.DataFrame(signed)
    if destination.exists():
        observed = pd.read_csv(destination, keep_default_na=False)
        if not resume or not observed.equals(expected):
            raise FileExistsError("existing signed mapping review differs")
    else:
        atomic_csv(expected, destination)
    return destination


__all__ = ["prepare_mapping_review_template", "sign_mapping_review"]
