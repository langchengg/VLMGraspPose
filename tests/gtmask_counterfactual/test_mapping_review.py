from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

from gtmask_counterfactual.io import (
    artifact_record,
    atomic_json,
    canonical_sha256,
)
from gtmask_counterfactual.mapping_review import (
    prepare_mapping_review_template,
    sign_mapping_review,
)
from gtmask_counterfactual.mapping_pipeline import validate_manual_mapping_qa


def _json(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return atomic_json(path, payload)


def _fixture(root: Path) -> None:
    output = root / "03_gt_mask_registry"
    registry = pd.DataFrame(
        [
            {
                "sample_id": f"s{index:03d}",
                "query_type": "name",
                "language_prompt": f"pick {index}",
                "mapping_status": "PASS",
                "pixel_qa_status": "P2_MAPPING_QA_PASS",
            }
            for index in range(150)
        ]
    )
    registry_path = output / "gt_mask_registry.parquet"
    registry_path.parent.mkdir(parents=True)
    registry.to_parquet(registry_path, index=False)
    sheet = output / "mapping_pixel_qa/sheet.png"
    sheet.parent.mkdir(parents=True)
    sheet.write_bytes(b"synthetic sheet")
    ids = registry["sample_id"].tolist()
    contact_path = _json(
        output / "mapping_pixel_qa/MAPPING_PIXEL_QA_CONTACT_SHEETS.json",
        {
            "schema_version": 1,
            "status": "RENDERED_PENDING_MANUAL_QA",
            "case_count": 150,
            "selected_sample_ids_sha256": canonical_sha256(sorted(ids)),
            "sheets": [{**artifact_record(sheet), "sample_ids": ids}],
        },
    )
    _json(
        output / "GT_MASK_MAPPING_AUDIT.json",
        {
            "schema_version": 1,
            "status": "PENDING_MANUAL_QA",
            "pixel_qa_status": "P2_MAPPING_QA_PENDING_MANUAL",
            "contact_sheets": {"artifact": artifact_record(contact_path)},
            "outputs": {"gt_mask_registry": artifact_record(registry_path)},
        },
    )


def test_prepare_and_sign_mapping_review(tmp_path: Path) -> None:
    _fixture(tmp_path)
    template_path = prepare_mapping_review_template(tmp_path)
    template = pd.read_csv(template_path, keep_default_na=False)
    assert len(template) == 150
    assert not template["review_status"].any()
    review = template.copy()
    review["review_status"] = "PASS"
    review["reviewer"] = "human@example.invalid"
    review["reviewed_at_utc"] = "2026-08-15T12:00:00Z"
    review_path = tmp_path / "review.csv"
    review.to_csv(review_path, index=False)
    signed_path = sign_mapping_review(tmp_path, review_path)
    signed = pd.read_csv(signed_path, keep_default_na=False)
    assert len(signed) == 150
    assert signed["review_signature"].str.fullmatch(r"[0-9a-f]{64}").all()

    # The production CLI must keep ISO timestamps as strings.  Letting Arrow
    # infer a timestamp changes ``T`` into a space on string conversion and
    # therefore invalidates an otherwise correct content signature.
    rows = pacsv.read_csv(
        signed_path,
        convert_options=pacsv.ConvertOptions(
            column_types={
                name: pa.string()
                for name in (
                    "sample_id",
                    "review_status",
                    "reviewer",
                    "reviewed_at_utc",
                    "review_notes",
                    "review_signature",
                )
            }
        ),
    ).to_pylist()
    selected = [{"sample_id": value} for value in template["sample_id"]]
    assert validate_manual_mapping_qa(selected, rows)["status"] == "PASS"


def test_mapping_review_rejects_incomplete_decision_and_tamper(tmp_path: Path) -> None:
    _fixture(tmp_path)
    template_path = prepare_mapping_review_template(tmp_path)
    with pytest.raises(ValueError, match="decision is incomplete"):
        sign_mapping_review(tmp_path, template_path)
    audit_path = tmp_path / "03_gt_mask_registry/GT_MASK_MAPPING_AUDIT.json"
    audit_path.write_text(audit_path.read_text().replace("PENDING", "CHANGED", 1))
    with pytest.raises(ValueError, match="self-hash differs"):
        prepare_mapping_review_template(tmp_path, resume=True)
