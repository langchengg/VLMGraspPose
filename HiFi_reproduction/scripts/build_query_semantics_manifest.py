#!/usr/bin/env python3
"""Build query-text-only semantic manifests and a separate offline parser audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.query_semantics import parse_query  # noqa: E402


def main() -> int:
    annotations_root = PROJECT_ROOT.parent / "crog_reproduction/OCID-VLG/refer/unique"
    output = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/query_semantics"
    output.mkdir(parents=True, exist_ok=True)
    audits = {}
    for split in ("train", "val", "test"):
        records = json.loads(
            (annotations_root / f"{split}_expressions.json").read_text(encoding="utf-8")
        )["data"]
        rows = []
        correct = 0
        relation_total = 0
        relation_detected = 0
        for record in records:
            parsed = parse_query(str(record["question"]))
            rows.append(
                {
                    "split": split,
                    "question_index": int(record["question_index"]),
                    "scene_id": str(record["image_filename"]),
                    **parsed.to_dict(),
                }
            )
            expected_category = str(record["target"]).rsplit("_", 1)[0]
            correct += int(parsed.target_category == expected_category)
            if "relation" in str(record["template_filename"]):
                relation_total += 1
                relation_detected += int(parsed.pairwise_relation is not None)
        frame = pd.DataFrame(rows)
        forbidden = set(frame.columns) & {
            "answer",
            "target",
            "target_instance_id",
            "answer_instance_value",
            "gt_mask",
        }
        if forbidden or bool(frame["uses_answer_instance"].any()):
            raise RuntimeError(f"query semantics leaked answer fields: {sorted(forbidden)}")
        frame.to_parquet(output / f"{split}_query_semantics.parquet", index=False)
        audits[split] = {
            "samples": len(frame),
            "target_category_correct_offline": correct,
            "target_category_accuracy_offline": correct / len(frame),
            "official_relation_templates": relation_total,
            "relation_detected": relation_detected,
            "relation_detection_recall": (
                relation_detected / relation_total if relation_total else None
            ),
            "query_types": frame["query_type"].value_counts().sort_index().to_dict(),
            "inference_manifest_has_answer_fields": False,
        }
    payload = {
        "schema_version": 1,
        "parser_version": "ocidvlg_lexical_v1",
        "audits": audits,
        "offline_audit_only_uses_public_target_category": True,
        "inference_parser_uses_query_text_only": True,
    }
    (output / "parser_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
