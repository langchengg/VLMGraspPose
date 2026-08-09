#!/usr/bin/env python3
"""Run static and fail-closed runtime leakage audits."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.leakage_guard import RuntimeFileAccessGuard, static_scan  # noqa: E402


INFERENCE_MODULES = (
    PROJECT_ROOT / "src/segmentation/query_semantics.py",
    PROJECT_ROOT / "src/segmentation/camera_intrinsics.py",
    PROJECT_ROOT / "src/segmentation/sam3_proposal_generator.py",
    PROJECT_ROOT / "src/segmentation/sam3_text_proposals.py",
    PROJECT_ROOT / "src/segmentation/sam3_visual_proposals.py",
    PROJECT_ROOT / "src/segmentation/sam3_automatic_proposals.py",
    PROJECT_ROOT / "src/segmentation/proposal_features.py",
    PROJECT_ROOT / "src/segmentation/clip_candidate_features.py",
    PROJECT_ROOT / "src/segmentation/spatial_relation_features.py",
    PROJECT_ROOT / "src/segmentation/depth_mask_features.py",
    PROJECT_ROOT / "src/segmentation/second_stage_refiner.py",
    PROJECT_ROOT / "scripts/generate_sam3_proposal_bank.py",
    PROJECT_ROOT / "scripts/extract_clip_candidate_features.py",
    PROJECT_ROOT / "scripts/extract_proposal_features.py",
    PROJECT_ROOT / "scripts/generate_stage2_refinements.py",
    PROJECT_ROOT / "scripts/finalize_locked_sam3_candidates.py",
)


def main() -> int:
    output = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/leakage_audit"
    output.mkdir(parents=True, exist_ok=True)
    scans = {str(path): static_scan(path) for path in INFERENCE_MODULES}
    violations = {path: values for path, values in scans.items() if values}
    payload = {
        "status": "PASSED" if not violations else "FAILED",
        "modules": [str(path) for path in INFERENCE_MODULES],
        "violations": violations,
    }
    (output / "static_scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    forbidden_log = output / "forbidden_access_test.log"
    try:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ground_truth_mask.png"
            with RuntimeFileAccessGuard(output / "runtime_file_access.jsonl"):
                path.open("rb")
    except RuntimeError as error:
        forbidden_log.write_text(f"EXPECTED_BLOCK: {error}\n", encoding="utf-8")
    else:
        raise RuntimeError("runtime guard failed to block a GT path")
    split_source = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/splits/overlap_audit.json"
    (output / "split_overlap_audit.json").write_bytes(split_source.read_bytes())
    if violations:
        raise SystemExit(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
