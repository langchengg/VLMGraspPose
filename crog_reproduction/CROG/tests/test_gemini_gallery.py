from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw

from failure_analysis.gemini_crog_evidence_v1.gallery import (
    CATEGORY_REQUESTS,
    CONSENSUS_SAFE,
    ER2_DIRECT,
    ER2_MODEL,
    ER2_SAFE,
    FLASH_DIRECT,
    FLASH_MODEL,
    FLASH_SAFE,
    LOCKED_PRIMARY,
    Q_ONLY,
    build_evaluation_gallery,
    select_gallery_cases,
)


def _stable(sample_id: str, candidate: str) -> str:
    return f"{sample_id}/{candidate}"


def _outcome(
    sample_id: str,
    method: str,
    *,
    q_correct: bool,
    selected_correct: bool,
    selected: str = "candidate_1",
    switch: bool = True,
) -> dict:
    return {
        "sample_id": sample_id,
        "method": method,
        "selected_stable_candidate_id": _stable(sample_id, selected),
        "q_only_stable_candidate_id": _stable(sample_id, "candidate_0"),
        "legacy_q_only_correct": q_correct,
        "legacy_selected_correct": selected_correct,
        "corrected_q_only_correct": q_correct,
        "corrected_selected_correct": selected_correct,
        "switch": switch,
        "keep": not switch,
        "technical_fallback": False,
        "abstain": False,
    }


def _decision(
    sample_id: str,
    model: str,
    *,
    selected: str,
    abstain: bool = False,
    fallback: bool = False,
) -> dict:
    return {
        "sample_id": sample_id,
        "model_id": model,
        "protocol": "P1_full_crog_evidence",
        "replicate_id": 0,
        "selected_candidate_id": selected,
        "selected_stable_candidate_id": _stable(sample_id, selected),
        "abstain": abstain,
        "technical_fallback": fallback,
        "lifecycle_status": "TECHNICAL_FALLBACK" if fallback else "SUCCEEDED",
        "decision": "abstain" if abstain else "switch",
        "confidence": 0.85,
        "score_margin_top1_top2": 0.2,
        "global_reason_codes": ["target_alignment"],
    }


def test_preregistered_selection_is_deterministic_and_globally_deduplicated():
    outcomes = [
        _outcome("s1", ER2_DIRECT, q_correct=False, selected_correct=True),
        _outcome("s2", ER2_DIRECT, q_correct=True, selected_correct=False),
        _outcome("s3", FLASH_DIRECT, q_correct=False, selected_correct=True),
        _outcome("s4", FLASH_DIRECT, q_correct=True, selected_correct=False),
        _outcome("s5", CONSENSUS_SAFE, q_correct=False, selected_correct=False),
        _outcome("s9", ER2_DIRECT, q_correct=False, selected_correct=False),
        _outcome("s9", FLASH_DIRECT, q_correct=False, selected_correct=False),
        _outcome(
            "s10",
            LOCKED_PRIMARY,
            q_correct=True,
            selected_correct=True,
            selected="candidate_0",
            switch=False,
        ),
    ]
    decisions = [
        _decision("s6", ER2_MODEL, selected="candidate_1"),
        _decision("s6", FLASH_MODEL, selected="candidate_2"),
        _decision("s7", ER2_MODEL, selected="candidate_0", abstain=True),
        _decision("s8", FLASH_MODEL, selected="candidate_0", fallback=True),
    ]
    requested = {category: 1 for category in CATEGORY_REQUESTS}
    result = select_gallery_cases(
        per_method_outcomes=outcomes,
        per_model_decisions=decisions,
        evidence_sample_ids={f"s{index}" for index in range(1, 11)},
        requested_counts=requested,
    )
    assert [case["category"] for case in result["cases"]] == list(CATEGORY_REQUESTS)
    assert [case["sample_id"] for case in result["cases"]] == [
        f"s{index}" for index in range(1, 11)
    ]
    assert len({case["sample_id"] for case in result["cases"]}) == 10
    assert all(row["selected"] == 1 for row in result["categories"].values())


def test_selection_reports_only_true_within_category_shortfall():
    outcomes = [
        _outcome("s", ER2_DIRECT, q_correct=False, selected_correct=True),
        _outcome("s", FLASH_DIRECT, q_correct=False, selected_correct=True),
    ]
    result = select_gallery_cases(
        per_method_outcomes=outcomes,
        per_model_decisions=[],
        evidence_sample_ids={"s"},
        requested_counts={category: 2 for category in CATEGORY_REQUESTS},
    )
    assert result["categories"]["er2_recovered"] == {
        "requested": 2,
        "eligible_unique": 1,
        "available_unique_within_category": 1,
        "selected": 1,
        "shortfall": 1,
        "selected_sample_ids": ["s"],
    }
    assert result["categories"]["flash_recovered"]["eligible_unique"] == 1
    assert result["categories"]["flash_recovered"]["selected"] == 1
    assert result["categories"]["flash_recovered"]["shortfall"] == 1
    assert len(result["cases"]) == 2


def _write_fixture(tmp_path: Path, *, evaluation_overlay: bool = False):
    sample_id = "multiple:test:00000001"
    source = tmp_path / "evidence" / "shard_000000"
    source.mkdir(parents=True)
    board = source / "boards" / "sample_00000001.png"
    board.parent.mkdir()
    image = Image.new("RGB", (720, 720), "#FAFAF7")
    draw = ImageDraw.Draw(image)
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#56B4E9", "#A3BE8C"]
    labels = ["RGB + Top-5", "Predicted M", "Predicted Q", "Angle", "Width", "Candidates A-E"]
    for index, (color, label) in enumerate(zip(colors, labels, strict=True)):
        x = (index % 2) * 360
        y = (index // 2) * 240
        draw.rectangle((x, y, x + 359, y + 239), fill=color)
        draw.text((x + 16, y + 16), label, fill="white")
    image.save(board)
    board_hash = hashlib.sha256(board.read_bytes()).hexdigest()
    layers = {
        "sample_id": sample_id,
        "image_sha256": board_hash,
        "evaluation_overlay_included": evaluation_overlay,
        "panels": [
            "rgb",
            "predicted_m",
            "predicted_q",
            "predicted_angle",
            "predicted_width",
            "candidate_cards",
        ],
    }
    (board.with_suffix(".layers.json")).write_text(json.dumps(layers), encoding="utf-8")
    mapping = {
        "display_to_candidate": {
            letter: f"candidate_{index}" for index, letter in enumerate("ABCDE")
        },
        "candidate_to_display": {
            f"candidate_{index}": letter for index, letter in enumerate("ABCDE")
        },
    }
    request = {
        "sample_id": sample_id,
        "frame_id": "frame.png",
        "board_path": str(board),
        "board_sha256": board_hash,
        "metadata": "GT-free evidence",
        "mapping": mapping,
    }
    (source / "request_manifest.jsonl").write_text(
        json.dumps(request) + "\n", encoding="utf-8"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": sample_id,
                    "frame_id": "frame.png",
                    "language_instruction": "Pick up the blue cup",
                    "board_path": str(board),
                    "board_sha256": board_hash,
                }
            ]
        ),
        source / "sample_evidence.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": sample_id,
                    "candidate_id": f"candidate_{index}",
                    "stable_candidate_id": _stable(sample_id, f"candidate_{index}"),
                    "original_q_rank": index,
                }
                for index in range(5)
            ]
        ),
        source / "candidate_evidence.parquet",
    )
    evidence_index = tmp_path / "evidence_index.json"
    evidence_index.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "sample_id": sample_id,
                        "source": str(source),
                        "board_sha256": board_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    outcomes = [
        _outcome(
            sample_id,
            Q_ONLY,
            q_correct=False,
            selected_correct=False,
            selected="candidate_0",
            switch=False,
        ),
        _outcome(sample_id, ER2_DIRECT, q_correct=False, selected_correct=True),
        _outcome(sample_id, ER2_SAFE, q_correct=False, selected_correct=True),
        _outcome(sample_id, FLASH_DIRECT, q_correct=False, selected_correct=False),
        _outcome(sample_id, FLASH_SAFE, q_correct=False, selected_correct=False),
        _outcome(sample_id, CONSENSUS_SAFE, q_correct=False, selected_correct=True),
    ]
    decisions = [
        _decision(sample_id, ER2_MODEL, selected="candidate_1"),
        _decision(sample_id, FLASH_MODEL, selected="candidate_2"),
    ]
    outcomes_path = tmp_path / "per_method_outcomes.parquet"
    decisions_path = tmp_path / "per_model_decisions.parquet"
    pq.write_table(pa.Table.from_pylist(outcomes), outcomes_path)
    pq.write_table(pa.Table.from_pylist(decisions), decisions_path)
    return sample_id, board, board_hash, outcomes_path, decisions_path, evidence_index


def test_build_gallery_preserves_gt_free_board_and_separates_evaluation_panel(tmp_path):
    sample_id, board, board_hash, outcomes, decisions, evidence_index = _write_fixture(
        tmp_path
    )
    requested = {category: 0 for category in CATEGORY_REQUESTS}
    requested["er2_recovered"] = 1
    result = build_evaluation_gallery(
        per_method_outcomes_path=outcomes,
        per_model_decisions_path=decisions,
        evidence_index_path=evidence_index,
        output_dir=tmp_path / "gallery",
        requested_counts=requested,
    )
    assert result["case_count"] == 1
    case = result["cases"][0]
    assert case["sample_id"] == sample_id
    assert case["input_board_sha256_before"] == board_hash
    assert case["input_board_sha256_after"] == board_hash
    assert hashlib.sha256(board.read_bytes()).hexdigest() == board_hash
    assert case["input_board_evaluation_overlay_included"] is False
    assert case["evaluation_panel_in_composite"] is True
    assert Path(case["image_path"]).is_file()
    assert Image.open(case["image_path"]).width > Image.open(case["image_path"]).height
    assert result["safety"] == {
        "input_boards_byte_identical": True,
        "input_board_evaluation_overlay_included": False,
        "gt_only_in_separate_evaluation_panel": True,
        "cross_category_reuse_allowed": True,
        "samples_unique_within_each_category": True,
    }
    assert (tmp_path / "gallery/index.html").is_file()
    assert (tmp_path / "gallery/gallery.json").is_file()
    assert (tmp_path / "gallery/summary.md").is_file()


def test_gallery_fails_closed_if_board_layers_report_evaluation_overlay(tmp_path):
    _, _, _, outcomes, decisions, evidence_index = _write_fixture(
        tmp_path, evaluation_overlay=True
    )
    requested = {category: 0 for category in CATEGORY_REQUESTS}
    requested["er2_recovered"] = 1
    with pytest.raises(ValueError, match="evaluation overlay"):
        build_evaluation_gallery(
            per_method_outcomes_path=outcomes,
            per_model_decisions_path=decisions,
            evidence_index_path=evidence_index,
            output_dir=tmp_path / "gallery",
            requested_counts=requested,
        )


def test_duplicate_saved_outcome_is_rejected():
    row = _outcome("s", ER2_DIRECT, q_correct=False, selected_correct=True)
    with pytest.raises(ValueError, match="duplicate outcome"):
        select_gallery_cases(
            per_method_outcomes=[row, row],
            per_model_decisions=[],
            evidence_sample_ids={"s"},
        )
