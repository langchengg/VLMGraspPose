from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from failure_analysis.reranking_v3.schema import canonical_json, sha256_bytes, sha256_file
from failure_analysis.reranking_v3.subgroup_reporting import (
    METHODS,
    SUBGROUP_FIELDS,
    TRACKS,
    UNKNOWN,
    build_subgroup_report,
    main,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _checksum(sample_id: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{sample_id}:{candidate_id}".encode()).hexdigest()


def _synthetic_artifacts(tmp_path: Path) -> dict[str, object]:
    sample_ids = [f"multiple:val:{index:08d}" for index in range(1, 5)]
    candidate_records = []
    metadata_records = []
    corrected_records = []
    legacy_records = []
    ranking_records = {method: [] for method in METHODS}
    corrected_positive = ("c1", None, "c0", "c3")
    legacy_positive = ("c2", None, "c0", "c3")
    v2_tops = ("c1", "c0", "c1", "c3")
    v3_tops = ("c1", "c2", "c0", "c3")
    expression_families = ("relation", "name", "attribute", None)
    complexities = ("complex", "simple", "simple", None)
    grounding = (False, False, True, None)

    for row, sample_id in enumerate(sample_ids):
        ids = [f"c{index}" for index in range(5)]
        checksums = [_checksum(sample_id, candidate_id) for candidate_id in ids]
        candidates = [
            {
                "candidate_id": candidate_id,
                "candidate_checksum": checksum,
                "q_rank": index,
                "q_raw": 0.90 - 0.05 * index - 0.01 * row,
                "features": {
                    "mask_support": {"value": 0.10 + 0.16 * index},
                    "predicted_mask_uncertainty": 0.08 + 0.10 * index,
                    "angle_confidence": 0.20 + 0.15 * index,
                    "width_consistency": 0.30 + 0.12 * index,
                },
            }
            for index, (candidate_id, checksum) in enumerate(zip(ids, checksums, strict=True))
        ]
        candidate_records.append(
            {"split": "val", "sample_id": row + 1, "candidates": candidates}
        )
        metadata = {
            "sample_id": sample_id,
            "expression_family": expression_families[row],
            "token_length": (5, 2, 8, None)[row],
            "query_complexity": complexities[row],
            "depth_valid_fraction": (1.0, 0.0, 0.5, None)[row],
            "object_size": ("small", "medium", "large", None)[row],
            "candidate_metadata": [
                {
                    "candidate_id": candidate_id,
                    "candidate_checksum": checksum,
                    "mask_support": 0.10 + 0.16 * index,
                    "predicted_mask_uncertainty": 0.08 + 0.10 * index,
                    "angle_confidence": 0.20 + 0.15 * index,
                    "width_consistency": 0.30 + 0.12 * index,
                }
                for index, (candidate_id, checksum) in enumerate(
                    zip(ids, checksums, strict=True)
                )
            ],
        }
        if grounding[row] is not None:
            metadata["grounding_failure"] = grounding[row]
        metadata_records.append(metadata)

        def labels(positive: str | None) -> list[dict]:
            return [
                {
                    "candidate_id": candidate_id,
                    "candidate_checksum": checksum,
                    "candidate_correct": candidate_id == positive,
                }
                for candidate_id, checksum in zip(ids, checksums, strict=True)
            ]

        corrected_records.append(
            {"sample_id": sample_id, "candidate_labels": labels(corrected_positive[row])}
        )
        legacy_records.append(
            {"sample_id": sample_id, "candidate_labels": labels(legacy_positive[row])}
        )

        q_order = ids
        v2_order = [v2_tops[row], *[value for value in ids if value != v2_tops[row]]]
        v3_order = [v3_tops[row], *[value for value in ids if value != v3_tops[row]]]
        ranking_records["q_only"].append(
            {
                "sample_id": sample_id,
                "candidate_order": q_order,
                "candidate_checksum_ids": ids,
                "candidate_checksums": checksums,
            }
        )
        # The frozen V2 format has no checksum vector.  Exact candidate IDs are
        # still enforced and labels provide the checksum-bound evaluation join.
        ranking_records["v2_locked_primary"].append(
            {"sample_id": sample_id, "candidate_order": v2_order}
        )
        ranking_records["v3_locked_primary"].append(
            {
                "sample_id": sample_id,
                "candidate_order": v3_order,
                "candidate_checksum_ids": ids,
                "candidate_checksums": checksums,
                "candidate_probability_ids": ids,
                "uncertainty": [0.05 + 0.10 * index for index in range(5)],
                "candidate_evidence": [
                    {"mask_support": 0.10 + 0.16 * index} for index in range(5)
                ],
                "selection": {"selected_candidate_id": v3_order[0]},
            }
        )

    paths = {
        "candidates": _write_jsonl(tmp_path / "candidates.jsonl", candidate_records),
        "metadata": _write_jsonl(tmp_path / "metadata.jsonl", metadata_records),
        "corrected": _write_jsonl(tmp_path / "corrected.jsonl", corrected_records),
        "legacy": _write_jsonl(tmp_path / "legacy.jsonl", legacy_records),
        "q_only": _write_jsonl(tmp_path / "q.jsonl", ranking_records["q_only"]),
        "v2_locked_primary": _write_jsonl(
            tmp_path / "v2.jsonl", ranking_records["v2_locked_primary"]
        ),
        "v3_locked_primary": _write_jsonl(
            tmp_path / "v3.jsonl", ranking_records["v3_locked_primary"]
        ),
    }
    return {
        "paths": paths,
        "records": {
            "candidates": candidate_records,
            "metadata": metadata_records,
            "corrected": corrected_records,
            "legacy": legacy_records,
            "rankings": ranking_records,
        },
    }


def _build_kwargs(bundle: dict[str, object], output: Path) -> dict[str, object]:
    paths = bundle["paths"]
    assert isinstance(paths, dict)
    return {
        "candidates_path": paths["candidates"],
        "metadata_path": paths["metadata"],
        "corrected_labels_path": paths["corrected"],
        "legacy_labels_path": paths["legacy"],
        "rankings": {method: paths[method] for method in METHODS},
        "output_dir": output,
    }


def test_subgroup_report_is_dual_track_three_method_complete_and_hashed(tmp_path: Path) -> None:
    bundle = _synthetic_artifacts(tmp_path)
    output = tmp_path / "subgroups"
    result = build_subgroup_report(**_build_kwargs(bundle, output))

    assert result["status"] == "complete"
    assert result["sample_count"] == 4
    assert result["tracks"] == list(TRACKS)
    assert result["methods"] == list(METHODS)
    assert result["subgroup_fields"] == list(SUBGROUP_FIELDS)
    assert result["rows"]
    assert {row["track"] for row in result["rows"]} == set(TRACKS)
    assert {row["method"] for row in result["rows"]} == set(METHODS)
    assert {row["subgroup_field"] for row in result["rows"]} == set(SUBGROUP_FIELDS)

    json_path = output / "subgroup_metrics.json"
    csv_path = output / "subgroup_metrics.csv"
    provenance_path = output / "subgroup_provenance.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    content_hash = payload.pop("content_sha256")
    payload.pop("content_hash_scope")
    assert sha256_bytes(canonical_json(payload).encode()) == content_hash
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(result["rows"])
    assert csv_path.stat().st_size > 0 and provenance_path.stat().st_size > 0

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["analysis_boundary"].startswith("evaluation-only")
    assert provenance["subgroup_content_sha256"] == content_hash
    assert provenance["checksum_validation"]["corrected_labels"] is True
    assert provenance["checksum_validation"]["legacy_labels"] is True
    assert provenance["checksum_validation"]["analysis_metadata"] == "verified"
    assert provenance["checksum_validation"]["rankings"]["q_only"] == "verified"
    assert provenance["checksum_validation"]["rankings"]["v3_locked_primary"] == "verified"
    assert (
        provenance["checksum_validation"]["rankings"]["v2_locked_primary"]
        == "candidate_id_set_verified_checksum_not_declared"
    )
    for identity in provenance["inputs"].values():
        if isinstance(identity, dict) and "path" in identity:
            assert sha256_file(identity["path"]) == identity["sha256"]


def test_subgroup_semantics_preserve_unknown_and_failure_stages(tmp_path: Path) -> None:
    bundle = _synthetic_artifacts(tmp_path)
    result = build_subgroup_report(**_build_kwargs(bundle, tmp_path / "subgroups"))
    rows = result["rows"]

    def one(track: str, method: str, field: str, value: object) -> dict:
        matches = [
            row
            for row in rows
            if row["track"] == track
            and row["method"] == method
            and row["subgroup_field"] == field
            and row["subgroup_value"] == value
        ]
        assert len(matches) == 1
        return matches[0]

    switched = one(
        "corrected_scientific", "v3_locked_primary", "v2_switch", "switch"
    )
    assert switched["support"] == 3
    assert switched["reference_correct"] == 1
    assert switched["method_correct"] == 3
    assert switched["delta_vs_q"] == pytest.approx(2 / 3)

    candidate_set = one(
        "corrected_scientific",
        "v3_locked_primary",
        "candidate_set_failure",
        "failure",
    )
    assert candidate_set["support"] == 1 and candidate_set["method_correct"] == 0
    ranking = one(
        "corrected_scientific", "v3_locked_primary", "ranking_failure", "failure"
    )
    assert ranking["support"] == 2 and ranking["method_correct"] == 2
    assert one(
        "corrected_scientific",
        "q_only",
        "original_first_correct_rank",
        "none",
    )["support"] == 1
    assert one(
        "corrected_scientific",
        "v3_locked_primary",
        "corrected_legacy_disagreement",
        "disagree",
    )["support"] == 1

    unknown = one(
        "corrected_scientific", "v3_locked_primary", "expression_family", UNKNOWN
    )
    assert unknown["support"] == 1
    assert result["unknown_counts"]["corrected_scientific"]["expression_family"] == 1
    assert result["unknown_counts"]["corrected_scientific"]["grounding_failure"] == 1


def test_subgroup_output_is_immutable(tmp_path: Path) -> None:
    bundle = _synthetic_artifacts(tmp_path)
    output = tmp_path / "subgroups"
    kwargs = _build_kwargs(bundle, output)
    build_subgroup_report(**kwargs)
    original_hash = sha256_file(output / "subgroup_metrics.json")
    with pytest.raises(FileExistsError, match="immutable subgroup output"):
        build_subgroup_report(**kwargs)
    assert sha256_file(output / "subgroup_metrics.json") == original_hash


@pytest.mark.parametrize(
    "corruption",
    ["label_checksum", "ranking_checksum", "ranking_pool", "metadata_cohort"],
)
def test_subgroup_join_fails_closed_on_identity_corruption(
    tmp_path: Path, corruption: str
) -> None:
    bundle = _synthetic_artifacts(tmp_path)
    paths = bundle["paths"]
    records = bundle["records"]
    assert isinstance(paths, dict) and isinstance(records, dict)
    if corruption == "label_checksum":
        corrected = records["corrected"]
        assert isinstance(corrected, list)
        corrected[0]["candidate_labels"][0]["candidate_checksum"] = "wrong"
        _write_jsonl(paths["corrected"], corrected)
        match = "candidate checksum differs"
    elif corruption == "ranking_checksum":
        rankings = records["rankings"]
        assert isinstance(rankings, dict)
        rankings["v3_locked_primary"][0]["candidate_checksums"][0] = "wrong"
        _write_jsonl(paths["v3_locked_primary"], rankings["v3_locked_primary"])
        match = "candidate checksum differs"
    elif corruption == "ranking_pool":
        rankings = records["rankings"]
        assert isinstance(rankings, dict)
        rankings["v2_locked_primary"][0]["candidate_order"][0] = "unknown_candidate"
        _write_jsonl(paths["v2_locked_primary"], rankings["v2_locked_primary"])
        match = "changed the frozen candidate pool"
    else:
        metadata = records["metadata"]
        assert isinstance(metadata, list)
        _write_jsonl(paths["metadata"], metadata[:-1])
        match = "cohort differs"
    output = tmp_path / "subgroups"
    with pytest.raises(ValueError, match=match):
        build_subgroup_report(**_build_kwargs(bundle, output))
    assert not output.exists()


def test_independent_cli_entrypoint_uses_explicit_artifacts(tmp_path: Path, capsys) -> None:
    bundle = _synthetic_artifacts(tmp_path)
    paths = bundle["paths"]
    assert isinstance(paths, dict)
    output = tmp_path / "cli-subgroups"
    assert (
        main(
            [
                "--candidates",
                str(paths["candidates"]),
                "--metadata",
                str(paths["metadata"]),
                "--corrected-labels",
                str(paths["corrected"]),
                "--legacy-labels",
                str(paths["legacy"]),
                "--q-ranking",
                str(paths["q_only"]),
                "--v2-ranking",
                str(paths["v2_locked_primary"]),
                "--v3-ranking",
                str(paths["v3_locked_primary"]),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "complete"
    assert len(status["content_sha256"]) == 64
