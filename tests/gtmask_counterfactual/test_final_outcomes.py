from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gtmask_counterfactual import final_outcomes as module
from gtmask_counterfactual.io import artifact_record, atomic_json, atomic_parquet


def _source_frames(tmp_path: Path) -> tuple[Path, Path]:
    unified = (
        tmp_path
        / "unified"
        / "09_formal_test"
        / "per_sample_decisions.parquet"
    )
    d1 = (
        tmp_path
        / "d1"
        / "09_formal_test"
        / "formal_candidate_score_decision_bundle.parquet"
    )
    atomic_parquet(
        pd.DataFrame(
            [
                {
                    "sample_id": sample,
                    "system_name": system,
                    "selected_correct": sample == "s0",
                }
                for system in ("g1_gated_primary", "c1_gated_primary")
                for sample in ("s0", "s1")
            ]
        ),
        unified,
    )
    atomic_parquet(
        pd.DataFrame(
            [
                {
                    "sample_id": sample,
                    "system_name": system,
                    "selected_correct": sample == "s0",
                    "is_selected": sample == "s0",
                    "no_output": sample == "s1",
                }
                for system in (
                    "d1_top5_r7_gated",
                    "d1_top10_locked",
                    "d1_allnms_locked",
                )
                for sample in ("s0", "s1")
            ]
        ),
        d1,
    )
    return unified, d1


def test_materialize_final_outcomes_uses_only_locked_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_synthetic"
    lock = atomic_json(
        root / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json",
        {"status": "LOCKED"},
    )
    sample = atomic_parquet(
        pd.DataFrame({"sample_id": ["s0", "s1"]}),
        root / "02_sample_manifest/counterfactual_manifest.parquet",
    )
    unified, d1 = _source_frames(tmp_path)
    inventory = {
        (record["path"], record["sha256"], record["bytes"])
        for record in (artifact_record(unified), artifact_record(d1))
    }
    monkeypatch.setattr(
        module,
        "verify_protocol_lock",
        lambda _root: {"sample_manifest": artifact_record(sample)},
    )
    monkeypatch.setattr(module, "_locked_source_inventory", lambda _lock: inventory)

    def fake_authority(
        run_dir: Path,
        *,
        protocol_lock: Path,
        final_outcomes: dict[str, object],
        selector_sources: dict[str, dict[str, object]],
        routes: tuple[str, ...],
    ) -> Path:
        assert Path(run_dir) == root
        assert protocol_lock == lock
        assert routes == ("G1", "C1", "D1")
        assert selector_sources["G1"] == selector_sources["C1"]
        assert selector_sources["D1"] != selector_sources["G1"]
        assert final_outcomes == artifact_record(
            root / module.FINAL_OUTCOMES_RELATIVE_PATH
        )
        return atomic_json(
            root / "04_predicted_replay/FINAL_OUTCOMES_AUTHORITY.json",
            {"status": "LOCKED"},
        )

    monkeypatch.setattr(module, "write_final_outcomes_authority", fake_authority)
    output, authority = module.materialize_frozen_final_outcomes(
        root, expected_count=2
    )
    frame = pd.read_parquet(output).sort_values(["route", "sample_id"])
    assert frame.groupby("route").size().to_dict() == {"C1": 2, "D1": 2, "G1": 2}
    assert frame.groupby("route")["final_correct"].sum().to_dict() == {
        "C1": 1,
        "D1": 1,
        "G1": 1,
    }
    assert authority.is_file()

    module.materialize_frozen_final_outcomes(root, resume=True, expected_count=2)
    changed = pd.read_parquet(output)
    changed.loc[0, "final_correct"] = not bool(changed.loc[0, "final_correct"])
    changed.to_parquet(output, index=False)
    with pytest.raises(module.PostprocessContractError, match="differ"):
        module.materialize_frozen_final_outcomes(
            root, resume=True, expected_count=2
        )


def test_materialize_rejects_an_unlocked_formal_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_synthetic"
    sample = atomic_parquet(
        pd.DataFrame({"sample_id": ["s0", "s1"]}),
        root / "02_sample_manifest/counterfactual_manifest.parquet",
    )
    monkeypatch.setattr(
        module,
        "verify_protocol_lock",
        lambda _root: {"sample_manifest": artifact_record(sample)},
    )
    monkeypatch.setattr(module, "_locked_source_inventory", lambda _lock: set())
    with pytest.raises(module.PostprocessContractError, match="found 0"):
        module.materialize_frozen_final_outcomes(root, expected_count=2)
