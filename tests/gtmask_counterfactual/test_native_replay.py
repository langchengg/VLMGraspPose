from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gtmask_counterfactual.native_replay import (
    NativeReplayError,
    assert_exact_native_replay,
    load_replay_sample,
    replay_label_free_samples,
    write_replay_sample,
)


def _candidate() -> dict[str, object]:
    return {
        "sample_id": "s1",
        "route": "g1",
        "candidate_id": "native_peak_001",
        "native_rank": 1,
        "native_score": 0.9,
        "cx_px": 10.0,
        "cy_px": 20.0,
        "theta_deg": 5.0,
        "width_px": 30.0,
        "height_px": 15.0,
    }


def test_exact_native_replay_rejects_identity_and_numeric_drift() -> None:
    frozen = pd.DataFrame([_candidate()])
    replay = frozen.rename(
        columns={"width_px": "jaw_width_px", "height_px": "rectangle_height_px"}
    ).copy()
    replay["method"] = "G1"
    assert assert_exact_native_replay(replay, frozen, route="g1")["status"] == "PASS"
    changed = replay.copy()
    changed.loc[0, "candidate_id"] = "other"
    with pytest.raises(NativeReplayError, match="identity"):
        assert_exact_native_replay(changed, frozen, route="g1")
    changed = replay.copy()
    changed.loc[0, "native_score"] += 1e-8
    with pytest.raises(NativeReplayError, match="numeric"):
        assert_exact_native_replay(changed, frozen, route="g1")


def test_label_free_driver_never_requests_labels() -> None:
    requests: list[object] = []

    class Loader:
        def load(self, row: object, *, mask_source: str, labels: object) -> object:
            requests.append(labels)
            assert mask_source == "predicted"
            return SimpleNamespace(
                rgb="rgb",
                depth_m="depth",
                binary_mask="mask",
                probability="prob",
                scene_id="scene",
            )

    class Module:
        @staticmethod
        def infer_one(
            **kwargs: object,
        ) -> tuple[dict[str, object], list[dict[str, object]]]:
            return {"sample_id": kwargs["sample"].sample_id}, []

    output: list[object] = []
    replay_label_free_samples(
        [{"sample_id": "s1"}],
        route="g1",
        module=Module(),
        model=object(),
        device=object(),
        config=object(),
        sample_loader=Loader(),
        sample_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        on_sample=lambda row, candidates: output.append((row, candidates)),
    )
    assert requests == [None]
    assert len(output) == 1


def test_predicted_replay_shard_is_strictly_resumable(tmp_path: Path) -> None:
    root = tmp_path
    sample = {"sample_id": "s1", "status": "ok"}
    candidates = [_candidate()]
    write_replay_sample(
        root,
        route="g1",
        sample_id="s1",
        sample=sample,
        candidates=candidates,
    )
    saved_sample, saved_candidates = load_replay_sample(
        root, route="g1", sample_id="s1"
    )
    assert saved_sample == sample
    assert saved_candidates == candidates
    changed = [dict(candidates[0], native_score=0.1)]
    with pytest.raises(NativeReplayError, match="differs"):
        write_replay_sample(
            root,
            route="g1",
            sample_id="s1",
            sample=sample,
            candidates=changed,
        )
