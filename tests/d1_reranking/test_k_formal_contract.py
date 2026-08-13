from __future__ import annotations

import ast
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from d1_reranking.execution import artifact_record
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from tools.d1_reranking import build_d1_formal_inputs as producer
from tools.unified_reranking import apply_locked_matrix_cell as unified_test_wrapper


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def _candidate_frame() -> pd.DataFrame:
    geometry = "a" * 64
    return pd.DataFrame(
        {
            "route": ["D1"],
            "sample_id": ["s0"],
            "candidate_id": ["c0"],
            "candidate_geometry_sha256": [geometry],
            "native_rank": [1],
            "native_score": [0.75],
            "cx_px": [10.0],
            "cy_px": [20.0],
            "theta_deg": [15.0],
            "width_px": [30.0],
            "height_px": [12.0],
        }
    )


def _formal_fixture(root: Path) -> Path:
    candidates_path = root / "02_candidates/test/top5.parquet"
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    _candidate_frame().to_parquet(candidates_path, index=False)
    _manifest(
        root / "02_candidates/test_manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {"top5": artifact_record(candidates_path)},
        },
    )
    paired_path = root / "01_manifests/d1_paired_manifest.parquet"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sample_id": ["s0", "s1"], "scene_id": ["scene0", "scene1"]}
    ).to_parquet(paired_path, index=False)
    return candidates_path


def test_formal_normalizer_is_label_free_and_preserves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    original = pd.read_parquet

    def guarded_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        source = str(path).lower()
        if "labels" in source or "ground_truth" in source:
            raise AssertionError("Test labels are forbidden")
        columns = kwargs.get("columns")
        if columns is not None and any(
            "success" in str(column).lower() or "jacquard" in str(column).lower()
            for column in columns
        ):
            raise AssertionError("Test label columns are forbidden")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    result = producer._write_normalized(
        root,
        name="d1_top5_r0",
        pool="top5",
        kind="R0",
        source_manifests={},
        score_record=None,
        decision_record=None,
        decision_key="selected_candidate_id",
        geometry_key=None,
        resume=False,
    )
    decisions = original(result["artifacts"]["per_sample_decisions"]["path"])
    no_output = decisions.loc[decisions["sample_id"].eq("s1")].iloc[0]
    assert no_output["selected_source_route"] == ""
    assert no_output["selected_candidate_id"] == ""
    assert no_output["selected_geometry_sha256"] == ""


def test_normalizer_resume_rejects_tamper_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    result = producer._write_normalized(
        root,
        name="d1_top5_r0",
        pool="top5",
        kind="R0",
        source_manifests={},
        score_record=None,
        decision_record=None,
        decision_key="selected_candidate_id",
        geometry_key=None,
        resume=False,
    )
    decisions_path = Path(result["artifacts"]["per_sample_decisions"]["path"])
    decisions_path.write_bytes(b"tampered")
    tampered_sha = sha256_file(decisions_path)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        producer._write_normalized(
            root,
            name="d1_top5_r0",
            pool="top5",
            kind="R0",
            source_manifests={},
            score_record=None,
            decision_record=None,
            decision_key="selected_candidate_id",
            geometry_key=None,
            resume=True,
        )
    assert sha256_file(decisions_path) == tampered_sha


def test_normalizer_rejects_selected_geometry_tamper() -> None:
    candidates = _candidate_frame()
    denominator = pd.DataFrame({"sample_id": ["s0"]})
    decisions = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "selected_candidate_id": ["c0"],
            "selected_geometry_sha256": ["b" * 64],
        }
    )
    with pytest.raises(RuntimeError, match="ID/geometry"):
        producer._normalized_decisions(
            denominator,
            candidates,
            decisions,
            selected_column="selected_candidate_id",
            geometry_column="selected_geometry_sha256",
        )


def test_unified_native_text_loader_never_invokes_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class Booster:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

        def num_feature(self) -> int:
            return 2

        def num_trees(self) -> int:
            return 1

        def predict(self, matrix: np.ndarray) -> np.ndarray:
            return np.asarray(matrix, dtype=float).sum(axis=1)

    monkeypatch.setattr(
        unified_test_wrapper, "_lightgbm", SimpleNamespace(Booster=Booster)
    )
    monkeypatch.setattr(
        pickle,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pickle.load is forbidden")
        ),
    )
    path = tmp_path / "model.txt"
    path.write_text("native model", encoding="utf-8")
    restored = unified_test_wrapper._load_native_lightgbm_ranker(path)
    np.testing.assert_array_equal(
        restored.predict(np.asarray([[1.0, 2.0]])), np.asarray([3.0])
    )
    assert calls == [{"model_file": str(path)}]


def test_unified_loader_rejects_pickle_without_native_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        unified_test_wrapper,
        "_lightgbm",
        SimpleNamespace(Booster=lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        pickle,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pickle.load is forbidden")
        ),
    )
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"payload": "not a native model"}))
    with pytest.raises(RuntimeError, match="exactly one native model string"):
        unified_test_wrapper._load_native_lightgbm_ranker(path)


def test_formal_producer_imports_lightgbm_before_torch_and_has_no_pickle_load() -> None:
    source = Path(producer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
    lightgbm_line = min(line for name, line in imports if name == "lightgbm")
    torch_line = min(line for name, line in imports if name == "torch")
    assert lightgbm_line < torch_line
    assert "pickle.load" not in source
    assert "candidate_success" not in source
