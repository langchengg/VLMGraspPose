#!/usr/bin/env python3
"""Apply a locked reranker bundle once to the formal-test candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.experiment_lock import (  # noqa: E402
    complete_formal_test_once,
    consume_formal_test_once,
    verify_lock,
)
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    validate_artifact_identity,
)
from src.grasping.reranking_v1.models import (  # noqa: E402
    GeometryGateConfig,
    RegularizedLinearRanker,
    TorchCandidateRanker,
    assert_feature_identity_invariant,
    attach_scores,
    geometry_gated_q_scores,
    geometry_risk,
    q_only_scores,
    q_softmask_rule_scores,
    validate_candidate_contract,
)
from src.grasping.reranking_v1.method_namespace import (  # noqa: E402
    FULL_NMS_INTERNAL_METHODS,
    GQCNN_TOP5_INTERNAL_ALIASES,
    TABULAR_FORMAL_METHOD_PROTOCOLS,
    public_method_name,
)
from src.grasping.reranking_v1.safe_switch import (  # noqa: E402
    SafeSwitchGate,
    apply_safe_switch,
    build_switch_features,
    candidate_predictions_from_switch,
)


FULL_METHODS = FULL_NMS_INTERNAL_METHODS
TOP5_ALIASES = GQCNN_TOP5_INTERNAL_ALIASES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table: {path}")


def _compact(scored: pd.DataFrame, *, protocol: str) -> pd.DataFrame:
    columns = [
        "sample_id",
        "scene_id",
        "candidate_id",
        "candidate_identity_sha256",
        "original_gqcnn_rank",
        "reranker_method",
        "reranker_score",
        "reranker_rank",
    ]
    output = scored.loc[:, columns].copy()
    output["reranker_method"] = output["reranker_method"].map(
        public_method_name
    )
    output["protocol"] = protocol
    return output


def _required_bundle_paths(
    training_root: Path, bundle: Mapping[str, Any]
) -> list[Path]:
    relative = [
        "inference_bundle.json",
        str(bundle["safe_switch_gate"]),
        str(bundle["safe_switch_selection"]),
        *map(str, bundle["learned_models"].values()),
    ]
    return [(training_root / value).resolve() for value in relative]


def _assert_paths_locked(
    lock: Mapping[str, Any], required: Sequence[Path]
) -> None:
    locked = {
        Path(item["path"]).resolve(): str(item["sha256"])
        for item in lock["artifacts"].values()
    }
    missing = [str(path) for path in required if path not in locked]
    if missing:
        raise ValueError(f"formal inference artifact is not experiment-locked: {missing}")
    changed = [
        str(path)
        for path in required
        if _sha256(path) != locked[path]
    ]
    if changed:
        raise ValueError(f"locked formal inference artifact changed: {changed}")


def _score_candidate_frame_once(
    frame: pd.DataFrame,
    *,
    training_root: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bundle_path = training_root / "inference_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_candidate_contract(frame, require_label=False)
    selected_features = tuple(map(str, bundle["feature_columns"]))
    missing = sorted(set(selected_features) - set(frame.columns))
    if missing:
        raise ValueError(f"locked inference features missing: {missing}")

    rule = bundle["rule_methods"]
    full: dict[str, pd.DataFrame] = {
        "q_only": attach_scores(frame, q_only_scores(frame), method="q_only"),
        "q_softmask_rule": attach_scores(
            frame,
            q_softmask_rule_scores(
                frame, **rule["q_softmask_rule"]
            ),
            method="q_softmask_rule",
        ),
        "geometry_gated_q": attach_scores(
            frame,
            geometry_gated_q_scores(
                frame, GeometryGateConfig(**rule["geometry_gated_q"])
            ),
            method="geometry_gated_q",
        ),
    }
    loaded: dict[str, RegularizedLinearRanker | TorchCandidateRanker] = {}
    for method, relative in bundle["learned_models"].items():
        path = training_root / str(relative)
        if method == "regularized_linear_ranker":
            model = RegularizedLinearRanker.load(path)
        else:
            model = TorchCandidateRanker.load(path, device=device)
        if tuple(model.feature_columns) != selected_features:
            raise ValueError(f"locked feature list mismatch for {method}")
        loaded[str(method)] = model
        full[str(method)] = attach_scores(
            frame, model.predict_scores(frame), method=str(method)
        )

    residual = full["residual_mlp"].copy()
    residual["geometry_risk"] = geometry_risk(residual)[0]
    switch_features = build_switch_features(residual)
    gate = SafeSwitchGate.load(training_root / str(bundle["safe_switch_gate"]))
    confidence = gate.predict_confidence(switch_features)
    selection = json.loads(
        (training_root / str(bundle["safe_switch_selection"])).read_text(
            encoding="utf-8"
        )
    )
    decisions = apply_safe_switch(
        switch_features,
        confidence,
        threshold=float(selection["threshold"]),
        force_no_switch=bool(selection["force_no_switch"]),
    )
    full["residual_mlp_safe_switch"] = candidate_predictions_from_switch(
        residual, decisions
    )

    full_parts: list[pd.DataFrame] = []
    for method in FULL_METHODS:
        scored = full[method]
        assert_feature_identity_invariant(frame, scored)
        full_parts.append(_compact(scored, protocol="full_nms"))

    top5_source = frame.loc[frame["original_gqcnn_rank"] <= 5].copy()
    top5_parts: list[pd.DataFrame] = []
    for source_method, alias in TOP5_ALIASES.items():
        if source_method == "q_only":
            values = q_only_scores(top5_source)
        else:
            values = loaded[source_method].predict_scores(top5_source)
        scored = attach_scores(top5_source, values, method=alias)
        assert_feature_identity_invariant(top5_source, scored)
        top5_parts.append(_compact(scored, protocol="gqcnn_top5"))
    predictions = pd.concat([*full_parts, *top5_parts], ignore_index=True)

    expected_full = set(
        zip(
            frame["sample_id"].astype(str),
            frame["candidate_id"].astype(str),
            strict=True,
        )
    )
    expected_top5 = set(
        zip(
            top5_source["sample_id"].astype(str),
            top5_source["candidate_id"].astype(str),
            strict=True,
        )
    )
    for (protocol, method), group in predictions.groupby(
        ["protocol", "reranker_method"], sort=False
    ):
        actual = set(
            zip(
                group["sample_id"].astype(str),
                group["candidate_id"].astype(str),
                strict=True,
            )
        )
        expected = expected_full if protocol == "full_nms" else expected_top5
        if actual != expected or len(group) != len(expected):
            raise AssertionError(f"candidate pool changed for {protocol}/{method}")
    actual_pairs = {
        (str(protocol), str(method))
        for protocol, method in predictions[
            ["protocol", "reranker_method"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    if actual_pairs != set(TABULAR_FORMAL_METHOD_PROTOCOLS):
        raise AssertionError(
            "tabular formal method/protocol namespace differs from the "
            "registered repeated-FiLM contract"
        )
    audit = {
        "methods": sorted(predictions["reranker_method"].unique()),
        "full_candidate_rows_per_method": len(frame),
        "top5_candidate_rows_per_method": len(top5_source),
        "candidate_identity_invariant": True,
        "candidate_pool_modified": False,
        "device": device,
        "device_requested": device,
        "resolved_devices": {
            method: (
                str(model.device)
                if isinstance(model, TorchCandidateRanker)
                else "cpu"
            )
            for method, model in loaded.items()
        },
        "safe_switch": {
            "threshold": float(selection["threshold"]),
            "force_no_switch": bool(selection["force_no_switch"]),
            "switch_count": int(decisions["switch_applied"].sum()),
        },
    }
    return predictions, decisions, audit


def score_candidate_frame(
    frame: pd.DataFrame,
    *,
    training_root: Path,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score a frame, falling back from automatic MPS to CPU on MPS failures."""

    try:
        return _score_candidate_frame_once(
            frame, training_root=training_root, device=device
        )
    except RuntimeError as error:
        lowered = str(error).lower()
        if device != "auto" or not any(
            token in lowered
            for token in ("mps", "not implemented", "unsupported", "placeholder")
        ):
            raise
        predictions, decisions, audit = _score_candidate_frame_once(
            frame, training_root=training_root, device="cpu"
        )
        audit["device_requested"] = device
        audit["device_fallback_reason"] = str(error)
        return predictions, decisions, audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--sample-universe", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--experiment-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = args.per_candidate.expanduser().resolve()
    universe_path = args.sample_universe.expanduser().resolve()
    training_root = args.training_root.expanduser().resolve()
    lock_path = args.experiment_lock.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    lock = verify_lock(lock_path)
    bundle = json.loads(
        (training_root / "inference_bundle.json").read_text(encoding="utf-8")
    )
    validate_artifact_identity(lock, context="formal experiment lock")
    validate_artifact_identity(bundle, context="formal inference bundle")
    required = [
        candidate_path,
        universe_path,
        *_required_bundle_paths(training_root, bundle),
    ]
    _assert_paths_locked(lock, required)
    if list(map(str, bundle["feature_columns"])) != list(
        map(str, lock["selected_feature_list"])
    ):
        raise ValueError("training bundle feature list disagrees with experiment lock")
    selection = json.loads(
        (training_root / str(bundle["safe_switch_selection"])).read_text()
    )
    validate_artifact_identity(
        selection, context="formal safe-switch selection"
    )
    if not np.isclose(
        float(selection["threshold"]),
        float(lock["safe_switch_threshold"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("safe-switch threshold disagrees with experiment lock")
    locked_force_no_switch = lock.get("ranking_parameters", {}).get(
        "safe_switch_force_no_switch"
    )
    if locked_force_no_switch is None or bool(locked_force_no_switch) != bool(
        selection["force_no_switch"]
    ):
        raise ValueError(
            "safe-switch force-no-switch decision is absent from or "
            "disagrees with the experiment lock ranking_parameters"
        )
    locked_device = str(
        lock.get("ranking_parameters", {}).get("tabular_inference_device", "")
    )
    if args.device != locked_device:
        raise ValueError(
            "formal inference device request disagrees with experiment lock"
        )

    universe = _read_table(universe_path)
    required_universe = {"sample_id", "scene_id"}
    if not required_universe <= set(universe.columns):
        raise ValueError("sample universe requires sample_id and scene_id")
    if universe["sample_id"].duplicated().any():
        raise ValueError("sample universe contains duplicate sample IDs")
    if universe["sample_id"].nunique() != int(lock["expected_test_sample_count"]):
        raise ValueError("formal-test sample count disagrees with experiment lock")
    frame = _read_table(candidate_path)
    validate_candidate_contract(
        frame, require_label=False, allowed_splits={"test"}
    )
    if not set(frame["sample_id"].astype(str)).issubset(
        set(universe["sample_id"].astype(str))
    ):
        raise ValueError("candidate samples are outside the locked sample universe")

    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--per-candidate",
        str(candidate_path),
        "--sample-universe",
        str(universe_path),
        "--training-root",
        str(training_root),
        "--experiment-lock",
        str(lock_path),
        "--output-root",
        str(output_root),
        "--device",
        args.device,
    ]
    consume_formal_test_once(
        lock_path, output_root=output_root, invocation=invocation
    )
    predictions_path = output_root / "reranker_predictions.parquet"
    manifest_path = output_root / "inference_manifest.json"
    if manifest_path.is_file():
        decisions_path = output_root / "safe_switch_decisions.parquet"
        if not predictions_path.is_file() or not decisions_path.is_file():
            raise FileExistsError("completed formal inference artifacts are incomplete")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_artifact_identity(
            existing, context="formal inference manifest"
        )
        if existing.get("lock_content_sha256") != lock["manifest_content_sha256"]:
            raise FileExistsError("existing formal output belongs to another lock")
        expected_manifest_values = {
            "per_candidate": str(candidate_path),
            "per_candidate_sha256": _sha256(candidate_path),
            "sample_universe": str(universe_path),
            "sample_universe_sha256": _sha256(universe_path),
            "training_root": str(training_root),
            "predictions": str(predictions_path),
            "safe_switch_decisions": str(decisions_path),
            "expected_test_sample_count": int(lock["expected_test_sample_count"]),
        }
        mismatched = sorted(
            key
            for key, expected in expected_manifest_values.items()
            if existing.get(key) != expected
        )
        if mismatched:
            raise ValueError(
                "existing formal inference manifest changed or belongs to "
                f"different inputs: {mismatched}"
            )
        for path, hash_key in (
            (predictions_path, "predictions_sha256"),
            (decisions_path, "safe_switch_decisions_sha256"),
        ):
            if not isinstance(existing.get(hash_key), str):
                raise ValueError(
                    f"existing formal inference manifest omits {hash_key}"
                )
            if _sha256(path) != existing[hash_key]:
                raise ValueError(f"existing formal inference artifact changed: {path}")
        complete_formal_test_once(lock_path, manifest_path=manifest_path)
        return existing

    predictions, decisions, audit = score_candidate_frame(
        frame, training_root=training_root, device=args.device
    )
    decisions_path = output_root / "safe_switch_decisions.parquet"
    predictions_temporary = output_root / f".predictions.{os.getpid()}.tmp.parquet"
    decisions_temporary = output_root / f".decisions.{os.getpid()}.tmp.parquet"
    predictions.to_parquet(
        predictions_temporary, index=False, compression="zstd"
    )
    decisions.to_parquet(
        decisions_temporary, index=False, compression="zstd"
    )
    os.replace(predictions_temporary, predictions_path)
    os.replace(decisions_temporary, decisions_path)
    manifest = {
        "schema_version": 1,
        **identity_payload(),
        "lock_path": str(lock_path),
        "lock_content_sha256": lock["manifest_content_sha256"],
        "per_candidate": str(candidate_path),
        "per_candidate_sha256": _sha256(candidate_path),
        "sample_universe": str(universe_path),
        "sample_universe_sha256": _sha256(universe_path),
        "training_root": str(training_root),
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "safe_switch_decisions": str(decisions_path),
        "safe_switch_decisions_sha256": _sha256(decisions_path),
        "expected_test_sample_count": int(lock["expected_test_sample_count"]),
        **audit,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    complete_formal_test_once(lock_path, manifest_path=manifest_path)
    (output_root / "run_command.txt").write_text(
        " ".join(map(shlex.quote, invocation)) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
