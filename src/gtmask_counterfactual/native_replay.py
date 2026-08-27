"""Label-free exact replay adapter for the frozen G1/C1 native pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from .execution import FROZEN_NATIVE_INFERENCE_SHA256, NATIVE_INFERENCE
from .io import artifact_record, atomic_json, atomic_parquet, canonical_sha256, sha256_file


class NativeReplayError(RuntimeError):
    """The label-free replay differs from the frozen native candidate table."""


class NativeReplayMissingError(NativeReplayError, FileNotFoundError):
    """A strict-resume shard does not exist yet."""


def load_frozen_native_module(path: str | Path = NATIVE_INFERENCE) -> ModuleType:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise NativeReplayError("frozen native inference source is absent or unsafe")
    if sha256_file(source) != FROZEN_NATIVE_INFERENCE_SHA256:
        raise NativeReplayError("frozen native inference source SHA-256 differs")
    specification = importlib.util.spec_from_file_location(
        "_gtmask_frozen_native_inference", source
    )
    if specification is None or specification.loader is None:
        raise NativeReplayError("cannot load frozen native inference source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def canonical_replay_candidates(frame: pd.DataFrame, *, route: str) -> pd.DataFrame:
    """Normalize either replay or frozen table to the exact comparison schema."""

    aliases = {
        "jaw_width_px": "width_px",
        "rectangle_height_px": "height_px",
    }
    result = frame.rename(columns=aliases).copy()
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise NativeReplayError(f"candidate table misses columns: {missing}")
    route_name = route.lower()
    if route_name not in {"g1", "c1"}:
        raise NativeReplayError(f"unsupported native replay route: {route}")
    if (
        "route" in result
        and not result["route"].astype(str).str.lower().eq(route_name).all()
    ):
        raise NativeReplayError("candidate route differs")
    if (
        "method" in result
        and not result["method"].astype(str).str.lower().eq(route_name).all()
    ):
        raise NativeReplayError("candidate method differs")
    result["sample_id"] = result["sample_id"].astype(str)
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["native_rank"] = pd.to_numeric(result["native_rank"], errors="raise").astype(
        int
    )
    if (
        result.duplicated(["sample_id", "candidate_id"]).any()
        or result.duplicated(["sample_id", "native_rank"]).any()
    ):
        raise NativeReplayError("candidate identities or native ranks are duplicated")
    for column in (
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if not result[column].map(math.isfinite).all():
            raise NativeReplayError(f"candidate {column} is non-finite")
    columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    return (
        result[columns]
        .sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def assert_exact_native_replay(
    replay: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    route: str,
) -> dict[str, Any]:
    """Compare exact keys/order and the frozen Parquet serializer values."""
    observed = canonical_replay_candidates(replay, route=route)
    expected = canonical_replay_candidates(frozen, route=route)
    keys = ["sample_id", "candidate_id", "native_rank"]
    if not observed[keys].equals(expected[keys]):
        raise NativeReplayError("candidate identity/rank replay differs")
    numeric = [
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    differences = {
        column: float((observed[column] - expected[column]).abs().max())
        for column in numeric
    }
    bad = {name: value for name, value in differences.items() if value != 0.0}
    if bad:
        raise NativeReplayError(f"candidate numeric replay differs: {bad}")
    return {
        "status": "PASS",
        "route": route.lower(),
        "candidate_count": len(observed),
        "sample_count_with_output": observed["sample_id"].nunique(),
        "serializer_atol": 0.0,
        "maximum_absolute_differences": differences,
        "raw_test_ground_truth_rows_read": 0,
    }


def _publish_exact_frame(path: Path, frame: pd.DataFrame) -> Path:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise NativeReplayError(f"predicted replay frame is unsafe: {path}")
        try:
            pd.testing.assert_frame_equal(
                pd.read_parquet(path), frame, check_dtype=False, check_exact=True
            )
        except AssertionError as error:
            raise NativeReplayError(
                f"existing predicted replay frame differs: {path}"
            ) from error
        return path
    return atomic_parquet(frame, path)


def write_canonical_replay_frames(
    run_dir: str | Path,
    *,
    route: str,
    sample_ids: Sequence[str],
    candidates: pd.DataFrame,
) -> tuple[Path, Path]:
    """Publish the exact label-free frames consumed by postprocess assembly."""

    route_name = route.lower()
    canonical = canonical_replay_candidates(candidates, route=route_name)
    canonical.insert(1, "route", route_name.upper())
    canonical.insert(2, "branch", "predicted")
    identities = [str(value) for value in sample_ids]
    if (
        len(identities) != len(set(identities))
        or any(not value for value in identities)
        or not set(canonical["sample_id"]).issubset(identities)
    ):
        raise NativeReplayError("canonical predicted replay denominator differs")
    counts = canonical.groupby("sample_id").size().reindex(identities, fill_value=0)
    per_sample = pd.DataFrame(
        {
            "sample_id": identities,
            "route": route_name.upper(),
            "branch": "predicted",
            "candidate_count": counts.to_numpy(dtype=int),
            "no_output": counts.eq(0).to_numpy(dtype=bool),
            "technical_failure": False,
            "status": np.where(counts.eq(0), "NO_OUTPUT", "COMPLETE"),
        }
    )
    output = Path(run_dir).expanduser().resolve() / "04_predicted_replay" / route_name
    candidate_path = _publish_exact_frame(output / "per_candidate.parquet", canonical)
    sample_path = _publish_exact_frame(output / "per_sample.parquet", per_sample)
    # Force both records now so callers cannot publish a manifest before bytes
    # are durably present.
    artifact_record(candidate_path)
    artifact_record(sample_path)
    return candidate_path, sample_path


def replay_label_free_samples(
    deployment_rows: Sequence[Mapping[str, Any]],
    *,
    route: str,
    module: ModuleType,
    model: Any,
    device: Any,
    config: Any,
    sample_loader: Any,
    sample_factory: Callable[..., Any],
    on_sample: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], None],
) -> None:
    """Drive the frozen inference primitive without constructing a label reader."""

    for deployment in deployment_rows:
        arrays = sample_loader.load(deployment, mask_source="predicted", labels=None)
        sample = sample_factory(
            sample_id=str(deployment["sample_id"]),
            rgb=arrays.rgb,
            depth_m=arrays.depth_m,
            predicted_mask=arrays.binary_mask,
            probability_map=arrays.probability,
            mask_source="predicted",
            oracle_mask=None,
            metadata={"scene_id": arrays.scene_id},
        )
        row, candidates = module.infer_one(
            method=route.lower(),
            model=model,
            device=device,
            config=config,
            sample=sample,
            raw_path=None,
        )
        on_sample(row, candidates)


def write_replay_sample(
    run_dir: str | Path,
    *,
    route: str,
    sample_id: str,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Path:
    """Publish one predicted-replay sample under its P3 namespace."""

    identity = str(sample_id).strip()
    if not identity:
        raise NativeReplayError("predicted replay sample_id is empty")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    destination = (
        Path(run_dir).expanduser().resolve()
        / "04_predicted_replay"
        / route.lower()
        / "samples"
        / digest[:2]
        / f"{digest}.json"
    )
    payload = {
        "sample_id": identity,
        "sample": dict(sample),
        "candidates": [dict(row) for row in candidates],
    }
    normalized = json.loads(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    )
    normalized["content_sha256"] = canonical_sha256(normalized)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise NativeReplayError(f"predicted replay shard is unsafe: {destination}")
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise NativeReplayError(
                f"predicted replay shard is unreadable: {destination}"
            ) from error
        if existing != normalized:
            raise NativeReplayError(
                f"existing predicted replay shard differs: {destination}"
            )
        return destination
    return atomic_json(destination, normalized)


def load_replay_sample(
    run_dir: str | Path, *, route: str, sample_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one self-hashed immutable sample shard for strict resume."""

    identity = str(sample_id).strip()
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    path = (
        Path(run_dir).expanduser().resolve()
        / "04_predicted_replay"
        / route.lower()
        / "samples"
        / digest[:2]
        / f"{digest}.json"
    )
    if path.is_symlink() or not path.is_file():
        raise NativeReplayMissingError(
            f"predicted replay shard is absent or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeReplayError(
            f"predicted replay shard is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict) or payload.get("sample_id") != identity:
        raise NativeReplayError("predicted replay shard sample identity differs")
    unsigned = dict(payload)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise NativeReplayError("predicted replay shard self hash differs")
    sample = payload.get("sample")
    candidates = payload.get("candidates")
    if (
        not isinstance(sample, dict)
        or not isinstance(candidates, list)
        or any(not isinstance(row, dict) for row in candidates)
    ):
        raise NativeReplayError("predicted replay shard schema differs")
    return dict(sample), [dict(row) for row in candidates]


__all__ = [
    "NativeReplayError",
    "NativeReplayMissingError",
    "assert_exact_native_replay",
    "canonical_replay_candidates",
    "load_replay_sample",
    "load_frozen_native_module",
    "replay_label_free_samples",
    "write_canonical_replay_frames",
    "write_replay_sample",
]
