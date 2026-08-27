"""Outcome-blind RGB-D near-duplicate map construction.

This module deliberately has no imports from prediction, metric, evaluator,
ranking, or result modules. Its only model-facing input is the locked formal
sample manifest, used to recover observation identities and source RGB-D paths.

The 64-bit pHash/dHash definitions are small local implementations informed by
Johannes Buchner's BSD-2-Clause ImageHash reference. No ImageHash code or
dependency is vendored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import scipy
from scipy.fft import dctn
from skimage.metrics import structural_similarity

from .common import (
    PREREGISTRATION_SHA256,
    atomic_frame,
    atomic_json,
    canonical_sha256,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)


FORMAL_MANIFESTS = {
    "train": "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_train.parquet",
    "validation": "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_validation.parquet",
    "test": "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet",
}
REQUIRED_COLUMNS = {
    "sample_id",
    "scene_id",
    "rgbd_pair_sha256",
    "source_rgb_path",
    "source_rgb_sha256",
    "source_depth_path",
    "source_depth_sha256",
    "split",
}
CANONICAL_SIZE = (256, 256)


def array_sha256(array: np.ndarray, *, invalid_mask: np.ndarray | None = None) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    if invalid_mask is not None:
        mask = np.ascontiguousarray(invalid_mask, dtype=np.uint8)
        digest.update(mask.tobytes())
        values = np.where(mask.astype(bool), contiguous, 0)
        digest.update(np.ascontiguousarray(values).tobytes())
    else:
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _bits_to_hex(bits: np.ndarray) -> str:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if flat.size != 64:
        raise ValueError("hash must contain exactly 64 bits")
    return f"{int(''.join(str(int(value)) for value in flat), 2):016x}"


def phash64(luminance: np.ndarray) -> str:
    image = cv2.resize(
        np.asarray(luminance, dtype=np.float32), (32, 32), interpolation=cv2.INTER_AREA
    )
    coefficients = dctn(image, type=2, norm="ortho")[:8, :8]
    return _bits_to_hex(coefficients > np.median(coefficients))


def dhash64(luminance: np.ndarray) -> str:
    image = cv2.resize(
        np.asarray(luminance, dtype=np.float32), (9, 8), interpolation=cv2.INTER_AREA
    )
    return _bits_to_hex(image[:, 1:] > image[:, :-1])


def hamming64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _split_scene(scene_id: str) -> tuple[str, str]:
    if "," not in scene_id:
        return scene_id, scene_id
    sequence, frame = scene_id.rsplit(",", 1)
    return sequence, frame


def _frame_number(frame_id: str) -> int | None:
    digits = "".join(character for character in frame_id if character.isdigit())
    return int(digits) if digits else None


def _decode_rgb(path: Path) -> np.ndarray:
    # Same primitives as OCIDVLGDataset.get_image_from_path.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot decode RGB: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _decode_depth_metres(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same depth rule as OCIDVLGDataset.get_depth_from_path: uint16 mm / 1000.
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot decode depth: {path}")
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"formal OCID depth must be uint16 millimetres: {path}")
    depth = raw.astype(np.float32) / np.float32(1000.0)
    valid = np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    return raw, depth, valid


def _luminance(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32)
    return (
        np.float32(0.299) * values[..., 0]
        + np.float32(0.587) * values[..., 1]
        + np.float32(0.114) * values[..., 2]
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_observation_manifest(repo: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    for split, relative in FORMAL_MANIFESTS.items():
        path = (repo / relative).resolve()
        identities[relative] = sha256_file(path)
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"split field mismatch in {path}")
        for observation_id, group in frame.groupby("rgbd_pair_sha256", sort=True):
            stable = [
                "scene_id",
                "source_rgb_path",
                "source_rgb_sha256",
                "source_depth_path",
                "source_depth_sha256",
            ]
            if any(group[column].astype(str).nunique() != 1 for column in stable):
                raise ValueError(f"inconsistent observation metadata: {observation_id}")
            first = group.iloc[0]
            sequence_id, frame_id = _split_scene(str(first["scene_id"]))
            rgb_path = Path(str(first["source_rgb_path"])).resolve()
            depth_path = Path(str(first["source_depth_path"])).resolve()
            if sha256_file(rgb_path) != str(first["source_rgb_sha256"]):
                raise ValueError(f"RGB file hash mismatch: {observation_id}")
            if sha256_file(depth_path) != str(first["source_depth_sha256"]):
                raise ValueError(f"depth file hash mismatch: {observation_id}")
            rgb = _decode_rgb(rgb_path)
            raw_depth, _, _ = _decode_depth_metres(depth_path)
            if rgb.shape[:2] != raw_depth.shape:
                raise ValueError(f"RGB/depth shape mismatch: {observation_id}")
            rows.append(
                {
                    "observation_id": str(observation_id),
                    "split": split,
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "frame_number": _frame_number(frame_id),
                    "scene_id": str(first["scene_id"]),
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "image_width": int(rgb.shape[1]),
                    "image_height": int(rgb.shape[0]),
                    "tuple_ids": json.dumps(sorted(group["sample_id"].astype(str))),
                    "number_of_queries": int(len(group)),
                    "source_manifest": str(path),
                    "source_rgb_file_sha256": str(first["source_rgb_sha256"]),
                    "source_depth_file_sha256": str(first["source_depth_sha256"]),
                }
            )
    result = pd.DataFrame(rows).sort_values(["split", "observation_id"]).reset_index(drop=True)
    expected = {"train": 26295, "validation": 3778, "test": 7675}
    counts = (
        result.groupby("split", sort=True)["number_of_queries"].sum().astype(int).to_dict()
    )
    if counts != expected:
        raise AssertionError(f"formal tuple denominators changed: {counts}")
    return result, identities


def _fingerprint_one(row: pd.Series, arrays_dir: Path) -> dict[str, Any]:
    rgb = _decode_rgb(Path(row["rgb_path"]))
    raw_depth, depth, valid = _decode_depth_metres(Path(row["depth_path"]))
    canonical_rgb = cv2.resize(rgb, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)
    canonical_luminance = _luminance(canonical_rgb).astype(np.float32)
    canonical_depth = cv2.resize(depth, CANONICAL_SIZE, interpolation=cv2.INTER_NEAREST)
    canonical_valid = cv2.resize(
        valid.astype(np.uint8), CANONICAL_SIZE, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    canonical_depth[~canonical_valid] = np.nan
    array_path = arrays_dir / f"{row['observation_id']}.npz"
    if not array_path.exists():
        _atomic_npz(
            array_path,
            rgb=canonical_rgb,
            luminance=canonical_luminance,
            depth=canonical_depth,
            valid=canonical_valid,
        )
    finite = canonical_depth[canonical_valid]
    return {
        "observation_id": row["observation_id"],
        "decoded_rgb_sha256": array_sha256(rgb),
        "decoded_depth_sha256": array_sha256(raw_depth),
        "canonical_rgb_sha256": array_sha256(canonical_rgb),
        "canonical_depth_sha256": array_sha256(
            np.nan_to_num(canonical_depth, nan=0.0), invalid_mask=canonical_valid
        ),
        "canonical_valid_sha256": array_sha256(canonical_valid.astype(np.uint8)),
        "phash64": phash64(canonical_luminance),
        "dhash64": dhash64(canonical_luminance),
        "median_valid_depth_m": float(np.median(finite)) if finite.size else np.nan,
        "valid_depth_fraction": float(canonical_valid.mean()),
        "canonical_array_path": str(array_path.resolve()),
        "canonical_array_sha256": sha256_file(array_path),
    }


def _hamming_to_all(value: str, candidates: np.ndarray) -> np.ndarray:
    target = np.uint64(int(value, 16))
    xor = np.bitwise_xor(candidates, target)
    bytes_view = xor.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(bytes_view, axis=1).sum(axis=1).astype(np.int16)


def retrieve_candidate_pairs(observations: pd.DataFrame, fingerprints: pd.DataFrame) -> pd.DataFrame:
    merged = observations.merge(fingerprints, on="observation_id", validate="one_to_one")
    train = merged[merged["split"] == "train"].sort_values("observation_id").reset_index(drop=True)
    test = merged[merged["split"] == "test"].sort_values("observation_id").reset_index(drop=True)
    train_hashes = np.asarray([int(value, 16) for value in train["phash64"]], dtype=np.uint64)
    by_sequence: dict[str, set[int]] = defaultdict(set)
    by_rgb: dict[str, set[int]] = defaultdict(set)
    by_depth: dict[str, set[int]] = defaultdict(set)
    for index, row in train.iterrows():
        by_sequence[str(row["sequence_id"])].add(int(index))
        by_rgb[str(row["decoded_rgb_sha256"])].add(int(index))
        by_depth[str(row["decoded_depth_sha256"])].add(int(index))
    pairs: list[dict[str, Any]] = []
    for _, test_row in test.iterrows():
        distances = _hamming_to_all(str(test_row["phash64"]), train_hashes)
        ordering = sorted(
            range(len(train)), key=lambda index: (int(distances[index]), str(train.iloc[index]["observation_id"]))
        )
        top20 = set(ordering[:20])
        same_sequence = by_sequence.get(str(test_row["sequence_id"]), set())
        exact_rgb = by_rgb.get(str(test_row["decoded_rgb_sha256"]), set())
        exact_depth = by_depth.get(str(test_row["decoded_depth_sha256"]), set())
        for train_index in sorted(top20 | same_sequence | exact_rgb | exact_depth):
            train_row = train.iloc[train_index]
            reasons = []
            if train_index in same_sequence:
                reasons.append("same_sequence")
            if train_index in top20:
                reasons.append("global_phash_top20")
            if train_index in exact_rgb:
                reasons.append("exact_rgb_hash")
            if train_index in exact_depth:
                reasons.append("exact_depth_hash")
            train_id = str(train_row["observation_id"])
            test_id = str(test_row["observation_id"])
            pairs.append(
                {
                    "pair_id": hashlib.sha256(f"{train_id}|{test_id}".encode()).hexdigest(),
                    "train_observation_id": train_id,
                    "test_observation_id": test_id,
                    "retrieval_reasons": ";".join(reasons),
                    "retrieval_phash_hamming": int(distances[train_index]),
                }
            )
    return pd.DataFrame(pairs).sort_values(["test_observation_id", "train_observation_id"]).reset_index(drop=True)


@lru_cache(maxsize=96)
def _load_arrays(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return tuple(np.asarray(payload[name]) for name in ("rgb", "luminance", "depth", "valid"))  # type: ignore[return-value]


def _aligned_views(
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, ...], float, float, float]:
    train_rgb, train_lum, train_depth, train_valid = train
    test_rgb, test_lum, test_depth, test_valid = test
    shift, response = cv2.phaseCorrelate(
        train_lum.astype(np.float32), test_lum.astype(np.float32)
    )
    dx, dy = float(shift[0]), float(shift[1])
    transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    size = (test_lum.shape[1], test_lum.shape[0])
    warped_rgb = cv2.warpAffine(
        train_rgb, transform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_lum = cv2.warpAffine(
        train_lum, transform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_depth = cv2.warpAffine(
        np.nan_to_num(train_depth, nan=0.0),
        transform,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_valid = cv2.warpAffine(
        train_valid.astype(np.uint8),
        transform,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    spatial = cv2.warpAffine(
        np.ones_like(train_valid, dtype=np.uint8),
        transform,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    ys, xs = np.nonzero(spatial)
    if not len(xs):
        # Phase correlation can return a shift outside the finite overlap for an
        # unrelated/repetitive pair. Preserve that shift (which necessarily
        # rejects both tiers) and expose unaligned views only so all requested
        # diagnostic fields remain serialisable; they cannot make the pair pass
        # because translation_magnitude exceeds the registered bounds.
        views = (
            train_rgb,
            test_rgb,
            train_lum,
            test_lum,
            train_depth,
            test_depth,
            train_valid,
            test_valid,
        )
        return views, dx, dy, float(response)
    crop = np.s_[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    views = (
        warped_rgb[crop],
        test_rgb[crop],
        warped_lum[crop],
        test_lum[crop],
        warped_depth[crop],
        test_depth[crop],
        warped_valid[crop],
        test_valid[crop],
    )
    return views, dx, dy, float(response)


def _pair_metrics(pair: pd.Series, lookup: pd.DataFrame) -> dict[str, Any]:
    train = lookup.loc[str(pair["train_observation_id"])]
    test = lookup.loc[str(pair["test_observation_id"])]
    train_arrays = _load_arrays(str(train["canonical_array_path"]))
    test_arrays = _load_arrays(str(test["canonical_array_path"]))
    views, dx, dy, response = _aligned_views(train_arrays, test_arrays)
    train_rgb, test_rgb, train_lum, test_lum, train_depth, test_depth, train_valid, test_valid = views
    # A large phase-correlation shift can leave a sliver smaller than SSIM's
    # minimum 7x7 window.  Such a pair is already outside both registered
    # translation bounds, so keep it as a diagnostic candidate and make the
    # non-estimable similarity explicit rather than fabricating padding.
    if min(train_lum.shape) < 7:
        lum_ssim = np.nan
    else:
        lum_ssim = float(
            structural_similarity(
                train_lum.astype(np.float32),
                test_lum.astype(np.float32),
                data_range=255.0,
            )
        )
    rgb_mae = float(np.mean(np.abs(train_rgb.astype(np.float32) - test_rgb.astype(np.float32))) / 255.0)
    joint = train_valid & test_valid & np.isfinite(train_depth) & np.isfinite(test_depth)
    valid_overlap = float(joint.mean())
    if joint.any():
        reference = max(
            float(np.median(np.concatenate([train_depth[joint], test_depth[joint]]))), 1e-6
        )
        errors = np.abs(train_depth[joint] - test_depth[joint]) / reference
        median_relative = float(np.median(errors))
        p95_relative = float(np.percentile(errors, 95))
    else:
        median_relative = np.nan
        p95_relative = np.nan
    exact_decoded = (
        str(train["decoded_rgb_sha256"]) == str(test["decoded_rgb_sha256"])
        and str(train["decoded_depth_sha256"]) == str(test["decoded_depth_sha256"])
    )
    exact_canonical = (
        str(train["canonical_rgb_sha256"]) == str(test["canonical_rgb_sha256"])
        and str(train["canonical_valid_sha256"]) == str(test["canonical_valid_sha256"])
        and str(train["canonical_depth_sha256"]) == str(test["canonical_depth_sha256"])
    )
    phash_distance = hamming64(str(train["phash64"]), str(test["phash64"]))
    dhash_distance = hamming64(str(train["dhash64"]), str(test["dhash64"]))
    magnitude = float(math.hypot(dx, dy))
    no_usable_depth = not bool(joint.any())
    exact = bool(exact_decoded or exact_canonical)
    strict_depth = (
        phash_distance <= 4
        and dhash_distance <= 6
        and magnitude <= 4.0
        and lum_ssim >= 0.990
        and rgb_mae <= 0.015
        and valid_overlap >= 0.95
        and median_relative <= 0.005
    )
    strict_rgb_only = (
        no_usable_depth
        and phash_distance <= 2
        and dhash_distance <= 3
        and magnitude <= 2.0
        and lum_ssim >= 0.995
        and rgb_mae <= 0.010
    )
    moderate_depth = (
        phash_distance <= 8
        and dhash_distance <= 10
        and magnitude <= 8.0
        and lum_ssim >= 0.980
        and rgb_mae <= 0.030
        and valid_overlap >= 0.90
        and median_relative <= 0.015
    )
    moderate_rgb_only = (
        no_usable_depth
        and phash_distance <= 4
        and dhash_distance <= 6
        and magnitude <= 4.0
        and lum_ssim >= 0.990
        and rgb_mae <= 0.015
    )
    train_number = train.get("frame_number")
    test_number = test.get("frame_number")
    frame_difference = (
        abs(int(test_number) - int(train_number))
        if pd.notna(train_number) and pd.notna(test_number)
        else np.nan
    )
    return {
        **pair.to_dict(),
        "train_sequence_id": str(train["sequence_id"]),
        "test_sequence_id": str(test["sequence_id"]),
        "train_frame_id": str(train["frame_id"]),
        "test_frame_id": str(test["frame_id"]),
        "phash_hamming": phash_distance,
        "dhash_hamming": dhash_distance,
        "luminance_ssim": lum_ssim,
        "normalised_rgb_mae": rgb_mae,
        "valid_depth_overlap": valid_overlap,
        "median_relative_depth_error": median_relative,
        "p95_relative_depth_error": p95_relative,
        "translation_dx_px": dx,
        "translation_dy_px": dy,
        "translation_magnitude_px": magnitude,
        "phase_response": response,
        "same_sequence": str(train["sequence_id"]) == str(test["sequence_id"]),
        "frame_index_difference": frame_difference,
        "exact_rgb_hash": str(train["decoded_rgb_sha256"]) == str(test["decoded_rgb_sha256"]),
        "exact_depth_hash": str(train["decoded_depth_sha256"]) == str(test["decoded_depth_sha256"]),
        "rgb_only_match": bool(no_usable_depth and (strict_rgb_only or moderate_rgb_only)),
        "exact_match": exact,
        "strict_match": bool(strict_depth or strict_rgb_only),
        "moderate_match": bool(moderate_depth or moderate_rgb_only),
    }


def _tier_exclusion(
    observations: pd.DataFrame, pairs: pd.DataFrame, tier: str
) -> pd.DataFrame:
    if tier == "exact":
        flagged = pairs[pairs["exact_match"]]
    elif tier == "strict":
        flagged = pairs[pairs["exact_match"] | pairs["strict_match"]]
    elif tier == "moderate":
        flagged = pairs[
            pairs["exact_match"] | pairs["strict_match"] | pairs["moderate_match"]
        ]
    else:
        raise ValueError(tier)
    test = observations[observations["split"] == "test"].set_index("observation_id")
    rows: list[dict[str, Any]] = []
    for observation_id, group in flagged.groupby("test_observation_id", sort=True):
        source = test.loc[str(observation_id)]
        tuple_ids = json.loads(str(source["tuple_ids"]))
        rows.append(
            {
                "tier": tier,
                "test_observation_id": str(observation_id),
                "sequence_id": str(source["sequence_id"]),
                "frame_id": str(source["frame_id"]),
                "test_tuple_ids": json.dumps(tuple_ids),
                "number_of_test_tuples": len(tuple_ids),
                "train_duplicate_ids": json.dumps(sorted(group["train_observation_id"].astype(str).unique())),
                "pair_ids": json.dumps(sorted(group["pair_id"].astype(str))),
                "removal_reason": f"new_preregistered_rgbd_{tier}_near_duplicate",
            }
        )
    return pd.DataFrame(rows, columns=[
        "tier", "test_observation_id", "sequence_id", "frame_id", "test_tuple_ids",
        "number_of_test_tuples", "train_duplicate_ids", "pair_ids", "removal_reason",
    ])


def _verify_locked_outputs(output: Path, lock: dict[str, Any]) -> None:
    if lock.get("status") != "LOCKED" or lock.get("protocol_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("duplicate map lock contract mismatch")
    for relative, expected in lock.get("output_sha256", {}).items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"locked duplicate output mismatch: {relative}")
    fingerprints_path = output / "fingerprints.parquet"
    if fingerprints_path.is_file():
        fingerprints = pd.read_parquet(
            fingerprints_path,
            columns=["canonical_array_path", "canonical_array_sha256"],
        )
        for row in fingerprints.itertuples(index=False):
            path = Path(str(row.canonical_array_path))
            if not path.is_file() or sha256_file(path) != str(row.canonical_array_sha256):
                raise RuntimeError(f"locked canonical array mismatch: {path}")


def build_duplicate_map(repo: Path, run_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    repo = repo.resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    output = run_dir / "duplicate_audit"
    lock_path = output / "DUPLICATE_MAP_LOCK.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        _verify_locked_outputs(output, lock)
        if resume:
            return {"status": "LOCKED", "resumed": True, **lock["counts"]}
        raise FileExistsError("duplicate map is immutable; use --resume to verify it")

    observations, input_hashes = build_observation_manifest(repo)
    atomic_frame(output / "canonical_observations.parquet", observations)
    arrays_dir = output / "canonical_arrays"
    fingerprint_rows = [_fingerprint_one(row, arrays_dir) for _, row in observations.iterrows()]
    fingerprints = pd.DataFrame(fingerprint_rows).sort_values("observation_id").reset_index(drop=True)
    atomic_frame(output / "fingerprints.parquet", fingerprints)

    retrieval = retrieve_candidate_pairs(observations, fingerprints)
    merged = observations.merge(fingerprints, on="observation_id", validate="one_to_one")
    lookup = merged.set_index("observation_id", drop=False)
    metric_rows = [_pair_metrics(pair, lookup) for _, pair in retrieval.iterrows()]
    all_pairs = pd.DataFrame(metric_rows).sort_values(
        ["test_observation_id", "phash_hamming", "train_observation_id"]
    ).reset_index(drop=True)
    atomic_frame(output / "all_candidate_pairs.parquet", all_pairs)
    exact_pairs = all_pairs[all_pairs["exact_match"]].copy()
    strict_pairs = all_pairs[all_pairs["strict_match"]].copy()
    moderate_pairs = all_pairs[all_pairs["moderate_match"]].copy()
    atomic_frame(output / "exact_pairs.csv", exact_pairs)
    atomic_frame(output / "strict_pairs.csv", strict_pairs)
    atomic_frame(output / "moderate_pairs.csv", moderate_pairs)
    exclusions: dict[str, pd.DataFrame] = {}
    for tier in ("exact", "strict", "moderate"):
        exclusions[tier] = _tier_exclusion(observations, all_pairs, tier)
        atomic_frame(output / f"test_observation_exclusion_{tier}.csv", exclusions[tier])

    output_names = [
        "canonical_observations.parquet",
        "fingerprints.parquet",
        "all_candidate_pairs.parquet",
        "exact_pairs.csv",
        "strict_pairs.csv",
        "moderate_pairs.csv",
        "test_observation_exclusion_exact.csv",
        "test_observation_exclusion_strict.csv",
        "test_observation_exclusion_moderate.csv",
    ]
    counts = {
        "unique_train_observations": int((observations["split"] == "train").sum()),
        "unique_validation_observations": int((observations["split"] == "validation").sum()),
        "unique_test_observations": int((observations["split"] == "test").sum()),
        "candidate_pairs": int(len(all_pairs)),
        "exact_pairs": int(len(exact_pairs)),
        "strict_pairs": int(len(strict_pairs)),
        "moderate_pairs": int(len(moderate_pairs)),
    }
    for tier, frame in exclusions.items():
        counts[f"{tier}_excluded_observations"] = int(len(frame))
        counts[f"{tier}_excluded_tuples"] = int(frame["number_of_test_tuples"].sum()) if len(frame) else 0
    lock = {
        "schema_version": 1,
        "status": "LOCKED",
        "map_kind": "new_independently_preregistered_rgbd_audit",
        "legacy_map_recovered": False,
        "previous_approximate_count_reproduced": False,
        "created_at": pd.Timestamp.now(tz="Europe/London").isoformat(),
        "protocol_sha256": PREREGISTRATION_SHA256,
        "protocol_definition_sha256": canonical_sha256({
            "canonical_size": CANONICAL_SIZE,
            "retrieval": ["same_sequence", "global_phash_top20", "exact_rgb_or_depth_hash"],
            "tiers": {
                "exact": "decoded RGB+depth identical OR canonical RGB+valid-mask+valid-depth identical",
                "strict": [4, 6, 4, 0.990, 0.015, 0.95, 0.005],
                "moderate": [8, 10, 8, 0.980, 0.030, 0.90, 0.015],
            },
        }),
        "input_manifest_sha256": input_hashes,
        "output_sha256": {name: sha256_file(output / name) for name in output_names},
        "thresholds": {
            "strict": {"phash_max": 4, "dhash_max": 6, "translation_px_max": 4, "ssim_min": 0.990, "rgb_mae_max": 0.015, "depth_overlap_min": 0.95, "median_relative_depth_error_max": 0.005},
            "moderate": {"phash_max": 8, "dhash_max": 10, "translation_px_max": 8, "ssim_min": 0.980, "rgb_mae_max": 0.030, "depth_overlap_min": 0.90, "median_relative_depth_error_max": 0.015},
        },
        "counts": counts,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
        },
    }
    # The lock is the terminal write. It is never rewritten by this function.
    atomic_json(lock_path, lock)
    return {"status": "LOCKED", "resumed": False, **counts}


def assert_builder_import_boundary() -> None:
    forbidden = {
        "prediction",
        "predictions",
        "metric",
        "metrics",
        "outcome",
        "outcomes",
        "recovered",
        "harmful",
        "formal_test",
    }
    names = set(globals())
    violations = sorted(name for name in names if name.lower() in forbidden)
    if violations:
        raise AssertionError(f"outcome-blind import boundary violated: {violations}")


def exclusion_tuple_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    result: set[str] = set()
    for value in frame.get("test_tuple_ids", pd.Series(dtype=str)).astype(str):
        result.update(str(item) for item in json.loads(value))
    return result


def stable_hash_order(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
