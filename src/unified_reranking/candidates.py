"""Freeze canonical order-only candidate pools without reading candidate labels."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_cache import common_asset_records
from .hashing import atomic_json, canonical_sha256, sha256_file


ROUTE_NAMES = {"CROG": "crog", "G1": "g1", "C1": "c1"}
GEOMETRY_COLUMNS = (
    "candidate_id",
    "native_rank",
    "native_score",
    "cx_px",
    "cy_px",
    "theta_deg",
    "width_px",
    "height_px",
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _fair_route_constants(fair_run: Path, route: str) -> dict[str, str]:
    source = fair_run / "02_predictions" / f"{ROUTE_NAMES[route]}_native_predictions.parquet"
    columns = [
        "checkpoint_sha256",
        "selected_config_sha256",
        "native_decoder_config_sha256",
    ]
    first = pd.read_parquet(source, columns=columns).iloc[0]
    values = {name: str(first[name]) for name in columns}
    if any(value in {"", "None", "nan"} for value in values.values()):
        raise ValueError(f"incomplete fair candidate lineage constants for {route}")
    return values


def _normalise(frame: pd.DataFrame, *, route: str, split: str, constants: dict[str, str]) -> pd.DataFrame:
    required = {"sample_id", *GEOMETRY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate columns missing for {route}/{split}: {missing}")
    result = frame.loc[:, ["sample_id", *GEOMETRY_COLUMNS]].copy()
    result.insert(1, "frame_id", "")
    result.insert(2, "scene_id", "")
    result.insert(3, "route", route)
    result.insert(4, "split", split)
    result["native_rank"] = result["native_rank"].astype("int16")
    result["valid"] = True
    result["source_checkpoint_sha256"] = constants["checkpoint_sha256"]
    result["generator_config_sha256"] = constants["native_decoder_config_sha256"]
    result["selected_config_sha256"] = constants["selected_config_sha256"]
    result["candidate_geometry_sha256"] = [
        canonical_sha256(
            [
                route,
                sample_id,
                candidate_id,
                int(rank),
                *[float(value) for value in values],
            ]
        )
        for sample_id, candidate_id, rank, values in zip(
            result["sample_id"],
            result["candidate_id"],
            result["native_rank"],
            result[["cx_px", "cy_px", "theta_deg", "width_px", "height_px"]].to_numpy(),
            strict=True,
        )
    ]
    if result[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError(f"duplicate candidate identity for {route}/{split}")
    finite = np.isfinite(
        result[["native_score", "cx_px", "cy_px", "theta_deg", "width_px", "height_px"]]
        .to_numpy(float)
    )
    if not finite.all() or (result[["width_px", "height_px"]].to_numpy(float) <= 0).any():
        raise ValueError(f"invalid candidate geometry for {route}/{split}")
    return result


def _attach_context(candidates: pd.DataFrame, paired: pd.DataFrame, *, route: str, split: str) -> pd.DataFrame:
    context = paired[["sample_id", "frame_id", "scene_id"]]
    result = candidates.drop(columns=["frame_id", "scene_id"]).merge(
        context, on="sample_id", how="left", validate="many_to_one"
    )
    if result[["frame_id", "scene_id"]].isna().any().any():
        raise ValueError(f"candidate/sample context mismatch for {route}/{split}")
    leading = ["sample_id", "frame_id", "scene_id", "route", "split"]
    return result.loc[:, leading + [c for c in result.columns if c not in leading]]


def _load_fair(fair_run: Path, route: str, split: str, paired: pd.DataFrame) -> pd.DataFrame:
    if split != "test":
        raise ValueError("the finalized fair source contains G1/C1 candidates for Test only")
    source = fair_run / "02_predictions" / f"{ROUTE_NAMES[route]}_native_predictions.parquet"
    original = pd.read_parquet(source)
    rename = {"jaw_width_px": "width_px", "rectangle_height_px": "height_px"}
    normal = _normalise(original.rename(columns=rename), route=route, split=split, constants=_fair_route_constants(fair_run, route))
    return _attach_context(normal, paired, route=route, split=split)


def _load_crog_development(
    old_features: Path,
    paired: pd.DataFrame,
    fair_run: Path,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_name = {"train": "train", "validation": "val", "test": "test"}[split]
    source = old_features / f"candidates_crog_frozen_top5_{source_name}.parquet"
    columns = [
        "sample_id", "scene_id", "language_instruction", "image_path", "depth_path",
        "candidate_id", "original_rank", "q_raw", "x_px", "y_px", "angle_rad",
        "width_px", "height_px",
    ]
    old = pd.read_parquet(source, columns=columns)
    old["question_index"] = old["sample_id"].str.rsplit(":", n=1).str[-1].astype("int64")
    context = paired[
        ["sample_id", "question_index", "scene_id", "language", "source_rgb_path", "source_depth_path"]
    ]
    joined = old.drop(columns="sample_id").merge(
        context, on="question_index", how="inner", validate="many_to_one", suffixes=("_old", "_paired")
    )
    expected = len(paired) * 5
    if len(joined) != expected:
        raise ValueError(f"CROG {split} expected {expected} candidates, observed {len(joined)}")
    mismatch = {
        "scene": int((joined["scene_id_old"] != joined["scene_id_paired"]).sum()),
        "language": int((joined["language_instruction"] != joined["language"]).sum()),
        "rgb_path": int((joined["image_path"] != joined["source_rgb_path"]).sum()),
        "depth_path": int((joined["depth_path"] != joined["source_depth_path"]).sum()),
    }
    if any(mismatch.values()):
        raise ValueError(f"CROG/source identity mismatch for {split}: {mismatch}")
    candidate = joined.rename(
        columns={
            "original_rank": "native_rank",
            "q_raw": "native_score",
            "x_px": "cx_px",
            "y_px": "cy_px",
        }
    )
    candidate["theta_deg"] = np.degrees(candidate["angle_rad"].astype(float))
    normal = _normalise(candidate, route="CROG", split=split, constants=_fair_route_constants(fair_run, "CROG"))
    normal = _attach_context(normal, paired, route="CROG", split=split)
    return normal, {"source": str(source.resolve()), "source_sha256": sha256_file(source), "identity_mismatches": mismatch}


def _verify_crog_test_exact(crog: pd.DataFrame, fair_run: Path) -> dict[str, Any]:
    fair = _load_fair(
        fair_run,
        "CROG",
        "test",
        pd.DataFrame(crog[["sample_id", "frame_id", "scene_id"]].drop_duplicates()),
    )
    columns = ["sample_id", *GEOMETRY_COLUMNS]
    left = crog[columns].sort_values(["sample_id", "native_rank"]).reset_index(drop=True)
    right = fair[columns].sort_values(["sample_id", "native_rank"]).reset_index(drop=True)
    if not left[["sample_id", "candidate_id", "native_rank"]].equals(right[["sample_id", "candidate_id", "native_rank"]]):
        raise ValueError("historical CROG Test candidate identity differs from fair source")
    maximum = {}
    for column in ("native_score", "cx_px", "cy_px", "theta_deg", "width_px", "height_px"):
        difference = np.abs(left[column].to_numpy(float) - right[column].to_numpy(float))
        maximum[column] = float(difference.max(initial=0.0))
    if any(value > 1e-12 for value in maximum.values()):
        raise ValueError(f"historical CROG candidate values differ from fair source: {maximum}")
    return {"rows": len(left), "maximum_absolute_difference": maximum, "match": True}


def _pool_descriptor(frame: pd.DataFrame, path: Path, all_samples: list[str]) -> dict[str, Any]:
    counts = frame.groupby("sample_id", sort=False).size().reindex(all_samples, fill_value=0)
    grouped = {
        str(sample_id): rows.sort_values("native_rank")
        for sample_id, rows in frame.groupby("sample_id", sort=False)
    }
    contracts = []
    for sample_id in all_samples:
        rows = grouped.get(str(sample_id))
        contracts.append(
            [
                sample_id,
                [
                    [
                        row.candidate_id,
                        int(row.native_rank),
                        float(row.native_score),
                        float(row.cx_px),
                        float(row.cy_px),
                        float(row.theta_deg),
                        float(row.width_px),
                        float(row.height_px),
                    ]
                    for row in (() if rows is None else rows.itertuples(index=False))
                ],
            ]
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "samples": len(all_samples),
        "samples_with_candidates": int((counts > 0).sum()),
        "no_output_samples": int((counts == 0).sum()),
        "max_candidates": int(counts.max()),
        "membership_geometry_score_sha256": canonical_sha256(contracts),
    }


def _verified_source(path_value: Any, digest_value: Any, label: str) -> Path:
    path = Path(str(path_value)).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"candidate source lineage file is missing: {label}")
    digest = str(digest_value)
    if sha256_file(path) != digest:
        raise ValueError(f"candidate source lineage hash mismatch: {label}")
    return path


def _validate_geometry_hashes(frame: pd.DataFrame, *, route: str, split: str) -> None:
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
        "candidate_geometry_sha256",
        "route",
        "split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"canonical candidate pool misses provenance columns: {route}/{split}: {missing}")
    if (
        set(frame["route"].astype(str)) != {route}
        or set(frame["split"].astype(str)) != {split}
        or frame[["sample_id", "candidate_id"]].duplicated().any()
    ):
        raise ValueError(f"canonical candidate identity route/split mismatch: {route}/{split}")
    expected = [
        canonical_sha256(
            [
                route,
                row.sample_id,
                row.candidate_id,
                int(row.native_rank),
                float(row.cx_px),
                float(row.cy_px),
                float(row.theta_deg),
                float(row.width_px),
                float(row.height_px),
            ]
        )
        for row in frame.itertuples(index=False)
    ]
    if frame["candidate_geometry_sha256"].astype(str).tolist() != expected:
        raise ValueError(f"canonical candidate geometry hash mismatch: {route}/{split}")


def _candidate_contract_rows(frame: pd.DataFrame) -> list[list[Any]]:
    columns = ["sample_id", *GEOMETRY_COLUMNS, "candidate_geometry_sha256"]
    ordered = frame[columns].sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    )
    return [
        [
            str(row.sample_id),
            str(row.candidate_id),
            int(row.native_rank),
            *[
                float(value)
                for value in (
                    row.native_score,
                    row.cx_px,
                    row.cy_px,
                    row.theta_deg,
                    row.width_px,
                    row.height_px,
                )
            ],
            str(row.candidate_geometry_sha256),
        ]
        for row in ordered.itertuples(index=False)
    ]


def _assert_candidate_source_match(
    canonical: pd.DataFrame, expected: pd.DataFrame, *, route: str, split: str
) -> None:
    _validate_geometry_hashes(canonical, route=route, split=split)
    _validate_geometry_hashes(expected, route=route, split=split)
    if _candidate_contract_rows(canonical) != _candidate_contract_rows(expected):
        raise ValueError(f"canonical candidate pool differs from audited source: {route}/{split}")


def _fair_source_audit(run_dir: Path) -> tuple[dict[str, Any], Path]:
    path = run_dir / "00_audit" / "fair_source_audit.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("fair source audit is missing for candidate provenance")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("status") != "PASS":
        raise ValueError("fair source audit is not PASS")
    fair_run = Path(str(audit.get("fair_run", ""))).resolve()
    if not fair_run.is_dir():
        raise ValueError("fair source audit has no available fair run")
    return audit, fair_run


def _audited_fair_candidate_source(
    audit: dict[str, Any], *, route: str
) -> Path:
    key = f"{route.lower()}_candidates"
    record = dict(audit.get("artifacts", {})).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"fair source audit does not bind {key}")
    return _verified_source(record.get("path"), record.get("sha256"), key)


def _load_crog_lineage_source(
    source: Path, paired: pd.DataFrame, fair_run: Path, split: str
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "scene_id",
        "language_instruction",
        "image_path",
        "depth_path",
        "candidate_id",
        "original_rank",
        "q_raw",
        "x_px",
        "y_px",
        "angle_rad",
        "width_px",
        "height_px",
    ]
    old = pd.read_parquet(source, columns=columns)
    old["question_index"] = old["sample_id"].astype(str).str.rsplit(":", n=1).str[-1].astype(
        "int64"
    )
    context = paired[
        [
            "sample_id",
            "question_index",
            "scene_id",
            "language",
            "source_rgb_path",
            "source_depth_path",
        ]
    ]
    joined = old.drop(columns="sample_id").merge(
        context,
        on="question_index",
        how="inner",
        validate="many_to_one",
        suffixes=("_old", "_paired"),
    )
    if joined.empty:
        raise ValueError(f"CROG audited source/sample coverage mismatch: {split}")
    mismatch = (
        (joined["scene_id_old"] != joined["scene_id_paired"])
        | (joined["language_instruction"] != joined["language"])
        | (joined["image_path"] != joined["source_rgb_path"])
        | (joined["depth_path"] != joined["source_depth_path"])
    )
    if mismatch.any():
        raise ValueError(f"CROG audited source context mismatch: {split}")
    candidate = joined.rename(
        columns={
            "original_rank": "native_rank",
            "q_raw": "native_score",
            "x_px": "cx_px",
            "y_px": "cy_px",
        }
    )
    candidate["theta_deg"] = np.degrees(candidate["angle_rad"].astype(float))
    expected = _normalise(
        candidate,
        route="CROG",
        split=split,
        constants=_fair_route_constants(fair_run, "CROG"),
    )
    return _attach_context(expected, paired, route="CROG", split=split)


def _load_generated_lineage_source(
    *,
    run_dir: Path,
    route: str,
    split: str,
    paired: pd.DataFrame,
    fair_run: Path,
    lineage: dict[str, Any],
) -> pd.DataFrame:
    expected_root = run_dir / "02_candidates" / "native_work" / f"{route.lower()}_{split}"
    source = _verified_source(lineage.get("source"), lineage.get("source_sha256"), f"{route}/{split}")
    manifest_path = _verified_source(
        lineage.get("run_manifest"), lineage.get("run_manifest_sha256"), f"{route}/{split} manifest"
    )
    if source != (expected_root / "candidates.parquet").resolve() or manifest_path != (
        expected_root / "run_manifest.json"
    ).resolve():
        raise ValueError(f"generated candidate lineage path mismatch: {route}/{split}")
    manifest = _verify_generated_run_manifest(manifest_path, expected_root)
    constants = _fair_route_constants(fair_run, route)
    expected_lineage = {
        "checkpoint_sha256": constants["checkpoint_sha256"],
        "selected_config_sha256": constants["selected_config_sha256"],
        "native_decoder_config_sha256": constants["native_decoder_config_sha256"],
    }
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("tag") != "formal"
        or manifest.get("locked_inference_source_sha256")
        != lineage.get("locked_inference_source_sha256")
        or any(str(manifest.get(key)) != value for key, value in expected_lineage.items())
    ):
        raise ValueError(f"generated candidate run-manifest lineage mismatch: {route}/{split}")
    original = pd.read_parquet(source)
    if any(set(original[key].astype(str)) != {value} for key, value in expected_lineage.items()):
        raise ValueError(f"generated candidate row lineage mismatch: {route}/{split}")
    expected = _normalise(
        original.rename(
            columns={"jaw_width_px": "width_px", "rectangle_height_px": "height_px"}
        ),
        route=route,
        split=split,
        constants=constants,
    )
    return _attach_context(expected, paired, route=route, split=split)


def _verify_generated_artifact(
    record: Any, expected_path: Path, *, name: str
) -> int:
    if not isinstance(record, dict):
        raise ValueError(f"generated candidate {name} artifact record is missing")
    path = expected_path.resolve()
    if (
        Path(str(record.get("path", ""))).resolve() != path
        or not path.is_file()
        or path.is_symlink()
        or record.get("sha256") != sha256_file(path)
        or record.get("bytes") != path.stat().st_size
    ):
        raise ValueError(f"generated candidate {name} artifact is stale")
    rows = int(pd.read_parquet(path).shape[0])
    if record.get("rows") != rows:
        raise ValueError(f"generated candidate {name} row count is stale")
    return rows


def _verify_generated_run_manifest(
    manifest_path: Path, expected_root: Path
) -> dict[str, Any]:
    """Verify the merged inference output and every content-addressed shard."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "COMPLETE":
        raise ValueError("generated candidate run manifest is not COMPLETE")
    code_records = manifest.get("inference_code_records")
    if not isinstance(code_records, list) or not code_records:
        raise ValueError("generated candidate inference code inventory is missing")
    for record in code_records:
        if not isinstance(record, dict):
            raise ValueError("generated candidate inference code record is malformed")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or path.is_symlink() or record.get("sha256") != sha256_file(path):
            raise ValueError("generated candidate inference code source is stale")
    if manifest.get("inference_code_bundle_sha256") != canonical_sha256(code_records):
        raise ValueError("generated candidate inference code bundle hash is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "per_sample",
        "candidates",
    }:
        raise ValueError("generated candidate merged artifact inventory is invalid")
    sample_rows = _verify_generated_artifact(
        artifacts["per_sample"], expected_root / "per_sample.parquet", name="per-sample"
    )
    candidate_rows = _verify_generated_artifact(
        artifacts["candidates"], expected_root / "candidates.parquet", name="candidates"
    )
    if (
        manifest.get("sample_count") != sample_rows
        or manifest.get("candidate_count") != candidate_rows
    ):
        raise ValueError("generated candidate merged row counts are inconsistent")
    chunks = manifest.get("chunk_manifests")
    if (
        not isinstance(chunks, list)
        or len(chunks) != int(manifest.get("shards", -1))
        or not chunks
    ):
        raise ValueError("generated candidate shard inventory is incomplete")
    cursor = 0
    chunk_candidate_rows = 0
    seen: set[Path] = set()
    source_sample_path = Path(str(manifest.get("source_samples", ""))).resolve()
    source_rows = pd.read_parquet(source_sample_path).to_dict("records")[:sample_rows]
    asset_cache: dict[str, str] = {}
    for index, record in enumerate(chunks):
        if not isinstance(record, dict):
            raise ValueError("generated candidate shard record is malformed")
        chunk_path = Path(str(record.get("path", ""))).resolve()
        shard_root = (expected_root / "shards").resolve()
        if (
            chunk_path in seen
            or chunk_path.parent.parent != shard_root
            or chunk_path.name != "manifest.json"
            or not chunk_path.is_file()
            or record.get("sha256") != sha256_file(chunk_path)
        ):
            raise ValueError(f"generated candidate shard manifest is stale: {index}")
        seen.add(chunk_path)
        chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        if chunk.get("status") != "COMPLETE" or (
            int(record.get("start", -1)) != cursor
            or int(chunk.get("start", -1)) != cursor
            or int(record.get("stop", -1)) != int(chunk.get("stop", -2))
        ):
            raise ValueError("generated candidate shard ranges are not contiguous")
        stop = int(chunk["stop"])
        if stop <= cursor or chunk.get("sample_rows") != stop - cursor:
            raise ValueError("generated candidate shard sample count is invalid")
        if record.get("sample_identity_sha256") != chunk.get(
            "sample_identity_sha256"
        ) or record.get("artifacts") != chunk.get("artifacts"):
            raise ValueError("generated candidate shard summary differs from its manifest")
        expected_samples = source_rows[cursor:stop]
        if (
            chunk.get("sample_identity_sha256")
            != canonical_sha256([str(row["sample_id"]) for row in expected_samples])
            or chunk.get("input_asset_identity_sha256")
            != canonical_sha256(common_asset_records(expected_samples, asset_cache))
        ):
            raise ValueError("generated candidate shard input identity is stale")
        for name in (
            "method",
            "split",
            "tag",
            "device",
            "locked_inference_source_sha256",
            "inference_code_bundle_sha256",
            "source_samples_sha256",
            "checkpoint_sha256",
            "selected_config_sha256",
            "native_decoder_config_sha256",
            "save_raw_maps",
        ):
            if chunk.get(name) != manifest.get(name):
                raise ValueError(f"generated candidate shard/run {name} mismatch")
        _verify_generated_artifact(
            chunk["artifacts"]["per_sample"],
            chunk_path.parent / "per_sample.parquet",
            name=f"shard-{index}-per-sample",
        )
        chunk_candidate_rows += _verify_generated_artifact(
            chunk["artifacts"]["candidates"],
            chunk_path.parent / "candidates.parquet",
            name=f"shard-{index}-candidates",
        )
        cursor = stop
    if cursor != sample_rows or chunk_candidate_rows != candidate_rows:
        raise ValueError("generated candidate shard totals differ from merged outputs")
    source_records = (
        (manifest.get("source_samples"), manifest.get("source_samples_sha256"), "samples"),
        (manifest.get("checkpoint"), manifest.get("checkpoint_sha256"), "checkpoint"),
        (manifest.get("selected_config"), manifest.get("selected_config_sha256"), "configuration"),
        (
            manifest.get("locked_inference_source"),
            manifest.get("locked_inference_source_sha256"),
            "locked inference source",
        ),
    )
    for raw_path, expected_sha, name in source_records:
        source = Path(str(raw_path or "")).resolve()
        if not source.is_file() or source.is_symlink() or sha256_file(source) != expected_sha:
            raise ValueError(f"generated candidate {name} source is stale")
    return manifest


def _validate_candidate_origins(run_dir: Path, previous: dict[str, Any]) -> None:
    pools = dict(previous.get("pools", {}))
    audit, fair_run = _fair_source_audit(run_dir)
    for route in ("CROG", "G1", "C1"):
        _audited_fair_candidate_source(audit, route=route)
        for split in ("train", "validation", "test"):
            all_path = run_dir / "02_candidates" / f"{route.lower()}_{split}_all.parquet"
            top5_path = run_dir / "02_candidates" / f"{route.lower()}_{split}_top5.parquet"
            if not all_path.is_file() or not top5_path.is_file():
                continue
            paired = pd.read_parquet(run_dir / "01_manifests" / f"paired_{split}.parquet")
            canonical = pd.read_parquet(all_path)
            top5 = pd.read_parquet(top5_path)
            _validate_geometry_hashes(canonical, route=route, split=split)
            _validate_geometry_hashes(top5, route=route, split=split)
            expected_top5 = canonical.loc[canonical["native_rank"] <= 5].copy()
            if _candidate_contract_rows(top5) != _candidate_contract_rows(expected_top5):
                raise ValueError(f"canonical Top-5 is not an exact All-pool subset: {route}/{split}")
            if route in {"G1", "C1"} and split == "test":
                expected = _load_fair(fair_run, route, split, paired)
            else:
                lineage = dict(pools.get(f"{route}/{split}/all", {})).get("lineage")
                if not isinstance(lineage, dict):
                    raise ValueError(f"candidate source lineage is missing: {route}/{split}")
                if route == "CROG":
                    source = _verified_source(
                        lineage.get("source"), lineage.get("source_sha256"), f"{route}/{split}"
                    )
                    expected = _load_crog_lineage_source(source, paired, fair_run, split)
                else:
                    expected = _load_generated_lineage_source(
                        run_dir=run_dir,
                        route=route,
                        split=split,
                        paired=paired,
                        fair_run=fair_run,
                        lineage=lineage,
                    )
            _assert_candidate_source_match(canonical, expected, route=route, split=split)


def build_available_candidate_pools(
    run_dir: Path,
    fair_run: Path,
    old_feature_dir: Path,
) -> dict[str, Any]:
    audit: dict[str, Any] = {"status": "PARTIAL_AWAITING_G1_C1_DEVELOPMENT_INFERENCE", "pools": {}}
    paired = {
        split: pd.read_parquet(run_dir / "01_manifests" / f"paired_{split}.parquet")
        for split in ("train", "validation", "test")
    }
    for split in ("train", "validation", "test"):
        crog, lineage = _load_crog_development(old_feature_dir, paired[split], fair_run, split)
        if split == "test":
            audit["crog_test_fair_exact_crosscheck"] = _verify_crog_test_exact(crog, fair_run)
        for pool, frame in (("all", crog), ("top5", crog.loc[crog["native_rank"] <= 5].copy())):
            path = run_dir / "02_candidates" / f"crog_{split}_{pool}.parquet"
            _atomic_parquet(frame, path)
            audit["pools"][f"CROG/{split}/{pool}"] = {
                **_pool_descriptor(frame, path, paired[split]["sample_id"].tolist()),
                "lineage": lineage,
            }
    for route in ("G1", "C1"):
        full = _load_fair(fair_run, route, "test", paired["test"])
        source = fair_run / "02_predictions" / f"{ROUTE_NAMES[route]}_native_predictions.parquet"
        lineage = {
            "source_kind": "audited_fair_native_predictions",
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
        }
        for pool, frame in (("all", full), ("top5", full.loc[full["native_rank"] <= 5].copy())):
            path = run_dir / "02_candidates" / f"{route.lower()}_test_{pool}.parquet"
            _atomic_parquet(frame, path)
            audit["pools"][f"{route}/test/{pool}"] = {
                **_pool_descriptor(frame, path, paired["test"]["sample_id"].tolist()),
                "lineage": lineage,
            }
    atomic_json(run_dir / "02_candidates" / "candidate_contract_hashes.json", audit)
    return audit


def ingest_generated_candidate_pool(
    run_dir: Path,
    fair_run: Path,
    *,
    route: str,
    split: str,
) -> dict[str, Any]:
    """Freeze one freshly generated G1/C1 development pool."""

    route = route.upper()
    if route not in {"G1", "C1"} or split not in {"train", "validation"}:
        raise ValueError("generated pool must be G1/C1 Train or Validation")
    source_root = run_dir / "02_candidates" / "native_work" / f"{route.lower()}_{split}"
    manifest_path = source_root / "run_manifest.json"
    source_path = source_root / "candidates.parquet"
    if not manifest_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(f"generated candidate pool is incomplete: {source_root}")
    manifest = _verify_generated_run_manifest(manifest_path, source_root)
    if manifest.get("status") != "COMPLETE" or manifest.get("tag") != "formal":
        raise ValueError(f"generated pool lacks a formal COMPLETE manifest: {source_root}")
    constants = _fair_route_constants(fair_run, route)
    expected_lineage = {
        "checkpoint_sha256": constants["checkpoint_sha256"],
        "selected_config_sha256": constants["selected_config_sha256"],
        "native_decoder_config_sha256": constants["native_decoder_config_sha256"],
    }
    for key, expected in expected_lineage.items():
        if str(manifest.get(key)) != expected:
            raise ValueError(
                f"generated {route}/{split} lineage mismatch for {key}: "
                f"{manifest.get(key)} != {expected}"
            )
    original = pd.read_parquet(source_path)
    for key, expected in expected_lineage.items():
        if set(original[key].astype(str)) != {expected}:
            raise ValueError(f"candidate-row lineage mismatch for {route}/{split}/{key}")
    candidate = _normalise(
        original.rename(columns={"jaw_width_px": "width_px", "rectangle_height_px": "height_px"}),
        route=route,
        split=split,
        constants=constants,
    )
    paired = pd.read_parquet(run_dir / "01_manifests" / f"paired_{split}.parquet")
    candidate = _attach_context(candidate, paired, route=route, split=split)
    contract_path = run_dir / "02_candidates" / "candidate_contract_hashes.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for pool, frame in (("all", candidate), ("top5", candidate.loc[candidate["native_rank"] <= 5].copy())):
        destination = run_dir / "02_candidates" / f"{route.lower()}_{split}_{pool}.parquet"
        _atomic_parquet(frame, destination)
        contract["pools"][f"{route}/{split}/{pool}"] = {
            **_pool_descriptor(frame, destination, paired["sample_id"].tolist()),
            "lineage": {
                "source": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "run_manifest": str(manifest_path.resolve()),
                "run_manifest_sha256": sha256_file(manifest_path),
                "locked_inference_source_sha256": manifest.get("locked_inference_source_sha256"),
            },
        }
    expected = {
        f"{route_name}/{split_name}/{pool}"
        for route_name in ("CROG", "G1", "C1")
        for split_name in ("train", "validation", "test")
        for pool in ("all", "top5")
    }
    missing = sorted(expected.difference(contract["pools"]))
    contract["missing_pools"] = missing
    contract["status"] = "COMPLETE" if not missing else "PARTIAL_AWAITING_G1_C1_DEVELOPMENT_INFERENCE"
    atomic_json(contract_path, contract)
    return {
        "route": route,
        "split": split,
        "status": "COMPLETE",
        "all": contract["pools"][f"{route}/{split}/all"],
        "top5": contract["pools"][f"{route}/{split}/top5"],
        "remaining_missing_pools": missing,
    }


def regenerate_candidate_contract_hashes(run_dir: Path) -> dict[str, Any]:
    """Regenerate the mutable candidate inventory from current canonical files.

    The inventory is a readiness audit, not a formal-lock input.  Content-bound
    consumers continue to lock the candidate Parquet files themselves.
    """

    root = run_dir.resolve()
    if (root / "08_lock" / "FORMAL_TEST_LOCK.json").exists() or (
        root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    ).exists():
        raise PermissionError("candidate contract inventory cannot change after lock/claim")
    path = root / "02_candidates" / "candidate_contract_hashes.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            previous = value
    pools: dict[str, Any] = {}
    expected = {
        f"{route.upper()}/{split}/{pool}"
        for route in ("crog", "g1", "c1")
        for split in ("train", "validation", "test")
        for pool in ("all", "top5")
    }
    existing_keys = {
        f"{route.upper()}/{split}/{pool}"
        for route in ("crog", "g1", "c1")
        for split in ("train", "validation", "test")
        for pool in ("all", "top5")
        if (root / "02_candidates" / f"{route}_{split}_{pool}.parquet").is_file()
    }
    if existing_keys == expected:
        _validate_candidate_origins(root, previous)
        fair_audit, _fair_run = _fair_source_audit(root)
    else:
        fair_audit = {}
    for route in ("crog", "g1", "c1"):
        for split in ("train", "validation", "test"):
            sample_path = root / "01_manifests" / f"paired_{split}.parquet"
            if not sample_path.is_file():
                continue
            sample_ids = pd.read_parquet(sample_path, columns=["sample_id"])[
                "sample_id"
            ].astype(str).tolist()
            for pool in ("all", "top5"):
                candidate_path = (
                    root / "02_candidates" / f"{route}_{split}_{pool}.parquet"
                )
                if not candidate_path.is_file():
                    continue
                frame = pd.read_parquet(candidate_path)
                key = f"{route.upper()}/{split}/{pool}"
                record = _pool_descriptor(frame, candidate_path, sample_ids)
                old = dict(previous.get("pools", {})).get(key)
                if isinstance(old, dict) and isinstance(old.get("lineage"), dict):
                    record["lineage"] = old["lineage"]
                elif route in {"g1", "c1"} and split == "test":
                    source_record = dict(fair_audit.get("artifacts", {})).get(
                        f"{route}_candidates"
                    )
                    if isinstance(source_record, dict):
                        record["lineage"] = {
                            "source_kind": "audited_fair_native_predictions",
                            "source": str(Path(str(source_record["path"])).resolve()),
                            "source_sha256": str(source_record["sha256"]),
                        }
                pools[key] = record
    missing = sorted(expected.difference(pools))
    audit: dict[str, Any] = {
        key: value
        for key, value in previous.items()
        if key not in {"status", "pools", "missing_pools", "regenerated_from_current_files"}
    }
    audit.update(
        {
            "status": "COMPLETE"
            if not missing
            else "PARTIAL_AWAITING_G1_C1_DEVELOPMENT_INFERENCE",
            "pools": pools,
            "missing_pools": missing,
            "regenerated_from_current_files": True,
            "source_lineage_verified": existing_keys == expected,
        }
    )
    atomic_json(path, audit)
    return audit


def verify_candidate_contract_hashes(run_dir: Path) -> dict[str, Any]:
    """Fail if the mutable candidate inventory is stale against current files."""

    root = run_dir.resolve()
    path = root / "02_candidates" / "candidate_contract_hashes.json"
    if not path.is_file():
        raise ValueError("candidate contract inventory is missing; regenerate P10 inputs")
    observed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict) or not isinstance(observed.get("pools"), dict):
        raise ValueError("candidate contract inventory is malformed")
    expected_keys = {
        f"{route.upper()}/{split}/{pool}"
        for route in ("crog", "g1", "c1")
        for split in ("train", "validation", "test")
        for pool in ("all", "top5")
    }
    if (
        set(observed["pools"]) != expected_keys
        or observed.get("status") != "COMPLETE"
        or observed.get("missing_pools") != []
        or observed.get("regenerated_from_current_files") is not True
        or observed.get("source_lineage_verified") is not True
    ):
        raise ValueError(
            "candidate contract inventory is incomplete or was not regenerated from current files"
        )
    _validate_candidate_origins(root, observed)
    for key, record in observed["pools"].items():
        try:
            route, split, pool = str(key).split("/")
        except ValueError as error:
            raise ValueError(f"invalid candidate contract inventory key: {key}") from error
        expected_path = (
            root
            / "02_candidates"
            / f"{route.lower()}_{split}_{pool}.parquet"
        ).resolve()
        if Path(str(record.get("path", ""))).resolve() != expected_path:
            raise ValueError(f"candidate contract inventory path is stale: {key}")
        sample_path = root / "01_manifests" / f"paired_{split}.parquet"
        if not sample_path.is_file() or not expected_path.is_file():
            raise ValueError(f"candidate contract inventory source is missing: {key}")
        sample_ids = pd.read_parquet(sample_path, columns=["sample_id"])[
            "sample_id"
        ].astype(str).tolist()
        expected_record = _pool_descriptor(
            pd.read_parquet(expected_path), expected_path, sample_ids
        )
        checked_fields = {
            "path",
            "sha256",
            "rows",
            "samples",
            "samples_with_candidates",
            "no_output_samples",
            "max_candidates",
            "membership_geometry_score_sha256",
        }
        if {name: record.get(name) for name in checked_fields} != {
            name: expected_record.get(name) for name in checked_fields
        }:
            raise ValueError(f"candidate contract inventory is stale: {key}")
    return observed
