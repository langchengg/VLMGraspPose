"""Deterministically audit Train/Validation Top-15 union oracle headroom."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage


ROUTES = ("crog", "g1", "c1")
SPLITS = ("train", "validation")
HEADROOM_THRESHOLD = 0.0025


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _read_route(run_dir: Path, route: str, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise PermissionError("union headroom is development-only; Test labels are forbidden")
    candidates_path = run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
    labels_path = run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
    candidates = pd.read_parquet(
        candidates_path,
        columns=["sample_id", "candidate_id", "native_rank"],
    )
    labels = pd.read_parquet(
        labels_path,
        columns=["sample_id", "candidate_id", "candidate_success"],
    )
    keys = ["sample_id", "candidate_id"]
    for name, frame in (("candidates", candidates), ("labels", labels)):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{route}/{split} {name} has invalid candidate keys")
    if candidates.groupby("sample_id").size().max() > 5:
        raise ValueError(f"{route}/{split} exceeds the frozen Top-5 contract")
    joined = candidates.merge(labels, on=keys, validate="one_to_one")
    if len(joined) != len(candidates) or len(joined) != len(labels):
        raise ValueError(f"{route}/{split} labels do not exactly cover candidates")
    joined["route"] = route.upper()
    joined["route_candidate_id"] = joined["route"] + ":" + joined["candidate_id"].astype(str)
    return joined


def analyze_split(run_dir: Path, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return aggregate and per-sample union oracle outcomes for one dev split."""

    if split not in SPLITS:
        raise PermissionError("union headroom is development-only; Test labels are forbidden")
    manifest_path = run_dir / "01_manifests" / f"paired_{split}.parquet"
    manifest = pd.read_parquet(manifest_path, columns=["sample_id", "scene_id"])
    if manifest["sample_id"].astype(str).duplicated().any() or manifest.empty:
        raise ValueError(f"paired {split} manifest has invalid sample IDs")
    result = manifest.copy()
    route_frames: list[pd.DataFrame] = []
    rates: dict[str, float] = {}
    counts: dict[str, int] = {}
    for route in ROUTES:
        frame = _read_route(run_dir, route, split)
        route_frames.append(frame)
        per_sample = frame.groupby("sample_id", sort=False).agg(
            **{
                f"{route}_candidate_count": ("candidate_id", "size"),
                f"{route}_oracle": ("candidate_success", "max"),
            }
        ).reset_index()
        result = result.merge(per_sample, on="sample_id", how="left", validate="one_to_one")
        result[f"{route}_candidate_count"] = result[f"{route}_candidate_count"].fillna(0).astype(int)
        result[f"{route}_oracle"] = result[f"{route}_oracle"].fillna(False).astype(bool)
        counts[route] = int(result[f"{route}_oracle"].sum())
        rates[route] = float(result[f"{route}_oracle"].mean())
    union = pd.concat(route_frames, ignore_index=True)
    # The primary union deliberately does not deduplicate geometry or identity.
    if union.duplicated(["sample_id", "route_candidate_id"]).any():
        raise RuntimeError("route-qualified primary union identities are not unique")
    per_union = union.groupby("sample_id", sort=False).agg(
        union_candidate_count=("route_candidate_id", "size"),
        union_oracle=("candidate_success", "max"),
    ).reset_index()
    result = result.merge(per_union, on="sample_id", how="left", validate="one_to_one")
    result["union_candidate_count"] = result["union_candidate_count"].fillna(0).astype(int)
    result["union_oracle"] = result["union_oracle"].fillna(False).astype(bool)
    if (result["union_candidate_count"] > 15).any():
        raise RuntimeError("primary union exceeds 15 frozen candidates")
    best_route = min(ROUTES, key=lambda route: (-rates[route], ROUTES.index(route)))
    union_rate = float(result["union_oracle"].mean())
    gain = union_rate - rates[best_route]
    summary = {
        "split": split,
        "sample_count": int(len(result)),
        "candidate_count": int(union["route_candidate_id"].size),
        "maximum_candidates_per_sample": int(result["union_candidate_count"].max()),
        "route_oracle_successes": counts,
        "route_oracle_rates": rates,
        "route_oracle_at_5": rates,
        "best_single_route": best_route.upper(),
        "best_single_route_oracle": float(rates[best_route]),
        "union_oracle_successes": int(result["union_oracle"].sum()),
        "union_oracle": union_rate,
        "union_oracle_at_all": union_rate,
        "union_oracle_at_15": union_rate,
        "union_gain_over_best_single": float(gain),
        "union_gain_over_best_single_pp": float(100.0 * gain),
        "primary_union_deduplication": "NONE",
    }
    return summary, result


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE" or value.get("signature_sha256") != signature:
        raise RuntimeError("immutable union-headroom output exists with a different signature")
    for record in value.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("resumable union-headroom artifact hash mismatch")
    return value


def run(run_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    sources: dict[str, Any] = {}
    for split in SPLITS:
        sources[f"paired_{split}"] = _record(
            run_dir / "01_manifests" / f"paired_{split}.parquet"
        )
        for route in ROUTES:
            sources[f"{route}_{split}_candidates"] = _record(
                run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
            )
            sources[f"{route}_{split}_labels"] = _record(
                run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
            )
    configuration = {
        "splits": list(SPLITS),
        "routes": [route.upper() for route in ROUTES],
        "maximum_union_candidates": 15,
        "primary_union_deduplication": "NONE",
        "validation_headroom_threshold": HEADROOM_THRESHOLD,
        "validation_headroom_threshold_pp": 100.0 * HEADROOM_THRESHOLD,
        "test_access": "NONE",
    }
    sources["implementation_tool"] = _record(Path(__file__))
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output_dir = (output_dir or run_dir / "07_validation" / "union_headroom").resolve()
    marker = output_dir / "manifest.json"
    resumed = _resume(marker, signature)
    if resumed is not None:
        return resumed

    summaries: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for split in SPLITS:
        summary, samples = analyze_split(run_dir, split)
        summaries[split] = summary
        path = output_dir / f"{split}_per_sample_oracles.parquet"
        _atomic_parquet(path, samples)
        artifacts[f"{split}_per_sample"] = _record(path)
    validation_gain = float(summaries["validation"]["union_gain_over_best_single"])
    if validation_gain < HEADROOM_THRESHOLD:
        decision = "NO_UNION_HEADROOM"
        training_status = "NOT_APPLICABLE_NO_HEADROOM"
        nms_status = "NOT_APPLICABLE_NO_HEADROOM"
    else:
        decision = "UNION_HEADROOM_AVAILABLE"
        training_status = "ELIGIBLE_FIXED_LAMBDAMART_AND_DEEPSETS"
        nms_status = "ELIGIBLE_PREDECLARED_SENSITIVITY_ONLY"
    decision_path = output_dir / f"{decision}.json"
    atomic_json(
        decision_path,
        {
            "decision": decision,
            "complex_union_training": training_status,
            "deterministic_union_nms_secondary": nms_status,
            "observed_validation_gain": validation_gain,
            "threshold": HEADROOM_THRESHOLD,
            "test_labels_read": False,
        },
    )
    artifacts["decision"] = _record(decision_path)
    manifest = {
        "status": "COMPLETE",
        "signature_sha256": signature,
        "decision": decision,
        "complex_union_training": training_status,
        "deterministic_union_nms_secondary": nms_status,
        "configuration": configuration,
        "sources": sources,
        "summaries": summaries,
        "artifacts": artifacts,
        "test_access": "NONE",
        "test_labels_read": False,
    }
    atomic_json(marker, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P9",
        substage="top15_union_oracle_headroom",
        route="cross_route",
        pool="primary_union_top15_no_dedup",
        method="deterministic_oracle_audit",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(run_dir, output_dir=args.output_dir)
        marker = Path(args.output_dir or run_dir / "07_validation" / "union_headroom") / "manifest.json"
        state["artifact_path"] = str(marker.resolve())
        state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HEADROOM_THRESHOLD", "analyze_split", "run"]
