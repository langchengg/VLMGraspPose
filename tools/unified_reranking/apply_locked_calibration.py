"""Apply a Validation-selected full-Train calibrator without loading labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.calibration import (
    PROBABILITY_EPSILON,
    assert_order_invariant,
    calibrator_from_serialized,
)
from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _formal_lock_exists(run_dir: Path) -> bool:
    return (run_dir / "08_lock" / "FORMAL_TEST_LOCK.json").is_file() or (
        run_dir / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    ).is_file()


def run(run_dir: Path, route: str, split: str) -> dict[str, object]:
    if split != "test":
        raise ValueError("this label-free application command is reserved for Test")
    manifest_path = run_dir / "05_calibration" / f"{route}_calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("route calibration is not complete")
    verify_artifact_records_recursive(
        manifest.get("sources", {}),
        name=f"{route} calibration sources",
        require_at_least_one=True,
    )
    verify_artifact_records_recursive(
        manifest.get("artifacts", {}),
        name=f"{route} calibration outputs",
        require_at_least_one=True,
    )
    selected = str(manifest["selected_method"])
    serialized = manifest["calibrators"]["full_train"][selected]
    calibrator = calibrator_from_serialized(serialized)
    candidate_path = run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
    output_path = run_dir / "05_calibration" / f"{route}_{split}.parquet"
    application_manifest_path = (
        run_dir / "05_calibration" / f"{route}_{split}_application_manifest.json"
    )
    sources = {
        "candidate_manifest": _record(candidate_path),
        "calibration_manifest": _record(manifest_path),
        "tool": _record(Path(__file__)),
    }
    configuration = {
        "route": route,
        "split": split,
        "selected_method": selected,
        "candidate_labels_loaded": False,
    }
    signature = canonical_sha256(
        {"configuration": configuration, "sources": sources}
    )
    if application_manifest_path.is_file():
        previous = json.loads(application_manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "COMPLETE_LABEL_FREE"
            and previous.get("signature_sha256") == signature
        ):
            verify_artifact_records_recursive(
                previous.get("sources", {}),
                name=f"{route} Test calibration resumable sources",
                require_at_least_one=True,
            )
            verify_artifact_records_recursive(
                previous.get("artifacts", {}),
                name=f"{route} Test calibration resumable outputs",
                require_at_least_one=True,
            )
            return previous
        if _formal_lock_exists(run_dir):
            raise RuntimeError(
                f"{route} Test calibration signature drift after formal locking"
            )
    candidates = pd.read_parquet(
        candidate_path,
        columns=["sample_id", "candidate_id", "native_rank", "native_score"],
    )
    probability = calibrator.predict(candidates["native_score"])
    output = candidates.copy()
    output["calibrated_native_probability"] = probability
    output["base_logit"] = np.log(probability) - np.log1p(-probability)
    assert_order_invariant(output, "calibrated_native_probability")
    if not np.isfinite(output["base_logit"]).all():
        raise RuntimeError("calibration produced non-finite base logits")
    if not output["calibrated_native_probability"].between(
        PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON
    ).all():
        raise RuntimeError("calibration probability clip contract failed")
    _atomic_parquet(output_path, output)
    result = {
        "status": "COMPLETE_LABEL_FREE",
        "route": route.upper(),
        "split": split,
        "selected_method": selected,
        "configuration": configuration,
        "signature_sha256": signature,
        "candidate_rows": len(output),
        "candidate_bearing_samples": int(output["sample_id"].nunique()),
        "candidate_order_invariant": True,
        "candidate_manifest": sources["candidate_manifest"],
        "calibration_manifest": sources["calibration_manifest"],
        "artifact": _record(output_path),
        "sources": sources,
        "artifacts": {"calibrated_candidates": _record(output_path)},
        "candidate_labels_loaded": False,
    }
    atomic_json(application_manifest_path, result)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_calibration",
            "route": route.upper(),
            "allowed": ["candidate_geometry", "native_score"],
            "output_manifest": str(application_manifest_path.resolve()),
            "output_manifest_sha256": sha256_file(application_manifest_path),
            "candidate_labels_loaded": False,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument("--split", choices=("test",), default="test")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P6",
        substage=f"label_free_calibration_{args.route}_{args.split}",
        route=args.route,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(run_dir, args.route, args.split)
        artifact = Path(str(result["artifact"]["path"]))
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
