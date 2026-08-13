"""Apply the locked full-Train D1 calibrator to a label-free Test pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.calibration import (  # noqa: E402
    PROBABILITY_EPSILON,
    assert_order_invariant,
    calibrator_from_serialized,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.test_access_guard import append_access_log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool", choices=("top5", "top10", "allnms"), default="top5")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(run_dir: Path, *, pool: str, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    calibration_manifest_path = (
        root / "05_calibration" / pool / "calibration_manifest.json"
    )
    calibration_manifest = load_verified_json(
        calibration_manifest_path, name=f"D1 {pool} calibration manifest"
    )
    verify_artifact_records_recursive(
        {
            "sources": calibration_manifest.get("sources"),
            "artifacts": calibration_manifest.get("artifacts"),
        },
        name=f"D1 {pool} calibration",
        require_at_least_one=True,
    )
    selected = str(calibration_manifest.get("selected_method", ""))
    calibrator_payload = (
        calibration_manifest.get("calibrators", {})
        if isinstance(calibration_manifest.get("calibrators"), dict)
        else {}
    )
    if not calibrator_payload:
        raise ValueError("D1 calibration manifest omits serialized calibrators")
    full_train = calibrator_payload.get("full_train", {})
    if selected not in full_train:
        raise ValueError("D1 selected full-Train calibrator is absent")

    candidate_manifest_path = root / "02_candidates" / "test_manifest.json"
    candidate_manifest = load_verified_json(
        candidate_manifest_path, name="D1 Test candidate manifest"
    )
    candidate_record = candidate_manifest.get("artifacts", {}).get(pool)  # type: ignore[union-attr]
    candidate_path = verified_artifact_path(
        candidate_record or {}, name=f"D1 Test/{pool} candidates"
    )
    sources = {
        "calibration_manifest": artifact_record(calibration_manifest_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "calibration_primitive": artifact_record(
            ROOT / "src/unified_reranking/calibration.py"
        ),
        "tool": artifact_record(Path(__file__)),
    }
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "split": "test",
        "pool": pool,
        "selected_method": selected,
        "probability_clip": [PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON],
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = root / "05_calibration" / pool
    manifest_path = output / "test_application_manifest.json"
    destination = output / "d1_test.parquet"
    if manifest_path.is_file():
        existing = load_verified_json(
            manifest_path, name=f"D1 Test/{pool} calibration application"
        )
        unsigned = dict(existing)
        observed_content = unsigned.pop("content_sha256", None)
        if observed_content != canonical_sha256(unsigned):
            raise RuntimeError("D1 Test calibration application content hash mismatch")
        if (
            resume
            and existing.get("source_signature_sha256") == signature
            and existing.get("configuration") == configuration
            and existing.get("sources") == sources
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name=f"D1 Test/{pool} calibration resume",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 Test calibration application contract differs")

    candidates = pd.read_parquet(
        candidate_path,
        columns=["sample_id", "candidate_id", "native_rank", "native_score"],
    )
    model = calibrator_from_serialized(full_train[selected])
    result = candidates.copy()
    result["calibrated_native_probability"] = model.predict(result["native_score"])
    probability = np.clip(
        result["calibrated_native_probability"].to_numpy(float),
        PROBABILITY_EPSILON,
        1 - PROBABILITY_EPSILON,
    )
    result["base_logit"] = np.log(probability) - np.log1p(-probability)
    assert_order_invariant(result, "calibrated_native_probability")
    artifact = atomic_parquet(result, destination)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": configuration,
        "source_signature_sha256": signature,
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": {"test_predictions": artifact_record(artifact)},
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "stage": f"d1_calibration_{pool}_test",
            "candidate_test_labels_read": False,
            "inputs": [sources["candidates"], sources["calibration_manifest"]],
            "output_manifest": artifact_record(manifest_path),
        },
    )
    return manifest


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    manifest_path = (
        root / "05_calibration" / args.pool / "test_application_manifest.json"
    )
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P8",
        substage=f"d1_calibration_test_{args.pool}",
        route="D1",
        pool=args.pool,
        method="locked_full_train_calibrator",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, pool=args.pool, resume=args.resume)
        state["artifact_path"] = str(manifest_path)
        state["artifact_sha256"] = sha256_file(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
