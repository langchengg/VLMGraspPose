"""Fit frozen grouped-OOF D1 q calibration using development labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.calibration import (  # noqa: E402
    CALIBRATION_METHODS,
    RELIABILITY_BINS,
    join_development_calibration,
    reliability_rows,
    top1_success_numerator,
)
from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.io import atomic_parquet, atomic_pickle  # noqa: E402
from d1_reranking.provenance import load_source_closure  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.calibration import grouped_oof_calibration  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pool", choices=("top5", "top10", "allnms"), default="top5")
    return parser.parse_args()


def _content_manifest(path: Path, *, name: str) -> dict[str, object]:
    value = load_verified_json(path, name=name)
    unsigned = dict(value)
    observed = unsigned.pop("content_sha256", None)
    if observed != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash mismatch")
    return value


def _candidate_manifest(
    run_dir: Path, split: str, closure_path: Path
) -> tuple[dict[str, object], Path]:
    base = (
        run_dir / "02_candidates"
        if split == "test"
        else run_dir / "02_candidates" / split
    )
    path = base / ("test_manifest.json" if split == "test" else "manifest.json")
    manifest = _content_manifest(path, name=f"D1 {split} candidate manifest")
    configuration = manifest.get("configuration")
    closure_record = (
        configuration.get("source_contract", {}).get("source_closure")
        if isinstance(configuration, dict)
        else None
    )
    expected_closure = artifact_record(closure_path)
    closure_matches = isinstance(closure_record, dict) and all(
        closure_record.get(key) == expected_closure.get(key)
        for key in ("path", "sha256")
    )
    if not isinstance(configuration, dict) or (
        configuration.get("route") != "D1"
        or configuration.get("split") != split
        or not closure_matches
    ):
        raise RuntimeError(f"D1 {split} candidate/source-closure binding differs")
    verify_artifact_records_recursive(
        manifest.get("artifacts"),
        name=f"D1 {split} candidate artifacts",
        require_at_least_one=True,
    )
    return manifest, path


def _development_table(
    run_dir: Path, split: str, pool: str, closure_path: Path
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    manifest, manifest_path = _candidate_manifest(run_dir, split, closure_path)
    candidate_record = manifest.get("artifacts", {}).get(pool)  # type: ignore[union-attr]
    if not isinstance(candidate_record, dict):
        raise ValueError(f"D1 {split}/{pool} candidate record is absent")
    candidate_path = verified_artifact_path(
        candidate_record, name=f"D1 {split}/{pool} candidates"
    )
    expected_candidate_path = (
        run_dir / "02_candidates" / split / f"d1_{pool}_candidates.parquet"
    ).resolve()
    if candidate_path != expected_candidate_path:
        raise RuntimeError(f"D1 {split}/{pool} candidate path differs")
    candidate_hashes_path = verified_artifact_path(
        manifest.get("artifacts", {}).get("candidate_hashes", {}),  # type: ignore[union-attr]
        name=f"D1 {split} candidate hashes",
    )
    label_manifest_path = (
        run_dir / "03_features" / split / pool / "labels" / "manifest.json"
    )
    label_manifest = _content_manifest(
        label_manifest_path, name=f"D1 {split}/{pool} label manifest"
    )
    expected_sources = {
        "source_closure": artifact_record(closure_path),
        "candidate_manifest": artifact_record(manifest_path),
        "candidate_hashes": artifact_record(candidate_hashes_path),
        "candidates": artifact_record(candidate_path),
    }
    label_sources = label_manifest.get("sources")
    if (
        label_manifest.get("split") != split
        or label_manifest.get("pool") != pool
        or label_manifest.get("candidate_test_labels_read") is not False
        or not isinstance(label_sources, dict)
        or any(
            label_sources.get(key) != value for key, value in expected_sources.items()
        )
    ):
        raise RuntimeError(
            "D1 development label manifest semantics/source binding differ"
        )
    verify_artifact_records_recursive(
        {"sources": label_sources, "artifact": label_manifest.get("artifact")},
        name=f"D1 {split}/{pool} development labels",
        require_at_least_one=True,
    )
    label_path = verified_artifact_path(
        label_manifest.get("artifact", {}), name=f"D1 {split}/{pool} labels"
    )
    frame = join_development_calibration(
        pd.read_parquet(candidate_path), pd.read_parquet(label_path)
    )
    return frame, {
        "source_closure": artifact_record(closure_path),
        "candidate_manifest": artifact_record(manifest_path),
        "candidate_hashes": artifact_record(candidate_hashes_path),
        "candidates": artifact_record(candidate_path),
        "label_manifest": artifact_record(label_manifest_path),
        "labels": artifact_record(label_path),
    }


def _write_reliability_pdf(path: Path, rows: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    for method, group in rows.groupby("method", sort=True):
        observed = group.dropna(subset=["mean_probability", "observed_frequency"])
        axis.plot(
            observed["mean_probability"],
            observed["observed_frequency"],
            marker="o",
            linewidth=1.5,
            label=str(method),
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability",
        ylabel="Observed frequency",
    )
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        temporary, format="pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)
    temporary.replace(path)


def run(run_dir: Path, *, pool: str, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    if pool not in {"top5", "top10", "allnms"}:
        raise ValueError(f"unsupported D1 calibration pool: {pool}")
    assert_writable_prelock(root)
    closure_path, _closure = load_source_closure(root)
    train, train_sources = _development_table(root, "train", pool, closure_path)
    validation, validation_sources = _development_table(
        root, "validation", pool, closure_path
    )
    folds_path = root / "04_splits" / "fold_assignments.parquet"
    denominator_path = root / "01_manifests" / "d1_paired_validation.parquet"
    sources: dict[str, object] = {
        "train": train_sources,
        "validation": validation_sources,
        "fold_assignments": artifact_record(folds_path),
        "validation_denominator": artifact_record(denominator_path),
        "calibration_primitive": artifact_record(
            ROOT / "src/unified_reranking/calibration.py"
        ),
        "tool": artifact_record(Path(__file__)),
    }
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "pool": pool,
        "methods": list(CALIBRATION_METHODS),
        "reliability_bins": RELIABILITY_BINS,
        "probability_clip": [1e-4, 1 - 1e-4],
        "selection_scope": "Train grouped OOF fit; official Validation selection",
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = root / "05_calibration" / pool
    manifest_path = output / "calibration_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            resume
            and existing.get("status") == "COMPLETE"
            and existing.get("source_signature_sha256") == signature
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name=f"D1 {pool} calibration resume",
                require_at_least_one=True,
            )
            unsigned = dict(existing)
            observed_content = unsigned.pop("content_sha256", None)
            if observed_content != canonical_sha256(unsigned):
                raise RuntimeError("D1 calibration manifest content hash mismatch")
            return existing
        raise RuntimeError(
            "D1 calibration manifest exists with a different or corrupt contract"
        )

    folds = pd.read_parquet(folds_path)
    oof, val, calibrators, metadata = grouped_oof_calibration(train, folds, validation)
    native_numerator = top1_success_numerator(validation)
    calibrated_numerator = top1_success_numerator(val, "calibrated_native_probability")
    if native_numerator != calibrated_numerator:
        raise RuntimeError("D1 monotone calibration changed native Validation J@1")

    oof_path = atomic_parquet(oof, output / "d1_train_oof.parquet")
    validation_path = atomic_parquet(val, output / "d1_validation.parquet")
    rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    for method in CALIBRATION_METHODS:
        metrics_rows.append(
            {
                "method": method,
                **metadata["validation_metrics"][method],
                "metric_rank_sum": metadata["metric_rank_sums"][method],
                "selected": method == metadata["selected_method"],
            }
        )
        rows.extend(
            {"method": method, **row}
            for row in reliability_rows(val, f"calibrated_probability_{method}")
        )
    metrics_path = output / "calibration_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    reliability_path = atomic_parquet(
        pd.DataFrame(rows), output / "reliability_bins.parquet"
    )
    diagram_path = output / "reliability_diagram.pdf"
    _write_reliability_pdf(diagram_path, pd.DataFrame(rows))
    calibrator_path = atomic_pickle(
        {
            "schema_version": 1,
            "route": "D1",
            "pool": pool,
            "selected_method": metadata["selected_method"],
            "calibrators": calibrators,
            "source_signature_sha256": signature,
        },
        output / "calibrator.pkl",
    )
    audit_path = output / "CALIBRATION_AUDIT.md"
    atomic_text(
        audit_path,
        "# D1 q calibration audit\n\n"
        f"- Pool: `{pool}`.\n"
        f"- Selected method: **{metadata['selected_method']}**.\n"
        f"- Validation native/calibrated J@1 numerator: {native_numerator}/{calibrated_numerator}.\n"
        "- Candidate order is unchanged by monotone calibration.\n"
        "- Test candidate labels opened: no.\n",
    )
    artifacts = {
        "train_oof": artifact_record(oof_path),
        "validation": artifact_record(validation_path),
        "calibrator": artifact_record(calibrator_path),
        "calibration_metrics": artifact_record(metrics_path),
        "reliability_bins": artifact_record(reliability_path),
        "reliability_diagram": artifact_record(diagram_path),
        "audit_report": artifact_record(audit_path),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": configuration,
        "source_signature_sha256": signature,
        "candidate_test_labels_read": False,
        "selected_method": metadata["selected_method"],
        "selection_rule": metadata["selection_rule"],
        "validation_metrics": metadata["validation_metrics"],
        "metric_rank_sums": metadata["metric_rank_sums"],
        "calibrators": calibrators,
        "validation_native_j1_numerator": native_numerator,
        "validation_calibrated_j1_numerator": calibrated_numerator,
        "validation_denominator": int(
            pd.read_parquet(denominator_path, columns=["sample_id"])[
                "sample_id"
            ].nunique()
        ),
        "sources": sources,
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    manifest_path = root / "05_calibration" / args.pool / "calibration_manifest.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P8",
        substage=f"d1_calibration_{args.pool}",
        route="D1",
        pool=args.pool,
        method="platt_isotonic_validation_selection",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, pool=args.pool, resume=args.resume)
        state["artifact_path"] = str(manifest_path)
        state["artifact_sha256"] = sha256_file(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
