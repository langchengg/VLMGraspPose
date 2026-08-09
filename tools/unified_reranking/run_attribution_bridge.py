"""Run the pre-Test Train/Validation two-by-two attribution bridge."""

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

from unified_reranking.attribution_bridge import candidate_membership_overlap, evaluate_attribution_bridge
from unified_reranking.evaluator_adapter import build_candidate_labels
from unified_reranking.hashing import atomic_json, atomic_text, sha256_file
from unified_reranking.ledger import ledger_stage


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def consolidate_development_bridges(run_dir: Path) -> Path | None:
    """Materialize the exact Train/Validation bridge table once all cells exist."""

    rows: list[pd.DataFrame] = []
    for route in ("g1", "c1"):
        for split in ("train", "validation"):
            manifest_path = (
                run_dir
                / "11_attribution_bridge"
                / f"bridge_{route}_{split}_top5_manifest.json"
            )
            if not manifest_path.is_file():
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "COMPLETE":
                return None
            record = manifest.get("artifacts", {}).get("table", {})
            table_path = Path(str(record.get("path", ""))).resolve()
            if not table_path.is_file() or sha256_file(table_path) != record.get("sha256"):
                raise RuntimeError(f"bridge table hash mismatch: {route}/{split}")
            table = pd.read_csv(table_path)
            if table.empty:
                raise RuntimeError(f"bridge table is empty: {route}/{split}")
            table["route"] = route.upper()
            table["split"] = split
            rows.append(table)
    output = run_dir / "07_validation" / "bridge_train_validation.csv"
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["route", "split"], kind="mergesort"
    )
    _atomic_csv(output, combined)
    alias = run_dir / "11_attribution_bridge" / "bridge_train_validation.csv"
    _atomic_csv(alias, combined)
    if sha256_file(alias) != sha256_file(output):
        raise RuntimeError("Train/Validation bridge alias differs from canonical table")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("g1", "c1"))
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--pool", choices=("top5", "all"), default="top5")
    parser.add_argument(
        "--historical-run",
        type=Path,
        default=ROOT / "HiFi_reproduction/runs/g1_c1_complete_reranking_20260806T084131Z",
    )
    parser.add_argument(
        "--modular-run",
        type=Path,
        default=ROOT / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500",
    )
    return parser.parse_args()


def _historical_source(root: Path, route: str, split: str, pool: str) -> Path:
    if pool == "top5":
        return root / "data" / f"frozen_{route}_{split}_top5_candidates.parquet"
    suffix = "train_candidates" if split == "train" else "validation_allnms_candidates"
    return root / "data" / f"frozen_{route}_{suffix}.parquet"


def _normalise_historical(source: Path, route: str, split: str) -> pd.DataFrame:
    original = pd.read_parquet(source)
    required = {
        "sample_id",
        "stable_candidate_id",
        "original_rank",
        "original_score",
        "raw_network_quality",
        "stored_center_mask_support",
        "center_x",
        "center_y",
        "angle_deg",
        "width_px",
        "height_px",
    }
    missing = sorted(required.difference(original.columns))
    if missing:
        raise ValueError(f"historical candidate source missing: {missing}")
    renamed = original.rename(
        columns={
            "stable_candidate_id": "candidate_id",
            "original_rank": "native_rank",
            "center_x": "cx_px",
            "center_y": "cy_px",
            "angle_deg": "theta_deg",
        }
    ).copy()
    renamed["native_score"] = pd.to_numeric(renamed["original_score"], errors="raise")
    renamed["route"] = route.upper()
    renamed["split"] = split
    if renamed[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("historical candidate identities are not unique")
    return renamed


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    historical_run = args.historical_run.resolve()
    modular_run = args.modular_run.resolve()
    output = run_dir / "11_attribution_bridge"
    output.mkdir(parents=True, exist_ok=True)
    source = _historical_source(historical_run, args.route, args.split, args.pool)
    historical = _normalise_historical(source, args.route, args.split)
    historical_candidate_path = output / f"historical_candidates_{args.route}_{args.split}_{args.pool}.parquet"
    _atomic_parquet(historical_candidate_path, historical)

    evaluator_path = run_dir / "configs" / "canonical_evaluator.py"
    evaluator_sha = sha256_file(evaluator_path)
    label_source = modular_run / "manifests" / f"{args.split}_labels.parquet"
    historical_label_path = output / f"historical_labels_{args.route}_{args.split}_{args.pool}.parquet"
    label_descriptor = build_candidate_labels(
        historical_candidate_path,
        label_source,
        historical_label_path,
        evaluator_path,
        evaluator_sha,
        split=args.split,
    )
    historical_labels = pd.read_parquet(historical_label_path)[
        ["sample_id", "candidate_id", "candidate_success"]
    ]
    historical = historical.merge(
        historical_labels, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    historical = historical.rename(columns={"cx_px": "cx_px", "cy_px": "cy_px"})

    fair_candidate_path = run_dir / "02_candidates" / f"{args.route}_{args.split}_{args.pool}.parquet"
    fair_label_path = run_dir / "03_features" / f"candidate_labels_{args.route}_{args.split}_{args.pool}.parquet"
    common_name = f"{args.route}_{args.split}" if args.pool == "top5" else f"{args.route}_{args.split}_all"
    common_path = run_dir / "03_features" / "common" / common_name / "candidate_features.parquet"
    fair = pd.read_parquet(fair_candidate_path)
    fair = fair.merge(
        pd.read_parquet(fair_label_path)[["sample_id", "candidate_id", "candidate_success"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    fair = fair.merge(
        pd.read_parquet(common_path)[["sample_id", "candidate_id", "p_center"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    denominator = pd.read_parquet(
        run_dir / "01_manifests" / f"paired_{args.split}.parquet", columns=["sample_id"]
    )["sample_id"].astype(str).tolist()
    table, decisions = evaluate_attribution_bridge(denominator, fair, historical)
    table.insert(0, "route", args.route.upper())
    table.insert(1, "split", args.split)
    table.insert(2, "pool_contract", args.pool)
    decision_path = output / f"bridge_decisions_{args.route}_{args.split}_{args.pool}.parquet"
    table_path = output / f"bridge_{args.route}_{args.split}_{args.pool}.csv"
    _atomic_parquet(decision_path, decisions)
    _atomic_csv(table_path, table)

    residual = (
        pd.to_numeric(historical["original_score"], errors="raise").to_numpy(float)
        - pd.to_numeric(historical["raw_network_quality"], errors="raise").to_numpy(float)
        * np.clip(pd.to_numeric(historical["stored_center_mask_support"], errors="raise").to_numpy(float), 0, 1)
    )
    compatibility = {
        "formula": "historical original_score = raw_network_quality * stored_center_mask_support; jaw exponent zero",
        "maximum_absolute_residual": float(np.abs(residual).max(initial=0.0)),
        "compatible": bool(np.abs(residual).max(initial=0.0) <= 1e-5),
        "fair_transfer_formula": "fair native_score * common p_center",
    }
    if not compatibility["compatible"]:
        raise RuntimeError("historical selector decomposition is not compatible with pure transfer")
    overlap = candidate_membership_overlap(fair, historical)
    manifest = {
        "status": "COMPLETE",
        "route": args.route.upper(),
        "split": args.split,
        "pool_contract": args.pool,
        "test_access": False,
        "selector_compatibility": compatibility,
        "candidate_membership_overlap": overlap,
        "historical_labels": label_descriptor,
        "sources": {
            "fair_candidates": {"path": str(fair_candidate_path.resolve()), "sha256": sha256_file(fair_candidate_path)},
            "fair_labels": {"path": str(fair_label_path.resolve()), "sha256": sha256_file(fair_label_path)},
            "common_features": {"path": str(common_path.resolve()), "sha256": sha256_file(common_path)},
            "historical_candidates": {"path": str(source), "sha256": sha256_file(source)},
            "label_source": {"path": str(label_source), "sha256": sha256_file(label_source)},
            "evaluator": {"path": str(evaluator_path), "sha256": evaluator_sha},
        },
        "artifacts": {
            "table": {"path": str(table_path.resolve()), "sha256": sha256_file(table_path)},
            "decisions": {"path": str(decision_path.resolve()), "sha256": sha256_file(decision_path)},
            "historical_labels": {"path": str(historical_label_path.resolve()), "sha256": sha256_file(historical_label_path)},
        },
    }
    atomic_json(output / f"bridge_{args.route}_{args.split}_{args.pool}_manifest.json", manifest)
    atomic_text(
        output / "BRIDGE_DESIGN.md",
        "# Two-by-two attribution bridge\n\n"
        "The bridge crosses the frozen fair Gaussian candidate pool with the compatible "
        "historical `q × centre-support` selector, and the historical probability-gated/NMS "
        "pool with both raw network quality and its historical selector. Train and Validation "
        "candidate labels are recomputed with the frozen canonical evaluator. Test is excluded "
        "from this command and may run only after the primary formal lock. No bridge result may "
        "replace the validation-declared primary order-only method.\n",
    )
    consolidate_development_bridges(run_dir)
    return manifest


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P10",
        substage=f"bridge_{args.route}_{args.split}_{args.pool}",
        route=args.route,
        evidence_track="attribution_bridge",
        pool=args.pool,
        method="two_by_two_selector_pool_bridge",
        feature_set="selector_decomposition",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args)
        artifact = Path(str(result["artifacts"]["table"]["path"]))
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["consolidate_development_bridges", "run"]
