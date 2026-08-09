"""Materialize selected three-seed ranker scores on label-free frozen Test features."""

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
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.unified_reranking.apply_locked_matrix_cell import run as apply_cell
from tools.unified_reranking.select_primary_rankers import _augment_decisions
from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import select_order_only
from unified_reranking.training import FORMAL_SEEDS
from unified_reranking.test_access_guard import append_access_log


ROUTES = ("crog", "g1", "c1")


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE" or value.get("signature_sha256") != signature:
        raise RuntimeError("immutable label-free Test ranker output signature mismatch")
    verify_artifact_records_recursive(
        {"sources": value.get("sources"), "artifacts": value.get("artifacts")},
        name="label-free Test ranker",
        require_at_least_one=True,
    )
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _source_validation_cells(ensemble_manifest: dict[str, Any]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for record in ensemble_manifest["sources"]["matrix_manifests"]:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("selected Validation ensemble source hash mismatch")
        value = json.loads(path.read_text(encoding="utf-8"))
        configuration = value["configuration"]
        if configuration.get("mode") != "validation":
            continue
        seed = int(configuration["seed"])
        if seed in result:
            raise RuntimeError(f"duplicate Validation source cell for seed {seed}")
        result[seed] = path
    if set(result) != set(FORMAL_SEEDS):
        raise RuntimeError("selected ensemble does not contain exactly three Validation seeds")
    return result


def run(run_dir: Path) -> dict[str, Any]:
    selection_path = run_dir / "07_validation" / "selected_primary_ungated.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_LOCKED":
        raise RuntimeError("primary ungated rankers are not Validation-locked")
    denominator_path = run_dir / "01_manifests" / "paired_test.parquet"
    sample_ids = pd.read_parquet(
        denominator_path, columns=["sample_id"]
    )["sample_id"].astype(str).tolist()
    artifacts: dict[str, Any] = {}
    for route in ROUTES:
        selected = selection["selections"][route]
        validation_ensemble_path = Path(selected["validation_manifest"])
        validation_ensemble = json.loads(
            validation_ensemble_path.read_text(encoding="utf-8")
        )
        if validation_ensemble.get("status") != "COMPLETE":
            raise RuntimeError(f"selected Validation ensemble is incomplete: {route}")
        source_cells = _source_validation_cells(validation_ensemble)
        candidate_path = run_dir / "02_candidates" / f"{route}_test_top5.parquet"
        candidates = pd.read_parquet(candidate_path)
        predictions = candidates[["sample_id", "candidate_id", "native_rank"]].copy()
        applications: dict[str, Any] = {}
        for seed in FORMAL_SEEDS:
            application = apply_cell(run_dir, source_cells[seed])
            frame = pd.read_parquet(application["artifact"]["path"])[
                ["sample_id", "candidate_id", "score"]
            ]
            predictions = predictions.merge(
                frame.rename(columns={"score": f"score_seed_{seed}"}),
                on=["sample_id", "candidate_id"],
                how="left",
                validate="one_to_one",
            )
            application_manifest = (
                Path(application["artifact"]["path"]).parent / "manifest.json"
            )
            applications[str(seed)] = _record(application_manifest)
        score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
        if predictions[score_columns].isna().any().any():
            raise RuntimeError("selected Test seed scores do not cover frozen candidates")
        predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
        decisions = select_order_only(
            sample_ids, predictions, score_column="ensemble_score"
        )
        decisions = _augment_decisions(candidates, predictions, decisions)
        decisions["prediction_source"] = "test_label_free"
        output = run_dir / "08_lock" / "label_free_test_rankers" / route
        sources = {
            "selection": _record(selection_path),
            "validation_ensemble": _record(validation_ensemble_path),
            "candidates": _record(candidate_path),
            "denominator": _record(denominator_path),
            "implementation_tool": _record(Path(__file__)),
            "applications": applications,
        }
        configuration = {
            "route": route,
            "primary_track": selected["primary_track"],
            "method_code": selected["method_code"],
            "encoder": selected["encoder"],
            "loss": selected["loss"],
            "ensemble_id": selected["ensemble_id"],
            "seeds": list(FORMAL_SEEDS),
            "candidate_test_labels_read": False,
        }
        signature = canonical_sha256(
            {"configuration": configuration, "sources": sources}
        )
        marker = output / "manifest.json"
        resumed = _resume(marker, signature)
        if resumed is not None:
            artifacts[route] = resumed["artifacts"]
            continue
        prediction_path = output / "per_candidate_scores.parquet"
        decision_path = output / "per_sample_decisions.parquet"
        _atomic_parquet(prediction_path, predictions)
        _atomic_parquet(decision_path, decisions)
        manifest = {
            "status": "COMPLETE",
            "route": route,
            "candidate_test_labels_read": False,
            "label_free_test_inference": True,
            "signature_sha256": signature,
            "configuration": configuration,
            "selection": selected,
            "sources": sources,
            "artifacts": {
                "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
                "decisions": {"path": str(decision_path.resolve()), "sha256": sha256_file(decision_path)},
            },
        }
        atomic_json(marker, manifest)
        artifacts[route] = manifest["artifacts"]
    summary = {
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "routes": artifacts,
    }
    summary_path = run_dir / "08_lock" / "label_free_test_rankers" / "manifest.json"
    atomic_json(summary_path, summary)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "selected_ranker_test_ensemble_application",
            "output_manifest": str(summary_path.resolve()),
            "output_manifest_sha256": sha256_file(summary_path),
            "candidate_labels_opened_as_table": False,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P11_PRELOCK",
        substage="selected_label_free_test_rankers",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(run_dir)
        artifact = run_dir / "08_lock" / "label_free_test_rankers" / "manifest.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
