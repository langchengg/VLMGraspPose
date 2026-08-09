#!/usr/bin/env python3
"""Start and finish the fail-closed provenance transaction for trainable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json  # noqa: E402
from src.grasping.g1_c1_safe_rerank.contracts import sha256_file  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import (  # noqa: E402
    BACKENDS,
    METHODS,
    POOLS,
    SEEDS,
)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("start", "finish"))
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _code_paths() -> list[Path]:
    return sorted(
        {
            *(
                path
                for root in (
                    PROJECT_ROOT / "src" / "grasping" / "g1_c1_safe_rerank",
                    PROJECT_ROOT / "tools" / "g1_c1_rerank",
                    REPOSITORY_ROOT / "reranking",
                )
                for path in root.rglob("*.py")
                if path.is_file()
            )
        }
    )


def _snapshot() -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
        for path in _code_paths()
    }


def _main_models(run: Path) -> list[Path]:
    return [
        run / "05_models" / backend / pool / f"{method}_seed{seed}.joblib"
        for backend in BACKENDS
        for pool in POOLS
        for method in METHODS
        for seed in SEEDS
    ]


def _temperature_models(run: Path) -> list[Path]:
    return sorted((run / "05_models" / "r5_temperature_cv").rglob("*.joblib"))


def _eligible_artifacts(run: Path) -> list[Path]:
    roots = (
        run / "04_calibration",
        run / "05_models",
        run / "06_oof",
        run / "07_validation",
        run / "checkpoints",
    )
    return sorted(
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    )


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _start(run: Path, destination: Path) -> None:
    main_existing = [path for path in _main_models(run) if path.is_file()]
    temperature_existing = _temperature_models(run)
    validation_roots = (
        run / "07_validation" / "candidate_scores",
        run / "07_validation" / "metrics",
        run / "07_validation" / "outcomes",
        run / "07_validation" / "tables",
        run / "07_validation" / "r1_train_cv",
        run / "07_validation" / "r5_temperature_cv",
        run / "07_validation" / "loss_selection",
    )
    active_validation = [
        path for root in validation_roots for path in root.rglob("*") if path.is_file()
    ]
    if main_existing or temperature_existing or active_validation:
        raise RuntimeError("matrix provenance start requires zero eligible model/validation artifacts")
    payload = {
        "status": "RUNNING",
        "started_local": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "command": " ".join(sys.argv),
        "clean_start_counts": {
            "main_matrix_models": 0,
            "r5_temperature_cv_models": 0,
            "candidate_scores": 0,
            "metrics": 0,
        },
        "trainable_code_snapshot": _snapshot(),
        "completion_requirement": "finish must verify unchanged code, 180 main models, 60 nested R5 temperature-CV models, and hash every active trainable artifact",
    }
    _exclusive_json(destination, payload)


def _finish(run: Path, destination: Path) -> None:
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if payload.get("status") not in {"RUNNING", "PASS"}:
        raise RuntimeError("matrix provenance finish requires RUNNING or already verified PASS status")
    snapshot = payload.get("trainable_code_snapshot", {})
    if not snapshot or snapshot != _snapshot():
        raise RuntimeError("trainable code changed after provenance start")
    main_models = _main_models(run)
    missing = [str(path) for path in main_models if not path.is_file()]
    if missing:
        raise RuntimeError(f"main matrix model inventory incomplete: {len(missing)}")
    temperature_models = _temperature_models(run)
    expected_temperature = len(BACKENDS) * len(POOLS) * 5 * 3
    if len(temperature_models) != expected_temperature:
        raise RuntimeError(
            f"R5 temperature-CV model inventory incomplete: {len(temperature_models)} != {expected_temperature}"
        )
    artifacts = [_artifact(path) for path in _eligible_artifacts(run)]
    digest = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("status") == "PASS":
        expected_counts = {
            "main_matrix_models": len(main_models),
            "r5_temperature_cv_models": len(temperature_models),
            "all_active_trainable_artifacts": len(artifacts),
        }
        if payload.get("completion_counts") != expected_counts:
            raise RuntimeError("completed matrix provenance counts drifted")
        if payload.get("matrix_artifacts") != artifacts:
            raise RuntimeError("completed matrix provenance artifact inventory drifted")
        if payload.get("matrix_artifact_inventory_sha256") != digest:
            raise RuntimeError("completed matrix provenance inventory digest drifted")
        return
    payload.update(
        {
            "status": "PASS",
            "completed_local": datetime.now().astimezone().isoformat(),
            "completion_counts": {
                "main_matrix_models": len(main_models),
                "r5_temperature_cv_models": len(temperature_models),
                "all_active_trainable_artifacts": len(artifacts),
            },
            "matrix_artifact_inventory_sha256": digest,
            "matrix_artifacts": artifacts,
            "quarantined_artifacts_eligible": False,
        }
    )
    atomic_json(destination, payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    run = args.run_dir.expanduser().resolve()
    destination = run / "00_audit" / "FINAL_MATRIX_CODE_PROVENANCE.json"
    if args.stage == "start":
        _start(run, destination)
    else:
        if not destination.is_file():
            raise FileNotFoundError(destination)
        _finish(run, destination)
    print(json.dumps({"status": args.stage.upper(), "path": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
