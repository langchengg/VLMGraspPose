#!/usr/bin/env python3
"""Apply a validation-locked VLM safe switch without reading ground truth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.experiment_lock import (  # noqa: E402
    complete_formal_stage_once,
    consume_formal_stage_once,
    selected_vlm_contract,
    verify_lock,
)
from src.grasping.reranking_v1.features import FORBIDDEN_GT_COLUMNS  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
    local_vlm_preregistration_payload,
    validate_artifact_identity,
    validate_config_identity,
    validate_matching_artifact_identity,
)
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.models import validate_candidate_contract  # noqa: E402
from src.grasping.reranking_v1.vlm_safe_switch import (  # noqa: E402
    apply_vlm_safe_switch_policy,
)


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_locked(lock: dict, path: Path, *, artifact_name: str) -> None:
    resolved = path.expanduser().resolve()
    item = lock["artifacts"].get(artifact_name, {})
    if (
        Path(str(item.get("path", ""))).resolve() != resolved
        or item.get("sha256") != sha256_file(resolved)
    ):
        raise ValueError(f"artifact is absent from or changed since lock: {resolved}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--vlm-results", type=Path, required=True)
    parser.add_argument("--formal-vlm-manifest", type=Path, required=True)
    parser.add_argument("--sample-universe", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--experiment-lock", type=Path, required=True)
    parser.add_argument("--formal-inference-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    candidate_path = args.per_candidate.expanduser().resolve()
    results_path = args.vlm_results.expanduser().resolve()
    vlm_manifest_path = args.formal_vlm_manifest.expanduser().resolve()
    universe_path = args.sample_universe.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve()
    lock_path = args.experiment_lock.expanduser().resolve()
    formal_manifest_path = args.formal_inference_manifest.expanduser().resolve()
    root = args.output_root.expanduser().resolve()
    lock = verify_lock(lock_path)
    config_path = Path(lock["artifacts"]["config"]["path"]).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("locked VLM safe-switch config must be a mapping")
    validate_config_identity(config)
    preregistered = local_vlm_preregistration_payload(config)
    assert_locked(lock, candidate_path, artifact_name="test_per_candidate")
    assert_locked(
        lock, selection_path, artifact_name="vlm_safe_switch_selection"
    )
    assert_locked(lock, universe_path, artifact_name="test_sample_universe")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        selection, context="formal VLM safe-switch selection"
    )
    validate_matching_artifact_identity(
        selection,
        lock,
        context="formal VLM safe-switch selection",
    )
    if (
        Path(str(selection.get("config", ""))).resolve() != config_path
        or selection.get("config_sha256") != sha256_file(config_path)
        or {
            key: selection.get(key)
            for key in (
                "geometry_risk_column",
                "geometry_risk_threshold",
                "geometry_semantics",
            )
        }
        != {
            key: preregistered[key]
            for key in (
                "geometry_risk_column",
                "geometry_risk_threshold",
                "geometry_semantics",
            )
        }
    ):
        raise ValueError(
            "formal VLM safe-switch geometry differs from preregistered config"
        )
    selected_vlm = selected_vlm_contract(lock)
    source_variant = selected_vlm["source_variant"]
    expected_source_method = selected_vlm["source_method"]
    locked_vlm = lock.get("ranking_parameters", {}).get("vlm_safe_switch", {})
    if (
        selection.get("selection_split") != "validation"
        or selection.get("source_method") != expected_source_method
        or selection.get("source_variant") != source_variant
        or selection.get("method") != locked_vlm.get("method")
    ):
        raise ValueError(
            "VLM safe-switch source differs from the validation-selected VLM "
            "in the immutable lock"
        )
    expected_stage = (
        "VLM_VISUAL" if source_variant == "visual" else "VLM_METADATA"
    )
    formal_test_ledger = lock_path.with_name(
        f"{lock_path.name}.FORMAL_TEST_STARTED.json"
    )
    formal_test = json.loads(formal_test_ledger.read_text(encoding="utf-8"))
    validate_matching_artifact_identity(
        formal_test,
        lock,
        context="formal reranker start ledger",
    )
    if formal_manifest_path != (
        Path(str(formal_test["output_root"])) / "inference_manifest.json"
    ).resolve():
        raise ValueError("formal inference manifest is outside guarded output")
    formal = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        formal, context="formal reranker inference manifest"
    )
    validate_matching_artifact_identity(
        formal,
        lock,
        context="formal reranker inference manifest",
    )
    if formal.get("lock_content_sha256") != lock["manifest_content_sha256"]:
        raise ValueError("formal inference manifest belongs to another lock")
    if not Path(formal["predictions"]).is_file():
        raise ValueError("locked formal reranker predictions are absent")
    if sha256_file(Path(formal["predictions"])) != formal["predictions_sha256"]:
        raise ValueError("locked formal reranker predictions changed")
    vlm_stage_ledger = lock_path.with_name(
        f"{lock_path.name}.FORMAL_{expected_stage}_STARTED.json"
    )
    vlm_stage = json.loads(vlm_stage_ledger.read_text(encoding="utf-8"))
    validate_matching_artifact_identity(
        vlm_stage,
        lock,
        context="formal VLM start ledger",
    )
    if vlm_manifest_path != (
        Path(str(vlm_stage["output_root"])) / "formal_vlm_manifest.json"
    ).resolve():
        raise ValueError("formal VLM manifest is outside selected guarded stage")
    formal_vlm = json.loads(vlm_manifest_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        formal_vlm, context="formal VLM execution manifest"
    )
    validate_matching_artifact_identity(
        formal_vlm,
        lock,
        context="formal VLM execution manifest",
    )
    if (
        formal_vlm.get("completed") is not True
        or formal_vlm.get("stage") != expected_stage
        or formal_vlm.get("variant") != source_variant
        or formal_vlm.get("source_method") != expected_source_method
        or Path(
            str(formal_vlm.get("validation_selection_path", ""))
        ).resolve()
        != Path(selected_vlm["validation_selection_path"]).resolve()
        or formal_vlm.get("validation_selection_sha256")
        != selected_vlm["validation_selection_sha256"]
        or formal_vlm.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or Path(str(formal_vlm.get("formal_inference_manifest", ""))).resolve()
        != formal_manifest_path
        or formal_vlm.get("formal_inference_manifest_sha256")
        != sha256_file(formal_manifest_path)
        or formal_vlm.get("model_digest") != selection.get("vlm_model_digest")
        or formal_vlm.get("model_digest") != lock.get("vlm_model_digest")
    ):
        raise ValueError("VLM results are not a completed guarded formal stage")
    if (
        Path(str(formal_vlm.get("results_jsonl"))).resolve() != results_path
        or sha256_file(results_path) != formal_vlm.get("results_jsonl_sha256")
    ):
        raise ValueError("formal VLM result artifact changed")
    vlm_completion_path = lock_path.with_name(
        f"{lock_path.name}.FORMAL_{expected_stage}_COMPLETED.json"
    )
    vlm_completion = json.loads(
        vlm_completion_path.read_text(encoding="utf-8")
    )
    validate_matching_artifact_identity(
        vlm_completion,
        lock,
        context="formal VLM completion ledger",
    )
    if (
        vlm_completion.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or vlm_completion.get("stage") != expected_stage
        or Path(str(vlm_completion.get("start_ledger", ""))).resolve()
        != vlm_stage_ledger
        or vlm_completion.get("start_ledger_sha256")
        != sha256_file(vlm_stage_ledger)
        or Path(str(vlm_completion.get("manifest_path", ""))).resolve()
        != vlm_manifest_path
        or vlm_completion.get("manifest_sha256")
        != sha256_file(vlm_manifest_path)
    ):
        raise ValueError("selected formal VLM completion ledger is invalid")

    candidates = pd.read_parquet(candidate_path)
    leaked = sorted(set(candidates.columns) & set(FORBIDDEN_GT_COLUMNS))
    if leaked:
        raise ValueError(f"GT columns reached VLM safe-switch inference: {leaked}")
    validate_candidate_contract(
        candidates, require_label=False, allowed_splits={"test"}
    )
    universe = pd.read_parquet(universe_path)
    if "sample_id" not in universe:
        raise ValueError("sample universe requires sample_id")
    universe_ids = universe["sample_id"].astype(str)
    if universe_ids.duplicated().any():
        raise ValueError("sample universe contains duplicate sample IDs")
    if len(universe_ids) != int(lock["expected_test_sample_count"]):
        raise ValueError("sample universe disagrees with experiment lock")
    if not set(candidates["sample_id"].astype(str)) <= set(universe_ids):
        raise ValueError("candidate samples fall outside formal sample universe")
    expected_locked = {
        "method": selection["method"],
        "threshold_kind": selection["threshold_kind"],
        "threshold": selection.get("threshold"),
        "geometry_risk_column": selection["geometry_risk_column"],
        "geometry_risk_threshold": selection["geometry_risk_threshold"],
        "geometry_semantics": selection["geometry_semantics"],
    }
    if locked_vlm != expected_locked:
        raise ValueError("VLM safe-switch selection disagrees with experiment lock")
    geometry_column = str(selection["geometry_risk_column"])
    if geometry_column not in candidates.columns:
        raise ValueError(f"locked geometry feature missing: {geometry_column}")

    results = jsonl(results_path)
    if len(results) != len({str(row["sample_id"]) for row in results}):
        raise ValueError("VLM results contain duplicate sample IDs")
    if {str(row["sample_id"]) for row in results} != set(universe_ids):
        raise ValueError("formal VLM results do not cover the complete test universe")
    if int(formal_vlm.get("sample_count", -1)) != len(universe_ids):
        raise ValueError("formal VLM manifest sample count mismatch")

    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--per-candidate",
        str(candidate_path),
        "--vlm-results",
        str(results_path),
        "--formal-vlm-manifest",
        str(vlm_manifest_path),
        "--sample-universe",
        str(universe_path),
        "--selection",
        str(selection_path),
        "--experiment-lock",
        str(lock_path),
        "--formal-inference-manifest",
        str(formal_manifest_path),
        "--output-root",
        str(root),
    ]
    consume_formal_stage_once(
        lock_path,
        stage="VLM_APPLY",
        output_root=root,
        invocation=invocation,
    )
    completed_manifest_path = root / "manifest.json"
    if completed_manifest_path.is_file():
        existing = json.loads(completed_manifest_path.read_text(encoding="utf-8"))
        validate_artifact_identity(
            existing, context="completed formal VLM apply manifest"
        )
        for path_key, hash_key in (
            ("predictions", "predictions_sha256"),
            ("decisions", "decisions_sha256"),
        ):
            artifact = Path(str(existing[path_key])).resolve()
            if (
                not artifact.is_file()
                or sha256_file(artifact) != existing[hash_key]
            ):
                raise ValueError(f"formal VLM apply artifact changed: {artifact}")
        if existing.get("lock_content_sha256") != lock["manifest_content_sha256"]:
            raise FileExistsError("formal VLM apply output belongs to another lock")
        complete_formal_stage_once(
            lock_path,
            stage="VLM_APPLY",
            manifest_path=completed_manifest_path,
        )
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    prediction_frame, decision_frame = apply_vlm_safe_switch_policy(
        candidates, results, selection
    )

    root.mkdir(parents=True, exist_ok=True)
    prediction_path = root / "predictions.parquet"
    decision_path = root / "decisions.parquet"
    prediction_temporary = root / f".predictions.{os.getpid()}.tmp.parquet"
    decision_temporary = root / f".decisions.{os.getpid()}.tmp.parquet"
    prediction_frame.to_parquet(
        prediction_temporary, index=False, compression="zstd"
    )
    decision_frame.to_parquet(
        decision_temporary, index=False, compression="zstd"
    )
    os.replace(prediction_temporary, prediction_path)
    os.replace(decision_temporary, decision_path)
    manifest = {
        "schema_version": 1,
        **identity_payload(),
        "experiment_lock": str(lock_path),
        "lock_content_sha256": lock["manifest_content_sha256"],
        "formal_inference_manifest": str(formal_manifest_path),
        "formal_inference_manifest_sha256": sha256_file(formal_manifest_path),
        "per_candidate": str(candidate_path),
        "per_candidate_sha256": sha256_file(candidate_path),
        "vlm_results": str(results_path),
        "vlm_results_sha256": sha256_file(results_path),
        "formal_vlm_manifest": str(vlm_manifest_path),
        "formal_vlm_manifest_sha256": sha256_file(vlm_manifest_path),
        "formal_vlm_stage": expected_stage,
        "source_method": expected_source_method,
        "source_variant": source_variant,
        "validation_selection_path": selected_vlm[
            "validation_selection_path"
        ],
        "validation_selection_sha256": selected_vlm[
            "validation_selection_sha256"
        ],
        "sample_universe": str(universe_path),
        "sample_universe_sha256": sha256_file(universe_path),
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "decisions": str(decision_path),
        "decisions_sha256": sha256_file(decision_path),
        "sample_count": len(universe_ids),
        "nonempty_sample_count": len(decision_frame),
        "valid_empty_sample_count": len(universe_ids) - len(decision_frame),
        "prediction_rows": len(prediction_frame),
        "switch_count": int(decision_frame["switch_applied"].astype(bool).sum()),
        "candidate_pool_modified": False,
        "candidate_identity_invariant": True,
        "GT_columns_used": False,
    }
    temporary = root / f".manifest.json.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(root / "manifest.json")
    complete_formal_stage_once(
        lock_path,
        stage="VLM_APPLY",
        manifest_path=root / "manifest.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
