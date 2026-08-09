#!/usr/bin/env python3
"""Freeze the validation-selected SAM3 proposal method before benchmark inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.selective_sam3_vg.io import sha256_file, stable_json_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=PROJECT_ROOT / "configs/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--revision-reason",
        help="Required with --force so a superseded lock remains auditable.",
    )
    return parser.parse_args()


def _hash_files(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def _tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_hash_manifest(root: Path) -> None:
    hashes = json.loads((root / "hashes.json").read_text(encoding="utf-8"))
    mismatches = [
        name
        for name, expected in hashes.items()
        if not (root / name).is_file() or sha256_file(root / name) != expected
    ]
    if mismatches:
        raise RuntimeError(f"selector artifact checksum mismatch: {root}: {mismatches}")


def main() -> int:
    args = parse_args()
    config_root = args.config_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    locked_path = config_root / "LOCKED.yaml"
    formal_path = artifact_root / "formal_lock.json"
    microbatch_evidence = (
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/validation_pilot/"
        "text_microbatch4_prompt16_equivalence.json"
    )
    topk_resize_evidence = (
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/validation_pilot/"
        "text_topk_resize_equivalence.json"
    )
    automatic_sharing_evidence = (
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/validation_pilot/"
        "automatic_shared_model_equivalence.json"
    )
    if (locked_path.exists() or formal_path.exists()) and not args.force:
        raise FileExistsError("formal lock already exists; use a new experiment name")
    if args.force and not args.revision_reason:
        raise ValueError("--force requires --revision-reason")
    previous_lock = (
        json.loads(formal_path.read_text(encoding="utf-8"))
        if formal_path.is_file()
        else None
    )
    required = [
        artifact_root / "stage1_selector/model.pkl",
        artifact_root / "stage1_selector/oof_predictions.parquet",
        artifact_root / "stage1_selector/oof_selected_candidates.parquet",
        artifact_root / "stage1_selector/training_groups.json",
        artifact_root / "stage1_selector/validation_metrics.json",
        artifact_root / "stage1_selector/hashes.json",
        artifact_root / "final_selector/model.pkl",
        artifact_root / "final_selector/gate.json",
        artifact_root / "final_selector/oof_predictions.parquet",
        artifact_root / "final_selector/oof_selected_candidates.parquet",
        artifact_root / "final_selector/oof_gate_decisions.parquet",
        artifact_root / "final_selector/training_groups.json",
        artifact_root / "final_selector/validation_metrics.json",
        artifact_root / "final_selector/hashes.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/proposal_generation_train_manifest.parquet",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/proposal_generation_val_manifest.parquet",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage1_train/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage1/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage2_train/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage2/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/features/candidate_features_train.parquet",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/features/candidate_features_validation.parquet",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/features_stage2/candidate_features_train.parquet",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/features_stage2/candidate_features_validation.parquet",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/cache/clip_embeddings/manifest_stage1_train.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/cache/clip_embeddings/manifest_stage1_val.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/cache/clip_embeddings/manifest_stage2_train.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/cache/clip_embeddings/manifest_stage2_val.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/report/ablation_results.csv",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/report/ablation_audit.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/leakage_audit/static_scan.json",
        microbatch_evidence,
        topk_resize_evidence,
        automatic_sharing_evidence,
    ]
    required.extend(
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1"
        / namespace
        / f"{name}_{split}.json"
        for namespace in ("features", "features_stage2")
        for split in ("train", "val")
        for name in ("feature_schema", "feature_statistics", "leakage_audit")
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot lock incomplete selector artifacts: {missing}")
    _verify_hash_manifest(artifact_root / "stage1_selector")
    _verify_hash_manifest(artifact_root / "final_selector")
    leakage = json.loads(
        (
            PROJECT_ROOT
            / "outputs/sam3_proposal_bank_p90_v1/leakage_audit/static_scan.json"
        ).read_text(encoding="utf-8")
    )
    if leakage.get("status") != "PASSED" or leakage.get("violations"):
        raise RuntimeError("GT-free inference static leakage audit did not pass")
    expected = {"train": 26295, "val": 3778}
    for split, count in expected.items():
        manifest = pd.read_parquet(
            PROJECT_ROOT
            / f"outputs/sam3_proposal_bank_p90_v1/proposal_generation_{split}_manifest.parquet",
            columns=["sample_id", "terminal_status"],
        )
        if manifest["sample_id"].nunique() != count or set(
            manifest["terminal_status"].astype(str)
        ) != {"COMPLETE"}:
            raise RuntimeError(f"incomplete Stage-1 proposal manifest for {split}")
        stage1_name = "oracle_stage1_train" if split == "train" else "oracle_stage1"
        stage2_name = "oracle_stage2_train" if split == "train" else "oracle_stage2"
        stage1 = json.loads(
            (
                PROJECT_ROOT
                / f"outputs/sam3_proposal_bank_p90_v1/{stage1_name}/summary.json"
            ).read_text(encoding="utf-8")
        )
        stage2 = json.loads(
            (
                PROJECT_ROOT
                / f"outputs/sam3_proposal_bank_p90_v1/{stage2_name}/summary.json"
            ).read_text(encoding="utf-8")
        )
        if int(stage1["p_at_90_denominator"]) != count:
            raise RuntimeError(f"incomplete Stage-1 oracle for {split}")
        if int(stage2["samples"]) != count:
            raise RuntimeError(f"incomplete Stage-2 oracle for {split}")
        feature_name = "validation" if split == "val" else split
        for namespace in ("features", "features_stage2"):
            feature_file = pq.ParquetFile(
                PROJECT_ROOT
                / "outputs/sam3_proposal_bank_p90_v1"
                / namespace
                / f"candidate_features_{feature_name}.parquet"
            )
            if feature_file.num_row_groups != count:
                raise RuntimeError(f"incomplete {namespace} row groups for {split}")
        for proposal_stage in ("stage1", "stage2"):
            clip_manifest = json.loads(
                (
                    PROJECT_ROOT
                    / "outputs/sam3_proposal_bank_p90_v1/cache/clip_embeddings"
                    / f"manifest_{proposal_stage}_{split}.json"
                ).read_text(encoding="utf-8")
            )
            if (
                clip_manifest.get("status") != "COMPLETE"
                or int(clip_manifest.get("samples", 0)) != count
                or bool(clip_manifest.get("fine_tuned", True))
            ):
                raise RuntimeError(
                    f"incomplete frozen CLIP {proposal_stage} evidence for {split}"
                )
    stage1_metrics = json.loads(
        (artifact_root / "stage1_selector/validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    final_metrics = json.loads(
        (artifact_root / "final_selector/validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    for name, metrics in (("Stage-1", stage1_metrics), ("final", final_metrics)):
        if sorted(metrics.get("development_splits", [])) != ["train", "val"]:
            raise RuntimeError(f"{name} selector was not trained on train+val development data")
        if int(metrics.get("samples", 0)) != sum(expected.values()):
            raise RuntimeError(f"{name} selector sample count is incomplete")
        if int(metrics.get("selection_samples", 0)) != expected["val"]:
            raise RuntimeError(f"{name} validation selection count is incomplete")
        selector_name = "stage1_selector" if name == "Stage-1" else "final_selector"
        oof_rows = int(
            pq.ParquetFile(
                artifact_root / selector_name / "oof_predictions.parquet"
            ).metadata.num_rows
        )
        if int(metrics.get("oof_eligible_rows", -1)) != oof_rows:
            raise RuntimeError(f"{name} OOF row-count evidence is inconsistent")
        audited_rows = sum(
            int(value)
            for value in metrics.get("training_subset_audit", {})
            .get("eligible_candidates_by_split", {})
            .values()
        )
        if audited_rows != oof_rows:
            raise RuntimeError(f"{name} OOF predictions do not cover every eligible row")
        selected_rows = int(
            pq.ParquetFile(
                artifact_root / selector_name / "oof_selected_candidates.parquet"
            ).metadata.num_rows
        )
        if selected_rows != 4 * sum(expected.values()):
            raise RuntimeError(f"{name} OOF method decisions are incomplete")
    gate_rows = int(
        pq.ParquetFile(
            artifact_root / "final_selector/oof_gate_decisions.parquet"
        ).metadata.num_rows
    )
    if gate_rows != 3 * expected["val"]:
        raise RuntimeError("final validation gate decisions are incomplete")
    proposal = yaml.safe_load((config_root / "proposal_generation.yaml").read_text())
    microbatch = json.loads(microbatch_evidence.read_text(encoding="utf-8"))
    configured_microbatch = int(proposal["pcs_micro_batch_size"])
    if configured_microbatch > 1 and (
        microbatch.get("status") != "PASS_OPERATIONALLY_EQUIVALENT"
        or int(microbatch.get("safe_micro_batch_size", 1)) != configured_microbatch
        or microbatch.get("device") != "cpu"
        or microbatch.get("dtype") != "float32"
        or microbatch.get("model_revision") != proposal["model"]["revision"]
    ):
        raise RuntimeError(
            "configured PCS micro-batch lacks matching CPU/float32 equivalence evidence"
        )
    topk_resize = json.loads(topk_resize_evidence.read_text(encoding="utf-8"))
    if (
        topk_resize.get("status") != "PASS_OPERATIONALLY_EQUIVALENT"
        or int(topk_resize.get("mask_mismatches", -1)) != 0
        or topk_resize.get("device") != "cpu"
        or topk_resize.get("dtype") != "float32"
        or topk_resize.get("model_revision") != proposal["model"]["revision"]
    ):
        raise RuntimeError("PCS top-k-before-resize lacks matching equivalence evidence")
    automatic_sharing = json.loads(
        automatic_sharing_evidence.read_text(encoding="utf-8")
    )
    if (
        automatic_sharing.get("status") != "PASS_NUMERICALLY_EQUIVALENT"
        or automatic_sharing.get("device") != "cpu"
        or automatic_sharing.get("dtype") != "float32"
        or automatic_sharing.get("model_revision") != proposal["model"]["revision"]
    ):
        raise RuntimeError(
            "shared automatic/visual Tracker model lacks matching equivalence evidence"
        )
    model_root = (PROJECT_ROOT / proposal["model"]["local_path"]).resolve()
    config_files = [
        path
        for path in config_root.glob("*.yaml")
        if path.name != "LOCKED.yaml"
    ]
    code_files = [PROJECT_ROOT / "tools/export_anygrasp_inputs.py"] + list(
        (PROJECT_ROOT / "src/segmentation").rglob("*.py")
    ) + [
        path
        for path in (PROJECT_ROOT / "scripts").glob("*.py")
        if any(
            token in path.name
            for token in (
                "proposal",
                "sam3",
                "query_semantics",
                "p90_selector",
                "conservative_mask_gate",
                "second_stage",
                "final_mask",
                "stage2",
                "refinement",
                "clip_candidate",
                "canonical",
            )
        )
    ]
    artifact_files = [
        path
        for directory in (
            artifact_root / "stage1_selector",
            artifact_root / "final_selector",
        )
        for path in directory.iterdir()
        if path.is_file()
    ]
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    validation_evidence_files = [
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage1_train/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage1/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage2_train/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/oracle_stage2/summary.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/report/ablation_results.csv",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/report/ablation_audit.json",
        PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/leakage_audit/static_scan.json",
        microbatch_evidence,
        topk_resize_evidence,
        automatic_sharing_evidence,
    ]
    validation_evidence_files.extend(
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1"
        / namespace
        / f"{name}_{split}.json"
        for namespace in ("features", "features_stage2")
        for split in ("train", "val")
        for name in ("feature_schema", "feature_statistics", "leakage_audit")
    )
    validation_evidence_files.extend(
        PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1/cache/clip_embeddings"
        / f"manifest_{stage}_{split}.json"
        for stage in ("stage1", "stage2")
        for split in ("train", "val")
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "sam3_proposal_bank_p90_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "chronology": (
            "re-locked after GT-free Stage-1 benchmark proposal/feature extraction "
            "and before learned benchmark selection, Stage-2 refinement, final "
            "output lock, or ground-truth evaluation"
            if previous_lock is not None
            else "locked after grouped development/validation selection and before locked benchmark inference"
        ),
        "supersedes_formal_lock_sha256": (
            previous_lock.get("formal_lock_sha256")
            if previous_lock is not None
            else None
        ),
        "lock_revision_reason": args.revision_reason,
        "NO_TEST_GT_USED_FOR_SELECTION": True,
        "posthoc_benchmark_caveat": (
            "The 7,675-sample set previously influenced aggregate design; this is a locked post-hoc benchmark, not a pristine confirmatory test."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "model": {
            "id": "facebook/sam3",
            "revision": proposal["model"]["revision"],
            "local_path": str(model_root),
            "tree_sha256": _tree_identity(model_root),
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
        "config_hashes": _hash_files(config_files),
        "code_hashes": _hash_files(code_files),
        "selector_artifact_hashes": _hash_files(artifact_files),
        "validation_evidence_hashes": _hash_files(validation_evidence_files),
        "training_group_hashes": _hash_files(
            [
                artifact_root / "stage1_selector/training_groups.json",
                artifact_root / "final_selector/training_groups.json",
                PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/splits/train_groups.txt",
                PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/splits/validation_groups.txt",
            ]
        ),
        "git_status_at_lock": git_status,
    }
    payload["formal_lock_sha256"] = stable_json_sha256(payload)
    locked_config = {
        "schema_version": 1,
        "immutable_after_benchmark_inference": True,
        "formal_lock_sha256": payload["formal_lock_sha256"],
        "NO_TEST_GT_USED_FOR_SELECTION": True,
        "model": payload["model"],
        "configuration": {
            path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in sorted(config_files)
        },
        "stage1_selector_model_sha256": sha256_file(
            artifact_root / "stage1_selector/model.pkl"
        ),
        "final_selector_model_sha256": sha256_file(
            artifact_root / "final_selector/model.pkl"
        ),
        "final_gate_sha256": sha256_file(artifact_root / "final_selector/gate.json"),
        "evaluator_semantics": "binary prediction; strict sample IoU > threshold for P@X",
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_formal = formal_path.with_name(f".{formal_path.name}.tmp")
    temporary_formal.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_formal.replace(formal_path)
    temporary_locked = locked_path.with_name(f".{locked_path.name}.tmp")
    temporary_locked.write_text(
        yaml.safe_dump(locked_config, sort_keys=True), encoding="utf-8"
    )
    temporary_locked.replace(locked_path)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
