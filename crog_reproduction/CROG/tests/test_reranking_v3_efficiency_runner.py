from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from failure_analysis.reranking_v3.efficiency_runner import (
    load_feature_catalog_descriptor,
    load_sample_ids,
    load_v2_prior_npz,
    run_selected_model_efficiency_audit,
)
from failure_analysis.reranking_v3.experiment_config import ENSEMBLE_SEEDS
from failure_analysis.reranking_v3.models.fullchain_ranker import FullChainRanker


class _OneSampleCatalog:
    def __init__(self, root: Path):
        self.sample_id = "multiple:val:00000001"
        self.ids = [f"candidate-{value}" for value in range(5)]
        self.checksums = [f"checksum-{value}" for value in range(5)]
        self.schema = {
            "head_feature_names": ["g0_q_raw", "g1_peak"],
            "depth_feature_names": ["relative_depth"],
        }
        self.locations = {self.sample_id: object()}
        self.candidate_records = {
            self.sample_id: {
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "candidate_checksum": checksum,
                        "q_rank": index,
                    }
                    for index, (candidate_id, checksum) in enumerate(
                        zip(self.ids, self.checksums, strict=True)
                    )
                ],
            },
        }
        feature_dir = root / "features"
        feature_dir.mkdir()
        (feature_dir / "shard.npz").write_bytes(b"synthetic-feature-bytes")
        candidate_path = root / "candidates.jsonl"
        candidate_path.write_text("{}\n", encoding="utf-8")
        self.artifact_dirs = (feature_dir,)
        self.head_override_dirs = ()
        self.candidate_feature_paths = (candidate_path,)

    def provenance(self):
        return {"kind": "synthetic_one_sample_catalog", "row_count": 1}

    def candidate_identity(self, allowed_ids):
        if set(allowed_ids) != {self.sample_id}:
            raise ValueError("unexpected test cohort")
        return {
            self.sample_id: {
                "candidate_ids": self.ids,
                "candidate_checksums": self.checksums,
            },
        }

    def iter_batches(
        self, *, allowed_ids, labels, priors, normalizers, batch_size, seed, shuffle, array_keys,
    ):
        assert set(allowed_ids) == {self.sample_id}
        assert labels is None
        assert set(priors) == {self.sample_id}
        assert tuple(array_keys) == ("head_features",)
        q = np.asarray([[0.9, 0.7, 0.5, 0.3, 0.1]], dtype=np.float32)
        head = np.stack((q, np.square(q)), axis=-1)
        yield {
            "sample_ids": np.asarray([self.sample_id]),
            "records": [{"candidate_ids": self.ids, "candidate_checksums": self.checksums}],
            "head_features": head,
            "q": q,
            "prior": np.asarray([priors[self.sample_id]], dtype=np.float32),
        }


def _write_checkpoints(tmp_path: Path) -> list[Path]:
    config = {
        "head_dim": 2,
        "depth_dim": 1,
        "hidden_dim": 8,
        "alpha": 0.5,
        "use_depth": False,
        "use_prior": True,
        "use_crop": False,
        "use_latent": False,
        "text_mode": "none",
        "use_attention": False,
        "use_set": False,
        "latent_layers": "all",
        "dropout": 0.0,
    }
    torch.manual_seed(1)
    model = FullChainRanker(**config)
    normalizers = {
        "head_features": {
            "median": [0.0, 0.0], "mean": [0.0, 0.0], "scale": [1.0, 1.0],
        },
        "depth_features": {
            "median": [0.0], "mean": [0.0], "scale": [1.0],
        },
        "prior": {
            "median": [0.0] * 80, "mean": [0.0] * 80, "scale": [1.0] * 80,
        },
    }
    paths = []
    for seed in ENSEMBLE_SEEDS:
        path = tmp_path / f"seed-{seed}.pt"
        torch.save({
            "schema_version": "3.0.0",
            "kind": "fcer_checkpoint",
            "status": "complete",
            "state_dict": model.state_dict(),
            "config": config,
            "normalizers": normalizers,
            "head_mask": [True, True],
            "seed": seed,
            "parameter_count": sum(value.numel() for value in model.parameters()),
        }, path)
        paths.append(path)
    return paths


def test_real_one_sample_cpu_selected_ensemble_efficiency_audit(tmp_path: Path) -> None:
    catalog = _OneSampleCatalog(tmp_path)
    checkpoints = _write_checkpoints(tmp_path)
    priors = {catalog.sample_id: np.zeros((5, 80), dtype=np.float32)}
    output = tmp_path / "efficiency.json"
    report = run_selected_model_efficiency_audit(
        catalog=catalog,
        sample_ids={catalog.sample_id},
        priors=priors,
        checkpoint_paths=checkpoints,
        output_path=output,
        device="cpu",
        batch_size=1,
        warmup=1,
        repeat=1,
    )
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == report
    assert report["labels_read"] is False
    assert report["coverage"]["candidate_count"] == 5
    assert report["coverage"]["ensemble_identity_equal_across_seeds"] is True
    assert report["protocol"]["checkpoint_loading_in_timed_region"] is False
    assert report["efficiency"]["per_expression_latency"]["median"] > 0.0
    assert report["efficiency"]["throughput_samples_per_second"]["mean"] > 0.0
    assert report["memory"]["mps_unified_memory_boundary_maximum"] is None
    assert report["model"]["ensemble_size"] == 3
    assert report["model"]["parameter_count_per_seed"] > 0
    assert report["disk"]["checkpoints"]["total_bytes"] == sum(path.stat().st_size for path in checkpoints)
    assert report["disk"]["feature_catalog"]["total_bytes"] > 0
    assert report["disk"]["output_bytes"] > 0
    assert len(report["content_sha256"]) == 64
    with pytest.raises(FileExistsError):
        run_selected_model_efficiency_audit(
            catalog=catalog,
            sample_ids={catalog.sample_id},
            priors=priors,
            checkpoint_paths=checkpoints,
            output_path=output,
            device="cpu",
            warmup=1,
            repeat=1,
        )


def test_input_loaders_reject_duplicate_ids_and_align_prior_subset(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps({"sample_ids": ["a", "b"]}), encoding="utf-8")
    assert load_sample_ids(ids_path) == {"a", "b"}
    ids_path.write_text(json.dumps(["a", "a"]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_sample_ids(ids_path)

    prior_path = tmp_path / "prior.npz"
    np.savez_compressed(
        prior_path,
        sample_ids=np.asarray(["unused", "a"]),
        prior=np.zeros((2, 5, 80), dtype=np.float32),
        valid=np.asarray([False, True]),
    )
    priors, evidence = load_v2_prior_npz(prior_path, sample_ids={"a"})
    assert set(priors) == {"a"}
    assert evidence["unused_source_row_count"] == 1
    assert evidence["all_selected_rows_valid"] is True


def test_catalog_descriptor_rejects_non_label_free_unknown_fields(tmp_path: Path) -> None:
    descriptor = tmp_path / "catalog.json"
    descriptor.write_text(json.dumps({
        "artifact_dirs": ["features"],
        "candidate_feature_paths": ["candidates.jsonl"],
        "labels": ["forbidden.jsonl"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown feature catalog descriptor fields"):
        load_feature_catalog_descriptor(descriptor)
