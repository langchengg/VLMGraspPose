"""Experiment initialization and deterministic scene-cluster bootstrap."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .experiment_store import ExperimentStore, ManifestCountMismatch


REQUIRED_LIMITATIONS = {
    "single-view TSDF adaptation",
    "no 6-DoF ground truth in OCID-VLG",
    "no robot execution validation",
}


def _mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row):
        return asdict(row)
    raise TypeError(f"rows must be mappings or dataclasses, got {type(row)!r}")


def validate_truthfulness_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate provenance and limitation fields required for an honest run."""

    result = dict(metadata)
    required_nonempty = (
        "repository_url",
        "repository_branch",
        "repository_commit",
        "checkpoint_sha256",
    )
    missing = [name for name in required_nonempty if not str(result.get(name, "")).strip()]
    if missing:
        raise ValueError(f"missing reproducibility metadata: {', '.join(missing)}")
    if result["repository_url"] != "https://github.com/ethz-asl/vgn":
        raise ValueError("repository_url must identify the official ETH Zürich VGN repository")
    if result["repository_branch"] != "corl2020":
        raise ValueError("repository_branch must be corl2020 for this experiment")
    commit = str(result["repository_commit"])
    if len(commit) != 40 or any(character not in "0123456789abcdefABCDEF" for character in commit):
        raise ValueError("repository_commit must be a full 40-character Git commit hash")
    checkpoint_hash = str(result["checkpoint_sha256"])
    if len(checkpoint_hash) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in checkpoint_hash
    ):
        raise ValueError("checkpoint_sha256 must be a 64-character SHA256 digest")
    if result.get("tsdf_mode") != "single_view_adaptation":
        raise ValueError("tsdf_mode must disclose single_view_adaptation")
    if result.get("score_source") != "official_vgn_processed_quality":
        raise ValueError("score_source must be official_vgn_processed_quality")
    if result.get("custom_reranking") is not False:
        raise ValueError("custom_reranking must be exactly false")
    limitations = result.get("limitations")
    if not isinstance(limitations, (list, tuple, set)):
        raise ValueError("limitations must be a list-like value")
    missing_limitations = REQUIRED_LIMITATIONS - {str(value) for value in limitations}
    if missing_limitations:
        raise ValueError(
            "limitations omit required disclosures: " + ", ".join(sorted(missing_limitations))
        )
    return result


def manifest_fingerprint(rows: Iterable[Mapping[str, Any] | Any]) -> str:
    """Hash stable manifest identity fields without depending on file ordering."""

    identities = []
    for raw in rows:
        row = _mapping(raw)
        identities.append(
            (
                int(row["dataset_index"]),
                str(row["sample_id"]),
                str(row["scene_id"]),
            )
        )
    identities.sort()
    digest = hashlib.sha256()
    for dataset_index, sample_id, scene_id in identities:
        digest.update(f"{dataset_index}\0{sample_id}\0{scene_id}\n".encode("utf-8"))
    return digest.hexdigest()


def bootstrap_experiment(
    store: ExperimentStore,
    rows: Iterable[Mapping[str, Any] | Any],
    run_metadata: Mapping[str, Any],
    *,
    manifest_count: int | None = None,
) -> dict[str, Any]:
    """Validate provenance and register the exact full experiment manifest."""

    materialized = list(rows)
    expected = len(materialized) if manifest_count is None else int(manifest_count)
    if len(materialized) != expected:
        raise ManifestCountMismatch(
            f"materialized manifest has {len(materialized)} rows, expected {expected}"
        )
    metadata = validate_truthfulness_metadata(run_metadata)
    metadata["manifest_count"] = expected
    metadata["manifest_identity_sha256"] = manifest_fingerprint(materialized)
    store.initialize_run(metadata, expected)
    registered = store.register_samples(materialized, expected_count=expected)
    scenes = {str(_mapping(row)["scene_id"]) for row in materialized}
    return {
        "run_id": store.run_id,
        "manifest_count": expected,
        "registered_sample_count": registered,
        "scene_cluster_count": len(scenes),
        "manifest_identity_sha256": metadata["manifest_identity_sha256"],
    }


def _group_by_cluster(
    rows: Sequence[Mapping[str, Any] | Any], cluster_key: str
) -> dict[str, list[Mapping[str, Any]]]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = _mapping(raw)
        cluster = str(row.get(cluster_key, "")).strip()
        if not cluster:
            raise ValueError(f"row is missing non-empty cluster field {cluster_key!r}")
        clusters[cluster].append(row)
    if not clusters:
        raise ValueError("scene-cluster bootstrap requires at least one row")
    return dict(clusters)


def _replicate_seed(seed: int, replicate_index: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{int(replicate_index)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def select_scene_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any] | Any],
    *,
    seed: int = 42,
    replicate_index: int = 0,
    cluster_key: str = "scene_id",
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Draw scene IDs with replacement and include every row in each draw.

    Returning selected cluster IDs makes duplicate scene draws explicit and
    lets callers audit that language expressions from one scene stay grouped.
    """

    clusters = _group_by_cluster(rows, cluster_key)
    cluster_ids = sorted(clusters)
    rng = random.Random(_replicate_seed(seed, replicate_index))
    selected_ids = [rng.choice(cluster_ids) for _ in range(len(cluster_ids))]
    selected_rows: list[Mapping[str, Any]] = []
    for cluster_id in selected_ids:
        selected_rows.extend(clusters[cluster_id])
    return selected_rows, selected_ids


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_interval(
    rows: Sequence[Mapping[str, Any] | Any],
    value: str | Callable[[Mapping[str, Any]], float | None],
    *,
    replicates: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
    cluster_key: str = "scene_id",
) -> dict[str, Any]:
    """Percentile interval for a mean, resampling exact scene clusters."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    def extract(row: Mapping[str, Any]) -> float | None:
        raw = value(row) if callable(value) else row.get(value)
        if raw is None or raw == "":
            return None
        number = float(raw)
        return number if math.isfinite(number) else None

    source_values = [extract(_mapping(row)) for row in rows]
    source_finite = [number for number in source_values if number is not None]
    if not source_finite:
        raise ValueError("no finite values are available for bootstrap")
    estimates: list[float] = []
    for replicate_index in range(replicates):
        selected, _ = select_scene_cluster_bootstrap(
            rows,
            seed=seed,
            replicate_index=replicate_index,
            cluster_key=cluster_key,
        )
        values = [extract(row) for row in selected]
        finite = [number for number in values if number is not None]
        if finite:
            estimates.append(sum(finite) / len(finite))
    if not estimates:
        raise ValueError("bootstrap replicates contain no finite values")
    alpha = 1.0 - confidence
    return {
        "estimate": sum(source_finite) / len(source_finite),
        "ci_lower": _percentile(estimates, alpha / 2.0),
        "ci_upper": _percentile(estimates, 1.0 - alpha / 2.0),
        "confidence": confidence,
        "method": "scene_cluster_percentile_bootstrap",
        "replicates": replicates,
        "seed": int(seed),
        "cluster_key": cluster_key,
        "cluster_count": len(_group_by_cluster(rows, cluster_key)),
        "finite_sample_count": len(source_finite),
    }
