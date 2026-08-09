from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .feature_store import FeatureCatalog, fit_streaming_normalizers
from .models.fullchain_ranker import FullChainRanker
from .schema import artifact_identity, atomic_write_json
from .training import fullchain_loss


MODEL_CONFIG_KEYS = {
    "hidden_dim", "alpha", "use_depth", "use_prior", "use_crop",
    "use_latent", "text_mode", "use_attention", "use_set", "latent_layers", "dropout", "head_dim", "depth_dim",
}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return result


def head_group_mask(feature_names: list[str], groups: list[int]) -> np.ndarray:
    allowed = {f"g{int(value)}_" for value in groups}
    mask = np.asarray(
        [any(str(name).startswith(prefix) for prefix in allowed) for name in feature_names],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError("head feature group selection is empty")
    return mask


def model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


def required_array_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    result = ["head_features"]
    if config.get("use_depth", False): result.append("depth_features")
    if config.get("use_latent", True) or config.get("text_mode", "token") == "token": result.append("latent_rois")
    if config.get("use_attention", True): result.append("attention_rois")
    if config.get("text_mode", "token") == "token": result.extend(("tokens", "sentence", "dynamic", "token_ids"))
    elif config.get("text_mode") == "sentence": result.append("sentence")
    if config.get("use_crop", True): result.append("crops")
    return tuple(dict.fromkeys(result))


def _normalizers_to_json(normalizers: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    return {
        name: {key: np.asarray(value).tolist() for key, value in fields.items()}
        for name, fields in normalizers.items()
    }


def _normalizers_from_json(payload: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
        for name, fields in payload.items()
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def torch_feature_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    head_mask: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    head = np.asarray(batch["head_features"], dtype=np.float32).copy()
    head[:, :, ~head_mask] = 0.0
    mapping = {"head": head, "q": batch["q"]}
    source_to_target = {"depth_features":"depth","latent_rois":"latent","attention_rois":"attention_roi","tokens":"tokens","sentence":"sentence","dynamic":"dynamic","token_ids":"token_ids","crops":"crops"}
    for source in required_array_keys(config):
        if source != "head_features": mapping[source_to_target[source]] = batch[source]
    if config.get("use_prior", True): mapping["prior"] = batch["prior"]
    inputs = {}
    for name, value in mapping.items():
        tensor = torch.from_numpy(np.asarray(value))
        if name != "token_ids":
            tensor = tensor.float()
        inputs[name] = tensor.to(device)
    labels = None
    if "labels" in batch:
        labels = torch.from_numpy(np.asarray(batch["labels"], dtype=np.float32)).to(device)
    return inputs, labels


def train_ranker_streaming(
    *,
    catalog: FeatureCatalog,
    train_ids: set[str],
    labels: Mapping[str, np.ndarray],
    priors: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    output_path: str | Path,
    seed: int,
    device: str,
    epochs: int = 5,
    batch_size: int = 16,
    normalizers: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact["status"] != "complete":
            raise ValueError(f"incomplete checkpoint cannot be resumed: {path}")
        return {"checkpoint": artifact_identity(path), "history": artifact["history"], "resumed": True, "parameters": int(artifact["parameter_count"])}
    if set(labels) != train_ids:
        raise ValueError("training label lookup must exactly match train IDs")
    if set(priors) != train_ids:
        raise ValueError("training prior lookup must exactly match train IDs")
    seed_everything(seed)
    torch_device = resolve_device(device)
    fitted = (
        fit_streaming_normalizers(catalog, allowed_ids=train_ids, priors=priors)
        if normalizers is None else
        {name: {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()} for name, fields in normalizers.items()}
    )
    mask = head_group_mask(catalog.schema["head_feature_names"], list(config["head_groups"]))
    runtime_config=dict(config)
    runtime_config["head_dim"]=len(catalog.schema["head_feature_names"])
    runtime_config["depth_dim"]=len(catalog.schema["depth_feature_names"])
    model = FullChainRanker(**model_kwargs(runtime_config)).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    history = []
    for epoch in range(int(epochs)):
        model.train()
        parts = []
        seen = 0
        for batch in catalog.iter_batches(
            allowed_ids=train_ids,
            labels=labels,
            priors=priors,
            normalizers=fitted,
            batch_size=batch_size,
            seed=seed + epoch,
            shuffle=True,
            array_keys=required_array_keys(config),
        ):
            inputs, target = torch_feature_batch(batch, device=torch_device, head_mask=mask, config=config)
            if target is None:
                raise AssertionError("training target was not joined")
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            loss, loss_parts = fullchain_loss(
                output,
                target,
                beta_abs=float(config.get("beta_abs", 1.0)),
                beta_pair=float(config.get("beta_pair", 0.25)),
                beta_any=float(config.get("beta_any", 0.5)),
                beta_res=float(config.get("beta_res", 0.01)),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            parts.append(loss_parts)
            seen += len(batch["sample_ids"])
        if seen != len(train_ids):
            raise AssertionError(f"training epoch silently skipped rows: {seen}/{len(train_ids)}")
        history.append({key: float(np.mean([value[key] for value in parts])) for key in parts[0]})
    payload = {
        "schema_version": "3.0.0",
        "kind": "fcer_checkpoint",
        "status": "complete",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": runtime_config,
        "normalizers": _normalizers_to_json(fitted),
        "head_mask": mask.tolist(),
        "seed": int(seed),
        "epochs": int(epochs),
        "train_count": len(train_ids),
        "train_ids_sha256": __import__("hashlib").sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "history": history,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "feature_provenance": catalog.provenance(),
    }
    _atomic_torch_save(payload, path)
    return {"checkpoint": artifact_identity(path), "history": history, "resumed": False, "parameters": payload["parameter_count"]}


@torch.no_grad()
def predict_ranker_streaming(
    *,
    catalog: FeatureCatalog,
    sample_ids: set[str],
    priors: Mapping[str, np.ndarray],
    checkpoint_path: str | Path,
    device: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if artifact.get("status") != "complete":
        raise ValueError("FCER checkpoint is incomplete")
    model = FullChainRanker(**model_kwargs(artifact["config"]))
    model.load_state_dict(artifact["state_dict"], strict=True)
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    normalizers = _normalizers_from_json(artifact["normalizers"])
    mask = np.asarray(artifact["head_mask"], dtype=bool)
    collected: dict[str, list[np.ndarray]] = {
        "scores": [], "residual": [], "probabilities": [], "any_probability": [],
        "embeddings": [], "token_attention": [], "q": [],
    }
    ordered_ids: list[str] = []
    candidate_ids: list[list[str]] = []
    candidate_checksums: list[list[str]] = []
    q_ranks: list[list[int]] = []
    for batch in catalog.iter_batches(
        allowed_ids=sample_ids,
        labels=None,
        priors=priors,
        normalizers=normalizers,
        batch_size=batch_size,
        seed=0,
        shuffle=False,
        array_keys=required_array_keys(artifact["config"]),
    ):
        inputs, _ = torch_feature_batch(batch, device=torch_device, head_mask=mask, config=artifact["config"])
        result = model(**inputs)
        for name, source in (
            ("scores", result["scores"]),
            ("residual", result["residual"]),
            ("probabilities", result["absolute_probability"]),
            ("any_probability", result["any_probability"]),
            ("embeddings", result["embedding"]),
            ("token_attention", result["token_attention"]),
            ("q", inputs["q"]),
        ):
            collected[name].append(source.detach().float().cpu().numpy())
        ordered_ids.extend(map(str, batch["sample_ids"]))
        for sample_id, record in zip(batch["sample_ids"], batch["records"], strict=True):
            frozen = catalog.candidate_records.get(str(sample_id))
            if frozen is None:
                raise ValueError("FCER prediction requires explicit frozen candidate q-rank records")
            candidates = frozen["candidates"]
            ids = [str(value["candidate_id"]) for value in candidates]
            checksums = [str(value["candidate_checksum"]) for value in candidates]
            if ids != list(map(str, record["candidate_ids"])) or checksums != list(map(str, record["candidate_checksums"])):
                raise ValueError(f"FCER prediction candidate identity mismatch: {sample_id}")
            candidate_ids.append(ids)
            candidate_checksums.append(checksums)
            ranks = [int(value["q_rank"]) for value in candidates]
            if sorted(ranks) != list(range(5)):
                raise ValueError(f"FCER prediction q-rank identity mismatch: {sample_id}")
            q_ranks.append(ranks)
    if set(ordered_ids) != sample_ids or len(ordered_ids) != len(sample_ids):
        raise AssertionError("prediction coverage or uniqueness failure")
    output = {name: np.concatenate(values) for name, values in collected.items()}
    output["sample_ids"] = np.asarray(ordered_ids)
    output["candidate_ids"] = np.asarray(candidate_ids)
    output["candidate_checksums"] = np.asarray(candidate_checksums)
    output["q_ranks"] = np.asarray(q_ranks, dtype=np.int64)
    output["checkpoint_sha256"] = artifact_identity(checkpoint_path)["sha256"]
    output["producing_seed"] = int(artifact["seed"])
    return output


def save_predictions(path: str | Path, predictions: Mapping[str, Any], **metadata: Any) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    arrays = {key: value for key, value in predictions.items() if isinstance(value, np.ndarray)}
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": "3.0.0",
        "kind": "fcer_predictions",
        "status": "complete",
        "prediction": artifact_identity(output),
        "row_count": len(arrays["sample_ids"]),
        "unique_sample_count": len(set(map(str, arrays["sample_ids"]))),
        "unique_candidate_count": len(arrays["sample_ids"]) * 5,
        "checkpoint_sha256": predictions["checkpoint_sha256"],
        "producing_seed": predictions["producing_seed"],
        "candidate_checksums_persisted": "candidate_checksums" in arrays,
        "q_ranks_persisted": "q_ranks" in arrays,
        **metadata,
    }
    atomic_write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def fold_assignments(split_manifest: str | Path, train_ids: set[str]) -> dict[str, int]:
    payload = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    selected=[row for row in payload["rows"] if str(row["sample_id"]) in train_ids]
    result = {str(row["sample_id"]): int(row["oof_fold"]) for row in selected}
    if set(result) != train_ids or set(result.values()) != {0, 1, 2}:
        raise ValueError("OOF fold coverage differs from the training cohort")
    group_folds:dict[str,set[int]]={}
    for row in selected: group_folds.setdefault(str(row["sequence_id"]),set()).add(int(row["oof_fold"]))
    leaked=[group for group,folds in group_folds.items() if len(folds)!=1]
    if leaked: raise ValueError(f"OOF group spans folds: {leaked[:5]}")
    return result
