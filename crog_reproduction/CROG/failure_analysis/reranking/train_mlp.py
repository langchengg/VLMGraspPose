import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import label_map, validate_label_candidate_join
from .schema import INFERENCE_FEATURE_ALLOWLIST, inference_vector, read_jsonl


class MLPRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, values):
        return self.network(values).squeeze(-1)


def _reject_test_records(records, description):
    offending = [record["sample_id"] for record in records if record.get("split") == "test"]
    if offending:
        raise ValueError(f"official test split is locked and cannot be used for {description}")


def _join(records, labels_by_id):
    groups = []
    for record in records:
        label_record = labels_by_id.get(str(record["sample_id"]))
        if label_record is None:
            raise ValueError(f"missing labels for sample {record['sample_id']}")
        validate_label_candidate_join(record, label_record)
        validity = label_map(label_record)
        groups.append(
            {
                "sample_id": record["sample_id"],
                "scene_id": record.get("scene_id", str(record["sample_id"])),
                "candidates": record.get("candidates", []),
                "labels": [float(validity[candidate["candidate_id"]]) for candidate in record.get("candidates", [])],
            }
        )
    return groups


def group_train_validation_split(groups, seed=17, validation_fraction=0.2):
    scene_ids = sorted({str(group["scene_id"]) for group in groups})
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    validation_count = max(1, int(round(len(scene_ids) * validation_fraction))) if len(scene_ids) > 1 else 0
    validation_scenes = set(scene_ids[:validation_count])
    train = [group for group in groups if str(group["scene_id"]) not in validation_scenes]
    validation = [group for group in groups if str(group["scene_id"]) in validation_scenes]
    if not train or not validation:
        raise ValueError("at least two frame groups are required for grouped train/validation split")
    return train, validation


def _raw_vectors(groups, fields):
    rows = []
    for group in groups:
        for candidate in group["candidates"]:
            rows.append(inference_vector(candidate, fields))
    if not rows:
        raise ValueError("no candidates available for MLP training")
    return np.asarray(rows, dtype=object)


def fit_preprocessor(groups, fields=INFERENCE_FEATURE_ALLOWLIST):
    raw = _raw_vectors(groups, fields)
    numeric = np.empty(raw.shape, dtype=np.float64)
    imputer = np.zeros(raw.shape[1], dtype=np.float64)
    for column in range(raw.shape[1]):
        values = np.asarray(
            [np.nan if value is None else float(value) for value in raw[:, column]],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        imputer[column] = float(np.median(finite)) if finite.size else 0.0
        values[~np.isfinite(values)] = imputer[column]
        numeric[:, column] = values
    mean = numeric.mean(axis=0)
    scale = numeric.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return imputer, mean, scale


def transform_candidate(candidate, fields, imputer, mean, scale):
    raw = inference_vector(candidate, fields)
    values = np.asarray(
        [np.nan if value is None else float(value) for value in raw], dtype=np.float64
    )
    values[~np.isfinite(values)] = imputer[~np.isfinite(values)]
    return ((values - mean) / scale).astype(np.float32)


def _group_tensors(group, fields, imputer, mean, scale, device):
    values = np.stack(
        [transform_candidate(candidate, fields, imputer, mean, scale) for candidate in group["candidates"]]
    )
    return (
        torch.from_numpy(values).to(device),
        torch.tensor(group["labels"], dtype=torch.float32, device=device),
    )


def _loss(scores, targets, positive_weight):
    positives = targets > 0.5
    mixed = bool(positives.any() and (~positives).any())
    if mixed:
        listwise = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positives], dim=0)
    else:
        listwise = scores.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(
        scores,
        targets,
        pos_weight=torch.tensor(float(positive_weight), device=scores.device),
    )
    return listwise + 0.2 * bce, mixed


def _epoch_loss(model, groups, fields, imputer, mean, scale, device, positive_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    mixed_count = no_positive = all_positive = 0
    ordered = list(groups)
    for group in ordered:
        if not group["candidates"]:
            continue
        values, targets = _group_tensors(group, fields, imputer, mean, scale, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            scores = model(values)
            loss, mixed = _loss(scores, targets, positive_weight)
            if training:
                loss.backward()
                optimizer.step()
        total += float(loss.detach().cpu())
        mixed_count += int(mixed)
        no_positive += int(not bool((targets > 0.5).any()))
        all_positive += int(bool((targets > 0.5).all()))
    return total / max(1, len(ordered)), {
        "mixed_groups": mixed_count,
        "no_positive_groups": no_positive,
        "all_positive_groups": all_positive,
    }


def _calibration(groups):
    width_rhos = []
    depth_mads = []
    contact_differences = []
    for group in groups:
        for candidate, label in zip(group["candidates"], group["labels"]):
            if not label:
                continue
            diagnostics = candidate.get("diagnostics", {})
            observed = diagnostics.get("object_width_px")
            predicted = candidate.get("width_px")
            if observed is not None and predicted is not None and observed > 0 and predicted > 0:
                width_rhos.append(math.log(float(predicted) / float(observed)))
            depth_mad = candidate.get("features", {}).get("depth_mad_m", {}).get("value")
            contact = candidate.get("features", {}).get("contact_depth_difference_m", {}).get("value")
            if depth_mad is not None and np.isfinite(depth_mad):
                depth_mads.append(float(depth_mad))
            if contact is not None and np.isfinite(contact):
                contact_differences.append(float(contact))
    result = {
        "width_mu_rho": None,
        "width_sigma_rho": None,
        "tau_variance_m": None,
        "tau_balance_m": None,
    }
    if width_rhos:
        median = float(np.median(width_rhos))
        mad = float(np.median(np.abs(np.asarray(width_rhos) - median)))
        result["width_mu_rho"] = median
        result["width_sigma_rho"] = max(1.4826 * mad, 0.10)
    if depth_mads:
        result["tau_variance_m"] = max(float(np.percentile(depth_mads, 90)), 0.005)
    if contact_differences:
        result["tau_balance_m"] = max(float(np.percentile(contact_differences, 90)), 0.005)
    return result


def train_mlp(
    feature_records,
    label_records,
    *,
    validation_feature_records=None,
    validation_label_records=None,
    fields=INFERENCE_FEATURE_ALLOWLIST,
    seed=17,
    validation_fraction=0.2,
    epochs=50,
    patience=8,
    learning_rate=1e-3,
    device="cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    fields = tuple(fields)
    for name in fields:
        if name not in INFERENCE_FEATURE_ALLOWLIST:
            raise ValueError(f"MLP feature is not allowlisted: {name}")
    feature_records = list(feature_records)
    label_records = list(label_records)
    _reject_test_records(feature_records, "MLP training")
    labels_by_id = {str(record["sample_id"]): record for record in label_records}
    all_groups = _join(feature_records, labels_by_id)
    if validation_feature_records is None:
        train_groups, validation_groups = group_train_validation_split(
            all_groups, seed=seed, validation_fraction=validation_fraction
        )
    else:
        validation_feature_records = list(validation_feature_records)
        _reject_test_records(validation_feature_records, "MLP validation")
        validation_labels = {
            str(record["sample_id"]): record for record in validation_label_records
        }
        train_groups = all_groups
        validation_groups = _join(validation_feature_records, validation_labels)

    train_scenes = {str(group["scene_id"]) for group in train_groups}
    validation_scenes = {str(group["scene_id"]) for group in validation_groups}
    if train_scenes & validation_scenes:
        raise AssertionError("same RGB-D frame crossed train/validation split")
    imputer, mean, scale = fit_preprocessor(train_groups, fields)
    positives = sum(sum(group["labels"]) for group in train_groups)
    candidate_count = sum(len(group["labels"]) for group in train_groups)
    negatives = candidate_count - positives
    positive_weight = float(negatives / positives) if positives else 1.0
    torch_device = torch.device(device)
    model = MLPRanker(len(fields) * 3).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_state = None
    best_validation = math.inf
    stale = 0
    history = []
    group_stats = None
    for epoch in range(int(epochs)):
        train_loss, group_stats = _epoch_loss(
            model, train_groups, fields, imputer, mean, scale, torch_device, positive_weight, optimizer
        )
        validation_loss, validation_stats = _epoch_loss(
            model, validation_groups, fields, imputer, mean, scale, torch_device, positive_weight
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_group_stats": group_stats,
                "validation_group_stats": validation_stats,
            }
        )
        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    manifest = {
        "seed": int(seed),
        "group_key": "scene_id (RGB-D frame)",
        "train_sample_ids": [group["sample_id"] for group in train_groups],
        "validation_sample_ids": [group["sample_id"] for group in validation_groups],
        "train_scene_ids": sorted(train_scenes),
        "validation_scene_ids": sorted(validation_scenes),
    }
    artifact = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": len(fields) * 3,
        "fields": fields,
        "imputer": imputer,
        "mean": mean,
        "scale": scale,
        "manifest": manifest,
        "calibration": _calibration(train_groups),
        "history": history,
        "positive_weight": positive_weight,
        "stacking_risk": (
            "CROG predictions from a checkpoint trained on the same train split are in-sample; "
            "out-of-fold CROG predictions were not generated."
        ),
    }
    return artifact


def save_artifact(artifact, output_path, *, overwrite=False):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError("MLP artifact exists; pass --overwrite explicitly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_payload = {
        "training": artifact["manifest"],
        "run": artifact.get("run_manifest"),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path


def load_mlp_scorer(path, device="cpu"):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    fields = tuple(artifact["fields"])
    model = MLPRanker(int(artifact["input_dim"]))
    model.load_state_dict(artifact["model_state_dict"])
    model.to(torch.device(device)).eval()
    imputer = np.asarray(artifact["imputer"], dtype=np.float64)
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)

    def scorer(candidate):
        vector = transform_candidate(candidate, fields, imputer, mean, scale)
        with torch.no_grad():
            value = model(torch.from_numpy(vector).to(device).unsqueeze(0))
        return float(value.squeeze().cpu())

    scorer.artifact = artifact
    return scorer


def train_mlp_paths(
    features_path,
    labels_path,
    *,
    validation_features_path=None,
    validation_labels_path=None,
    **kwargs,
):
    features = list(read_jsonl(features_path))
    labels = list(read_jsonl(labels_path))
    validation_features = (
        list(read_jsonl(validation_features_path)) if validation_features_path else None
    )
    validation_labels = (
        list(read_jsonl(validation_labels_path)) if validation_labels_path else None
    )
    return train_mlp(
        features,
        labels,
        validation_feature_records=validation_features,
        validation_label_records=validation_labels,
        **kwargs,
    )
