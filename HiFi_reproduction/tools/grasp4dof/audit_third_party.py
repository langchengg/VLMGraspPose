#!/usr/bin/env python3
"""Audit pinned official 4-DoF grasp backends and the Mac runtime.

The command never downloads code or weights and never writes inside the
third-party checkouts.  Every pickle-bearing checkpoint is byte-verified
before it is deserialised.  Outputs are restricted to ``--run-dir``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party_src"
DEFAULT_GR_REPO = THIRD_PARTY_ROOT / "grconvnet"
DEFAULT_GG_REPO = THIRD_PARTY_ROOT / "ggcnn"
DEFAULT_GG_CHECKPOINT_ROOT = THIRD_PARTY_ROOT / "checkpoints/ggcnn2"

GR_REPOSITORY = "https://github.com/skumra/robotic-grasping.git"
GR_COMMIT = "bdd49367f8619be94123fb3187c2f8ad5100ef46"
GR_RELEASE = "v0.3.0"
GR_LICENSE_SHA256 = "07809e1c6ba70e7027376faf131a5792bfe2f8acedc1b48a373ddfca8a964708"

GG_REPOSITORY = "https://github.com/dougsm/ggcnn.git"
GG_COMMIT = "0c50aa7600e8a30d44c5c85cebd6e3394a81f30e"
GG_RELEASE = "v0.1"
GG_RELEASE_TAG_COMMIT = "ad48bc5f768fe0a9ba9fd47729638e0aed46e47b"
GG_POST_2020_ARCH_COMMIT = "8104fd4426acc507359658e99dc8ec1f66d1dec6"
GG_LICENSE_SHA256 = "506b8c229124c1edd9ee7765b09b442410d7161d14226bf0a96c6b4a75a33549"

GR_SOURCE_FILES = {
    "inference/models/grconvnet3.py": "8fbdf124cd7710a1a3f50121dcec583a919d109900bb23010995ad4a8ed87b81",
    "inference/post_process.py": "a03721d4b9ad610cc2debe76e2104519ca5ad5fe217d7ea0b9a243e3d9c185a7",
    "hardware/device.py": "97a161bd4246ddcd9426d91054fcbcdb47344fe883820877857e27c065df589f",
    "train_network.py": "df66dd4c029131cae47b3675beeca8cad386e86de4608c416607120d14428ea7",
}
GG_SOURCE_FILES = {
    "models/ggcnn2.py": "52ae0ba25a9da3e87b7fae894558442f41dc610da0c3d485126b345e350c1224",
    "models/common.py": "1e2fbc2ee9640d0957c7af79071ba5f62be76ae07d311d6895963b19b348b385",
    "train_ggcnn.py": "4283dc24bb234248a7d769fe58f1bf8125bfddf82972bac6ea0d1b47c5999ca1",
    "eval_ggcnn.py": "9c13a980858ae4b7a2970fd4cc2159ff8181fad0e8eb1753c6697247544a172b",
}

GR_CHECKPOINTS = {
    "jacquard_rgbd": {
        "relative_path": "trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93",
        "sha256": "adfb2cbbb8df2708a732e12ddc4db114f3ec399ffb5d403ca75c5b5b9e769171",
        "bytes": 7_661_004,
        "input_channels": 4,
        "dataset": "Jacquard",
        "modalities": ["depth", "rgb"],
    },
    "jacquard_depth": {
        "relative_path": "trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94",
        "sha256": "67d8d2a948bd24f68547bbfd851f6ae2b229f07cf7ed1b75ed8c6730c6626bc6",
        "bytes": 7_629_902,
        "input_channels": 1,
        "dataset": "Jacquard",
        "modalities": ["depth"],
    },
    "cornell_rgbd": {
        "relative_path": "trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98",
        "sha256": "e6c03afc0f8266d29ec92141088cc6a557215fabc2681d4ca5a162f451853d28",
        "bytes": 7_640_481,
        "input_channels": 4,
        "dataset": "Cornell",
        "modalities": ["depth", "rgb"],
    },
}
GG_CHECKPOINTS = {
    "release_archive": {
        "relative_path": "ggcnn2_weights_cornell.zip",
        "sha256": "f71e3575fe70bea6817239f9fad98264af5cfd95dd9bbe69ff4144696f34f972",
        "bytes": 509_324,
    },
    "state_dict": {
        "relative_path": "ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt",
        "sha256": "865d538a51d427f7ee84defc99e093bdf51eeb0627c302068037e52188c11d1c",
        "bytes": 270_922,
    },
    "full_pickle": {
        "relative_path": "ggcnn2_weights_cornell/epoch_50_cornell",
        "sha256": "105f82f405c8758e664fb704f225d0e68f67c6efe034a9a3e95d6030d1002515",
        "bytes": 295_090,
    },
}

UV_SYNC_ANOMALY = {
    "status": "corrected_before_model_smoke",
    "persistent_runtime_fault": False,
    "initial_command": "uv pip sync requirements-grasp4dof-macos.txt",
    "initial_observation": (
        "initial sync installed only the 14 direct pins; the first import torch "
        "failed with ModuleNotFoundError: No module named 'typing_extensions'"
    ),
    "correction": (
        "uv pip install -r requirements-grasp4dof-macos.txt resolved and installed "
        "the transitive dependency closure before any model smoke test"
    ),
    "corrected_transitive_examples": [
        "typing-extensions",
        "filelock",
        "fsspec",
        "jinja2",
        "markupsafe",
        "networkx",
        "sympy",
        "mpmath",
        "setuptools",
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grconvnet-repo", type=Path, default=DEFAULT_GR_REPO)
    parser.add_argument("--ggcnn-repo", type=Path, default=DEFAULT_GG_REPO)
    parser.add_argument(
        "--ggcnn-checkpoint-root", type=Path, default=DEFAULT_GG_CHECKPOINT_ROOT
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed_size = path.stat().st_size
    if observed_size != int(expected["bytes"]):
        raise ValueError(
            f"{label} size mismatch: expected {expected['bytes']}, got {observed_size}"
        )
    observed_sha = sha256_file(path)
    if observed_sha != expected["sha256"]:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected['sha256']}, got {observed_sha}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": observed_size,
        "sha256": observed_sha,
        "verified_before_deserialisation": True,
    }


def command_text(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _normalise_remote(value: str) -> str:
    return value.strip().removesuffix("/").removesuffix(".git")


def audit_git_checkout(repo: Path, *, expected_remote: str, expected_commit: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    if not (repo / ".git").is_dir():
        raise FileNotFoundError(f"not a Git checkout: {repo}")
    remote = command_text("git", "remote", "get-url", "origin", cwd=repo)
    if _normalise_remote(remote) != _normalise_remote(expected_remote):
        raise ValueError(f"unexpected origin for {repo}: {remote}")
    commit = command_text("git", "rev-parse", "HEAD", cwd=repo)
    if commit != expected_commit:
        raise ValueError(
            f"unexpected commit for {repo}: expected {expected_commit}, got {commit}"
        )
    porcelain = command_text(
        "git", "status", "--porcelain=v1", "--untracked-files=all", cwd=repo
    )
    if porcelain:
        raise ValueError(f"third-party checkout is dirty: {repo}\n{porcelain}")
    try:
        exact_tag = command_text("git", "describe", "--tags", "--exact-match", cwd=repo)
    except subprocess.CalledProcessError:
        exact_tag = None
    created = datetime.fromtimestamp(repo.stat().st_birthtime, timezone.utc).isoformat()
    return {
        "path": str(repo),
        "origin": remote,
        "commit": commit,
        "commit_date": command_text("git", "show", "-s", "--format=%cI", "HEAD", cwd=repo),
        "exact_tag": exact_tag,
        "clean": True,
        "modified_files": [],
        "local_checkout_created_utc": created,
    }


def audit_license(repo: Path, *, expected_sha256: str) -> dict[str, Any]:
    license_path = repo / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"missing license: {license_path}")
    text = license_path.read_text(encoding="utf-8")
    observed = sha256_file(license_path)
    if observed != expected_sha256:
        raise ValueError(f"license hash mismatch: {license_path}: {observed}")
    required = (
        "Redistribution and use in source and binary forms",
        "Neither the name of the copyright holder nor the names of its contributors",
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS",
    )
    if not all(fragment in text for fragment in required):
        raise ValueError(f"license is not the expected BSD 3-Clause text: {license_path}")
    return {
        "spdx": "BSD-3-Clause",
        "path": str(license_path.resolve()),
        "bytes": license_path.stat().st_size,
        "sha256": observed,
        "redistribution_allowed": True,
        "conditions": [
            "retain copyright notice, conditions, and disclaimer",
            "do not use contributor names for endorsement without permission",
        ],
    }


def audit_source_files(repo: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative, expected_sha in expected.items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source file: {path}")
        observed = sha256_file(path)
        if observed != expected_sha:
            raise ValueError(f"pinned source hash mismatch: {path}: {observed}")
        result[relative] = {"path": str(path.resolve()), "sha256": observed}
    return result


@contextlib.contextmanager
def pinned_module_root(root: Path, prefix: str) -> Iterator[None]:
    """Temporarily make one verified checkout authoritative for pickle imports."""

    root = root.resolve()
    prior = {
        name: module
        for name, module in list(sys.modules.items())
        if name == prefix or name.startswith(prefix + ".")
    }
    for name in prior:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
        else:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(root))
        sys.modules.update(prior)
        importlib.invalidate_caches()


def _assert_module_in_repo(module: Any, repo: Path) -> str:
    source = Path(inspect.getfile(module)).resolve()
    try:
        source.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"module escaped pinned checkout: {source}") from exc
    return str(source)


def validate_state_dict_shapes(
    model: torch.nn.Module, state: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} is not a non-empty state_dict")
    if any(not isinstance(key, str) for key in state):
        raise ValueError(f"{label} has non-string keys")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{label} has non-tensor values")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError(f"{label} contains non-finite tensors")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = {
        key: {"expected": list(expected[key].shape), "observed": list(state[key].shape)}
        for key in sorted(set(expected) & set(state))
        if tuple(expected[key].shape) != tuple(state[key].shape)
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{label} key/shape mismatch: missing={missing}, "
            f"unexpected={unexpected}, shapes={mismatched}"
        )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"{label} strict load failed: {incompatible}")
    return {
        "strict_load_success": True,
        "state_keys": len(state),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": {},
    }


def audit_grconvnet(repo: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    repo = repo.expanduser().resolve()
    git = audit_git_checkout(repo, expected_remote=GR_REPOSITORY, expected_commit=GR_COMMIT)
    license_info = audit_license(repo, expected_sha256=GR_LICENSE_SHA256)
    source_files = audit_source_files(repo, GR_SOURCE_FILES)
    checkpoint_rows: list[dict[str, Any]] = []
    primary_model: torch.nn.Module | None = None

    with pinned_module_root(repo, "inference"):
        module = importlib.import_module("inference.models.grconvnet3")
        module_path = _assert_module_in_repo(module, repo)
        model_type = module.GenerativeResnet
        for checkpoint_id, specification in GR_CHECKPOINTS.items():
            checkpoint_path = repo / specification["relative_path"]
            verified = verify_file(
                checkpoint_path, specification, label=f"GR-ConvNet {checkpoint_id}"
            )
            # The pinned official artifacts are complete torch.save(model) pickles.
            # Explicit weights_only=False is permitted only after exact byte and
            # checkout verification above.
            loaded = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            if loaded.__class__.__module__ != "inference.models.grconvnet3" or not isinstance(
                loaded, model_type
            ):
                raise ValueError(
                    f"{checkpoint_id} deserialised as unexpected class: "
                    f"{loaded.__class__.__module__}.{loaded.__class__.__name__}"
                )
            input_channels = int(loaded.conv1.in_channels)
            if input_channels != int(specification["input_channels"]):
                raise ValueError(
                    f"{checkpoint_id} input-channel mismatch: {input_channels}"
                )
            fresh = model_type(
                input_channels=input_channels,
                channel_size=int(loaded.conv1.out_channels),
                dropout=bool(loaded.dropout),
                prob=float(loaded.dropout_pos.p),
            )
            strict = validate_state_dict_shapes(
                fresh, loaded.state_dict(), label=f"GR-ConvNet {checkpoint_id} state"
            )
            row = {
                "checkpoint_id": checkpoint_id,
                **verified,
                "dataset": specification["dataset"],
                "modalities": specification["modalities"],
                "input_channels": input_channels,
                "serialization": "trusted_official_full_model_pickle",
                "map_location": "cpu",
                "weights_only": False,
                "trusted_basis": (
                    "exact repository remote+commit+clean status and exact checkpoint "
                    "size+SHA-256 were verified before unpickling"
                ),
                "class": f"{loaded.__class__.__module__}.{loaded.__class__.__name__}",
                "parameter_count": sum(parameter.numel() for parameter in loaded.parameters()),
                **strict,
            }
            checkpoint_rows.append(row)
            if checkpoint_id == "jacquard_rgbd":
                primary_model = fresh.eval()
    if primary_model is None:
        raise RuntimeError("primary GR-ConvNet checkpoint was not audited")

    manifest = {
        "schema_version": 1,
        "audited_utc": utc_now(),
        "status": "PASS",
        "repository": GR_REPOSITORY,
        "release": GR_RELEASE,
        "pinned_commit": GR_COMMIT,
        "git": git,
        "license": license_info,
        "source_files": source_files,
        "checkpoints": checkpoint_rows,
        "primary_pretrained_transfer_checkpoint_id": "jacquard_rgbd",
        "architecture": {
            "class": "inference.models.grconvnet3.GenerativeResnet",
            "module_path": module_path,
            "family": "GR-ConvNet configurable variant 3",
            "encoder": "three convolutions",
            "residual_blocks": 5,
            "decoder": "three transposed convolutions",
            "output_heads": ["quality", "cos_2theta", "sin_2theta", "width"],
        },
        "input_contract": {
            "rgbd_channel_order": ["depth", "red", "green", "blue"],
            "rgb": "float32 / 255 then subtract per-image mean",
            "depth": "float32 subtract per-image mean then clip to [-1, 1]",
        },
        "training_contract": {
            "loss": "sum of four Smooth-L1 losses",
            "width_target": "clip to output_size/2 then divide by output_size/2",
        },
        "post_processing_contract": {
            "angle": "0.5 * atan2(sin_2theta, cos_2theta)",
            "width": "network width multiplied by fixed 150 pixels",
            "gaussian_sigma": {"quality": 2.0, "angle": 2.0, "width": 1.0},
            "official_peak_decoder": {
                "min_distance": 20,
                "threshold_abs": 0.2,
            },
        },
        "provenance_disclosures": [
            {
                "id": "gr_input_size_and_width_scale_ambiguity",
                "severity": "must_disclose",
                "fact": (
                    "paired arch.txt files report 224x224 output, the README Jacquard "
                    "example uses input-size 300, width targets scale by output_size/2, "
                    "and official post-processing always multiplies width by 150"
                ),
                "decision": (
                    "do not claim an unambiguous official training scale; lock resize "
                    "and width decoding on validation and record the selected rule"
                ),
            }
        ],
        "mac_adapter_requirements": [
            "replace CUDA/CPU-only device helper with MPS then CPU selection",
            "load full pickle on CPU after hash verification and export/use a state_dict",
            "guard torch.cuda.empty_cache",
            "replace removed NumPy np.int/np.float aliases in adapter paths",
            "exclude pyrealsense2 from the offline benchmark dependency set",
        ],
        "reuse_decision": "adapt pinned official source through a project-local adapter",
    }
    return manifest, primary_model


def audit_ggcnn2(
    repo: Path, checkpoint_root: Path
) -> tuple[dict[str, Any], torch.nn.Module]:
    repo = repo.expanduser().resolve()
    checkpoint_root = checkpoint_root.expanduser().resolve()
    git = audit_git_checkout(repo, expected_remote=GG_REPOSITORY, expected_commit=GG_COMMIT)
    license_info = audit_license(repo, expected_sha256=GG_LICENSE_SHA256)
    source_files = audit_source_files(repo, GG_SOURCE_FILES)
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    for checkpoint_id, specification in GG_CHECKPOINTS.items():
        checkpoint_rows[checkpoint_id] = {
            "checkpoint_id": checkpoint_id,
            **verify_file(
                checkpoint_root / specification["relative_path"],
                specification,
                label=f"GG-CNN2 {checkpoint_id}",
            ),
        }

    state_path = checkpoint_root / GG_CHECKPOINTS["state_dict"]["relative_path"]
    with pinned_module_root(repo, "models"):
        module = importlib.import_module("models.ggcnn2")
        module_path = _assert_module_in_repo(module, repo)
        model = module.GGCNN2(input_channels=1)
        bilinear_layers = sum(
            isinstance(layer, torch.nn.UpsamplingBilinear2d) for layer in model.modules()
        )
        transposed_convolutions = sum(
            isinstance(layer, torch.nn.ConvTranspose2d) for layer in model.modules()
        )
        if bilinear_layers != 2 or transposed_convolutions != 0:
            raise ValueError(
                "GG-CNN2 source is not the post-2020-07 bilinear-upsample architecture"
            )
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        strict = validate_state_dict_shapes(model, state, label="GG-CNN2 state_dict")
    checkpoint_rows["state_dict"].update(
        {
            "serialization": "state_dict",
            "map_location": "cpu",
            "weights_only": True,
            "architecture_commit": GG_COMMIT,
            **strict,
        }
    )
    checkpoint_rows["full_pickle"].update(
        {
            "serialization": "official_full_model_pickle_not_used",
            "reason_not_loaded": "the official state_dict is safer and loads strictly",
        }
    )
    checkpoint_rows["release_archive"].update(
        {
            "official_release_asset_url": (
                "https://github.com/dougsm/ggcnn/releases/download/v0.1/"
                "ggcnn2_weights_cornell.zip"
            ),
            "github_asset_id": 22_741_062,
        }
    )
    manifest = {
        "schema_version": 1,
        "audited_utc": utc_now(),
        "status": "PASS",
        "repository": GG_REPOSITORY,
        "release": GG_RELEASE,
        "pinned_commit": GG_COMMIT,
        "git": git,
        "license": license_info,
        "source_files": source_files,
        "checkpoints": list(checkpoint_rows.values()),
        "primary_pretrained_transfer_checkpoint_id": "state_dict",
        "architecture": {
            "class": "models.ggcnn2.GGCNN2",
            "module_path": module_path,
            "variant": "post-2020-07 bilinear-upsample GG-CNN2",
            "input_channels": 1,
            "input_size": [300, 300],
            "bilinear_upsampling_layers": bilinear_layers,
            "transposed_convolution_layers": transposed_convolutions,
            "dilations": [2, 4],
            "output_heads": ["quality", "cos_2theta", "sin_2theta", "width"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "input_contract": {
            "modalities": ["depth"],
            "depth": "float32 subtract per-image mean then clip to [-1, 1]",
        },
        "training_contract": {
            "loss": "sum of four mean-squared-error losses",
            "width_target": "clip to 150 pixels then divide by 150",
        },
        "post_processing_contract": {
            "angle": "0.5 * atan2(sin_2theta, cos_2theta)",
            "width": "network width multiplied by 150 pixels",
            "gaussian_sigma": {"quality": 2.0, "angle": 2.0, "width": 1.0},
            "official_peak_decoder": {"min_distance": 20, "threshold_abs": 0.2},
        },
        "provenance_disclosures": [
            {
                "id": "gg_release_tag_source_mismatch",
                "severity": "must_disclose",
                "fact": (
                    f"release tag {GG_RELEASE} points to {GG_RELEASE_TAG_COMMIT}, while "
                    f"GG-CNN2 changed to bilinear upsampling in {GG_POST_2020_ARCH_COMMIT} "
                    "before the official checkpoint asset was uploaded"
                ),
                "decision": (
                    f"load the official state_dict strictly against pinned post-change "
                    f"source commit {GG_COMMIT}, not the older release-tag source tree"
                ),
            }
        ],
        "mac_adapter_requirements": [
            "replace hard-coded cuda:0 with MPS then CPU selection",
            "load the state_dict with map_location=cpu and weights_only=True",
            "run scikit-image post-processing on CPU after tensor transfer",
        ],
        "reuse_decision": "adapt pinned official source through a project-local adapter",
    }
    return manifest, model.eval()


def _synchronise(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def smoke_forward(
    model: torch.nn.Module,
    *,
    model_name: str,
    device_name: str,
    input_shape: tuple[int, int, int, int],
    expected_output_shape: tuple[int, int, int, int],
) -> dict[str, Any]:
    device = torch.device(device_name)
    candidate = copy.deepcopy(model).eval().to(device)
    input_tensor = torch.zeros(input_shape, dtype=torch.float32, device=device)
    before_allocated = (
        int(torch.mps.current_allocated_memory()) if device.type == "mps" else None
    )
    _synchronise(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = candidate(input_tensor)
    _synchronise(device)
    elapsed = time.perf_counter() - started
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise ValueError(f"{model_name} on {device_name} did not return four maps")
    shapes = [list(output.shape) for output in outputs]
    expected = list(expected_output_shape)
    if any(shape != expected for shape in shapes):
        raise ValueError(
            f"{model_name} on {device_name} output mismatch: {shapes} != {expected}"
        )
    finite = [bool(torch.isfinite(output).all().item()) for output in outputs]
    if not all(finite):
        raise ValueError(f"{model_name} on {device_name} produced non-finite output")
    after_allocated = (
        int(torch.mps.current_allocated_memory()) if device.type == "mps" else None
    )
    del outputs, input_tensor, candidate
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "status": "PASS",
        "model": model_name,
        "device": device_name,
        "dtype": "float32",
        "input_shape": list(input_shape),
        "output_shapes": shapes,
        "outputs_finite": finite,
        "wall_time_seconds": elapsed,
        "mps_allocated_before_bytes": before_allocated,
        "mps_allocated_after_forward_bytes": after_allocated,
    }


def audit_device_smoke(
    gr_model: torch.nn.Module,
    gg_model: torch.nn.Module,
    *,
    include_mps: bool = True,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    specifications = (
        (gr_model, "GR-ConvNet Jacquard RGB-D", (1, 4, 224, 224), (1, 1, 224, 224)),
        (gg_model, "GG-CNN2 Cornell depth", (1, 1, 300, 300), (1, 1, 300, 300)),
    )
    for model, name, input_shape, output_shape in specifications:
        entries.append(
            smoke_forward(
                model,
                model_name=name,
                device_name="cpu",
                input_shape=input_shape,
                expected_output_shape=output_shape,
            )
        )
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    if include_mps and mps_available:
        for model, name, input_shape, output_shape in specifications:
            entries.append(
                smoke_forward(
                    model,
                    model_name=name,
                    device_name="mps",
                    input_shape=input_shape,
                    expected_output_shape=output_shape,
                )
            )
        mps_status = "PASS"
    elif include_mps:
        mps_status = "NOT_AVAILABLE"
    else:
        mps_status = "NOT_REQUESTED_IN_UNIT_TEST"
    return {
        "schema_version": 1,
        "audited_utc": utc_now(),
        "status": "PASS",
        "formal_dtype": "float32",
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_status": mps_status,
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "post_processing_device": "cpu (NumPy/scikit-image after .cpu())",
        "entries": entries,
    }


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_lock_text() -> str:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return "".join(
        f"{name}=={packages[name]}\n" for name in sorted(packages, key=str.casefold)
    )


def _optional_command(*args: str) -> str | None:
    try:
        return command_text(*args)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def audit_environment(run_dir: Path, *, package_lock_sha256: str) -> dict[str, Any]:
    disk = shutil.disk_usage(run_dir)
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
    except (ImportError, AttributeError):
        value = _optional_command("sysctl", "-n", "hw.memsize")
        memory_bytes = int(value) if value else None
    return {
        "schema_version": 1,
        "audited_utc": utc_now(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "macos_version": _optional_command("sw_vers", "-productVersion"),
        "macos_build": _optional_command("sw_vers", "-buildVersion"),
        "machine": platform.machine(),
        "chip": _optional_command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": memory_bytes,
        "disk_path": str(run_dir.resolve()),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pytorch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": {
            name: package_version(distribution)
            for name, distribution in {
                "numpy": "numpy",
                "opencv": "opencv-python-headless",
                "scikit_image": "scikit-image",
                "scipy": "scipy",
                "pyarrow": "pyarrow",
                "pandas": "pandas",
                "pillow": "pillow",
                "matplotlib": "matplotlib",
                "psutil": "psutil",
                "shapely": "shapely",
                "pyyaml": "PyYAML",
                "pytest": "pytest",
                "typing_extensions": "typing-extensions",
            }.items()
        },
        "package_lock_path": str((run_dir / "package_lock.txt").resolve()),
        "package_lock_sha256": package_lock_sha256,
        "environment_setup_anomaly": UV_SYNC_ANOMALY,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def run_audit(
    *,
    run_dir: Path,
    gr_repo: Path = DEFAULT_GR_REPO,
    gg_repo: Path = DEFAULT_GG_REPO,
    gg_checkpoint_root: Path = DEFAULT_GG_CHECKPOINT_ROOT,
    include_mps: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    gr_manifest, gr_model = audit_grconvnet(gr_repo)
    gg_manifest, gg_model = audit_ggcnn2(gg_repo, gg_checkpoint_root)
    smoke = audit_device_smoke(gr_model, gg_model, include_mps=include_mps)
    lock_text = package_lock_text()
    lock_path = run_dir / "package_lock.txt"
    write_text(lock_path, lock_text)
    lock_sha = sha256_file(lock_path)
    environment = audit_environment(run_dir, package_lock_sha256=lock_sha)
    write_json(run_dir / "third_party/grconvnet_source_manifest.json", gr_manifest)
    write_json(run_dir / "third_party/ggcnn2_source_manifest.json", gg_manifest)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "audit/basic_device_smoke.json", smoke)
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "outputs": [
            str(run_dir / "third_party/grconvnet_source_manifest.json"),
            str(run_dir / "third_party/ggcnn2_source_manifest.json"),
            str(run_dir / "environment.json"),
            str(lock_path),
            str(run_dir / "audit/basic_device_smoke.json"),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(
        run_dir=args.run_dir,
        gr_repo=args.grconvnet_repo,
        gg_repo=args.ggcnn_repo,
        gg_checkpoint_root=args.ggcnn_checkpoint_root,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
