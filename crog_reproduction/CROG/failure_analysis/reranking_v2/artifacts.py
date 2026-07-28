from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from . import SCHEMA_VERSION
from .schema import (
    artifact_identity,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    code_fingerprint,
    sha256_bytes,
    sha256_file,
)


def runtime_metadata(device: str) -> dict[str, Any]:
    try:
        import cv2
        import scipy
        import skimage

        versions = {
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
        }
    except ImportError:
        versions = {}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        **versions,
    }


def atomic_savez_compressed(path: str | Path, **arrays: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def artifact_fingerprint(
    *,
    kind: str,
    config: dict[str, Any],
    inputs: Iterable[dict[str, Any]],
    split_manifest_sha256: str | None,
    checkpoint_sha256: str | None,
    evaluator_sha256: str | None,
    code_sha256: str,
    seed: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "config": config,
        "inputs": list(inputs),
        "split_manifest_sha256": split_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "code_sha256": code_sha256,
        "seed": int(seed),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


class ArtifactRun:
    """Atomic, resumable artifact directory with an immutable run identity."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        kind: str,
        repo_root: str | Path,
        config: dict[str, Any],
        inputs: Iterable[str | Path] = (),
        split_manifest: str | Path | None = None,
        checkpoint: str | Path | None = None,
        evaluator_source: str | Path | None = None,
        seed: int = 17,
        device: str = "cpu",
        resume: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "artifact_manifest.json"
        self.success_path = self.output_dir / "SUCCESS"
        self.resume = bool(resume)
        input_identities = [artifact_identity(path) for path in inputs]
        code_sha = code_fingerprint(repo_root)
        split_sha = sha256_file(split_manifest) if split_manifest else None
        checkpoint_sha = sha256_file(checkpoint) if checkpoint else None
        evaluator_sha = sha256_file(evaluator_source) if evaluator_source else None
        fingerprint = artifact_fingerprint(
            kind=kind,
            config=config,
            inputs=input_identities,
            split_manifest_sha256=split_sha,
            checkpoint_sha256=checkpoint_sha,
            evaluator_sha256=evaluator_sha,
            code_sha256=code_sha,
            seed=seed,
        )
        self.expected = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "status": "running",
            "fingerprint": fingerprint,
            "inputs": input_identities,
            "split_manifest_sha256": split_sha,
            "checkpoint_sha256": checkpoint_sha,
            "evaluator_sha256": evaluator_sha,
            "code_sha256": code_sha,
            "config": config,
            "seed": int(seed),
            "runtime": runtime_metadata(device),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "row_count": 0,
            "unique_id_count": 0,
            "outputs": [],
        }

    def prepare(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if current.get("fingerprint") != self.expected["fingerprint"]:
                raise ValueError("artifact fingerprint mismatch; refusing reuse")
            if current.get("status") == "complete":
                for identity in current.get("outputs", []):
                    output = Path(identity["path"])
                    if (
                        not output.exists()
                        or sha256_file(output) != identity["sha256"]
                    ):
                        raise ValueError(
                            f"completed artifact output changed: {output}"
                        )
                expected_marker = str(current["result_sha256"])
                observed_marker = (
                    self.success_path.read_text(encoding="utf-8").strip()
                    if self.success_path.exists()
                    else None
                )
                if observed_marker != expected_marker:
                    atomic_write_text(
                        self.success_path, expected_marker + "\n"
                    )
                self.expected = current
                return current
            if not self.resume:
                raise FileExistsError("incomplete artifact exists; pass --resume")
            self.expected = current
            return current
        if any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"unmanaged files exist in new artifact directory: {self.output_dir}"
            )
        atomic_write_json(self.manifest_path, self.expected)
        return self.expected

    @property
    def is_complete(self) -> bool:
        if (
            self.expected.get("status") != "complete"
            or not self.success_path.exists()
        ):
            return False
        return (
            self.success_path.read_text(encoding="utf-8").strip()
            == self.expected.get("result_sha256")
        )

    def complete(
        self,
        *,
        outputs: Iterable[str | Path],
        row_count: int,
        unique_ids: Iterable[str],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identities = [artifact_identity(path) for path in outputs]
        ids = list(map(str, unique_ids))
        if len(ids) != len(set(ids)):
            raise ValueError("artifact IDs are not unique")
        manifest = {
            **self.expected,
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "row_count": int(row_count),
            "unique_id_count": len(ids),
            "outputs": identities,
            **(extra or {}),
        }
        manifest["result_sha256"] = sha256_bytes(
            canonical_json(
                {
                    "fingerprint": manifest["fingerprint"],
                    "row_count": manifest["row_count"],
                    "unique_id_count": manifest["unique_id_count"],
                    "outputs": identities,
                }
            ).encode()
        )
        atomic_write_json(self.manifest_path, manifest)
        atomic_write_text(
            self.success_path,
            manifest["result_sha256"] + "\n",
        )
        self.expected = manifest
        return manifest


@contextmanager
def atomic_output(path: str | Path, mode: str = "wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open(mode) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
