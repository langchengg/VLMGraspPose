"""Fail-closed D1 Case-B adapter for locked original-resolution GT masks.

The frozen D1 loader is deliberately prediction-only.  This module preserves
that loader and its mask-processing implementation byte-for-byte: an ephemeral
view changes only the two prediction-only provenance declarations that the
frozen loader checks.  The returned sample still points at the canonical
oracle bundle and exposes the oracle metadata.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import errno
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import tempfile
from types import ModuleType
from typing import Any, Iterable, Mapping

from .execution import (
    FROZEN_D1_CANDIDATE_SCRIPT,
    FROZEN_D1_CANDIDATE_SCRIPT_SHA256,
    FROZEN_D1_CONFIG,
    FROZEN_D1_CONFIG_SHA256,
    FROZEN_D1_SCORER_SCRIPT,
    FROZEN_D1_SCORER_SCRIPT_SHA256,
    FROZEN_D1_SOURCE,
    FROZEN_DOCKER_IMAGE,
    FROZEN_DOCKER_IMAGE_ID,
    REPOSITORY_ROOT,
)
from .candidate_matching import stable_candidate_id
from .io import (
    artifact_record,
    atomic_parquet,
    canonical_sha256,
    exclusive_json,
    exclusive_text,
)
from .io import sha256_file


FROZEN_D1_LOADER = (
    FROZEN_D1_SOURCE / "source_snapshot/src/grasping/ocid_vlg_grasp_adapter.py"
)
FROZEN_D1_LOADER_SHA256 = (
    "afb954cc5959e5948cbd7abc1303b876364d53bd8c478691b135d3a032e36b60"
)
FROZEN_MODEL_DIR = REPOSITORY_ROOT / "HiFi_reproduction/models/gqcnn-official/GQCNN-2.1"
FROZEN_MODEL_CONFIG_SHA256 = (
    "eb5bc17089a39bd8fe6c801010c25a6a79a898d64181180feb5cf69aa630ff6f"
)
# Every byte consumed by the pinned GQ-CNN inference loader.  Locking only
# ``config.json`` would leave the TensorFlow checkpoint and normalisation
# arrays mutable after the protocol was assembled.
FROZEN_MODEL_RUNTIME_SHA256 = {
    "architecture.json": "bf8214e1285be28879291184e86a84f56076c4ca612cdad3a874d2e1414122e2",
    "checkpoint": "5684f3a7bb9f2d1b34f771e425138f3d2d268e3cf2c12637c60163d5c18a297d",
    "config.json": FROZEN_MODEL_CONFIG_SHA256,
    "mean.npy": "3072110cde105c8a14c293935e372e7224f3ff3a4064e3f02b332853bcc37acd",
    "model.ckpt.data-00000-of-00001": "f36db44416664db0dc46174db69166db3398ebd7c08f07a5d7c813a9ebd04313",
    "model.ckpt.index": "38a00428b0a0471056904bf02abcaed825aac2a59c8de168b598d0b1d6aad28e",
    "model.ckpt.meta": "a567ea53cfe21ae3988230e1a6473f3d01d82b0f5315d9b4a21bc235d54a1cf9",
    "pose_mean.npy": "496dedb4cc1477932abb1a40dc0aa9a9611af31385df289dcb59ac0fdf1c7668",
    "pose_std.npy": "d947aa65e2b20775dddc4db2ef72842fd1082fdf9b57b9a341d8502628343223",
    "std.npy": "5edc600f83347a911b7e6e0b320bb3b90338464029bcbff4d83decd81bafdf3e",
}
CASE_B_MANIFEST_NAME = "D1_CASE_B_BUNDLE_MANIFEST.json"
ORACLE_MASK_SOURCE = "locked_gt_mask_original_resolution"
PREDICTED_MASK_SOURCE = "predicted_mask_original_resolution"
REQUIRED_BUNDLE_MEMBERS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
    "checksums.sha256",
)
BYTE_IDENTICAL_MEMBERS = (
    "color.png",
    "depth.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
)
_CHECKSUM_MEMBERS = frozenset(REQUIRED_BUNDLE_MEMBERS) - {"checksums.sha256"}
_SAFE_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class D1AdapterError(RuntimeError):
    """The requested D1 operation differs from the frozen Case-B contract."""


class D1MachineBlocker(D1AdapterError):
    """The host cannot safely materialise or resume the D1 operation."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class CandidateInventory:
    samples: int
    nonempty: int
    empty: int
    candidates: int

    def __post_init__(self) -> None:
        values = (self.samples, self.nonempty, self.empty, self.candidates)
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise D1AdapterError(
                "candidate inventory counts must be non-negative integers"
            )
        if self.nonempty + self.empty != self.samples:
            raise D1AdapterError("candidate inventory does not partition all samples")


def _regular_file(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=False)
    if Path(path).expanduser().is_symlink() or not source.is_file():
        raise D1AdapterError(f"{label} must be a regular non-symlink file: {source}")
    return source


def _executable_file(path: str | Path, *, label: str) -> Path:
    """Resolve a normal executable symlink while rejecting absent/non-executable targets."""

    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file() or not os.access(source, os.X_OK):
        raise D1AdapterError(f"{label} must resolve to an executable file: {source}")
    return source


def _safe_id(value: Any) -> str:
    sample_id = str(value)
    if _SAFE_SAMPLE_ID.fullmatch(sample_id) is None or sample_id in {".", ".."}:
        raise D1AdapterError(f"unsafe sample_id: {sample_id!r}")
    return sample_id


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1AdapterError(f"cannot parse {label}: {source}") from error
    if not isinstance(value, dict):
        raise D1AdapterError(f"{label} must contain one JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    source = _regular_file(path, label=label)
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise D1AdapterError(f"{label} row {line_number} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise D1AdapterError(f"cannot parse {label}: {source}") from error
    return rows


def _checksum_map(bundle: Path) -> dict[str, str]:
    checksum_file = _regular_file(bundle / "checksums.sha256", label="bundle checksums")
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2:
            raise D1AdapterError(f"malformed checksum row {line_number} in {bundle}")
        digest, name = pieces[0], pieces[1].lstrip("* ")
        if Path(name).name != name or name in result:
            raise D1AdapterError(f"unsafe or duplicate checksum member: {name!r}")
        member = _regular_file(bundle / name, label="bundle member")
        actual = sha256_file(member)
        if actual != digest:
            raise D1AdapterError(f"bundle checksum differs: {bundle / name}")
        result[name] = actual
    if set(result) != _CHECKSUM_MEMBERS:
        raise D1AdapterError(
            "frozen bundle checksum inventory differs: "
            f"missing={sorted(_CHECKSUM_MEMBERS - set(result))} "
            f"extra={sorted(set(result) - _CHECKSUM_MEMBERS)}"
        )
    return result


def _verify_frozen_file(path: Path, expected: str, *, label: str) -> Path:
    source = _regular_file(path, label=label)
    if sha256_file(source) != expected:
        raise D1AdapterError(f"{label} hash differs: {source}")
    return source


def verify_frozen_d1_sources() -> dict[str, dict[str, Any]]:
    """Rehash every frozen D1 source used by the adapter and commands."""

    sources = {
        "candidate_script": (
            FROZEN_D1_CANDIDATE_SCRIPT,
            FROZEN_D1_CANDIDATE_SCRIPT_SHA256,
        ),
        "scorer_script": (FROZEN_D1_SCORER_SCRIPT, FROZEN_D1_SCORER_SCRIPT_SHA256),
        "config": (FROZEN_D1_CONFIG, FROZEN_D1_CONFIG_SHA256),
        "loader": (FROZEN_D1_LOADER, FROZEN_D1_LOADER_SHA256),
    }
    result: dict[str, dict[str, Any]] = {}
    for label, (path, expected) in sources.items():
        result[label] = artifact_record(
            _verify_frozen_file(path, expected, label=f"frozen D1 {label}")
        )
    for name, expected in FROZEN_MODEL_RUNTIME_SHA256.items():
        label = f"model_runtime/{name}"
        result[label] = artifact_record(
            _verify_frozen_file(
                FROZEN_MODEL_DIR / name,
                expected,
                label=f"frozen D1 {label}",
            )
        )
    return result


def assert_gt_bulk_authority(
    authority: Mapping[str, Any], *, registry_path: Path
) -> dict[str, Any]:
    """Validate an already-claimed P4 authority before any GT pixel read."""

    if authority.get("d1_candidate_generation_authorized") is not True:
        raise PermissionError(
            "locked authority does not permit D1 GT candidate generation"
        )
    registry = authority.get("gt_mask_registry")
    if not isinstance(registry, Mapping):
        raise D1AdapterError("locked authority lacks the GT registry artifact")
    requested = _regular_file(registry_path, label="GT registry")
    if Path(str(registry.get("path", ""))).expanduser().resolve(
        strict=False
    ) != requested or registry.get("sha256") != sha256_file(requested):
        raise D1AdapterError("GT registry differs from locked authority")
    routes = authority.get("routes")
    d1 = routes.get("d1") if isinstance(routes, Mapping) else None
    if not isinstance(d1, Mapping):
        raise D1AdapterError("locked authority lacks a D1 route contract")
    if (
        d1.get("case") != "B"
        or d1.get("mask_affects_raw_sampling") is not True
        or d1.get("raw_candidate_regeneration_required") is not True
        or d1.get("filter_only_primary_allowed") is not False
    ):
        raise D1AdapterError("D1 authority does not require Case-B raw regeneration")
    return dict(d1)


def _index_rows(
    rows: Iterable[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _safe_id(row.get("sample_id"))
        if sample_id in indexed:
            raise D1AdapterError(f"duplicate {label} sample_id: {sample_id}")
        indexed[sample_id] = dict(row)
    if not indexed:
        raise D1AdapterError(f"{label} is empty")
    return indexed


def _link_verified(
    source: Path, destination: Path, *, expected_sha256: str
) -> dict[str, Any]:
    source = _regular_file(source, label="hardlink source")
    if sha256_file(source) != expected_sha256:
        raise D1AdapterError(f"hardlink source hash differs: {source}")
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        code = (
            "D1_HARDLINK_CROSS_DEVICE"
            if error.errno == errno.EXDEV
            else "D1_HARDLINK_UNAVAILABLE"
        )
        raise D1MachineBlocker(code, f"{source} -> {destination}: {error}") from error
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) != (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise D1AdapterError(f"materialised member is not a hardlink: {destination}")
    if (
        sha256_file(source) != expected_sha256
        or sha256_file(destination) != expected_sha256
    ):
        raise D1AdapterError(f"hardlink bytes changed during materialisation: {source}")
    return {
        "source": str(source),
        "source_sha256": expected_sha256,
        "output_sha256": expected_sha256,
        "storage": "hardlink",
        "same_device_inode_verified": True,
    }


def _verify_linked_member(
    source: Path, destination: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Rebuild the immutable hardlink receipt for one resumed member."""

    source = _regular_file(source, label="resumed hardlink source")
    output = _regular_file(destination, label="resumed hardlink output")
    if (
        sha256_file(source) != expected_sha256
        or sha256_file(output) != expected_sha256
        or (source.stat().st_dev, source.stat().st_ino)
        != (output.stat().st_dev, output.stat().st_ino)
    ):
        raise D1AdapterError(f"resumed hardlink invariant differs: {destination}")
    return {
        "source": str(source),
        "source_sha256": expected_sha256,
        "output_sha256": expected_sha256,
        "storage": "hardlink",
        "same_device_inode_verified": True,
    }


def _oracle_metadata(
    predicted: Mapping[str, Any],
    *,
    sample_id: str,
    destination: Path,
    gt_mask: Path,
    gt_sha256: str,
    predicted_mask_sha256: str,
    registry_path: Path,
    adapter_sha256: str,
    loader_sha256: str,
) -> dict[str, Any]:
    if str(predicted.get("sample_id")) != sample_id:
        raise D1AdapterError(f"predicted metadata sample_id differs: {sample_id}")
    if predicted.get("mask_source") != PREDICTED_MASK_SOURCE:
        raise D1AdapterError(f"source bundle is not prediction-only: {sample_id}")
    if predicted.get("oracle_artifacts_exported") is not False:
        raise D1AdapterError(f"source bundle oracle flag is unsafe: {sample_id}")
    values = dict(predicted)
    original_values = {
        key: predicted.get(key)
        for key in (
            "mask_source",
            "oracle_artifacts_exported",
            "output_bundle",
            "prediction_mask",
            "prediction_mask_sha256",
            "materialization",
        )
    }
    values.update(
        {
            "mask_source": ORACLE_MASK_SOURCE,
            "oracle_artifacts_exported": True,
            "output_bundle": str(destination),
            "prediction_mask": str(destination / "target_mask.png"),
            "prediction_mask_sha256": gt_sha256,
            "materialization": {
                name: (
                    "hardlink_locked_gt_registry"
                    if name == "target_mask.png"
                    else "hardlink_frozen_predicted_bundle"
                )
                for name in (*BYTE_IDENTICAL_MEMBERS, "target_mask.png")
            },
            "d1_case_b_oracle_provenance": {
                "schema_version": 1,
                "scientific_role": "post-formal oracle stage-replacement diagnostic",
                "case": "B",
                "raw_candidate_regeneration_required": True,
                "filter_only_primary_allowed": False,
                "probability_input_allowed": False,
                "gt_mask_registry": artifact_record(registry_path),
                "gt_mask_source_path": str(gt_mask),
                "gt_mask_source_sha256": gt_sha256,
                "predicted_target_mask_sha256": predicted_mask_sha256,
                "frozen_loader_sha256": loader_sha256,
                "oracle_adapter_sha256": adapter_sha256,
                "replaced_predicted_metadata_values": original_values,
            },
        }
    )
    return values


def _checksum_text(hashes: Mapping[str, str]) -> str:
    return "".join(f"{hashes[name]}  {name}\n" for name in REQUIRED_BUNDLE_MEMBERS[:-1])


def _verify_resumed_oracle_sample(
    *,
    destination: Path,
    source_bundle: Path,
    predicted_hashes: Mapping[str, str],
    gt_mask: Path,
    gt_sha256: str,
    sample_id: str,
    registry_path: Path,
    adapter_sha256: str,
    loader_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Verify a sample published before a root-manifest crash, without rewriting it."""

    if destination.is_symlink() or not destination.is_dir():
        raise D1AdapterError(
            f"resumed oracle sample is missing or unsafe: {destination}"
        )
    members = {
        name: _verify_linked_member(
            source_bundle / name,
            destination / name,
            expected_sha256=predicted_hashes[name],
        )
        for name in BYTE_IDENTICAL_MEMBERS
    }
    members["target_mask.png"] = _verify_linked_member(
        gt_mask,
        destination / "target_mask.png",
        expected_sha256=gt_sha256,
    )
    expected_metadata = _oracle_metadata(
        _load_json(source_bundle / "metadata.json", label="predicted metadata"),
        sample_id=sample_id,
        destination=destination,
        gt_mask=gt_mask,
        gt_sha256=gt_sha256,
        predicted_mask_sha256=predicted_hashes["target_mask.png"],
        registry_path=registry_path,
        adapter_sha256=adapter_sha256,
        loader_sha256=loader_sha256,
    )
    observed_metadata = _load_json(
        destination / "metadata.json", label="resumed oracle metadata"
    )
    if observed_metadata != expected_metadata:
        raise D1AdapterError(f"resumed oracle metadata differs: {sample_id}")
    output_hashes = {
        **{name: predicted_hashes[name] for name in BYTE_IDENTICAL_MEMBERS},
        "target_mask.png": gt_sha256,
        "metadata.json": sha256_file(destination / "metadata.json"),
    }
    if _checksum_map(destination) != output_hashes:
        raise D1AdapterError(f"resumed oracle checksum closure differs: {sample_id}")
    expected_checksum_text = _checksum_text(output_hashes)
    if (destination / "checksums.sha256").read_text(
        encoding="utf-8"
    ) != expected_checksum_text:
        raise D1AdapterError(f"resumed checksum serialization differs: {sample_id}")
    return members, output_hashes


def _verify_existing_root(
    output_root: Path, *, registry_path: Path, expected_denominator_count: int
) -> dict[str, Any]:
    manifest = _load_json(
        output_root / CASE_B_MANIFEST_NAME, label="D1 bundle manifest"
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if manifest.get("content_sha256") != canonical_sha256(unsigned):
        raise D1AdapterError("D1 bundle manifest content hash differs")
    registry = manifest.get("gt_mask_registry")
    if not isinstance(registry, Mapping):
        raise D1AdapterError("D1 bundle manifest lacks GT registry")
    if (
        Path(str(registry.get("path", ""))).resolve(strict=False) != registry_path
        or registry.get("sha256") != sha256_file(registry_path)
        or manifest.get("denominator_sample_count") != expected_denominator_count
    ):
        raise D1AdapterError("existing D1 bundle root has a different identity")
    return manifest


def build_isolated_gt_bundle_root(
    *,
    predicted_root: Path,
    output_root: Path,
    registry_path: Path,
    registry_rows: Iterable[Mapping[str, Any]],
    authority: Mapping[str, Any],
    expected_count: int | None = None,
    frozen_loader_source: Path = FROZEN_D1_LOADER,
    expected_loader_sha256: str = FROZEN_D1_LOADER_SHA256,
    adapter_source: Path | None = None,
    resume: bool = False,
) -> Path:
    """Materialise the canonical D1 oracle root without copying large members.

    ``registry_rows`` must only be obtained after ``authority`` was returned by
    the exactly-once P4 execution guard.  The CLI enforces that ordering.
    """

    registry = _regular_file(registry_path, label="GT registry")
    assert_gt_bulk_authority(authority, registry_path=registry)
    predicted = predicted_root.expanduser().resolve(strict=False)
    output = output_root.expanduser().resolve(strict=False)
    if not predicted.is_dir() or predicted.is_symlink():
        raise D1AdapterError(f"predicted bundle root is missing or unsafe: {predicted}")
    if (
        output == predicted
        or output in predicted.parents
        or predicted in output.parents
    ):
        raise D1AdapterError("predicted and oracle bundle roots must be disjoint")
    predicted_rows = _index_rows(
        _load_jsonl(predicted / "manifest.jsonl", label="predicted manifest"),
        label="predicted manifest",
    )
    gt_rows = _index_rows(registry_rows, label="GT registry")
    if set(predicted_rows) != set(gt_rows):
        missing = sorted(set(predicted_rows) - set(gt_rows))[:5]
        extra = sorted(set(gt_rows) - set(predicted_rows))[:5]
        raise D1AdapterError(
            f"GT/predicted coverage differs: missing={missing} extra={extra}"
        )
    count = len(predicted_rows)
    if expected_count is not None and count != expected_count:
        raise D1AdapterError(
            f"bundle count {count} differs from locked count {expected_count}"
        )
    expected_count = count
    evaluable_ids = [
        sample_id
        for sample_id in predicted_rows
        if (
            gt_rows[sample_id].get("mapping_status") == "PASS"
            and gt_rows[sample_id].get("pixel_qa_status") == "P2_MAPPING_QA_PASS"
            and gt_rows[sample_id].get("bulk_gt_pixels_read") is True
        )
    ]
    unresolved_ids = sorted(set(predicted_rows).difference(evaluable_ids))
    if (output / CASE_B_MANIFEST_NAME).exists():
        if not resume:
            raise FileExistsError(
                f"D1 oracle bundle exists; pass --resume: {output / CASE_B_MANIFEST_NAME}"
            )
        _verify_existing_root(
            output,
            registry_path=registry,
            expected_denominator_count=count,
        )
        verify_isolated_bundle_root(
            output, expected_loader_sha256=expected_loader_sha256
        )
        return output

    loader = _verify_frozen_file(
        frozen_loader_source, expected_loader_sha256, label="frozen D1 loader"
    )
    adapter = _regular_file(
        adapter_source or Path(__file__), label="D1 oracle adapter source"
    )
    adapter_sha256 = sha256_file(adapter)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError(
            f"partial D1 oracle bundle exists; pass --resume: {output}"
        )
    manifest_rows: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for manifest_index, sample_id in enumerate(evaluable_ids):
        source_row = predicted_rows[sample_id]
        gt_row = gt_rows[sample_id]
        if (
            source_row.get("ready") is not True
            or source_row.get("ready_for_anygrasp") is not True
        ):
            raise D1AdapterError(f"predicted bundle is not ready: {sample_id}")
        if source_row.get("blockers") not in (None, []):
            raise D1AdapterError(f"predicted bundle has blockers: {sample_id}")
        # Only the exact P2 PASS partition is allowed to expose GT pixels to
        # the frozen candidate generator.  Every other denominator member is
        # carried by the root manifest as a technical/no-output complement.
        source_bundle = predicted / sample_id
        if not source_bundle.is_dir() or source_bundle.is_symlink():
            raise D1AdapterError(
                f"predicted bundle is missing or unsafe: {source_bundle}"
            )
        predicted_hashes = _checksum_map(source_bundle)
        gt_mask = _regular_file(
            gt_row.get("original_gt_mask_path", ""), label="locked GT mask"
        )
        gt_sha256 = str(gt_row.get("original_gt_mask_sha256", ""))
        if not gt_sha256 or sha256_file(gt_mask) != gt_sha256:
            raise D1AdapterError(f"locked GT mask hash differs: {sample_id}")
        destination = output / sample_id
        staging = output / f".{sample_id}.{os.getpid()}.staging"
        members: dict[str, dict[str, Any]]
        output_hashes: dict[str, str]
        if destination.exists():
            if not resume:
                raise FileExistsError(
                    f"partial oracle sample exists; pass --resume: {destination}"
                )
            members, output_hashes = _verify_resumed_oracle_sample(
                destination=destination,
                source_bundle=source_bundle,
                predicted_hashes=predicted_hashes,
                gt_mask=gt_mask,
                gt_sha256=gt_sha256,
                sample_id=sample_id,
                registry_path=registry,
                adapter_sha256=adapter_sha256,
                loader_sha256=expected_loader_sha256,
            )
        else:
            staging.mkdir(mode=0o700)
            try:
                members = {}
                for name in BYTE_IDENTICAL_MEMBERS:
                    members[name] = _link_verified(
                        source_bundle / name,
                        staging / name,
                        expected_sha256=predicted_hashes[name],
                    )
                members["target_mask.png"] = _link_verified(
                    gt_mask, staging / "target_mask.png", expected_sha256=gt_sha256
                )
                metadata = _oracle_metadata(
                    _load_json(source_bundle / "metadata.json", label="predicted metadata"),
                    sample_id=sample_id,
                    destination=destination,
                    gt_mask=gt_mask,
                    gt_sha256=gt_sha256,
                    predicted_mask_sha256=predicted_hashes["target_mask.png"],
                    registry_path=registry,
                    adapter_sha256=adapter_sha256,
                    loader_sha256=expected_loader_sha256,
                )
                exclusive_json(staging / "metadata.json", metadata)
                output_hashes = {
                    **{
                        name: predicted_hashes[name]
                        for name in BYTE_IDENTICAL_MEMBERS
                    },
                    "target_mask.png": gt_sha256,
                    "metadata.json": sha256_file(staging / "metadata.json"),
                }
                exclusive_text(
                    staging / "checksums.sha256", _checksum_text(output_hashes)
                )
                _checksum_map(staging)
                os.replace(staging, destination)
            finally:
                if staging.exists():
                    for child in staging.iterdir():
                        child.unlink()
                    staging.rmdir()

        row = dict(source_row)
        row.update(
            {
                "bundle_dir": str(destination),
                "path": str(destination),
                "manifest_index": manifest_index,
                "selected_source": ORACLE_MASK_SOURCE,
                "target_mask_sha256": gt_sha256,
                "oracle_artifacts_exported": True,
                "d1_case": "B",
            }
        )
        manifest_rows.append(row)
        sample_records.append(
            {
                "sample_id": sample_id,
                "source_predicted_bundle": str(source_bundle),
                "output_bundle": str(destination),
                "predicted_target_mask_sha256": predicted_hashes["target_mask.png"],
                "oracle_target_mask_sha256": gt_sha256,
                "members": {
                    **members,
                    "metadata.json": {
                        "storage": "generated_oracle_provenance",
                        "output_sha256": output_hashes["metadata.json"],
                    },
                    "checksums.sha256": {
                        "storage": "generated",
                        "output_sha256": sha256_file(destination / "checksums.sha256"),
                    },
                },
            }
        )

    manifest_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in manifest_rows
    )
    exclusive_text(output / "manifest.jsonl", manifest_text)
    root_manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "COMPLETE",
        "route": "d1",
        "branch": "gt_oracle",
        "d1_case": "B",
        "raw_candidate_regeneration_required": True,
        "filter_only_primary_allowed": False,
        "probability_input_allowed": False,
        # ``sample_count`` is the exact executable universe consumed by the
        # frozen loader/scorer.  The separate denominator fields prevent an
        # unresolved mapping from either aborting the whole route or silently
        # disappearing from the scientific denominator.
        "sample_count": len(evaluable_ids),
        "denominator_sample_count": expected_count,
        "evaluable_sample_count": len(evaluable_ids),
        "unresolved_sample_count": len(unresolved_ids),
        "unresolved_sample_ids_sha256": canonical_sha256(unresolved_ids),
        "unresolved_samples": [
            {
                "sample_id": sample_id,
                "mapping_status": gt_rows[sample_id].get("mapping_status"),
                "pixel_qa_status": gt_rows[sample_id].get("pixel_qa_status"),
                "bulk_gt_pixels_read": gt_rows[sample_id].get("bulk_gt_pixels_read"),
            }
            for sample_id in unresolved_ids
        ],
        "technical_complement_required": True,
        "predicted_manifest": artifact_record(predicted / "manifest.jsonl"),
        "oracle_manifest": artifact_record(output / "manifest.jsonl"),
        "gt_mask_registry": artifact_record(registry),
        "frozen_loader": artifact_record(loader),
        "oracle_adapter": artifact_record(adapter),
        "byte_identical_members": list(BYTE_IDENTICAL_MEMBERS),
        "replacement_members": ["target_mask.png", "metadata.json", "checksums.sha256"],
        "hardlink_copy_fallback_allowed": False,
        "samples": sample_records,
    }
    root_manifest["content_sha256"] = canonical_sha256(root_manifest)
    exclusive_json(output / CASE_B_MANIFEST_NAME, root_manifest)
    return output


def verify_isolated_bundle_root(
    root: Path, *, expected_loader_sha256: str = FROZEN_D1_LOADER_SHA256
) -> dict[str, Any]:
    """Rehash the complete isolated inventory before candidate generation."""

    bundle_root = root.expanduser().resolve(strict=False)
    manifest = _load_json(
        bundle_root / CASE_B_MANIFEST_NAME, label="D1 bundle manifest"
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    if manifest.get("content_sha256") != canonical_sha256(unsigned):
        raise D1AdapterError("D1 bundle manifest content hash differs")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("d1_case") != "B"
        or manifest.get("raw_candidate_regeneration_required") is not True
        or manifest.get("filter_only_primary_allowed") is not False
        or manifest.get("probability_input_allowed") is not False
    ):
        raise D1AdapterError("D1 bundle root is not an executable Case-B oracle root")
    loader = manifest.get("frozen_loader")
    if (
        not isinstance(loader, Mapping)
        or loader.get("sha256") != expected_loader_sha256
    ):
        raise D1AdapterError("D1 bundle root binds a different frozen loader")
    for label in (
        "frozen_loader",
        "oracle_adapter",
        "predicted_manifest",
        "gt_mask_registry",
    ):
        record = manifest.get(label)
        if not isinstance(record, Mapping):
            raise D1AdapterError(f"D1 bundle root lacks {label}")
        source = _regular_file(record.get("path", ""), label=label)
        if sha256_file(source) != record.get("sha256"):
            raise D1AdapterError(f"D1 bundle root {label} hash differs")
    rows = _index_rows(
        _load_jsonl(bundle_root / "manifest.jsonl", label="D1 oracle manifest"),
        label="D1 oracle manifest",
    )
    if len(rows) != manifest.get("sample_count"):
        raise D1AdapterError("D1 manifest sample count differs")
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(rows):
        raise D1AdapterError("D1 bundle inventory count differs")
    unresolved = manifest.get("unresolved_samples")
    if not isinstance(unresolved, list):
        raise D1AdapterError("D1 bundle lacks its unresolved complement")
    unresolved_ids = sorted(
        _safe_id(record.get("sample_id"))
        for record in unresolved
        if isinstance(record, Mapping)
    )
    if len(unresolved_ids) != len(unresolved) or len(set(unresolved_ids)) != len(
        unresolved_ids
    ):
        raise D1AdapterError("D1 unresolved complement has invalid identities")
    executable_ids = set(rows)
    if executable_ids.intersection(unresolved_ids):
        raise D1AdapterError("D1 executable and unresolved partitions overlap")
    if (
        manifest.get("sample_count") != len(executable_ids)
        or manifest.get("evaluable_sample_count") != len(executable_ids)
        or manifest.get("unresolved_sample_count") != len(unresolved_ids)
        or manifest.get("denominator_sample_count")
        != len(executable_ids) + len(unresolved_ids)
        or manifest.get("unresolved_sample_ids_sha256")
        != canonical_sha256(unresolved_ids)
        or manifest.get("technical_complement_required") is not True
    ):
        raise D1AdapterError("D1 executable/denominator partition differs")
    for record in records:
        if not isinstance(record, Mapping):
            raise D1AdapterError("D1 bundle inventory row is malformed")
        sample_id = _safe_id(record.get("sample_id"))
        if sample_id not in rows:
            raise D1AdapterError(f"D1 bundle inventory has unknown sample: {sample_id}")
        hashes = _checksum_map(bundle_root / sample_id)
        if hashes["target_mask.png"] != record.get("oracle_target_mask_sha256"):
            raise D1AdapterError(f"D1 oracle target mask differs: {sample_id}")
        for name in BYTE_IDENTICAL_MEMBERS:
            member = record.get("members", {}).get(name, {})
            if not isinstance(member, Mapping) or hashes[name] != member.get(
                "source_sha256"
            ):
                raise D1AdapterError(f"D1 byte invariant differs: {sample_id}/{name}")
            source = _regular_file(
                member.get("source", ""), label="linked predicted member"
            )
            output = bundle_root / sample_id / name
            if sha256_file(source) != hashes[name] or (
                source.stat().st_dev,
                source.stat().st_ino,
            ) != (output.stat().st_dev, output.stat().st_ino):
                raise D1AdapterError(
                    f"D1 hardlink invariant differs: {sample_id}/{name}"
                )
        mask_member = record.get("members", {}).get("target_mask.png", {})
        if not isinstance(mask_member, Mapping):
            raise D1AdapterError(f"D1 target-mask inventory differs: {sample_id}")
        mask_source = _regular_file(
            mask_member.get("source", ""), label="linked locked GT mask"
        )
        mask_output = bundle_root / sample_id / "target_mask.png"
        if sha256_file(mask_source) != hashes["target_mask.png"] or (
            mask_source.stat().st_dev,
            mask_source.stat().st_ino,
        ) != (mask_output.stat().st_dev, mask_output.stat().st_ino):
            raise D1AdapterError(f"D1 GT-mask hardlink invariant differs: {sample_id}")
    return manifest


def make_oracle_bundle_index_class(
    frozen_adapter_module: ModuleType,
    *,
    expected_loader_sha256: str = FROZEN_D1_LOADER_SHA256,
) -> type[Any]:
    """Create the narrow adapter injected into the byte-frozen candidate runner."""

    base_index = frozen_adapter_module.OcidVlgBundleIndex
    source_path = Path(inspect.getsourcefile(base_index) or "").resolve(strict=False)
    _verify_frozen_file(
        source_path, expected_loader_sha256, label="injected frozen loader"
    )

    class D1OracleBundleIndex:
        def __init__(self, dataset_root: Path, mask_root: Path, *, split: str = "test"):
            if split != "test":
                raise D1AdapterError("D1 oracle adapter is frozen to the test split")
            self.dataset_root = Path(dataset_root).expanduser().resolve()
            self.mask_root = Path(mask_root).expanduser().resolve()
            self.split = split
            verify_isolated_bundle_root(
                self.mask_root, expected_loader_sha256=expected_loader_sha256
            )
            self.rows = _load_jsonl(
                self.mask_root / "manifest.jsonl", label="D1 oracle manifest"
            )
            self.by_id = _index_rows(self.rows, label="D1 oracle manifest")
            self.manifest_path = self.mask_root / "manifest.jsonl"

        def sample_ids(
            self, sample_id: str | None = None, limit: int | None = None
        ) -> list[str]:
            if sample_id is not None:
                sample_id = _safe_id(sample_id)
                if sample_id not in self.by_id:
                    raise KeyError(f"unknown sample_id: {sample_id}")
                return [sample_id]
            identifiers = [str(row["sample_id"]) for row in self.rows]
            if limit is not None:
                if limit <= 0:
                    raise ValueError("sample limit must be positive")
                identifiers = identifiers[: int(limit)]
            return identifiers

        def iter_samples(
            self, sample_ids: Iterable[str], **kwargs: Any
        ) -> Iterable[Any]:
            for sample_id in sample_ids:
                yield self.load_sample(sample_id, **kwargs)

        def load_sample(self, sample_id: str, **kwargs: Any) -> Any:
            sample_id = _safe_id(sample_id)
            if kwargs.get("mask_source", "binary_prediction") != "binary_prediction":
                raise D1AdapterError(
                    "D1 oracle adapter forbids predicted-probability mask input"
                )
            actual_bundle = self.mask_root / sample_id
            actual_metadata = _load_json(
                actual_bundle / "metadata.json", label="D1 oracle metadata"
            )
            if (
                actual_metadata.get("mask_source") != ORACLE_MASK_SOURCE
                or actual_metadata.get("oracle_artifacts_exported") is not True
            ):
                raise D1AdapterError(f"D1 oracle provenance differs: {sample_id}")
            with tempfile.TemporaryDirectory(prefix="d1-oracle-loader-") as temporary:
                view_root = Path(temporary)
                view_bundle = view_root / sample_id
                view_bundle.mkdir()
                for name in REQUIRED_BUNDLE_MEMBERS:
                    if name in {"metadata.json", "checksums.sha256"}:
                        continue
                    expected = sha256_file(actual_bundle / name)
                    _link_verified(
                        actual_bundle / name,
                        view_bundle / name,
                        expected_sha256=expected,
                    )
                normalised = dict(actual_metadata)
                normalised["mask_source"] = PREDICTED_MASK_SOURCE
                normalised["oracle_artifacts_exported"] = False
                exclusive_json(view_bundle / "metadata.json", normalised)
                hashes = {
                    name: sha256_file(view_bundle / name)
                    for name in REQUIRED_BUNDLE_MEMBERS[:-1]
                }
                exclusive_text(view_bundle / "checksums.sha256", _checksum_text(hashes))
                source_row = dict(self.by_id[sample_id])
                source_row.update(
                    {
                        "bundle_dir": str(view_bundle),
                        "path": str(view_bundle),
                        "mask_source": PREDICTED_MASK_SOURCE,
                        "oracle_artifacts_exported": False,
                    }
                )
                exclusive_text(
                    view_root / "manifest.jsonl",
                    json.dumps(source_row, sort_keys=True, ensure_ascii=False) + "\n",
                )
                frozen_index = base_index(
                    self.dataset_root, view_root, split=self.split
                )
                sample = frozen_index.load_sample(sample_id, **kwargs)
                return replace(
                    sample,
                    bundle_dir=actual_bundle,
                    metadata=actual_metadata,
                )

    D1OracleBundleIndex.__name__ = "D1OracleBundleIndex"
    return D1OracleBundleIndex


def candidate_inventory_from_summary(
    candidate_root: Path, *, expected_sample_ids: set[str] | None = None
) -> CandidateInventory:
    """Derive scorer expectations from a completed Case-B candidate summary."""

    root = candidate_root.expanduser().resolve(strict=False)
    summary = _regular_file(root / "summary.csv", label="D1 candidate summary")
    rows = list(csv.DictReader(summary.open(encoding="utf-8", newline="")))
    identifiers = [_safe_id(row.get("sample_id")) for row in rows]
    if not rows or len(set(identifiers)) != len(rows):
        raise D1AdapterError("D1 candidate summary is empty or has duplicate samples")
    if expected_sample_ids is not None and set(identifiers) != set(expected_sample_ids):
        missing = sorted(set(expected_sample_ids).difference(identifiers))[:5]
        extra = sorted(set(identifiers).difference(expected_sample_ids))[:5]
        raise D1AdapterError(
            f"D1 candidate summary sample universe differs: missing={missing} extra={extra}"
        )
    statuses = {str(row.get("status")) for row in rows}
    if not statuses.issubset({"success_nonempty", "success_empty"}):
        raise D1AdapterError(
            f"D1 candidate summary is not terminal: {sorted(statuses)}"
        )
    counts: list[int] = []
    for row in rows:
        try:
            count = int(row["post_nms_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise D1AdapterError("D1 candidate summary has an invalid count") from error
        if count < 0 or (row["status"] == "success_empty") != (count == 0):
            raise D1AdapterError("D1 candidate status/count contract differs")
        counts.append(count)
    return CandidateInventory(
        samples=len(rows),
        nonempty=sum(count > 0 for count in counts),
        empty=sum(count == 0 for count in counts),
        candidates=sum(counts),
    )


def _verified_scored_json(path: Path, *, sample_id: str) -> list[dict[str, Any]]:
    payload = _load_json(path, label="D1 scored candidate payload")
    metadata = payload.get("metadata")
    records = payload.get("candidates")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("sample_id") != sample_id
        or metadata.get("scoring_status") != "scored_nonempty"
        or not isinstance(records, list)
        or not records
    ):
        raise D1AdapterError(f"D1 scored payload identity differs: {sample_id}")
    result: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    previous_key: tuple[float, str] | None = None
    for expected_rank, item in enumerate(records, 1):
        if not isinstance(item, Mapping):
            raise D1AdapterError(f"D1 scored candidate row is malformed: {sample_id}")
        row = dict(item)
        source_id = str(row.get("candidate_id", ""))
        try:
            rank = int(row["gqcnn_rank"])
            score = float(row["gqcnn_q_value"])
            source_index = int(row["source_candidate_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise D1AdapterError(
                f"D1 scored candidate rank/score is malformed: {sample_id}"
            ) from error
        if (
            not source_id
            or source_id in source_ids
            or source_index < 0
            or rank != expected_rank
            or not math.isfinite(score)
        ):
            raise D1AdapterError(f"D1 scored candidate identity differs: {sample_id}")
        key = (-score, source_id)
        if previous_key is not None and key < previous_key:
            raise D1AdapterError(f"D1 scored candidate ordering differs: {sample_id}")
        previous_key = key
        source_ids.add(source_id)
        result.append(row)
    return result


def _scored_marker(
    sample_dir: Path, *, sample_id: str, expected_candidate_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    marker_path = _regular_file(
        sample_dir / "_SCORING_COMPLETE.json", label="D1 scoring marker"
    )
    marker = _load_json(marker_path, label="D1 scoring marker")
    status = marker.get("scoring_status")
    if (
        marker.get("sample_id") != sample_id
        or status not in {"scored_nonempty", "skipped_valid_empty"}
        or int(marker.get("source_candidate_count", -1))
        != expected_candidate_count
        or int(marker.get("gqcnn_scored_count", -1))
        != expected_candidate_count
    ):
        raise D1AdapterError(f"D1 scoring marker identity/count differs: {sample_id}")
    if (status == "skipped_valid_empty") != (expected_candidate_count == 0):
        raise D1AdapterError(f"D1 scoring empty status differs: {sample_id}")
    required = marker.get("required_files")
    hashes = marker.get("required_file_hashes")
    if not isinstance(required, list) or not isinstance(hashes, Mapping):
        raise D1AdapterError(f"D1 scoring marker inventory is malformed: {sample_id}")
    expected_required = (
        {"scoring_metadata.json"}
        if status == "skipped_valid_empty"
        else {
            "gqcnn_scored_candidates.npz",
            "gqcnn_scored_candidates.json",
            "gqcnn_scored_candidates.csv",
            "gqcnn_top1.json",
            "gqcnn_top5.json",
            "scoring_metadata.json",
        }
    )
    if set(required) != expected_required or set(hashes) != expected_required:
        raise D1AdapterError(f"D1 scoring marker inventory differs: {sample_id}")
    for name in required:
        if Path(str(name)).name != str(name):
            raise D1AdapterError(f"unsafe D1 scoring member: {name!r}")
        member = _regular_file(sample_dir / str(name), label="D1 scored member")
        if sha256_file(member) != hashes[name]:
            raise D1AdapterError(f"D1 scored member hash differs: {sample_id}/{name}")
    records = (
        []
        if status == "skipped_valid_empty"
        else _verified_scored_json(
            sample_dir / "gqcnn_scored_candidates.json", sample_id=sample_id
        )
    )
    if len(records) != expected_candidate_count:
        raise D1AdapterError(f"D1 scored candidate count differs: {sample_id}")
    return marker, records


def _write_or_verify_parquet(frame: Any, path: Path) -> Path:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        import pandas as pd

        existing = pd.read_parquet(destination)
        if list(existing.columns) != list(frame.columns) or not existing.equals(frame):
            raise D1AdapterError(f"existing D1 canonical Parquet differs: {destination}")
        return destination
    return atomic_parquet(frame, destination)


def assemble_d1_scored_outputs(
    *,
    run_dir: Path,
    bundle_root: Path,
    candidate_root: Path,
    scored_root: Path,
    protocol_lock_path: Path,
    execution_claim_path: Path,
    branch: str = "gt_oracle",
    expected_denominator_count: int = 7_675,
    expected_loader_sha256: str = FROZEN_D1_LOADER_SHA256,
) -> Path:
    """Close frozen D1 scoring into one denominator-preserving route output.

    This function is deliberately label-free.  It verifies the scorer's
    immutable per-sample payloads, rebuilds the exact q-value ordering and
    adds the unresolved P2 partition only as technical/no-output sample rows.
    Correctness labels and first-positive ranks are computed later by the
    post-lock evaluator, never here.
    """

    import pandas as pd

    root = run_dir.expanduser().resolve(strict=False)
    if branch not in {"predicted", "gt_oracle"}:
        raise D1AdapterError(f"unsupported D1 output branch: {branch}")
    bundle_path = bundle_root.expanduser().resolve(strict=False)
    candidates_path = candidate_root.expanduser().resolve(strict=False)
    scores_path = scored_root.expanduser().resolve(strict=False)
    for label, path in (
        ("bundle root", bundle_path),
        ("candidate root", candidates_path),
        ("scored root", scores_path),
    ):
        if path != root and root not in path.parents:
            raise D1AdapterError(f"D1 {label} must be isolated inside the new run")
    protocol = _regular_file(protocol_lock_path, label="counterfactual protocol lock")
    claim = _regular_file(execution_claim_path, label="counterfactual execution claim")
    if protocol != root / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json" or (
        claim != root / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
    ):
        raise D1AdapterError("D1 protocol/claim paths differ from the new run")
    claim_value = _load_json(claim, label="D1 secondary execution claim")
    if (
        claim_value.get("scope") != "d1_secondary"
        or claim_value.get("status") != "RUNNING"
        or int(claim_value.get("execution_count", -1)) != 1
        or claim_value.get("protocol_lock_file_sha256") != sha256_file(protocol)
    ):
        raise D1AdapterError("D1 secondary execution claim differs")
    bundle = verify_isolated_bundle_root(
        bundle_path, expected_loader_sha256=expected_loader_sha256
    )
    if int(bundle.get("denominator_sample_count", -1)) != int(
        expected_denominator_count
    ):
        raise D1AdapterError("D1 bundle denominator differs")
    executable_ids = {
        _safe_id(record.get("sample_id"))
        for record in bundle.get("samples", [])
        if isinstance(record, Mapping)
    }
    unresolved_records = bundle.get("unresolved_samples")
    if not isinstance(unresolved_records, list):
        raise D1AdapterError("D1 bundle lacks unresolved samples")
    unresolved_ids = {
        _safe_id(record.get("sample_id"))
        for record in unresolved_records
        if isinstance(record, Mapping)
    }
    if len(executable_ids) + len(unresolved_ids) != expected_denominator_count:
        raise D1AdapterError("D1 bundle partition does not preserve the denominator")

    inventory = candidate_inventory_from_summary(
        candidates_path, expected_sample_ids=executable_ids
    )
    summary_path = _regular_file(candidates_path / "summary.csv", label="D1 summary")
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
    candidate_counts = {
        _safe_id(row.get("sample_id")): int(row["post_nms_count"])
        for row in summary_rows
    }
    progress = _load_json(scores_path / "progress.json", label="D1 scorer progress")
    statistics = _load_json(
        scores_path / "run_statistics.json", label="D1 scorer statistics"
    )
    expected_root_counts = {
        "total_samples": inventory.samples,
        "terminal_samples": inventory.samples,
        "scored_candidates": inventory.candidates,
    }
    if any(int(progress.get(key, -1)) != value for key, value in expected_root_counts.items()):
        raise D1AdapterError("D1 scorer progress counts differ")
    if (
        int(progress.get("completed_nonempty_samples", -1)) != inventory.nonempty
        or int(progress.get("skipped_empty_samples", -1)) != inventory.empty
        or int(progress.get("failed_samples", -1)) != 0
        or int(progress.get("remaining_candidates", -1)) != 0
    ):
        raise D1AdapterError("D1 scorer progress is not a clean terminal run")
    expected_statistics = {
        "total_samples": inventory.samples,
        "terminal_samples": inventory.samples,
        "scored_nonempty_samples": inventory.nonempty,
        "skipped_valid_empty_samples": inventory.empty,
        "failed_samples": 0,
        "corrupt_committed_samples": 0,
        "expected_candidates": inventory.candidates,
        "scored_candidates": inventory.candidates,
        "finite_q_values": inventory.candidates,
        "invalid_q_values": 0,
    }
    if any(
        int(statistics.get(key, -1)) != value
        for key, value in expected_statistics.items()
    ):
        raise D1AdapterError("D1 scorer statistics differ")
    scorer_manifest_rows = _load_jsonl(
        scores_path / "scoring_manifest.jsonl", label="D1 scorer manifest"
    )
    scorer_manifest_by_id = _index_rows(
        scorer_manifest_rows, label="D1 scorer manifest"
    )
    if set(scorer_manifest_by_id) != executable_ids:
        raise D1AdapterError("D1 scorer manifest sample universe differs")
    scored_sources: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for sample_id in sorted(executable_ids):
        marker, records = _scored_marker(
            scores_path / sample_id,
            sample_id=sample_id,
            expected_candidate_count=candidate_counts[sample_id],
        )
        root_row = scorer_manifest_by_id[sample_id]
        if (
            root_row.get("scoring_status") != marker.get("scoring_status")
            or int(root_row.get("source_candidate_count", -1))
            != candidate_counts[sample_id]
            or int(root_row.get("gqcnn_scored_count", -1))
            != candidate_counts[sample_id]
            or root_row.get("top1_candidate_id") != marker.get("top1_candidate_id")
            or root_row.get("source_candidate_sha256")
            != marker.get("source_candidate_sha256")
            or root_row.get("model_config_hash") != marker.get("model_config_hash")
        ):
            raise D1AdapterError(
                f"D1 root/per-sample scoring provenance differs: {sample_id}"
            )
        normalized: list[dict[str, Any]] = []
        for item in records:
            try:
                center_x = float(item.get("center_u_px", item.get("centre_u_px")))
                center_y = float(item.get("center_v_px", item.get("centre_v_px")))
                angle = item.get("angle_deg")
                angle_deg = (
                    float(angle)
                    if angle is not None
                    else math.degrees(float(item["angle_rad"]))
                )
                width_px = float(item["width_px"])
                height_px = float(
                    item.get("height_px", item.get("rectangle_height_px", 20.0))
                )
                source_index = int(item["source_candidate_index"])
                native_rank = int(item["gqcnn_rank"])
                score = float(item["gqcnn_q_value"])
            except (KeyError, TypeError, ValueError) as error:
                raise D1AdapterError(
                    f"D1 scored geometry is malformed: {sample_id}"
                ) from error
            geometry = {
                "cx_px": center_x,
                "cy_px": center_y,
                "theta_deg": angle_deg,
                "width_px": width_px,
                "height_px": height_px,
            }
            candidate_id = stable_candidate_id(
                sample_id=sample_id,
                route="d1",
                branch=branch,
                source_candidate_index=source_index,
                candidate=geometry,
            )
            row = {
                "sample_id": sample_id,
                "route": "D1",
                "branch": branch,
                "candidate_id": candidate_id,
                "source_candidate_id": str(item["candidate_id"]),
                "source_candidate_index": source_index,
                "native_rank": native_rank,
                "native_score": score,
                **geometry,
                "jaw_width_px": width_px,
                "rectangle_height_px": height_px,
                "pool_top5": native_rank <= 5,
                "pool_top10": native_rank <= 10,
                "pool_allnms": True,
                "q_value_is_calibrated_probability": False,
            }
            normalized.append(row)
        candidate_rows.extend(normalized)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "route": "D1",
                "branch": branch,
                "candidate_count": len(normalized),
                "candidate_count_top5": min(5, len(normalized)),
                "candidate_count_top10": min(10, len(normalized)),
                "no_output": not normalized,
                "technical_failure": False,
                "status": "NO_OUTPUT" if not normalized else "COMPLETE",
                "native_candidate_id": (
                    None if not normalized else normalized[0]["candidate_id"]
                ),
                "native_score": None if not normalized else normalized[0]["native_score"],
            }
        )
        scored_sources[sample_id] = {
            "marker": artifact_record(scores_path / sample_id / "_SCORING_COMPLETE.json"),
            "payload": (
                None
                if not normalized
                else artifact_record(
                    scores_path / sample_id / "gqcnn_scored_candidates.json"
                )
            ),
            "candidate_count": len(normalized),
            "scoring_status": marker["scoring_status"],
        }
    for item in sorted(unresolved_records, key=lambda row: str(row.get("sample_id"))):
        sample_id = _safe_id(item.get("sample_id"))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "route": "D1",
                "branch": branch,
                "candidate_count": 0,
                "candidate_count_top5": 0,
                "candidate_count_top10": 0,
                "no_output": True,
                "technical_failure": True,
                "status": "TECHNICAL_FAILURE",
                "native_candidate_id": None,
                "native_score": None,
            }
        )
    if len(sample_rows) != expected_denominator_count:
        raise D1AdapterError("D1 canonical sample output lost the denominator")

    canonical_root = root / "06_gtmask_predictions/d1" / branch
    candidates_frame = pd.DataFrame(candidate_rows).sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    )
    samples_frame = pd.DataFrame(sample_rows).sort_values(
        ["sample_id"], kind="mergesort"
    )
    candidate_output = _write_or_verify_parquet(
        candidates_frame.reset_index(drop=True), canonical_root / "per_candidate.parquet"
    )
    sample_output = _write_or_verify_parquet(
        samples_frame.reset_index(drop=True), canonical_root / "per_sample.parquet"
    )
    source_inventory: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "sample_count": len(scored_sources),
        "candidate_count": len(candidate_rows),
        "samples": scored_sources,
    }
    source_inventory["content_sha256"] = canonical_sha256(source_inventory)
    inventory_path = canonical_root / "scored_source_inventory.json"
    if inventory_path.exists():
        if _load_json(inventory_path, label="D1 scored source inventory") != source_inventory:
            raise D1AdapterError("existing D1 scored source inventory differs")
    else:
        exclusive_json(inventory_path, source_inventory)
    required_roots = {
        "candidate_summary": summary_path,
        "candidate_run_config": candidates_path / "run_config.json",
        "scorer_run_config": scores_path / "run_config.json",
        "scorer_progress": scores_path / "progress.json",
        "scorer_summary": scores_path / "summary.csv",
        "scorer_manifest": scores_path / "scoring_manifest.jsonl",
        "scorer_statistics": scores_path / "run_statistics.json",
    }
    source_records = {
        label: artifact_record(_regular_file(path, label=label))
        for label, path in required_roots.items()
    }
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "COMPLETE",
        "route": "d1",
        "branch": branch,
        "d1_case": "B",
        "sample_count": expected_denominator_count,
        "evaluable_sample_count": len(executable_ids),
        "technical_complement_count": len(unresolved_ids),
        "candidate_count": len(candidate_rows),
        "top5_candidate_count": sum(
            min(5, count) for count in candidate_counts.values()
        ),
        "top10_candidate_count": sum(
            min(10, count) for count in candidate_counts.values()
        ),
        "allnms_candidate_count": inventory.candidates,
        "raw_candidate_regeneration_executed": True,
        "filter_only_primary_used": False,
        "q_value_ranking": "raw q descending; exact ties by source candidate_id ascending",
        "q_value_is_calibrated_probability": False,
        "protocol_lock": artifact_record(protocol),
        "execution_claim": artifact_record(claim),
        "bundle_manifest": artifact_record(bundle_path / CASE_B_MANIFEST_NAME),
        "gt_mask_registry": dict(bundle["gt_mask_registry"]),
        "source_artifacts": source_records,
        "scored_source_inventory": artifact_record(inventory_path),
        "per_sample": artifact_record(sample_output),
        "per_candidate": artifact_record(candidate_output),
        "candidates": artifact_record(candidate_output),
        "candidate_id_namespace": "sha256(sample,route,branch,source_index,geometry)",
    }
    result["content_sha256"] = canonical_sha256(result)
    destination = canonical_root / "manifest.json"
    if destination.exists():
        if _load_json(destination, label="D1 canonical output manifest") != result:
            raise D1AdapterError("existing D1 canonical output manifest differs")
        return destination
    exclusive_json(destination, result)
    return destination


def _candidate_arguments(
    *,
    dataset_root: Path,
    mask_root: Path,
    output_root: Path,
    resume: bool,
    config_path: Path = FROZEN_D1_CONFIG,
) -> list[str]:
    arguments = [
        "--dataset-root",
        str(dataset_root.expanduser().resolve()),
        "--mask-root",
        str(mask_root.expanduser().resolve()),
        "--output-dir",
        str(output_root.expanduser().resolve()),
        "--config",
        str(_regular_file(config_path, label="D1 candidate config")),
        "--mode",
        "candidate-only",
        "--num-candidates",
        "256",
        "--top-k",
        "30",
        "--seed",
        "42",
        "--sample-seed-mode",
        "stable-sha256",
        "--seed-namespace",
        "hierfilm-modular-formal-v1",
        "--visualize-policy",
        "none",
        "--status-every",
        "25",
        "--checkpoint-every",
        "25",
        "--max-failures",
        "1",
    ]
    if resume:
        arguments.extend(["--resume", "--verify-existing", "--retry-failures"])
    return arguments


def build_predicted_replay_command(
    *,
    python: Path,
    dataset_root: Path,
    predicted_root: Path,
    output_root: Path,
    source_view_root: Path | None = None,
    resume: bool = True,
) -> list[str]:
    """Build the GT-free frozen predicted replay command."""

    verify_frozen_d1_sources()
    script = (
        FROZEN_D1_CANDIDATE_SCRIPT
        if source_view_root is None
        else source_view_root.expanduser().resolve()
        / "scripts/run_hifics_dexnet_candidates.py"
    )
    config = (
        FROZEN_D1_CONFIG
        if source_view_root is None
        else source_view_root.expanduser().resolve()
        / "configs/dexnet_candidates_formal_no_refinement.yaml"
    )
    return [
        str(_executable_file(python, label="Python launcher")),
        str(_regular_file(script, label="D1 source-view candidate script")),
        *_candidate_arguments(
            dataset_root=dataset_root,
            mask_root=predicted_root,
            output_root=output_root,
            resume=resume,
            config_path=config,
        ),
    ]


def build_gt_candidate_command(
    *,
    python: Path,
    launcher: Path,
    run_dir: Path,
    protocol_lock: Path,
    registry_path: Path,
    resource_gate: Path,
    dataset_root: Path,
    bundle_root: Path,
    output_root: Path,
    resume: bool = True,
) -> list[str]:
    """Build the guarded launcher command for Case-B raw regeneration."""

    verify_frozen_d1_sources()
    return [
        str(_executable_file(python, label="Python launcher")),
        str(_regular_file(launcher, label="D1 launcher")),
        "execute-candidates",
        "--run-dir",
        str(run_dir.expanduser().resolve()),
        "--protocol-lock",
        str(protocol_lock.expanduser().resolve()),
        "--registry",
        str(registry_path.expanduser().resolve()),
        "--resource-gate",
        str(resource_gate.expanduser().resolve()),
        "--dataset-root",
        str(dataset_root.expanduser().resolve()),
        "--bundle-root",
        str(bundle_root.expanduser().resolve()),
        "--candidate-root",
        str(output_root.expanduser().resolve()),
        *(["--resume"] if resume else []),
    ]


def build_frozen_scorer_command(
    *,
    docker: Path,
    candidate_root: Path,
    output_root: Path,
    inventory: CandidateInventory,
    model_dir: Path = FROZEN_MODEL_DIR,
    source_view_root: Path | None = None,
    resume: bool = True,
) -> list[str]:
    """Build the network-isolated Docker command for the frozen scorer."""

    verify_frozen_d1_sources()
    source_snapshot = (
        FROZEN_D1_SOURCE / "source_snapshot"
        if source_view_root is None
        else source_view_root.expanduser().resolve()
    )
    _regular_file(
        source_snapshot / "scripts/run_full_gqcnn_scoring.py",
        label="D1 source-view scorer script",
    )
    model = model_dir.expanduser().resolve()
    for name, expected in FROZEN_MODEL_RUNTIME_SHA256.items():
        _verify_frozen_file(
            model / name,
            expected,
            label=f"GQ-CNN model runtime member {name}",
        )
    command = [
        str(_executable_file(docker, label="Docker CLI")),
        "run",
        "--rm",
        "--pull",
        "never",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--init",
        "-v",
        f"{source_snapshot}:/workspace:ro",
        "-v",
        f"{candidate_root.expanduser().resolve()}:/candidates:ro",
        "-v",
        f"{model}:/models/GQCNN-2.1:ro",
        "-v",
        f"{output_root.expanduser().resolve()}:/scored:rw",
        "-w",
        "/workspace",
        # Run the immutable content identity, not the mutable convenience tag
        # inspected by the preflight.  ``--pull=never`` also prevents the
        # daemon from contacting a registry between inspect and run.
        FROZEN_DOCKER_IMAGE_ID,
        "python",
        "scripts/run_full_gqcnn_scoring.py",
        "--candidate-root",
        "/candidates",
        "--output-root",
        "/scored",
        "--model-dir",
        "/models/GQCNN-2.1",
        "--model-name",
        "GQCNN-2.1",
        "--docker-image",
        FROZEN_DOCKER_IMAGE,
        "--docker-image-id",
        FROZEN_DOCKER_IMAGE_ID,
        "--expected-model-config-hash",
        FROZEN_MODEL_CONFIG_SHA256,
        "--seed",
        "42",
        "--expected-samples",
        str(inventory.samples),
        "--expected-nonempty",
        str(inventory.nonempty),
        "--expected-empty",
        str(inventory.empty),
        "--expected-candidates",
        str(inventory.candidates),
        "--batch-size",
        "100",
        "--log-every",
        "100",
    ]
    if resume:
        command.extend(["--resume", "--verify-existing", "--retry-failed"])
    return command


def machine_blocker_payload(
    *,
    blocker_code: str,
    detail: str,
    resume_command: Iterable[str],
    docker_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic, non-filter-only D1 resume contract."""

    command = [str(piece) for piece in resume_command]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "MACHINE_BLOCKED",
        "route": "d1",
        "branch": "gt_oracle",
        "d1_case": "B",
        "blocker_code": blocker_code,
        "detail": detail,
        "raw_candidate_regeneration_required": True,
        "filter_only_primary_allowed": False,
        "execution_attempted": False,
        "docker": dict(docker_probe),
        "resume_command": command,
        "resume_command_shell": shlex.join(command),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "BYTE_IDENTICAL_MEMBERS",
    "CASE_B_MANIFEST_NAME",
    "CandidateInventory",
    "D1AdapterError",
    "D1MachineBlocker",
    "FROZEN_D1_LOADER_SHA256",
    "ORACLE_MASK_SOURCE",
    "assert_gt_bulk_authority",
    "build_frozen_scorer_command",
    "build_gt_candidate_command",
    "build_isolated_gt_bundle_root",
    "build_predicted_replay_command",
    "candidate_inventory_from_summary",
    "machine_blocker_payload",
    "make_oracle_bundle_index_class",
    "verify_frozen_d1_sources",
    "verify_isolated_bundle_root",
]
