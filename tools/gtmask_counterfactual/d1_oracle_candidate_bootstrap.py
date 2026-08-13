#!/usr/bin/env python3
"""Run the frozen D1 sampler against a hash-bound Case-B oracle bundle.

This file intentionally uses only the Python standard library.  It runs inside
the audited GQ-CNN sampler environment, which does not contain pandas/pyarrow,
and injects the narrow metadata adapter before the frozen candidate entrypoint
is imported.  Scientific sampling/filtering code remains byte-for-byte in the
content-addressed source view.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable


REQUIRED_MEMBERS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
    "checksums.sha256",
)
ORACLE_MASK_SOURCE = "locked_original_binary_gt_mask"
PREDICTED_MASK_SOURCE = "predicted_mask_original_resolution"


class OracleBootstrapError(RuntimeError):
    """The isolated oracle runtime differs from its parent-verified inputs."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OracleBootstrapError(f"expected a regular input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OracleBootstrapError(f"cannot parse {label}: {path}") from error
    if not isinstance(value, dict):
        raise OracleBootstrapError(f"{label} must contain one object")
    return value


def checksum_map(directory: Path) -> dict[str, str]:
    lines = (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    result: dict[str, str] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or Path(parts[1].lstrip("* ")).name != parts[1].lstrip("* "):
            raise OracleBootstrapError(f"malformed checksum inventory: {directory}")
        expected, name = parts[0], parts[1].lstrip("* ")
        if name in result or sha256_file(directory / name) != expected:
            raise OracleBootstrapError(f"bundle member hash differs: {directory}/{name}")
        result[name] = expected
    if set(result) != set(REQUIRED_MEMBERS) - {"checksums.sha256"}:
        raise OracleBootstrapError(f"bundle member inventory differs: {directory}")
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def checksum_text(values: dict[str, str]) -> str:
    return "".join(f"{values[name]}  {name}\n" for name in sorted(values))


def link_verified(source: Path, destination: Path) -> None:
    expected = sha256_file(source)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination, follow_symlinks=False)
    if sha256_file(destination) != expected:
        raise OracleBootstrapError(f"temporary member differs: {source.name}")


def verified_bundle_index(
    base_index: type[Any], *, bundle_root: Path, bundle_manifest_sha256: str
) -> type[Any]:
    root_manifest_path = bundle_root / "D1_CASE_B_BUNDLE_MANIFEST.json"
    if sha256_file(root_manifest_path) != bundle_manifest_sha256:
        raise OracleBootstrapError("Case-B bundle manifest differs from parent authority")
    root_manifest = load_object(root_manifest_path, label="Case-B bundle manifest")
    unsigned = {k: v for k, v in root_manifest.items() if k != "content_sha256"}
    if root_manifest.get("content_sha256") != canonical_sha256(unsigned):
        raise OracleBootstrapError("Case-B bundle manifest content hash differs")
    oracle_record = root_manifest.get("oracle_manifest")
    if not isinstance(oracle_record, dict):
        raise OracleBootstrapError("Case-B bundle lacks its oracle manifest")
    manifest_path = Path(str(oracle_record.get("path", ""))).resolve()
    if (
        manifest_path != (bundle_root / "manifest.jsonl").resolve()
        or sha256_file(manifest_path) != oracle_record.get("sha256")
    ):
        raise OracleBootstrapError("Case-B oracle manifest record differs")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row.get("sample_id")): row for row in rows}
    if (
        len(rows) != len(by_id)
        or len(rows) != int(root_manifest.get("evaluable_sample_count", -1))
    ):
        raise OracleBootstrapError("Case-B executable sample universe differs")

    class OracleBundleIndex:
        def __init__(self, dataset_root: Path, mask_root: Path, *, split: str = "test"):
            if split != "test" or Path(mask_root).resolve() != bundle_root:
                raise OracleBootstrapError("oracle loader route/split authority differs")
            self.dataset_root = Path(dataset_root).resolve()
            self.mask_root = bundle_root
            self.split = split
            self.rows = rows
            self.by_id = by_id
            self.manifest_path = manifest_path

        def sample_ids(
            self, sample_id: str | None = None, limit: int | None = None
        ) -> list[str]:
            identifiers = [str(row["sample_id"]) for row in self.rows]
            if sample_id is not None:
                if sample_id not in self.by_id:
                    raise KeyError(f"unknown sample_id: {sample_id}")
                return [sample_id]
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
            if sample_id not in self.by_id:
                raise KeyError(f"unknown sample_id: {sample_id}")
            if kwargs.get("mask_source", "binary_prediction") != "binary_prediction":
                raise OracleBootstrapError("oracle runtime forbids probability input")
            actual = self.mask_root / sample_id
            checksum_map(actual)
            metadata = load_object(actual / "metadata.json", label="oracle metadata")
            if (
                metadata.get("mask_source") != ORACLE_MASK_SOURCE
                or metadata.get("oracle_artifacts_exported") is not True
            ):
                raise OracleBootstrapError(f"oracle provenance differs: {sample_id}")
            with tempfile.TemporaryDirectory(prefix="d1-oracle-loader-") as temporary:
                view_root = Path(temporary)
                view_bundle = view_root / sample_id
                view_bundle.mkdir()
                for name in REQUIRED_MEMBERS:
                    if name not in {"metadata.json", "checksums.sha256"}:
                        link_verified(actual / name, view_bundle / name)
                normalized = dict(metadata)
                normalized["mask_source"] = PREDICTED_MASK_SOURCE
                normalized["oracle_artifacts_exported"] = False
                atomic_json(view_bundle / "metadata.json", normalized)
                hashes = {
                    name: sha256_file(view_bundle / name)
                    for name in REQUIRED_MEMBERS
                    if name != "checksums.sha256"
                }
                (view_bundle / "checksums.sha256").write_text(
                    checksum_text(hashes), encoding="utf-8"
                )
                row = dict(self.by_id[sample_id])
                row.update(
                    {
                        "bundle_dir": str(view_bundle),
                        "path": str(view_bundle),
                        "mask_source": PREDICTED_MASK_SOURCE,
                        "oracle_artifacts_exported": False,
                    }
                )
                (view_root / "manifest.jsonl").write_text(
                    json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                sample = base_index(
                    self.dataset_root, view_root, split=self.split
                ).load_sample(sample_id, **kwargs)
                return replace(sample, bundle_dir=actual, metadata=metadata)

    OracleBundleIndex.__name__ = "D1OracleBundleIndex"
    return OracleBundleIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--source-view-manifest-sha256", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("candidate_arguments", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_view = args.source_view.resolve()
    source_manifest = source_view / "SOURCE_VIEW_MANIFEST.json"
    if sha256_file(source_manifest) != args.source_view_manifest_sha256:
        raise OracleBootstrapError("source-view manifest differs from parent authority")
    candidate_script = source_view / "scripts/run_hifics_dexnet_candidates.py"
    sys.path.insert(0, str(source_view))
    specification = importlib.util.spec_from_file_location(
        "_gtmask_frozen_d1_candidates", candidate_script
    )
    if specification is None or specification.loader is None:
        raise OracleBootstrapError("cannot load frozen candidate entrypoint")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    adapter_module = sys.modules.get(module.OcidVlgBundleIndex.__module__)
    if adapter_module is None:
        raise OracleBootstrapError("frozen candidate loader module was not imported")
    module.OcidVlgBundleIndex = verified_bundle_index(
        adapter_module.OcidVlgBundleIndex,
        bundle_root=args.bundle_root.resolve(),
        bundle_manifest_sha256=args.bundle_manifest_sha256,
    )
    candidate_arguments = list(args.candidate_arguments)
    if candidate_arguments and candidate_arguments[0] == "--":
        candidate_arguments = candidate_arguments[1:]
    previous = sys.argv
    try:
        sys.argv = [str(candidate_script), *candidate_arguments]
        return int(module.main() or 0)
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
