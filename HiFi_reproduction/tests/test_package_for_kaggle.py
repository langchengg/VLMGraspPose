import csv
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_for_kaggle.sh"
REQUIRED_FILES = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, ready_for_anygrasp=True, checksum_files=REQUIRED_FILES):
    run_dir = tmp_path / "run"
    input_root = run_dir / "anygrasp_input_predicted_mask"
    sample_dir = input_root / "sample-001"
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (4, 3)).save(sample_dir / "color.png")
    Image.new("I;16", (4, 3)).save(sample_dir / "depth.png")
    Image.new("L", (4, 3), 255).save(sample_dir / "target_mask.png")
    np.save(
        sample_dir / "target_probability.npy",
        np.ones((3, 4), dtype=np.float32),
    )
    (sample_dir / "language.txt").write_text("pick the object\n")
    (sample_dir / "intrinsics.json").write_text(
        json.dumps({"source": "derived_from_organized_pcd", "depth_scale": 1000.0})
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mask_source": "predicted_mask_original_resolution",
                "oracle_artifacts_exported": False,
            }
        )
    )
    if checksum_files is not None:
        (sample_dir / "checksums.sha256").write_text(
            "".join(
                f"{_sha256(sample_dir / name)}  {name}\n"
                for name in checksum_files
            )
        )
    row = {
        "sample_id": "sample-001",
        "ready": True,
        "ready_for_anygrasp": ready_for_anygrasp,
        "blockers": [],
    }
    (input_root / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    with (input_root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow({**row, "blockers": "[]"})
    subset = run_dir / "anygrasp_verified_subset"
    subset.mkdir()
    (subset / "README.txt").write_text("format verification only\n")
    (run_dir / "predictions").mkdir()
    (run_dir / "predictions/intermediate.npy").write_bytes(b"excluded")
    (run_dir / "checkpoint.pth").write_bytes(b"excluded")
    return run_dir


def _run(run_dir, output_prefix, *options, expected_samples=1):
    environment = {
        **os.environ,
        "HIFI_ANYGRASP_EXPECTED_SAMPLES": str(expected_samples),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *options, str(run_dir), str(output_prefix)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )


def test_dry_run_and_deterministic_package_accept_complete_integrity_metadata(tmp_path):
    run_dir = _fixture(tmp_path)
    output_prefix = tmp_path / "package"

    result = _run(run_dir, output_prefix, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "validated_manifest_rows=1 expected_samples=1" in result.stdout
    assert "validated_ready_samples=1" in result.stdout
    assert "dry_run=PASS archive_not_created=true" in result.stdout
    assert not output_prefix.with_suffix(".tar.gz").exists()

    packaged = _run(run_dir, output_prefix)
    assert packaged.returncode == 0, packaged.stderr
    archive = output_prefix.with_suffix(".tar.gz")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    assert archive.is_file() and checksum.is_file()
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
        package_manifest_name = next(
            name for name in names if name.endswith("/manifests/package_manifest.json")
        )
        package_manifest = json.load(handle.extractfile(package_manifest_name))
    assert package_manifest["source_manifest_total_rows"] == 1
    assert package_manifest["expected_sample_count"] == 1
    assert any(name.endswith("/sample-001/target_probability.npy") for name in names)
    assert not any("predictions" in name or name.endswith("checkpoint.pth") for name in names)
    first_digest = _sha256(archive)
    archive.unlink()
    checksum.unlink()
    packaged_again = _run(run_dir, output_prefix)
    assert packaged_again.returncode == 0, packaged_again.stderr
    assert _sha256(archive) == first_digest


def test_rejects_ready_bundle_without_source_checksums(tmp_path):
    run_dir = _fixture(tmp_path, checksum_files=None)

    result = _run(run_dir, tmp_path / "package", "--dry-run")

    assert result.returncode != 0
    assert "checksums.sha256" in result.stderr


def test_rejects_checksums_that_do_not_cover_target_probability(tmp_path):
    covered = tuple(name for name in REQUIRED_FILES if name != "target_probability.npy")
    run_dir = _fixture(tmp_path, checksum_files=covered)

    result = _run(run_dir, tmp_path / "package", "--dry-run")

    assert result.returncode != 0
    assert "checksum coverage" in result.stderr
    assert "target_probability.npy" in result.stderr


def test_rejects_partial_manifest_against_explicit_expected_count(tmp_path):
    run_dir = _fixture(tmp_path)

    result = _run(
        run_dir,
        tmp_path / "package",
        "--dry-run",
        expected_samples=2,
    )

    assert result.returncode != 0
    assert "expected 2" in result.stderr


def test_requires_ready_for_anygrasp_in_addition_to_ready(tmp_path):
    run_dir = _fixture(tmp_path, ready_for_anygrasp=False)

    result = _run(run_dir, tmp_path / "package", "--dry-run")

    assert result.returncode != 0
    assert "ready_for_anygrasp" in result.stderr
